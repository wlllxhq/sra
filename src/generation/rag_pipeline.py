"""
RAG完整流水线
串联：检索 → 重排序 → 缓存检查 → 构建Prompt → 调用大模型 → 流式/非流式输出
"""
import os
import time
from openai import OpenAI
from loguru import logger

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.prompt_template import RAG_SYSTEM_PROMPT, build_rag_prompt, FALLBACK_PROMPT
from src.generation.cache import SemanticCache
from src.generation.streaming import (
    StreamEvent, create_reference_events, create_chunk_event,
    create_done_event, create_error_event, create_thinking_event,
)


class RAGPipeline:
    """
    RAG完整流水线

    流程：
    1. 检查语义缓存 → 命中则直接返回
    2. 混合检索（向量 + BM25 + RRF融合）
    3. 重排序（可选）
    4. 降级检查（无结果时友好提示）
    5. 构建Prompt
    6. 调用大模型生成回答
    7. 写入缓存
    """

    def __init__(self, retriever: HybridRetriever,
                 reranker: Reranker = None,
                 cache: SemanticCache = None):
        self.retriever = retriever
        self.reranker = reranker
        self.cache = cache or SemanticCache(vector_store=retriever.vector_store)
        # 正式评测需保证各题相互独立。默认保持缓存开启，仅在启动进程时
        # 显式设置 SEMANTIC_CACHE_ENABLED=false 才关闭缓存读写。
        self.cache_enabled = os.getenv(
            "SEMANTIC_CACHE_ENABLED", "true"
        ).strip().lower() not in {"0", "false", "no", "off"}

        # 初始化大模型客户端
        from src.utils.config import get_config
        config = get_config()
        self.client = OpenAI(
            api_key=config["SILICONFLOW_API_KEY"],
            base_url=config["SILICONFLOW_BASE_URL"],
        )
        self.model = config["CHAT_MODEL"]

        # 统计信息
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "fallback_count": 0,
            "total_tokens": 0,
            "total_time": 0,
        }

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        非流式RAG查询

        Args:
            question: 用户问题
            top_k: 检索的参考文档数

        Returns:
            dict: {
                "answer": str,
                "references": list[dict],
                "retrieval_count": int,
                "from_cache": bool,
                "elapsed_time": float,
            }
        """
        start_time = time.time()
        self.stats["total_queries"] += 1

        # Step 1: 检查缓存
        if self.cache_enabled:
            cached = self.cache.get(question)
            if cached:
                self.stats["cache_hits"] += 1
                cached["elapsed_time"] = time.time() - start_time
                return cached

        # Step 2: 混合检索
        candidates = self.retriever.search(question, top_k=top_k * 4)

        # Step 3: 重排序
        if self.reranker and candidates:
            candidates = self.reranker.rerank(question, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        # Step 4: 降级检查
        if not candidates:
            self.stats["fallback_count"] += 1
            logger.info("检索无结果，返回降级回答")
            return {
                "answer": FALLBACK_PROMPT,
                "references": [],
                "retrieval_count": 0,
                "from_cache": False,
                "elapsed_time": time.time() - start_time,
            }

        # Step 5: 构建Prompt
        user_prompt = build_rag_prompt(question, candidates)

        # Step 6: 调用大模型
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=8192,
            )
            answer = response.choices[0].message.content
            msg = response.choices[0].message

            # 处理推理模型（如deepseek-flash）：content为空时从reasoning_content提取
            if not answer and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                logger.info(f"推理模型: content为空, reasoning_content长度={len(msg.reasoning_content)}")
                # reasoning_content是思考过程，取最后一段作为回答
                answer = msg.reasoning_content

            # 统计Token消耗
            if response.usage:
                self.stats["total_tokens"] += response.usage.total_tokens

        except Exception as e:
            logger.error(f"大模型调用失败: {e}")
            answer = (
                "抱歉，AI模型调用出现异常。请稍后重试。\n\n"
                f"错误信息: {str(e)}\n\n"
                "以下是检索到的相关文献，您可以直接阅读：\n\n"
            )
            for i, ref in enumerate(candidates):
                answer += f"\n【文献{i+1}】{ref.get('source', '')} 第{ref.get('page', '?')}页:\n"
                answer += ref["content"][:200] + "\n"

        result = {
            "answer": answer,
            "references": candidates,
            "retrieval_count": len(candidates),
            "from_cache": False,
            "elapsed_time": time.time() - start_time,
        }

        # Step 7: 写入缓存（只缓存有效的回答，不缓存错误/空回答）
        if (
            self.cache_enabled
            and answer
            and len(answer) > 20
            and "AI模型调用出现异常" not in answer
        ):
            self.cache.put(question, answer, candidates, len(candidates))

        self.stats["total_time"] += result["elapsed_time"]
        logger.info(
            f"RAG查询完成: 耗时{result['elapsed_time']:.2f}s, "
            f"检索{len(candidates)}条, 回答{len(answer)}字"
        )

        return result

    def query_stream(self, question: str, top_k: int = 5):
        """
        流式RAG查询（生成器）

        yield StreamEvent对象，供FastAPI SSE输出

        Args:
            question: 用户问题
            top_k: 检索的参考文档数
        """
        start_time = time.time()
        self.stats["total_queries"] += 1

        # Step 1: 发送"检索中"状态
        yield create_thinking_event("正在检索相关文献...")

        # Step 2: 检查缓存
        if self.cache_enabled:
            cached = self.cache.get(question)
            if cached:
                self.stats["cache_hits"] += 1
                yield create_thinking_event("命中缓存，直接返回")
                for event in create_reference_events(cached.get("references", [])):
                    yield event
                # 分块输出缓存的答案
                answer = cached["answer"]
                chunk_size = 20
                for i in range(0, len(answer), chunk_size):
                    yield create_chunk_event(answer[i:i+chunk_size])
                yield create_done_event(answer, cached.get("references", []),
                                       time.time() - start_time)
                return

        # Step 3: 混合检索
        candidates = self.retriever.search(question, top_k=top_k * 4)
        yield create_thinking_event(
            f"检索到 {len(candidates)} 条候选文献，正在排序..."
        )

        # Step 4: 重排序
        if self.reranker and candidates:
            candidates = self.reranker.rerank(question, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        # Step 5: 发送引用信息
        for event in create_reference_events(candidates):
            yield event

        # Step 6: 降级检查
        if not candidates:
            self.stats["fallback_count"] += 1
            yield create_chunk_event(FALLBACK_PROMPT)
            yield create_done_event(FALLBACK_PROMPT, [], time.time() - start_time)
            return

        # Step 7: 构建Prompt并流式生成
        user_prompt = build_rag_prompt(question, candidates)
        yield create_thinking_event("正在生成回答...")

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=8192,
                stream=True,
            )

            full_answer = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_answer += text
                    yield create_chunk_event(text)

        except Exception as e:
            logger.error(f"大模型流式调用失败: {e}")
            yield create_error_event(f"AI模型调用异常: {str(e)}")
            full_answer = f"调用失败: {str(e)}"

        # Step 8: 写入缓存（只缓存有效的回答）
        if (
            self.cache_enabled
            and full_answer
            and len(full_answer) > 20
            and "调用失败" not in full_answer
        ):
            self.cache.put(question, full_answer, candidates, len(candidates))

        elapsed = time.time() - start_time
        self.stats["total_time"] += elapsed
        yield create_done_event(full_answer, candidates, elapsed)

    def get_stats(self) -> dict:
        """获取RAG流水线统计信息"""
        avg_time = (
            self.stats["total_time"] / self.stats["total_queries"]
            if self.stats["total_queries"] > 0 else 0
        )
        return {
            **self.stats,
            "avg_time": round(avg_time, 2),
            "cache_enabled": self.cache_enabled,
            "cache_stats": self.cache.get_stats(),
        }

    # ---- 供Agent调用的接口 ----

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        纯检索接口（供Agent作为工具调用）
        不走生成流程，只返回检索结果
        """
        candidates = self.retriever.search(query, top_k=top_k * 4)
        if self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]
        return candidates

    def generate(self, question: str, context: list[dict]) -> str:
        """
        纯生成接口（供Agent调用）
        给定问题和上下文，直接生成回答
        """
        user_prompt = build_rag_prompt(question, context)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=8192,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return f"生成失败: {str(e)}"
