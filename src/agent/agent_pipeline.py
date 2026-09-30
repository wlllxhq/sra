"""
Agent流水线
===========
串联ReAct循环、工具集、记忆管理，输出SSE事件流。
提供流式和非流式两种调用方式。
"""
import os
import time
from typing import Generator

from loguru import logger
from openai import OpenAI

from .memory import MemoryManager, init_memory_manager
from .react_loop import ReActLoop
from .tools import register_all_tools, get_tool_descriptions


class AgentPipeline:
    """
    Agent流水线

    初始化流程:
        1. 注册8个工具
        2. 初始化记忆管理器
        3. 创建ReAct循环实例
    """

    def __init__(self, rag_pipeline=None, vector_store=None,
                 max_rounds: int = 5):
        """
        Args:
            rag_pipeline: RAGPipeline实例（模块二）
            vector_store: VectorStore实例（模块一）
            max_rounds: 最大推理轮次
        """
        # 1. 注册工具
        register_all_tools(rag_pipeline=rag_pipeline, vector_store=vector_store)
        self.tool_descriptions = get_tool_descriptions()
        logger.info(f"Agent工具注册完成:\n{self.tool_descriptions}")

        # 2. 初始化记忆管理器
        self.memory_manager = init_memory_manager(
            rag_pipeline=rag_pipeline,
            window_size=10,
        )

        # 3. 初始化LLM客户端（复用模块二的配置）
        from src.utils.config import get_config
        config = get_config()
        self.client = OpenAI(
            api_key=config["SILICONFLOW_API_KEY"],
            base_url=config["SILICONFLOW_BASE_URL"],
        )
        self.model = config["CHAT_MODEL"]

        # 4. 创建ReAct循环
        self.react_loop = ReActLoop(
            llm_client=self.client,
            model=self.model,
            memory_manager=self.memory_manager,
            max_rounds=max_rounds,
        )

        # 统计
        self.stats = {
            "total_queries": 0,
            "total_time": 0,
        }

    def query_stream(self, question: str, session_id: str = "default",
                     top_k: int = 5) -> Generator[str, None, None]:
        """
        流式Agent查询（SSE格式）

        Args:
            question: 用户问题
            session_id: 会话ID
            top_k: 检索条数

        Yields:
            str: SSE格式的事件字符串
        """
        import json

        self.stats["total_queries"] += 1
        start_time = time.time()

        logger.info(f"[{session_id}] Agent查询: {question[:80]}...")

        try:
            for event in self.react_loop.run(
                question=question,
                session_id=session_id,
                top_k=top_k,
                tool_descriptions=self.tool_descriptions,
            ):
                # 转为SSE格式
                sse_data = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                yield sse_data

        except Exception as e:
            logger.error(f"Agent流式查询异常: {e}", exc_info=True)
            # 必须发送error事件收尾（交接说明红线3）
            error_event = {
                "type": "error",
                "content": f"Agent推理过程异常: {str(e)}",
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

        finally:
            elapsed = time.time() - start_time
            self.stats["total_time"] += elapsed
            logger.info(f"[{session_id}] Agent查询完成: {elapsed:.2f}s")

    def query(self, question: str, session_id: str = "default",
              top_k: int = 5) -> dict:
        """
        非流式Agent查询（收集所有事件，返回最终结果）

        Args:
            question: 用户问题
            session_id: 会话ID
            top_k: 检索条数

        Returns:
            dict: {
                "answer": str,
                "references": list[dict],
                "from_cache": False,
                "elapsed_time": float,
                "trace": list[dict],  # ReAct轨迹
            }
        """
        import json

        self.stats["total_queries"] += 1
        start_time = time.time()

        answer = ""
        references = []
        trace = []
        token_count = 0
        tool_token_counts: dict = {}

        for event in self.react_loop.run(
            question=question,
            session_id=session_id,
            top_k=top_k,
            tool_descriptions=self.tool_descriptions,
        ):
            event_type = event.get("type", "")

            # 收集轨迹事件
            if event_type in ("thought", "action", "observation", "thinking"):
                trace.append(event)

            elif event_type == "reference":
                references.append({
                    "source": event.get("doc", ""),
                    "filename": event.get("doc", ""),
                    "page": event.get("page", 0),
                    "content": event.get("content", ""),
                    "score": event.get("score", 0),
                })

            elif event_type == "done":
                answer = event.get("content", "")
                token_count = event.get("token_count", 0)
                tool_token_counts = event.get("tool_token_counts", {}) or {}

            elif event_type == "error":
                answer = f"Agent执行出错: {event.get('content', '')}"

        elapsed = time.time() - start_time
        self.stats["total_time"] += elapsed

        return {
            "answer": answer,
            "references": references,
            "retrieval_count": len(references),
            "from_cache": False,
            "elapsed_time": elapsed,
            "trace": trace,
            "token_count": token_count,
            "tool_token_counts": tool_token_counts,
        }

    def get_stats(self) -> dict:
        """获取Agent统计信息"""
        avg_time = (
            self.stats["total_time"] / max(1, self.stats["total_queries"])
        )

        react_stats = self.react_loop.get_stats()
        memory_stats = self.memory_manager.get_session_stats()

        return {
            "total_queries": self.stats["total_queries"],
            "avg_response_time": round(avg_time, 2),
            "total_rounds": react_stats.get("total_rounds", 0),
            "avg_rounds_per_query": react_stats.get("avg_rounds_per_query", 0),
            "avg_rounds_per_session": react_stats.get("avg_rounds_per_session", 0),
            "total_tool_calls": react_stats.get("total_tool_calls", 0),
            "tool_success_count": react_stats.get("tool_success_count", 0),
            "tool_fail_count": react_stats.get("tool_fail_count", 0),
            "tool_success_rate": react_stats.get("tool_success_rate", 0),
            "tool_call_counts": react_stats.get("tool_call_counts", {}),
            "tool_token_counts": react_stats.get("tool_token_counts", {}),
            "tool_avg_time_ms": react_stats.get("tool_avg_time_ms", {}),
            "total_llm_tokens": react_stats.get("total_llm_tokens", 0),
            "active_sessions": memory_stats.get("total_sessions", 0),
        }


# 全局Agent流水线实例
_agent_pipeline: AgentPipeline | None = None


def get_agent_pipeline() -> AgentPipeline | None:
    """获取全局Agent流水线实例"""
    return _agent_pipeline


def init_agent_pipeline(rag_pipeline=None, vector_store=None,
                        max_rounds: int | None = None) -> AgentPipeline:
    """初始化全局Agent流水线"""
    global _agent_pipeline
    if max_rounds is None:
        try:
            max_rounds = int(os.getenv("AGENT_MAX_ROUNDS", "5"))
        except ValueError:
            logger.warning("AGENT_MAX_ROUNDS无效，回退为5")
            max_rounds = 5
    max_rounds = max(1, min(max_rounds, 10))
    _agent_pipeline = AgentPipeline(
        rag_pipeline=rag_pipeline,
        vector_store=vector_store,
        max_rounds=max_rounds,
    )
    logger.info(f"Agent流水线初始化完成，最大推理轮次={max_rounds}")
    return _agent_pipeline
