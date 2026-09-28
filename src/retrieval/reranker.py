"""
重排序模块
使用cross-encoder模型对候选结果进行精细打分
在混合检索之后使用，进一步提升排序精度
"""
from loguru import logger


class Reranker:
    """
    重排序器
    对混合检索的Top-N候选结果进行精细打分重排
    使用cross-encoder架构，精度高于bi-encoder（向量检索）
    """

    def __init__(self, model_name: str = None):
        from src.utils.config import get_config
        config = get_config()

        self.model_name = model_name or config.get(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.reranker = None
        self._load_failed = False

        # 如果配置为 none，直接禁用
        if self.model_name.lower() == "none":
            self._load_failed = True

    def _load_model(self):
        """懒加载Reranker模型"""
        if self.reranker is not None or self._load_failed:
            return

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"加载Reranker模型: {self.model_name}")
            self.reranker = CrossEncoder(self.model_name)
            logger.info("Reranker模型加载完成")
        except ImportError:
            logger.warning(
                "sentence-transformers未安装，重排序功能不可用。"
                "安装命令: pip install sentence-transformers"
            )
            self._load_failed = True
        except Exception as e:
            logger.warning(f"Reranker模型加载失败: {e}，将跳过重排序")
            self._load_failed = True

    def rerank(self, query: str, candidates: list[dict],
               top_k: int = 5) -> list[dict]:
        """
        对候选结果重排序

        Args:
            query: 查询文本
            candidates: 候选结果列表（来自混合检索）
            top_k: 返回重排后的top_k

        Returns:
            list[dict]: 重排后的结果，新增rerank_score字段
        """
        if not candidates:
            return []

        # 如果候选数已经 <= top_k，且reranker不可用，直接返回
        if len(candidates) <= top_k:
            return candidates

        self._load_model()

        # Reranker不可用时，降级返回原结果
        if self.reranker is None:
            logger.debug("Reranker不可用，返回原始排序结果")
            return candidates[:top_k]

        try:
            # 构造(query, passage)对
            pairs = [[query, c["content"]] for c in candidates]

            # 计算相关性分数（CrossEncoder.predict返回numpy数组）
            scores = self.reranker.predict(pairs)

            # 确保scores是可迭代的
            if hasattr(scores, '__iter__'):
                scores = list(scores)
            else:
                scores = [scores]

            # 附加rerank分数
            for c, score in zip(candidates, scores):
                c["rerank_score"] = float(score)
                c["retrieval_method"] = "hybrid_rrf_rerank"

            # 按rerank分数降序排序
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

            logger.debug(
                f"重排序完成: {len(candidates)}条候选 → Top-{top_k}, "
                f"Top-1 rerank_score={candidates[0]['rerank_score']:.4f}"
            )

            return candidates[:top_k]

        except Exception as e:
            logger.error(f"重排序失败: {e}，返回原始结果")
            return candidates[:top_k]

    @property
    def available(self) -> bool:
        """检查reranker是否可用"""
        self._load_model()
        return self.reranker is not None
