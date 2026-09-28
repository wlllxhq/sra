"""
语义缓存模块
对相似问题直接返回缓存答案，节省大模型调用时间和成本
"""
import time
import numpy as np
from loguru import logger


class SemanticCache:
    """
    语义缓存
    使用Embedding相似度判断问题是否与已缓存的问题相似
    相似度高时直接返回缓存答案，无需调用大模型
    """

    def __init__(self, vector_store=None, similarity_threshold: float = 0.95):
        """
        Args:
            vector_store: 复用的VectorStore（借用其embed_query方法）
            similarity_threshold: 相似度阈值，高于此值视为相同问题
        """
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.cache: list[dict] = []  # {query, query_vec, answer, references, timestamp}
        self.hit_count = 0
        self.total_count = 0

    def get(self, query: str) -> dict | None:
        """
        查询缓存

        Args:
            query: 用户问题

        Returns:
            dict | None: 命中缓存则返回缓存结果，否则返回None
        """
        self.total_count += 1

        if not self.cache or self.vector_store is None:
            return None

        # 编码查询
        query_vec = self.vector_store.embed_query(query)

        # 计算与缓存中所有问题的相似度
        for entry in self.cache:
            similarity = float(np.dot(query_vec.flatten(), entry["query_vec"].flatten()))

            if similarity >= self.similarity_threshold:
                self.hit_count += 1
                logger.info(
                    f"缓存命中 (相似度={similarity:.4f}): "
                    f"'{query[:50]}...' → 命中 '{entry['query'][:50]}...'"
                )
                return {
                    "answer": entry["answer"],
                    "references": entry["references"],
                    "retrieval_count": entry.get("retrieval_count", 0),
                    "from_cache": True,
                    "cache_similarity": similarity,
                }

        return None

    def put(self, query: str, answer: str, references: list[dict],
            retrieval_count: int = 0) -> None:
        """
        写入缓存

        Args:
            query: 用户问题
            answer: 生成的回答
            references: 引用来源
            retrieval_count: 检索到的文档数
        """
        if self.vector_store is None:
            return

        query_vec = self.vector_store.embed_query(query)

        self.cache.append({
            "query": query,
            "query_vec": query_vec,
            "answer": answer,
            "references": references,
            "retrieval_count": retrieval_count,
            "timestamp": time.time(),
        })

        logger.debug(f"缓存写入: '{query[:50]}...' (缓存大小: {len(self.cache)})")

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.hit_count = 0
        self.total_count = 0
        logger.info("缓存已清空")

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        if self.total_count == 0:
            return 0.0
        return self.hit_count / self.total_count

    @property
    def size(self) -> int:
        """缓存大小"""
        return len(self.cache)

    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        return {
            "size": self.size,
            "hit_count": self.hit_count,
            "total_count": self.total_count,
            "hit_rate": self.hit_rate,
        }
