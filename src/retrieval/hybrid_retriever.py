"""向量检索与BM25的RRF混合检索，包含本地跨语言扩展和来源去冗余。"""

from __future__ import annotations

import os
from collections import defaultdict

from loguru import logger

from .bm25_retriever import BM25Retriever
from .query_expansion import expand_cross_language_query
from .vector_store import VectorStore


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        logger.warning(f"{name}配置无效，回退为{default}")
        value = default
    return max(minimum, min(value, maximum))


class HybridRetriever:
    """RRF混合检索，并避免单篇论文的重复分块挤占TopK。"""

    def __init__(self, vector_store: VectorStore, bm25: BM25Retriever,
                 rrf_k: int = 60, enable_query_expansion: bool | None = None,
                 max_chunks_per_source: int | None = None,
                 vector_source_reserve: int | None = None):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.enable_query_expansion = (
            _env_bool("ENABLE_CROSS_LANGUAGE_EXPANSION", True)
            if enable_query_expansion is None else enable_query_expansion
        )
        self.max_chunks_per_source = (
            _env_int("HYBRID_MAX_CHUNKS_PER_SOURCE", 2, 1, 20)
            if max_chunks_per_source is None else max(1, max_chunks_per_source)
        )
        self.vector_source_reserve = (
            _env_int("HYBRID_VECTOR_SOURCE_RESERVE", 2, 0, 20)
            if vector_source_reserve is None else max(0, vector_source_reserve)
        )

    @staticmethod
    def _source_key(result: dict) -> str:
        source = str(result.get("source", "")).replace("\\", "/").strip()
        return source.casefold() or str(result.get("chunk_id", ""))

    def _prepare_query(self, query: str) -> tuple[str, list[str]]:
        if not self.enable_query_expansion:
            return query, []
        expanded, aliases = expand_cross_language_query(query)
        if aliases:
            logger.debug(f"跨语言查询扩展: {len(aliases)}个英文术语")
        return expanded, aliases

    def search(self, query: str, top_k: int = 5,
               candidate_multiplier: int = 4) -> list[dict]:
        candidate_k = top_k * candidate_multiplier
        retrieval_query, _ = self._prepare_query(query)

        vector_results = self.vector_store.search(
            retrieval_query, top_k=candidate_k
        )
        bm25_results = self.bm25.search(
            retrieval_query, top_k=candidate_k
        )

        logger.debug(
            f"检索候选: 向量{len(vector_results)}条, BM25{len(bm25_results)}条"
        )
        results = self._rrf_fuse(vector_results, bm25_results, top_k)

        if results:
            logger.debug(
                f"RRF融合完成: Top-1 score={results[0].get('rrf_score', 0):.4f}, "
                f"来源={results[0].get('source', 'N/A')}"
            )
        return results

    def search_with_scores(self, query: str, top_k: int = 5) -> dict:
        candidate_k = top_k * 4
        retrieval_query, aliases = self._prepare_query(query)
        vector_results = self.vector_store.search(
            retrieval_query, top_k=candidate_k
        )
        bm25_results = self.bm25.search(
            retrieval_query, top_k=candidate_k
        )
        return {
            "results": self._rrf_fuse(vector_results, bm25_results, top_k),
            "vector_results": vector_results,
            "bm25_results": bm25_results,
            "expanded_query": retrieval_query,
            "query_aliases": aliases,
        }

    def _rrf_fuse(self, vector_results, bm25_results, top_k):
        rrf_scores = defaultdict(float)
        result_map = {}

        for rank, result in enumerate(vector_results, 1):
            key = result["chunk_id"]
            rrf_scores[key] += 1.0 / (self.rrf_k + rank)
            result_map[key] = result

        for rank, result in enumerate(bm25_results, 1):
            key = result["chunk_id"]
            rrf_scores[key] += 1.0 / (self.rrf_k + rank)
            result_map.setdefault(key, result)

        sorted_keys = sorted(
            rrf_scores, key=lambda key: rrf_scores[key], reverse=True
        )
        selected_keys: list[str] = []
        selected_set: set[str] = set()
        source_counts: dict[str, int] = defaultdict(int)

        def add(key: str) -> None:
            if key in selected_set or len(selected_keys) >= top_k:
                return
            source = self._source_key(result_map[key])
            if source_counts[source] >= self.max_chunks_per_source:
                return
            selected_keys.append(key)
            selected_set.add(key)
            source_counts[source] += 1

        # 向量通道对跨语言查询更稳定，先保留前N个不同来源，避免其被
        # 中文BM25与同源重复分块的叠加分数完全挤出。
        reserved_sources: set[str] = set()
        for result in vector_results:
            if len(reserved_sources) >= self.vector_source_reserve:
                break
            source = self._source_key(result)
            if source in reserved_sources:
                continue
            add(result["chunk_id"])
            reserved_sources.add(source)

        for key in sorted_keys:
            add(key)
            if len(selected_keys) >= top_k:
                break

        results = []
        for key in selected_keys:
            result = result_map[key].copy()
            result["rrf_score"] = rrf_scores[key]
            result["retrieval_method"] = "hybrid_rrf_crosslingual"
            results.append(result)
        return results
