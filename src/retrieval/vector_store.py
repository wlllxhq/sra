"""
FAISS向量存储
负责文本向量化、FAISS索引构建、向量检索
支持增量更新和持久化
"""
import os
import json
import numpy as np
import faiss
from loguru import logger
from sentence_transformers import SentenceTransformer


class VectorStore:
    """基于FAISS的向量存储与检索"""

    def __init__(self, model_name: str = None, dimension: int = None):
        from src.utils.config import get_config
        config = get_config()

        self.model_name = model_name or config.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
        self.dimension = dimension or int(config.get("VECTOR_DIMENSION", "1024"))
        self.embed_model = None
        self.index = None
        self.chunks_meta: list[dict] = []

    def _load_model(self):
        """懒加载Embedding模型"""
        if self.embed_model is None:
            logger.info(f"加载Embedding模型: {self.model_name}")
            self.embed_model = SentenceTransformer(self.model_name)
            logger.info("Embedding模型加载完成")

    def embed_texts(self, texts: list[str],
                    batch_size: int = 64,
                    show_progress: bool = True) -> np.ndarray:
        """
        将文本列表编码为向量矩阵

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条

        Returns:
            np.ndarray: shape=(len(texts), dimension) 的向量矩阵
        """
        self._load_model()
        embeddings = self.embed_model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # L2归一化，内积等价于余弦相似度
            show_progress_bar=show_progress,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        将查询编码为向量
        bge模型推荐在query前加指令前缀以提升检索效果
        """
        self._load_model()
        # bge模型的query编码指令
        query_with_prefix = f"为这个句子生成表示以用于检索: {query}"
        embedding = self.embed_model.encode(
            [query_with_prefix],
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)

    def build_index(self, chunks: list) -> None:
        """
        从Chunk列表构建FAISS索引

        Args:
            chunks: Chunk对象列表（来自chunking模块）
        """
        logger.info(f"开始构建FAISS索引: {len(chunks)}个文本块")

        texts = [c.content for c in chunks]
        vectors = self.embed_texts(texts)

        # 选择索引类型
        n_vectors = len(vectors)
        if n_vectors < 100:
            # 小数据量用Flat索引（精确搜索）
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(vectors)
            logger.info(f"使用IndexFlatIP (精确搜索, {n_vectors}个向量)")
        else:
            # 大数据量用IVF索引（近似搜索，更快）
            nlist = min(100, n_vectors // 10)
            nlist = max(1, nlist)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(vectors)
            self.index.add(vectors)
            self.index.nprobe = min(20, nlist)  # 搜索时检查的聚类数
            logger.info(f"使用IndexIVFFlat (近似搜索, {n_vectors}个向量, nlist={nlist})")

        # 保存元信息
        self.chunks_meta = [
            {
                "content": c.content,
                "source": c.source,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        logger.info(f"FAISS索引构建完成: {self.index.ntotal}个向量")

    def add_chunks(self, chunks: list) -> None:
        """
        增量添加新的文本块到索引（无需重建全量索引）

        Args:
            chunks: 新增的Chunk列表
        """
        if self.index is None:
            self.build_index(chunks)
            return

        texts = [c.content for c in chunks]
        vectors = self.embed_texts(texts)

        self.index.add(vectors)

        new_meta = [
            {
                "content": c.content,
                "source": c.source,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        self.chunks_meta.extend(new_meta)

        logger.info(f"增量添加 {len(chunks)} 个向量，当前总计 {self.index.ntotal} 个")

    @staticmethod
    def _normalize_source(source: str) -> str:
        """统一路径分隔符，便于跨 Windows/POSIX 匹配同一来源。"""
        return os.path.normcase(os.path.normpath(str(source).replace("\\", "/")))

    @classmethod
    def _source_matches(cls, source: str, target: str) -> bool:
        source_norm = cls._normalize_source(source)
        target_norm = cls._normalize_source(target)
        return (
            source_norm == target_norm
            or os.path.basename(source_norm) == os.path.basename(target_norm)
        )

    def _rebuild_index_from_vectors(self, vectors: np.ndarray) -> None:
        """
        用现成的向量矩阵重建索引（不重新编码）。

        FAISS 不支持原位删向量，删除文档需重建索引；而向量本就在原索引里，
        取回后按行重排即可，避免对整库重跑 Embedding（CPU 上数百块需数分钟）。
        """
        n_vectors = int(vectors.shape[0])
        if n_vectors < 100:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(vectors)
        else:
            nlist = max(1, min(100, n_vectors // 10))
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(vectors)
            self.index.add(vectors)
            self.index.nprobe = min(20, nlist)
        logger.info(f"索引已按现有向量重建: {n_vectors}个向量（未重新编码）")

    def _all_vectors(self) -> np.ndarray:
        """取回索引中已有的全部向量，行序与 chunks_meta 对齐。"""
        n_total = self.index.ntotal if self.index is not None else 0
        if n_total == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        # IVF 索引需 direct map 才能按 id 重建；Flat 索引本身即可直接重建
        make_direct_map = getattr(self.index, "make_direct_map", None)
        if callable(make_direct_map):
            try:
                make_direct_map()
            except Exception as e:  # 个别索引类型不支持，忽略
                logger.warning(f"make_direct_map 失败，将回退重新编码: {e}")
        return np.asarray(self.index.reconstruct_n(0, n_total), dtype=np.float32)

    def delete_by_source(self, source: str) -> int:
        """删除指定来源的全部向量，并用剩余向量重建索引（不重新编码）。"""
        keep_rows = [
            i for i, meta in enumerate(self.chunks_meta)
            if not self._source_matches(meta.get("source", ""), source)
        ]
        removed = len(self.chunks_meta) - len(keep_rows)
        if removed == 0:
            return 0

        kept_meta = [self.chunks_meta[i] for i in keep_rows]

        if not kept_meta:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            all_vectors = None
            try:
                all_vectors = self._all_vectors()
            except Exception as e:
                logger.warning(f"取回现有向量失败，回退重新编码: {e}")

            if all_vectors is not None and all_vectors.shape[0] == len(self.chunks_meta):
                self._rebuild_index_from_vectors(all_vectors[keep_rows])
            else:
                # 索引不支持重建或与元数据不一致（异常情况）：回退到重新编码，保证正确性
                from src.chunking.base_chunker import Chunk

                self.build_index([
                    Chunk(
                        content=meta["content"],
                        source=meta["source"],
                        page=meta["page"],
                        chunk_id=meta["chunk_id"],
                        chunk_index=meta["chunk_index"],
                    )
                    for meta in kept_meta
                ])

        self.chunks_meta = kept_meta
        logger.info(f"已删除来源 {source} 的 {removed} 个向量")
        return removed

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回最相关的top_k个结果

        Returns:
            list[dict]: 检索结果列表，每个包含 content, source, page, score 等
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS索引为空，无法检索")
            return []

        # 实际检索数量不超过索引总量
        actual_k = min(top_k, self.index.ntotal)

        query_vec = self.embed_query(query)
        scores, indices = self.index.search(query_vec, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_meta):
                continue
            meta = self.chunks_meta[idx].copy()
            meta["score"] = float(score)
            meta["retrieval_method"] = "vector"
            results.append(meta)

        return results

    def save(self, path: str = None) -> None:
        """持久化索引到磁盘"""
        if path is None:
            from src.utils.config import get_config
            path = os.path.join(get_config().get("INDEX_DIR", "index"), "vector")

        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks_meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(self.chunks_meta, f, ensure_ascii=False, indent=2)

        # 保存配置信息
        with open(os.path.join(path, "config.json"), "w",
                  encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "dimension": self.dimension,
                "total_vectors": self.index.ntotal,
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"索引已保存到: {path}")

    def load(self, path: str = None) -> bool:
        """
        从磁盘加载索引

        Returns:
            bool: 是否加载成功
        """
        if path is None:
            from src.utils.config import get_config
            path = os.path.join(get_config().get("INDEX_DIR", "index"), "vector")

        index_path = os.path.join(path, "faiss.index")
        meta_path = os.path.join(path, "chunks_meta.json")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.warning(f"索引文件不存在: {path}")
            return False

        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunks_meta = json.load(f)

        logger.info(f"索引加载成功: {self.index.ntotal}个向量, {len(self.chunks_meta)}条元信息")
        return True

    @property
    def total_vectors(self) -> int:
        """索引中的向量总数"""
        return self.index.ntotal if self.index else 0
