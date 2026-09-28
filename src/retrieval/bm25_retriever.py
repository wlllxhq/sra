"""
BM25关键词检索器
基于关键词精确匹配，与向量检索互补
使用jieba进行中文分词
"""
import jieba
from rank_bm25 import BM25Okapi
from loguru import logger


class BM25Retriever:
    """基于BM25算法的关键词检索器"""

    def __init__(self):
        self.bm25: BM25Okapi = None
        self.chunks_meta: list[dict] = []
        # 可添加专业术语到jieba词典
        self._custom_words_added = False

    def _add_custom_words(self):
        """添加领域专业术语到jieba词典"""
        if self._custom_words_added:
            return
        # 添加常见AI/学术术语
        words = [
            "Transformer", "Attention", "BERT", "GPT", "LLM",
            "深度学习", "机器学习", "神经网络", "卷积", "注意力机制",
            "数据集", "损失函数", "梯度下降", "反向传播",
            "BatchNorm", "LayerNorm", "Dropout", "ReLU",
            "ResNet", "ViT", "CLIP", "Diffusion",
            "知识图谱", "迁移学习", "联邦学习",
        ]
        for word in words:
            jieba.add_word(word)
        self._custom_words_added = True

    def _tokenize(self, text: str) -> list[str]:
        """中文分词"""
        self._add_custom_words()
        # 使用精确模式分词
        tokens = list(jieba.cut(text, cut_all=False))
        # 过滤空白和过短的token
        tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 0]
        return tokens

    def build_index(self, chunks: list) -> None:
        """
        从Chunk列表构建BM25索引

        Args:
            chunks: Chunk对象列表
        """
        logger.info(f"开始构建BM25索引: {len(chunks)}个文本块")

        # 分词
        tokenized_corpus = [self._tokenize(c.content) for c in chunks]

        # 构建BM25索引
        self.bm25 = BM25Okapi(tokenized_corpus)

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

        logger.info(f"BM25索引构建完成: {len(chunks)}个文档, "
                     f"词表大小约{len(self.bm25.idf)}")

    def add_chunks(self, chunks: list) -> None:
        """
        增量添加（BM25需要重建索引）

        注意：rank_bm25库不支持增量更新，需要重建
        对于大量增量更新，建议将新旧chunk合并后重新build_index
        """
        # 合并元信息
        existing_chunks_meta = self.chunks_meta.copy()
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
        all_meta = existing_chunks_meta + new_meta

        # 重建索引
        all_texts = [m["content"] for m in all_meta]
        tokenized = [self._tokenize(t) for t in all_texts]
        self.bm25 = BM25Okapi(tokenized)
        self.chunks_meta = all_meta

        logger.info(f"BM25索引重建完成: {len(all_meta)}个文档")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        BM25关键词检索

        Args:
            query: 查询文本
            top_k: 返回数量（通常多取一些，供后续RRF融合）

        Returns:
            list[dict]: 检索结果列表
        """
        if self.bm25 is None:
            logger.warning("BM25索引为空，无法检索")
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # 取top_k
        actual_k = min(top_k, len(scores))
        top_indices = scores.argsort()[::-1][:actual_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self.chunks_meta[idx].copy()
            meta["score"] = float(scores[idx])
            meta["retrieval_method"] = "bm25"
            results.append(meta)

        return results
