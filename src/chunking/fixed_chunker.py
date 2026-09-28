"""
固定大小切分器
按固定字符数切分，不考虑语义边界
用于对比实验：验证语义感知切分是否优于简单切分
"""
from .base_chunker import BaseChunker, Chunk
from src.data_loader.base_loader import Document
from loguru import logger


class FixedChunker(BaseChunker):
    """
    固定大小切分
    每 chunk_size 个字符切一刀，末尾保留 chunk_overlap 个字符的重叠
    """

    def chunk_text(self, text: str) -> list[str]:
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size

            # 如果剩余部分小于 chunk_size，直接取完
            if end >= text_len:
                chunks.append(text[start:])
                break

            chunk = text[start:end]
            chunks.append(chunk)

            # 下一个块从 end - overlap 开始
            start = end - self.chunk_overlap

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks = self._make_chunks(documents)
        logger.info(
            f"固定切分完成: {len(documents)}页文档 → {len(chunks)}个文本块 "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks
