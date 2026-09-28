"""
语义切分器
基于句子边界切分，保证每个chunk内包含完整的句子
用于对比实验
"""
import re
from .base_chunker import BaseChunker, Chunk
from src.data_loader.base_loader import Document
from loguru import logger


class SemanticChunker(BaseChunker):
    """
    语义切分（基于句子边界）
    先按句子拆分，再将句子组合成不超过chunk_size的块
    保证不在句子中间断开
    """

    # 中英文句子结束符
    SENTENCE_ENDINGS = re.compile(
        r'(?<=[。！？.!?])\s*'
    )

    def chunk_text(self, text: str) -> list[str]:
        if not text:
            return []

        # 按句子拆分
        sentences = self.SENTENCE_ENDINGS.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text] if text.strip() else []

        # 将句子组合成块
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # 单句超过chunk_size时，强制切分
            if sentence_len > self.chunk_size:
                # 先保存当前累积的块
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # 强制切分长句
                start = 0
                while start < sentence_len:
                    end = min(start + self.chunk_size, sentence_len)
                    chunks.append(sentence[start:end])
                    start = end - self.chunk_overlap if end < sentence_len else sentence_len
                continue

            # 当前块加上新句子是否超限
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunks.append("".join(current_chunk))

                # 保留重叠：取最后一个句子的尾部作为重叠
                overlap_text = current_chunk[-1] if current_chunk else ""
                if len(overlap_text) > self.chunk_overlap:
                    overlap_text = overlap_text[-self.chunk_overlap:]

                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_length = len(overlap_text) + sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len

        # 最后一块
        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks = self._make_chunks(documents)
        logger.info(
            f"语义切分完成: {len(documents)}页文档 → {len(chunks)}个文本块 "
            f"(chunk_size={self.chunk_size})"
        )
        return chunks
