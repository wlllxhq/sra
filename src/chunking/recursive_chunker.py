"""
递归字符切分器（主力方案）
按优先级递归使用不同分隔符切分文本，尽量在语义边界处断开
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from .base_chunker import BaseChunker, Chunk
from src.data_loader.base_loader import Document


class RecursiveChunker(BaseChunker):
    """
    递归字符切分
    分隔符优先级：双换行 > 单换行 > 句号 > 感叹号 > 问号 > 空格 > 无
    适合中英文混合的学术论文
    """

    # 中英文分隔符，按优先级排列
    SEPARATORS = [
        "\n\n",          # 段落分隔
        "\n",            # 换行
        "。",            # 中文句号
        "！",            # 中文感叹号
        "？",            # 中文问号
        ". ",            # 英文句号（后面跟空格）
        "! ",            # 英文感叹号
        "? ",            # 英文问号
        "；",            # 中文分号
        "; ",            # 英文分号
        "，",            # 中文逗号
        ", ",            # 英文逗号
        " ",             # 空格
        "",              # 无分隔符（字符级切分，最后手段）
    ]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.SEPARATORS,
            length_function=len,
        )

    def chunk_text(self, text: str) -> list[str]:
        return self.splitter.split_text(text)

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks = self._make_chunks(documents)
        logger.info(
            f"递归切分完成: {len(documents)}页文档 → {len(chunks)}个文本块 "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks
