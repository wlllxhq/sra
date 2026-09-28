"""
文本分块器基类
定义统一的Chunk数据格式和分块器接口
"""
from dataclasses import dataclass
from src.data_loader.base_loader import Document


@dataclass
class Chunk:
    """文本块"""
    content: str        # 文本内容
    source: str         # 来源文件路径
    page: int           # 页码
    chunk_id: str       # 唯一标识符
    chunk_index: int    # 全局序号
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def filename(self) -> str:
        from pathlib import Path
        return Path(self.source).name

    @property
    def char_count(self) -> int:
        return len(self.content)


class BaseChunker:
    """分块器基类"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        将文档列表切分为文本块列表

        Args:
            documents: Document列表

        Returns:
            Chunk列表
        """
        raise NotImplementedError

    def chunk_text(self, text: str) -> list[str]:
        """将单段文本切分为字符串列表（子类实现）"""
        raise NotImplementedError

    def _make_chunks(self, documents: list[Document]) -> list[Chunk]:
        """通用：将文档列表切分并生成Chunk对象"""
        chunks = []
        global_idx = 0

        for doc in documents:
            texts = self.chunk_text(doc.content)
            for i, text in enumerate(texts):
                if text.strip():
                    chunks.append(Chunk(
                        content=text.strip(),
                        source=doc.source,
                        page=doc.page,
                        chunk_id=f"{doc.source}_p{doc.page}_c{i}",
                        chunk_index=global_idx,
                        metadata={
                            "total_pages": doc.total_pages,
                            "chunker": self.__class__.__name__,
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                        }
                    ))
                    global_idx += 1

        return chunks
