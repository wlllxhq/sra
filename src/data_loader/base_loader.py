"""
文档加载器基类
定义统一的文档数据格式和加载器接口
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """统一的文档格式，每个Document代表一页内容"""
    content: str            # 文本内容
    source: str             # 来源文件路径
    page: int               # 页码（从1开始）
    total_pages: int = 0    # 总页数
    metadata: dict = field(default_factory=dict)  # 额外元数据

    @property
    def filename(self) -> str:
        """提取文件名"""
        return Path(self.source).name

    @property
    def char_count(self) -> int:
        return len(self.content)


class BaseLoader(ABC):
    """文档加载器基类"""

    SUPPORTED_EXTENSIONS: list[str] = []

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        """
        加载文档，返回Document列表（按页拆分）

        Args:
            file_path: 文件路径

        Returns:
            list[Document]: 按页拆分的文档列表
        """
        pass

    def can_load(self, file_path: str) -> bool:
        """判断是否能加载该文件"""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS


class DocumentLoaderManager:
    """
    文档加载管理器
    根据文件后缀自动选择合适的加载器
    """

    def __init__(self):
        self.loaders: list[BaseLoader] = []

    def register(self, loader: BaseLoader):
        """注册一个加载器"""
        self.loaders.append(loader)

    def load(self, file_path: str) -> list[Document]:
        """自动选择加载器并加载文档"""
        for loader in self.loaders:
            if loader.can_load(file_path):
                return loader.load(file_path)
        raise ValueError(
            f"不支持的文件格式: {file_path}。"
            f"支持的格式: {[ext for l in self.loaders for ext in l.SUPPORTED_EXTENSIONS]}"
        )

    def load_batch(self, file_paths: list[str],
                   on_progress=None) -> list[Document]:
        """
        批量加载文档

        Args:
            file_paths: 文件路径列表
            on_progress: 进度回调函数 on_progress(current, total, filename, status)

        Returns:
            所有文档的Document列表
        """
        all_docs = []
        total = len(file_paths)

        for i, fp in enumerate(file_paths):
            filename = Path(fp).name
            try:
                if on_progress:
                    on_progress(i + 1, total, filename, "loading")
                docs = self.load(fp)
                all_docs.extend(docs)
                if on_progress:
                    on_progress(i + 1, total, filename, "success")
            except Exception as e:
                if on_progress:
                    on_progress(i + 1, total, filename, f"failed: {e}")

        return all_docs
