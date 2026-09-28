# 延迟导入，避免未安装依赖时整体报错
def __getattr__(name):
    if name == "VectorStore":
        from .vector_store import VectorStore
        return VectorStore
    elif name == "BM25Retriever":
        from .bm25_retriever import BM25Retriever
        return BM25Retriever
    elif name == "HybridRetriever":
        from .hybrid_retriever import HybridRetriever
        return HybridRetriever
    elif name == "Reranker":
        from .reranker import Reranker
        return Reranker
    raise AttributeError(f"module 'retrieval' has no attribute '{name}'")

__all__ = ["VectorStore", "BM25Retriever", "HybridRetriever", "Reranker"]
