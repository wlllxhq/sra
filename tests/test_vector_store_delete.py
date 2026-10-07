import sys
import types
import unittest
from dataclasses import dataclass


class _IndexFlatIP:
    def __init__(self, dimension):
        self.dimension = dimension
        self.ntotal = 0


fake_faiss = types.ModuleType("faiss")
fake_faiss.IndexFlatIP = _IndexFlatIP
sys.modules.setdefault("faiss", fake_faiss)

fake_sentence_transformers = types.ModuleType("sentence_transformers")
fake_sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", fake_sentence_transformers)

fake_loguru = types.ModuleType("loguru")
fake_loguru.logger = types.SimpleNamespace(
    info=lambda *args, **kwargs: None,
    warning=lambda *args, **kwargs: None,
    error=lambda *args, **kwargs: None,
    debug=lambda *args, **kwargs: None,
)
sys.modules.setdefault("loguru", fake_loguru)


@dataclass
class _Chunk:
    content: str
    source: str
    page: int
    chunk_id: str
    chunk_index: int


fake_chunking = types.ModuleType("src.chunking")
fake_chunking.__path__ = []
fake_base_chunker = types.ModuleType("src.chunking.base_chunker")
fake_base_chunker.Chunk = _Chunk
sys.modules.setdefault("src.chunking", fake_chunking)
sys.modules.setdefault("src.chunking.base_chunker", fake_base_chunker)

from src.retrieval.vector_store import VectorStore


class VectorStoreDeleteTest(unittest.TestCase):
    def _store(self):
        store = object.__new__(VectorStore)
        store.dimension = 3
        store.chunks_meta = [
            {
                "content": "保留",
                "source": "uploads/keep.pdf",
                "page": 1,
                "chunk_id": "keep-1",
                "chunk_index": 0,
            },
            {
                "content": "删除一",
                "source": "uploads\\drop.pdf",
                "page": 1,
                "chunk_id": "drop-1",
                "chunk_index": 0,
            },
            {
                "content": "删除二",
                "source": "uploads/drop.pdf",
                "page": 2,
                "chunk_id": "drop-2",
                "chunk_index": 1,
            },
        ]
        store.index = SimpleIndex(3)

        def fake_build(chunks):
            store.chunks_meta = [
                {
                    "content": chunk.content,
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ]
            store.index = SimpleIndex(len(chunks))

        store.build_index = fake_build
        return store

    def test_delete_matches_cross_platform_paths_and_rebuilds(self):
        store = self._store()
        removed = store.delete_by_source("/tmp/uploads/drop.pdf")
        self.assertEqual(removed, 2)
        self.assertEqual(store.index.ntotal, 1)
        self.assertEqual([meta["chunk_id"] for meta in store.chunks_meta], ["keep-1"])

    def test_delete_last_source_creates_empty_index(self):
        store = self._store()
        store.chunks_meta = [store.chunks_meta[1]]
        removed = store.delete_by_source("uploads/drop.pdf")
        self.assertEqual(removed, 1)
        self.assertEqual(store.index.ntotal, 0)
        self.assertEqual(store.chunks_meta, [])


class SimpleIndex:
    def __init__(self, count):
        self.ntotal = count


if __name__ == "__main__":
    unittest.main()
