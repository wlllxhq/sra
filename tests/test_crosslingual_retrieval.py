import sys
import types
import unittest


fake_loguru = types.ModuleType("loguru")
fake_loguru.logger = types.SimpleNamespace(
    info=lambda *args, **kwargs: None,
    warning=lambda *args, **kwargs: None,
    error=lambda *args, **kwargs: None,
    debug=lambda *args, **kwargs: None,
)
sys.modules.setdefault("loguru", fake_loguru)

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_expansion import expand_cross_language_query


def _result(chunk_id, source):
    return {
        "chunk_id": chunk_id,
        "source": source,
        "content": chunk_id,
        "page": 1,
        "score": 1.0,
    }


class _Retriever:
    def __init__(self, results):
        self.results = results
        self.last_query = None

    def search(self, query, top_k):
        self.last_query = query
        return self.results[:top_k]


class CrossLanguageRetrievalTest(unittest.TestCase):
    def test_query_expansion_is_term_based_and_keeps_original(self):
        query = "弱监督鱼类分割的固定预算结果如何？"
        expanded, aliases = expand_cross_language_query(query)
        self.assertTrue(expanded.startswith(query))
        self.assertIn("weakly supervised", aliases)
        self.assertIn("fixed annotation budget", aliases)

        untouched, aliases = expand_cross_language_query("普通中文问题")
        self.assertEqual(untouched, "普通中文问题")
        self.assertEqual(aliases, [])

    def test_reserves_two_vector_sources_and_caps_source_duplicates(self):
        vector = _Retriever([
            _result("a1", "paper-a.pdf"),
            _result("a2", "paper-a.pdf"),
            _result("b1", "paper-b.pdf"),
            _result("c1", "paper-c.pdf"),
        ])
        bm25 = _Retriever([
            _result("d1", "paper-d.pdf"),
            _result("d2", "paper-d.pdf"),
            _result("d3", "paper-d.pdf"),
            _result("a1", "paper-a.pdf"),
        ])
        retriever = HybridRetriever(
            vector, bm25, rrf_k=60,
            enable_query_expansion=False,
            max_chunks_per_source=2,
            vector_source_reserve=2,
        )

        results = retriever.search("query", top_k=5, candidate_multiplier=2)
        sources = [r["source"] for r in results]
        self.assertEqual(sources[:2], ["paper-a.pdf", "paper-b.pdf"])
        self.assertLessEqual(sources.count("paper-d.pdf"), 2)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r["retrieval_method"] == "hybrid_rrf_crosslingual" for r in results))

    def test_search_passes_expanded_query_to_both_channels(self):
        vector = _Retriever([_result("a1", "paper-a.pdf")])
        bm25 = _Retriever([_result("b1", "paper-b.pdf")])
        retriever = HybridRetriever(
            vector, bm25,
            enable_query_expansion=True,
            max_chunks_per_source=2,
            vector_source_reserve=1,
        )
        retriever.search("鲑鱼多目标跟踪", top_k=2)
        self.assertIn("salmon", vector.last_query)
        self.assertIn("multi-object tracking MOT", bm25.last_query)


if __name__ == "__main__":
    unittest.main()
