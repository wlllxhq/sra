import sys
import types
import unittest
from types import SimpleNamespace

fake_loguru = types.ModuleType("loguru")
fake_loguru.logger = SimpleNamespace(
    info=lambda *args, **kwargs: None,
    warning=lambda *args, **kwargs: None,
    error=lambda *args, **kwargs: None,
    debug=lambda *args, **kwargs: None,
)
sys.modules.setdefault("loguru", fake_loguru)

from src.agent import react_loop as react_loop_module
from src.agent.react_loop import ReActLoop
from src.agent.tools.base_tool import ToolResult


class _Memory:
    def __init__(self):
        self.sessions = {}

    def _session(self, session_id):
        return self.sessions.setdefault(session_id, SimpleNamespace(messages=[]))

    def add_user_message(self, session_id, content):
        self._session(session_id).messages.append(SimpleNamespace(role="user", content=content))

    def add_assistant_message(self, session_id, content, metadata=None):
        self._session(session_id).messages.append(SimpleNamespace(role="assistant", content=content))

    def get_context_prompt(self, session_id):
        return ""

    def get_session(self, session_id):
        return self._session(session_id)


class _LLM:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        message = SimpleNamespace(content=next(self.outputs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(total_tokens=10),
        )


class _Tool:
    def safe_execute(self, **kwargs):
        return ToolResult(
            content="找到相关段落",
            detail="完整检索正文",
            metadata={
                "references": [
                    {
                        "chunk_id": "paper_p3_c0",
                        "source": "uploads/paper.pdf",
                        "page": 3,
                        "content": "证据一",
                        "score": 0.8,
                    },
                    {
                        "chunk_id": "paper_p3_c1",
                        "source": "uploads\\paper.pdf",
                        "page": 3,
                        "content": "证据二",
                        "score": 0.7,
                    },
                ]
            },
        )


class ReActReferenceMetricsTest(unittest.TestCase):
    def test_reference_dedup_and_round_statistics(self):
        memory = _Memory()
        loop = ReActLoop(
            llm_client=_LLM([
                'Thought: 检索\nAction: knowledge_search\nArgs: {"query": "鱼体", "top_k": 5}',
                "Thought: 信息充分\nFinal Answer: 结论[来源: paper.pdf, 第3页]",
            ]),
            model="fake",
            memory_manager=memory,
            max_rounds=3,
        )

        original_get_tool = react_loop_module.get_tool
        react_loop_module.get_tool = lambda name: _Tool()
        try:
            events = list(loop.run("问题", "session-1", 5, "tools"))
        finally:
            react_loop_module.get_tool = original_get_tool

        references = [event for event in events if event["type"] == "reference"]
        done = next(event for event in events if event["type"] == "done")
        self.assertEqual(len(references), 1)
        self.assertEqual(done["reference_count"], 1)
        self.assertEqual(done["rounds_used"], 2)

        stats = loop.get_stats()
        self.assertEqual(stats["total_runs"], 1)
        self.assertEqual(stats["total_rounds"], 2)
        self.assertEqual(stats["avg_rounds_per_query"], 2.0)
        self.assertEqual(stats["avg_rounds_per_session"], 2.0)

        loop.llm_client = _LLM(["Thought: 直接回答\nFinal Answer: 完成"])
        list(loop.run("第二问", "session-1", 5, "tools"))
        stats = loop.get_stats()
        self.assertEqual(stats["total_runs"], 2)
        self.assertEqual(stats["total_rounds"], 3)
        self.assertEqual(stats["avg_rounds_per_query"], 1.5)


if __name__ == "__main__":
    unittest.main()
