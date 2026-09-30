"""
工具: knowledge_qa（知识库问答）
================================
端到端知识库问答：检索+生成完整流程。
对应后端 /api/query 接口，但直接调用RAGPipeline.query()避免HTTP回环。
"""
from .base_tool import ToolBase, ToolResult


class KnowledgeQATool(ToolBase):
    """知识库问答工具"""

    def __init__(self, rag_pipeline=None):
        self._rag_pipeline = rag_pipeline

    @property
    def name(self) -> str:
        return "knowledge_qa"

    @property
    def description(self) -> str:
        return "端到端知识库问答：自动检索相关文献并用大模型生成带引用来源的回答"

    @property
    def parameters(self) -> dict[str, str]:
        return {
            "question": "用户问题",
            "top_k": "检索参考文档数，默认5",
        }

    def execute(self, question: str = "", top_k: int = 5, **kwargs) -> ToolResult:
        if not question:
            return ToolResult(success=False, error="缺少question参数")

        if self._rag_pipeline is None:
            return ToolResult(success=False, error="RAGPipeline未初始化")

        # 直接调用模块二的query方法
        result = self._rag_pipeline.query(question, top_k=top_k)

        answer = result.get("answer", "")
        references = result.get("references", [])

        # 构建摘要
        ref_summary = ""
        if references:
            ref_names = []
            for r in references[:3]:
                source = r.get("source", "未知")
                filename = source.split("\\")[-1].split("/")[-1] if source else "未知"
                ref_names.append(filename)
            ref_summary = f"，引用了{len(references)}篇文献（{', '.join(ref_names)}等）"

        return ToolResult(
            success=True,
            content=f"问答完成{ref_summary}",
            detail=f"回答:\n{answer}\n\n引用数: {len(references)}",
            metadata={
                "answer": answer,
                "references": references,
                "from_cache": result.get("from_cache", False),
                "elapsed_time": result.get("elapsed_time", 0),
            },
        )
