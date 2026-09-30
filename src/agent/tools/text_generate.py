"""
工具: text_generate（文本生成）
===============================
给定问题和上下文，调用大模型生成回答（不走检索流程）。
对应后端 /api/generate 接口。
"""
from .base_tool import ToolBase, ToolResult


class TextGenerateTool(ToolBase):
    """文本生成工具"""

    def __init__(self, rag_pipeline=None):
        self._rag_pipeline = rag_pipeline

    @property
    def name(self) -> str:
        return "text_generate"

    @property
    def description(self) -> str:
        return "基于提供的上下文信息，调用大模型生成回答（不经过检索，需要已有上下文）"

    @property
    def parameters(self) -> dict[str, str]:
        return {
            "question": "需要回答的问题",
            "context": "上下文信息列表，每项包含content/source/page",
        }

    def execute(self, question: str = "", context: list = None, **kwargs) -> ToolResult:
        if not question:
            return ToolResult(success=False, error="缺少question参数")

        if self._rag_pipeline is None:
            return ToolResult(success=False, error="RAGPipeline未初始化")

        ctx = context or []

        # 直接调用模块二的generate方法
        answer = self._rag_pipeline.generate(question, ctx)

        if not answer or "失败" in answer:
            return ToolResult(
                success=False,
                error=f"生成失败: {answer}",
            )

        return ToolResult(
            success=True,
            content=f"生成完成，回答长度{len(answer)}字",
            detail=f"回答:\n{answer}",
            metadata={"answer": answer, "context_count": len(ctx)},
        )
