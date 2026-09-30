"""
工具: text_summarize（文本摘要）
==================================
对检索到的文档内容生成摘要。
"""
from .base_tool import ToolBase, ToolResult


class TextSummarizeTool(ToolBase):
    """文本摘要工具"""

    def __init__(self, rag_pipeline=None):
        self._rag_pipeline = rag_pipeline

    @property
    def name(self) -> str:
        return "text_summarize"

    @property
    def description(self) -> str:
        return "对提供的文本内容生成简洁摘要，提取关键信息"

    @property
    def parameters(self) -> dict[str, str]:
        return {
            "text": "需要摘要的文本内容",
            "max_length": "摘要最大长度（字数），默认200",
        }

    def execute(self, text: str = "", max_length: int = 200, **kwargs) -> ToolResult:
        if not text:
            return ToolResult(success=False, error="缺少text参数")

        if self._rag_pipeline is None:
            return ToolResult(success=False, error="RAGPipeline未初始化")

        # 使用RAGPipeline的生成能力来生成摘要
        question = f"请用不超过{max_length}字对以下内容进行简洁摘要，提取关键信息："
        context = [{"content": text, "source": "待摘要文本", "page": 1}]

        try:
            summary = self._rag_pipeline.generate(question, context)

            if not summary or "失败" in summary:
                # 降级：简单截取前max_length字
                summary = text[:max_length] + ("..." if len(text) > max_length else "")

            return ToolResult(
                success=True,
                content=f"摘要完成（{len(summary)}字）",
                detail=f"原文长度: {len(text)}字\n摘要:\n{summary}",
                metadata={
                    "summary": summary,
                    "original_length": len(text),
                    "summary_length": len(summary),
                },
            )

        except Exception as e:
            # 降级：简单截取
            summary = text[:max_length] + ("..." if len(text) > max_length else "")
            return ToolResult(
                success=True,
                content=f"摘要完成（降级模式，{len(summary)}字）",
                detail=f"降级原因: {str(e)}\n摘要:\n{summary}",
                metadata={"summary": summary, "fallback": True},
            )
