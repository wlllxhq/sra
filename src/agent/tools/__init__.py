"""
Agent工具集
==========
注册并管理所有可用工具。

工具命名规范: snake_case 英文
工具注册表: TOOL_REGISTRY（dict，tool_name → ToolBase实例）
"""
from .base_tool import ToolBase, ToolResult
from .knowledge_search import KnowledgeSearchTool
from .knowledge_qa import KnowledgeQATool
from .text_generate import TextGenerateTool
from .calculator import CalculatorTool
from .paper_meta import PaperMetaTool
from .paper_compare import PaperCompareTool
from .text_summarize import TextSummarizeTool
from .keyword_extract import KeywordExtractTool

# 全局工具注册表
TOOL_REGISTRY: dict[str, ToolBase] = {}


def register_all_tools(rag_pipeline=None, vector_store=None):
    """
    注册全部8个工具到全局注册表

    Args:
        rag_pipeline: RAGPipeline实例（供knowledge_search/knowledge_qa/text_generate使用）
        vector_store: VectorStore实例（供paper_meta使用）
    """
    tools = [
        KnowledgeSearchTool(rag_pipeline=rag_pipeline),
        KnowledgeQATool(rag_pipeline=rag_pipeline),
        TextGenerateTool(rag_pipeline=rag_pipeline),
        CalculatorTool(),
        PaperMetaTool(vector_store=vector_store),
        PaperCompareTool(rag_pipeline=rag_pipeline),
        TextSummarizeTool(rag_pipeline=rag_pipeline),
        KeywordExtractTool(),
    ]
    for tool in tools:
        TOOL_REGISTRY[tool.name] = tool

    return TOOL_REGISTRY


def get_tool(name: str) -> ToolBase | None:
    """按名称获取工具实例"""
    return TOOL_REGISTRY.get(name)


def get_all_tools() -> list[ToolBase]:
    """获取所有已注册的工具"""
    return list(TOOL_REGISTRY.values())


def get_tool_descriptions() -> str:
    """生成工具描述文本（供ReAct系统提示词使用）"""
    lines = []
    for tool in TOOL_REGISTRY.values():
        lines.append(f"- {tool.name}: {tool.description}")
        if tool.parameters:
            params_str = ", ".join(
                f"{k}({v})" for k, v in tool.parameters.items()
            )
            lines.append(f"  参数: {params_str}")
    return "\n".join(lines)


__all__ = [
    "ToolBase", "ToolResult",
    "TOOL_REGISTRY",
    "register_all_tools", "get_tool", "get_all_tools",
    "get_tool_descriptions",
]
