"""
工具: paper_compare（论文对比）
================================
对比两篇或多篇论文的内容，分析异同点。
"""
from .base_tool import ToolBase, ToolResult


class PaperCompareTool(ToolBase):
    """论文对比工具"""

    def __init__(self, rag_pipeline=None):
        self._rag_pipeline = rag_pipeline

    @property
    def name(self) -> str:
        return "paper_compare"

    @property
    def description(self) -> str:
        return "对比两篇或多篇论文的内容，分析它们在方法、结论、数据集等方面的异同"

    @property
    def parameters(self) -> dict[str, str]:
        return {
            "papers": "要对比的论文名称列表，如['paper1.pdf', 'paper2.pdf']",
            "aspect": "对比维度，如'方法'、'数据集'、'结论'等",
        }

    def execute(self, papers: list = None, aspect: str = "方法", **kwargs) -> ToolResult:
        if not papers or len(papers) < 2:
            return ToolResult(success=False, error="至少需要指定2篇论文进行对比")

        if self._rag_pipeline is None:
            return ToolResult(success=False, error="RAGPipeline未初始化")

        # 分别检索每篇论文的相关内容
        paper_contents = []

        for paper_name in papers:
            # 使用论文名称+对比维度作为查询
            query = f"{paper_name} {aspect}"
            results = self._rag_pipeline.search(query, top_k=3)

            if not results:
                paper_contents.append({
                    "name": paper_name,
                    "content": "未找到相关内容",
                    "found": False,
                })
            else:
                # 合并检索结果
                combined_content = "\n\n".join([
                    f"[页码{r.get('page', '?')}]: {r.get('content', '')}"
                    for r in results
                ])
                paper_contents.append({
                    "name": paper_name,
                    "content": combined_content,
                    "found": True,
                    "results": results,
                })

        # 检查是否至少找到了一篇论文的内容
        found_papers = [p for p in paper_contents if p["found"]]
        if len(found_papers) < 2:
            not_found = [p["name"] for p in paper_contents if not p["found"]]
            return ToolResult(
                success=False,
                error=f"无法找到足够论文内容进行对比，缺失: {', '.join(not_found)}",
            )

        # 构建对比摘要
        summary_parts = []
        for p in found_papers:
            summary_parts.append(f"{p['name']}（检索到{len(p.get('results', []))}段相关内容）")

        detail_parts = []
        for p in found_papers:
            detail_parts.append(f"【{p['name']}】\n{p['content'][:500]}")

        return ToolResult(
            success=True,
            content=f"对比{len(found_papers)}篇论文的{aspect}: " + "; ".join(summary_parts),
            detail="\n\n".join(detail_parts),
            metadata={
                "papers": paper_contents,
                "aspect": aspect,
                "comparison_ready": True,
            },
        )
