"""
工具: paper_meta（论文元信息）
================================
查询已上传论文的元信息，如文件名、页码、分块数等。
"""
from .base_tool import ToolBase, ToolResult


class PaperMetaTool(ToolBase):
    """论文元信息工具"""

    def __init__(self, vector_store=None):
        self._vector_store = vector_store

    @property
    def name(self) -> str:
        return "paper_meta"

    @property
    def description(self) -> str:
        return "查询知识库中已上传论文的元信息，包括文件名、页码、分块数量等"

    @property
    def parameters(self) -> dict[str, str]:
        return {
            "filename": "指定文件名（可选，不指定则返回全部）",
        }

    def execute(self, filename: str = None, **kwargs) -> ToolResult:
        if self._vector_store is None:
            return ToolResult(success=False, error="VectorStore未初始化")

        # 获取所有分块的元数据
        if not hasattr(self._vector_store, 'chunks_meta') or not self._vector_store.chunks_meta:
            return ToolResult(
                success=True,
                content="知识库为空，尚未上传任何论文",
                detail="未找到任何分块元数据",
            )

        # 按文件聚合统计
        paper_stats = {}
        for meta in self._vector_store.chunks_meta:
            source = meta.get("source", "未知")
            # 提取文件名
            fname = source.split("\\")[-1].split("/")[-1] if source else "未知"

            if fname not in paper_stats:
                paper_stats[fname] = {
                    "filename": fname,
                    "source": source,
                    "chunks": 0,
                    "pages": set(),
                }

            paper_stats[fname]["chunks"] += 1
            page = meta.get("page", 0)
            if page:
                paper_stats[fname]["pages"].add(page)

        # 如果指定了文件名，过滤
        if filename:
            filtered = {k: v for k, v in paper_stats.items() if filename.lower() in k.lower()}
            if not filtered:
                return ToolResult(
                    success=True,
                    content=f"未找到包含'{filename}'的论文",
                    detail="可用论文列表：" + "\n".join(paper_stats.keys()),
                )
            paper_stats = filtered

        # 构建结果
        summary_parts = []
        detail_parts = []

        for fname, stats in paper_stats.items():
            pages = sorted(stats["pages"])
            page_range = f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0]) if pages else "未知"

            summary_parts.append(f"{fname}（{stats['chunks']}个分块，页码{page_range}）")
            detail_parts.append(
                f"文件名: {fname}\n"
                f"路径: {stats['source']}\n"
                f"分块数: {stats['chunks']}\n"
                f"页码范围: {page_range}"
            )

        return ToolResult(
            success=True,
            content=f"共{len(paper_stats)}篇论文: " + "; ".join(summary_parts[:5]),
            detail="\n\n".join(detail_parts),
            metadata={"papers": list(paper_stats.values()), "count": len(paper_stats)},
        )
