"""
模块四 · 左栏文档管理组件 doc_panel.py
======================================
对应作业要求 5.4.3「左侧面板：文档上传与管理（批量上传、进度显示、文档列表、删除功能）」。
所有 HTTP 调用一律经 services/rag_client.py，不在组件层直接发请求。
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# 让组件可独立被 import（app.py 已在项目根，这里双保险）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.frontend.services import rag_client  # noqa: E402

# 与后端 routes.py supported 列表保持一致
ACCEPTED_TYPES = ["pdf", "docx", "txt", "md", "csv"]


def _human_size(n: int) -> str:
    """字节数转可读大小。"""
    val = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024 or unit == "GB":
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} GB"


def render_doc_panel() -> None:
    """在传入的容器/侧栏中渲染文档管理面板。"""

    st.subheader("📄 知识库文档")

    # ---- 上传（支持多选批量）----
    files = st.file_uploader(
        "上传论文/文档（支持 PDF、Word、TXT、MD、CSV，可多选）",
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        key="doc_uploader",
    )

    if files and st.button("🚀 上传并构建索引", use_container_width=True,
                           key="btn_upload"):
        progress = st.progress(0.0)
        status_area = st.status("正在处理文档…", expanded=True)
        ok_count, fail_count = 0, 0

        with status_area:
            for i, f in enumerate(files, 1):
                st.write(f"⏳ [{i}/{len(files)}] {f.name} — 解析、分块、向量化中…")
                result = rag_client.upload_document(f.getvalue(), f.name)
                if result.get("status") == "success":
                    ok_count += 1
                    st.write(f"✅ {f.name}：{result.get('pages', '?')} 页 / "
                             f"{result.get('chunks', '?')} 个文本块")
                else:
                    fail_count += 1
                    st.write(f"❌ {f.name}：{result.get('error') or result.get('detail', '处理失败')}")
                progress.progress(i / len(files))

        status_area.update(
            label=f"上传完成：成功 {ok_count} 个，失败 {fail_count} 个",
            state="error" if fail_count else "complete",
        )
        if ok_count:
            st.rerun()

    # ---- 文档列表 + 删除 ----
    st.divider()
    docs = rag_client.list_documents()
    st.caption(f"当前知识库共 {len(docs)} 个文档")

    if not docs:
        st.info("知识库为空。请先上传论文，或确认后端服务已启动。")
        return

    for idx, doc in enumerate(docs):
        name = doc.get("filename", "未知文件")
        size = _human_size(doc.get("file_size", 0))
        status = doc.get("status", "unknown")
        badge = {"indexed": "🟢", "processing": "🟡", "failed": "🔴"}.get(status, "⚪")

        col_name, col_del = st.columns([5, 1])
        with col_name:
            st.markdown(f"{badge} **{name}**")
            st.caption(f"{size} · {status}")
        with col_del:
            st.write("")  # 对齐微调
            if st.button("🗑", key=f"del_doc_{idx}_{name}",
                         help=f"删除 {name}"):
                res = rag_client.delete_document(name)
                if res.get("status") == "success":
                    st.toast(f"已删除 {name}")
                    st.rerun()
                else:
                    st.error(f"删除失败：{res.get('error', '未知错误')}")
