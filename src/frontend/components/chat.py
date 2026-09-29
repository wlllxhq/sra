"""
模块四 · 中央对话组件 chat.py
=============================
对应作业要求 5.4.3「中央主区域：对话窗口（流式输出打字机效果、Markdown 渲染、引用标记动态出现）」。

注意：Streamlit 规定 st.chat_input 只能在脚本顶层调用，因此输入由 app.py 获取后
通过 render_chat(session, question) 传入，本组件负责渲染历史与消费事件流。

事件消费规则（rag_client.NormalizedEvent.kind）：
    status      → 顶部状态行（打字机开始前显示"正在检索…"等），同时记入轨迹时间线
    delta       → 累积进答案缓冲区，节流刷新（打字机效果）
    reference   → 先于正文动态出现在答案上方（引用标记）
    thought /
    action /
    observation → 记入本轮轨迹 trace（右栏渲染，模块三接入后生效）
    final       → 收尾：耗时/Token/引用数写入 message 元数据
    error       → 红色提示区
    unknown     → 降级为灰色 caption 原样展示，绝不崩溃
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterator, List

import streamlit as st

from src.frontend.services import rag_client
from src.frontend.components import trace_panel

# 后端 Prompt 约定：回答中引用格式为 [来源: 文档名, 第X页]
_CITATION_RE = re.compile(r"\[来源:\s*([^\],]+?)\s*,\s*第\s*(\d+|[?？])\s*页\]")


def _highlight_citations(text: str) -> str:
    """把引用标记渲染为醒目徽章。"""
    def repl(m: re.Match) -> str:
        doc, page = m.group(1).strip(), m.group(2)
        return f"**`📎{doc}·第{page}页`**"
    return _CITATION_RE.sub(repl, text)


def _ref_chips(refs: List[Dict[str, Any]]) -> str:
    return "  ".join(
        f"`📎[{r.get('ref_index', i + 1)}] {r.get('doc', '?')}·P{r.get('page', '?')}`"
        for i, r in enumerate(refs)
    )


def _render_history(messages: List[Dict[str, Any]]) -> None:
    """重放历史消息（含引用徽章与错误态）。"""
    for msg in messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
            continue
        with st.chat_message("assistant"):
            refs = msg.get("references") or []
            if refs:
                st.markdown(_ref_chips(refs))
            if msg.get("error"):
                st.error(msg["error"])
            if msg.get("content"):
                st.markdown(_highlight_citations(msg["content"]))
            meta = msg.get("meta") or {}
            bits = []
            if meta.get("elapsed_time"):
                bits.append(f"耗时 {meta['elapsed_time']:.1f}s")
            if meta.get("token_count"):
                bits.append(f"Token {meta['token_count']}")
            if bits:
                st.caption(" · ".join(bits))


def _draw_live_trace(placeholder: Any, trace: List[Dict[str, Any]]) -> None:
    """把当前轨迹树重绘到右栏占位槽（无害空操作：无占位槽时跳过）。"""
    if placeholder is None:
        return
    with placeholder.container():
        trace_panel._render_trace_tree(trace)


def _consume_stream(events: Iterator[rag_client.NormalizedEvent],
                    trace_placeholder: Any = None) -> Dict[str, Any]:
    """
    消费归一化事件流并实时渲染，返回本轮完整消息 dict。
    打字机节流：距上次刷新 ≥0.12s 或累计 8 个 chunk 才重绘一次，
    避免 placeholder 高频重绘导致卡顿。

    trace_placeholder：右栏预留的占位槽。每收到一条轨迹事件就在此重绘推理轨迹，
    使右栏 Agent 轨迹随流式事件实时生长（否则脚本自上而下执行，右栏要等本函数
    把整条事件流消费完才渲染，轨迹会"一次性蹦出"）。
    """
    status_line = st.empty()
    refs_area = st.empty()
    answer_area = st.empty()

    buffer: List[str] = []
    references: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    err: str = ""
    shown_any = False

    last_draw = time.time()
    chunk_since_draw = 0

    def draw() -> None:
        nonlocal last_draw, chunk_since_draw
        if buffer:
            answer_area.markdown(_highlight_citations("".join(buffer)) + "▍")
        last_draw = time.time()
        chunk_since_draw = 0

    for ev in events:
        if ev.kind == "status":
            status_line.caption(f"⏳ {ev.text}")
            trace.append({"type": "status", "text": ev.text})
            _draw_live_trace(trace_placeholder, trace)

        elif ev.kind == "reference":
            references.append({
                "ref_index": ev.ref_index, "doc": ev.doc,
                "page": ev.page, "score": ev.score, "content": ev.text,
            })
            refs_area.markdown(_ref_chips(references))

        elif ev.kind == "delta":
            if not shown_any:
                status_line.empty()   # 正文开始，收起状态行
                shown_any = True
            buffer.append(ev.text)
            chunk_since_draw += 1
            if chunk_since_draw >= 8 or (time.time() - last_draw) >= 0.12:
                draw()

        elif ev.kind in ("thought", "action", "observation"):
            trace.append({
                "type": ev.kind, "text": ev.text,
                "tool": ev.tool, "args": ev.args,
            })
            _draw_live_trace(trace_placeholder, trace)

        elif ev.kind == "final":
            draw()
            meta = {
                "elapsed_time": ev.elapsed_time,
                "token_count": ev.token_count,
                "reference_count": ev.reference_count,
                "tool_token_counts": dict(ev.tool_token_counts),
            }

        elif ev.kind == "error":
            err = ev.text
            status_line.error(f"❌ {ev.text}")

        else:  # unknown：未知事件类型，降级展示（向后兼容）
            st.caption(f"ℹ️ 未识别事件 [{ev.payload.get('type', '?')}]: {ev.text[:120]}")

    # 收尾：去掉光标，落最终文本
    final_text = "".join(buffer)
    if final_text:
        answer_area.markdown(_highlight_citations(final_text))
    else:
        answer_area.empty()
    if not shown_any:
        status_line.empty()

    return {
        "role": "assistant",
        "content": final_text,
        "references": references,
        "trace": trace,
        "meta": meta,
        "error": err or None,
    }


def render_chat(session: Dict[str, Any], question: str = "",
                trace_placeholder: Any = None) -> None:
    """
    渲染对话区。question 由 app.py 顶层的 st.chat_input 传入；
    为空时仅回放历史消息，不发起新一轮问答。
    session 结构（app.py 维护）：
        {session_id, messages: [...], traces: [...]}
    trace_placeholder：右栏占位槽，流式期间实时刷新推理轨迹（可为 None）。
    """
    if not question:
        _render_history(session["messages"])
        if not session["messages"]:
            st.info("💡 上传论文后开始提问。支持事实、对比、归纳、推理性问题，"
                    "回答将带引用来源标注。")
        return

    # 先回放历史，再追加本轮用户消息
    session["messages"].append({"role": "user", "content": question})
    _render_history(session["messages"])

    with st.chat_message("assistant"):
        events = rag_client.stream_chat(
            question,
            session_id=session["session_id"],
            top_k=int(st.session_state.get("top_k", 5)),
        )
        new_msg = _consume_stream(events, trace_placeholder)

    session["messages"].append(new_msg)
    if new_msg.get("trace"):
        session.setdefault("traces", []).extend(new_msg["trace"])
