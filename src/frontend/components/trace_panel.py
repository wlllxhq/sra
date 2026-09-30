"""
模块四 · 右栏可观测性面板 trace_panel.py
=========================================
对应作业要求：
    5.4.2 可观测性与监控 —— Token 消耗、检索命中率、响应延迟、工具调用成功率
    5.4.3 右侧面板 —— Agent 推理过程可视化（Thought → Action → Observation 轨迹树）

数据来自两处：
    - 会话内事件流（chat.py 收集的 references / trace / meta）→ 渲染本轮引用与轨迹树
    - GET /api/stats + GET /health → 系统级监控指标（自动轮询刷新）

模块三（Agent 决策层）接入前，轨迹树区域显示占位说明，仅展示后端 thinking 状态；
接入后 thought/action/observation 事件自动出现在此树中，前端零改动。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import streamlit as st

from src.frontend.services import rag_client


# ---------------------------------------------------------------------------
# 本轮引用来源卡片
# ---------------------------------------------------------------------------

def _render_references(refs: List[Dict[str, Any]]) -> None:
    if not refs:
        return
    st.markdown("**📚 本轮引用来源**")
    for r in refs:
        with st.container(border=True):
            head = f"📎 [{r.get('ref_index', '?')}] {r.get('doc', '未知文档')} · 第 {r.get('page', '?')} 页"
            if r.get("score"):
                head += f" · 相关度 {r['score']:.3f}"
            st.markdown(head)
            content = (r.get("content") or "").strip()
            if content:
                st.caption(content[:120] + ("…" if len(content) > 120 else ""))


# ---------------------------------------------------------------------------
# Agent 推理轨迹树（模块三可视化位）
# ---------------------------------------------------------------------------

_TRACE_ICON = {
    "thought": "🧠 Thought",
    "action": "🔧 Action",
    "observation": "👁 Observation",
    "status": "💭 状态",
}


def _render_trace_tree(trace: List[Dict[str, Any]]) -> None:
    st.markdown("**🧭 Agent 推理轨迹**")

    # 会话状态里的 thinking 事件（rag_client 归一化后 kind 可能带 thinking 或 status 两类，
    # 都归入"状态/思考"时间线，兼容模块三上线后真实轨迹）
    status_items = [t for t in trace if t.get("type") in ("thinking", "status")]
    react_items = [t for t in trace if t.get("type") in _TRACE_ICON and t.get("type") not in ("thinking", "status")]

    if not trace:
        st.caption("本轮暂无推理轨迹。发出提问后，这里将实时展示 "
                   "**Thought → Action → Observation** 推理轨迹树。")
        if not rag_client.use_agent():
            st.caption("💡 当前为 RAG 直连；用顶栏“问答通道”切到 Agent 编排，"
                       "即可展示完整推理轨迹树。")
        return

    if react_items:
        for item in react_items:
            icon = _TRACE_ICON.get(item.get("type", ""), "•")
            body = item.get("text", "")
            if item.get("type") == "action" and item.get("tool"):
                head = f"{icon} **{item['tool']}**"
                if item.get("args"):
                    with st.expander(head, expanded=False):
                        st.markdown(f"入参：`{json.dumps(item['args'], ensure_ascii=False)[:300]}`")
                        st.text(body[:200])
                else:
                    st.markdown(head)
            elif item.get("type") == "observation" and item.get("tool"):
                st.markdown(f"{icon} `{item['tool']}` 返回")
                st.caption((body or "")[:150])
            else:
                st.markdown(f"{icon} {body[:120]}")
    else:
        st.caption("系统处理过程（thinking）：")
        for item in status_items:
            st.write(f"- 💭 {item.get('text', '')}")


# ---------------------------------------------------------------------------
# 本轮元数据
# ---------------------------------------------------------------------------

def _render_round_meta(meta: Dict[str, Any]) -> None:
    if not meta:
        return
    st.markdown("**⏱ 本轮指标**")
    c1, c2, c3 = st.columns(3)
    c1.metric("耗时", f"{meta.get('elapsed_time', 0):.1f}s")
    c2.metric("Token 合计", f"{meta.get('token_count', 0)}")
    c3.metric("引用数", f"{meta.get('reference_count', 0)}")


# ---------------------------------------------------------------------------
# Token 消耗（按工具分别统计）：本轮 + 累计
# ---------------------------------------------------------------------------

def _fmt_ms(ms: Any) -> str:
    try:
        ms = float(ms or 0)
    except (TypeError, ValueError):
        return "—"
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    if ms >= 1:
        return f"{ms:.0f}ms"
    return "—"


def _render_tool_token_table(round_meta: Dict[str, Any], astats: Dict[str, Any]) -> None:
    """
    按工具分别统计 Token 消耗：本轮（最近一次提问）+ 累计（后端进程启动至今）。

    归因规则：一轮推理触发某工具调用，则该轮 LLM 消耗记到该工具头上；最终回答
    （Final Answer / 强制收敛）不调用工具，其 Token 计入"(生成轮·不归属工具)"，
    以保证各行合计与总 Token 完全对账。
    """
    round_meta = round_meta or {}
    round_tokens = {str(k): int(v or 0) for k, v in (round_meta.get("tool_token_counts") or {}).items()}
    cum_tokens = {str(k): int(v or 0) for k, v in (astats.get("tool_token_counts") or {}).items()}
    call_counts = astats.get("tool_call_counts") or {}
    avg_ms = astats.get("tool_avg_time_ms") or {}

    round_total = int(round_meta.get("token_count", 0) or 0) or sum(round_tokens.values())
    cum_total = int(astats.get("total_llm_tokens", 0) or 0) or sum(cum_tokens.values())

    if not (round_tokens or cum_tokens or round_total or cum_total):
        return

    st.markdown("**🔢 Token 消耗（按工具分别统计）**")

    tools = sorted(set(round_tokens) | set(cum_tokens) | set(call_counts),
                   key=lambda t: (-round_tokens.get(t, 0), -cum_tokens.get(t, 0), t))

    rows: List[List[str]] = [["工具", "调用次数", "平均耗时", "本轮 Token", "累计 Token"]]
    round_attr = cum_attr = 0
    for tool in tools:
        rt, ct = round_tokens.get(tool, 0), cum_tokens.get(tool, 0)
        round_attr += rt
        cum_attr += ct
        rows.append([tool, str(call_counts.get(tool, 0) or 0),
                     _fmt_ms(avg_ms.get(tool)), str(rt), str(ct)])

    round_gen = max(0, round_total - round_attr)
    cum_gen = max(0, cum_total - cum_attr)
    if round_gen or cum_gen:
        rows.append(["(生成轮·不归属工具)", "—", "—", str(round_gen), str(cum_gen)])

    rows.append(["合计", str(astats.get("total_tool_calls", 0) or 0), "—",
                 str(round_total), str(cum_total)])
    st.table(rows)
    st.caption("本轮=最近一次提问（Agent 通道）；累计=后端进程启动至今。"
               "生成轮为不调用工具的最终作答轮，按差额对账，故合计与总 Token 一致。")


# ---------------------------------------------------------------------------
# 系统级监控（/api/stats + /health）
# ---------------------------------------------------------------------------

def _render_system_monitor(round_meta: Dict[str, Any] | None = None) -> None:
    st.markdown("---")
    st.markdown("**📊 系统监控**")

    health_data = rag_client.health()

    if health_data.get("status") != "ok":
        st.error(f"⛔ 后端服务不可用：{health_data.get('error', '连接失败')}",
                 icon="🔴")
        st.caption("请确认后端已启动：`python main.py`")
        return

    stats_data = rag_client.stats() or {}

    c1, c2 = st.columns(2)
    c1.metric("向量总数", stats_data.get("vector_count", health_data.get("vector_count", 0)))
    c2.metric("文档数", health_data.get("document_count", 0))

    c3, c4 = st.columns(2)
    c3.metric("RAG直连累计提问", stats_data.get("total_queries", 0))
    c4.metric("RAG直连缓存命中", f"{stats_data.get('cache_hits', 0)}")

    if stats_data.get("total_queries", 0) > 0:
        hit_rate = stats_data.get("cache_hit_rate") or 0
        avg_time = stats_data.get("avg_response_time") or 0
        st.metric("缓存命中率", f"{hit_rate:.1%}")
        st.metric("平均响应", f"{avg_time:.2f}s")
        st.metric("累计 Token", stats_data.get("total_tokens", 0))

    if rag_client.use_agent():
        st.caption("🤖 当前走 Agent 通道（USE_AGENT=true）")
        astats = rag_client.agent_stats()
        if astats:
            st.markdown("**🤖 Agent 指标**")
            a1, a2, a3 = st.columns(3)
            a1.metric("Agent提问数", astats.get("total_queries", 0))
            a2.metric("工具成功率", f"{astats.get('tool_success_rate', 0):.1%}")
            a3.metric("平均响应", f"{astats.get('avg_response_time', 0):.1f}s")
            a4, a5 = st.columns(2)
            a4.metric("工具调用总数", astats.get("total_tool_calls", 0))
            a5.metric("活跃会话", astats.get("active_sessions", 0))
            # 按工具分别统计 Token 消耗（本轮 + 累计，作业 5.4.2）
            _render_tool_token_table(round_meta or {}, astats)
    else:
        st.caption("🔌 当前走 RAG 直连（USE_AGENT=false）")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def render_trace_panel(session: Dict[str, Any]) -> None:
    """右栏渲染：最新一轮的引用/轨迹/指标 + 系统监控。"""
    msgs = [m for m in session.get("messages", []) if m.get("role") == "assistant"]
    last = msgs[-1] if msgs else None
    round_meta = (last or {}).get("meta") or {}

    if last:
        _render_references(last.get("references") or [])
        st.divider()
        _render_round_meta(round_meta)
        st.divider()

    # 轨迹树只显示最新一轮（历史轨迹保留在 messages 里可回溯）
    _render_trace_tree((last or {}).get("trace") or [])
    _render_system_monitor(round_meta)
