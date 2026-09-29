"""
模块四 · 系统集成与前端展示 —— Streamlit 主入口 app.py
======================================================
作业要求对应：
    5.4.1 RAG + Agent 深度融合     → services/rag_client.py 的 USE_AGENT 开关与事件归一化
    5.4.2 可观测性与监控           → components/trace_panel.py（引用/轨迹树/Token/耗时/命中率）
    5.4.3 前端界面开发             → 本文件三栏布局 + 会话管理（新建/切换/删除，JSON 落盘）
    5.4.4 端到端联调与数据格式统一 → 前端仅消费 NormalizedEvent 单一契约，session_id 全链路透传

启动方式（两个终端）：
    终端1（后端）:  python main.py
    终端2（前端）:  streamlit run src/frontend/app.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------- 路径与导入 ----------
_FRONTEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FRONTEND_DIR.parents[1]          # smart-research-assistant/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.frontend.services import rag_client       # noqa: E402
from src.frontend.components import chat, doc_panel, trace_panel  # noqa: E402

SESSIONS_DIR = Path(os.getenv("RAG_SESSIONS_DIR", str(_FRONTEND_DIR / "sessions")))  # 会话持久化目录（测试可用环境变量隔离）
SESSIONS_DIR.mkdir(exist_ok=True, parents=True)

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="智能科研助理 · RAG+Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 轻量自定义样式（浅色主题兜底：.streamlit/config.toml 为主，此处双保险）
st.markdown("""
<style>
.stApp { background-color: #f7f8fa; }
header[data-testid="stHeader"] { background-color: #f7f8fa; }
.small-caption { font-size: 0.8rem; color: #888; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# 会话持久化（5.4.3 对话历史管理：新建会话、切换历史会话、删除会话）
# ===========================================================================

def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _load_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"session_id": session_id,
            "title": "新会话",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "messages": []}


def _save_session(session: dict) -> None:
    session["updated_at"] = datetime.now().isoformat(timespec="seconds")
    try:
        _session_path(session["session_id"]).write_text(
            json.dumps(session, ensure_ascii=False, indent=2),
            encoding="utf-8")
    except OSError as e:
        st.toast(f"会话保存失败：{e}", icon="⚠️")


def _list_sessions() -> list[dict]:
    """按更新时间倒序列出全部会话元信息。"""
    items = []
    for p in SESSIONS_DIR.glob("*.json"):
        # macOS 在外置盘上可能生成 AppleDouble 元数据文件（._*.json），
        # 它们不是 UTF-8 JSON，会话列表应直接忽略。
        if p.name.startswith("._"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            items.append({
                "session_id": data.get("session_id", p.stem),
                "title": data.get("title", "未命名"),
                "updated_at": data.get("updated_at", ""),
                "count": len(data.get("messages", [])),
            })
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items


def _new_session() -> str:
    return f"s_{uuid.uuid4().hex[:8]}"


def _session_has_content(session: dict) -> bool:
    """空会话（无任何消息）不落盘，避免新建/切换/删除时留下大量空会话文件。"""
    return bool(session.get("messages"))


def _prune_empty_sessions(keep_id: str | None = None) -> None:
    """清理磁盘上的空会话文件（0 条消息），当前会话 keep_id 除外。"""
    for p in SESSIONS_DIR.glob("*.json"):
        if p.name.startswith("._") or (keep_id and p.stem == keep_id):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if not data.get("messages"):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


# ===========================================================================
# 状态初始化
# ===========================================================================

def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("top_k", 5)
    if "session_id" not in ss:
        existing = _list_sessions()
        ss["session_id"] = existing[0]["session_id"] if existing else _new_session()
    if "session" not in ss or ss.get("session_id") != ss["session"].get("session_id"):
        ss["session"] = _load_session(ss["session_id"])
    # 清理历史遗留的空会话文件（旧版新建/删除会立即落盘空会话）；每个浏览器会话只扫一次
    if not ss.get("_sessions_pruned"):
        _prune_empty_sessions(keep_id=ss["session_id"])
        ss["_sessions_pruned"] = True


_init_state()


# ===========================================================================
# 顶栏：标题 + 通道切换 + 检索参数
# ===========================================================================

# 首次进入：以 .env 的 USE_AGENT 作为默认通道
if "use_agent" not in st.session_state:
    st.session_state["use_agent"] = rag_client.use_agent()

with st.container():
    col_title, col_agent, col_topk = st.columns([4.2, 3.6, 2])
    col_title.title("🔬 智能科研助理")
    with col_agent:
        choice = st.radio(
            "问答通道",
            ["🤖 Agent 编排", "🔌 RAG 直连"],
            index=0 if st.session_state["use_agent"] else 1,
            horizontal=True,
            key="channel_radio",
            help="Agent：ReAct多轮推理+工具+记忆（慢、全面）；直连：一次检索生成（快、省Token）。切换即时生效",
        )
        st.session_state["use_agent"] = choice.startswith("🤖")
    with col_topk:
        st.session_state["top_k"] = st.slider(
            "检索条数 top_k", 1, 20, st.session_state["top_k"],
            help="每次回答参考的文献段落数（透传后端）")

# 把通道选择同步到环境变量——适配层每次提问实时读取，下一问立即生效（无需重启）
os.environ["USE_AGENT"] = "true" if st.session_state["use_agent"] else "false"

# 后端连通性提示条
_probe = rag_client.health()
if _probe.get("status") != "ok":
    st.error(f"⛔ 后端服务不可用（{_probe.get('error', '未知原因')}）。"
             f"请在项目根目录运行 `python main.py` 后刷新本页面。")


# ===========================================================================
# 左栏（st.sidebar）：文档管理 + 会话管理
# ===========================================================================

with st.sidebar:
    doc_panel.render_doc_panel()

    st.divider()
    st.subheader("💬 会话管理")

    sess_col_new, sess_col_del = st.columns([1, 1])
    with sess_col_new:
        if st.button("➕ 新建会话", use_container_width=True):
            # 仅当当前会话有内容才落盘，空会话直接丢弃（不产生新的空会话文件）
            if _session_has_content(st.session_state["session"]):
                _save_session(st.session_state["session"])
            st.session_state["session_id"] = _new_session()
            st.session_state["session"] = _load_session(st.session_state["session_id"])
            _prune_empty_sessions(keep_id=st.session_state["session_id"])
            st.rerun()
    with sess_col_del:
        if st.button("🗑 删除当前", use_container_width=True):
            sid = st.session_state["session_id"]
            try:
                _session_path(sid).unlink(missing_ok=True)
            except OSError as e:
                st.toast(f"删除失败：{e}", icon="⚠️")
            # 删除后回到一个"内存态"空会话，不落盘（否则会凭空多出一个新会话文件）
            st.session_state["session_id"] = _new_session()
            st.session_state["session"] = _load_session(st.session_state["session_id"])
            _prune_empty_sessions(keep_id=st.session_state["session_id"])
            st.rerun()

    sessions = _list_sessions()
    if sessions:
        options = {f"{s['title']} · {s['updated_at'][:16].replace('T', ' ')} · {s['count']}条": s["session_id"]
                   for s in sessions}
        # 若当前会话尚未落盘（新会话无消息），补一个占位标签避免 selectbox 异常
        labels = list(options.keys())
        default_idx = 0
        cur_sid = st.session_state["session_id"]
        if cur_sid not in options.values():
            labels.insert(0, "（当前新会话）")
            options["（当前新会话）"] = cur_sid
        else:
            default_idx = list(options.values()).index(cur_sid)

        choice_label = st.selectbox("切换历史会话", labels, index=default_idx,
                                    label_visibility="collapsed")
        picked_sid = options[choice_label]
        if picked_sid != cur_sid:
            # 切换前保存当前会话（仅当有内容，避免留下空会话文件）
            if _session_has_content(st.session_state["session"]):
                _save_session(st.session_state["session"])
            st.session_state["session_id"] = picked_sid
            st.session_state["session"] = _load_session(picked_sid)
            _prune_empty_sessions(keep_id=picked_sid)
            st.rerun()


# ===========================================================================
# 主区域：中央对话（2/3） + 右侧可观测面板（1/3）
# ===========================================================================

col_chat, col_trace = st.columns([2.3, 1], gap="medium")

# chat_input 必须在顶层调用（Streamlit 约束），先取问题再分发渲染
question = st.chat_input("向知识库提问，例如：这篇论文的主要方法是什么？")

session = st.session_state["session"]

# 两栏各自独立滚动：给每个面板一个固定高度的滚动容器，右栏内容再长也能单独滚到底，
# 不必被中间对话的长度拖着走（高度可用 RAG_PANEL_HEIGHT 环境变量覆盖，单位 px）。
PANEL_HEIGHT = int(os.getenv("RAG_PANEL_HEIGHT", "760"))

with col_chat:
    st.markdown("**💬 对话**")
    chat_box = st.container(height=PANEL_HEIGHT)

with col_trace:
    st.markdown("**📡 可观测性**")
    trace_box = st.container(height=PANEL_HEIGHT)

# 占位槽必须在事件流消费前创建（脚本自上而下执行），且放在右栏滚动容器内：
# 流式期间由 chat 实时重绘推理轨迹，流结束后整体替换为完整面板（同一槽位，不重复）。
with trace_box:
    panel_slot = st.empty()

with chat_box:
    chat.render_chat(session, question or "", panel_slot)

with panel_slot.container():
    trace_panel.render_trace_panel(session)

# 有新回合时自动更新会话标题（取首问前 18 字）并落盘
if question:
    user_msgs = [m for m in session["messages"] if m["role"] == "user"]
    if len(user_msgs) == 1:
        session["title"] = question[:18] or "新会话"
    _save_session(session)
