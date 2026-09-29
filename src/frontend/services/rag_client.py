"""
模块四 · 前端服务适配层 rag_client.py
=====================================

职责：
1. 封装后端（模块一/二 FastAPI 服务）全部 REST + SSE 接口，前端组件禁止直接发 HTTP 请求；
2. 将 SSE 原始事件归一化为标准事件流（NormalizedEvent），UI 层只消费归一化事件；
3. 预留模块三（Agent 决策层）接入位：
   - USE_AGENT 开关：false → /api/query（RAG 直连，当前默认）
                     true  → /api/agent/query（Agent ReAct 循环，模块三交付后启用）
   - Agent 模式端点未就绪时自动降级回 RAG 直连，并在事件流中明确告知；
   - 按《模块三事件协议》解析 thought/action/observation 事件，未知类型不崩溃。

环境变量（.env 或系统变量）：
    RAG_API_BASE_URL   后端地址，默认 http://127.0.0.1:8000
    USE_AGENT          是否走 Agent 通道，默认 false
    RAG_TIMEOUT_CONNECT 连接超时秒数，默认 10
    RAG_TIMEOUT_READ    读取超时秒数，默认 120（大模型生成较慢）
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import requests

# Streamlit 进程不会自动读 .env（后端由 src/utils/config.py 加载），此处显式加载项目根 .env。
# 已有环境变量优先（命令行 set 的不被覆盖）。
_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)
except ImportError:
    # 兜底：当前解释器未装 python-dotenv 时，手动解析 KEY=VALUE（跳过注释/空行，不覆盖已有变量）
    if _ENV_FILE.exists():
        try:
            for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                _v = _v.split("#", 1)[0].strip().strip('"').strip("'")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v
        except OSError:
            pass

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")


def use_agent() -> bool:
    """每次调用实时读取 USE_AGENT（Streamlit 每次重跑都会重新执行本函数，
    修改 .env 后只需刷新页面，无需重启进程）。"""
    return os.getenv("USE_AGENT", "false").strip().lower() in ("1", "true", "yes")

CONNECT_TIMEOUT = float(os.getenv("RAG_TIMEOUT_CONNECT", "10"))
READ_TIMEOUT = float(os.getenv("RAG_TIMEOUT_READ", "120"))

RAG_QUERY_ENDPOINT = "/api/query"          # 模块二提供
AGENT_QUERY_ENDPOINT = "/api/agent/query"  # 模块三预留（交付前不存在，自动降级）


def _timeouts() -> tuple:
    return (CONNECT_TIMEOUT, READ_TIMEOUT)


# ---------------------------------------------------------------------------
# 归一化事件定义 —— UI 层唯一消费的数据契约（5.4.1 数据格式统一）
# ---------------------------------------------------------------------------

# kind 取值：
#   status      系统状态提示        （来自后端 thinking 事件）
#   reference   引用来源卡片        （来自后端 reference 事件）
#   delta       回答增量文本块      （来自后端 chunk 事件）
#   thought     Agent 推理思考      （模块三协议，预留）
#   action      Agent 工具调用      （模块三协议，预留）
#   observation Agent 工具返回观察  （模块三协议，预留）
#   final       生成完成汇总        （来自后端 done 事件）
#   error       错误信息            （来自后端 error 事件或本地异常）
#   unknown     未知/未来新增类型   （原样携带，UI 降级为纯文本展示，绝不崩溃）


@dataclass
class NormalizedEvent:
    """归一化后的流式事件。所有前端组件只认这个结构。"""
    kind: str
    text: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    # ---- 引用卡片专用字段（kind == "reference"）----
    ref_index: int = 0
    doc: str = ""
    page: int = 0
    score: float = 0.0

    # ---- final 事件专用字段 ----
    elapsed_time: float = 0.0
    token_count: int = 0
    reference_count: int = 0
    # 本轮按工具分别统计的Token消耗（Agent通道 done 事件携带；RAG直连为空）
    tool_token_counts: Dict[str, int] = field(default_factory=dict)

    # ---- trace 事件专用字段（thought/action/observation）----
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


# 后端原始 SSE type → 归一化 kind 映射表
# 未知 type 一律落到 unknown，保证向后兼容（模块三上线、后端扩展事件都不用改前端）
_TYPE_MAP = {
    "thinking": "status",
    "reference": "reference",
    "chunk": "delta",
    "done": "final",
    "error": "error",
    # ↓ 模块三事件协议（《frontend/docs/agent_event_protocol.md》）
    "thought": "thought",
    "action": "action",
    "observation": "observation",
}


def _normalize(raw: Dict[str, Any]) -> NormalizedEvent:
    """把一条后端原始事件 dict 转为 NormalizedEvent。"""
    rtype = str(raw.get("type", ""))
    kind = _TYPE_MAP.get(rtype, "unknown")
    content = raw.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)

    ev = NormalizedEvent(kind=kind, text=content, payload=raw)

    if kind == "reference":
        ev.ref_index = int(raw.get("ref_index", 0) or 0)
        ev.doc = str(raw.get("doc", ""))
        ev.page = int(raw.get("page", 0) or 0)
        try:
            ev.score = float(raw.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            ev.score = 0.0
    elif kind == "final":
        ev.elapsed_time = float(raw.get("elapsed_time", 0.0) or 0.0)
        ev.token_count = int(raw.get("token_count", 0) or 0)
        ev.reference_count = int(raw.get("reference_count", 0) or 0)
        raw_tokens = raw.get("tool_token_counts")
        if isinstance(raw_tokens, dict):
            for tool, val in raw_tokens.items():
                try:
                    ev.tool_token_counts[str(tool)] = int(val or 0)
                except (TypeError, ValueError):
                    continue
    elif kind in ("thought", "action", "observation"):
        # 字段容错解析：协议约定的键取不到时用常见别名兜底，再取不到留空
        ev.tool = str(raw.get("tool", raw.get("tool_name", raw.get("action", ""))) or "")
        args = raw.get("args", raw.get("arguments", raw.get("params", None)))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        ev.args = args if isinstance(args, dict) else {}
    return ev


# ---------------------------------------------------------------------------
# SSE 帧解析
# ---------------------------------------------------------------------------

def _iter_sse_events(resp: requests.Response) -> Iterator[Dict[str, Any]]:
    """
    逐行读取 SSE 流，按空行分帧，解析 data: 前缀后的 JSON。
    容忍：多行 data、注释行(: ping)、JSON 解析失败（跳过并报 error）。
    """
    data_lines: List[str] = []
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        line = line.strip("\r")

        if line == "":  # 帧边界
            if data_lines:
                blob = "\n".join(data_lines)
                data_lines = []
                try:
                    obj = json.loads(blob)
                    if isinstance(obj, dict):
                        yield obj
                    else:
                        yield {"type": "error",
                               "content": f"SSE 帧非对象: {blob[:200]}"}
                except json.JSONDecodeError:
                    yield {"type": "error",
                           "content": f"SSE JSON 解析失败: {blob[:200]}"}
            continue

        if line.startswith(":"):       # SSE 注释（keep-alive ping）
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        # 其他字段（event:/id:/retry:）当前后端未使用，忽略


# ---------------------------------------------------------------------------
# 核心接口 1：流式问答（前端对话主流程）
# ---------------------------------------------------------------------------

def stream_chat(question: str,
                session_id: str = "default",
                top_k: int = 5) -> Iterator[NormalizedEvent]:
    """
    发起一次问答并返回归一化事件流。

    USE_AGENT=false：POST /api/query（模块二 RAG 直连）
    USE_AGENT=true ：POST /api/agent/query（模块三），
                     端点不存在(404)/不可用时自动降级 /api/query 并发 status 告知。

    本函数永不抛异常——所有网络/解析错误转为 error 事件，由 UI 层展示（5.3.3 错误恢复的前端侧对应）。
    """
    body = {
        "question": question,
        "session_id": session_id,   # 当前后端未消费；模块三接管会话记忆后立即生效
        "top_k": top_k,
        "stream": True,
    }

    endpoint = AGENT_QUERY_ENDPOINT if use_agent() else RAG_QUERY_ENDPOINT

    try:
        resp = requests.post(f"{DEFAULT_BASE_URL}{endpoint}",
                             json=body, stream=True, timeout=_timeouts())

        # Agent 端点尚未交付：降级到 RAG 直连，明确告知用户（而非静默切换）
        if resp.status_code == 404 and endpoint == AGENT_QUERY_ENDPOINT:
            resp.close()
            yield NormalizedEvent(
                kind="status",
                text="⚠️ Agent 通道尚未就绪（/api/agent/query 404），已自动降级为 RAG 直连模式。")
            resp = requests.post(f"{DEFAULT_BASE_URL}{RAG_QUERY_ENDPOINT}",
                                 json=body, stream=True, timeout=_timeouts())

        if resp.status_code != 200:
            detail = resp.text[:300]
            resp.close()
            yield NormalizedEvent(
                kind="error",
                text=f"后端返回 HTTP {resp.status_code}：{detail}")
            return

        got_final = False
        for raw in _iter_sse_events(resp):
            ev = _normalize(raw)
            if ev.kind == "final":
                got_final = True
            yield ev
        resp.close()

        if not got_final:
            # 流意外结束（代理截断/服务重启等），前端不至于永远转圈
            yield NormalizedEvent(
                kind="error",
                text="响应流意外结束（未收到完成事件），请重试或查看后端日志。")

    except requests.exceptions.ConnectionError:
        yield NormalizedEvent(
            kind="error",
            text=f"无法连接后端服务 {DEFAULT_BASE_URL}，请确认已运行 python main.py。")
    except requests.exceptions.Timeout:
        yield NormalizedEvent(
            kind="error",
            text=f"请求超时（连接 {CONNECT_TIMEOUT:.0f}s / 读取 {READ_TIMEOUT:.0f}s），"
                 f"可能模型首次加载或网络较慢，请重试。")
    except Exception as e:  # 最后防线：任何未预见异常都不允许击穿 UI
        yield NormalizedEvent(kind="error", text=f"前端服务层异常：{e}")


# ---------------------------------------------------------------------------
# 核心接口 2：非流式问答（调试与评测批跑用，QA 模块五会复用）
# ---------------------------------------------------------------------------

def chat(question: str, session_id: str = "default",
         top_k: int = 5) -> Dict[str, Any]:
    """非流式 POST /api/query，返回后端 QueryResponse 原样 dict。异常时返回 {error: ...}。"""
    body = {"question": question, "session_id": session_id,
            "top_k": top_k, "stream": False}
    endpoint = AGENT_QUERY_ENDPOINT if use_agent() else RAG_QUERY_ENDPOINT
    try:
        resp = requests.post(f"{DEFAULT_BASE_URL}{endpoint}", json=body,
                             timeout=_timeouts())
        if resp.status_code == 404 and endpoint == AGENT_QUERY_ENDPOINT:
            resp = requests.post(f"{DEFAULT_BASE_URL}{RAG_QUERY_ENDPOINT}",
                                 json=body, timeout=_timeouts())
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 文档管理（5.4.3 左侧面板）
# ---------------------------------------------------------------------------

def upload_document(file_bytes: bytes, filename: str,
                    timeout: Optional[float] = None) -> Dict[str, Any]:
    """
    上传文档并建索引。大 PDF（CPU 向量化约 8s/块）耗时很长，默认读取超时放宽到 15 分钟。
    返回 {status, filename, pages, chunks, message} 或 {error: ...}。
    """
    try:
        resp = requests.post(
            f"{DEFAULT_BASE_URL}/api/documents/upload",
            files={"file": (filename, file_bytes)},
            timeout=(CONNECT_TIMEOUT, timeout or 900),
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def list_documents() -> List[Dict[str, Any]]:
    """GET /api/documents → [{filename, file_size, status}, ...]；失败返回空列表并由 status_error 记录。"""
    try:
        resp = requests.get(f"{DEFAULT_BASE_URL}/api/documents",
                            timeout=_timeouts())
        resp.raise_for_status()
        return resp.json().get("documents", [])
    except Exception:
        return []


def delete_document(filename: str) -> Dict[str, Any]:
    """DELETE /api/documents/{filename}。文件名 URL 编码由 requests 处理。"""
    try:
        resp = requests.delete(f"{DEFAULT_BASE_URL}/api/documents/{filename}",
                               timeout=_timeouts())
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 系统监控（5.4.2 可观测性 + 健康检查）
# ---------------------------------------------------------------------------

def health() -> Dict[str, Any]:
    """GET /health → {status, vector_count, document_count, cache_size}；不可达时 {status: 'down', error:...}。"""
    try:
        resp = requests.get(f"{DEFAULT_BASE_URL}/health",
                            timeout=(CONNECT_TIMEOUT, 10))
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "down", "error": str(e)}


def stats() -> Dict[str, Any]:
    """GET /api/stats → 查询数/缓存命中/平均耗时/Token/向量数；失败返回 {}。"""
    try:
        resp = requests.get(f"{DEFAULT_BASE_URL}/api/stats",
                            timeout=_timeouts())
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def backend_up() -> bool:
    """快速判断后端是否在（前端启动时探测用）。"""
    return health().get("status") == "ok"


def agent_stats() -> Dict[str, Any]:
    """GET /api/agent/stats → Agent级指标（推理轮次/工具成功率/活跃会话）；
    端点未挂载(404)或失败返回 {}。"""
    try:
        resp = requests.get(f"{DEFAULT_BASE_URL}/api/agent/stats",
                            timeout=_timeouts())
        if resp.status_code != 200:
            return {}
        return resp.json()
    except Exception:
        return {}


__all__ = [
    "NormalizedEvent",
    "stream_chat", "chat",
    "upload_document", "list_documents", "delete_document",
    "health", "stats", "backend_up", "agent_stats",
    "use_agent", "DEFAULT_BASE_URL",
]
