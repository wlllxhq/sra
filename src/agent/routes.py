"""
Agent路由
=========
提供 /api/agent/query 和 /api/agent/stats 两个端点。

端点说明:
    POST /api/agent/query  — Agent问答（SSE流式或非流式）
    GET  /api/agent/stats  — Agent专属统计指标

请求体沿用后端现有 QueryRequest（不修改既有字段名，交接说明红线1）。
"""
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from src.api.schemas import QueryRequest, QueryResponse, ReferenceResponse

router = APIRouter(prefix="/api/agent", tags=["Agent"])

# 全局Agent流水线（在app.py启动时注入）
_agent_pipeline = None


def init_agent_routes(agent_pipeline):
    """注入Agent流水线实例"""
    global _agent_pipeline
    _agent_pipeline = agent_pipeline
    logger.info("Agent路由初始化完成")


@router.post("/query", summary="Agent问答（ReAct循环）")
async def agent_query(req: QueryRequest):
    """
    Agent问答接口

    - 基于ReAct核心循环的多轮推理
    - 支持8个工具调用
    - 基于session_id的会话记忆隔离
    - 流式输出SSE事件（thought/action/observation/chunk/done）
    """
    if _agent_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Agent服务未就绪，请检查初始化"
        )

    if req.stream:
        # 流式输出
        async def event_stream():
            for sse_event in _agent_pipeline.query_stream(
                question=req.question,
                session_id=req.session_id,
                top_k=req.top_k,
            ):
                yield sse_event

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        # 非流式输出
        result = _agent_pipeline.query(
            question=req.question,
            session_id=req.session_id,
            top_k=req.top_k,
        )

        # 转换references格式
        refs = []
        for r in result.get("references", []):
            refs.append(ReferenceResponse(
                source=r.get("source", ""),
                filename=r.get("filename", ""),
                page=r.get("page", 0),
                content=r.get("content", "")[:200],
                score=r.get("score", 0),
            ))

        return QueryResponse(
            answer=result["answer"],
            references=refs,
            retrieval_count=result.get("retrieval_count", 0),
            from_cache=result.get("from_cache", False),
            elapsed_time=result.get("elapsed_time", 0),
        )


@router.get("/stats", summary="Agent专属统计指标")
async def agent_stats():
    """
    Agent级统计指标

    包括:
    - 总查询数 / 平均响应时间
    - 总推理轮次 / 平均轮次
    - 工具调用统计（成功/失败/按工具分布）
    - 活跃会话数
    """
    if _agent_pipeline is None:
        return {
            "total_queries": 0,
            "avg_response_time": 0,
            "total_tool_calls": 0,
            "tool_success_rate": 0,
            "active_sessions": 0,
        }

    return _agent_pipeline.get_stats()
