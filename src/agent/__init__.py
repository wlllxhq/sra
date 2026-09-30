"""
模块三 · Agent决策层
===================
基于ReAct核心循环的智能Agent，支持工具调用与多轮对话记忆管理。

子模块:
    - react_loop: ReAct核心循环（手写实现）
    - memory: 多轮对话记忆管理（会话隔离 + 滑动窗口）
    - agent_pipeline: Agent流水线（SSE事件输出）
    - tools: 工具集（8个工具）
    - routes: FastAPI路由（/api/agent/query, /api/agent/stats）
"""
