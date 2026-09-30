"""
多轮对话记忆管理
================
基于session_id的会话隔离 + 滑动窗口 + 摘要压缩。

记忆结构:
    - 短期记忆: 最近N轮对话（滑动窗口）
    - 长期记忆: 会话摘要（当短期记忆满时自动压缩）
    - 上下文注入: 将记忆注入ReAct系统提示词
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List
from loguru import logger


@dataclass
class Message:
    """单条对话消息"""
    role: str  # "user" | "assistant" | "tool"
    content: str
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class Session:
    """会话记忆"""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    summary: str = ""  # 长期摘要
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "messages": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in self.messages
            ],
            "summary": self.summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class MemoryManager:
    """
    记忆管理器

    功能:
        1. 会话隔离: 不同session_id互不串话
        2. 滑动窗口: 保留最近window_size轮对话
        3. 摘要压缩: 超出窗口时自动压缩旧对话为摘要
        4. 指代消解: 在上下文中保留最近的实体，支持"它"、"这个"等指代
    """

    def __init__(self, window_size: int = 10, rag_pipeline=None):
        """
        Args:
            window_size: 滑动窗口大小（保留最近N轮对话，每轮=user+assistant）
            rag_pipeline: RAGPipeline实例（用于生成摘要）
        """
        self.window_size = window_size
        self._rag_pipeline = rag_pipeline
        self.sessions: Dict[str, Session] = defaultdict(lambda: None)

    def get_session(self, session_id: str) -> Session:
        """获取或创建会话"""
        if session_id not in self.sessions or self.sessions[session_id] is None:
            import time
            self.sessions[session_id] = Session(
                session_id=session_id,
                created_at=time.time(),
                updated_at=time.time(),
            )
            logger.info(f"创建新会话: {session_id}")
        return self.sessions[session_id]

    def add_user_message(self, session_id: str, content: str) -> None:
        """添加用户消息"""
        import time
        session = self.get_session(session_id)
        session.messages.append(Message(
            role="user",
            content=content,
            timestamp=time.time(),
        ))
        session.updated_at = time.time()
        logger.debug(f"[{session_id}] 用户消息: {content[:50]}...")

    def add_assistant_message(self, session_id: str, content: str,
                              metadata: dict = None) -> None:
        """添加助手回复"""
        import time
        session = self.get_session(session_id)
        session.messages.append(Message(
            role="assistant",
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
        ))
        session.updated_at = time.time()

        # 检查是否需要压缩
        self._maybe_compress(session)

        logger.debug(f"[{session_id}] 助手回复: {content[:50]}...")

    def _maybe_compress(self, session: Session) -> None:
        """当消息数超过窗口大小时，压缩旧消息为摘要"""
        # 每轮包含user+assistant两条消息
        max_messages = self.window_size * 2

        if len(session.messages) <= max_messages:
            return

        # 需要压缩的消息数
        compress_count = len(session.messages) - max_messages
        if compress_count <= 0:
            return

        # 取出要压缩的消息（从最早开始）
        to_compress = session.messages[:compress_count]
        session.messages = session.messages[compress_count:]

        # 生成摘要
        compress_text = "\n".join([
            f"{'用户' if m.role == 'user' else '助手'}: {m.content}"
            for m in to_compress
        ])

        # 如果已有摘要，合并压缩
        if session.summary:
            compress_text = f"【之前的对话摘要】\n{session.summary}\n\n【新的对话】\n{compress_text}"

        # 调用大模型生成摘要
        if self._rag_pipeline:
            try:
                summary_prompt = "请用不超过200字总结以下对话的关键信息，保留重要的实体、结论和上下文："
                context = [{"content": compress_text, "source": "对话历史", "page": 1}]
                new_summary = self._rag_pipeline.generate(summary_prompt, context)

                if new_summary and len(new_summary) > 10:
                    session.summary = new_summary
                    logger.info(f"[{session.summary[:50]}...")
                else:
                    # 降级：简单截取
                    session.summary = compress_text[:200] + "..."
                    logger.warning(f"摘要生成失败，使用降级摘要")
            except Exception as e:
                logger.error(f"摘要压缩失败: {e}")
                # 降级：简单截取
                session.summary = compress_text[:200] + "..."
        else:
            # 无RAGPipeline时直接截取
            session.summary = compress_text[:200] + "..."
            logger.warning("RAGPipeline未初始化，使用降级摘要")

    def get_context_prompt(self, session_id: str) -> str:
        """
        获取会话上下文（供ReAct系统提示词注入）

        Returns:
            str: 格式化的上下文文本，包含：
                - 会话摘要（如果有）
                - 最近N轮对话
        """
        session = self.get_session(session_id)

        parts = []

        # 1. 会话摘要（长期记忆）
        if session.summary:
            parts.append(f"【对话历史摘要】\n{session.summary}")

        # 2. 最近对话（短期记忆）
        if session.messages:
            recent = session.messages[-(self.window_size * 2):]  # 最近N轮
            recent_text = "\n".join([
                f"{'用户' if m.role == 'user' else '助手'}: {m.content[:300]}"
                for m in recent
            ])
            parts.append(f"【最近对话】\n{recent_text}")

        if not parts:
            return "（这是一个新会话，暂无历史记录）"

        return "\n\n".join(parts)

    def clear_session(self, session_id: str) -> None:
        """清空指定会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"会话已清空: {session_id}")

    def get_session_stats(self) -> dict:
        """获取所有会话的统计信息"""
        stats = {
            "total_sessions": len(self.sessions),
            "sessions": [],
        }

        for sid, session in self.sessions.items():
            if session:
                stats["sessions"].append({
                    "session_id": sid,
                    "message_count": len(session.messages),
                    "has_summary": bool(session.summary),
                    "updated_at": session.updated_at,
                })

        return stats


# 全局记忆管理器实例（在routes.py中初始化）
_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """获取全局记忆管理器实例"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(window_size=10)
    return _memory_manager


def init_memory_manager(rag_pipeline=None, window_size: int = 10) -> MemoryManager:
    """初始化全局记忆管理器"""
    global _memory_manager
    _memory_manager = MemoryManager(window_size=window_size, rag_pipeline=rag_pipeline)
    logger.info(f"记忆管理器初始化完成: window_size={window_size}")
    return _memory_manager
