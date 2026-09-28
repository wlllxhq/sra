"""
流式输出模块
实现SSE（Server-Sent Events）格式的流式输出
支持打字机效果和引用标记动态出现
"""
import json
import time
from dataclasses import dataclass
from loguru import logger


@dataclass
class StreamEvent:
    """流式事件"""
    type: str       # chunk | reference | thinking | done | error
    content: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_sse(self) -> str:
        """转为SSE格式字符串"""
        data = {"type": self.type, "content": self.content}
        if self.metadata:
            data.update(self.metadata)
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def to_dict(self) -> dict:
        """转为字典"""
        data = {"type": self.type, "content": self.content}
        if self.metadata:
            data.update(self.metadata)
        return data


def create_reference_events(references: list[dict]) -> list[StreamEvent]:
    """为检索结果生成引用事件"""
    events = []
    for i, ref in enumerate(references):
        source = ref.get("source", "未知")
        filename = source.split("/")[-1] if "/" in source else source
        filename = filename.split("\\")[-1] if "\\" in filename else filename

        events.append(StreamEvent(
            type="reference",
            content=ref.get("content", "")[:100],
            metadata={
                "ref_index": i + 1,
                "doc": filename,
                "page": ref.get("page", 0),
                "score": ref.get("rrf_score", ref.get("score", 0)),
            }
        ))
    return events


def create_chunk_event(text: str) -> StreamEvent:
    """创建文本片段事件"""
    return StreamEvent(type="chunk", content=text)


def create_done_event(full_answer: str, references: list[dict],
                      elapsed_time: float = 0, token_count: int = 0) -> StreamEvent:
    """创建完成事件"""
    return StreamEvent(
        type="done",
        content=full_answer,
        metadata={
            "reference_count": len(references),
            "elapsed_time": round(elapsed_time, 2),
            "token_count": token_count,
        }
    )


def create_error_event(message: str) -> StreamEvent:
    """创建错误事件"""
    return StreamEvent(type="error", content=message)


def create_thinking_event(message: str) -> StreamEvent:
    """创建思考过程事件（展示Agent推理过程）"""
    return StreamEvent(type="thinking", content=message)
