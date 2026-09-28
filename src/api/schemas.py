"""
API请求/响应模型定义
使用Pydantic进行数据验证
"""
from pydantic import BaseModel, Field
from typing import Optional


# ---- 请求模型 ----

class QueryRequest(BaseModel):
    """RAG查询请求"""
    question: str = Field(..., description="用户问题", min_length=1)
    session_id: str = Field(default="default", description="会话ID")
    top_k: int = Field(default=5, ge=1, le=20, description="返回参考文档数")
    stream: bool = Field(default=True, description="是否流式输出")


class SearchRequest(BaseModel):
    """纯检索请求（供Agent调用）"""
    query: str = Field(..., description="检索查询", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class GenerateRequest(BaseModel):
    """纯生成请求（供Agent调用）"""
    question: str = Field(..., description="问题")
    context: list[dict] = Field(default=[], description="上下文")


# ---- 响应模型 ----

class ReferenceResponse(BaseModel):
    """引用来源"""
    source: str = Field(default="", description="文档路径")
    filename: str = Field(default="", description="文件名")
    page: int = Field(default=0, description="页码")
    content: str = Field(default="", description="段落内容")
    score: float = Field(default=0.0, description="相关性分数")
    retrieval_method: str = Field(default="", description="检索方法")


class QueryResponse(BaseModel):
    """RAG查询响应（非流式）"""
    answer: str = Field(..., description="生成的回答")
    references: list[ReferenceResponse] = Field(default=[], description="引用来源列表")
    retrieval_count: int = Field(default=0, description="检索到的文档数")
    from_cache: bool = Field(default=False, description="是否来自缓存")
    elapsed_time: float = Field(default=0.0, description="响应耗时（秒）")


class DocumentInfo(BaseModel):
    """文档信息"""
    filename: str
    file_size: int = 0
    pages: int = 0
    chunks: int = 0
    status: str = "unknown"  # indexed | processing | failed


class UploadResponse(BaseModel):
    """上传响应"""
    status: str = "success"
    filename: str
    pages: int = 0
    chunks: int = 0
    message: str = ""


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    vector_count: int = 0
    document_count: int = 0
    cache_size: int = 0


class StatsResponse(BaseModel):
    """系统统计信息"""
    total_queries: int = 0
    cache_hits: int = 0
    cache_hit_rate: float = 0.0
    avg_response_time: float = 0.0
    total_tokens: int = 0
    vector_count: int = 0
    document_count: int = 0
