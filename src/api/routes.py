"""
API路由定义
定义所有HTTP接口
"""
import os
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from .schemas import (
    QueryRequest, QueryResponse, SearchRequest, GenerateRequest,
    UploadResponse, DocumentInfo, HealthResponse, StatsResponse,
    ReferenceResponse,
)

router = APIRouter()

# 全局服务实例（在app.py中初始化）
_rag_pipeline = None
_vector_store = None
_chunker = None
_loader_manager = None

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv"}


def init_services(rag_pipeline, vector_store, chunker, loader_manager):
    """初始化服务实例"""
    global _rag_pipeline, _vector_store, _chunker, _loader_manager
    _rag_pipeline = rag_pipeline
    _vector_store = vector_store
    _chunker = chunker
    _loader_manager = loader_manager


def _safe_filename(filename: str) -> str:
    """只接受文件名本身，阻止目录穿越并兼容 Windows 路径。"""
    normalized = str(filename or "").replace("\\", "/")
    safe_name = Path(normalized).name
    if not safe_name or safe_name in {".", ".."} or safe_name != normalized:
        raise HTTPException(status_code=400, detail="文件名不合法")
    return safe_name


def _document_filenames(upload_dir: str) -> list[str]:
    """列出真实业务文档，忽略 macOS 资源分叉等隐藏文件。"""
    if not os.path.exists(upload_dir):
        return []
    return sorted(
        filename for filename in os.listdir(upload_dir)
        if not filename.startswith(".")
        and Path(filename).suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
        and os.path.isfile(os.path.join(upload_dir, filename))
    )


def _refresh_retrieval_state() -> None:
    """按向量库当前元数据重建BM25，并使旧语义缓存失效。"""
    if _rag_pipeline is None or _vector_store is None:
        return

    retriever = getattr(_rag_pipeline, "retriever", None)
    bm25 = getattr(retriever, "bm25", None)
    if bm25 is not None:
        if _vector_store.chunks_meta:
            from src.chunking.base_chunker import Chunk

            chunks = [
                Chunk(
                    content=meta["content"],
                    source=meta["source"],
                    page=meta["page"],
                    chunk_id=meta["chunk_id"],
                    chunk_index=meta["chunk_index"],
                )
                for meta in _vector_store.chunks_meta
            ]
            bm25.build_index(chunks)
        else:
            bm25.bm25 = None
            bm25.chunks_meta = []

    cache = getattr(_rag_pipeline, "cache", None)
    if cache is not None:
        cache.clear()


# ========== RAG查询接口 ==========

@router.post("/api/query", response_model=QueryResponse,
             summary="RAG问答查询")
async def query(req: QueryRequest):
    """
    RAG问答查询接口

    - 支持流式和非流式输出
    - 自动检索相关文献并生成回答
    - 回答中带有引用来源标注
    """
    if _rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG服务未就绪，请先上传文档构建索引")

    if req.stream:
        # 流式输出
        async def event_stream():
            for event in _rag_pipeline.query_stream(req.question, req.top_k):
                yield event.to_sse()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用nginx缓冲
            }
        )
    else:
        # 非流式输出
        result = _rag_pipeline.query(req.question, req.top_k)

        # 转换references格式
        refs = []
        for r in result.get("references", []):
            source = r.get("source", "")
            filename = Path(source).name if source else ""
            refs.append(ReferenceResponse(
                source=source,
                filename=filename,
                page=r.get("page", 0),
                content=r.get("content", "")[:200],
                score=r.get("rrf_score", r.get("score", 0)),
                retrieval_method=r.get("retrieval_method", ""),
            ))

        return QueryResponse(
            answer=result["answer"],
            references=refs,
            retrieval_count=result.get("retrieval_count", 0),
            from_cache=result.get("from_cache", False),
            elapsed_time=result.get("elapsed_time", 0),
        )


# ========== 纯检索接口（供Agent调用） ==========

@router.post("/api/search", summary="纯检索（供Agent调用）")
async def search(req: SearchRequest):
    """
    纯检索接口，不经过大模型生成
    返回检索到的相关文档段落
    """
    if _rag_pipeline is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    results = _rag_pipeline.search(req.query, req.top_k)
    return {"results": results, "count": len(results)}


# ========== 纯生成接口（供Agent调用） ==========

@router.post("/api/generate", summary="纯生成（供Agent调用）")
async def generate(req: GenerateRequest):
    """
    纯生成接口，给定问题和上下文直接生成回答
    不经过检索流程
    """
    if _rag_pipeline is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    answer = _rag_pipeline.generate(req.question, req.context)
    return {"answer": answer}


# ========== 文档管理接口 ==========

@router.post("/api/documents/upload", response_model=UploadResponse,
             summary="上传文档")
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档并构建索引

    支持格式: PDF, DOCX, TXT, MD, CSV
    上传后自动解析、切分、向量化
    """
    if _loader_manager is None or _chunker is None or _vector_store is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    # 检查文件格式
    filename = _safe_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，支持: {', '.join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))}"
        )

    # 保存文件
    from src.utils.config import get_config
    upload_dir = get_config()["UPLOAD_DIR"]
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    logger.info(f"文件已保存: {file_path} ({len(content)} bytes)")

    try:
        # 加载文档
        docs = _loader_manager.load(file_path)
        logger.info(f"文档解析完成: {len(docs)} 页")

        # 文本分块
        chunks = _chunker.chunk_documents(docs)
        logger.info(f"文本分块完成: {len(chunks)} 个块")

        # 同名文件按“替换”处理，避免重复上传造成重复向量与引用。
        removed = _vector_store.delete_by_source(file_path)
        if removed:
            logger.info(f"替换同名文档前移除旧向量: {removed} 个")

        # 增量添加到向量索引
        _vector_store.add_chunks(chunks)

        # 文档集变化后同步BM25并清空旧缓存。
        _refresh_retrieval_state()

        # 保存索引
        _vector_store.save()

        return UploadResponse(
            status="success",
            filename=filename,
            pages=len(docs),
            chunks=len(chunks),
            message=f"文档已成功上传并索引: {len(docs)}页, {len(chunks)}个文本块",
        )

    except Exception as e:
        logger.error(f"文档处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.get("/api/documents", summary="获取文档列表")
async def list_documents():
    """获取已上传的文档列表"""
    from src.utils.config import get_config
    upload_dir = get_config()["UPLOAD_DIR"]
    documents = []
    indexed_names = {
        str(meta.get("source", "")).replace("\\", "/").rsplit("/", 1)[-1]
        for meta in (_vector_store.chunks_meta if _vector_store else [])
    }

    if os.path.exists(upload_dir):
        for filename in _document_filenames(upload_dir):
            filepath = os.path.join(upload_dir, filename)
            documents.append(DocumentInfo(
                filename=filename,
                file_size=os.path.getsize(filepath),
                status="indexed" if filename in indexed_names else "pending",
            ))

    return {"documents": documents, "total": len(documents)}


@router.delete("/api/documents/{filename}", summary="删除文档")
async def delete_document(filename: str):
    """删除物理文件及其向量/BM25记录，并使语义缓存失效。"""
    from src.utils.config import get_config
    upload_dir = get_config()["UPLOAD_DIR"]
    filename = _safe_filename(filename)
    file_path = os.path.join(upload_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

    if _vector_store is None:
        raise HTTPException(status_code=503, detail="向量服务未初始化")

    try:
        removed = _vector_store.delete_by_source(file_path)
        _refresh_retrieval_state()
        _vector_store.save()
        os.remove(file_path)
    except Exception as e:
        logger.error(f"删除文档及索引失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")

    logger.info(f"文档已删除: {filename}，同步删除 {removed} 个向量")

    return {
        "status": "success",
        "message": f"文档 {filename} 已删除，同步删除 {removed} 个向量",
    }


# ========== 系统接口 ==========

@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """检查服务健康状态"""
    vector_count = _vector_store.total_vectors if _vector_store else 0
    cache_size = _rag_pipeline.cache.size if _rag_pipeline else 0

    from src.utils.config import get_config
    upload_dir = get_config()["UPLOAD_DIR"]
    doc_count = 0
    if os.path.exists(upload_dir):
        doc_count = len(_document_filenames(upload_dir))

    return HealthResponse(
        status="ok",
        vector_count=vector_count,
        document_count=doc_count,
        cache_size=cache_size,
    )


@router.get("/api/stats", response_model=StatsResponse, summary="系统统计信息")
async def get_stats():
    """获取RAG系统运行统计信息"""
    if _rag_pipeline is None:
        return StatsResponse()

    stats = _rag_pipeline.get_stats()
    cache_stats = stats.get("cache_stats", {})

    return StatsResponse(
        total_queries=stats.get("total_queries", 0),
        cache_hits=stats.get("cache_hits", 0),
        cache_hit_rate=cache_stats.get("hit_rate", 0),
        avg_response_time=stats.get("avg_time", 0),
        total_tokens=stats.get("total_tokens", 0),
        vector_count=_vector_store.total_vectors if _vector_store else 0,
    )
