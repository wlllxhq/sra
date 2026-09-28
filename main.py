"""
智能科研助理 - 主入口

启动方式:
    python main.py              # 启动API服务
    python main.py --cli        # 命令行交互模式
    python main.py --test       # 运行快速测试
"""
import sys
import os

# 将项目根目录加入Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def start_server():
    """启动FastAPI服务"""
    import uvicorn
    from src.utils.config import get_config

    config = get_config()
    host = config["HOST"]
    port = config["PORT"]

    print(f"""
╔══════════════════════════════════════════════════╗
║        智能科研助理 RAG API 服务                  ║
║                                                  ║
║   地址: http://{host}:{port}                       ║
║   文档: http://{host}:{port}/docs                  ║
║                                                  ║
║   Ctrl+C 停止服务                                ║
╚══════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "src.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


def cli_mode():
    """命令行交互模式（用于快速测试，不启动Web服务）"""
    from src.utils.config import get_config
    from src.data_loader.base_loader import DocumentLoaderManager
    from src.data_loader.pdf_loader import PDFLoader
    from src.data_loader.docx_loader import DocxLoader
    from src.data_loader.text_loader import TextLoader
    from src.chunking.recursive_chunker import RecursiveChunker
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import Reranker
    from src.generation.rag_pipeline import RAGPipeline
    from src.generation.cache import SemanticCache

    config = get_config()

    print("=" * 50)
    print("智能科研助理 - 命令行模式")
    print("=" * 50)

    # 初始化
    print("\n[1/4] 初始化文档加载器...")
    loader = DocumentLoaderManager()
    loader.register(PDFLoader())
    loader.register(DocxLoader())
    loader.register(TextLoader())

    print("[2/4] 初始化分块器...")
    chunker = RecursiveChunker(
        chunk_size=config["DEFAULT_CHUNK_SIZE"],
        chunk_overlap=config["DEFAULT_CHUNK_OVERLAP"],
    )

    print("[3/4] 初始化向量存储...")
    vector_store = VectorStore()

    # 尝试加载已有索引
    index_path = os.path.join(config["INDEX_DIR"], "vector")
    if os.path.exists(index_path):
        if vector_store.load(index_path):
            print(f"  已加载索引: {vector_store.total_vectors}个向量")
    else:
        print("  未找到已有索引")

    bm25 = BM25Retriever()
    hybrid = HybridRetriever(vector_store, bm25)
    reranker = Reranker()
    cache = SemanticCache(vector_store=vector_store)
    rag = RAGPipeline(hybrid, reranker, cache)

    print("[4/4] 初始化完成！")
    print()

    # 检查是否需要上传文档
    upload_dir = config["UPLOAD_DIR"]
    if not os.path.exists(upload_dir) or not os.listdir(upload_dir):
        print("提示: uploads目录为空，请先上传文档。")
        print("用法: 输入 !upload <文件路径> 上传文档")
        print()

    # 交互循环
    print("输入问题开始对话，输入 quit 退出")
    print("输入 !upload <路径> 上传文档")
    print("输入 !stats 查看统计信息")
    print("-" * 50)

    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if question.startswith("!upload "):
            file_path = question[8:].strip()
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                continue
            try:
                print(f"正在处理: {file_path}")
                docs = loader.load(file_path)
                print(f"  解析完成: {len(docs)} 页")
                chunks = chunker.chunk_documents(docs)
                print(f"  分块完成: {len(chunks)} 个块")
                vector_store.add_chunks(chunks)
                vector_store.save()
                print(f"  索引完成: 当前共 {vector_store.total_vectors} 个向量")

                # 重建BM25
                from src.chunking.base_chunker import Chunk
                # 需要所有chunk来重建BM25
                # 这里简化处理：从vector_store的meta重建
                print("  BM25索引重建中...")
                all_chunks_for_bm25 = []
                for meta in vector_store.chunks_meta:
                    c = Chunk(
                        content=meta["content"],
                        source=meta["source"],
                        page=meta["page"],
                        chunk_id=meta["chunk_id"],
                        chunk_index=meta["chunk_index"],
                    )
                    all_chunks_for_bm25.append(c)
                bm25.build_index(all_chunks_for_bm25)
                print("  全部完成！可以开始提问了。")

            except Exception as e:
                print(f"处理失败: {e}")
            continue

        if question.startswith("!stats"):
            stats = rag.get_stats()
            print(f"\n系统统计:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            continue

        if vector_store.total_vectors == 0:
            print("请先上传文档 (!upload <文件路径>)")
            continue

        # RAG查询
        try:
            result = rag.query(question)
            print(f"\n助理: {result['answer']}")
            if result.get("references"):
                print(f"\n--- 引用来源 ({len(result['references'])}条) ---")
                for i, ref in enumerate(result["references"], 1):
                    source = ref.get("source", "")
                    filename = os.path.basename(source) if source else "未知"
                    print(f"  [{i}] {filename} 第{ref.get('page', '?')}页 "
                          f"(相关度: {ref.get('rrf_score', ref.get('score', 0)):.4f})")
            print(f"\n[耗时 {result.get('elapsed_time', 0):.2f}s"
                  f"{' (缓存)' if result.get('from_cache') else ''}]")

        except Exception as e:
            print(f"查询失败: {e}")


def run_test():
    """快速测试"""
    print("=" * 50)
    print("快速测试")
    print("=" * 50)

    # 测试1: 文档加载
    print("\n[测试1] 文档加载器")
    from src.data_loader.pdf_loader import PDFLoader
    from src.data_loader.text_loader import TextLoader
    print("  PDFLoader: OK")
    print("  TextLoader: OK")

    # 测试2: 文本分块
    print("\n[测试2] 文本分块器")
    from src.chunking.recursive_chunker import RecursiveChunker
    from src.chunking.fixed_chunker import FixedChunker
    from src.chunking.semantic_chunker import SemanticChunker
    from src.data_loader.base_loader import Document

    test_doc = Document(
        content="这是测试文本。" * 100,
        source="test.txt",
        page=1,
    )

    for name, chunker in [("Recursive", RecursiveChunker(512, 50)),
                           ("Fixed", FixedChunker(512, 50)),
                           ("Semantic", SemanticChunker(512, 50))]:
        chunks = chunker.chunk_documents([test_doc])
        print(f"  {name}Chunker: {len(chunks)}个块 OK")

    print("\n所有基础测试通过！")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--cli":
            cli_mode()
        elif mode == "--test":
            run_test()
        else:
            print(f"未知参数: {mode}")
            print("用法: python main.py [--cli | --test]")
    else:
        start_server()