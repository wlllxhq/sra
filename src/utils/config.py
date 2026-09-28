"""
配置管理
统一读取.env文件和环境变量
"""
import os
from dotenv import load_dotenv

# 设置HuggingFace国内镜像（解决模型下载超时问题）
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 加载.env文件
_env_loaded = False


def _ensure_env_loaded():
    global _env_loaded
    if not _env_loaded:
        # 查找.env文件（从项目根目录）
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            # 尝试当前目录
            load_dotenv()
        _env_loaded = True


def get_config() -> dict:
    """获取所有配置项"""
    _ensure_env_loaded()
    return {
        # API配置
        "SILICONFLOW_API_KEY": os.getenv("SILICONFLOW_API_KEY", ""),
        "SILICONFLOW_BASE_URL": os.getenv("SILICONFLOW_BASE_URL",
                                           "https://api.siliconflow.cn/v1"),

        # 模型配置
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
        "CHAT_MODEL": os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "RERANKER_MODEL": os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "VECTOR_DIMENSION": os.getenv("VECTOR_DIMENSION", "1024"),

        # 检索配置
        "DEFAULT_CHUNK_SIZE": int(os.getenv("DEFAULT_CHUNK_SIZE", "512")),
        "DEFAULT_CHUNK_OVERLAP": int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50")),
        "DEFAULT_TOP_K": int(os.getenv("DEFAULT_TOP_K", "5")),
        "RRF_K": int(os.getenv("RRF_K", "60")),

        # 路径配置
        "UPLOAD_DIR": os.getenv("UPLOAD_DIR", "uploads"),
        "INDEX_DIR": os.getenv("INDEX_DIR", "index"),

        # 服务配置
        "HOST": os.getenv("HOST", "0.0.0.0"),
        "PORT": int(os.getenv("PORT", "8000")),
    }
