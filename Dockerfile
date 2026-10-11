# =============================================================================
# 智能科研助理 —— 统一镜像（后端 FastAPI + 前端 Streamlit 共用同一镜像）
#   构建：docker build -t smart-research-assistant .
#   运行：docker compose up -d   （见 docker-compose.yml）
# =============================================================================
FROM python:3.10-slim

# 基础环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    TZ=Asia/Shanghai

WORKDIR /app

# 系统依赖：faiss / PDF 解析所需运行库 + 编译工具（个别依赖需本地编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 依赖分层安装：先装 requirements，之后改代码不会触发重装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

# 预下载本地模型（Embedding + Reranker，约 1.3G），使镜像自包含、运行时可离线
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-large-zh-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 复制源码（.dockerignore 已排除 .env / .venv / uploads / index / logs 等）
COPY . .

EXPOSE 8000 8501

# 默认启动后端（前端在 docker-compose.yml 里用 command 覆盖）
CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
