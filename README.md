# 智能科研助理 —— 基于 RAG + Agent 的论文知识库问答系统

面向科研工作者的私有文献问答系统：以 **Agent 为大脑中枢**（ReAct 自主推理与工具调度），
以 **RAG 为知识工具**（混合检索 + 重排序），支持上传 PDF/Word/TXT 论文后以自然语言提问，
回答带**引用溯源**（文档名 + 页码），并完整可视化 Agent 的推理轨迹与运行指标。

南京农业大学 · 生产实习课程实践 ｜ 4 人小组项目

---

## 一、核心特性

- **双通道问答**：`USE_AGENT` 一键切换 RAG 直连（快、省 Token）与 Agent ReAct 通道（多步推理、工具组合），Agent 端点未就绪时前端自动降级并明示。
- **混合检索**：FAISS 向量检索 + BM25 关键词检索，RRF 倒数排名融合，接入 Reranker 精排。
- **引用可溯源**：答案中标注 `[来源: 文档名, 第X页]`，前端渲染为可核对的引用卡片。
- **手写 ReAct**：约 200 行核心循环（不依赖框架高层封装），Thought→Action→Observation 迭代，最大轮次防死循环 + 强制收敛 + 错误恢复。
- **8 个可调用工具**：`knowledge_search` / `knowledge_qa` / `text_generate` / `calculator` / `paper_meta` / `paper_compare` / `text_summarize` / `keyword_extract`。
- **多轮记忆**：按 `session_id` 会话隔离 + 10 轮滑动窗口 + 超阈值自动摘要压缩，支持指代消解。
- **可观测性**：Token 消耗、检索命中率、响应延迟、工具调用成功率、推理轨迹全链路展示。
- **分工具 Token 统计**：右栏按工具分别统计 Token 消耗（本轮 + 累计），生成轮单独对账，各行合计与总量一致。
- **实时推理轨迹**：Agent 的 Thought→Action→Observation 随事件流实时生长，而非等回答结束一次性展示。
- **双栏独立滚动**：对话区与可观测面板各自滚动，右栏内容再长也能单独拉到底。
- **工程化**：语义缓存、三档降级策略（无结果/低相关性/API 失败）、请求日志、健康检查、增量索引。

## 二、技术栈

| 层 | 选型 |
|---|---|
| 前端 | Streamlit + requests（SSE 流式消费） |
| 后端框架 | FastAPI + Uvicorn |
| 大模型 | DeepSeek / 硅基流动等 OpenAI 兼容 API（`openai` SDK） |
| Embedding | BAAI/bge-large-zh-v1.5（本地，sentence-transformers） |
| 重排序 | cross-encoder/ms-marco-MiniLM-L-6-v2（FlagEmbedding） |
| 向量库 | FAISS（支持增量更新与持久化） |
| 关键词检索 | rank_bm25 + jieba 分词 |
| 分块 | langchain-text-splitters（递归切分）+ 自实现固定/语义切分 |
| 可观测 | loguru 日志 + REST 统计接口 |

## 三、目录结构

```
smart-research-assistant/
├── main.py                    # 启动入口：python main.py / --cli / --test
├── requirements.txt           # 依赖清单
├── .env.example               # 配置模板（复制为 .env 后填入 API Key）
├── .streamlit/config.toml     # 前端浅色主题
├── src/
│   ├── data_loader/           # 模块一：PDF/DOCX/TXT 加载
│   ├── chunking/              # 模块一：固定/递归/语义三种分块策略
│   ├── retrieval/             # 模块一：向量库 + BM25 + RRF 混合检索 + Reranker
│   ├── generation/            # 模块二：Prompt 模板 / RAG 流水线 / 语义缓存 / SSE 流式
│   ├── agent/                 # 模块三：ReAct 循环 / 8 工具集 / 记忆管理 / Agent 路由
│   ├── api/                   # 模块一/二：FastAPI 路由与装配
│   ├── frontend/              # 模块四：Streamlit 界面 + 服务适配层 + 文档
│   └── utils/                 # 配置与日志
├── uploads/                   # 上传的文档（运行时生成）
├── index/                     # 向量索引与元信息（运行时生成）
└── logs/                      # 运行日志（运行时生成）
```

## 四、环境配置

```bash
# 1. 创建并激活虚拟环境
conda create -n rag_agent python=3.10 -y
conda activate rag_agent

# 2. 安装依赖（国内建议走镜像）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 配置 API Key
#    Windows: copy .env.example .env   （或用记事本新建 .env）
#    编辑 .env，填入你的密钥
```

`.env` 关键项（完整模板见 `.env.example`）：

```env
# 大模型（OpenAI 兼容）
SILICONFLOW_API_KEY=sk-你的密钥
SILICONFLOW_BASE_URL=https://api.deepseek.com
CHAT_MODEL=<以官网 API 文档示例中的模型名原文为准>

# 本地模型（无需密钥，首次自动经 HF 镜像下载）
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
VECTOR_DIMENSION=1024

# 检索
DEFAULT_CHUNK_SIZE=512
DEFAULT_TOP_K=5
RRF_K=60

# 服务
HOST=0.0.0.0
PORT=8000
```

前端 / Agent 相关开关（可选，未设置时取默认值）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `USE_AGENT` | `false` | 前端问答通道：`true` 走 Agent ReAct，`false` 走 RAG 直连 |
| `AGENT_MAX_ROUNDS` | `5` | Agent 最大推理轮次（自动收敛到 1~10） |
| `RAG_API_BASE_URL` | `http://127.0.0.1:8000` | 前端请求的后端地址 |
| `RAG_TIMEOUT_CONNECT` / `RAG_TIMEOUT_READ` | `10` / `120` | 前端连接 / 读取超时（秒），慢模型可调大读取超时 |
| `RAG_SESSIONS_DIR` | `src/frontend/sessions` | 前端会话 JSON 落盘目录 |
| `RAG_PANEL_HEIGHT` | `760` | 前端左右两栏滚动容器高度（px） |

> Embedding 与 Reranker 为本地开源模型，**无需密钥**，首次启动自动从 HuggingFace 镜像下载
> （bge-large-zh 约 1.2GB）。DeepSeek 不提供向量化服务，请勿修改 `EMBEDDING_MODEL`。
>
> 环境只用 conda 的 `rag_agent` 即可，**无需**在仓库内再建 `.venv`；若自建虚拟环境，请把 `.venv/`
> 加入 `.gitignore`，避免其被打进交付压缩包（依赖体积约 1.4GB）。

## 五、启动与使用

```bash
# 终端 1：后端 API（http://localhost:8000，文档 /docs）
python main.py

# 终端 2：前端界面（http://localhost:8501）
streamlit run src/frontend/app.py
```

使用流程：左栏上传论文 → 等待解析与向量化（大文档 CPU 推理需数分钟）→ 中央输入问题 →
查看流式回答与 📎 引用 → 右栏观察推理轨迹与 Token/耗时/命中率指标。

命令行模式（不起 Web 服务，便于快速验证）：`python main.py --cli`

### 5.1 一键部署（Docker）

无需手动配环境，用 Docker 起后端与前端两个容器：

```bash
cp .env.example .env        # 填好 SILICONFLOW_API_KEY / CHAT_MODEL / USE_AGENT
docker compose up -d --build
# 前端 http://localhost:8501   后端文档 http://localhost:8000/docs
```

- `Dockerfile`：统一镜像（Python 3.10 + 依赖 + **预下载的本地模型**），后端/前端共用；
- `docker-compose.yml`：`backend`（FastAPI，8000）+ `frontend`（Streamlit，8501）两服务，
  `uploads/`、`index/`、`logs/`、前端会话目录均挂载到宿主机，容器重建不丢数据；
- `.dockerignore`：排除 `.env`、`.venv`、`uploads/`、`index/` 等，避免密钥与数据进镜像。

> 首次 `--build` 会安装依赖并下载约 1.3G 的 Embedding/Reranker 模型，耗时较长（走 HF 镜像）；
> 之后启动很快。建议宿主机内存 ≥ 8G。

## 六、API 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/query` | POST | RAG 问答，`stream=true` 走 SSE（模块二） |
| `/api/agent/query` | POST | Agent ReAct 问答，SSE 含 thought/action/observation（模块三） |
| `/api/search` | POST | 纯检索，供 Agent/评测复用（模块二） |
| `/api/generate` | POST | 给定上下文纯生成（模块二） |
| `/api/documents` `/api/documents/upload` `/api/documents/{name}` | GET/POST/DELETE | 文档管理与索引构建（模块一） |
| `/health` | GET | 健康检查（向量数/文档数/缓存数） |
| `/api/stats` | GET | RAG 运行统计 |
| `/api/agent/stats` | GET | Agent 统计（推理轮次/工具成功率/调用分布/活跃会话） |

完整交互式文档：启动后端后访问 `http://localhost:8000/docs`。

## 七、测试

```bash
python main.py --test                                  # 基础模块自检
python src/agent_delivery/agent_smoke_test.py           # 模块三冒烟（12 项自检）
```

## 八、文档索引

| 文档 | 用途 |
|---|---|
| `环境配置说明.md`（本仓库根目录） | 完整环境搭建与变量配置说明（依赖、变量清单、模型、排查、自检清单） |
| `../相关使用说明书(1).md`（仓库同级） | 后端 API 对接总说明（前端/Agent 均参考） |
| `../后端相关报告(1).md`（仓库同级） | 模块一/二技术报告 |
| `src/frontend/docs/` | 模块四设计说明、前端使用手册、Agent 事件协议、架构图、联调清单、交接与修改记录 |
| `src/agent_delivery/README_模块三交付说明.md` | 模块三交付说明 |

> 建议交付前把上述两份组内文档一并复制进仓库 `docs/` 目录，代码压缩包内自包含。

## 九、团队分工

| 角色 | 覆盖模块 | 核心职责 |
|---|---|---|
| 后端开发工程师 | 模块一、二 | 文档处理与混合检索、RAG 生成层、向量数据库、FastAPI 接口 |
| Agent 开发工程师 | 模块三 | 手写 ReAct 循环、8 工具集、决策优化、多轮记忆管理 |
| 前端开发工程师 | 模块四 | Streamlit 界面、可观测性可视化、系统集成与端到端联调 |
| 测试与质量保证工程师 | 模块五 | 评测集构建、性能评估、Bad Case 分析、部署与交付 |

## 十、已知问题与改进方向

- **Agent 通道顶栏「检索条数 top_k」不生效**：该参数已透传到 `/api/agent/query`，但 ReAct 循环未消费它，实际每次检索条数由模型在 `Action` 的 `Args` 中自行决定
  （`knowledge_search` 默认 3 / 上限 10，`knowledge_qa` 默认 5）。
- **Agent 通道 Token 成本高**：每轮 ReAct 都会重发全量上下文，单问 Token 可达直连通道的十余倍；批量评测前请先估算成本并控制轮次。
- **两通道统计口径不同**：RAG 直连看 `/api/stats`，Agent 看 `/api/agent/stats` 或前端右栏，评测时勿混用。
- 可扩展方向：联网搜索工具、多模态 PDF 表格解析、并发场景与容器化部署。

> 早期版本遗留的「删除文档不清理向量/BM25/缓存」「Agent 引用跨轮重复计数」「会话管理空会话增生」等问题，均已在当前代码修复，见下节。

## 十一、开发踩坑记录

| # | 现象 | 排查 | 解决 |
|---|---|---|---|
| 1 | 删除知识库文档要等几分钟，库越大越慢 | 删除链路 `delete_by_source → build_index → embed_texts` 把剩余全部块重新跑一遍 Embedding（FAISS 不支持原位删向量，重建时丢掉了已算好的向量） | 改用 `index.reconstruct_n` 取回现有向量、按保留行重建索引，删除从分钟级降到秒级 |
| 2 | Agent 推理轨迹不实时，回答结束才一次性蹦出 | 后端 SSE 正常；Streamlit 自上而下执行，整条事件流在左侧循环里消费，右栏排在后面只能最后渲染 | 右栏占位槽提前到消费事件流之前创建，chat 每条事件到达即实时重绘，结束后替换为完整面板 |
| 3 | 会话列表凭空多出很多「新会话 · 0 条」 | 新建/删除/切换都会无条件 `_save_session` 一个空会话，158 字节的空文件被列表列出 | 空会话不落盘（`_session_has_content` 守卫），并清理历史遗留空文件 |

## 十二、许可

课程实践项目，内部使用。
