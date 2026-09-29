# 前端（模块四）启动使用说明

## 1. 环境要求

- Python 3.10+（模块一二三共用一个环境即可）
- 依赖已在 `requirements.txt` 登记：`streamlit>=1.30.0`、`requests>=2.31.0`

```bash
pip install streamlit requests
```

## 2. 启动（两个终端）

```bash
cd smart-research-assistant

# 终端 1：后端（模块一二）
python main.py                      # http://localhost:8000

# 终端 2：前端（模块四）
streamlit run src/frontend/app.py   # 默认 http://localhost:8501
```

浏览器打开 `http://localhost:8501`。后端未启动时页面会显示红色提示条并给出指引，
不会白屏。

## 3. 配置项（`.env` 或系统环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_API_BASE_URL` | `http://127.0.0.1:8000` | 后端地址；后端换机器/端口时修改 |
| `USE_AGENT` | `false` | `true` 时对话走 `/api/agent/query`（模块三）；未部署自动降级并提示 |
| `RAG_TIMEOUT_CONNECT` | `10` | 连接超时（秒） |
| `RAG_TIMEOUT_READ` | `120` | 读取超时（秒），长回答/慢模型可调大 |

## 4. 界面功能速查

- **左栏**：上传（多选批量、逐文件进度与结果）；文档列表（状态徽标、🗑 删除）；
  会话管理（➕新建、下拉切换历史、🗑删除当前；会话落盘 `src/frontend/sessions/*.json`）。
- **中央**：提问 → 打字机流式回答；`📎文档名·第X页` 引用徽章先于正文出现；
  历史消息含耗时/Token 元信息。
- **右栏**：本轮引用卡片（相关度+摘要）、本轮指标（耗时/Token/引用数）、
  Agent 推理轨迹树（RAG 模式显示 thinking 时间线；模块三接入后自动显示
  Thought→Action→Observation）、系统监控（向量数/文档数/提问数/缓存命中率/平均耗时/Token）。
- **顶栏**：通道徽标（RAG 直连 / Agent 通道）、top_k 滑块（1~20，透传后端）。

## 5. 常见问题

| 现象 | 原因与处理 |
|---|---|
| 红条"后端服务不可用" | 后端没起/端口不对：终端 1 `python main.py`，或改 `RAG_API_BASE_URL` |
| 上传后提问无引用 | 确认文档状态 🟢indexed；大 PDF 解析需 1~3 分钟，看进度区 |
| 回答很慢 | 首次加载模型慢；相同问题会命中语义缓存几乎秒回（thinking 显示"命中缓存"） |
| 打字机卡顿 | 已做 0.12s/8chunk 节流；仍卡则网络到模型 API 慢 |
| 切换 USE_AGENT 无效 | 环境变量在 `import` 时读取，改后需重启 streamlit |

## 6. 给模块五（QA）的对接提示

评测批跑请直接用 `src/frontend/services/rag_client.py` 的 `chat()`（非流式，返回完整
answer+references+耗时），或 `stream_chat()` 收集事件；无需起 Streamlit 界面。
