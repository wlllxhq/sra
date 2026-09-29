# 模块四设计说明：系统集成与前端展示

> 本文档对应课程实践模块四（5.4.1~5.4.4），可直接作为《项目交付书》第 4.4 节素材。
> 作者角色：前端开发工程师（Frontend）。

## 1. 模块定位与设计原则

模块四将模块一（文档处理与检索层）、模块二（RAG 生成层）已交付的 FastAPI 服务整合为
可交互的 Web 应用，并为模块三（Agent 决策层）预留标准接入位。三条设计原则：

1. **单一数据契约**：前端所有组件只消费适配层归一化后的 `NormalizedEvent`，不直接触碰
   后端原始 SSE 格式。后端升级、Agent 接入都不影响 UI 层代码。
2. **通道开关式集成**：`USE_AGENT` 环境变量一键切换 RAG 直连 / Agent 通道；Agent 端点
   未就绪时自动降级并在界面明示，保证联调期间系统始终可用。
3. **前端永不崩溃**：网络异常、HTTP 错误、SSE 坏帧、未知事件类型全部转为可见的
   status/error 提示，界面保持可操作（对应模块三的容错理念在前端侧的落地）。

## 2. 文件结构

```
src/frontend/
├── app.py                        # 主入口：三栏布局、顶栏、会话管理（新建/切换/删除/落盘）
├── services/
│   └── rag_client.py             # ★ 服务适配层：REST 封装 + SSE 归一化 + USE_AGENT 开关
├── components/
│   ├── chat.py                   # 中央对话：流式打字机、Markdown、引用徽章
│   ├── doc_panel.py              # 左栏：批量上传、进度、文档列表、删除
│   └── trace_panel.py            # 右栏：引用卡片、Agent 轨迹树、本轮/系统监控指标
├── sessions/                     # 会话持久化目录（运行时生成，每会话一个 JSON）
├── docs/
│   ├── module4_design.md         # 本文档
│   ├── frontend_usage.md         # 启动使用说明
│   └── agent_event_protocol.md   # 与模块三约定的事件协议
└── frontend_smoke_test.md        # 联调测试清单与结果
```

## 3. 与后端接口映射（5.4.4 数据格式统一）

| 界面功能区 | 后端接口 | 说明 |
|---|---|---|
| 上传并建索引 | `POST /api/documents/upload` | 支持 PDF/DOCX/TXT/MD/CSV，多选批量，逐文件进度 |
| 文档列表/删除 | `GET /api/documents`、`DELETE /api/documents/{name}` | 🟢indexed/🟡processing/🔴failed 状态徽标 |
| 对话（流式） | `POST /api/query`（stream=true） | 打字机 + 引用标记动态出现 |
| 对话（非流式） | `POST /api/query`（stream=false） | 供调试与模块五评测批跑复用 `rag_client.chat()` |
| 顶栏 top_k 滑块 | 透传 `top_k` 字段 | 1~20 |
| 系统监控面板 | `GET /health`、`GET /api/stats` | 向量数/文档数/查询数/缓存命中率/平均耗时/Token |
| 会话上下文 | 每次请求透传 `session_id` | 后端当前未消费；模块三接管记忆后立即生效，前端零改动 |

## 4. 事件归一化契约（核心设计）

`rag_client.NormalizedEvent`，kind 共 9 类：

| kind | 来源（后端原始 type） | UI 消费位置 |
|---|---|---|
| status | thinking | 对话区状态行 → 右栏轨迹时间线 |
| reference | reference（ref_index/doc/page/score/content） | 答案上方引用徽章行 + 右栏引用卡片 |
| delta | chunk | 打字机累积渲染（≥0.12s 或 8 chunk 节流重绘） |
| thought / action / observation | **模块三预留**（协议见 agent_event_protocol.md） | 右栏 Thought→Action→Observation 轨迹树 |
| final | done（elapsed_time/token_count/reference_count） | 本轮指标卡 |
| error | error / 本地网络异常 | 红色提示区 |
| unknown | 未来任何新增类型 | 灰色 caption 原样降级显示，**不崩溃** |

## 5. 模块三接入方式（预留位说明）

1. 模块三交付 `POST /api/agent/query`（请求体沿用 `QueryRequest`：question/session_id/top_k/stream，
   SSE 按协议扩展 thought/action/observation 三类事件，保留 reference/chunk/done/error 兼容）。
2. 前端启用：`.env` 设 `USE_AGENT=true`（或环境变量），无需改任何代码。
3. 端点不存在（404）时自动降级 RAG 直连并明示用户——两人可并行开发互不阻塞。
4. 会话记忆：模块三按 `session_id` 做滑动窗口/摘要压缩后，历史渲染与切换功能前端已就绪。

## 6. 可观测性实现（对应 5.4.2）

- **Token 消耗**：done 事件 `token_count` + `/api/stats` 累计值，两处展示。
- **检索命中率**：`/api/stats.cache_hit_rate`（语义缓存）+ 每条引用 score 相关度。
- **响应延迟**：done 事件 `elapsed_time`（本轮）与 `avg_response_time`（系统）。
- **决策路径**：右栏轨迹树——RAG 模式显示 thinking 时间线；Agent 模式自动切换为
  T→A→O 树（action 可展开看出入参 JSON）。
- **健康检查**：页面每次渲染实时探活 `/health`，后端宕机顶条红色横幅明示并给出恢复指引。

## 7. 已验证与待验证

已完成：全文件编译、事件归一化/SSE 解析/引用高亮单元测试、Streamlit AppTest 冒烟
（离线降级 + 模拟完整问答回合 + 会话落盘）。
待真机验证：见 `frontend_smoke_test.md` 联调清单（需后端环境 + API Key）。

## 8. 已知边界（如实记录）

- `session_id` 目前后端未消费，多轮"记得上文"要等模块三。
- 删除文档仅删文件，不清理向量索引中对应 chunks（后端接口现状），知识库一致性建议
  模块二后续在 DELETE 路由补 `vector_store.delete_by_source`。
- 打字机节流在超长回答（>2000 字）时右栏指标会等 final 事件才出，属预期行为。
