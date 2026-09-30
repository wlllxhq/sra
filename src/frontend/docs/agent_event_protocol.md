# 模块三（Agent 决策层）对接协议 v1.0

> 发起方：前端开发工程师（模块四）
> 约定对象：Agent 开发工程师（模块三）
> 状态：协议已冻结，按此实现即可无缝接入；如需改字段，先在本文件留言。

## 1. 接入方式总览

前端通过环境变量 `USE_AGENT` 切换通道（默认 `false`）：

| USE_AGENT | 前端请求端点 | 说明 |
|---|---|---|
| false | `POST /api/query` | 现状：模块二 RAG 直连 |
| true | `POST /api/agent/query` | 目标：Agent ReAct 循环；内部可继续调用 `/api/search`、`/api/generate` 作为工具 |

前端对 404 已做自动降级（回 RAG 直连并提示用户），因此**模块三未部署时前端照常工作**，
双方可完全并行开发。

## 2. 请求体（沿用后端现有 QueryRequest，字段名不变）

```json
{
  "question": "对比论文A和论文B的方法差异",
  "session_id": "s_08791a2b",
  "top_k": 5,
  "stream": true
}
```

- `session_id`：请模块三**务必消费**——多轮记忆（会话隔离/滑动窗口/摘要压缩）以它为键。
  前端每次提问都携带，且右栏历史/会话切换已就绪。
- 可选扩展字段按需在下方表格追加，勿改既有字段名。

## 3. SSE 事件协议

格式与后端现状一致：每条事件单独一行 `data: {JSON}\n\n`，UTF-8 不转义中文。

### 3.1 兼容保留（语义与模块二完全一致，前端已实现渲染）

| type | 必带字段 | 前端行为 |
|---|---|---|
| `thinking` | content | 状态行显示 + 轨迹时间线 |
| `reference` | content, ref_index, doc, page, score | 答案上方引用徽章 + 右栏引用卡片 |
| `chunk` | content | 打字机增量 |
| `done` | content, reference_count, elapsed_time, token_count | 本轮指标卡，回合收尾 |
| `error` | content | 红色提示区 |

**硬要求**：一轮响应必须恰好以 1 个 `done` 或 `error` 收尾（前端靠它判定回合结束，
无收尾事件会等待至读取超时 120s 并报流中断）。

### 3.2 新增三类（ReAct 轨迹，前端已预留渲染位，收到即显示）

| type | 字段 | 说明 |
|---|---|---|
| `thought` | content（推理文本，≤500字） | 每轮 ReAct 的 Thought |
| `action` | content（自然语言描述）、tool（工具名）、args（JSON 对象或可解析字符串） | 工具调用意图；前端可展开查看入参 |
| `observation` | content（结果摘要，≤300字）、tool（对应工具名） | 工具返回后的观察 |
| `final_answer_begin` | 可选 | 预留；若不需要区分"轨迹中的文本"与"最终回答"，直接用 chunk 即可 |

字段取键顺序（前端 `_normalize` 实际兼容的别名）：`tool`/`tool_name`/`action`，
`args`/`arguments`/`params`。建议直接按标准键输出。

### 3.3 一次 Agent 回合的事件流示例（两轮 ReAct + 工具调用）

```
data: {"type":"thinking","content":"Agent 接收任务，开始规划…"}
data: {"type":"thought","content":"需要对比两篇论文，先分别检索方法部分"}
data: {"type":"action","content":"检索论文A方法","tool":"knowledge_search","args":{"query":"Swin Transformer 方法 架构","top_k":3}}
data: {"type":"observation","content":"检索到3条相关段落","tool":"knowledge_search"}
data: {"type":"action","content":"检索论文B方法","tool":"knowledge_search","args":{"query":"ViT 方法 架构","top_k":3}}
data: {"type":"observation","content":"检索到3条相关段落","tool":"knowledge_search"}
data: {"type":"thought","content":"上下文已齐，调用生成器做对比"}
data: {"type":"action","content":"生成对比回答","tool":"text_generate","args":{"question":"对比两篇论文方法差异"}}
data: {"type":"reference","content":"Swin 采用移位窗口…","ref_index":1,"doc":"swin.pdf","page":3,"score":0.42}
data: {"type":"chunk","content":"两篇论文的核心差异在于…"}
data: {"type":"done","content":"两篇论文的核心差异在于……（完整回答）","reference_count":1,"elapsed_time":18.7,"token_count":3121}
```

### 3.4 工具命名（与使用说明书第 4.2 节注册名保持一致）

`knowledge_search` → `/api/search`；`knowledge_qa` → `/api/query`；
`text_generate` → `/api/generate`。其余工具（论文元信息/对比/摘要等）命名建议
snake_case 英文，前端不做白名单校验，未知 tool 名照常展示。

## 4. 验收标准（前端侧自动满足，无需前端改动）

1. 端点 `/api/agent/query` 返回 `Content-Type: text/event-stream`；
2. 3.2 三类事件任一出现，右栏轨迹树自动渲染（action 可展开看出入参）；
3. 会话记忆：同 `session_id` 连续提问"它指的是哪篇？"能正确消解指代；
4. 中途异常返回 `error` 事件收尾，或 HTTP 5xx（前端显示"后端返回 HTTP 5xx"）。

## 5. 前端联调支持

- 适配层：`src/frontend/services/rag_client.py`（`stream_chat` 已按本协议解析全部类型）
- 渲染层：`src/frontend/components/chat.py`（对话）、`trace_panel.py`（轨迹树）
- 协议测试：任意 SSE mock（如 `npx sse-test-server`）即可，或直接在 routes 加
  一个返回固定事件序列的桩端点供前端先行验证。
