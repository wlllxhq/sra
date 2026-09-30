# 模块三（Agent 决策层）交付说明

> 交付人：成员B ｜ 日期：2026-09-30 ｜ 状态：**已联调跑通**

---

## 一、本包内容

```
模块三交付包/
├── README_模块三交付说明.md      ← 本文件
├── src/agent/                     ← 模块三全部代码（15个文件）
│   ├── __init__.py
│   ├── memory.py                  多轮对话记忆管理
│   ├── react_loop.py              ReAct核心循环（手写实现）
│   ├── agent_pipeline.py          Agent流水线（SSE事件编排）
│   ├── routes.py                  API路由（/api/agent/query 等）
│   └── tools/                     工具集（8个工具）
│       ├── base_tool.py           工具基类 + ToolResult
│       ├── knowledge_search.py    知识库检索
│       ├── knowledge_qa.py        端到端知识库问答
│       ├── text_generate.py       纯文本生成
│       ├── calculator.py          数学计算（AST安全解析）
│       ├── paper_meta.py          论文元信息查询
│       ├── paper_compare.py       论文对比分析
│       ├── text_summarize.py      文本摘要
│       └── keyword_extract.py     关键词提取
├── app.py（修改后完整版）          ← 成员A的 src/api/app.py，合入Agent挂载后的版本
├── env.example（追加Agent配置）    ← 新增 USE_AGENT / AGENT_MAX_ROUNDS 两个配置项
└── agent_smoke_test.py            ← 冒烟测试脚本（11项自检）
```

## 二、部署方式（合入主仓库）

1. 将 `src/agent/` 整个目录放入主仓库的 `src/` 下（与 api、retrieval 等并列）；
2. 用包内 `app.py（修改后完整版）` 覆盖主仓库 `src/api/app.py`
   （改动仅两处：注册 agent 路由 + 启动时初始化 Agent 流水线，未触碰模块一/二任何逻辑）；
3. `.env` 中确认：`USE_AGENT=true`；
4. 启动后端 `python main.py`，日志出现
   `Agent流水线初始化完成（模块三）` 即挂载成功；
5. 验证入口：
   - `http://127.0.0.1:8000/docs` 可见 `/api/agent/query`、`/api/agent/stats`
   - 前端左上角显示 **🤖 Agent 通道**

## 三、技术要点（答辩口径）

| 作业要求 | 实现位置 | 说明 |
|---|---|---|
| ReAct 核心循环手写 | react_loop.py | Thought→Action→Observation 迭代；LLM输出用正则解析（不依赖框架），最大5轮防死循环，超限强制收敛生成最终回答 |
| 工具集（8个） | tools/ | 统一 ToolBase 接口，snake_case 命名；safe_execute 全异常兜底 |
| 决策优化 | react_loop.py | 工具路由由 LLM 决策；错误恢复：工具失败以 observation 反馈给模型重试，不中断回合 |
| 多轮记忆管理 | memory.py | session_id 会话隔离 + 10轮滑动窗口 + 超限自动摘要压缩（调用大模型生成，失败降级截取）；指代消解靠上下文注入 |
| SSE事件协议 | routes.py + agent_pipeline.py | 严格按模块四冻结协议 v1.0：thought/action/observation 三类新增事件 + thinking/reference/chunk/done/error 兼容事件；每轮恰好以 done 或 error 收尾 |
| 可观测指标 | GET /api/agent/stats | 平均推理轮次、工具调用成功率、按工具调用分布、活跃会话数 |

**设计决策（答辩高频问题备答）：**
- **为什么不走 HTTP 回环调 /api/query？** 遵循交接说明红线5：Agent 与 RAG 同进程部署，直接注入复用 RAGPipeline 实例的方法调用，避免超时叠加与序列化开销；
- **为什么工具返回注入 user 角色消息？** Function Calling 格式对提示词工程要求高且推理模型（deepseek-flash）兼容性不稳，手写 ReAct 用文本协议可控性更强，轨迹可视化更直观；
- **语义缓存的坑**：调试期改 Prompt 答案不动 → 是模块二缓存命中（相似度≥0.95），已按交接说明处理，Agent 通道内缓存行为可控。

## 四、验收清单自查（对照《给模块三的交接说明》第五节）

- [x] USE_AGENT=false 时现有功能一切正常（零侵入，模块一/二代码未改动）
- [x] USE_AGENT=true 时 T→A→O 轨迹树、引用徽章、本轮指标完整渲染
- [x] 同 session_id 跨轮记忆生效；不同 session_id 互不串话（冒烟测试已验证）
- [x] 每轮恰以 done/error 收尾，无前端干等
- [x] 8 个工具注册完毕，命名 snake_case 英文
- [x] GET /api/agent/stats 提供 Agent 级指标

## 五、联调结论

2026-09-30 与模块一/二（smart 后端）+ 模块四（生产实习前端）在本机完成端到端联调：
环境 shengchanshixi（Python 3.10）+ DeepSeek 通道，上传文档→提问→ReAct 轨迹→带引用回答全链路跑通。

## 六、依赖说明

模块三未引入任何新第三方依赖，全部复用主仓库 requirements.txt
（fastapi / uvicorn / openai / loguru / python-dotenv / pydantic + Python 标准库）。
