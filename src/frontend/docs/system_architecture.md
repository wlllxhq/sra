# 系统架构图（交付书 4.4.1 / 第五章 5.1 素材）

> 绘制：模块四 ｜ 2026-09-30 ｜ 工具：Mermaid（交付书可用支持 Mermaid 的编辑器直接渲染，或粘贴至 draw.io / mermaid.live 导出 PNG）
> 分层与文字描述遵循《项目交付模板》："系统分为五层：展示层（Streamlit界面）、调度层（Agent决策）、生成层（RAG流水线）、检索层（混合检索+重排序）、数据层（向量数据库+文档存储）。各层通过标准化接口交互。"

## 图一：五层架构总览

```mermaid
flowchart TB
    subgraph L1["展示层（模块四 · Streamlit 前端）"]
        UI_CHAT["中央对话区<br/>流式打字机 / Markdown / 引用徽章"]
        UI_DOC["左栏文档管理<br/>批量上传 / 进度 / 列表 / 删除"]
        UI_TRACE["右栏可观测面板<br/>T→A→O 轨迹树 / 引用卡片 / 指标"]
        CLIENT["服务适配层 rag_client.py<br/>REST封装 + SSE 归一化（NormalizedEvent 9类）<br/>USE_AGENT 通道开关"]
        UI_CHAT & UI_DOC & UI_TRACE --> CLIENT
    end

    subgraph L2["调度层（模块三 · Agent 决策）"]
        AGENT_EP["POST /api/agent/query"]
        REACT["ReAct 手写核心循环<br/>Thought→Action→Observation ≤5轮<br/>强制收敛 + 异常 error 收尾"]
        TOOLS["工具集 ×8<br/>knowledge_search / knowledge_qa / text_generate<br/>calculator / paper_meta / paper_compare<br/>text_summarize / keyword_extract"]
        MEM["记忆管理 MemoryManager<br/>session_id 隔离 + 10轮滑动窗口 + 摘要压缩"]
        AGENT_EP --> REACT --> TOOLS
        REACT <--> MEM
    end

    subgraph L3["生成层（模块二 · RAG 生成）"]
        RAG_EP["POST /api/query（RAG直连）<br/>POST /api/generate（纯生成）"]
        PIPE["RAGPipeline<br/>Prompt模板 / 上下文拼接 / 引用溯源<br/>流式SSE / 语义缓存 / 三档降级 / 日志"]
        RAG_EP --> PIPE
    end

    subgraph L4["检索层（模块一 · 文档处理与检索）"]
        SRCH_EP["POST /api/search<br/>POST /api/documents/upload"]
        HYBRID["混合检索 HybridRetriever<br/>向量 + BM25 → RRF 融合"]
        RERANK["Reranker 精排<br/>cross-encoder"]
        LOAD["多格式加载 PDF/DOCX/TXT<br/>3种分块策略（固定/递归/语义）"]
        SRCH_EP --> HYBRID --> RERANK
        LOAD -.建索引.-> HYBRID
    end

    subgraph L5["数据层"]
        FAISS[("FAISS 向量索引<br/>faiss.index + chunks_meta")]
        BM25[("BM25 语料库")]
        EMB[("Embedding 模型<br/>bge-large-zh-v1.5（本地）")]
        UPLOADS[("uploads/ 原始文档")]
        CACHE[("语义缓存（内存）")]
    end

    EXT["LLM API（外部）<br/>DeepSeek / 硅基流动<br/>OpenAI 兼容接口"]

    CLIENT -- "USE_AGENT=true" --> AGENT_EP
    CLIENT -- "USE_AGENT=false" --> RAG_EP
    TOOLS -- "进程内直调（红线5：无HTTP回环）" --> PIPE
    TOOLS -- "进程内直调" --> HYBRID
    PIPE --> HYBRID
    HYBRID --> FAISS & BM25
    FAISS --- EMB
    PIPE -- "失败降级：回退纯模型生成" --> EXT
    REACT --> EXT
    PIPE --> EXT
    PIPE --- CACHE
    UPLOADS --> LOAD
    MEM -. "GET /health · /api/stats · /api/agent/stats" .-> CLIENT
```

## 图二：核心数据流时序（Agent 通道一轮问答）

```mermaid
sequenceDiagram
    participant U as 用户
    participant F as Streamlit 前端<br/>(rag_client)
    participant A as /api/agent/query<br/>(ReAct循环)
    participant M as MemoryManager
    participant T as knowledge_search
    participant R as 混合检索+Rerank
    participant L as LLM API

    U->>F: 提问（携带 session_id）
    F->>A: POST stream=true
    A->>M: get_context(session_id) 注入历史
    A->>L: System+工具描述+上下文+问题
    L-->>A: Thought + Action + Args
    A-->>F: SSE: thought / action
    A->>T: execute(query, top_k)
    T->>R: rag_pipeline.search()（进程内直调）
    R-->>T: top_k 段落（content/source/page/score）
    T-->>A: ToolResult（detail 正文注入模型上下文）
    A-->>F: SSE: observation / reference
    Note over A,L: 循环 ≤5 轮，超限强制收敛
    L-->>A: Final Answer
    A-->>F: SSE: chunk×N（打字机）
    A->>M: add_turn(session_id, Q, A)
    A-->>F: SSE: done（token/耗时/引用数）【恰一次收尾】
    F-->>U: 渲染答案+📎引用+轨迹树+指标
```

## 图三：RAG 直连通道（USE_AGENT=false，数据流对照）

```mermaid
flowchart LR
    Q["用户问题<br/>/api/query"] --> C{"语义缓存<br/>相似度≥0.95?"}
    C -- 命中 --> R0["秒回缓存答案<br/>thinking: 命中缓存"]
    C -- 未命中 --> S["混合检索 top_k×4<br/>向量+BM25 → RRF"]
    S --> RR["Reranker 精排 → top_k"]
    RR -- 无结果 --> FB["降级：FALLBACK 提示文案"]
    RR --> P["build_rag_prompt<br/>引用编号【参考文献i】"]
    P --> LLM["LLM 流式生成<br/>[来源: 文档名, 第X页]"]
    LLM --> W["写缓存 + 日志<br/>Token/耗时/命中率"]
    W --> OUT["SSE: thinking→reference→chunk→done"]
```

## 交付书嵌入说明

- 4.4.1 用**图一**（五层架构）+ 模板既有文字描述；
- 第五章 5.2 模块接口定义可直接引用接口表（`docs/module4_design.md` §3），5.3 核心数据流向用**图二/图三**；
- 渲染导出：把 mermaid 代码块贴入 [mermaid.live](https://mermaid.live) → Export PNG/SVG → 插入 Word。
