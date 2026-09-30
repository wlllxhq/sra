import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ok, fail = [], []

def check(name, fn):
    try:
        fn()
        ok.append(name)
    except Exception as e:
        fail.append(f"{name}: {type(e).__name__}: {e}")

check("import memory", lambda: __import__("src.agent.memory", fromlist=["MemoryManager"]))
check("import tools_pkg", lambda: __import__("src.agent.tools", fromlist=["register_all_tools"]))
check("import react_loop", lambda: __import__("src.agent.react_loop", fromlist=["ReActLoop"]))
check("import agent_pipeline", lambda: __import__("src.agent.agent_pipeline", fromlist=["AgentPipeline"]))
check("import agent_routes", lambda: __import__("src.agent.routes", fromlist=["router"]))

def reg_tools():
    from src.agent.tools import register_all_tools, TOOL_REGISTRY
    register_all_tools()
    assert len(TOOL_REGISTRY) == 8, f"工具数={len(TOOL_REGISTRY)}"
check("register 8 tools", reg_tools)

def tool_names():
    from src.agent.tools import TOOL_REGISTRY, register_all_tools
    register_all_tools()
    expect = {"knowledge_search","knowledge_qa","text_generate","calculator",
              "paper_meta","paper_compare","text_summarize","keyword_extract"}
    assert set(TOOL_REGISTRY) == expect, f"差异: {set(TOOL_REGISTRY) ^ expect}"
check("tool names match protocol", tool_names)

def calc():
    from src.agent.tools import CalculatorTool
    t = CalculatorTool()
    r = t.safe_execute(expression="2+3*4")
    assert r.success and "14" in r.content, r.content
    r2 = t.safe_execute(expression="mean([1,2,3,4])")
    assert r2.success and "2.5" in r2.content, r2.content
    r3 = t.safe_execute(expression="__import__('os')")
    assert not r3.success, "不安全表达式应被拒绝"
check("calculator safe_eval", calc)

def kw():
    from src.agent.tools import KeywordExtractTool
    t = KeywordExtractTool()
    r = t.safe_execute(text="本文提出基于注意力机制的Transformer模型，用于作物病虫害识别。注意力机制提升识别准确率。")
    assert r.success and r.metadata.get("keywords"), r.content
check("keyword_extract", kw)

def memory():
    from src.agent.memory import MemoryManager
    import time
    m = MemoryManager(window_size=2)
    m.add_user_message("s1", "问题一")
    m.add_assistant_message("s1", "回答一")
    m.add_user_message("s1", "问题二")
    m.add_assistant_message("s1", "回答二")
    m.add_user_message("s1", "问题三")
    m.add_assistant_message("s1", "回答三")
    s = m.get_session("s1")
    ctx = m.get_context_prompt("s1")
    assert "对话历史摘要" in ctx or len(s.messages) <= 4, f"窗口未生效 msgs={len(s.messages)}"
    # 会话隔离
    m.add_user_message("s2", "另一个会话")
    assert "问题一" not in m.get_context_prompt("s2"), "会话串话"
check("memory window + isolation", memory)

def parse():
    from src.agent.react_loop import ReActLoop
    from src.agent.memory import MemoryManager
    loop = ReActLoop(llm_client=None, model="x", memory_manager=MemoryManager())
    p1 = loop._parse_llm_output("Thought: 需要检索\nAction: knowledge_search\nArgs: {\"query\": \"test\", \"top_k\": 3}")
    assert p1["type"] == "action" and p1["action"] == "knowledge_search", p1
    assert p1["args"].get("query") == "test", p1
    p2 = loop._parse_llm_output("Thought: 完成\nFinal Answer: 这是最终答案，引用[来源: a.pdf, 第3页]")
    assert p2["type"] == "final" and "最终答案" in p2["final_answer"], p2
check("react output parser", parse)

def route_contract():
    from src.agent.routes import router
    paths = {r.path for r in router.routes}
    assert "/api/agent/query" in paths, paths
    assert "/api/agent/stats" in paths, paths
check("agent endpoints", route_contract)

print("\n=== 冒烟测试结果 ===")
for x in ok:
    print(f"  [PASS] {x}")
for x in fail:
    print(f"  [FAIL] {x}")
print(f"\n通过 {len(ok)} / 失败 {len(fail)}")
sys.exit(1 if fail else 0)
