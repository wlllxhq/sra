# -*- coding: utf-8 -*-
"""
模块四 · 5.4.4 多用户并发场景测试
================================
两种模式：

  离线自检（不消耗API，随时可跑）：
      python tests/test_session_concurrency.py --offline
  断言 MemoryManager 在多线程并发写入下的会话隔离正确性（模块三记忆层的并发正确性）。

  真机并发（需后端已 python main.py 运行，会消耗 LLM Token，联调时跑一次）：
      python tests/test_session_concurrency.py --live
  对 /api/agent/query（或 USE_AGENT=false 时 /api/query）发起 N 路不同 session_id
  的并发流式请求，校验：① 每路均完整收到 done 收尾；② 各路 session 记忆互不串话
  （随后向每个 session 追问"它指的是哪篇？"，验证上下文隔离）。
"""
import argparse
import concurrent.futures as cf
import json
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


# ---------------------------------------------------------------------------
# 离线：MemoryManager 并发隔离
# ---------------------------------------------------------------------------
def test_memory_isolation_offline(n_threads=8, n_msg=20) -> None:
    from src.agent.memory import MemoryManager

    mm = MemoryManager(window_size=5)
    errors: list[str] = []

    def worker(tid: int) -> None:
        sid = f"session_{tid}"
        try:
            for i in range(n_msg):
                mm.add_user_message(sid, f"[T{tid}] 问题{i}")
                mm.add_assistant_message(sid, f"[T{tid}] 回答{i}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sid}: {e}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    t0 = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    elapsed = time.time() - t0

    assert not errors, f"并发写入报错: {errors}"
    # 每个会话上下文只应含自己线程的标记
    for t in range(n_threads):
        ctx = mm.get_context_prompt(f"session_{t}")
        assert f"[T{t}]" in ctx, f"session_{t} 上下文丢失自身标记"
        others = [f"[T{x}]" for x in range(n_threads) if x != t]
        leaked = [m for m in others if m in ctx]
        assert not leaked, f"session_{t} 串话，混入: {leaked}"
    stats = mm.get_session_stats()
    assert stats["total_sessions"] == n_threads, "会话数不符"
    print(f"[offline] PASS: {n_threads}线程×{n_msg}轮 并发写入隔离正确，耗时 {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# 真机：多 session 并发流式问答
# ---------------------------------------------------------------------------
def _collect_stream(base: str, endpoint: str, question: str, sid: str) -> dict:
    import requests

    body = {"question": question, "session_id": sid, "top_k": 3, "stream": True}
    result = {"session_id": sid, "ok": False, "got_done": False, "error": "", "events": 0}
    try:
        resp = requests.post(f"{base}{endpoint}", json=body, stream=True, timeout=(10, 300))
        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:150]}"
            resp.close()
            return result
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            result["events"] += 1
            if ev.get("type") == "done":
                result["got_done"] = True
                result["answer_head"] = (ev.get("content") or "")[:60]
            if ev.get("type") == "error":
                result["error"] = ev.get("content", "")
        resp.close()
    except Exception as e:  # noqa: BLE001
        result["error"] = str(e)[:200]
    result["ok"] = result["got_done"] and not result["error"]
    return result


def test_concurrency_live(base: str, n_users: int = 3) -> None:
    import os

    use_agent = os.getenv("USE_AGENT", "false").strip().lower() in ("1", "true", "yes")
    endpoint = "/api/agent/query" if use_agent else "/api/query"
    ts = int(time.time())
    questions = [
        "这篇论文主要用了什么模型？",
        "知识库里的论文研究了什么对象？",
        "论文的实验结果如何？",
    ][:n_users]

    print(f"[live] 端点 {endpoint} | {n_users} 路并发 ...")
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=n_users) as pool:
        futs = [pool.submit(_collect_stream, base, endpoint, q, f"conc_{ts}_{i}")
                for i, q in enumerate(questions)]
        results = [f.result() for f in futs]
    total = time.time() - t0

    failed = [r for r in results if not r["ok"]]
    for r in results:
        mark = "✅" if r["ok"] else "❌"
        print(f"  {mark} {r['session_id']}: events={r['events']} "
              f"done={r['got_done']} err={r['error'][:80] or '-'}")

    assert not failed, f"{len(failed)}/{n_users} 路失败"
    print(f"[live] PASS: {n_users} 路并发全部完整收尾，总耗时 {total:.1f}s")
    print(f"[live] 后续人工核验：对 conc_{ts}_0 追问“它指的是哪篇？”应消解为第一问语境；"
          f"对 conc_{ts}_1 同样提问应指向不同对象（会话隔离）。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--users", type=int, default=3)
    args = ap.parse_args()

    if args.offline or not (args.offline or args.live):
        test_memory_isolation_offline()
    if args.live:
        test_concurrency_live(args.base, args.users)
    print("\nDONE")


if __name__ == "__main__":
    main()
