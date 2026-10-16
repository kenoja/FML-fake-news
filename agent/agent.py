"""The graph and the resumable runner.

    load_context -> plan -> implement (one task per step, loops) -> run_pipeline -> verify
        verify --(gate failures, bounded)--> implement
        verify -> build_site -> review --(below 10/10, rounds left)--> plan
        review -> finalize -> END

`python -m agent run` executes steps until the time budget is used up, a usage limit is hit,
or the cycle is finished. Every finished node is checkpointed, so the next call continues
exactly where this one stopped. Output always ends with one line `STATUS: <word>`.
"""
from __future__ import annotations

import time
import traceback

from langgraph.graph import END, START, StateGraph

from . import memory, nodes, tools
from .state import AgentState


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    for name in ("load_context", "plan", "implement", "run_pipeline", "verify", "build_site", "review", "finalize"):
        g.add_node(name, getattr(nodes, name))
    g.add_edge(START, "load_context")
    g.add_edge("load_context", "plan")
    g.add_edge("plan", "implement")
    g.add_conditional_edges("implement", nodes.after_implement, ["implement", "run_pipeline"])
    g.add_edge("run_pipeline", "verify")
    g.add_conditional_edges("verify", nodes.after_verify, ["implement", "build_site"])
    g.add_edge("build_site", "review")
    g.add_conditional_edges("review", nodes.after_review, ["plan", "finalize"])
    g.add_edge("finalize", END)
    return g


# Nodes that may take up to limits.task_timeout_minutes (Claude calls, tests + full pipeline).
LONG_NODES = {"plan", "implement", "review", "run_pipeline"}

# For `langgraph dev` / LangGraph Studio (see langgraph.json). The runner below compiles its own
# copy with the SQLite checkpointer.
graph = build_graph().compile()


def run(cycle: int | None = None, minutes: float | None = None, use_git: bool = True) -> str:
    """Advance the current cycle. Returns CONTINUE, DONE, PAUSED or FAILED."""
    if cycle is None:
        cycle, _ = nodes.latest_requirements()
    branch = f"{tools.load_config('agent')['branch_prefix']}{cycle:02d}"
    lim = tools.load_config("agent")["limits"]
    budget_s = 60 * (minutes if minutes is not None else lim["run_minutes"])
    if use_git:
        tools.checkout_cycle_branch(branch)

    runs = memory.count_run(cycle)
    if runs > lim["max_runs_per_cycle"]:
        return _finish(cycle, branch, "FAILED", f"max_runs_per_cycle ({lim['max_runs_per_cycle']}) exceeded", use_git)

    saver, conn = memory.checkpointer(cycle)
    app = build_graph().compile(checkpointer=saver)
    cfg = memory.thread_config(cycle)
    snapshot = app.get_state(cfg)
    if snapshot.values and not snapshot.next:
        conn.close()
        return _finish(cycle, branch, "DONE", "cycle already finished; nothing to do", use_git, push=False)

    start = time.monotonic()
    step_s = 60 * lim["task_timeout_minutes"]
    status, detail = "CONTINUE", ""
    try:
        inputs = None if snapshot.values else {}
        for update in app.stream(inputs, cfg, stream_mode="updates"):
            node = next(iter(update))
            print(f"[{memory.now()}] finished {node}", flush=True)
            memory.reset_timeouts()
            if node == "finalize":
                status, detail = "DONE", update[node].get("stop_reason", "")
                break
            elapsed = time.monotonic() - start
            upcoming = app.get_state(cfg).next
            # Never start a long step that could not finish inside this run's time budget.
            if elapsed > budget_s or (set(upcoming) & LONG_NODES and elapsed + step_s > budget_s):
                detail = f"stopped after {elapsed / 60:.0f} min to stay within the run budget; next node: {upcoming}"
                break
    except tools.UsageLimitReached as exc:
        status, detail = "PAUSED", f"usage limit reached - resume later. {exc}"
    except TimeoutError as exc:
        # The step ran out of time. Its partial work is kept (committed below) and the same step is
        # retried on the next run, because the checkpoint did not advance past it.
        n = memory.count_timeout()
        if n >= 3:
            status, detail = "FAILED", (f"the same step timed out {n} times in a row ({exc}). Make the task smaller "
                                        "or raise limits.task_timeout_minutes in config/agent.yaml.")
        else:
            detail = f"a step ran out of time ({exc}); its partial work is saved and it will be retried (attempt {n + 1} of 3)"
    except Exception as exc:  # fail loud, keep the checkpoint, let a human look
        status, detail = "FAILED", f"{type(exc).__name__}: {exc}\n\n```\n{traceback.format_exc()[-3000:]}\n```"
    finally:
        conn.close()

    if status == "DONE" and use_git:
        state = _final_state(cycle)
        url = tools.open_pull_request(branch, f"Cycle {cycle}: agent results",
                                      state.get("review_summary", "") + "\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)")
        detail += f"\n\nPull request: {url}"
    return _finish(cycle, branch, status, detail, use_git)


def _final_state(cycle: int) -> dict:
    saver, conn = memory.checkpointer(cycle)
    try:
        return build_graph().compile(checkpointer=saver).get_state(memory.thread_config(cycle)).values
    finally:
        conn.close()


def _finish(cycle: int, branch: str, status: str, detail: str, use_git: bool, push: bool = True) -> str:
    memory.write_status(cycle, status, detail)
    if use_git and push:
        tools.commit_and_push(branch, f"agent: cycle {cycle} run -> {status}\n\n{detail[:600]}")
    print(detail)
    print(f"STATUS: {status}")
    return status
