"""Graph nodes. LLM calls only where judgment is needed (plan, implement, review);
everything that code can answer (tests, gates, site build, routing) is plain code."""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import memory, tools
from .state import AgentState, Task

REPO = tools.REPO


def _event(node: str, note: str) -> dict:
    return {"history": [{"node": node, "at": memory.now(), "note": note[:500]}]}


def _limits() -> dict:
    return tools.load_config("agent")["limits"]


def _models() -> dict:
    return tools.load_config("agent")["models"]


def _data_dirs() -> list[Path]:
    try:
        from fml.paths import data_dir
        return [data_dir()]
    except Exception:
        return []


def _repo_tree(max_lines: int = 200) -> str:
    skip = {".git", ".agent_state", "site", "__pycache__", ".pytest_cache", "node_modules"}
    lines = []
    for p in sorted(REPO.rglob("*")):
        rel = p.relative_to(REPO)
        if any(part in skip or part.endswith(".egg-info") for part in rel.parts) or p.is_dir():
            continue
        lines.append(str(rel))
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... {len(lines) - max_lines} more files"]
    return "\n".join(lines)


def latest_requirements() -> tuple[int, Path]:
    files = sorted((REPO / "requirements").glob("cycle-*.md"))
    if not files:
        raise FileNotFoundError("requirements/ has no cycle-NN.md file")
    cycle = int(re.search(r"cycle-(\d+)", files[-1].stem).group(1))
    return cycle, files[-1]


# --------------------------------------------------------------------------------------

def load_context(state: AgentState) -> dict:
    cycle, req = latest_requirements()
    return {"cycle": cycle, "requirements_path": str(req.relative_to(REPO)), "status": "running",
            "tasks": [], "gate_failures": [], "gate_fix_attempts": 0, "review_round": 0,
            "must_fix": [], **_event("load_context", f"cycle {cycle}, requirements {req.name}")}


TASKS_SCHEMA = {
    "type": "object",
    "properties": {"tasks": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "title": {"type": "string"}, "section": {"type": "string"},
                       "instructions": {"type": "string"}, "acceptance": {"type": "string"}},
        "required": ["id", "title", "section", "instructions", "acceptance"]}},
        "notes": {"type": "string"}},
    "required": ["tasks"],
}


def plan(state: AgentState) -> dict:
    """Turn requirements + open issues into an ordered task list (judgment -> LLM)."""
    lim = _limits()
    mem = memory.load_memory()
    open_issues = state.get("must_fix") or []
    task_prompt = tools.prompt("plan",
        cycle=state["cycle"],
        requirements_path=state["requirements_path"],
        review_round=state.get("review_round", 0),
        max_tasks=lim["max_tasks_per_plan"],
        open_issues=json.dumps(open_issues, indent=1, ensure_ascii=False) or "[]",
        gate_failures="\n".join(state.get("gate_failures") or []) or "none",
        lessons="\n".join(f"- {x}" for x in mem["lessons"]) or "none yet",
        repo_tree=_repo_tree(),
    )
    res = tools.run_claude(task_prompt, model=_models()["agent"], tools=tools.READ_ONLY_TOOLS,
                           json_schema=TASKS_SCHEMA, max_turns=lim["max_turns_per_task"],
                           timeout_s=lim["task_timeout_minutes"] * 60, extra_dirs=_data_dirs())
    tasks = (res.structured or {}).get("tasks", [])[: lim["max_tasks_per_plan"]]
    for t in tasks:
        t["source"] = "review" if open_issues else "plan"
    if not tasks:
        raise RuntimeError(f"plan returned no tasks: {res.text[-500:]}")
    return {"tasks": tasks, **_event("plan", f"{len(tasks)} tasks: " + "; ".join(t["title"] for t in tasks))}


def implement(state: AgentState) -> dict:
    """Do exactly one task with Claude Code, then hand the rest of the queue back."""
    lim = _limits()
    task: Task = state["tasks"][0]
    task_prompt = tools.prompt("implement",
        cycle=state["cycle"], requirements_path=state["requirements_path"],
        task=json.dumps(task, indent=1, ensure_ascii=False),
        remaining="\n".join(f"- {t['title']}" for t in state["tasks"][1:]) or "none",
    )
    res = tools.run_claude(task_prompt, model=_models()["agent"], tools=tools.EDIT_TOOLS,
                           max_turns=lim["max_turns_per_task"], timeout_s=lim["task_timeout_minutes"] * 60,
                           extra_dirs=_data_dirs())
    done = {**task, "result": res.text[-1500:]}
    return {"tasks": state["tasks"][1:], "completed": [done],
            **_event("implement", f"{task['id']} {task['title']} ({res.turns} turns)")}


def run_pipeline(state: AgentState) -> dict:
    """Tests and the full pipeline. Pure code: if code can answer, code answers."""
    # Both together must fit inside limits.task_timeout_minutes (see LONG_NODES in agent.py).
    budget = _limits()["task_timeout_minutes"] * 60
    t_code, t_out = tools.run_cmd(["python", "-m", "pytest", "-q", "-x"], timeout_s=int(budget * 0.4))
    p_code, p_out = tools.run_cmd(["python", "-m", "fml.cli", "run-all"], timeout_s=int(budget * 0.6))
    log = f"$ pytest -q -x (exit {t_code})\n{t_out[-3000:]}\n\n$ fml run-all (exit {p_code})\n{p_out[-3000:]}"
    return {"pipeline_log": log, "gate_failures": tools.run_gates(t_code == 0, p_code == 0),
            **_event("run_pipeline", f"pytest exit {t_code}, run-all exit {p_code}")}


def verify(state: AgentState) -> dict:
    """Turn failed hard gates into fix tasks (bounded), or move on."""
    fails = state.get("gate_failures") or []
    attempts = state.get("gate_fix_attempts", 0)
    if fails and attempts < _limits()["max_gate_fix_attempts"]:
        task: Task = {"id": f"gate-fix-{attempts + 1}", "title": "Fix failed hard gates", "section": "infra",
                      "source": "gate",
                      "instructions": "These deterministic checks failed. Fix the causes, not the checks:\n"
                                      + "\n".join(f"- {f}" for f in fails)
                                      + f"\n\nLast test/pipeline output:\n{state.get('pipeline_log', '')[-2500:]}",
                      "acceptance": "`python -m pytest -q` and `fml run-all` succeed and every listed gate passes."}
        return {"tasks": [task], "gate_fix_attempts": attempts + 1,
                **_event("verify", f"{len(fails)} gate failures -> fix attempt {attempts + 1}")}
    note = "all gates pass" if not fails else f"{len(fails)} gate failures remain after {attempts} attempts"
    return {"tasks": [], **_event("verify", note)}


def build_site(state: AgentState) -> dict:
    code, out = tools.run_cmd(["python", "-m", "sitegen", "build", "--strict", "--cycle", str(state["cycle"])], timeout_s=600)
    fails = list(state.get("gate_failures") or [])
    if code != 0:
        fails.append(f"site build failed: {out[-800:]}")
    return {"gate_failures": fails, **_event("build_site", f"exit {code}")}


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "score": {"type": "integer", "minimum": 0, "maximum": 10},
            "evidence": {"type": "string"}}, "required": ["id", "score", "evidence"]}},
        "must_fix": {"type": "array", "items": {"type": "object", "properties": {
            "criterion": {"type": "string"}, "issue": {"type": "string"}, "where": {"type": "string"},
            "fix": {"type": "string"}}, "required": ["criterion", "issue", "where", "fix"]}},
        "summary": {"type": "string"},
    },
    "required": ["scores", "must_fix", "summary"],
}


def review(state: AgentState) -> dict:
    """A fresh Claude session with read-only tools scores the work against the rubric."""
    lim = _limits()
    rubric = tools.load_config("rubric")
    task_prompt = tools.prompt("review",
        cycle=state["cycle"], requirements_path=state["requirements_path"],
        rubric=json.dumps(rubric["criteria"], indent=1, ensure_ascii=False),
        gate_failures="\n".join(state.get("gate_failures") or []) or "none",
    )
    res = tools.run_claude(task_prompt, model=_models()["reviewer"], tools=tools.READ_ONLY_TOOLS,
                           json_schema=REVIEW_SCHEMA, max_turns=lim["max_turns_per_task"],
                           timeout_s=lim["task_timeout_minutes"] * 60)
    out = res.structured or {}
    ids = {c["id"] for c in rubric["criteria"]}
    scores = {s["id"]: {"score": int(s["score"]), "evidence": s["evidence"]}
              for s in out.get("scores", []) if s.get("id") in ids}
    for missing in ids - scores.keys():  # an unscored criterion counts as failed, loudly
        scores[missing] = {"score": 0, "evidence": "reviewer did not score this criterion"}
    rnd = state.get("review_round", 0) + 1
    low = ", ".join(f"{k}={v['score']}" for k, v in sorted(scores.items()) if v["score"] < 10)
    return {"scores": scores, "must_fix": out.get("must_fix", []), "review_summary": out.get("summary", ""),
            "review_round": rnd, "gate_fix_attempts": 0,
            **_event("review", f"round {rnd}: below 10 -> {low or 'none'}")}


def finalize(state: AgentState) -> dict:
    """Log what happened (AI-use log, memory), and open the PR. The runner commits and pushes."""
    target = tools.load_config("agent")["target_score"]
    scores = state.get("scores") or {}
    reached = bool(scores) and all(v["score"] >= target for v in scores.values()) and not state.get("gate_failures")
    verdict = "reached 10/10 on every criterion" if reached else "STOPPED BELOW 10/10 (round limit reached)"
    table = "\n".join(f"| {k} | {v['score']} | {v['evidence'][:160]} |" for k, v in sorted(scores.items()))
    open_items = "\n".join(f"- [{i['criterion']}] {i['issue']} ({i['where']})" for i in state.get("must_fix") or [])
    gates = "\n".join(f"- {g}" for g in state.get("gate_failures") or []) or "- none"
    done_tasks = "\n".join(f"- {t.get('id')}: {t.get('title')} ({t.get('source', 'plan')})"
                           for t in state.get("completed") or [])
    body = (f"**Verdict:** {verdict} after {state.get('review_round', 0)} review round(s).\n\n"
            f"**Requirements:** `{state['requirements_path']}`\n\n"
            f"### What the AI produced (tasks completed)\n{done_tasks or '- none'}\n\n"
            f"### What was checked\nHard gates still failing:\n{gates}\n\n"
            f"| criterion | score | evidence |\n|---|---|---|\n{table}\n\n"
            f"### Open reviewer issues\n{open_items or '- none'}\n\n"
            f"### Reviewer summary\n{state.get('review_summary', '')}\n")
    memory.append_markdown("docs/ai-log.md", f"Cycle {state['cycle']} - agent run ({memory.now()[:10]})", body)
    memory.record_cycle(state["cycle"], reached_target=reached, rounds=state.get("review_round", 0),
                        scores={k: v["score"] for k, v in scores.items()})
    if not reached:
        memory.add_lesson(f"Cycle {state['cycle']} ended below target; open issues: "
                          + "; ".join(i["issue"][:80] for i in (state.get("must_fix") or [])[:5]))
    return {"status": "done", "stop_reason": verdict, "review_summary": body,
            **_event("finalize", verdict)}


# ---- routing (plain code) -----------------------------------------------------------

def after_implement(state: AgentState) -> str:
    return "implement" if state.get("tasks") else "run_pipeline"


def after_verify(state: AgentState) -> str:
    return "implement" if state.get("tasks") else "build_site"


def after_review(state: AgentState) -> str:
    target = tools.load_config("agent")["target_score"]
    scores = state.get("scores") or {}
    perfect = scores and all(v["score"] >= target for v in scores.values()) and not state.get("gate_failures")
    if perfect or state.get("review_round", 0) >= _limits()["max_review_rounds"]:
        return "finalize"
    return "plan"
