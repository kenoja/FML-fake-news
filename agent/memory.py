"""Two kinds of memory.

Short-term: the LangGraph checkpoint for the current cycle (.agent_state/cycle-NN.sqlite).
  It is committed to the cycle branch after every run, so the next run - possibly hours later,
  on a fresh cloud machine - continues from the last finished node.

Long-term: plain files that survive across cycles and that humans can read and edit:
  .agent_state/memory.json  - lessons and scores the agent carries into the next cycle
  docs/decisions.md         - decisions and conflicts (never blended; see AGENTS.md)
  docs/ai-log.md            - what the AI produced, what was checked/changed, what we learned
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

REPO = Path(__file__).resolve().parents[1]
STATE_DIR = REPO / ".agent_state"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def checkpointer(cycle: int) -> tuple[SqliteSaver, sqlite3.Connection]:
    STATE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(STATE_DIR / f"cycle-{cycle:02d}.sqlite", check_same_thread=False)
    return SqliteSaver(conn), conn


def thread_config(cycle: int) -> dict:
    return {"configurable": {"thread_id": f"cycle-{cycle:02d}"}, "recursion_limit": 500}


# ---- long-term memory ----------------------------------------------------------------

def _memory_path() -> Path:
    return STATE_DIR / "memory.json"


def load_memory() -> dict:
    p = _memory_path()
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"lessons": [], "cycles": {}, "runs": {}}


def save_memory(mem: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    _memory_path().write_text(json.dumps(mem, indent=2, ensure_ascii=False), encoding="utf-8")


def add_lesson(text: str) -> None:
    mem = load_memory()
    if text not in mem["lessons"]:
        mem["lessons"].append(text)
        mem["lessons"] = mem["lessons"][-40:]
    save_memory(mem)


def count_run(cycle: int) -> int:
    mem = load_memory()
    key = f"{cycle:02d}"
    mem["runs"][key] = mem["runs"].get(key, 0) + 1
    save_memory(mem)
    return mem["runs"][key]


def record_cycle(cycle: int, **fields) -> None:
    mem = load_memory()
    mem["cycles"].setdefault(f"{cycle:02d}", {}).update(fields)
    save_memory(mem)


def append_markdown(rel_path: str, heading: str, body: str) -> None:
    path = REPO / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    path.write_text(existing.rstrip() + f"\n\n## {heading}\n\n{body.strip()}\n", encoding="utf-8")


def write_status(cycle: int, status: str, detail: str) -> None:
    """A tiny human-readable status file; the routine prints it at the end of every run."""
    STATE_DIR.mkdir(exist_ok=True)
    (STATE_DIR / "last_run.md").write_text(
        f"# Last agent run\n\n- cycle: {cycle}\n- status: {status}\n- at: {now()}\n\n{detail.strip()}\n",
        encoding="utf-8")
