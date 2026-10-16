"""Stand-in for Claude Code, used only when FML_AGENT_FAKE_LLM=1 (tests and dry runs).

It lets the graph, checkpointing and routing be tested without spending usage.
"""
from __future__ import annotations

from .tools import ClaudeResult, load_config


def respond(task_prompt: str, json_schema: dict | None) -> ClaudeResult:
    props = (json_schema or {}).get("properties", {})
    if "tasks" in props:
        return ClaudeResult("planned", {"tasks": [{
            "id": "t1", "title": "fake task", "section": "infra",
            "instructions": "do nothing", "acceptance": "nothing"}]}, 0.0, 1)
    if "scores" in props:
        ids = [c["id"] for c in load_config("rubric")["criteria"]]
        return ClaudeResult("reviewed", {"scores": [{"id": i, "score": 10, "evidence": "fake"} for i in ids],
                                         "must_fix": [], "summary": "fake review"}, 0.0, 1)
    return ClaudeResult("implemented (fake)", None, 0.0, 1)
