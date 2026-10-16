"""The state the graph carries from node to node. It is checkpointed after every node,
so a run can stop at any point and the next run continues from the last finished node."""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class Task(TypedDict, total=False):
    id: str
    title: str
    section: str          # notebook section slug this task serves, or "infra"
    instructions: str     # what to do, self-contained
    acceptance: str       # how we know it is done (checkable)
    source: str           # "plan" | "gate" | "review"


class Event(TypedDict, total=False):
    node: str
    at: str               # ISO timestamp
    note: str


class AgentState(TypedDict, total=False):
    cycle: int
    requirements_path: str
    status: Literal["running", "done", "failed"]
    stop_reason: str

    # work queue: replaced as a whole by plan/implement/verify
    tasks: list[Task]
    completed: Annotated[list[Task], operator.add]

    # quality loop
    gate_failures: list[str]
    gate_fix_attempts: int
    review_round: int
    scores: dict[str, dict]          # criterion id -> {"score": int, "evidence": str}
    must_fix: list[dict]             # reviewer issues still open
    review_summary: str

    pipeline_log: str                # tail of the last test + pipeline run
    history: Annotated[list[Event], operator.add]
