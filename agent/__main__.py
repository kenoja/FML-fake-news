"""CLI: python -m agent run | status | reset-cycle N

Dependencies are checked before the graph is imported, so a fresh cloud machine can start
with a bare `python -m agent run`.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def ensure_installed() -> None:
    if all(importlib.util.find_spec(m) for m in ("langgraph", "sklearn", "fml", "sitegen")):
        return
    print("Installing the project (pip install -e .[dev]) ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", "-e", f"{REPO}[dev]"],
                   check=True)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="advance the current cycle for up to --minutes, then save and stop")
    r.add_argument("--cycle", type=int)
    r.add_argument("--minutes", type=float)
    r.add_argument("--no-git", action="store_true", help="do not switch branches, commit or push")
    sub.add_parser("status", help="print the last run summary")
    reset = sub.add_parser("reset-cycle", help="delete the checkpoint of cycle N (start it over)")
    reset.add_argument("cycle", type=int)
    args = parser.parse_args()

    if args.cmd == "status":
        p = REPO / ".agent_state" / "last_run.md"
        print(p.read_text(encoding="utf-8") if p.is_file() else "no run yet")
        return 0
    if args.cmd == "reset-cycle":
        p = REPO / ".agent_state" / f"cycle-{args.cycle:02d}.sqlite"
        p.unlink(missing_ok=True)
        print(f"removed {p.name}")
        return 0

    ensure_installed()
    from .agent import run
    status = run(cycle=args.cycle, minutes=args.minutes, use_git=not args.no_git)
    return 0 if status in {"CONTINUE", "DONE", "PAUSED"} else 1


if __name__ == "__main__":
    sys.exit(main())
