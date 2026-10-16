import argparse
import re
from pathlib import Path

from .build import build

REPO = Path(__file__).resolve().parents[1]


def latest_cycle() -> int:
    files = sorted((REPO / "requirements").glob("cycle-*.md"))
    return int(re.search(r"cycle-(\d+)", files[-1].stem).group(1)) if files else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="python -m sitegen")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="render content/ into site/")
    b.add_argument("--cycle", type=int)
    b.add_argument("--strict", action="store_true", help="fail if any code/output cell cannot be built")
    a = p.parse_args()
    build(a.cycle or latest_cycle(), strict=a.strict)
