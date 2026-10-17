"""Command line entry point: `fml <command>`.

`data-summary` works from the start. `run-all` calls fml.pipeline.run_all(), which the agent
writes in cycle 1; until then it fails loudly instead of pretending to work.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data-summary", help="print document and website counts")
    sub.add_parser("prepare-data", help="clean, describe lengths, create/read the frozen split, write results")
    sub.add_parser("run-all",help="reproduce every result in results/ from the data repo")
    final = sub.add_parser("evaluate-final", help="score one model on the TEST split (logged; see AGENTS.md)")
    final.add_argument("model_id")
    agree = sub.add_parser("stance-agreement", help="Cohen's kappa between human and Claude stance codes")
    agree.add_argument("csv", nargs="?", default="results/tables/stance_check_sample.csv")
    args = parser.parse_args(argv)

    if args.command == "stance-agreement":
        from .errors import human_agreement

        print(json.dumps(human_agreement(Path(args.csv)), indent=2))
        return 0

    if args.command == "data-summary":
        from .data import load_corpus, summary

        print(json.dumps(summary(load_corpus()), indent=2))
        return 0

    if args.command == "prepare-data":
        from .data_report import run_data_stage

        run_data_stage()
        return 0

    try:
        from . import pipeline  # written by the agent
    except ImportError as exc:
        print(f"fml {args.command}: not implemented yet ({exc}). See AGENTS.md, section 'Contracts'.",
              file=sys.stderr)
        return 2
    if args.command == "run-all":
        pipeline.run_all()
    else:
        pipeline.evaluate_final(args.model_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
