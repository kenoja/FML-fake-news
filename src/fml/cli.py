"""Command line entry point: `fml <command>`.

`data-summary` works from the start. `run-all` calls fml.pipeline.run_all(), which the agent
writes in cycle 1; until then it fails loudly instead of pretending to work.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data-summary", help="print document and website counts")
    sub.add_parser("prepare-data", help="clean, describe lengths, create/read the frozen split, write results")
    sub.add_parser("run-all",help="reproduce every result in results/ from the data repo")
    final = sub.add_parser("evaluate-final", help="score one model on the TEST split (logged; see AGENTS.md)")
    final.add_argument("model_id")
    args = parser.parse_args(argv)

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
        pipeline.freeze_model(args.model_id)        # write the validation-chosen setting before the first test look
        result = pipeline.evaluate_final(args.model_id)
        print(json.dumps({k: result[k] for k in ("model_id", "config_hash", "n_docs", "confusion")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
