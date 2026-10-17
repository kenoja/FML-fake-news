"""Shared helpers for writing results/: metrics.json, CSV tables and figures.

One writer for everything keeps the numbers in one place, so prose can quote results/ files
instead of typing numbers by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import REPO_ROOT

CONSPIRACY_COLOUR = "#8c3b76"
MAINSTREAM_COLOUR = "#2b7a6a"
METRIC_KEYS = ("data", "cleaning", "split", "baseline", "ablation",
               "llm", "comparison", "errors")
DPI = 150
TABLE_DECIMALS = 3  # floats in CSV tables are written rounded, so reruns and diffs stay stable


def results_dir() -> Path:
    return REPO_ROOT / "results"


def write_metrics(section_key: str, values: dict, path: Path | None = None) -> Path:
    """Merge `values` under `section_key` into metrics.json.

    Merging (not overwriting the file) lets each stage write its own part and re-run alone.
    The file always contains every required top-level key, so readers never hit a missing one.
    """
    if section_key not in METRIC_KEYS:
        raise KeyError(f"unknown metrics section {section_key!r}; allowed: {METRIC_KEYS}")
    path = path or results_dir() / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for key in METRIC_KEYS:
        current.setdefault(key, {})
    current[section_key] = values
    path.write_text(json.dumps(current, indent=2, sort_keys=False, default=float) + "\n", encoding="utf-8")
    return path


def write_table(name: str, df: pd.DataFrame, directory: Path | None = None) -> Path:
    """Write a CSV table to results/tables/<name>.csv."""
    directory = directory or results_dir() / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name if name.endswith(".csv") else f"{name}.csv")
    df.round(TABLE_DECIMALS).to_csv(path, index=False)
    return path


def savefig(fig, name: str, directory: Path | None = None) -> Path:
    """Save a matplotlib figure to results/figures/<name>.png at 150 dpi and close it."""
    import matplotlib.pyplot as plt

    directory = directory or results_dir() / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path
