"""Helpers that write results/: metrics.json, CSV tables and figures."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import REPO_ROOT

CONSPIRACY_COLOUR = "#8c3b76"
MAINSTREAM_COLOUR = "#2b7a6a"
METRIC_KEYS = ("data", "cleaning", "split", "word_counts", "tfidf", "tfidf_features",
               "llm", "comparison", "errors")
DPI = 150
TABLE_DECIMALS = 3          # tables are rounded, so two runs give identical files


def results_dir() -> Path:
    """The results folder of this repository."""
    return REPO_ROOT / "results"


def write_metrics(section_key: str, values: dict, path: Path | None = None) -> Path:
    """Put `values` under `section_key` in metrics.json, keeping the other sections."""
    if section_key not in METRIC_KEYS:
        raise KeyError(f"unknown metrics section {section_key!r}; allowed: {METRIC_KEYS}")
    path = path or results_dir() / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for key in METRIC_KEYS:
        current.setdefault(key, {})             # every key is always present
    current[section_key] = values
    path.write_text(json.dumps(current, indent=2, default=float) + "\n", encoding="utf-8")
    return path


def write_table(name: str, df: pd.DataFrame, directory: Path | None = None) -> Path:
    """Write a rounded CSV table to results/tables/<name>.csv."""
    directory = directory or results_dir() / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name if name.endswith(".csv") else f"{name}.csv")
    three_decimals(df.round(TABLE_DECIMALS)).to_csv(path, index=False)
    return path


def three_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """Write every score column (all values from 0 to 1) with exactly three decimals: 0.480, 1.000."""
    out = df.copy()
    for column in out.select_dtypes("float").columns:
        values = out[column].dropna()
        if values.any() and values.between(0, 1).all():              # shares and scores, not settings such as C
            out[column] = out[column].map(lambda x: "" if pd.isna(x) else f"{x:.{TABLE_DECIMALS}f}")
    return out


def savefig(fig, name: str, directory: Path | None = None) -> Path:
    """Save a matplotlib figure to results/figures/<name>.png and close it."""
    import matplotlib.pyplot as plt

    directory = directory or results_dir() / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path
