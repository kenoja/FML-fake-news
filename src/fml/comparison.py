"""Comparison of the four final models, built from saved test predictions only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import bootstrap_ci, confusion, mcnemar_exact

SCORES = ("precision", "recall", "f1")


def model_row(name: str, y, pred) -> dict:
    """One table row: confusion counts, then each score with its 95% interval."""
    intervals = bootstrap_ci(y, pred)
    row = {"model": name, "documents": int(len(y)), "conspiracy": int(np.sum(y)), **confusion(y, pred)}
    for score in SCORES:
        row[score] = intervals[score]["estimate"]
        row[f"{score}_ci_low"] = intervals[score]["ci_low"]
        row[f"{score}_ci_high"] = intervals[score]["ci_high"]
    return row


def score_cell(value: float, low: float, high: float) -> str:
    """A score with its 95% interval, e.g. '0.598 (0.508 to 0.682)'."""
    if low == high:                     # no spread (the baseline), so an interval says nothing
        return f"{value:.3f}"
    return f"{value:.3f} ({low:.3f} to {high:.3f})"


def compact_table(table, names: dict):
    """Plain rows for the report: each score with its 95% interval in one cell, three decimals."""
    rows = []
    for _, r in table.iterrows():
        row = {"Model": names[r["model"]]}
        for score in SCORES:
            column = "F1" if score == "f1" else score.capitalize()
            row[column] = score_cell(r[score], r[score + "_ci_low"], r[score + "_ci_high"])
        rows.append(row)
    return pd.DataFrame(rows)


def mcnemar_pair(y, pred_a, pred_b, name_a: str, name_b: str) -> dict:
    """McNemar test for two models: who is right where the other is wrong, and the exact p-value."""
    test = mcnemar_exact(y, pred_a, pred_b)
    return {"a": name_a, "b": name_b,
            "a_only_correct": test["a_only_correct"], "b_only_correct": test["b_only_correct"],
            "both_correct": test["both_correct"], "both_wrong": test["both_wrong"],
            "p_value": test["p_value"], "p_text": "< 0.001" if test["p_value"] < 0.001 else f"{test['p_value']:.3f}"}   # p always with three decimals


def validation_website_table(websites, y, said_conspiracy) -> pd.DataFrame:
    """Per conspiracy website in validation: articles and the share the LLM calls conspiracy (plain names, no domains)."""
    frame = pd.DataFrame({"website": np.asarray(websites), "label": np.asarray(y, dtype=bool),
                          "hit": np.asarray(said_conspiracy, dtype=bool)})
    frame = frame[frame["label"]]                            # conspiracy articles only
    sizes = frame.groupby("website")["hit"].agg(["size", "mean"]).sort_values("size", ascending=False)
    names = {site: f"Validation website {i}" for i, site in enumerate(sizes.index, start=1)}
    return pd.DataFrame({"Website": sizes.index.map(names), "Conspiracy articles": sizes["size"].to_numpy(),
                         "Recall: LLM": sizes["mean"].round(4).to_numpy()})
