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


def compact_table(table, names: dict):
    """Four plain rows for the report: each score with its 95% interval in one cell, three decimals."""
    rows = []
    for _, r in table.iterrows():
        row = {"Model": names[r["model"]]}
        for score in SCORES:
            row[score.capitalize() if score != "f1" else "F1"] = (
                f"{r[score]:.3f} ({r[score + '_ci_low']:.3f} to {r[score + '_ci_high']:.3f})")
        rows.append(row)
    return pd.DataFrame(rows)


def mcnemar_pair(y, pred_a, pred_b, name_a: str, name_b: str) -> dict:
    """McNemar test for two models: who is right where the other is wrong, and the exact p-value."""
    test = mcnemar_exact(y, pred_a, pred_b)
    return {"a": name_a, "b": name_b,
            "a_only_correct": test["a_only_correct"], "b_only_correct": test["b_only_correct"],
            "both_correct": test["both_correct"], "both_wrong": test["both_wrong"],
            "p_value": test["p_value"], "p_text": "< 0.0001" if test["p_value"] < 0.0001 else f"{test['p_value']:.3f}"}
