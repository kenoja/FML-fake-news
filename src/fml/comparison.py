"""Comparison tables built from saved test predictions only (nothing is fitted or scored here)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import METRIC_NAMES, bootstrap_ci, classification_metrics, confusion, mcnemar_exact, paired_delta_ci


def model_row(name: str, y, pred) -> dict:
    """One table row: counts, then every metric with its 95% interval."""
    intervals = bootstrap_ci(y, pred)
    row = {"model": name, "documents": int(len(y)), "conspiracy": int(np.sum(y)), **confusion(y, pred)}
    for metric in METRIC_NAMES:
        row[metric] = intervals[metric]["estimate"]
        row[f"{metric}_ci_low"] = intervals[metric]["ci_low"]
        row[f"{metric}_ci_high"] = intervals[metric]["ci_high"]
    return row


def pairwise(y, pred_a, pred_b, name_a: str, name_b: str) -> dict:
    """McNemar test of two models plus the F1 difference (b minus a) with its interval."""
    test = mcnemar_exact(y, pred_a, pred_b)
    return {"a": name_a, "b": name_b,
            "a_only_correct": test["a_only_correct"], "b_only_correct": test["b_only_correct"],
            "both_correct": test["both_correct"], "both_wrong": test["both_wrong"], "mcnemar_p": test["p_value"],
            "delta_f1_b_minus_a": paired_delta_ci(y, pred_a, pred_b)}


def length_bucket_f1(words, y, preds: dict[str, np.ndarray], edges=(150, 250)) -> pd.DataFrame:
    """Conspiracy F1 per model by article length (words); NaN when a bucket has no conspiracy page."""
    w, y = np.asarray(words), np.asarray(y, dtype=bool)
    bounds = [0, *edges, int(w.max()) + 1]
    rows = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        m = (w >= lo) & (w < hi)
        row = {"bucket": f"{lo}-{hi - 1} words" if hi <= bounds[-2] else f">= {lo} words",
               "documents": int(m.sum()), "conspiracy": int(y[m].sum())}
        for name, p in preds.items():
            row[f"f1_{name}"] = classification_metrics(y[m], np.asarray(p)[m])["f1"] if y[m].any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
