"""Compare TF-IDF alone with TF-IDF plus each extra feature group, on validation only."""
from __future__ import annotations

import pandas as pd

from .models import tune_variant

HELPS_BY = 0.01      # a group "helps" only if validation F1 rises by at least this much


def verdict(delta_f1):
    """Plain label for a change in validation F1."""
    if delta_f1 >= HELPS_BY:
        return "helps"
    if delta_f1 <= -HELPS_BY:
        return "hurts"
    return "no clear gain"


def run_ablation(train, valid, variants, reference, c_grid=None):
    """Tune every variant on validation; return the table (one row each) and the fitted pipelines."""
    pipes, rows = {}, []
    for model_id in variants:
        pipe, row = tune_variant(train, valid, variants[model_id]["features"], c_grid)
        row["model_id"] = model_id
        row["label"] = variants[model_id]["label"]
        pipes[model_id] = pipe
        rows.append(row)
    table = pd.DataFrame(rows, columns=["model_id", "label", "C", "class_weight", "f1"])
    return add_verdicts(table, reference), pipes


def add_verdicts(table, reference):
    """Compare every model with the reference model and label the change."""
    is_reference = table["model_id"] == reference
    table["delta_f1"] = table["f1"] - table.loc[is_reference, "f1"].iloc[0]
    table["verdict"] = [verdict(d) for d in table["delta_f1"]]
    table.loc[is_reference, "verdict"] = "reference"
    return table


def choose_best(table: pd.DataFrame, reference: str) -> dict:
    """Pick the best row, but keep the plain reference unless another row clearly helps."""
    best = table.sort_values("f1", ascending=False, kind="stable").iloc[0]
    chosen = best["model_id"] if best["verdict"] == "helps" else reference
    rule = f"best validation F1, but only if it beats TF-IDF alone by at least {HELPS_BY}"
    return {"chosen": chosen, "rule": rule, "f1": float(table.set_index("model_id").loc[chosen, "f1"])}


def verdict_sentence(table: pd.DataFrame, reference: str) -> str:
    """One sentence that answers: do the extra features help?"""
    extras = table[table["model_id"] != reference]
    helping = extras.loc[extras["verdict"] == "helps", "label"].tolist()
    best_gain = float(extras["delta_f1"].max())
    if not helping:
        return (f"No extra feature group raised validation F1 by {HELPS_BY} or more "
                f"(the largest gain was {best_gain:.3f}), so the extra features do not help.")
    return f"These groups raised validation F1 by {HELPS_BY} or more: {', '.join(helping)}."
