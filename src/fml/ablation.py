"""Compare TF-IDF alone with TF-IDF plus each extra feature group, on validation only."""
from __future__ import annotations

import pandas as pd

from .models import C_GRID, tune_on_validation

HELPS_BY = 0.01      # a group "helps" only if validation F1 rises by at least this much


def verdict(delta_f1: float) -> str:
    """Plain label for a change in validation F1."""
    if delta_f1 >= HELPS_BY:
        return "helps"
    if delta_f1 <= -HELPS_BY:
        return "hurts"
    return "no clear gain"


def run_ablation(train: pd.DataFrame, valid: pd.DataFrame, variants: dict[str, dict], reference: str,
                 c_grid=C_GRID) -> tuple[pd.DataFrame, dict]:
    """Tune every variant on validation; return the table (one row each) and the fitted pipelines."""
    pipes, rows = {}, []
    for model_id, variant in variants.items():
        pipe, grid = tune_on_validation(train, valid, variant["features"], c_grid=c_grid)
        best = grid.query("selected").iloc[0]               # the (C, class weight) with the best F1
        pipes[model_id] = pipe
        rows.append({"model_id": model_id, "label": variant["label"], "C": float(best["C"]),
                     "class_weight": best["class_weight"], "f1": float(best["f1"])})
    table = pd.DataFrame(rows)
    reference_f1 = float(table.loc[table["model_id"] == reference, "f1"].iloc[0])
    table["delta_f1"] = table["f1"] - reference_f1
    table["verdict"] = [verdict(d) for d in table["delta_f1"]]
    table.loc[table["model_id"] == reference, "verdict"] = "reference"
    return table, pipes


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
