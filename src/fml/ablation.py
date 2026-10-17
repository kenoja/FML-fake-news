"""Compare TF-IDF alone with TF-IDF plus each extra feature group, on validation only."""
from __future__ import annotations

import pandas as pd

from .models import feature_matrix, fit_features, tune_on_validation
from .paths import project_config

THRESHOLD = 0.01     # a group counts as "above threshold" only if validation F1 rises by this much


def verdict(delta_f1):
    """Plain label for a change in validation F1."""
    if delta_f1 >= THRESHOLD:
        return "above threshold"
    if delta_f1 <= -THRESHOLD:
        return "hurts"
    return "no clear gain"


def run_ablation(train, valid, variants, reference, c_grid=None):
    """Tune every variant on validation; return the table (one row each)."""
    class_weight = project_config()["lr"]["class_weight"]
    rows = []
    for model_id, variant in variants.items():
        blocks = fit_features(train, variant["features"])       # fitted on training articles only
        X_train, X_valid = feature_matrix(blocks, train), feature_matrix(blocks, valid)
        _, grid = tune_on_validation(X_train, train["label"], X_valid, valid["label"], c_grid)
        best = grid[grid["chosen"]].iloc[0]
        rows.append({"model_id": model_id, "label": variant["label"], "C": float(best["C"]),
                     "class_weight": str(class_weight or "none"), "f1": float(best["f1"])})
    table = pd.DataFrame(rows, columns=["model_id", "label", "C", "class_weight", "f1"])
    return add_verdicts(table, reference)


def add_verdicts(table, reference):
    """Compare every model with the reference model and label the change."""
    is_reference = table["model_id"] == reference
    table["delta_f1"] = table["f1"] - table.loc[is_reference, "f1"].iloc[0]
    table["verdict"] = [verdict(d) for d in table["delta_f1"]]
    table.loc[is_reference, "verdict"] = "reference"
    return table


def choose_best(table: pd.DataFrame, reference: str) -> dict:
    """Pick the best row, but keep the plain reference unless another row is clearly above the threshold."""
    best = table.sort_values("f1", ascending=False, kind="stable").iloc[0]
    chosen = best["model_id"] if best["verdict"] == "above threshold" else reference
    rule = f"best validation F1, but only if it beats TF-IDF alone by at least {THRESHOLD}"
    return {"chosen": chosen, "rule": rule, "f1": float(table.set_index("model_id").loc[chosen, "f1"])}


def verdict_sentence(table: pd.DataFrame, reference: str) -> str:
    """One sentence that answers: do the extra features help?"""
    extras = table[table["model_id"] != reference]
    helping = extras.loc[extras["verdict"] == "above threshold", "label"].tolist()
    best_gain = float(extras["delta_f1"].max())
    if not helping:
        return (f"No extra feature group raised validation F1 by {THRESHOLD} or more "
                f"(the largest gain was {best_gain:.3f}), so the extra features do not help.")
    return f"These groups raised validation F1 by {THRESHOLD} or more: {', '.join(helping)}."
