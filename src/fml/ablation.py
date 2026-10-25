"""Compare TF-IDF alone with TF-IDF plus each extra feature group, on validation only."""
from __future__ import annotations

import pandas as pd

from .models import feature_matrix, fit_features, tune_on_validation
from .paths import project_config


def run_ablation(train, valid, variants, reference, c_grid=None):
    """Tune every variant on validation; return the table (one row each)."""
    class_weight = project_config()["lr"]["class_weight"]
    rows, n_columns = [], {}
    for model_id, variant in variants.items():
        blocks = fit_features(train, variant["features"])       # fitted on training articles only
        X_train, X_valid = feature_matrix(blocks, train), feature_matrix(blocks, valid)
        n_columns[model_id] = int(X_train.shape[1])             # phrase columns plus engineered columns
        _, grid = tune_on_validation(X_train, train["label"], X_valid, valid["label"], c_grid)
        best = grid[grid["chosen"]].iloc[0]
        rows.append({"model_id": model_id, "label": variant["label"], "C": float(best["C"]),
                     "class_weight": str(class_weight or "none"), "f1": float(best["f1"])})
    table = pd.DataFrame(rows, columns=["model_id", "label", "C", "class_weight", "f1"])
    table["delta_f1"] = table["f1"] - table.loc[table["model_id"] == reference, "f1"].iloc[0]   # change vs reference
    table.attrs["n_columns"] = n_columns
    return table
