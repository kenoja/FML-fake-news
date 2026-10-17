"""Ablation: which feature groups help the logistic regression, judged on validation only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import METRIC_NAMES, bootstrap_ci, mcnemar_exact, paired_delta_ci
from .models import C_GRID, tune_on_validation


def metric_columns(doc: dict) -> dict[str, float]:
    """Flat table columns: estimate plus 95% interval, per metric."""
    row = {}
    for m in METRIC_NAMES:
        row[m] = doc[m]["estimate"]
        row[f"{m}_ci_low"] = doc[m]["ci_low"]
        row[f"{m}_ci_high"] = doc[m]["ci_high"]
    return row


def verdict(ci_low: float, ci_high: float) -> str:
    """helps / hurts only when the whole CI is on one side of 0; anything else is "no clear gain"."""
    if ci_low > 0:
        return "helps"
    if ci_high < 0:
        return "hurts"
    return "no clear gain"


def delta_columns(prefix: str, y, ref_pred, new_pred, n_boot: int) -> dict:
    """Score difference with its interval, a verdict and the McNemar test."""
    delta = paired_delta_ci(y, ref_pred, new_pred, n=n_boot)
    test = mcnemar_exact(y, ref_pred, new_pred)
    return {prefix: delta["estimate"], f"{prefix}_ci_low": delta["ci_low"], f"{prefix}_ci_high": delta["ci_high"],
            f"{prefix}_verdict": verdict(delta["ci_low"], delta["ci_high"]),
            f"mcnemar_p_{prefix.split('_')[-1]}": test["p_value"],
            f"{prefix}_only_ref_correct": test["a_only_correct"], f"{prefix}_only_new_correct": test["b_only_correct"]}


def run_ablation(train: pd.DataFrame, valid: pd.DataFrame, variants: dict[str, dict], reference: str,
                 full: str | None = None, drop_one: tuple[str, ...] = (), c_grid=C_GRID,
                 n_boot: int = 2000) -> tuple[pd.DataFrame, dict[str, object], dict[str, np.ndarray]]:
    """Tune and score every variant; return (table, fitted pipelines, validation predictions)."""
    y = valid["label"].to_numpy()
    pipes, preds, grids, rows = {}, {}, {}, []
    for model_id, v in variants.items():
        pipe, grid = tune_on_validation(train, valid, v["features"], c_grid=c_grid)
        pipes[model_id], preds[model_id], grids[model_id] = pipe, pipe.predict(valid), grid
    for model_id, v in variants.items():
        pred, sel = preds[model_id], grids[model_id].query("selected").iloc[0]
        row = {"model_id": model_id, "label": v["label"], "kind": v["kind"], "C": float(sel["C"]),
               "class_weight": sel["class_weight"], "validation_documents": int(len(y)),
               **metric_columns(bootstrap_ci(y, pred, n=n_boot))}
        if model_id != reference:
            row.update(delta_columns("delta_f1_vs_m1", y, preds[reference], pred, n_boot))
        if model_id in drop_one:
            row.update(delta_columns("contribution_f1_vs_m2", y, pred, preds[full], n_boot))
        rows.append(row)
    return pd.DataFrame(rows), pipes, preds


def coefficient_tables(pipe, model_id: str, k: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(top n-grams per class, standardised dense coefficients) of one fitted pipeline."""
    names = pipe.named_steps["features"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    is_tfidf = np.array([str(n).startswith("tfidf__") for n in names])
    grams = pd.DataFrame({"feature": [str(n)[len("tfidf__"):] for n in names[is_tfidf]], "coefficient": coef[is_tfidf]})
    grams["n"] = grams["feature"].str.count(" ") + 1
    # sort by coefficient, then by term, so ties always come out in the same order
    towards_conspiracy = grams.sort_values(["coefficient", "feature"], ascending=[False, True]).head(k)
    towards_mainstream = grams.sort_values(["coefficient", "feature"], ascending=[True, True]).head(k)
    top = [towards_conspiracy.assign(direction="conspiracy"), towards_mainstream.assign(direction="mainstream")]
    top = pd.concat(top)[["direction", "feature", "n", "coefficient"]].assign(model_id=model_id)
    dense = pd.DataFrame({"group": [str(n).split("__", 1)[0] for n in names[~is_tfidf]],
                          "feature": [str(n).split("__", 1)[1] for n in names[~is_tfidf]],
                          "coefficient_standardised": coef[~is_tfidf]}).assign(model_id=model_id)
    dense["size"] = dense["coefficient_standardised"].abs()
    dense = dense.sort_values(["size", "feature"], ascending=[False, True]).drop(columns="size")
    return top.reset_index(drop=True), dense.reset_index(drop=True)


def group_verdicts(table: pd.DataFrame, groups: dict[str, dict]) -> dict[str, dict]:
    """Verdict per feature group from two views, each judged by the bootstrap interval of delta-F1."""
    t = table.set_index("model_id")
    out = {}
    for g, ids in groups.items():
        a, d = t.loc[ids["added"]], t.loc[ids["dropped"]]
        va, vd = a["delta_f1_vs_m1_verdict"], d["contribution_f1_vs_m2_verdict"]
        out[g] = {
            "verdict": va if va == vd else "no clear gain",
            "added_to_words": {"model_id": ids["added"], "delta_f1": a["delta_f1_vs_m1"], "verdict": va,
                               "ci": [a["delta_f1_vs_m1_ci_low"], a["delta_f1_vs_m1_ci_high"]],
                               "mcnemar_p": a["mcnemar_p_m1"]},
            "contribution_in_m2": {"model_id": ids["dropped"], "delta_f1": d["contribution_f1_vs_m2"], "verdict": vd,
                                   "ci": [d["contribution_f1_vs_m2_ci_low"], d["contribution_f1_vs_m2_ci_high"]],
                                   "mcnemar_p": d["mcnemar_p_m2"]},
        }
    return out


def choose_between(table: pd.DataFrame, m1: str, m2: str) -> dict:
    """Pick M2 only if its delta-F1 over M1 is positive with the whole bootstrap interval above 0."""
    r = table.set_index("model_id").loc[m2]
    helps = r["delta_f1_vs_m1_verdict"] == "helps"
    return {"chosen": m2 if helps else m1, "rule": "M2 only if delta-F1 vs M1 has a bootstrap interval above 0 on validation",
            "delta_f1": r["delta_f1_vs_m1"], "ci": [r["delta_f1_vs_m1_ci_low"], r["delta_f1_vs_m1_ci_high"]]}


def epistemic_negation(train: pd.DataFrame, valid: pd.DataFrame, words_cfg: dict, c_grid=C_GRID,
                       n_boot: int = 2000) -> tuple[pd.DataFrame, dict]:
    """Words + epistemic group with plain vs negation-aware counts, validation only."""
    variants = {"epistemic_plain": {"label": "words + epistemic (plain counts)", "kind": "reference",
                                    "features": {**words_cfg, "dense_groups": ["epistemic"]}},
                "epistemic_negation_aware": {"label": "words + epistemic (negation-aware counts)", "kind": "variant",
                                             "features": {**words_cfg, "dense_groups": ["epistemic_negation"]}}}
    table, _, _ = run_ablation(train, valid, variants, reference="epistemic_plain", c_grid=c_grid, n_boot=n_boot)
    t = table.set_index("model_id")
    new = t.loc["epistemic_negation_aware"]
    summary = {"data": "filtered documents, split_v1, validation rows only (test never read)",
               "negation_window_tokens": 3, "negators": ["not", "no", "never", "*n't"],
               "f1_plain": t.loc["epistemic_plain", "f1"], "f1_negation_aware": new["f1"],
               "delta_f1": new["delta_f1_vs_m1"],
               "ci": [new["delta_f1_vs_m1_ci_low"], new["delta_f1_vs_m1_ci_high"]],
               "mcnemar_p": new["mcnemar_p_m1"], "verdict": new["delta_f1_vs_m1_verdict"],
               "selected_C": {m: float(t.loc[m, "C"]) for m in t.index}}
    return table, summary
