"""Head-to-head comparison on the test documents, computed from saved prediction files only.

Nothing here fits or scores a model: the predictions were produced once by `evaluate_final`, so this
module cannot be used to pick between models on test. Functions are pure (arrays in, numbers out) so
tests can re-compute every statistic by hand on tiny examples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import (METRIC_NAMES, bootstrap_ci, classification_metrics, confusion, mcnemar_exact,
                       paired_delta_ci)


def model_row(name: str, y, pred, sites) -> dict:
    """One table row: counts, conspiracy P/R/F1, macro-F1, balanced accuracy, with both bootstrap CIs."""
    doc, site = bootstrap_ci(y, pred), bootstrap_ci(y, pred, groups=sites)
    row = {"model": name, "documents": int(len(y)), "conspiracy": int(np.sum(y)), "websites": int(len(set(sites))),
           **confusion(y, pred)}
    for m in METRIC_NAMES:
        row.update({m: doc[m]["estimate"], f"{m}_ci_doc_low": doc[m]["ci_low"], f"{m}_ci_doc_high": doc[m]["ci_high"],
                    f"{m}_ci_site_low": site[m]["ci_low"], f"{m}_ci_site_high": site[m]["ci_high"]})
    return row


def pairwise(y, pred_a, pred_b, sites, name_a: str, name_b: str) -> dict:
    """McNemar exact test with the 2x2 discordant counts, plus paired-bootstrap delta-F1 (b minus a)."""
    mc = mcnemar_exact(y, pred_a, pred_b)
    return {"a": name_a, "b": name_b,
            "a_only_correct": mc["a_only_correct"], "b_only_correct": mc["b_only_correct"],
            "both_correct": mc["both_correct"], "both_wrong": mc["both_wrong"], "mcnemar_p": mc["p_value"],
            "delta_f1_b_minus_a": {"documents": paired_delta_ci(y, pred_a, pred_b),
                                   "websites": paired_delta_ci(y, pred_a, pred_b, groups=sites)}}


def per_website_table(sites, y, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Accuracy per website and model, with n docs: with few test websites this is the real picture."""
    df = pd.DataFrame({"website": np.asarray(sites), "label": np.asarray(y, dtype=bool)})
    for name, p in preds.items():
        df[name] = np.asarray(p, dtype=bool) == df["label"]
    g = df.groupby("website")
    out = pd.DataFrame({"documents": g.size(),
                        "label": g["label"].first().map({True: "conspiracy", False: "mainstream"})})
    for name in preds:
        out[f"accuracy_{name}"] = g[name].mean()
    return out.reset_index().sort_values(["label", "website"]).reset_index(drop=True)


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
