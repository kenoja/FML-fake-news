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


def duplicate_test_ids(pairs: pd.DataFrame, split: pd.Series) -> set:
    """Test documents with a near-duplicate partner in train or validation.

    `split` maps doc_id -> split name. Pairs inside the test split are kept: both texts are test
    documents, so neither can have been learned from.
    """
    if not len(pairs):
        return set()
    a, b = pairs["doc_id_a"], pairs["doc_id_b"]
    sa, sb = a.map(split), b.map(split)
    out = set(a[(sa == "test") & sb.isin(["train", "validation"])])
    return out | set(b[(sb == "test") & sa.isin(["train", "validation"])])


def no_duplicate_sensitivity(doc_ids, y, preds: dict[str, np.ndarray], excluded: set,
                             a: str, b: str, sites=None) -> tuple[pd.DataFrame, dict]:
    """Recompute conspiracy P/R/F1 on the test documents without near-duplicate partners.

    Why: a test text that is a near copy of a training text is partly "seen", which could flatter a
    model that memorises (the logistic regression) more than one that does not. The split stays
    frozen; this only quantifies how much the head-to-head depends on those documents.
    Returns a table (all test vs. remainder, per model) and a summary with F1 deltas (b minus a).
    With `sites` (website per document) every row also carries the document and website bootstrap
    intervals from `model_row`, computed from the saved predictions only.
    """
    ids, y = np.asarray(doc_ids), np.asarray(y, dtype=bool)
    keep = ~np.isin(ids, list(excluded))
    rows, f1 = [], {}
    for name, p in preds.items():
        p = np.asarray(p, dtype=bool)
        for setting, m in (("all_test", np.ones(len(y), dtype=bool)), ("without_duplicates", keep)):
            met = classification_metrics(y[m], p[m]) if m.any() else dict.fromkeys(METRIC_NAMES, float("nan"))
            row = {"model": name, "setting": setting, "documents": int(m.sum()),
                   "conspiracy": int(y[m].sum()), **{k: met[k] for k in ("precision", "recall", "f1")}}
            if sites is not None and m.any():
                full = model_row(name, y[m], p[m], np.asarray(sites)[m])
                row.update({k: v for k, v in full.items() if k not in ("model", "documents", "conspiracy", *row)})
            rows.append(row)
            f1[(name, setting)] = met["f1"]
    summary = {"n_excluded": int((~keep).sum()), "n_excluded_conspiracy": int(y[~keep].sum()),
               "n_remaining": int(keep.sum()), "n_remaining_conspiracy": int(y[keep].sum()),
               "f1_all_test": {n: f1[(n, "all_test")] for n in preds},
               "f1_without_duplicates": {n: f1[(n, "without_duplicates")] for n in preds},
               "delta_f1_without_minus_all": {n: f1[(n, "without_duplicates")] - f1[(n, "all_test")] for n in preds},
               "gap_f1_b_minus_a_all_test": f1[(b, "all_test")] - f1[(a, "all_test")],
               "gap_f1_b_minus_a_without_duplicates": f1[(b, "without_duplicates")] - f1[(a, "without_duplicates")]}
    summary["same_winner"] = bool((summary["gap_f1_b_minus_a_all_test"] > 0)
                                  == (summary["gap_f1_b_minus_a_without_duplicates"] > 0))
    return pd.DataFrame(rows), summary
