"""Evaluation toolkit: metrics, baselines, bootstrap confidence intervals, McNemar test.

Labels are booleans, True = conspiracy-oriented website (the positive class). Everything here is
pure: it takes arrays in and gives numbers out, so it can be tested on tiny hand-made examples.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import precision_recall_curve

METRIC_NAMES = ("precision", "recall", "f1", "macro_f1", "balanced_accuracy")
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261016


def confusion(y_true, y_pred) -> dict[str, int]:
    """Counts of true/false positives/negatives; "positive" means conspiracy."""
    t, p = np.asarray(y_true, dtype=bool), np.asarray(y_pred, dtype=bool)
    return {"tp": int((t & p).sum()), "fn": int((t & ~p).sum()),
            "fp": int((~t & p).sum()), "tn": int((~t & ~p).sum())}


def _div(a: float, b: float) -> float:
    """a / b, with 0 when b is 0 (a class that is never predicted has precision 0, not an error)."""
    return a / b if b else 0.0


def _f1(tp: float, fp: float, fn: float) -> float:
    return _div(2 * tp, 2 * tp + fp + fn)


def metrics_from_confusion(c: dict[str, int]) -> dict[str, float]:
    """Conspiracy-class precision/recall/F1 plus macro-F1 (both classes) and balanced accuracy."""
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    recall, specificity = _div(tp, tp + fn), _div(tn, tn + fp)
    f1_pos, f1_neg = _f1(tp, fp, fn), _f1(tn, fn, fp)
    return {"precision": _div(tp, tp + fp), "recall": recall, "f1": f1_pos,
            "macro_f1": (f1_pos + f1_neg) / 2, "balanced_accuracy": (recall + specificity) / 2}


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return metrics_from_confusion(confusion(y_true, y_pred))


def pr_curve_data(y_true, scores) -> dict[str, list[float]]:
    """Precision-recall curve points for the conspiracy class (higher score = more conspiracy)."""
    precision, recall, thresholds = precision_recall_curve(np.asarray(y_true, dtype=bool), scores)
    return {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thresholds.tolist()}


def majority_baseline(y_train, y_eval) -> dict[str, float]:
    """Always predict the majority class of the TRAINING labels."""
    majority = bool(np.mean(np.asarray(y_train, dtype=bool)) >= 0.5)
    return classification_metrics(y_eval, np.full(len(y_eval), majority))


def stratified_random_baseline(y_train, y_eval, seed: int, repeats: int = 200) -> dict[str, float]:
    """Guess at random with the training class shares; metrics averaged over `repeats` guesses."""
    rng = np.random.default_rng(seed)
    share = float(np.mean(np.asarray(y_train, dtype=bool)))
    runs = [classification_metrics(y_eval, rng.random(len(y_eval)) < share) for _ in range(repeats)]
    return {k: float(np.mean([r[k] for r in runs])) for k in METRIC_NAMES}


def _percentile_ci(values: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1 - level) / 2
    lo, hi = np.quantile(values, [alpha, 1 - alpha])
    return float(lo), float(hi)


def bootstrap_ci(y_true, y_pred, groups=None, n: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
                 level: float = 0.95, metric_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]]
                 = classification_metrics) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CI for every metric.

    Without `groups` documents are resampled one by one. With `groups` (website per document) whole
    websites are resampled with all their documents, which is the honest version here: documents of
    one site are not independent, and a split has only a few conspiracy sites.
    """
    t, p = np.asarray(y_true, dtype=bool), np.asarray(y_pred, dtype=bool)
    rng = np.random.default_rng(seed)
    if groups is None:
        draws = [rng.integers(0, len(t), len(t)) for _ in range(n)]
    else:
        g = np.asarray(groups)
        sites = np.unique(g)
        members = [np.flatnonzero(g == s) for s in sites]
        draws = [np.concatenate([members[k] for k in rng.integers(0, len(sites), len(sites))]) for _ in range(n)]
    samples = [metric_fn(t[i], p[i]) for i in draws]
    point = metric_fn(t, p)
    out = {}
    for name in point:
        lo, hi = _percentile_ci(np.array([s[name] for s in samples]), level)
        out[name] = {"estimate": float(point[name]), "ci_low": lo, "ci_high": hi}
    return out


def mcnemar_exact(y_true, pred_a, pred_b) -> dict[str, float]:
    """Exact McNemar test on paired predictions.

    Only documents where exactly one model is right carry information: `a_only` (A right, B wrong)
    and `b_only`. If the models were equally good these discordant pairs split 50/50, so the p-value
    is a two-sided exact binomial test on them.
    """
    t = np.asarray(y_true, dtype=bool)
    a_ok, b_ok = np.asarray(pred_a, dtype=bool) == t, np.asarray(pred_b, dtype=bool) == t
    a_only, b_only = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(min(a_only, b_only), n, 0.5).pvalue)
    return {"a_only_correct": a_only, "b_only_correct": b_only, "both_correct": int((a_ok & b_ok).sum()),
            "both_wrong": int((~a_ok & ~b_ok).sum()), "p_value": p}
