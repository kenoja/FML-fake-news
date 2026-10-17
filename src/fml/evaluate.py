"""Scoring tools: metrics, baselines, bootstrap confidence intervals and the McNemar test."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import precision_recall_curve

METRIC_NAMES = ("precision", "recall", "f1", "macro_f1", "balanced_accuracy")
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261016


def confusion(y_true, y_pred) -> dict[str, int]:
    """Count true/false positives/negatives; "positive" means conspiracy."""
    true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    return {"tp": int((true & pred).sum()), "fn": int((true & ~pred).sum()),
            "fp": int((~true & pred).sum()), "tn": int((~true & ~pred).sum())}


def divide(a: float, b: float) -> float:
    """Divide, but give 0 when the divisor is 0."""
    return a / b if b else 0.0


def f1_score(tp: float, fp: float, fn: float) -> float:
    """F1 from the three counts that matter."""
    return divide(2 * tp, 2 * tp + fp + fn)


def metrics_from_confusion(c: dict[str, int]) -> dict[str, float]:
    """Precision, recall and F1 of the conspiracy class, plus two overall scores."""
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    f1_conspiracy = f1_score(tp, fp, fn)
    f1_mainstream = f1_score(tn, fn, fp)
    return {"precision": divide(tp, tp + fp), "recall": recall, "f1": f1_conspiracy,
            "macro_f1": (f1_conspiracy + f1_mainstream) / 2,
            "balanced_accuracy": (recall + specificity) / 2}


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """All metrics for one set of predictions."""
    return metrics_from_confusion(confusion(y_true, y_pred))


def pr_curve_data(y_true, scores) -> dict[str, list[float]]:
    """Points of the precision-recall curve (higher score = more conspiracy)."""
    precision, recall, thresholds = precision_recall_curve(np.asarray(y_true, dtype=bool), scores)
    return {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thresholds.tolist()}


def majority_baseline(y_train, y_eval) -> dict[str, float]:
    """Always guess the majority class of the training labels."""
    majority = bool(np.mean(np.asarray(y_train, dtype=bool)) >= 0.5)
    guesses = np.full(len(y_eval), majority)
    return classification_metrics(y_eval, guesses)


def stratified_random_baseline(y_train, y_eval, seed: int, repeats: int = 200) -> dict[str, float]:
    """Guess at random with the training class shares, averaged over many guesses."""
    rng = np.random.default_rng(seed)
    share = float(np.mean(np.asarray(y_train, dtype=bool)))
    runs = []
    for _ in range(repeats):
        guesses = rng.random(len(y_eval)) < share
        runs.append(classification_metrics(y_eval, guesses))
    return {name: float(np.mean([run[name] for run in runs])) for name in METRIC_NAMES}


def percentile_interval(values, level: float) -> tuple[float, float]:
    """Lower and upper end that cover `level` of the values."""
    tail = (1 - level) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    return float(low), float(high)


def bootstrap_ci(y_true, y_pred, n: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
                 level: float = 0.95) -> dict[str, dict[str, float]]:
    """95% bootstrap interval for every metric, resampling documents."""
    true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        pick = rng.integers(0, len(true), len(true))      # draw documents with replacement
        samples.append(classification_metrics(true[pick], pred[pick]))
    point = classification_metrics(true, pred)
    result = {}
    for name in point:
        low, high = percentile_interval([s[name] for s in samples], level)
        result[name] = {"estimate": float(point[name]), "ci_low": low, "ci_high": high}
    return result


def paired_delta_ci(y_true, pred_ref, pred_new, n: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
                    level: float = 0.95, metric: str = "f1") -> dict[str, float]:
    """Bootstrap interval of the score difference (new minus reference) on the same documents."""
    true, ref, new = (np.asarray(v, dtype=bool) for v in (y_true, pred_ref, pred_new))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        pick = rng.integers(0, len(true), len(true))
        score_new = classification_metrics(true[pick], new[pick])[metric]
        score_ref = classification_metrics(true[pick], ref[pick])[metric]
        deltas.append(score_new - score_ref)
    low, high = percentile_interval(deltas, level)
    estimate = classification_metrics(true, new)[metric] - classification_metrics(true, ref)[metric]
    return {"estimate": float(estimate), "ci_low": low, "ci_high": high}


def mcnemar_exact(y_true, pred_a, pred_b) -> dict[str, float]:
    """Exact McNemar test: do two models make different mistakes more often than chance?"""
    true = np.asarray(y_true, dtype=bool)
    a_right = np.asarray(pred_a, dtype=bool) == true
    b_right = np.asarray(pred_b, dtype=bool) == true
    a_only = int((a_right & ~b_right).sum())          # only model A is right
    b_only = int((~a_right & b_right).sum())          # only model B is right
    disagreements = a_only + b_only
    p_value = 1.0 if disagreements == 0 else float(binomtest(min(a_only, b_only), disagreements, 0.5).pvalue)
    return {"a_only_correct": a_only, "b_only_correct": b_only,
            "both_correct": int((a_right & b_right).sum()),
            "both_wrong": int((~a_right & ~b_right).sum()), "p_value": p_value}
