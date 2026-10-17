"""The two entry points the CLI calls, plus the stages they are made of.

Stages are added one by one over the cycle. Until every stage exists `run_all` raises after running
the finished ones, so `fml run-all` can never look successful while results are missing
(AGENTS.md: fail loud).
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from . import errors  # noqa: E402
from .clean import clean_documents, website_names  # noqa: E402
from .evaluate import (BOOTSTRAP_SEED, N_BOOTSTRAP, bootstrap_ci, classification_metrics, confusion,  # noqa: E402
                       majority_baseline, mcnemar_exact, METRIC_NAMES, paired_delta_ci, pr_curve_data,
                       stratified_random_baseline)
from .models import (build_pipeline, choose_c, split_frames, tfidf_vectorizer, top_words,
                     tune_on_validation, word_counts_vectorizer)  # noqa: E402
from .paths import project_config  # noqa: E402
from .results import CONSPIRACY_COLOUR, MAINSTREAM_COLOUR, results_dir, savefig, write_metrics, write_table  # noqa: E402
from .split import load_split, split_path  # noqa: E402

M1_FEATURES = {"ngram_range": [1, 1], "min_df": 2, "sublinear_tf": True, "numeric_columns": []}
LENGTH_BUCKETS = (150, 250)
CLASS_NAMES = {True: "conspiracy", False: "mainstream"}
COLOURS = {"conspiracy": CONSPIRACY_COLOUR, "mainstream": MAINSTREAM_COLOUR}


# ---------------------------------------------------------------- stage: cleaning

@lru_cache(maxsize=1)
def _clean_documents_cached():
    """Cleaning takes ~20 s and every stage needs it; do it once per run (callers get copies)."""
    return clean_documents()


def analysis_frame() -> pd.DataFrame:
    """THE frame every model uses: on-topic, near-duplicates removed, text cleaned."""
    return _clean_documents_cached()[0].copy()


def plot_cleaning(log: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(log))
    for k, c in enumerate(("mainstream", "conspiracy")):
        vals = log[f"remaining_{c}"].tolist()
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=COLOURS[c], label=c)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, ["all documents", "on topic", "no near-duplicates"])
    ax.set_ylabel("Documents left")
    ax.set_title("Documents before and after each cleaning step")
    ax.legend()
    return fig


def round_floats(obj, digits: int = 3):
    """Round every float in a nested dict/list, so numbers in metrics.json read as quoted in the prose."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, digits) for v in obj]
    return obj


def cleaning_stage() -> dict:
    """Write the cleaning log, the before/after figure and metrics.json `cleaning`."""
    _, log, info = _clean_documents_cached()
    write_table("cleaning_log", log)
    savefig(plot_cleaning(log), "cleaning_before_after")
    write_metrics("cleaning", round_floats(info, 4))
    return info


def plot_confusion(c: dict[str, int], title: str):
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(mat, cmap="Greens")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > mat.max() / 2 else "black")
    ax.set_xticks([0, 1], ["mainstream", "conspiracy"])
    ax.set_yticks([0, 1], ["mainstream", "conspiracy"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True website type")
    ax.set_title(title)
    return fig


def website_name_tokens(df: pd.DataFrame, min_len: int = 4) -> set[str]:
    """Word pieces of all website names ("sitename" from "sitename.com"), to spot giveaways."""
    import re

    return {t for n in website_names(df) for t in re.findall(r"[a-z0-9]+", n) if len(t) >= min_len} - {"com", "www", "html"}


# ---------------------------------------------------------------- stages: word counts and TF-IDF

def run_logistic_analysis(name: str, make_vectorizer, title: str) -> dict:
    """Fit the vectorizer on train, choose C on validation, write tables, figure and metrics."""
    frames = split_frames(analysis_frame(), ("train", "validation"))
    train, valid = frames["train"], frames["validation"]
    vectorizer = make_vectorizer()
    X_train = vectorizer.fit_transform(train["txt"])        # vocabulary (and IDF): training articles only
    X_valid = vectorizer.transform(valid["txt"])
    model, grid = choose_c(X_train, train["label"], X_valid, valid["label"])
    write_table(f"{name}_grid", grid)
    write_table(f"{name}_top_words", top_words(vectorizer, model))
    counts = confusion(valid["label"], model.predict(X_valid))
    savefig(plot_confusion(counts, f"{title} (validation)"), f"{name}_validation_confusion")
    chosen = grid[grid["chosen"]].iloc[0]
    values = {"chosen_C": float(chosen["C"]), "class_weight": project_config()["lr"]["class_weight"],
              "validation": {"precision": float(chosen["precision"]), "recall": float(chosen["recall"]),
                             "f1": float(chosen["f1"])},
              "confusion": counts, "vocabulary_size": int(len(vectorizer.vocabulary_)),
              "documents": {"train": int(len(train)), "validation": int(len(valid))},
              "grid": grid.drop(columns="chosen").to_dict("records")}
    return round_floats(values)


def word_counts_stage() -> dict:
    """Analysis 1: logistic regression on word counts (validation only)."""
    values = run_logistic_analysis("word_counts", word_counts_vectorizer, "Word counts")
    write_metrics("word_counts", values)
    return values


def tfidf_stage() -> dict:
    """Analysis 2: logistic regression on TF-IDF weights (validation only)."""
    values = run_logistic_analysis("tfidf", tfidf_vectorizer, "TF-IDF")
    write_metrics("tfidf", values)
    return values


# ---------------------------------------------------------------- final test scoring

@dataclass
class ModelSpec:
    """Everything that defines a model: `config` is hashed, `fit_predict` trains on train, predicts test."""
    config: dict
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]
    fit_predict_proba: Callable[[pd.DataFrame, pd.DataFrame], tuple[np.ndarray, np.ndarray]] | None = None


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


MODEL_BUILDERS: dict[str, Callable[[], ModelSpec]] = {}


def read_access_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_frozen(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_model(model_id: str, builders: dict[str, Callable[[], ModelSpec]] | None = None,
                 frozen_path: Path | None = None, log_path: Path | None = None) -> dict:
    """Write the configuration chosen on validation to results/frozen_models.json BEFORE any test look.

    Why: `evaluate_final` refuses to score a model whose chosen config is not frozen, or whose hash
    differs from the frozen one, so the test split cannot influence which configuration is reported.
    An entry is never rewritten. If the access log already holds a look for this model (a freeze made
    after the fact), the entry says `retroactive: true` instead of pretending it came first.
    """
    builders = MODEL_BUILDERS if builders is None else builders
    digest = config_hash(builders[model_id]().config)
    frozen_path = frozen_path or results_dir() / "frozen_models.json"
    log_path = log_path or results_dir() / "test_access_log.jsonl"
    frozen = read_frozen(frozen_path)
    if model_id in frozen:
        if frozen[model_id]["config_hash"] != digest:
            raise PermissionError(f"{model_id} is frozen with another configuration; use a new model id.")
        return frozen[model_id]
    entry = {"model_id": model_id, "config_hash": digest, "selection_source": "validation",
             "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "retroactive": any(e["model_id"] == model_id for e in read_access_log(log_path))}
    frozen[model_id] = entry
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return entry


def evaluate_final(model_id: str, builders: dict[str, Callable[[], ModelSpec]] | None = None,
                   frame: pd.DataFrame | None = None, log_path: Path | None = None,
                   frozen_path: Path | None = None) -> dict:
    """Score one model on the locked TEST split and log the access in results/test_access_log.jsonl.

    The only code path that reads the test rows. A model id may be scored with one configuration only:
    a second, different config hash is refused BEFORE anything is scored (otherwise the test set could
    be used to pick between settings). Re-running the identical configuration is allowed and does not
    add a second log line.
    """
    builders = MODEL_BUILDERS if builders is None else builders
    if model_id not in builders:
        raise KeyError(f"unknown model_id {model_id!r}; known: {sorted(builders)}")
    spec = builders[model_id]()
    digest = config_hash(spec.config)
    log_path = log_path or results_dir() / "test_access_log.jsonl"
    frozen = read_frozen(frozen_path or results_dir() / "frozen_models.json").get(model_id)
    if frozen is None:
        raise PermissionError(f"{model_id} has no entry in frozen_models.json: freeze the configuration "
                              "chosen on validation (freeze_model) before the first test scoring.")
    if frozen["config_hash"] != digest:
        raise PermissionError(f"{model_id}: configuration hash {digest} differs from the frozen "
                              f"{frozen['config_hash']}; refusing to score on test.")
    previous = [e for e in read_access_log(log_path) if e["model_id"] == model_id]
    if any(e["config_hash"] != digest for e in previous):
        raise PermissionError(f"{model_id} was already scored on test with a different configuration; "
                              "a new configuration needs a new model id (and is a new test access).")

    if frame is None:
        frame = analysis_frame()
    split = load_split()
    train = frame[frame["doc_id"].isin(set(split.loc[split["split"] == "train", "doc_id"]))].reset_index(drop=True)
    test = frame[frame["doc_id"].isin(set(split.loc[split["split"] == "test", "doc_id"]))].reset_index(drop=True)
    if spec.fit_predict_proba is not None:
        pred, proba = spec.fit_predict_proba(train, test)
    else:
        pred, proba = spec.fit_predict(train, test), None
    y = test["label"].to_numpy()
    out = {"model_id": model_id, "config_hash": digest, "n_docs": int(len(test)),
           "confusion": confusion(y, pred),
           "metrics": bootstrap_ci(y, pred),
           "test_websites": int(test["website"].nunique()), "test_conspiracy": int(y.sum())}
    if proba is not None:       # per-document predictions: ids, label, prediction, probability; never text
        out["predictions"] = pd.DataFrame({"model_id": model_id, "doc_id": test["doc_id"], "label": y,
                                           "pred": np.asarray(pred, dtype=bool), "proba": proba})
    if not previous:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"model_id": model_id, "config_hash": digest,
                                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "n_docs": out["n_docs"]}) + "\n")
    return out


# ---------------------------------------------------------------- stage: tfidf_features

LR_WORDS_ID, LR_EXTENDED_ID = "lr_words_v2", "lr_extended"


def ablation_variants() -> dict[str, dict]:
    """The five models to compare: TF-IDF alone, plus each extra group, and plus all three."""
    ng = project_config()["features"]
    ngrams = {"ngram_range": list(ng["ngram_range"]), "min_df": ng["ngram_min_df"], "max_features": ng["ngram_max_features"]}
    words = dict(M1_FEATURES)
    models = {LR_WORDS_ID: ("TF-IDF alone", words | {"dense_groups": []}),
              "lr_plus_ngrams": ("+ n-grams (1-3)", words | ngrams | {"dense_groups": []}),
              "lr_plus_epistemic": ("+ certainty words", words | {"dense_groups": ["epistemic"]}),
              "lr_plus_style": ("+ style", words | {"dense_groups": ["style"]}),
              LR_EXTENDED_ID: ("+ all three", words | ngrams | {"dense_groups": ["epistemic", "style"]})}
    return {model_id: {"label": label, "features": features} for model_id, (label, features) in models.items()}


def plot_ablation(table: pd.DataFrame, chosen: str):
    """Horizontal bars of validation F1, one per model; the chosen model is highlighted."""
    rows = table.iloc[::-1].reset_index(drop=True)          # first model on top
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colours = ["#3b5b8c" if m == chosen else "#9a9a9a" for m in rows["model_id"]]
    ax.barh(rows.index, rows["f1"], color=colours, height=0.6)
    for i, f1 in enumerate(rows["f1"]):
        ax.text(f1, i, f" {f1:.3f}", va="center", fontsize=9)
    ax.set_yticks(rows.index, rows["label"])
    ax.set_xlim(max(0.0, float(rows["f1"].min()) - 0.05), 1.0)
    ax.set_xlabel("Validation F1 (conspiracy class)")
    ax.set_title("Do the extra features help? (blue = chosen model)", fontsize=10)
    fig.tight_layout()
    return fig


def _ablation_selected() -> dict:
    sel = json.loads((results_dir() / "metrics.json").read_text(encoding="utf-8"))["tfidf_features"].get("selected")
    if not sel:
        raise RuntimeError("run the ablation stage first: the tuned settings come from validation")
    return sel


def _lr_spec(model_id: str) -> ModelSpec:
    """Test-time spec of an ablation model: settings (features, C, class weight) fixed by validation tuning."""
    sel = _ablation_selected()[model_id]
    cw = None if sel["class_weight"] == "none" else sel["class_weight"]
    config = {"model_id": model_id, "features": sel["features"], "C": sel["C"], "class_weight": cw,
              "data": "filtered_documents_cleaned", "split": project_config()["split"]["file"]}

    def fit(train: pd.DataFrame):
        return build_pipeline(config["features"], C=config["C"], class_weight=cw).fit(train, train["label"])

    def fit_predict_proba(train: pd.DataFrame, test: pd.DataFrame):
        pipe = fit(train)
        return pipe.predict(test), pipe.predict_proba(test)[:, 1]

    return ModelSpec(config, lambda train, test: fit(train).predict(test), fit_predict_proba)


MODEL_BUILDERS.update({m: (lambda m=m: _lr_spec(m)) for m in (LR_WORDS_ID, LR_EXTENDED_ID)})


def ablation() -> dict:
    """Compare the five feature sets on validation and write the table, figure and metrics."""
    from .ablation import choose_best, run_ablation, verdict_sentence
    from .features.epistemic import lexicon_coverage

    frames = split_frames(analysis_frame(), ("train", "validation"))
    train, valid = frames["train"], frames["validation"]
    variants = ablation_variants()
    table, _ = run_ablation(train, valid, variants, reference=LR_WORDS_ID)
    chosen = choose_best(table, LR_WORDS_ID)
    write_table("ablation", table.round(4))
    savefig(plot_ablation(table, chosen["chosen"]), "ablation_f1")

    coverage = lexicon_coverage(train).round(4)             # how much of the data the word list reaches
    write_table("epistemic_coverage", coverage)

    selected = {row["model_id"]: {"features": variants[row["model_id"]]["features"], "C": row["C"],
                                  "class_weight": row["class_weight"]} for row in table.to_dict("records")}
    values = {"data": "filtered documents, split_v1, validation rows only",
              "protocol": "per model: C in C_GRID x class_weight in (none, balanced), best conspiracy-F1 on validation",
              "documents": {"train": int(len(train)), "validation": int(len(valid))},
              "selected": selected, "models": {row["model_id"]: row for row in table.to_dict("records")},
              "chosen_lr": chosen, "verdict_sentence": verdict_sentence(table, LR_WORDS_ID),
              "coverage": coverage.to_dict("records")}
    write_metrics("tfidf_features", round_floats(values, 4))
    return values


# ---------------------------------------------------------------- entry points

def feature_summary_stage() -> dict:
    """Feature table and style figure on training rows (imported late: features.summary imports this module)."""
    from .features.summary import features_stage

    return features_stage()


# ---------------------------------------------------------------- stage: llm

LLM_ID = "llm_haiku"


def llm_scores(valid: pd.DataFrame, variant: str, pass_id: int = 1) -> dict:
    """Ask the LLM about every validation article and score its answers against the dataset labels."""
    from . import llm

    answers = llm.classify(valid, variant, pass_id)
    said_conspiracy = (answers["label"] == "conspiracy").to_numpy()      # unreadable answers count as wrong
    scores = classification_metrics(valid["label"], said_conspiracy)
    return {"labels": answers["label"].to_numpy(), "confusion": confusion(valid["label"], said_conspiracy),
            "unreadable": int((answers["label"] == llm.UNREADABLE).sum()),
            "truncated_docs": int(answers["truncated"].sum()),
            **{name: scores[name] for name in ("precision", "recall", "f1")}}


def llm_stage() -> dict:
    """Analysis 4: try each prompt on validation, keep the best by F1, repeat it once to check stability."""
    from . import llm

    cfg = llm.llm_config()
    valid = split_frames(analysis_frame(), ("validation",))["validation"]
    variants = llm.check_variants(list(llm.PROMPTS))
    runs = {variant: llm_scores(valid, variant) for variant in variants}      # prompts are tried on validation only
    table = pd.DataFrame([{"variant": v, "documents": len(valid), "precision": r["precision"],
                           "recall": r["recall"], "f1": r["f1"]} for v, r in runs.items()])
    write_table("llm_prompt_variants", table.round(4))
    chosen = max(variants, key=lambda v: round(runs[v]["f1"], 4))             # best F1 wins; a tie keeps the first prompt
    first, second = runs[chosen], llm_scores(valid, chosen, pass_id=2)
    repeat_agreement = float((first["labels"] == second["labels"]).mean())     # share of equal answers
    savefig(plot_confusion(first["confusion"], f"LLM, prompt {chosen} (validation)"), "llm_validation_confusion")
    keys = ("precision", "recall", "f1", "confusion")
    values = {"model": cfg["model"], "question": "source: conspiracy-oriented or mainstream website",
              "max_input_words": cfg["max_input_words"], "max_output_tokens": cfg["max_output_tokens"],
              "documents": int(len(valid)),
              "variants": {v: {k: r[k] for k in keys} for v, r in runs.items()},
              "chosen_prompt": chosen, "validation": {k: first[k] for k in keys},
              "unreadable": first["unreadable"], "truncated_docs": first["truncated_docs"],
              "repeat_agreement": repeat_agreement}
    write_metrics("llm", round_floats(values, 4))
    return values


# ---------------------------------------------------------------- stage: comparison

def _score_from_llm(p1: pd.DataFrame) -> np.ndarray:
    """LLM confidence as a conspiracy score: confidence if it said conspiracy, 1 - confidence otherwise."""
    conf = p1["confidence"].astype(float).fillna(0.0).to_numpy()
    return np.where(p1["llm_label"].to_numpy() == "conspiracy", conf, 1 - conf)


def _nan_to_none_deep(obj):
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none_deep(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none_deep(v) for v in obj]
    return obj


def plot_comparison_f1(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(table))
    f1 = table["f1"].to_numpy()
    ci = table[["f1_ci_low", "f1_ci_high"]].to_numpy()
    ax.bar(x, f1, color=CONSPIRACY_COLOUR, width=0.55)
    ax.errorbar(x, f1, yerr=[f1 - ci[:, 0], ci[:, 1] - f1], fmt="none", ecolor="black", capsize=4,
                label="95% confidence interval")
    ax.set_xticks(x, table["model"])
    ax.set_ylabel("Conspiracy-class F1 (test)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Test F1 with 95% confidence interval")
    return fig


def plot_comparison_confusion(table: pd.DataFrame):
    fig, axes = plt.subplots(1, len(table), figsize=(3.4 * len(table), 3.6))
    for ax, (_, r) in zip(axes, table.iterrows()):
        mat = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(mat, cmap="Greens")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                    color="white" if v > mat.max() / 2 else "black")
        ax.set_xticks([0, 1], ["main.", "consp."])
        ax.set_yticks([0, 1], ["main.", "consp."])
        ax.set_xlabel("Predicted")
        ax.set_title(r["model"], fontsize=10)
    axes[0].set_ylabel("True website type")
    fig.tight_layout()
    return fig


def plot_comparison_pr(curves: dict[str, tuple[dict, float]], base_rate: float):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for (name, (c, ap)), colour in zip(curves.items(), (CONSPIRACY_COLOUR, MAINSTREAM_COLOUR)):
        ax.plot(c["recall"], c["precision"], color=colour, label=f"{name} (AP {ap:.2f})")
    ax.axhline(base_rate, color="grey", linestyle="--", label=f"always conspiracy ({base_rate:.2f})")
    ax.set_xlabel("Recall (conspiracy)")
    ax.set_ylabel("Precision (conspiracy)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Test: score threshold behaviour")
    ax.legend(loc="lower left", fontsize=8)
    return fig


def comparison_stage() -> dict:
    """Compare majority, lr_words, lr_extended and llm_haiku (pass 1) on the same test documents.

    Reads only the prediction files written by `evaluate_final` (plus website and length metadata of
    the same documents). No model is fitted or scored here, so this stage cannot select anything on test.
    """
    from .comparison import length_bucket_f1, model_row, pairwise

    frame = analysis_frame()
    split = load_split().set_index("doc_id")["split"]
    train_y = frame.loc[frame["doc_id"].map(split) == "train", "label"].to_numpy()
    lr = pd.read_csv(results_dir() / "tables" / "predictions_lr.csv")
    llm_all = pd.read_csv(results_dir() / "tables" / "predictions_llm.csv")
    p1 = llm_all[llm_all["pass"] == 1].set_index("doc_id")
    ids = p1.index
    for m in (LR_WORDS_ID, LR_EXTENDED_ID):
        if set(lr.loc[lr["model_id"] == m, "doc_id"]) != set(ids):
            raise RuntimeError(f"{m} and the LLM were not scored on the same test documents")
    meta = frame.set_index("doc_id").loc[ids]
    y = p1["label"].to_numpy(dtype=bool)
    if not (meta["label"].to_numpy(dtype=bool) == y).all() or (split.loc[ids] != "test").any():
        raise RuntimeError("prediction files disagree with split_v1 / dataset labels")
    words = meta["txt"].str.split().map(len).to_numpy()

    metrics = json.loads((results_dir() / "metrics.json").read_text(encoding="utf-8"))
    chosen_lr = metrics["tfidf_features"]["chosen_lr"]["chosen"]
    lr_by = {m: lr[lr["model_id"] == m].set_index("doc_id").loc[ids] for m in (LR_WORDS_ID, LR_EXTENDED_ID)}
    majority = bool(np.mean(train_y) >= 0.5)
    preds = {"majority": np.full(len(y), majority), LR_WORDS_ID: lr_by[LR_WORDS_ID]["pred"].to_numpy(dtype=bool),
             LR_EXTENDED_ID: lr_by[LR_EXTENDED_ID]["pred"].to_numpy(dtype=bool), LLM_ID: p1["pred"].to_numpy(dtype=bool)}
    table = pd.DataFrame([model_row(n, y, p) for n, p in preds.items()])
    write_table("comparison_test", table.round(4))
    savefig(plot_comparison_f1(table), "comparison_f1")
    savefig(plot_comparison_confusion(table), "comparison_confusion")

    scores = {chosen_lr: lr_by[chosen_lr]["proba"].to_numpy(), LLM_ID: _score_from_llm(p1)}
    curves = {n: (pr_curve_data(y, s), float(average_precision_score(y, s))) for n, s in scores.items()}
    savefig(plot_comparison_pr(curves, float(y.mean())), "comparison_pr")

    model_preds = {k: v for k, v in preds.items() if k != "majority"}
    lengths = length_bucket_f1(words, y, model_preds, LENGTH_BUCKETS)
    write_table("comparison_length_buckets", lengths.round(4))

    tests = {"lr_best_vs_llm": pairwise(y, preds[chosen_lr], preds[LLM_ID], chosen_lr, LLM_ID),
             "lr_words_vs_lr_extended": pairwise(y, preds[LR_WORDS_ID], preds[LR_EXTENDED_ID],
                                                 LR_WORDS_ID, LR_EXTENDED_ID)}
    f1_of = {r["model"]: r for r in table.to_dict("records")}
    h1 = {}
    for m in (chosen_lr, LLM_ID):
        h1[m] = {"f1": f1_of[m]["f1"], "majority_f1": f1_of["majority"]["f1"],
                 "delta_f1_vs_majority": paired_delta_ci(y, preds["majority"], preds[m]),
                 "beats_majority": bool(f1_of[m]["f1"] > f1_of["majority"]["f1"]),
                 "ci_excludes_majority": bool(f1_of[m]["f1_ci_low"] > f1_of["majority"]["f1"])}
    pair = (chosen_lr, LLM_ID)
    wins = {k: max(pair, key=lambda m: f1_of[m][k]) for k in ("precision", "recall", "f1")}
    values = {"data": "filtered documents, split_v1, TEST rows; predictions from evaluate_final files only",
              "models": list(preds), "lr_best": chosen_lr,
              "lr_best_rule": "chosen on validation (tfidf_features.chosen_lr)", "llm_pass_scored": 1,
              "bootstrap": {"resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "level": 0.95},
              "test_documents": int(len(y)), "test_conspiracy": int(y.sum()),
              "table": table.to_dict("records"), "mcnemar": tests, "h1": h1, "winner_by_metric": wins,
              "mainstream_accuracy": {m: float((preds[m][~y] == y[~y]).mean()) for m in pair},
              "average_precision": {n: c[1] for n, c in curves.items()},
              "length_buckets": lengths.to_dict("records"), "llm_stability": metrics["llm"]["repeat_agreement"]}
    write_metrics("comparison", round_floats(_nan_to_none_deep(values), 4))
    return values


# ---------------------------------------------------------------- stage: errors

def _example_candidates(coded: pd.DataFrame) -> pd.DataFrame:
    """Worked-example documents, chosen by rule (not by hand) so a rerun picks the same ones:
    a mainstream false positive that debunks, a conspiracy site without a claim, an endorsing conspiracy
    article both models missed, a mainstream article that endorses (label problem), plus one LR-only and
    one LLM-only error. First match by (set order, doc_id)."""
    wants = [("debunking mainstream false positive", lambda d: (~d["true_label"]) & (d["claude_stance"] == "rejects_or_debunks")),
             ("conspiracy site without a claim", lambda d: d["true_label"] & (d["claude_stance"] == "no_claim")),
             ("endorsing conspiracy article missed", lambda d: d["true_label"] & (d["claude_stance"] == "endorses")),
             ("mainstream site endorsing a claim", lambda d: (~d["true_label"]) & (d["claude_stance"] == "endorses")),
             ("LR-only error", lambda d: d["set"] == "lr_only_wrong"),
             ("LLM-only error", lambda d: d["set"] == "llm_only_wrong")]
    rows, used = [], set()
    for kind, rule in wants:
        d = coded[rule(coded) & ~coded["doc_id"].isin(used)]
        d = d.assign(_o=d["set"].map(errors.SETS.index)).sort_values(["_o", "doc_id"]).drop(columns="_o")
        if len(d):
            rows.append({**d.iloc[0].to_dict(), "kind": kind})
            used.add(d.iloc[0]["doc_id"])
    return pd.DataFrame(rows)


def errors_stage() -> dict:
    """Disagreement sets on TEST predictions -> stance coding (Sonnet, cached) -> verdicts -> tables.

    Reads only the prediction files written by `evaluate_final`; the LR is refitted on TRAIN only, solely to
    explain individual predictions (coef * tf-idf). Nothing is scored or selected on test.
    """
    cfg = errors.stance_config()
    seed = project_config()["split"]["seed"]
    lr_all = pd.read_csv(results_dir() / "tables" / "predictions_lr.csv")
    chosen = json.loads((results_dir() / "metrics.json").read_text(encoding="utf-8"))["comparison"]["lr_best"]
    lr = lr_all[lr_all["model_id"] == chosen]
    llm_p = pd.read_csv(results_dir() / "tables" / "predictions_llm.csv")
    llm_p = llm_p[llm_p["pass"] == 1]
    sets = errors.disagreement_sets(lr, llm_p)
    frame = analysis_frame().set_index("doc_id")
    sets["website"] = sets["doc_id"].map(frame["website"])
    to_code = errors.select_for_coding(sets, cfg["max_per_other_set"], seed)
    missing = errors.missing_stances(to_code["doc_id"])
    client = None
    if missing:
        client = errors.stance_client(cfg)          # raises loudly: no key and no complete cache
    stance = errors.code_stances(to_code.assign(txt=to_code["doc_id"].map(frame["txt"])), client=client)
    coded = to_code.merge(stance, on="doc_id")
    coded["verdict"] = [errors.classify_error(bool(t), s) for t, s in zip(coded["true_label"], coded["claude_stance"])]
    table = errors.error_classification_table(coded, sets)
    write_table("error_classification", table)

    usable = coded[coded["claude_stance"] != errors.UNREADABLE]
    sample = errors.stratified_sample(usable, cfg["human_sample_size"], seed)
    sample_path = results_dir() / "tables" / "stance_check_sample.csv"
    written = errors.write_check_sample(sample, sample_path)

    sel = _ablation_selected()[chosen]
    split = load_split().set_index("doc_id")["split"]
    f = frame.reset_index()
    train = f[f["doc_id"].map(split) == "train"]
    pipe = build_pipeline(sel["features"], C=sel["C"], class_weight=None if sel["class_weight"] == "none" else sel["class_weight"]
                          ).fit(train, train["label"])
    ex = _example_candidates(coded)
    contrib = errors.top_contributions(pipe, f.set_index("doc_id").loc[ex["doc_id"]].reset_index()) if len(ex) else []
    reasons = llm_p.set_index("doc_id")
    from . import llm as llm_mod
    cache = llm_mod.read_cache(llm_mod.cache_path())
    key = llm_mod.cache_key(reasons["variant"].iloc[0], 1)
    ex = pd.concat([ex.reset_index(drop=True), pd.DataFrame(contrib)], axis=1)
    ex["llm_label"] = ex["doc_id"].map(reasons["llm_label"])
    ex["llm_reason"] = [errors.scrub_justification(cache[(d, key)]["reason"]) for d in ex["doc_id"]]
    ex["true_label"] = ex["true_label"].map(CLASS_NAMES)
    write_table("error_examples", ex[["kind", "doc_id", "website", "set", "true_label", "error_type", "claude_stance",
                                      "claude_justification", "verdict", "llm_label", "llm_reason",
                                      "lr_top_conspiracy", "lr_top_mainstream"]])

    def share(mask_all, mask_hit):
        n = int(mask_all.sum())
        return {"n": n, "hit": int(mask_hit.sum()), "share": float(mask_hit.sum() / n) if n else None}

    fp, fn = ~usable["true_label"], usable["true_label"]
    nonendorse = usable["claude_stance"] != "endorses"
    h6 = {"fp_mainstream_not_endorsing_but_discusses": share(
              fp, fp & usable["claude_stance"].isin(["rejects_or_debunks", "reports_neutrally"])),
          "fp_mainstream_debunking": share(fp, fp & (usable["claude_stance"] == "rejects_or_debunks")),
          "fn_conspiracy_no_claim": share(fn, fn & (usable["claude_stance"] == "no_claim")),
          "fn_conspiracy_not_endorsing": share(fn, fn & nonendorse),
          "label_problem_overall": share(usable["verdict"].notna(), usable["verdict"] == errors.LABEL_PROBLEM)}
    totals = sets.groupby(["set", "error_type"]).size()
    val = f[f["doc_id"].map(split) == "validation"]
    try:
        probe = llm_mod.website_probe(llm_mod.probe_sample(val, seed))
    except RuntimeError as exc:             # no key and no cached probe: say so, never invent a share
        probe = {"status": "not run", "reason": str(exc)}
    values = {"data": "TEST rows, filtered documents; predictions from evaluate_final files only",
              "prior_knowledge_probe": probe,
              "lr_model": chosen, "llm_pass": 1, "stance_model": cfg["model"],
              "counts": {s: {"total": int((sets["set"] == s).sum()),
                             "conspiracy": int(((sets["set"] == s) & sets["true_label"]).sum()),
                             "mainstream": int(((sets["set"] == s) & ~sets["true_label"]).sum()),
                             "false_positive": int(totals.get((s, "false positive"), 0)),
                             "false_negative": int(totals.get((s, "false negative"), 0)),
                             "coded": int((coded["set"] == s).sum())} for s in errors.SETS},
              "coded_documents": int(len(coded)), "stance_unreadable": int((coded["claude_stance"] == errors.UNREADABLE).sum()),
              "stance_counts": usable["claude_stance"].value_counts().to_dict(),
              "verdict_counts": usable["verdict"].value_counts().to_dict(),
              "rule": "conspiracy site + (rejects|neutral|no_claim) -> label problem; mainstream site + endorses -> label problem; else model wrong",
              "h6": h6, "human_check": {"sample_file": "results/tables/stance_check_sample.csv",
                                        "file_rewritten_this_run": written}}
    write_metrics("errors", round_floats(values, 4))
    return values


STAGES: list[tuple[str, Callable[[], object]]] = [("cleaning", cleaning_stage),
                                                  ("word_counts", word_counts_stage), ("tfidf", tfidf_stage), ("features", feature_summary_stage),
                                                  ("tfidf_features", ablation), ("llm", llm_stage),
                                                  ("comparison", comparison_stage),
                                                  ("errors", errors_stage)]
PENDING_STAGES: list[str] = []


def write_run_info(seconds: float) -> Path:
    """Record how long `fml run-all` took and which software versions produced results/.

    Kept apart from metrics.json (whose top-level keys are a contract) so the report can
    quote runtime and versions from a file instead of typing them.
    """
    from importlib.metadata import version

    packages = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "anthropic", "pyyaml")
    info = {"run_all_seconds": round(seconds, 1),
            "python": platform.python_version(),
            "packages": {name: version(name) for name in packages}}
    path = results_dir() / "run_info.json"
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return path


def run_all() -> None:
    """Reproduce every file in results/ from the data repo (LLM answers come from the cache)."""
    from .data_report import run_data_stage

    started = time.perf_counter()
    run_data_stage()
    for _, stage in STAGES:
        stage()
    write_run_info(time.perf_counter() - started)
    if PENDING_STAGES:
        raise NotImplementedError("fml run-all: stages not implemented yet: " + ", ".join(PENDING_STAGES))
