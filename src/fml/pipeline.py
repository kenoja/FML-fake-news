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

from . import errors  # noqa: E402
from .clean import clean_documents, website_names  # noqa: E402
from .comparison import compact_table, mcnemar_pair, model_row  # noqa: E402
from .evaluate import BOOTSTRAP_SEED, N_BOOTSTRAP, bootstrap_ci, classification_metrics, confusion  # noqa: E402
from .models import (build_pipeline, choose_c, fit_logistic, split_frames, tfidf_vectorizer, top_words,
                     tune_on_validation, word_counts_vectorizer)  # noqa: E402
from .paths import project_config  # noqa: E402
from .results import CONSPIRACY_COLOUR, MAINSTREAM_COLOUR, results_dir, savefig, write_metrics, write_table  # noqa: E402
from .data_report import plot_split_sizes  # noqa: E402
from .split import SPLITS, load_split, split_path, split_summary  # noqa: E402

M1_FEATURES = {"ngram_range": [1, 1], "min_df": 2, "sublinear_tf": True, "numeric_columns": []}
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
    frame, log, info = _clean_documents_cached()
    write_table("cleaning_log", log)
    summ = split_summary(frame, load_split())       # the split file predates cleaning; count what is left
    write_table("split_summary", summ)
    savefig(plot_split_sizes(summ), "split_sizes")
    info = {**info, "split_after_cleaning": {s: int(summ.loc[summ["split"] == s, "documents"].sum()) for s in SPLITS}}
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

def run_logistic_analysis(name, make_vectorizer, title):
    """Fit the vectorizer on train, choose C on validation, write tables, figure and metrics."""
    frames = split_frames(analysis_frame(), ("train", "validation"))
    train, valid = frames["train"], frames["validation"]
    vectorizer = make_vectorizer()
    X_train = vectorizer.fit_transform(train["txt"])        # vocabulary (and IDF): training articles only
    X_valid = vectorizer.transform(valid["txt"])
    model, grid = choose_c(X_train, train["label"], X_valid, valid["label"])
    counts = confusion(valid["label"], model.predict(X_valid))
    write_analysis_files(name, title, grid, top_words(vectorizer, model), counts)
    return build_metrics_dict(grid, counts, vectorizer, train, valid)


def write_analysis_files(name, title, grid, words, counts):
    """Write the grid, the top words and the validation confusion matrix."""
    write_table(f"{name}_grid", grid)
    write_table(f"{name}_top_words", words)
    savefig(plot_confusion(counts, f"{title} (validation)"), f"{name}_validation_confusion")


def build_metrics_dict(grid, counts, vectorizer, train, valid):
    """Collect the numbers for metrics.json, rounded to 3 digits."""
    chosen = grid[grid["chosen"]].iloc[0]
    values = {"chosen_C": float(chosen["C"]), "class_weight": project_config()["lr"]["class_weight"]}
    values["validation"] = {"precision": float(chosen["precision"]), "recall": float(chosen["recall"]),
                            "f1": float(chosen["f1"])}
    values["confusion"] = counts
    values["vocabulary_size"] = int(len(vectorizer.vocabulary_))
    values["documents"] = {"train": int(len(train)), "validation": int(len(valid))}
    values["grid"] = grid.drop(columns="chosen").to_dict("records")
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
    details: Callable[[pd.DataFrame], pd.DataFrame] | None = None       # extra prediction columns, per test row


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
    predictions_path = log_path.parent / "tables" / f"predictions_{model_id}.csv"
    if previous and predictions_path.is_file():         # same configuration, already scored: reuse, never rescore
        return _summary_from_predictions(model_id, digest, pd.read_csv(predictions_path))

    if frame is None:
        frame = analysis_frame()
    split = load_split()
    train = frame[frame["doc_id"].isin(set(split.loc[split["split"] == "train", "doc_id"]))].reset_index(drop=True)
    test = frame[frame["doc_id"].isin(set(split.loc[split["split"] == "test", "doc_id"]))].reset_index(drop=True)
    if spec.fit_predict_proba is not None:
        pred, proba = spec.fit_predict_proba(train, test)
    else:
        pred, proba = spec.fit_predict(train, test), None
    table = pd.DataFrame({"model_id": model_id, "doc_id": test["doc_id"], "label": test["label"].to_numpy(),
                          "pred": np.asarray(pred, dtype=bool)})       # ids and answers only, never text
    if proba is not None:
        table["proba"] = np.round(proba, 4)
    if spec.details is not None:
        table = pd.concat([table, spec.details(test).reset_index(drop=True)], axis=1)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(predictions_path, index=False)
    if not previous:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"model_id": model_id, "config_hash": digest,
                                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "n_docs": int(len(test))}) + "\n")
    return _summary_from_predictions(model_id, digest, table)


def _summary_from_predictions(model_id: str, digest: str, table: pd.DataFrame) -> dict:
    """Confusion counts and bootstrap intervals from a stored predictions table."""
    y, pred = table["label"].to_numpy(dtype=bool), table["pred"].to_numpy(dtype=bool)
    return {"model_id": model_id, "config_hash": digest, "n_docs": int(len(table)),
            "confusion": confusion(y, pred), "metrics": bootstrap_ci(y, pred), "predictions": table}


# ---------------------------------------------------------------- stage: tfidf_features

LR_WORDS_ID, LR_EXTENDED_ID = "lr_words_v2", "lr_extended"


def ablation_variants():
    """The five models to compare: TF-IDF alone, plus each extra group, and plus all three."""
    settings = project_config()["features"]
    ngrams = {"ngram_range": list(settings["ngram_range"]), "min_df": settings["ngram_min_df"],
              "max_features": settings["ngram_max_features"]}
    plain = dict(M1_FEATURES, dense_groups=[])
    with_ngrams = dict(plain, **ngrams)
    return {LR_WORDS_ID: variant("TF-IDF alone", plain),
            "lr_plus_ngrams": variant("+ n-grams (1-3)", with_ngrams),
            "lr_plus_epistemic": variant("+ certainty words", dict(plain, dense_groups=["epistemic"])),
            "lr_plus_style": variant("+ style", dict(plain, dense_groups=["style"])),
            LR_EXTENDED_ID: variant("+ all three", dict(with_ngrams, dense_groups=["epistemic", "style"]))}


def variant(label, features):
    """One model to compare: a name for the table and its feature settings."""
    return {"label": label, "features": features}


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


def _metrics() -> dict:
    return json.loads((results_dir() / "metrics.json").read_text(encoding="utf-8"))


def _ablation_selected() -> dict:
    sel = _metrics()["tfidf_features"].get("selected")
    if not sel:
        raise RuntimeError("run the ablation stage first: the tuned settings come from validation")
    return sel


def _final_config(model_id: str, **settings) -> dict:
    return {"model_id": model_id, **settings, "data": "filtered_documents_cleaned",
            "split": project_config()["split"]["file"]}


def _logistic_spec(model_id: str, vectorizer, metrics_key: str) -> ModelSpec:
    """Word counts or TF-IDF on all words: C and class weight were chosen on validation (metrics.json)."""
    chosen = _metrics()[metrics_key]
    config = _final_config(model_id, features=metrics_key, C=chosen["chosen_C"], class_weight=chosen["class_weight"])

    def fit_predict_proba(train: pd.DataFrame, test: pd.DataFrame):
        vec = vectorizer()
        model = fit_logistic(vec.fit_transform(train["txt"]), train["label"], config["C"], config["class_weight"])
        X_test = vec.transform(test["txt"])                 # vocabulary comes from training articles only
        return model.predict(X_test), model.predict_proba(X_test)[:, 1]

    return ModelSpec(config, lambda train, test: fit_predict_proba(train, test)[0], fit_predict_proba)


def _features_spec(model_id: str) -> ModelSpec:
    """TF-IDF plus engineered features: the feature set, C and class weight chosen on validation."""
    name = _metrics()["tfidf_features"]["chosen_lr"]["chosen"]
    sel = _ablation_selected()[name]
    cw = None if sel["class_weight"] == "none" else sel["class_weight"]
    config = _final_config(model_id, feature_set=name, features=sel["features"], C=sel["C"], class_weight=cw)

    def fit_predict_proba(train: pd.DataFrame, test: pd.DataFrame):
        pipe = build_pipeline(config["features"], C=config["C"], class_weight=cw).fit(train, train["label"])
        return pipe.predict(test), pipe.predict_proba(test)[:, 1]

    return ModelSpec(config, lambda train, test: fit_predict_proba(train, test)[0], fit_predict_proba)


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
              "protocol": "per model: C and class weight from config lr (same grid as word counts and TF-IDF), best conspiracy-F1 on validation",
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

LLM_ID = "final_llm"


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


def _llm_spec(model_id: str) -> ModelSpec:
    """The LLM with the prompt chosen on validation; the config holds the prompt hash, so a changed prompt is caught."""
    from . import llm

    variant = _metrics()["llm"]["chosen_prompt"]
    config = _final_config(model_id, model=llm.llm_config()["model"], prompt=variant,
                           prompt_hash=llm.cache_key(variant, 1), pass_id=1)

    def answers(test: pd.DataFrame) -> pd.DataFrame:
        return llm.classify(test, variant, 1)               # from the cache first, from the API only on a miss

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame):
        return (answers(test)["label"] == "conspiracy").to_numpy()      # unreadable answers count as wrong

    def details(test: pd.DataFrame) -> pd.DataFrame:
        cache = llm.read_cache(llm.cache_path())
        records = [cache[(d, config["prompt_hash"])] for d in test["doc_id"]]
        return pd.DataFrame({"llm_label": [r["label"] for r in records],
                             "confidence": [r["confidence"] for r in records],
                             "pass": 1, "variant": variant})

    return ModelSpec(config, fit_predict, details=details)


MODEL_BUILDERS.update({"final_word_counts": lambda: _logistic_spec("final_word_counts", word_counts_vectorizer, "word_counts"),
                       "final_tfidf": lambda: _logistic_spec("final_tfidf", tfidf_vectorizer, "tfidf"),
                       "final_tfidf_features": lambda: _features_spec("final_tfidf_features"),
                       "final_llm": lambda: _llm_spec("final_llm")})
FINAL_IDS = tuple(MODEL_BUILDERS)


def final_stage() -> dict:
    """One final run: freeze the four validation-chosen settings, then score each on test exactly once."""
    return {model_id: (freeze_model(model_id), evaluate_final(model_id))[1]["n_docs"] for model_id in FINAL_IDS}


# ---------------------------------------------------------------- stage: comparison

WORDS_FINAL, TFIDF_FINAL, FEATURES_FINAL = "final_word_counts", "final_tfidf", "final_tfidf_features"


def read_final_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test predictions written by evaluate_final: (logistic regression models stacked, the LLM)."""
    tables = results_dir() / "tables"
    lr = pd.concat([pd.read_csv(tables / f"predictions_{m}.csv")
                    for m in (WORDS_FINAL, TFIDF_FINAL, FEATURES_FINAL)], ignore_index=True)
    return lr, pd.read_csv(tables / "predictions_final_llm.csv")


def best_final_lr(metrics: dict) -> str:
    """Of the two TF-IDF models, the one with the higher validation F1 (a tie keeps plain TF-IDF)."""
    plain, extra = metrics["tfidf"]["validation"]["f1"], metrics["tfidf_features"]["chosen_lr"]["f1"]
    return FEATURES_FINAL if extra > plain else TFIDF_FINAL


MODEL_NAMES = {"final_word_counts": "Word counts", "final_tfidf": "TF-IDF",
               "final_tfidf_features": "TF-IDF + features", "final_llm": "LLM"}


def plot_comparison_f1(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(table))
    f1 = table["f1"].to_numpy()
    ci = table[["f1_ci_low", "f1_ci_high"]].to_numpy()
    ax.bar(x, f1, color=CONSPIRACY_COLOUR, width=0.55)
    ax.errorbar(x, f1, yerr=[f1 - ci[:, 0], ci[:, 1] - f1], fmt="none", ecolor="black", capsize=4,
                label="95% confidence interval")
    ax.set_xticks(x, table["model"].map(MODEL_NAMES))
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
        ax.set_title(MODEL_NAMES[r["model"]], fontsize=10)
    axes[0].set_ylabel("True website type")
    fig.tight_layout()
    return fig


def comparison_stage() -> dict:
    """Compare the four final models on the same test documents, from the saved prediction files only."""
    lr, llm_all = read_final_predictions()
    llm = llm_all[llm_all["pass"] == 1].set_index("doc_id")
    ids = llm.index
    y = llm["label"].to_numpy(dtype=bool)
    preds = {m: lr[lr["model_id"] == m].set_index("doc_id").loc[ids]["pred"].to_numpy(dtype=bool)
             for m in (WORDS_FINAL, TFIDF_FINAL, FEATURES_FINAL)}           # same documents, same order
    preds[LLM_ID] = llm["pred"].to_numpy(dtype=bool)

    table = pd.DataFrame([model_row(name, y, pred) for name, pred in preds.items()])
    write_table("comparison_test", table)
    write_table("comparison_compact", compact_table(table, MODEL_NAMES))
    savefig(plot_comparison_f1(table), "comparison_f1")
    savefig(plot_comparison_confusion(table), "comparison_confusion")

    metrics = json.loads((results_dir() / "metrics.json").read_text(encoding="utf-8"))
    best_lr = best_final_lr(metrics)
    tests = {"words_vs_tfidf": mcnemar_pair(y, preds[WORDS_FINAL], preds[TFIDF_FINAL], WORDS_FINAL, TFIDF_FINAL),
             "tfidf_vs_features": mcnemar_pair(y, preds[TFIDF_FINAL], preds[FEATURES_FINAL], TFIDF_FINAL, FEATURES_FINAL),
             "best_lr_vs_llm": mcnemar_pair(y, preds[best_lr], preds[LLM_ID], best_lr, LLM_ID)}
    values = {"models": list(preds), "lr_best": best_lr, "test_documents": int(len(y)), "test_conspiracy": int(y.sum()),
              "bootstrap": {"resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "level": 0.95},
              "table": table.to_dict("records"), "mcnemar": tests, "llm_stability": metrics["llm"]["repeat_agreement"]}
    write_metrics("comparison", round_floats(values, 4))
    return values


# ---------------------------------------------------------------- stage: errors

def errors_stage() -> dict:
    """Test articles where the best LR and the LLM disagree -> stance coding (cached) -> tables."""
    cfg = errors.stance_config()
    seed = project_config()["split"]["seed"]
    lr_all, llm_p = read_final_predictions()
    chosen = json.loads((results_dir() / "metrics.json").read_text(encoding="utf-8"))["comparison"]["lr_best"]
    lr = lr_all[lr_all["model_id"] == chosen]
    llm_p = llm_p[llm_p["pass"] == 1]
    sets = errors.disagreement_sets(lr, llm_p)
    frame = analysis_frame().set_index("doc_id")
    sets["website"] = sets["doc_id"].map(frame["website"])
    to_code = errors.select_for_coding(sets, cfg["max_per_other_set"], seed)
    client = None
    if errors.missing_stances(to_code["doc_id"]):
        client = errors.stance_client(cfg)          # raises loudly: no key and no complete cache
    stance = errors.code_stances(to_code.assign(txt=to_code["doc_id"].map(frame["txt"])), client=client)
    coded = to_code.merge(stance, on="doc_id")

    unreadable = int((coded["claude_stance"] == errors.UNREADABLE).sum())
    coded = coded[coded["claude_stance"] != errors.UNREADABLE].reset_index(drop=True)   # footnote: these are dropped
    coded["verdict"] = [errors.classify_error(bool(t), s) for t, s in zip(coded["true_label"], coded["claude_stance"])]
    usable = coded
    write_table("error_classification", errors.error_classification_table(coded, sets))
    sample = errors.stratified_sample(usable, cfg["human_sample_size"], seed)
    written = errors.write_check_sample(sample, results_dir() / "tables" / "stance_check_sample.csv")

    errors.write_claude_answers(sample, results_dir() / "tables" / "stance_claude_answers.csv")

    both_right = lr[["doc_id", "label", "pred"]].merge(llm_p[["doc_id", "pred"]], on="doc_id", suffixes=("_lr", "_llm"))
    both_right = both_right[(both_right["pred_lr"] == both_right["label"]) & (both_right["pred_llm"] == both_right["label"])]
    ex = errors.pick_examples(sets, both_right.rename(columns={"label": "true_label"}))
    ex["website"] = ex["doc_id"].map(frame["website"])
    ex["excerpt"] = [" ".join(frame.loc[d, "txt"].split()[:errors.EXCERPT_WORDS]) + " ..." for d in ex["doc_id"]]
    ex["true_label"] = ex["true_label"].map(CLASS_NAMES)
    write_table("error_examples", ex[["cell", "doc_id", "website", "true_label", "excerpt"]])

    from . import llm as llm_mod
    split = load_split().set_index("doc_id")["split"]
    val = frame.reset_index()
    val = val[val["doc_id"].map(split) == "validation"]
    try:
        probe = llm_mod.website_probe(llm_mod.probe_sample(val, seed))
    except RuntimeError as exc:             # no key and no cached probe: say so, never invent a share
        probe = {"status": "not run", "reason": str(exc)}
    totals = sets.groupby(["set", "error_type"]).size()
    values = {"data": "TEST rows, filtered documents; predictions from evaluate_final files only",
              "prior_knowledge_probe": probe,
              "lr_model": chosen, "stance_model": cfg["model"],
              "counts": {s: {"total": int((sets["set"] == s).sum()),
                             "conspiracy": int(((sets["set"] == s) & sets["true_label"]).sum()),
                             "mainstream": int(((sets["set"] == s) & ~sets["true_label"]).sum()),
                             "false_positive": int(totals.get((s, "false positive"), 0)),
                             "false_negative": int(totals.get((s, "false negative"), 0)),
                             "coded": int((coded["set"] == s).sum())} for s in errors.SETS},
              "coded_documents": int(len(coded)),
              "footnote_unreadable_codes": unreadable,
              "stance_counts": usable["claude_stance"].value_counts().to_dict(),
              "verdict_counts": usable["verdict"].value_counts().to_dict(),
              "label_problem_share": {"label_problems": int((coded["verdict"] == errors.LABEL_PROBLEM).sum()),
                                      "of_coded": int(len(coded)),
                                      "source": "Claude stance coding only, not checked by people"},
              "rule": "conspiracy site + (rejects|neutral|no_claim) -> label problem; mainstream site + endorses -> label problem; else model wrong",
              "human_check": {"sample_file": "results/tables/stance_check_sample.csv", "rows": int(len(sample)),
                              "file_rewritten_this_run": written, "status": "to be filled by the team"}}
    write_metrics("errors", round_floats(values, 4))
    return values


STAGES: list[tuple[str, Callable[[], object]]] = [("cleaning", cleaning_stage),
                                                  ("word_counts", word_counts_stage), ("tfidf", tfidf_stage), ("features", feature_summary_stage),
                                                  ("tfidf_features", ablation), ("llm", llm_stage),
                                                  ("final", final_stage), ("comparison", comparison_stage),
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
