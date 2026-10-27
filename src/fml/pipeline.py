"""The two entry points the CLI calls (`run_all`, `evaluate_final`) and the stages they are made of."""
from __future__ import annotations

import hashlib
import json
import os
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
from .data import first_words  # noqa: E402
from .comparison import compact_table, mcnemar_pair, model_row, validation_website_table  # noqa: E402
from .evaluate import BOOTSTRAP_SEED, N_BOOTSTRAP, bootstrap_ci, classification_metrics, confusion  # noqa: E402
from .models import (feature_matrix, fit_features, fit_logistic, split_frames, tfidf_vectorizer, top_words,
                     tune_on_validation, word_counts_vectorizer)  # noqa: E402
from .methods_overview import write_methods_overview  # noqa: E402
from .paths import REPO_ROOT, project_config  # noqa: E402
from .results import CONSPIRACY_COLOUR, MAINSTREAM_COLOUR, results_dir, savefig, write_metrics, write_table  # noqa: E402
from .data_report import plot_split_sizes, site_concentration  # noqa: E402
from .split import SPLITS, load_split, split_path, split_summary, websites_per_split  # noqa: E402

M1_FEATURES = {"ngram_range": [1, 1], "min_df": 2, "sublinear_tf": True, "numeric_columns": []}
CLASS_NAMES = {True: "conspiracy", False: "mainstream"}
COLOURS = {"conspiracy": CONSPIRACY_COLOUR, "mainstream": MAINSTREAM_COLOUR}


# ---------------------------------------------------------------- stage: cleaning

@lru_cache(maxsize=1)
def _clean_documents_cached():
    """Cleaning takes ~20 s and every stage needs it; do it once per run (callers get copies)."""
    return clean_documents()


def analysis_frame() -> pd.DataFrame:
    """THE frame every model uses: on-topic, near-duplicates removed, text cut like the LLM's."""
    frame = _clean_documents_cached()[0].copy()
    max_words = project_config()["llm_classifier"]["max_input_words"]
    frame["txt"] = frame["txt"].map(lambda text: first_words(text, max_words))      # same 3,000 words as the LLM
    return frame


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


CLEANING_HEADERS = {"step": "Step", "removed": "Removed", "remaining": "Remaining", "removed_conspiracy": "Conspiracy removed",
                    "removed_mainstream": "Mainstream removed", "remaining_conspiracy": "Conspiracy remaining",
                    "remaining_mainstream": "Mainstream remaining"}
SPLIT_HEADERS = {"split": "Set", "class": "Class", "websites": "Websites", "documents": "Documents",
                 "share_of_documents": "Share of documents"}


def cleaning_stage() -> dict:
    """Write the cleaning log, the before/after figure and metrics.json `cleaning`."""
    frame, log, info = _clean_documents_cached()
    write_table("cleaning_log", log.rename(columns=CLEANING_HEADERS))
    summ = split_summary(frame, load_split())       # the split file predates cleaning; count what is left
    write_table("split_summary", summ.rename(columns=SPLIT_HEADERS))
    savefig(plot_split_sizes(summ), "split_sizes")
    split_metrics = json.loads((results_dir() / "metrics.json").read_text(encoding="utf-8"))["split"]
    write_metrics("split", {**split_metrics, "websites_per_split_after_cleaning": websites_per_split(summ)})
    info = {**info, "split_after_cleaning": {s: int(summ.loc[summ["split"] == s, "documents"].sum()) for s in SPLITS}}
    savefig(plot_cleaning(log), "cleaning_before_after")
    concentration, concentration_metrics = site_concentration(frame, load_split())
    write_table("site_concentration", concentration)
    info = {**info, **concentration_metrics}
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
    model, grid = tune_on_validation(X_train, train["label"], X_valid, valid["label"])
    counts = confusion(valid["label"], model.predict(X_valid))
    write_analysis_files(name, title, grid, top_words(vectorizer.get_feature_names_out(), model), counts)
    return build_metrics_dict(grid, counts, vectorizer, train, valid)


def write_analysis_files(name, title, grid, words, counts):
    """Write the grid, the top words and the validation confusion matrix."""
    write_table(f"{name}_grid", grid)
    for side in ("conspiracy", "mainstream"):       # one table for each side
        write_table(f"{name}_top_{side}_words", words[words["direction"] == side].drop(columns="direction"))
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


def freeze_model(model_id: str, builders: dict | None = None,
                 frozen_path: Path | None = None, log_path: Path | None = None) -> dict:
    """Record the configuration chosen on validation before any test look; an entry is never rewritten."""
    # evaluate_final refuses a model that is not frozen, so the test set cannot change the reported choice
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


def check_frozen_config(model_id: str, digest: str, frozen: dict | None) -> None:
    """Refuse to score a model whose configuration was not frozen on validation."""
    if frozen is None:
        raise PermissionError(f"{model_id} has no entry in frozen_models.json: freeze the configuration "
                              "chosen on validation (freeze_model) before the first test scoring.")
    if frozen["config_hash"] != digest:
        raise PermissionError(f"{model_id}: configuration hash {digest} differs from the frozen "
                              f"{frozen['config_hash']}; refusing to score on test.")


def score_on_test(model_id: str, spec: ModelSpec, frame: pd.DataFrame | None) -> pd.DataFrame:
    """Fit on train, predict test: one row per test article (ids and answers only, never text)."""
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
                          "pred": np.asarray(pred, dtype=bool)})
    if proba is not None:
        table["proba"] = np.round(proba, 4)
    if spec.details is not None:
        table = pd.concat([table, spec.details(test).reset_index(drop=True)], axis=1)
    return table


def log_test_access(log_path: Path, model_id: str, digest: str, n_docs: int) -> None:
    """Append one line to the test access log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"model_id": model_id, "config_hash": digest,
             "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_docs": int(n_docs)}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def earlier_test_accesses(model_id: str, digest: str, log_path: Path) -> list[dict]:
    """Earlier test scorings of this model; a different configuration is refused."""
    previous = [e for e in read_access_log(log_path) if e["model_id"] == model_id]
    if any(e["config_hash"] != digest for e in previous):
        raise PermissionError(f"{model_id} was already scored on test with a different configuration; "
                              "a new configuration needs a new model id (and is a new test access).")
    return previous


def evaluate_final(model_id: str, builders: dict | None = None,
                   frame: pd.DataFrame | None = None, log_path: Path | None = None,
                   frozen_path: Path | None = None) -> dict:
    """Score one model on the locked test split, once, and log the access."""
    builders = MODEL_BUILDERS if builders is None else builders
    if model_id not in builders:
        raise KeyError(f"unknown model_id {model_id!r}; known: {sorted(builders)}")
    spec = builders[model_id]()
    digest = config_hash(spec.config)
    log_path = log_path or results_dir() / "test_access_log.jsonl"
    # the frozen configuration is checked BEFORE anything is scored
    frozen = read_frozen(frozen_path or results_dir() / "frozen_models.json").get(model_id)
    check_frozen_config(model_id, digest, frozen)
    previous = earlier_test_accesses(model_id, digest, log_path)
    predictions_path = log_path.parent / "tables" / f"predictions_{model_id}.csv"
    if previous and predictions_path.is_file():         # same configuration, already scored: reuse, never rescore
        return _summary_from_predictions(model_id, digest, pd.read_csv(predictions_path))

    table = score_on_test(model_id, spec, frame)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(predictions_path, index=False)
    if not previous:
        log_test_access(log_path, model_id, digest, len(table))
    return _summary_from_predictions(model_id, digest, table)


def _summary_from_predictions(model_id: str, digest: str, table: pd.DataFrame) -> dict:
    """Confusion counts and bootstrap intervals from a stored predictions table."""
    y, pred = table["label"].to_numpy(dtype=bool), table["pred"].to_numpy(dtype=bool)
    return {"model_id": model_id, "config_hash": digest, "n_docs": int(len(table)),
            "confusion": confusion(y, pred), "metrics": bootstrap_ci(y, pred), "predictions": table}


# ---------------------------------------------------------------- stage: tfidf_features

LR_WORDS_ID, LR_EXTENDED_ID = "lr_words_v2", "lr_extended"


def ablation_variants():
    """The models to compare: TF-IDF alone, plus each extra group, plus all three, and certainty words plus style without n-grams."""
    settings = project_config()["features"]
    ngrams = {"ngram_range": list(settings["ngram_range"]), "min_df": settings["ngram_min_df"],
              "max_features": settings["ngram_max_features"]}
    plain = dict(M1_FEATURES, dense_groups=[])
    with_ngrams = dict(plain, **ngrams)
    return {LR_WORDS_ID: variant("TF-IDF alone", plain),
            "lr_plus_ngrams": variant("+ n-grams (1-3)", with_ngrams),
            "lr_plus_epistemic": variant("+ certainty words", dict(plain, dense_groups=["epistemic"])),
            "lr_plus_style": variant("+ style", dict(plain, dense_groups=["style"])),
            LR_EXTENDED_ID: variant("+ all three", dict(with_ngrams, dense_groups=["epistemic", "style"])),
            "lr_nongram_both": variant("+ certainty words and style (no n-grams)", dict(plain, dense_groups=["epistemic", "style"]))}


def variant(label, features):
    """One model to compare: a name for the table and its feature settings."""
    return {"label": label, "features": features}


def plot_ablation(table: pd.DataFrame):
    """Horizontal bars of validation F1, one per model; the final model (all three groups) is blue."""
    rows = table.iloc[::-1].reset_index(drop=True)          # first model on top
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colours = ["#3b5b8c" if m == LR_EXTENDED_ID else "#9a9a9a" for m in rows["model_id"]]
    ax.barh(rows.index, rows["f1"], color=colours, height=0.6)
    for i, f1 in enumerate(rows["f1"]):
        ax.text(f1, i, f" {f1:.3f}", va="center", fontsize=9)
    ax.set_yticks(rows.index, rows["label"])
    ax.set_xlim(max(0.0, float(rows["f1"].min()) - 0.05), 1.0)
    ax.set_xlabel("Validation F1 (conspiracy class)")
    ax.set_title("Do the extra features help? (blue = final model)", fontsize=10)
    fig.tight_layout()
    return fig


def _metrics() -> dict:
    return json.loads((results_dir() / "metrics.json").read_text(encoding="utf-8"))


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
    """TF-IDF plus n-grams, certainty words and style (always all three); C chosen on validation."""
    chosen = _metrics()["tfidf_features"]["validation"]
    cw = None if chosen["class_weight"] == "none" else chosen["class_weight"]
    features = ablation_variants()[LR_EXTENDED_ID]["features"]
    config = _final_config(model_id, feature_set=LR_EXTENDED_ID, features=features, C=chosen["C"], class_weight=cw)

    def fit_predict_proba(train: pd.DataFrame, test: pd.DataFrame):
        blocks = fit_features(train, config["features"])      # vocabulary and scalers: training articles only
        model = fit_logistic(feature_matrix(blocks, train), train["label"], config["C"], cw)
        X_test = feature_matrix(blocks, test)
        return model.predict(X_test), model.predict_proba(X_test)[:, 1]

    return ModelSpec(config, lambda train, test: fit_predict_proba(train, test)[0], fit_predict_proba)


def ablation() -> dict:
    """Score the feature sets on validation and write the table, figure and metrics."""
    from .ablation import run_ablation
    from .features.epistemic import LEXICON, lexicon_coverage

    frames = split_frames(analysis_frame(), ("train", "validation"))
    train, valid = frames["train"], frames["validation"]
    lexicon_sizes = {category: len(terms) for category, terms in LEXICON.items()}
    lexicon_sizes["total"] = sum(lexicon_sizes.values())      # size of the hand-made word lists
    table = run_ablation(train, valid, ablation_variants(), reference=LR_WORDS_ID)
    write_table("ablation", table.round(4))
    savefig(plot_ablation(table), "ablation_f1")

    coverage = lexicon_coverage(train).round(4)             # how much of the data the word list reaches
    write_table("epistemic_coverage", coverage)

    models = {row["model_id"]: row for row in table.to_dict("records")}
    values = {"data": "filtered documents, split_v1, validation rows only",
              "protocol": "per model: C and class weight from config lr (same grid as word counts and TF-IDF), best conspiracy-F1 on validation",
              "documents": {"train": int(len(train)), "validation": int(len(valid))},
              "final_model": LR_EXTENDED_ID, "n_columns_full": table.attrs["n_columns"][LR_EXTENDED_ID],
              "validation": models[LR_EXTENDED_ID], "models": models,
              "coverage": coverage.to_dict("records"),
              "lexicon_sizes": lexicon_sizes}
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


def choose_prompt(runs: dict, variants: list) -> str:
    """The prompt with the best validation F1; a tie keeps the earlier prompt."""
    chosen = variants[0]
    for variant in variants:
        if round(runs[variant]["f1"], 4) > round(runs[chosen]["f1"], 4):
            chosen = variant
    return chosen


def llm_values(cfg: dict, valid: pd.DataFrame, runs: dict, chosen: str, agreement: float) -> dict:
    """The numbers of analysis 4 that go into metrics.json."""
    keys = ("precision", "recall", "f1", "confusion")
    first = runs[chosen]
    return {"model": cfg["model"], "question": "source: conspiracy-oriented or mainstream website",
            "max_input_words": cfg["max_input_words"], "max_output_tokens": cfg["max_output_tokens"],
            "documents": int(len(valid)),
            "variants": {v: {k: r[k] for k in keys} for v, r in runs.items()},
            "chosen_prompt": chosen, "validation": {k: first[k] for k in keys},
            "unreadable": first["unreadable"], "truncated_docs": first["truncated_docs"],
            "repeat_agreement": agreement}


def llm_stage() -> dict:
    """Analysis 4: try each prompt on validation, keep the best by F1, repeat it once to check stability."""
    from . import llm

    cfg = llm.llm_config()
    valid = split_frames(analysis_frame(), ("validation",))["validation"]
    variants = llm.check_variants(list(llm.PROMPTS))
    runs = {variant: llm_scores(valid, variant) for variant in variants}      # prompts are tried on validation only
    table = pd.DataFrame([{"Prompt": v, "Articles": len(valid), "Precision": r["precision"],
                           "Recall": r["recall"], "F1": r["f1"]} for v, r in runs.items()])
    write_table("llm_prompt_variants", table.round(4))
    chosen = choose_prompt(runs, variants)
    second = llm_scores(valid, chosen, pass_id=2)                             # the same prompt once more
    agreement = float((runs[chosen]["labels"] == second["labels"]).mean())   # share of equal answers
    savefig(plot_confusion(runs[chosen]["confusion"], f"LLM, prompt {chosen} (validation)"), "llm_validation_confusion")
    sites = validation_website_table(valid["website"], valid["label"], runs[chosen]["labels"] == "conspiracy")
    write_table("validation_website_recall", sites)         # shows whether validation was an easier set of websites
    values = llm_values(cfg, valid, runs, chosen, agreement)
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


MODEL_BUILDERS.update({"final2_word_counts": lambda: _logistic_spec("final2_word_counts", word_counts_vectorizer, "word_counts"),
                       "final2_tfidf": lambda: _logistic_spec("final2_tfidf", tfidf_vectorizer, "tfidf"),
                       "final2_tfidf_features": lambda: _features_spec("final2_tfidf_features"),
                       "final_llm": lambda: _llm_spec("final_llm")})
FINAL_IDS = tuple(MODEL_BUILDERS)


def final_stage() -> dict:
    """One final run: freeze the four validation-chosen settings, then score each on test exactly once."""
    return {model_id: (freeze_model(model_id), evaluate_final(model_id))[1]["n_docs"] for model_id in FINAL_IDS}


# ---------------------------------------------------------------- stage: comparison

WORDS_FINAL, TFIDF_FINAL, FEATURES_FINAL = "final2_word_counts", "final2_tfidf", "final2_tfidf_features"
BASELINE_ID = "always_mainstream"


def read_final_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test predictions written by evaluate_final: (logistic regression models stacked, the LLM)."""
    tables = results_dir() / "tables"
    lr = pd.concat([pd.read_csv(tables / f"predictions_{m}.csv")
                    for m in (WORDS_FINAL, TFIDF_FINAL, FEATURES_FINAL)], ignore_index=True)
    return lr, pd.read_csv(tables / f"predictions_{LLM_ID}.csv")


def best_final_lr(metrics: dict) -> str:
    """Of the three logistic regressions, the one with the highest validation F1 (a tie keeps the earlier one)."""
    scores = {WORDS_FINAL: metrics["word_counts"]["validation"]["f1"], TFIDF_FINAL: metrics["tfidf"]["validation"]["f1"],
              FEATURES_FINAL: metrics["tfidf_features"]["validation"]["f1"]}
    return max(scores, key=scores.get)


MODEL_NAMES = {WORDS_FINAL: "Word counts", TFIDF_FINAL: "TF-IDF", FEATURES_FINAL: "TF-IDF + features",
               LLM_ID: "LLM", BASELINE_ID: "Always guess mainstream"}


def plot_comparison_f1(table: pd.DataFrame):
    table = table[table["model"] != BASELINE_ID].reset_index(drop=True)      # the baseline has F1 = 0: table only
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
    table = table[table["model"] != BASELINE_ID].reset_index(drop=True)
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


def plot_validation_vs_test(valid: dict, test: dict):
    """Confusion matrices on validation (top) and test (bottom) for the best logistic regression and the LLM."""
    fig, axes = plt.subplots(2, len(test), figsize=(3.4 * len(test), 6.4))
    for col, name in enumerate(test):
        for row, (counts, when) in enumerate(((valid[name], "validation"), (test[name], "test"))):
            mat = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
            ax = axes[row, col]
            ax.imshow(mat, cmap="Greens")
            for (i, j), v in np.ndenumerate(mat):
                ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                        color="white" if v > mat.max() / 2 else "black")
            ax.set_xticks([0, 1], ["main.", "consp."])
            ax.set_yticks([0, 1], ["main.", "consp."])
            ax.set_title(f"{name} ({when})", fontsize=10)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True website type")
    fig.tight_layout()
    return fig


def mcnemar_table(tests: dict) -> pd.DataFrame:
    """The three McNemar tests with plain headers: who is right alone, and the p-value."""
    rows = []
    for test in tests.values():
        rows.append({"First model": MODEL_NAMES.get(test["a"], "Best logistic regression"),
                     "Second model": MODEL_NAMES[test["b"]],
                     "Right only for the first": test["a_only_correct"],
                     "Right only for the second": test["b_only_correct"], "p-value": test["p_text"]})
    return pd.DataFrame(rows)


def align_predictions(lr: pd.DataFrame, llm_all: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """True labels and each model's predictions, on the same test documents in the same order."""
    llm = llm_all[llm_all["pass"] == 1].set_index("doc_id")
    y = llm["label"].to_numpy(dtype=bool)
    preds = {m: lr[lr["model_id"] == m].set_index("doc_id").loc[llm.index]["pred"].to_numpy(dtype=bool)
             for m in (WORDS_FINAL, TFIDF_FINAL, FEATURES_FINAL)}
    preds[LLM_ID] = llm["pred"].to_numpy(dtype=bool)
    return y, preds


def write_comparison_tables(y: np.ndarray, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Write the comparison tables and figures, one row per model plus the 'always guess mainstream' baseline."""
    baseline = np.zeros(len(y), dtype=bool)                  # never says conspiracy
    table = pd.DataFrame([model_row(name, y, pred) for name, pred in {**preds, BASELINE_ID: baseline}.items()])
    write_table("comparison_test", table)
    write_table("comparison_compact", compact_table(table, MODEL_NAMES))
    savefig(plot_comparison_f1(table), "comparison_f1")
    savefig(plot_comparison_confusion(table), "comparison_confusion")
    return table


def run_mcnemar_pairs(y: np.ndarray, preds: dict[str, np.ndarray], best_lr: str) -> dict:
    """McNemar test for the three pairs we compare."""
    pairs = {"words_vs_tfidf": (WORDS_FINAL, TFIDF_FINAL),
             "tfidf_vs_features": (TFIDF_FINAL, FEATURES_FINAL),
             "best_lr_vs_llm": (best_lr, LLM_ID)}
    return {name: mcnemar_pair(y, preds[first], preds[second], first, second) for name, (first, second) in pairs.items()}


def conspiracy_website_recall(y: np.ndarray, preds: dict[str, np.ndarray], websites: np.ndarray) -> pd.DataFrame:
    """For each conspiracy website in the test set: its article count and the recall of every model."""
    rows = []
    for site in sorted(set(websites[y])):
        mine = (websites == site) & y                        # conspiracy articles of this website
        rows.append({"website": site, "test_articles": int(mine.sum()),
                     **{f"recall_{m}": float(preds[m][mine].mean()) for m in preds}})
    return pd.DataFrame(rows)


def website_labels(y: np.ndarray, websites: np.ndarray) -> dict:
    """Plain labels (Website 1, 2, ...) for the conspiracy test websites, so real domains stay out of the text."""
    return {site: f"Website {i}" for i, site in enumerate(sorted(set(websites[y])), start=1)}


def test_websites(llm_all: pd.DataFrame) -> np.ndarray:
    """The website of every test document, in the same order as the aligned predictions."""
    doc_ids = llm_all[llm_all["pass"] == 1]["doc_id"]
    return doc_ids.map(analysis_frame().set_index("doc_id")["website"]).to_numpy()


def large_site(y: np.ndarray, websites: np.ndarray) -> str:
    """The website with the most conspiracy articles in the test set."""
    return pd.Series(websites[y]).value_counts().index[0]


def f1_without_site(y: np.ndarray, preds: dict[str, np.ndarray], websites: np.ndarray, site: str) -> dict:
    """Conspiracy-class F1 of every model after dropping all test articles of one website."""
    keep = websites != site                                  # leave the large hard site out
    return {"site": website_labels(y, websites)[site], "documents": int(keep.sum()), "conspiracy": int(y[keep].sum()),
            **{m: float(model_row(m, y[keep], preds[m][keep])["f1"]) for m in preds}}


def write_website_recall(y: np.ndarray, preds: dict[str, np.ndarray], websites: np.ndarray) -> pd.DataFrame:
    """Write the recall of every model for each conspiracy website in the test set."""
    per_website = conspiracy_website_recall(y, preds, websites)
    per_website["website"] = per_website["website"].map(website_labels(y, websites))      # hide real domains
    names = {"website": "Website", "test_articles": "Test articles"}
    names.update({f"recall_{m}": f"Recall: {MODEL_NAMES[m]}" for m in preds})      # plain column labels
    shown = per_website.copy()
    for column in shown.columns:
        if column.startswith("recall_"):
            shown[column] = shown[column].map("{:.3f}".format)       # three decimals in every cell
    write_table("test_website_recall", shown.rename(columns=names))
    return per_website


def validation_vs_test(metrics: dict, table: pd.DataFrame) -> pd.DataFrame:
    """Validation F1 and test F1 of the four final models side by side."""
    validation = {"final2_word_counts": metrics["word_counts"]["validation"]["f1"],
                  "final2_tfidf": metrics["tfidf"]["validation"]["f1"],
                  "final2_tfidf_features": metrics["tfidf_features"]["validation"]["f1"],
                  "final_llm": metrics["llm"]["validation"]["f1"]}
    rows = []
    for record in table.to_dict("records"):
        if record["model"] in validation:                      # the baseline has no validation run
            rows.append({"Model": MODEL_NAMES[record["model"]],
                         "Validation F1": f"{validation[record['model']]:.3f}",    # three decimals in every cell
                         "Test F1": f"{record['f1']:.3f}"})
    return pd.DataFrame(rows)


def comparison_stage() -> dict:
    """Compare the four final models on the same test documents, from the saved prediction files only."""
    lr, llm_all = read_final_predictions()
    y, preds = align_predictions(lr, llm_all)
    table = write_comparison_tables(y, preds)
    websites = test_websites(llm_all)
    per_website = write_website_recall(y, preds, websites)
    metrics = _metrics()
    best_lr = best_final_lr(metrics)
    mcnemar = run_mcnemar_pairs(y, preds, best_lr)
    write_table("mcnemar_tests", mcnemar_table(mcnemar))
    write_table("validation_vs_test", validation_vs_test(metrics, table))   # where the scores drop
    test_counts = {r["model"]: {k: int(r[k]) for k in ("tp", "fn", "fp", "tn")} for r in table.to_dict("records")}
    savefig(plot_validation_vs_test({"TF-IDF": metrics["tfidf"]["confusion"], "LLM": metrics["llm"]["validation"]["confusion"]},
                                    {"TF-IDF": test_counts[TFIDF_FINAL], "LLM": test_counts[LLM_ID]}),
            "validation_vs_test_confusion")
    values = {"models": list(preds), "lr_best": best_lr, "test_documents": int(len(y)), "test_conspiracy": int(y.sum()),
              "bootstrap": {"resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "level": 0.95},
              "baseline": {"model": BASELINE_ID, "accuracy": float((~y).mean())},
              "per_website": per_website.to_dict("records"),
              "f1_without_large_site": f1_without_site(y, preds, websites, large_site(y, websites)),
              "table": table.to_dict("records"), "mcnemar": mcnemar,
              "llm_stability": metrics["llm"]["repeat_agreement"]}
    write_metrics("comparison", round_floats(values, 4))
    return values


# ---------------------------------------------------------------- stage: errors

def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Test predictions of the best logistic regression and of the LLM (first pass), plus the model id."""
    lr_all, llm_all = read_final_predictions()
    chosen = _metrics()["comparison"]["lr_best"]
    return lr_all[lr_all["model_id"] == chosen], llm_all[llm_all["pass"] == 1], chosen


def code_disagreements(sets: pd.DataFrame, frame: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Ask for a stance on the chosen disagreement articles (cached) and drop unreadable answers."""
    to_code = errors.select_for_coding(sets, cfg["max_per_other_set"], seed)
    client = None
    if errors.missing_stances(to_code["doc_id"]):
        client = errors.stance_client(cfg)          # raises loudly: no key and no complete cache
    stance = errors.code_stances(to_code.assign(txt=to_code["doc_id"].map(frame["txt"])), client=client)
    coded = to_code.merge(stance, on="doc_id")
    coded = coded[coded["claude_stance"] != errors.UNREADABLE].reset_index(drop=True)
    coded["verdict"] = [errors.classify_error(bool(t), s) for t, s in zip(coded["true_label"], coded["claude_stance"])]
    return coded


def large_site_stances(coded: pd.DataFrame) -> pd.DataFrame:
    """Stances of the conspiracy-site articles the LLM missed, on the website where it missed the most."""
    missed = coded[(coded["llm_pred"] != coded["true_label"]) & coded["true_label"].astype(bool)]
    site = missed["website"].value_counts().idxmax()                 # the website with the most misses
    counts = missed[missed["website"] == site]["claude_stance"].value_counts()
    return pd.DataFrame({"stance": [s.replace("_", " ") for s in counts.index], "articles": counts.to_numpy()})


def pick_examples(coded: pd.DataFrame, frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Six disagreement examples with a short excerpt, picked by a seeded rule among coded articles."""
    ex = errors.pick_examples(coded, frame["txt"], seed)
    ex["website"] = ex["doc_id"].map(frame["website"])
    for col in ("true_label", "lr_pred", "llm_pred"):
        ex[col] = ex[col].map(CLASS_NAMES)
    return ex.rename(columns={"lr_pred": "lr_prediction", "llm_pred": "llm_prediction"})


def write_error_tables(coded: pd.DataFrame, sets: pd.DataFrame, examples: pd.DataFrame, sample: pd.DataFrame) -> bool:
    """Write the error tables and the stance check sample; return whether the sample file was rewritten."""
    tables = results_dir() / "tables"
    write_table("error_classification", errors.error_classification_table(coded, sets))
    written = errors.write_check_sample(sample, tables / "stance_check_sample.csv")
    errors.write_claude_answers(sample, tables / "stance_claude_answers.csv")
    write_table("stance_check_layout", errors.check_sheet_layout())
    write_table("error_examples", examples[["cell", "doc_id", "true_label", "lr_prediction",
                                            "llm_prediction", "stance", "excerpt"]])    # no website next to a label
    write_table("error_examples_plain", errors.plain_examples(examples))
    write_table("error_counts_plain", errors.plain_counts(coded, sets))
    return written


def check_sample(frame: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Validation articles for the human check: the existing sheet's, else a balanced draw; Claude's stance (cached)."""
    split = load_split().set_index("doc_id")["split"]
    pool = frame.reset_index()
    pool = pool[pool["doc_id"].map(split) == "validation"]
    pool = pool[["doc_id", "website", "label", "txt"]].reset_index(drop=True)
    sheet = results_dir() / "tables" / "stance_check_sample.csv"
    if sheet.is_file():                              # keep the 40 articles the team already has
        ids = pd.read_csv(sheet, dtype=str)["doc_id"]
        sample = pool[pool["doc_id"].isin(ids)].reset_index(drop=True)
    else:
        sample = errors.balanced_website_sample(pool, cfg["human_sample_size"], cfg["max_per_website"], seed)
    client = errors.stance_client(cfg) if errors.missing_stances(sample["doc_id"]) else None
    stance = errors.code_stances(sample, client=client)
    return sample.drop(columns="txt").merge(stance, on="doc_id")


def error_counts(sets: pd.DataFrame, coded: pd.DataFrame) -> dict:
    """Per disagreement set: sizes, label split, error types and how many were coded."""
    totals = sets.groupby(["set", "error_type"]).size()
    return {s: {"total": int((sets["set"] == s).sum()),
                "conspiracy": int(((sets["set"] == s) & sets["true_label"]).sum()),
                "mainstream": int(((sets["set"] == s) & ~sets["true_label"]).sum()),
                "false_positive": int(totals.get((s, "false positive"), 0)),
                "false_negative": int(totals.get((s, "false negative"), 0)),
                "coded": int((coded["set"] == s).sum())} for s in errors.SETS}


def label_problem_values(coded: pd.DataFrame) -> dict:
    """How many coded articles look like label problems (Claude's stance only)."""
    return {"label_problems": int((coded["verdict"] == errors.LABEL_PROBLEM).sum()),
            "of_coded": int(len(coded)),
            "source": "Claude stance coding; checked by the team on 40 validation articles (human_check)"}


def errors_values(chosen: str, cfg: dict, sets: pd.DataFrame, coded: pd.DataFrame, sample: pd.DataFrame, written: bool,
                  hand: dict) -> dict:
    """The numbers of the errors stage, as written to metrics.json."""
    return {"data": "TEST rows, filtered documents; predictions from evaluate_final files only",
            "lr_model": chosen, "stance_model": cfg["model"],
            "counts": error_counts(sets, coded),
            "coded_documents": int(len(coded)),
            "stance_counts": coded["claude_stance"].value_counts().to_dict(),
            "verdict_counts": coded["verdict"].value_counts().to_dict(),
            "label_problem_share": label_problem_values(coded),
            "rule": "conspiracy site + (rejects|neutral|no_claim) -> label problem; mainstream site + endorses -> label problem; else model wrong",
            "human_check": {"sample_file": "results/tables/stance_check_sample.csv", "rows": int(len(sample)),
                            "file_rewritten_this_run": written, **hand}}


def hand_check_stage(sample: pd.DataFrame) -> dict:
    """Compare the team's blind answers with each other and with the automatic coder."""
    folder = REPO_ROOT / errors.stance_config()["hand_check_dir"]
    codes = errors.read_hand_check(folder, sample["doc_id"])
    write_table("hand_check_agreement", errors.agreement_table(codes, sample))
    write_table("hand_check_verdicts", errors.verdicts_by_site_type(codes, sample))
    return errors.hand_check_values(codes, sample)


def errors_stage() -> dict:
    """Test articles where the best LR and the LLM disagree -> stance coding (cached) -> tables."""
    cfg = errors.stance_config()
    seed = project_config()["split"]["seed"]
    lr, llm, chosen = load_predictions()
    frame = analysis_frame().set_index("doc_id")
    sets = errors.disagreement_sets(lr, llm)
    sets["website"] = sets["doc_id"].map(frame["website"])
    coded = code_disagreements(sets, frame, cfg, seed)
    examples = pick_examples(coded, frame, seed)
    write_table("large_site_missed_stances", large_site_stances(coded))   # why the large site is missed
    sample = check_sample(frame, cfg, seed)
    written = write_error_tables(coded, sets, examples, sample)
    hand = hand_check_stage(sample)                                       # the team's blind check
    values = errors_values(chosen, cfg, sets, coded, sample, written, hand)
    write_metrics("errors", round_floats(values, 4))
    return values


STAGES: list[tuple[str, Callable[[], object]]] = [("cleaning", cleaning_stage),
                                                  ("word_counts", word_counts_stage), ("tfidf", tfidf_stage), ("features", feature_summary_stage),
                                                  ("tfidf_features", ablation), ("llm", llm_stage),
                                                  ("final", final_stage), ("comparison", comparison_stage),
                                                  ("errors", errors_stage)]
PENDING_STAGES: list[str] = []


def write_run_info(seconds: float, stage_seconds: dict[str, float] | None = None) -> Path:
    """Record how long `fml run-all` took and which software versions produced results/.

    Kept apart from metrics.json (whose top-level keys are a contract) so the report can
    quote runtime and versions from a file instead of typing them.
    """
    from importlib.metadata import version

    packages = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "anthropic", "pyyaml")
    info = {"run_all_seconds": round(seconds, 1),
            "python": platform.python_version(),
            "machine": {"system": platform.system(), "processor": platform.machine(), "cpus": os.cpu_count()},
            "stage_seconds": {name: round(value, 1) for name, value in (stage_seconds or {}).items()},
            "packages": {name: version(name) for name in packages}}
    path = results_dir() / "run_info.json"
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return path


def run_all() -> None:
    """Reproduce every file in results/ from the data repo (LLM answers come from the cache)."""
    from .data_report import run_data_stage

    started = time.perf_counter()
    run_data_stage()
    stage_seconds = {}
    for name, stage in STAGES:
        stage_started = time.perf_counter()
        stage()
        stage_seconds[name] = time.perf_counter() - stage_started
    write_run_info(time.perf_counter() - started, stage_seconds)
    write_methods_overview()
    if PENDING_STAGES:
        raise NotImplementedError("fml run-all: stages not implemented yet: " + ", ".join(PENDING_STAGES))
