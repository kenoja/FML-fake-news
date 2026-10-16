"""The two entry points the CLI calls, plus the stages they are made of.

Stages are added one by one over the cycle. Until every stage exists `run_all` raises after running
the finished ones, so `fml run-all` can never look successful while results are missing
(AGENTS.md: fail loud).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from .clean import clean_corpus, website_names  # noqa: E402
from .data import load_corpus, on_topic_mask  # noqa: E402
from .evaluate import (BOOTSTRAP_SEED, N_BOOTSTRAP, bootstrap_ci, classification_metrics, confusion,  # noqa: E402
                       majority_baseline, pr_curve_data, stratified_random_baseline)
from .models import build_pipeline, split_frames, top_coefficients, tune_on_validation  # noqa: E402
from .paths import project_config  # noqa: E402
from .results import CONSPIRACY_COLOUR, MAINSTREAM_COLOUR, results_dir, savefig, write_metrics, write_table  # noqa: E402
from .split import load_split, split_path  # noqa: E402
from .topic import mention_counts, short_title, term_matches  # noqa: E402

M1_ID = "m1_words_all_docs"
M1_FILTERED_ID = "m1_words_filtered"
M1_FEATURES = {"ngram_range": [1, 1], "min_df": 2, "sublinear_tf": True, "numeric_columns": []}
LENGTH_BUCKETS = (150, 250)
EXPECTED_FILTERED = {"documents": 2608, "conspiracy": 777, "mainstream": 1831}
SPOT_CHECK_SIZE = 6
CLASS_NAMES = {True: "conspiracy", False: "mainstream"}


# ---------------------------------------------------------------- data for the stages

def cleaned_frame(keep_all: bool = True) -> pd.DataFrame:
    """Corpus with cleaned text (URLs, site names, boilerplate removed) and a boolean `label`.

    `keep_all` keeps the two exact duplicates so the first run really covers all 3,146 documents
    that the frozen split lists; the split decides which part each one belongs to.
    """
    cleaned, _, _ = clean_corpus(load_corpus(), drop_exact=not keep_all)
    return cleaned.reset_index(drop=True)


# ---------------------------------------------------------------- stage: baseline_all_docs

def round_floats(obj, digits: int = 3):
    """Round every float in a nested dict/list, so numbers in metrics.json read as quoted in the prose."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, digits) for v in obj]
    return obj


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


def plot_pr(curve: dict, base_rate: float, ap: float, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(curve["recall"], curve["precision"], color=CONSPIRACY_COLOUR, label=f"logistic regression (AP {ap:.2f})")
    ax.axhline(base_rate, color=MAINSTREAM_COLOUR, linestyle="--", label=f"always conspiracy ({base_rate:.2f})")
    ax.set_xlabel("Recall (conspiracy)")
    ax.set_ylabel("Precision (conspiracy)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="upper right")
    return fig


def website_name_tokens(df: pd.DataFrame, min_len: int = 4) -> set[str]:
    """Word pieces of all website names ("sitename" from "sitename.com"), to spot giveaways."""
    import re

    return {t for n in website_names(df) for t in re.findall(r"[a-z0-9]+", n) if len(t) >= min_len} - {"com", "www", "html"}


def length_bucket_table(valid: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    words = valid["txt"].str.split().map(len)
    edges = [0, *LENGTH_BUCKETS, int(words.max()) + 1]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ((words >= lo) & (words < hi)).to_numpy()
        row = {"bucket": f"{lo}-{hi - 1} words" if hi <= edges[-2] else f">= {lo} words", "documents": int(m.sum()),
               "conspiracy": int(valid["label"].to_numpy()[m].sum())}
        row.update(classification_metrics(valid["label"].to_numpy()[m], pred[m]) if m.any() else {})
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_all_docs() -> dict:
    """M1: words-only TF-IDF (1-grams) + L2 logistic regression on ALL documents, VALIDATION metrics only."""
    df = cleaned_frame(keep_all=True)
    frames = split_frames(df, ("train", "validation"))
    train, valid = frames["train"], frames["validation"]

    pipe, grid = tune_on_validation(train, valid, M1_FEATURES)
    write_table(f"{M1_ID}_grid", grid)
    sel = grid[grid["selected"]].iloc[0]

    y, pred = valid["label"].to_numpy(), pipe.predict(valid)
    scores = pipe.decision_function(valid)
    sites = valid["website"].to_numpy()
    cis = {"documents": bootstrap_ci(y, pred), "websites": bootstrap_ci(y, pred, groups=sites)}
    c = confusion(y, pred)
    ap = float(average_precision_score(y, scores))

    savefig(plot_confusion(c, "Words-only LR, all documents (validation)"), "baseline_all_docs_confusion")
    savefig(plot_pr(pr_curve_data(y, scores), float(y.mean()), ap, "Precision-recall, validation"),
            "baseline_all_docs_pr")
    write_table(f"{M1_ID}_length_buckets", length_bucket_table(valid, pred))

    coefs = top_coefficients(pipe)
    write_table(f"{M1_ID}_top_coefficients", coefs)
    giveaways = sorted(set(coefs["feature"]) & website_name_tokens(df))

    values = {
        "model_id": M1_ID,
        "split_evaluated": "validation",
        "features": M1_FEATURES,
        "selected": {"C": float(sel["C"]), "class_weight": sel["class_weight"], "selection_metric": "f1 (validation)"},
        "grid_values_tried": int(len(grid)),
        "documents": {"all": int(len(df)), "train": int(len(train)), "validation": int(len(valid)),
                      "validation_conspiracy": int(y.sum()), "validation_websites": int(len(set(sites)))},
        "confusion": c,
        "validation": {**{k: v for k, v in cis["documents"].items()}},
        "validation_website_bootstrap": cis["websites"],
        "bootstrap": {"resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "level": 0.95},
        "average_precision": ap,
        "baselines": {
            "majority_class": majority_baseline(train["label"], y),
            "stratified_random": stratified_random_baseline(train["label"], y, BOOTSTRAP_SEED),
        },
        "website_name_tokens_in_top_coefficients": giveaways,
        "top_coefficient_note": ("No website-name token among the top 25 words per class."
                                 if not giveaways else
                                 "Website-name tokens remain among the top words: " + ", ".join(giveaways)),
    }
    write_metrics("baseline_all_docs", round_floats(values))
    return values


# ---------------------------------------------------------------- stage: topic_filter

def filter_to_topic(cleaned: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Rows of `cleaned` whose document mentions 9/11 according to the configured pattern.

    The pattern is applied to the ORIGINAL title and text (`raw`), not to the cleaned text: cleaning
    strips website names and boilerplate and must not change which documents count as on-topic. Rows
    keep their split because the frozen split is looked up by doc_id.
    """
    keep = set(raw.loc[on_topic_mask(raw), "doc_id"])
    return cleaned[cleaned["doc_id"].isin(keep)].reset_index(drop=True)


def analysis_frame() -> pd.DataFrame:
    """THE frame every experiment after the first run uses (off-topic documents removed).

    One shared function, so no later stage can quietly go back to the unfiltered corpus.
    """
    return filter_to_topic(cleaned_frame(keep_all=True), load_corpus())


def _count_row(level: str, split: str, label: str, sub: pd.DataFrame) -> dict:
    before, after = len(sub), int(sub["on_topic"].sum())
    return {"level": level, "split": split, "label": label, "documents_before": before, "documents_after": after,
            "removed": before - after, "removal_rate": (before - after) / before if before else 0.0,
            "websites_before": int(sub["website"].nunique()),
            "websites_after": int(sub.loc[sub["on_topic"], "website"].nunique())}


def topic_count_table(df: pd.DataFrame) -> pd.DataFrame:
    """Before/after counts overall, by class, by split and by split x class (`df` has split, label, on_topic)."""
    name = df["label"].map(CLASS_NAMES)
    splits = ("train", "validation", "test")
    rows = [_count_row("overall", "all", "all", df)]
    rows += [_count_row("class", "all", lab, df[name == lab]) for lab in ("conspiracy", "mainstream")]
    rows += [_count_row("split", sp, "all", df[df["split"] == sp]) for sp in splits]
    rows += [_count_row("split_class", sp, lab, df[(df["split"] == sp) & (name == lab)])
             for sp in splits for lab in ("conspiracy", "mainstream")]
    return pd.DataFrame(rows)


def plot_topic_filter(counts: pd.DataFrame):
    """Left: documents before/after per class. Right: share removed per split and class."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
    classes = [("conspiracy", CONSPIRACY_COLOUR), ("mainstream", MAINSTREAM_COLOUR)]
    cl = counts[counts["level"] == "class"].set_index("label")
    for i, (lab, col) in enumerate(classes):
        for dx, key, alpha in ((-0.2, "documents_before", 0.4), (0.2, "documents_after", 1.0)):
            a.bar(i + dx, cl.loc[lab, key], 0.4, color=col, alpha=alpha)
            a.text(i + dx, cl.loc[lab, key], f"{int(cl.loc[lab, key]):,}", ha="center", va="bottom", fontsize=9)
    a.set_xticks([-0.2, 0.2, 0.8, 1.2], ["before", "after", "before", "after"])
    top = cl["documents_before"].max()
    for i, (lab, col) in enumerate(classes):
        a.text(i, -0.13 * top, lab, ha="center", color=col, fontsize=10)
    a.set_ylabel("Documents")
    a.set_title("Documents before / after the filter")
    sc = counts[counts["level"] == "split_class"]
    w = 0.38
    for i, (lab, col) in enumerate(classes):
        sub = sc[sc["label"] == lab].set_index("split").loc[["train", "validation", "test"]]
        b.bar(np.arange(3) + (i - 0.5) * w, sub["removal_rate"] * 100, w, color=col, label=lab)
    b.set_xticks(range(3), ["train", "validation", "test"])
    b.set_ylabel("Documents removed (%)")
    b.set_title("Removal rate by split and class")
    b.legend()
    fig.tight_layout()
    return fig


def title_word_counts(titles: pd.Series, k: int = 15) -> pd.Series:
    """In how many titles each (non-stop) word occurs: a cheap look at what the removed documents were about."""
    import re

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    sets = titles.fillna("").str.lower().map(lambda t: set(re.findall(r"[a-z]{3,}", t)) - ENGLISH_STOP_WORDS)
    return pd.Series([w for ws in sets for w in ws], dtype=object).value_counts().head(k)


def term_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Per pattern term: kept documents it matches, and those matched by this term ONLY."""
    tm = term_matches(raw)
    keep = raw["on_topic"].to_numpy()
    lab = raw["label"].to_numpy()
    n_terms = tm.sum(axis=1).to_numpy()
    rows = []
    for term in tm.columns:
        m = tm[term].to_numpy() & keep
        only = m & (n_terms == 1)
        rows.append({"term": term, "documents_matching": int(m.sum()), "conspiracy": int((m & lab).sum()),
                     "mainstream": int((m & ~lab).sum()), "only_this_term": int(only.sum()),
                     "only_this_term_conspiracy": int((only & lab).sum()),
                     "only_this_term_mainstream": int((only & ~lab).sum())})
    return pd.DataFrame(rows).sort_values("documents_matching", ascending=False)


def topic_filter() -> dict:
    """Apply the off-topic filter AFTER the first run and document exactly what it does.

    Writes before/after counts, which pattern term admitted each kept document (esp. "pentagon", which
    can match defence stories unrelated to 9/11), a small spot check as title-only listings (titles cut
    to the excerpt limit) and what the removed documents were about. The split file is only read:
    removed documents keep their assignment and are simply not used later.
    """
    from scipy.stats import fisher_exact

    raw = load_corpus().merge(load_split(), on="doc_id", how="left", validate="one_to_one")
    if raw["split"].isna().any():
        raise RuntimeError("some documents are missing from the frozen split")
    raw["on_topic"] = on_topic_mask(raw).to_numpy()
    counts = topic_count_table(raw)
    write_table("topic_filter_counts", counts)
    ov = counts[counts["level"] == "overall"].iloc[0]
    cl = counts[counts["level"] == "class"].set_index("label")
    sp = counts[counts["level"] == "split"].set_index("split")
    got = {"documents": int(ov["documents_after"]), "conspiracy": int(cl.loc["conspiracy", "documents_after"]),
           "mainstream": int(cl.loc["mainstream", "documents_after"])}

    sites = (raw.groupby(["website", "label", "split"]).agg(documents=("doc_id", "size"), kept=("on_topic", "sum"))
             .reset_index().assign(removed=lambda d: d["documents"] - d["kept"]))
    sites["removal_rate"] = sites["removed"] / sites["documents"]
    sites["label"] = sites["label"].map(CLASS_NAMES)
    sites = sites.sort_values(["removed", "documents"], ascending=False)
    write_table("topic_filter_by_website", sites)

    terms = term_table(raw)
    write_table("topic_filter_terms", terms)
    tm = term_matches(raw)
    keep = raw["on_topic"].to_numpy()
    pent = raw[keep & tm["pentagon"].to_numpy() & (tm.sum(axis=1).to_numpy() == 1)].copy()
    pent["pentagon_mentions"] = mention_counts(pent, "pentagon").to_numpy()
    attack_rx = r"attack|terror|hijack|plane"
    removed = raw[~raw["on_topic"]]
    rng = np.random.default_rng(project_config()["split"]["seed"])
    picks = [("removed", removed), ("kept_by_pentagon_only", pent)]
    spot = pd.concat([d.iloc[rng.choice(len(d), min(SPOT_CHECK_SIZE, len(d)), replace=False)].assign(group=g)
                      for g, d in picks])
    spot = spot.assign(label=spot["label"].map(CLASS_NAMES), title=spot["title"].map(short_title),
                       words=spot["txt"].str.split().map(len),
                       pentagon_mentions=mention_counts(spot, "pentagon").to_numpy())
    write_table("topic_filter_spot_check", spot[["group", "doc_id", "website", "label", "split", "title", "words",
                                                 "pentagon_mentions"]])
    tw = [pd.DataFrame({"class": name, "title_word": s.index, "removed_documents": s.to_numpy()})
          for name, flag in (("conspiracy", True), ("mainstream", False))
          for s in [title_word_counts(removed.loc[removed["label"] == flag, "title"])]]
    write_table("topic_filter_removed_title_words", pd.concat(tw))

    tab = [[int(cl.loc[c, "removed"]), int(cl.loc[c, "documents_after"])] for c in ("conspiracy", "mainstream")]
    odds, p_fisher = fisher_exact(tab)
    values = {
        "pattern_source": "config/project.yaml: data.topic_filter_pattern",
        "documents_before": int(ov["documents_before"]), "documents_after": got["documents"],
        "removed": int(ov["removed"]),
        "after_by_class": {"conspiracy": got["conspiracy"], "mainstream": got["mainstream"]},
        "expected_after": EXPECTED_FILTERED, "matches_expected": got == EXPECTED_FILTERED,
        "removed_by_class": {c: int(cl.loc[c, "removed"]) for c in cl.index},
        "removal_rate_by_class": {c: float(cl.loc[c, "removal_rate"]) for c in cl.index},
        "class_share_before": {c: float(cl.loc[c, "documents_before"] / ov["documents_before"]) for c in cl.index},
        "class_share_after": {c: float(cl.loc[c, "documents_after"] / ov["documents_after"]) for c in cl.index},
        "removal_rate_difference_test": {
            "test": "Fisher exact on documents (ignores that documents of one website are alike)",
            "odds_ratio_removed_conspiracy_vs_mainstream": float(odds), "p_value": float(p_fisher)},
        "removal_rate_by_split": {s: float(sp.loc[s, "removal_rate"]) for s in sp.index},
        "share_of_removed_by_split": {s: float(sp.loc[s, "removed"] / ov["removed"]) for s in sp.index},
        "websites": {"before": int(ov["websites_before"]), "after": int(ov["websites_after"]),
                     "conspiracy_before": int(cl.loc["conspiracy", "websites_before"]),
                     "conspiracy_after": int(cl.loc["conspiracy", "websites_after"]),
                     "mainstream_before": int(cl.loc["mainstream", "websites_before"]),
                     "mainstream_after": int(cl.loc["mainstream", "websites_after"]),
                     "with_no_document_left": int((sites["kept"] == 0).sum())},
        "websites_with_most_removed": sites.head(5)[["website", "label", "removed", "documents"]].to_dict("records"),
        "terms": {r["term"]: {"matching": r["documents_matching"], "only_this_term": r["only_this_term"]}
                  for r in terms.to_dict("records")},
        "pentagon_only": {
            "documents": int(len(pent)), "conspiracy": int(pent["label"].sum()),
            "share_of_kept": float(len(pent) / keep.sum()),
            "single_mention": int((pent["pentagon_mentions"] == 1).sum()),
            "median_mentions": float(pent["pentagon_mentions"].median()) if len(pent) else 0.0,
            "with_attack_words": int(pent["txt"].str.contains(attack_rx, case=False, regex=True).sum()),
            "attack_words_pattern": attack_rx},
        "split_file_sha256": hashlib.sha256(split_path().read_bytes()).hexdigest(),
    }
    savefig(plot_topic_filter(counts), "topic_filter_before_after")
    write_metrics("topic_filter", round_floats(values, 4))
    return values


# ---------------------------------------------------------------- stage: baseline (filtered documents)

def comparison_row(setting: str, trained_on: str, evaluated_on: str, y, pred, sites) -> dict:
    d, w = bootstrap_ci(y, pred), bootstrap_ci(y, pred, groups=sites)
    return {"setting": setting, "trained_on": trained_on, "evaluated_on": evaluated_on, "documents": int(len(y)),
            "conspiracy": int(np.sum(y)), "websites": int(len(set(sites))),
            **{k: d[k]["estimate"] for k in ("precision", "recall", "f1", "macro_f1", "balanced_accuracy")},
            "f1_ci_doc_low": d["f1"]["ci_low"], "f1_ci_doc_high": d["f1"]["ci_high"],
            "f1_ci_site_low": w["f1"]["ci_low"], "f1_ci_site_high": w["f1"]["ci_high"]}


def baseline_filtered() -> dict:
    """M1 retrained on the filtered documents (same frozen split), validation metrics only.

    Also the like-for-like row: the ALL-documents model (settings of the first run) scored on the filtered
    validation documents. Validation before and after the filter contains different documents, so the two
    headline F1 values are not directly comparable; row B isolates the effect of the training set.
    """
    all_df = cleaned_frame(keep_all=True)
    df = analysis_frame()
    f = split_frames(df, ("train", "validation"))
    train, valid = f["train"], f["validation"]
    pipe, grid = tune_on_validation(train, valid, M1_FEATURES)
    write_table(f"{M1_FILTERED_ID}_grid", grid)
    sel = grid[grid["selected"]].iloc[0]
    y, pred, sites = valid["label"].to_numpy(), pipe.predict(valid), valid["website"].to_numpy()
    c, ap = confusion(y, pred), float(average_precision_score(y, pipe.decision_function(valid)))
    savefig(plot_confusion(c, "Words-only LR, filtered documents (validation)"), "baseline_filtered_confusion")
    coefs = top_coefficients(pipe)
    write_table(f"{M1_FILTERED_ID}_top_coefficients", coefs)

    first = json.loads((results_dir() / "metrics.json").read_text(encoding="utf-8"))["baseline_all_docs"]
    if not first:
        raise RuntimeError("run the baseline_all_docs stage first: its tuned settings are reused here")
    cw = None if first["selected"]["class_weight"] == "none" else first["selected"]["class_weight"]
    fa = split_frames(all_df, ("train", "validation"))
    old = build_pipeline(first["features"], C=first["selected"]["C"], class_weight=cw).fit(fa["train"], fa["train"]["label"])
    va = fa["validation"]
    on = va["doc_id"].isin(set(valid["doc_id"])).to_numpy()
    ya, pa, sa = va["label"].to_numpy(), old.predict(va), va["website"].to_numpy()
    rows = [comparison_row("A: all-docs model, all validation docs (first run)", "all", "all validation", ya, pa, sa),
            comparison_row("B: all-docs model, filtered validation docs (like-for-like)", "all", "filtered validation",
                           ya[on], pa[on], sa[on]),
            comparison_row("C: filtered model, filtered validation docs", "filtered", "filtered validation", y, pred, sites)]
    write_table("topic_filter_f1_comparison", pd.DataFrame(rows))
    off = ~on

    values = {
        "model_id": M1_FILTERED_ID, "split_evaluated": "validation", "features": M1_FEATURES,
        "data": "filtered (documents matching data.topic_filter_pattern), same split_v1",
        "selected": {"C": float(sel["C"]), "class_weight": sel["class_weight"], "selection_metric": "f1 (validation)"},
        "grid_values_tried": int(len(grid)),
        "documents": {"filtered": int(len(df)), "train": int(len(train)), "validation": int(len(valid)),
                      "validation_conspiracy": int(y.sum()), "validation_websites": int(len(set(sites)))},
        "confusion": c,
        "validation": bootstrap_ci(y, pred),
        "validation_website_bootstrap": bootstrap_ci(y, pred, groups=sites),
        "bootstrap": {"resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "level": 0.95},
        "average_precision": ap,
        "baselines": {"majority_class": majority_baseline(train["label"], y),
                      "stratified_random": stratified_random_baseline(train["label"], y, BOOTSTRAP_SEED)},
        "website_name_tokens_in_top_coefficients": sorted(set(coefs["feature"]) & website_name_tokens(all_df)),
        "comparison_with_all_docs": {
            "note": ("The validation sets differ (all validation documents vs only the on-topic ones), so the F1 values "
                     "of rows A and C are NOT directly comparable. Row B scores the all-documents model on exactly "
                     "the documents of row C. Differences are descriptive; no significance test is claimed."),
            "rows": rows,
            "f1_change_c_minus_b": rows[2]["f1"] - rows[1]["f1"],
            "f1_change_c_minus_a": rows[2]["f1"] - rows[0]["f1"],
            "all_docs_model_on_removed_validation_docs": {
                "documents": int(off.sum()), "conspiracy": int(ya[off].sum()),
                "accuracy": float((ya[off] == pa[off]).mean()) if off.any() else None,
                "predicted_conspiracy": int(pa[off].sum())}},
    }
    write_metrics("baseline", round_floats(values))
    return values


# ---------------------------------------------------------------- final test scoring

@dataclass
class ModelSpec:
    """Everything that defines a model: `config` is hashed, `fit_predict` trains on train, predicts test."""
    config: dict
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _m1_spec() -> ModelSpec:
    sel = json.loads((results_dir() / "metrics.json").read_text(encoding="utf-8"))["baseline_all_docs"]
    if not sel:
        raise RuntimeError("run the baseline_all_docs stage first: the tuned settings come from validation")
    cw = None if sel["selected"]["class_weight"] == "none" else sel["selected"]["class_weight"]
    config = {"model_id": M1_ID, "features": sel["features"], "C": sel["selected"]["C"], "class_weight": cw,
              "data": "all_documents_cleaned", "split": project_config()["split"]["file"]}

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return build_pipeline(config["features"], C=config["C"], class_weight=cw).fit(train, train["label"]).predict(test)

    return ModelSpec(config, fit_predict)


MODEL_BUILDERS: dict[str, Callable[[], ModelSpec]] = {M1_ID: _m1_spec}


def read_access_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_final(model_id: str, builders: dict[str, Callable[[], ModelSpec]] | None = None,
                   frame: pd.DataFrame | None = None, log_path: Path | None = None) -> dict:
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
    previous = [e for e in read_access_log(log_path) if e["model_id"] == model_id]
    if any(e["config_hash"] != digest for e in previous):
        raise PermissionError(f"{model_id} was already scored on test with a different configuration; "
                              "a new configuration needs a new model id (and is a new test access).")

    frame = cleaned_frame(keep_all=True) if frame is None else frame
    split = load_split()
    train = frame[frame["doc_id"].isin(set(split.loc[split["split"] == "train", "doc_id"]))].reset_index(drop=True)
    test = frame[frame["doc_id"].isin(set(split.loc[split["split"] == "test", "doc_id"]))].reset_index(drop=True)
    pred = spec.fit_predict(train, test)
    y = test["label"].to_numpy()
    out = {"model_id": model_id, "config_hash": digest, "n_docs": int(len(test)),
           "confusion": confusion(y, pred),
           "metrics_documents": bootstrap_ci(y, pred),
           "metrics_websites": bootstrap_ci(y, pred, groups=test["website"].to_numpy())}
    if not previous:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"model_id": model_id, "config_hash": digest,
                                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "n_docs": out["n_docs"]}) + "\n")
    return out


# ---------------------------------------------------------------- entry points

STAGES: list[tuple[str, Callable[[], object]]] = [("baseline_all_docs", baseline_all_docs), ("topic_filter", topic_filter),
                                                  ("baseline", baseline_filtered)]
PENDING_STAGES = ["ablation", "llm", "comparison", "errors"]


def run_all() -> None:
    """Reproduce every file in results/ from the data repo (LLM answers come from the cache)."""
    from .data_report import run_data_stage

    run_data_stage()
    for _, stage in STAGES:
        stage()
    if PENDING_STAGES:
        raise NotImplementedError("fml run-all: stages not implemented yet: " + ", ".join(PENDING_STAGES))
