"""Error analysis on the TEST predictions: who is wrong where, and is the *label* the problem?

The labels describe the WEBSITE, not the article. A mainstream outlet can run a piece that debunks 9/11
theories; a conspiracy site can run a plain news item. Such documents are "errors" only because the label
says something the text does not. To measure how often that happens, Claude Sonnet 5.5 codes each
disagreement document's stance, and an explicit rule (`classify_error`) turns stance + true label into
"model wrong" or "label does not describe the text".

Nothing here fits or scores a model: it reads the prediction files written by `evaluate_final`. The stance
cache, like the classifier cache, stores doc_id + prompt hash + answer and never article text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import llm
from .paths import REPO_ROOT, project_config

STANCES = ("endorses", "rejects_or_debunks", "reports_neutrally", "no_claim")
UNREADABLE = llm.UNREADABLE
SETS = ("lr_only_wrong", "llm_only_wrong", "both_wrong")
MODEL_WRONG, LABEL_PROBLEM = "model wrong", "label does not describe the text"
JUSTIFICATION_MAX_WORDS = 25
NAME_ALLOWLIST = {
    "The", "This", "That", "These", "Those", "It", "Its", "A", "An", "Both", "Although", "While", "Written",
    "Article", "Articles", "Some", "No", "Not", "Only", "Many", "Most", "Neither", "Either", "However",
    "Saudi", "Saudis", "Russia", "Russian", "American", "Americans", "US", "USA", "WTC", "FBI", "CIA", "NIST",
    "Pentagon", "World", "Trade", "Center", "Twin", "Towers", "Qaeda", "Al", "Iraq", "Afghanistan", "Islamic",
    "Muslim", "Muslims", "Israel", "Israeli", "COVID", "September", "Building", "Washington", "New", "York",
    "Congress", "Senate", "Arabia", "Arab", "Middle", "East", "Taliban", "ISIS", "UK", "EU", "NATO",
}
_CAP_TOKEN = re.compile(r"[A-Z][\w'’-]*")
CHECK_COLUMNS = ["doc_id", "website", "human_stance"]     # no Claude stance: the team reads blind
EXCERPT_WORDS = 29

STANCE_PROMPT = (
    "You code how a web article treats CONSPIRACY CLAIMS about the 9/11 attacks (for example: the attacks "
    "were an inside job, the official account is a lie or cover-up, the towers were brought down by "
    "explosives). Judge only what the article itself says, not which website it might come from.\n"
    "Choose exactly one stance:\n"
    '- "endorses": the article presents such a claim as true or likely true, or argues for it.\n'
    '- "rejects_or_debunks": the article discusses such claims in order to refute, criticise or debunk them.\n'
    '- "reports_neutrally": the article describes such claims or the debate around them without taking a side.\n'
    '- "no_claim": the article makes and discusses no such conspiracy claim (e.g. ordinary news, history, '
    "opinion on another subject).\n"
    'Do not explain your reasoning first. Your reply must START with the character { and be strict JSON '
    'only, no other text: {"stance": one of the four values, "justification": "at most 25 words"}'
)


def scrub_justification(text: str, max_words: int = JUSTIFICATION_MAX_WORDS) -> str:
    """Shorten to `max_words` and replace capitalised name-like tokens outside NAME_ALLOWLIST by "[name]".

    Why: Claude's justifications are public (CSV tables, report) and may name people from the articles. A
    word-level rule over-strips (a place may become "[name]") but never leaks a name, which matters more.
    """
    def repl(m: re.Match) -> str:
        base = re.sub(r"(['’]s|-\d+)$", "", m.group(0))
        return m.group(0) if base in NAME_ALLOWLIST else "[name]"

    out = _CAP_TOKEN.sub(repl, " ".join(str(text).split()[:max_words]))
    return re.sub(r"\[name\](?:\s+\[name\])+", "[name]", out)


def stance_config() -> dict:
    return project_config()["stance_coding"]


def stance_cache_path(cfg: dict | None = None) -> Path:
    return REPO_ROOT / (cfg or stance_config())["cache_file"]


def stance_key(cfg: dict | None = None) -> str:
    cfg = cfg or stance_config()
    return llm.prompt_hash(STANCE_PROMPT, cfg["model"], cfg["max_input_words"], 1)


# ---------------------------------------------------------------- disagreement sets

def disagreement_sets(lr: pd.DataFrame, llm_pred: pd.DataFrame) -> pd.DataFrame:
    """One row per test document that at least one model gets wrong, with its set."""
    # inputs have doc_id, label and pred (True = conspiracy)
    m = lr[["doc_id", "label", "pred"]].merge(llm_pred[["doc_id", "pred"]], on="doc_id", suffixes=("_lr", "_llm"))
    if len(m) != len(lr) or len(m) != len(llm_pred):
        raise RuntimeError("LR and LLM predictions cover different test documents")
    m["lr_wrong"] = m["pred_lr"] != m["label"]
    m["llm_wrong"] = m["pred_llm"] != m["label"]
    m = m[m["lr_wrong"] | m["llm_wrong"]].copy()
    m["set"] = np.select([m["lr_wrong"] & m["llm_wrong"], m["lr_wrong"]], ["both_wrong", "lr_only_wrong"],
                         "llm_only_wrong")
    # a mainstream article called conspiracy is a false positive, the reverse a false negative
    m["error_type"] = np.where(m["label"], "false negative", "false positive")
    return m.rename(columns={"pred_lr": "lr_pred", "pred_llm": "llm_pred", "label": "true_label"}) \
            .drop(columns=["lr_wrong", "llm_wrong"]).reset_index(drop=True)


def select_for_coding(sets: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Code ALL both-wrong documents; for the other two sets all of them, or a seeded sample of `cap`."""
    rng = np.random.default_rng(seed)
    parts = []
    for name in SETS:
        sub = sets[sets["set"] == name].sort_values("doc_id")
        if name != "both_wrong" and len(sub) > cap:
            sub = sub.iloc[np.sort(rng.choice(len(sub), size=cap, replace=False))]
        parts.append(sub)
    return pd.concat(parts).reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, seed: int, by: str = "set") -> pd.DataFrame:
    """`n` rows, spread over the groups of `by` in proportion to their size (largest remainder), seeded.

    Every group gets at least one row if n allows it, so a small set (LLM-only errors) is not lost.
    """
    if n > len(df):
        raise ValueError(f"cannot sample {n} rows from {len(df)}")
    rng = np.random.default_rng(seed)
    sizes = df[by].value_counts().sort_index()
    raw = sizes / sizes.sum() * n
    take = np.minimum(np.maximum(np.floor(raw).astype(int), 1), sizes)
    order = (raw - np.floor(raw)).sort_values(ascending=False).index
    while take.sum() < n:                    # hand out the remaining rows by largest remainder
        for g in order:
            if take.sum() < n and take[g] < sizes[g]:
                take[g] += 1
    while take.sum() > n:                    # the "at least one" floor may overshoot
        g = take.idxmax()
        take[g] -= 1
    picks = []
    for g, k in take.items():
        idx = df.index[df[by] == g].to_numpy()
        picks.extend(rng.choice(idx, size=int(k), replace=False))
    return df.loc[sorted(picks)].reset_index(drop=True)


def balanced_website_sample(df: pd.DataFrame, n: int, cap: int, seed: int) -> pd.DataFrame:
    """`n` rows, half per class, at most `cap` per website, taken one website at a time (seeded)."""
    rng = np.random.default_rng(seed)
    picks = []
    for label in (False, True):
        by_site = {}
        for site, sub in df[df["label"] == label].groupby("website"):
            by_site[site] = list(rng.permutation(sub.index.to_numpy())[:cap])
        sites = list(rng.permutation(sorted(by_site)))
        chosen = []
        while len(chosen) < n // 2 and any(by_site.values()):
            for site in sites:                    # one article per website per round
                if by_site[site] and len(chosen) < n // 2:
                    chosen.append(by_site[site].pop())
        if len(chosen) < n // 2:
            raise ValueError(f"only {len(chosen)} rows for label {label} with at most {cap} per website")
        picks.extend(chosen)
    return df.loc[sorted(picks)].reset_index(drop=True)


# ---------------------------------------------------------------- stance coding

def parse_stance(raw: str) -> dict:
    """Strict JSON -> {stance, justification}. Anything else is stance 'unreadable', never guessed."""
    import re

    bad = {"stance": UNREADABLE, "justification": ""}
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    if not text.startswith("{") and "{" in text:      # tolerate reasoning before the JSON, but only a full object
        text = text[text.index("{"):]
    try:
        obj = json.loads(text)
        stance =str(obj["stance"]).strip().lower()
    except (ValueError, KeyError, TypeError):
        return bad
    if stance not in STANCES:
        return bad
    just = scrub_justification(obj.get("justification", ""))
    return {"stance": stance, "justification": just}


def stance_client(cfg: dict | None = None):
    """Real client for the stance model. No key -> RuntimeError (no silent fallback)."""
    import os

    cfg = cfg or stance_config()
    env = llm.api_key_env_name(cfg)
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"No cached stance answer and no API key: set {env} or run with a complete cache "
                           f"({cfg['cache_file']}). Refusing to guess stances.")
    return llm.AnthropicClient(key, cfg)


def code_stances(df: pd.DataFrame, client=None, cache_file: Path | None = None,
                 cfg: dict | None = None) -> pd.DataFrame:
    """Stance and justification per row of `df` (columns doc_id, txt), asking the client only on a cache miss."""
    cfg = cfg or stance_config()
    path = cache_file or stance_cache_path(cfg)
    key = stance_key(cfg)
    cache = llm.read_cache(path)
    rows = []
    for doc_id, txt in zip(df["doc_id"], df["txt"]):
        rec = cache.get((doc_id, key))
        if rec is None:
            if client is None:
                client = stance_client(cfg)
            article, _ = llm.truncate_words(txt, cfg["max_input_words"])
            answer = parse_stance(client.complete(STANCE_PROMPT, article).text)
            rec = {"doc_id": doc_id, "prompt_hash": key, "answer": answer}   # no article text in the cache
            llm.append_cache(path, rec)
            cache[(doc_id, key)] = rec
        rows.append({"doc_id": doc_id, "claude_stance": rec["answer"]["stance"],
                     "claude_justification": scrub_justification(rec["answer"]["justification"])})
    return pd.DataFrame(rows)


def missing_stances(doc_ids, cache_file: Path | None = None, cfg: dict | None = None) -> int:
    cache = llm.read_cache(cache_file or stance_cache_path(cfg))
    key = stance_key(cfg)
    return int(sum((d, key) not in cache for d in doc_ids))


# ---------------------------------------------------------------- the rule

def classify_error(true_label: bool, stance: str) -> str:
    """Does the website label describe the text, or did the model get it wrong?"""
    if stance not in STANCES:
        raise ValueError(f"unknown stance {stance!r}: drop unreadable answers before applying the rule")
    if true_label:
        # conspiracy site: only an article that endorses a claim fits its label
        return MODEL_WRONG if stance == "endorses" else LABEL_PROBLEM
    # mainstream site: an article that endorses a claim does not fit its label
    # (a mainstream article that debunks theories still counts as mainstream)
    return LABEL_PROBLEM if stance == "endorses" else MODEL_WRONG


PLAIN_LABELS = {"lr_only_wrong": "Only the logistic regression is wrong", "llm_only_wrong": "Only the LLM is wrong",
                "both_wrong": "Both are wrong", "all": "All errors",
                "false positive": "Mainstream article called conspiracy",
                "false negative": "Conspiracy-site article called mainstream",
                "endorses": "Article endorses a conspiracy claim", "rejects_or_debunks": "Article debunks claims",
                "reports_neutrally": "Article reports claims neutrally", "no_claim": "Article makes no claim about 9/11",
                "not_sampled": "Not coded"}


def error_classification_table(coded: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Counts by group x error type, with plain labels: all errors, coded, and the coded ones per verdict."""
    rows = []
    for s in (*SETS, "all"):
        for et in ("false positive", "false negative", "all"):
            sel = lambda d: d[((d["set"] == s) | (s == "all")) & ((d["error_type"] == et) | (et == "all"))]  # noqa: E731
            c = sel(coded)
            rows.append({"Group": PLAIN_LABELS[s], "Error type": PLAIN_LABELS.get(et, "All errors"),
                         "Errors": len(sel(totals)), "Coded": len(c),
                         "Model wrong": int((c["verdict"] == MODEL_WRONG).sum()),
                         "Label does not describe the text": int((c["verdict"] == LABEL_PROBLEM).sum()),
                         PLAIN_LABELS["not_sampled"]: len(sel(totals)) - len(c),
                         **{PLAIN_LABELS[st]: int((c["claude_stance"] == st).sum()) for st in STANCES}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- human check

def write_check_sample(sample: pd.DataFrame, path: Path) -> bool:
    """Write the blind CSV for the team; False if the file already exists."""
    if path.is_file():
        return False                  # the team's tool is built on these doc_ids: never redraw them
    out = sample.copy()
    out["human_stance"] = ""          # left empty on purpose: the team fills it in
    path.parent.mkdir(parents=True, exist_ok=True)
    out[CHECK_COLUMNS].to_csv(path, index=False)
    return True


def check_sheet_layout() -> pd.DataFrame:
    """Plain description of the blind check sheet, one row per column (no article text, no model answer)."""
    stances = ", ".join(s.replace("_", " ") for s in STANCES)
    return pd.DataFrame({"Column": CHECK_COLUMNS,
                         "What it holds": ["Identifier of one validation article", "The website that published it",
                                           f"Left empty: the reader ticks one of {stances}"],
                         "Filled by": ["Prepared", "Prepared", "Each member, without seeing the model's answer"]})


def write_claude_answers(sample: pd.DataFrame, path: Path) -> None:
    """Claude's stance for the same documents, kept in a separate file so the team does not see it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample[["doc_id", "claude_stance", "claude_justification"]].to_csv(path, index=False)


def plain_counts(coded: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Three rows for the report: articles per group, how many were coded, and the label-problem share."""
    rows = []
    for s in SETS:
        c = coded[coded["set"] == s]
        problems = int((c["verdict"] == LABEL_PROBLEM).sum())
        rows.append({"Group": PLAIN_LABELS[s], "Articles": int((totals["set"] == s).sum()), "Coded": len(c),
                     "Label problems": problems, "Label-problem share": round(problems / len(c), 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- examples

BOILERPLATE = re.compile(r"click|cached|subscribe|cookie|posted by|copyright|share this|read more|sign up|"
                         r"follow us|advertis|newsletter|all rights|menu|http|www\.|@|\d{4}", re.IGNORECASE)
EXAMPLE_CASES = {"lr_only_wrong": "Only Claude was right", "llm_only_wrong": "Only the logistic regression was right",
                 "both_wrong": "Both models were wrong", "label_problem": "The label does not fit the text",
                 "false_positive": PLAIN_LABELS["false positive"], "false_negative": PLAIN_LABELS["false negative"]}
STANCE_WORDS = {"endorses": "endorses a claim", "rejects_or_debunks": "debunks claims",
                "reports_neutrally": "reports neutrally", "no_claim": "makes no claim"}


def plain_examples(examples: pd.DataFrame) -> pd.DataFrame:
    """The example table with plain column names, only the columns the report discusses."""
    return pd.DataFrame({"Case": examples["cell"].map(EXAMPLE_CASES), "Website type": examples["true_label"],
                         "Logistic regression said": examples["lr_prediction"],
                         "Claude said": examples["llm_prediction"],
                         "What the article does": examples["stance"].map(STANCE_WORDS),
                         "Excerpt": examples["excerpt"]})


def find_excerpt(text: str, words: int = EXCERPT_WORDS, scan: int = 400) -> str | None:
    """First sentence-start window of words that is no boilerplate and names nobody."""
    tokens = str(text).split()[:scan]
    for size in (words, 25, 20):                      # try shorter windows if no long one is clean
        for start in range(0, max(len(tokens) - size, 0) + 1):
            if start and not re.search(r"[.!?]$", tokens[start - 1]):
                continue                              # start only at the beginning of a sentence
            window = " ".join(tokens[start:start + size])
            if len(window.split()) < size or BOILERPLATE.search(window):
                continue
            if scrub_justification(window, size) == window:
                return window + " ..."
    return None


def pick_examples(sets: pd.DataFrame, texts: pd.Series, seed: int) -> pd.DataFrame:
    """One example per case, drawn with a seeded shuffle from documents with a clean excerpt."""
    # `sets` holds the coded documents; `texts` maps doc_id to the article text; each document is used once
    rng = np.random.default_rng(seed)
    pools = {"lr_only_wrong": sets[sets["set"] == "lr_only_wrong"], "llm_only_wrong": sets[sets["set"] == "llm_only_wrong"],
             "both_wrong": sets[sets["set"] == "both_wrong"],
             "label_problem": sets[sets.get("verdict", pd.Series("", index=sets.index)) == LABEL_PROBLEM],
             "false_positive": sets[sets["error_type"] == "false positive"],
             "false_negative": sets[sets["error_type"] == "false negative"]}
    rows, used = [], set()
    for case, pool in pools.items():
        pool = pool[~pool["doc_id"].isin(used)].sort_values("doc_id")
        for k in rng.permutation(len(pool)):
            row = pool.iloc[k]
            excerpt = find_excerpt(texts[row["doc_id"]])
            if excerpt:
                used.add(row["doc_id"])
                rows.append({"cell": case, "doc_id": row["doc_id"], "true_label": row["true_label"],
                             "lr_pred": row["lr_pred"], "llm_pred": row["llm_pred"],
                             "stance": row["claude_stance"], "excerpt": excerpt})
                break
    return pd.DataFrame(rows)
