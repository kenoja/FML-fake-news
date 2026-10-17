"""Zero-shot LLM classifier (Claude Haiku 5.5) with a cache that never holds article text.

The LLM answers the SOURCE question: would this article come from a conspiracy-oriented or a
mainstream website? It receives exactly the cleaned text the logistic regression receives
(column `txt` of `pipeline.analysis_frame()`), cut to the same word limit for every document.
Every answer is cached by (doc_id, prompt_hash), so reruns need no API key and no network.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd

from .paths import REPO_ROOT, project_config

LABELS = ("conspiracy", "mainstream")
PARSE_ERROR = "parse_error"
REASON_MAX_WORDS = 25
MAX_RETRIES = 4
BACKOFF_SECONDS = 2.0

_ANSWER_FORMAT = ('Answer with strict JSON only, no other text: '
                  '{"label": "conspiracy" or "mainstream", "confidence": a number from 0 to 1, '
                  '"reason": "at most 25 words"}')

PROMPT_V1 = (
    "Read the article below. Would it have been published on a conspiracy-oriented website or on a "
    "mainstream website?\n" + _ANSWER_FORMAT
)

PROMPT_V2 = (
    "You classify web articles by the kind of website that published them.\n"
    "A conspiracy-oriented website mainly promotes alternative explanations of events, claims that "
    "official accounts are lies or cover-ups, and distrusts mainstream institutions and media.\n"
    "A mainstream news website is an established news, reference or magazine outlet that reports "
    "events through sourced reporting and accepts official findings unless evidence contradicts them.\n"
    "Would the article below have been published on a conspiracy-oriented website or on a mainstream "
    "news website?\n" + _ANSWER_FORMAT
)

PROMPT_V3 = (
    "You classify web articles by the kind of website that published them: conspiracy-oriented or "
    "mainstream news.\n"
    "Work step by step before you answer: (1) note how the article treats sources, evidence and "
    "official accounts; (2) note its tone and how certain it sounds; (3) decide which kind of "
    "website would publish it.\n"
    "Do not decide by topic alone: mainstream sites also write about 9/11 and its conspiracy "
    "theories, and conspiracy sites also publish other stories. Judge how the article is written "
    "and argued.\n" + _ANSWER_FORMAT
)

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2, "v3": PROMPT_V3}


def llm_config() -> dict:
    return project_config()["llm_classifier"]


def check_variants(variants, max_variants: int | None = None) -> list[str]:
    """At most `max_prompt_variants` prompts may be tried: more would be tuning on validation."""
    variants = list(variants)
    limit = max_variants if max_variants is not None else llm_config()["max_prompt_variants"]
    if len(variants) > limit:
        raise ValueError(f"{len(variants)} prompt variants requested, config allows {limit}")
    unknown = [v for v in variants if v not in PROMPTS]
    if unknown:
        raise KeyError(f"unknown prompt variant(s) {unknown}; known: {sorted(PROMPTS)}")
    return variants


def prompt_hash(prompt: str, model: str, temperature: float, max_words: int, pass_id: int) -> str:
    """Cache key part. Any change to prompt, model, temperature, truncation or pass gives a new key,
    so a stale answer can never be mistaken for one from the current setting."""
    payload = json.dumps([prompt, model, temperature, max_words, pass_id])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def truncate_words(text: str, max_words: int) -> tuple[str, bool]:
    """Keep the first `max_words` whitespace-separated words; also say whether anything was cut."""
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]), True


def parse_response(raw: str) -> dict:
    """Strict JSON -> {label, confidence, reason}. Anything else is label 'parse_error', never guessed."""
    bad = {"label": PARSE_ERROR, "confidence": None, "reason": ""}
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        obj = json.loads(text)
        label = str(obj["label"]).strip().lower()
        conf = float(obj["confidence"])
    except (ValueError, KeyError, TypeError):
        return bad
    if label not in LABELS or not 0.0 <= conf <= 1.0:
        return bad
    reason = " ".join(str(obj.get("reason", "")).split()[:REASON_MAX_WORDS])
    return {"label": label, "confidence": conf, "reason": reason}


# ---------------------------------------------------------------- clients

@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


class Client(Protocol):
    def complete(self, prompt: str, article: str) -> Completion: ...


class FakeClient:
    """Deterministic, offline stand-in for tests: the answer depends only on the text's hash."""

    def __init__(self, fixed: str | None = None):
        self.calls = 0
        self.fixed = fixed

    def complete(self, prompt: str, article: str) -> Completion:
        self.calls += 1
        if self.fixed is not None:
            raw = self.fixed
        else:
            odd = int(hashlib.sha256(article.encode("utf-8")).hexdigest(), 16) % 2
            raw = json.dumps({"label": LABELS[odd], "confidence": 0.7, "reason": "fake answer"})
        return Completion(raw, len(prompt.split()) + len(article.split()), len(raw.split()))


def api_key_env_name(cfg: dict | None = None) -> str:
    """The env var holding the key. Anything starting with ANTHROPIC is refused: Claude Code would
    bill the agent's own work to that key."""
    name = (cfg or llm_config())["api_key_env"]
    if name.upper().startswith("ANTHROPIC"):
        raise ValueError(f"api_key_env {name!r} is not allowed; use a dedicated variable such as "
                         "LLM_CLASSIFIER_API_KEY")
    return name


class AnthropicClient:
    """Thin wrapper around the anthropic SDK with retries and exponential backoff."""

    def __init__(self, api_key: str, cfg: dict | None = None):
        import anthropic

        cfg = cfg or llm_config()
        self.model = cfg["model"]
        # Temperature 0 is the configured intent, but claude-haiku-5-5 rejects the parameter ("`temperature` is
        # deprecated for this model", HTTP 400) and the SDK has no argument for it, so it cannot be sent.
        # It stays in the cache key; decisions.md records the deviation and the stability pass measures it.
        self.temperature = cfg["temperature"]
        self.max_tokens = cfg.get("max_output_tokens", 150)
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0)

    def complete(self, prompt: str, article: str) -> Completion:
        delay = BACKOFF_SECONDS
        for attempt in range(MAX_RETRIES + 1):
            try:
                msg = self._client.messages.create(
                    model=self.model, max_tokens=self.max_tokens,
                    system=prompt, messages=[{"role": "user", "content": f"ARTICLE:\n{article}"}])
                text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
                return Completion(text, msg.usage.input_tokens, msg.usage.output_tokens)
            except (self._anthropic.RateLimitError, self._anthropic.APIConnectionError,
                    self._anthropic.InternalServerError):
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")


def make_client(cfg: dict | None = None) -> Client:
    """Real client from the configured env var. No key -> RuntimeError (no silent fallback)."""
    cfg = cfg or llm_config()
    env = api_key_env_name(cfg)
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(
            f"No cached LLM answer and no API key: set {env} or run with a complete cache "
            f"({cfg['cache_file']}). Refusing to guess answers.")
    return AnthropicClient(key, cfg)


# ---------------------------------------------------------------- cache

def cache_path(cfg: dict | None = None) -> Path:
    return REPO_ROOT / (cfg or llm_config())["cache_file"]


def read_cache(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            # First answer wins: if two runs wrote the same key (parallel runs), numbers must not
            # depend on which line a reader happens to see last.
            out.setdefault((rec["doc_id"], rec["prompt_hash"]), rec)
    return out


def dedupe_cache(path: Path) -> int:
    """Rewrite the cache keeping the first record per (doc_id, prompt_hash); returns lines dropped."""
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    kept = list(read_cache(path).values())
    path.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    return len(lines) - len(kept)


def append_cache(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- classification

def cache_key(variant: str, pass_id: int = 1, cfg: dict | None = None) -> str:
    cfg = cfg or llm_config()
    return prompt_hash(PROMPTS[variant], cfg["model"], cfg["temperature"], cfg["max_input_words"], pass_id)


def missing_count(df: pd.DataFrame, variant: str, pass_id: int = 1, cache_file: Path | None = None,
                  cfg: dict | None = None) -> int:
    """How many documents of `df` have no cached answer for this variant and pass (API calls needed)."""
    cfg = cfg or llm_config()
    cache = read_cache(cache_file or cache_path(cfg))
    key = cache_key(variant, pass_id, cfg)
    return int(sum((d, key) not in cache for d in df["doc_id"]))


def cached_records(df: pd.DataFrame, variant: str, pass_id: int = 1, cache_file: Path | None = None,
                   cfg: dict | None = None) -> list[dict]:
    """Cache records (tokens, reason, ...) of the documents in `df`; all must be cached."""
    cfg = cfg or llm_config()
    cache = read_cache(cache_file or cache_path(cfg))
    key = cache_key(variant, pass_id, cfg)
    return [cache[(d, key)] for d in df["doc_id"]]


def classify(df: pd.DataFrame, variant: str = "v1", pass_id: int = 1, client: Client | None = None,
             cache_file: Path | None = None, cfg: dict | None = None) -> pd.DataFrame:
    """Classify every row of `df` (columns `doc_id`, `txt`) and return one row per document.

    Order: cache lookup first; only on a miss is a client needed. Without an injected client and
    without the API key a miss raises RuntimeError. `pass_id` keeps the stability pass (2) apart
    from pass 1 in the cache. Output columns: doc_id, label, confidence, reason, truncated, cached.
    """
    cfg = cfg or llm_config()
    check_variants([variant])
    prompt = PROMPTS[variant]
    max_words = cfg["max_input_words"]
    key = prompt_hash(prompt, cfg["model"], cfg["temperature"], max_words, pass_id)
    path = cache_file or cache_path(cfg)
    cache = read_cache(path)
    rows = []
    for doc_id, txt in zip(df["doc_id"], df["txt"]):
        article, cut = truncate_words(txt, max_words)
        rec = cache.get((doc_id, key))
        cached = rec is not None
        if rec is None:
            if client is None:
                client = make_client(cfg)
            done = client.complete(prompt, article)
            rec = {"doc_id": doc_id, "prompt_hash": key, **parse_response(done.text), "pass": pass_id,
                   "variant": variant, "truncated": cut, "input_tokens": done.input_tokens,
                   "output_tokens": done.output_tokens, "at": datetime.now(timezone.utc).isoformat()}
            append_cache(path, rec)
            cache[(doc_id, key)] = rec
        rows.append({"doc_id": doc_id, "label": rec["label"], "confidence": rec["confidence"],
                     "reason": rec["reason"], "truncated": bool(rec["truncated"]), "cached": cached})
    return pd.DataFrame(rows)


def summarise(result: pd.DataFrame) -> dict:
    """Counts that must be reported: parse errors and truncated documents are never hidden."""
    return {"n_docs": int(len(result)),
            "parse_errors": int((result["label"] == PARSE_ERROR).sum()),
            "truncated_docs": int(result["truncated"].sum()),
            "max_input_words": llm_config()["max_input_words"]}


def usage_table(cache_file: Path | None = None, out_dir: Path | None = None) -> pd.DataFrame:
    """Token totals per variant and pass from the cache -> results/tables/llm_usage.csv.

    Cost is not computed here: prices change, so the report quotes the current price list.
    """
    from .results import write_table

    recs = list(read_cache(cache_file or cache_path()).values())
    cols = ["variant", "pass", "input_tokens", "output_tokens", "label"]
    df = pd.DataFrame(recs, columns=cols)
    if df.empty:
        table = pd.DataFrame(columns=["variant", "pass", "calls", "input_tokens", "output_tokens", "parse_errors"])
    else:
        table = (df.groupby(["variant", "pass"])
                   .agg(calls=("label", "size"), input_tokens=("input_tokens", "sum"),
                        output_tokens=("output_tokens", "sum"),
                        parse_errors=("label", lambda s: int((s == PARSE_ERROR).sum())))
                   .reset_index())
    write_table("llm_usage", table, out_dir)
    return table
