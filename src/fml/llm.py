"""Zero-shot LLM classifier (Claude Haiku 5.5). Answers are cached by document id and prompt, never with text."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from .paths import REPO_ROOT, project_config

LABELS = ("conspiracy", "mainstream")
UNREADABLE = "unreadable"
REASON_MAX_WORDS = 25
MAX_RETRIES = 8
WAIT_SECONDS = 5.0
MAX_WAIT_SECONDS = 60.0
PARALLEL_CALLS = 4

ANSWER_FORMAT = ('Answer with strict JSON only, no other text: '
                 '{"label": "conspiracy" or "mainstream", "confidence": a number from 0 to 1, '
                 '"reason": "at most 25 words"}')

# prompt 1: just the question
PROMPT_V1 = (
    "Read the article below. Would it have been published on a conspiracy-oriented website or on a "
    "mainstream website?\n" + ANSWER_FORMAT
)

# prompt 2: the question plus a short definition of both website types
PROMPT_V2 = (
    "You classify web articles by the kind of website that published them.\n"
    "A conspiracy-oriented website mainly promotes alternative explanations of events, claims that "
    "official accounts are lies or cover-ups, and distrusts mainstream institutions and media.\n"
    "A mainstream news website is an established news, reference or magazine outlet that reports "
    "events through sourced reporting and accepts official findings unless evidence contradicts them.\n"
    "Would the article below have been published on a conspiracy-oriented website or on a mainstream "
    "news website?\n" + ANSWER_FORMAT
)

# prompt 3: step-by-step advice and a warning not to decide by topic alone
PROMPT_V3 = (
    "You classify web articles by the kind of website that published them: conspiracy-oriented or "
    "mainstream news.\n"
    "Work step by step before you answer: (1) note how the article treats sources, evidence and "
    "official accounts; (2) note its tone and how certain it sounds; (3) decide which kind of "
    "website would publish it.\n"
    "Do not decide by topic alone: mainstream sites also write about 9/11 and its conspiracy "
    "theories, and conspiracy sites also publish other stories. Judge how the article is written "
    "and argued.\n" + ANSWER_FORMAT
)

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2, "v3": PROMPT_V3}


def llm_config() -> dict:
    return project_config()["llm_classifier"]


def check_variants(variants, max_variants: int | None = None) -> list[str]:
    """Allow at most `max_prompt_variants` prompts: more would be tuning on validation."""
    variants = list(variants)
    limit = max_variants if max_variants is not None else llm_config()["max_prompt_variants"]
    if len(variants) > limit:
        raise ValueError(f"{len(variants)} prompt variants requested, config allows {limit}")
    unknown = [v for v in variants if v not in PROMPTS]
    if unknown:
        raise KeyError(f"unknown prompt variant(s) {unknown}; known: {sorted(PROMPTS)}")
    return variants


def prompt_hash(prompt: str, model: str, max_words: int, pass_id: int, max_tokens: int | None = None) -> str:
    """Cache key part: any change to prompt, model, word limit, output limit or pass gives a new key."""
    parts = [prompt, model, max_words, pass_id]
    if max_tokens is not None:
        parts.append(max_tokens)
    return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()[:16]


def cache_key(variant: str, pass_id: int = 1, cfg: dict | None = None) -> str:
    cfg = cfg or llm_config()
    return prompt_hash(PROMPTS[variant], cfg["model"], cfg["max_input_words"], pass_id, cfg["max_output_tokens"])


def truncate_words(text: str, max_words: int) -> tuple[str, bool]:
    """Keep the first `max_words` words; also say whether anything was cut."""
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]), True


def parse_response(raw: str) -> dict:
    """Turn the model's JSON answer into label, confidence and reason; anything else is 'unreadable'."""
    unreadable = {"label": UNREADABLE, "confidence": None, "reason": ""}
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())      # remove a code fence if there is one
    try:
        answer = json.loads(text)
        label = str(answer["label"]).strip().lower()
        confidence = float(answer["confidence"])
    except (ValueError, KeyError, TypeError):
        return unreadable
    if label not in LABELS or not 0.0 <= confidence <= 1.0:
        return unreadable
    reason = " ".join(str(answer.get("reason", "")).split()[:REASON_MAX_WORDS])
    return {"label": label, "confidence": confidence, "reason": reason}


# ---------------------------------------------------------------- clients

@dataclass
class Completion:
    text: str
    output_tokens: int


class Client(Protocol):
    def complete(self, prompt: str, article: str) -> Completion: ...


class FakeClient:
    """Offline stand-in for tests: the answer depends only on the article text."""

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
        return Completion(raw, len(raw.split()))


def api_key_env_name(cfg: dict | None = None) -> str:
    """Name of the env var with the key; ANTHROPIC_* names are refused (they would bill the agent's work)."""
    name = (cfg or llm_config())["api_key_env"]
    if name.upper().startswith("ANTHROPIC"):
        raise ValueError(f"api_key_env {name!r} is not allowed; use a dedicated variable such as "
                         "LLM_CLASSIFIER_API_KEY")
    return name


class AnthropicClient:
    """Sends one article to the model; waits and retries when the service is busy."""

    def __init__(self, api_key: str, cfg: dict | None = None):
        import anthropic

        cfg = cfg or llm_config()
        self.model = cfg["model"]
        self.max_tokens = cfg["max_output_tokens"]
        self.busy_errors = (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=0)

    def complete(self, prompt: str, article: str) -> Completion:
        wait = WAIT_SECONDS
        for attempt in range(MAX_RETRIES + 1):
            try:
                # no temperature is sent: the model has no temperature setting, so default sampling is used
                message = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens,
                    system=prompt, messages=[{"role": "user", "content": f"ARTICLE:\n{article}"}])
            except self.busy_errors:
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(wait)
                wait = min(wait * 2, MAX_WAIT_SECONDS)
                continue
            text = "".join(block.text for block in message.content if block.type == "text")
            return Completion(text, message.usage.output_tokens)
        raise AssertionError("unreachable")


def make_client(cfg: dict | None = None) -> Client:
    """Real client from the key in the environment; no key means an error, never a guess."""
    cfg = cfg or llm_config()
    env = api_key_env_name(cfg)
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"No cached LLM answer and no API key: set {env} or use a complete cache "
                           f"({cfg['cache_file']}). Refusing to guess answers.")
    return AnthropicClient(key, cfg)


# ---------------------------------------------------------------- cache

def cache_path(cfg: dict | None = None) -> Path:
    return REPO_ROOT / (cfg or llm_config())["cache_file"]


def read_cache(path: Path) -> dict[tuple[str, str], dict]:
    """All cached answers by (doc_id, prompt_hash); if a key appears twice the first line wins."""
    if not path.exists():
        return {}
    cache = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            cache.setdefault((record["doc_id"], record["prompt_hash"]), record)
    return cache


def append_cache(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------- classification

def classify(df: pd.DataFrame, variant: str = "v1", pass_id: int = 1, client: Client | None = None,
             cache_file: Path | None = None, cfg: dict | None = None) -> pd.DataFrame:
    """Label every row of `df` (columns doc_id, txt): from the cache first, from the API only on a miss."""
    cfg = cfg or llm_config()
    check_variants([variant])
    prompt = PROMPTS[variant]
    key = prompt_hash(prompt, cfg["model"], cfg["max_input_words"], pass_id, cfg["max_output_tokens"])
    path = cache_file or cache_path(cfg)
    cache = read_cache(path)
    articles = {}
    for doc_id, txt in zip(df["doc_id"], df["txt"]):
        articles[doc_id] = truncate_words(txt, cfg["max_input_words"])          # (text, was it cut?)
    missing = [doc_id for doc_id in articles if (doc_id, key) not in cache]

    if missing and client is None:
        if not os.environ.get(api_key_env_name(cfg)):
            raise RuntimeError(f"No cached answer for doc_id {missing[0]} with prompt {variant} (hash {key}) "
                               f"and {api_key_env_name(cfg)} is not set; {len(missing)} answers are missing")
        client = make_client(cfg)

    def ask(doc_id: str) -> dict:
        article, cut = articles[doc_id]
        done = client.complete(prompt, article)
        record = {"doc_id": doc_id, "prompt_hash": key, **parse_response(done.text), "variant": variant,
                  "pass": pass_id, "truncated": cut, "output_tokens": done.output_tokens}
        append_cache(path, record)
        return record

    with ThreadPoolExecutor(PARALLEL_CALLS) as pool:           # ask for several articles at once
        for record in pool.map(ask, missing):
            cache[(record["doc_id"], key)] = record

    rows = [{"doc_id": d, "label": cache[(d, key)]["label"], "truncated": bool(cache[(d, key)]["truncated"])}
            for d in articles]
    return pd.DataFrame(rows)
