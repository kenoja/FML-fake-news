"""The 'four methods at a glance' table: what each method reads, how it is trained, how long it takes, what it costs."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from .llm import cache_path, llm_config
from .paths import project_config
from .results import results_dir, write_table

GAP_LIMIT_SECONDS = 60       # a longer pause between two cached answers is not time spent waiting for the API


def llm_usage(rows: list[dict]) -> dict:
    """Average tokens and seconds per article, from the cached answers that recorded them."""
    counted = [r for r in rows if "input_tokens" in r]
    times = sorted(datetime.fromisoformat(r["at"]) for r in counted)
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    busy = sum(g for g in gaps if g < GAP_LIMIT_SECONDS)          # keep only the gaps inside one batch
    return {"answers": len(counted),
            "input_tokens": sum(r["input_tokens"] for r in counted) / len(counted),
            "output_tokens": sum(r["output_tokens"] for r in counted) / len(counted),
            "seconds": busy / len(counted)}


def llm_cost_per_1000(usage: dict, cfg: dict) -> float:
    """Price in US dollars of 1,000 articles at list price."""
    per_article = (usage["input_tokens"] * cfg["price_input_per_mtok_usd"]
                   + usage["output_tokens"] * cfg["price_output_per_mtok_usd"]) / 1_000_000
    return per_article * 1000


def rough_time(seconds: float) -> str:
    """Runtime in words, so the table does not change from one run to the next."""
    if seconds < 15:
        return "a few seconds"
    if seconds < 120:
        return "about a minute"
    return f"about {round(seconds / 60)} minutes"


def methods_table(info: dict, metrics: dict, usage: dict, cfg: dict, lr_cfg: dict) -> pd.DataFrame:
    """One row per method, filled from run_info.json, metrics.json and the answer cache."""
    seconds = info["stage_seconds"]
    reads = f"first {cfg['max_input_words']:,} words of each article"
    lr_settings = (f"C from {lr_cfg['c_grid']}, class weight {lr_cfg['class_weight']}, "
                   f"chosen on validation")
    lr_cost = "free (runs locally)"
    rows = [
        ("1. Word counts + logistic regression", "word counts", "yes", lr_settings,
         rough_time(seconds['word_counts']), lr_cost),
        ("2. TF-IDF + logistic regression", "TF-IDF weights", "yes", lr_settings,
         rough_time(seconds['tfidf']), lr_cost),
        ("3. TF-IDF + features + logistic regression", "TF-IDF weights, n-grams, certainty words, style",
         "yes", lr_settings, rough_time(seconds['tfidf_features']), lr_cost),
        (f"4. LLM ({metrics['llm']['model']})", "the article text itself, with a written question", "no",
         f"prompt {metrics['llm']['chosen_prompt']} of {len(metrics['llm']['variants'])}, default sampling",
         f"about {usage['seconds']:.1f} seconds per article",
         f"about ${llm_cost_per_1000(usage, cfg):.2f} per 1,000 articles"),
    ]
    table = pd.DataFrame(rows, columns=["Method", "Input", "Trained on our data", "Settings", "Runtime", "Cost"])
    table.insert(2, "Reads", reads)
    return table


def write_methods_overview() -> pd.DataFrame:
    """Write results/tables/methods_overview.csv (needs run_info.json, metrics.json and the answer cache)."""
    info = json.loads((results_dir() / "run_info.json").read_text(encoding="utf-8"))
    metrics = json.loads((results_dir() / "metrics.json").read_text(encoding="utf-8"))
    cache = [json.loads(line) for line in cache_path().read_text(encoding="utf-8").splitlines() if line.strip()]
    table = methods_table(info, metrics, llm_usage(cache), llm_config(), project_config()["lr"])
    write_table("methods_overview", table)
    return table
