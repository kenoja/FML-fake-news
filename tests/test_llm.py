"""LLM classifier: the cache must make reruns free, fail loudly without a key, and never hold text."""
from __future__ import annotations

import pandas as pd
import pytest

from fml import llm

SENTINEL = "zebrafish quantum sentinel sentence that must never reach the cache file"
CFG = {"model": "claude-haiku-5-5", "temperature": 0, "api_key_env": "LLM_CLASSIFIER_API_KEY",
       "max_prompt_variants": 3, "max_input_words": 5, "cache_file": "unused"}


def frame(*texts):
    return pd.DataFrame({"doc_id": [f"d{i}" for i in range(len(texts))], "txt": list(texts)})


def test_cache_hit_makes_zero_client_calls(tmp_path):
    cache = tmp_path / "c.jsonl"
    df = frame("a b c", "d e f")
    first = llm.FakeClient()
    llm.classify(df, client=first, cache_file=cache, cfg=CFG)
    assert first.calls == 2
    second = llm.FakeClient()
    out = llm.classify(df, client=second, cache_file=cache, cfg=CFG)
    assert second.calls == 0 and out["cached"].all()


def test_missing_key_and_cache_miss_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_CLASSIFIER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LLM_CLASSIFIER_API_KEY"):
        llm.classify(frame("a b"), cache_file=tmp_path / "c.jsonl", cfg=CFG)


def test_cache_never_contains_article_text(tmp_path):
    cache = tmp_path / "c.jsonl"
    llm.classify(frame(SENTINEL), client=llm.FakeClient(), cache_file=cache, cfg={**CFG, "max_input_words": 50})
    body = cache.read_text()
    assert "zebrafish" not in body and "sentinel" not in body
    assert '"doc_id"' in body and '"prompt_hash"' in body


def test_wrong_env_var_name_is_rejected():
    forbidden = "ANTHROPIC" + "_API_KEY"
    with pytest.raises(ValueError):
        llm.api_key_env_name({**CFG, "api_key_env": forbidden})
    assert llm.api_key_env_name(CFG) == "LLM_CLASSIFIER_API_KEY"
    assert llm.llm_config()["api_key_env"] == "LLM_CLASSIFIER_API_KEY"


@pytest.mark.parametrize("raw", ["not json", '{"label": "maybe", "confidence": 0.5}',
                                 '{"label": "conspiracy", "confidence": 3}', '{"label": "conspiracy"}'])
def test_parse_error_is_recorded_and_counted(tmp_path, raw):
    out = llm.classify(frame("a b"), client=llm.FakeClient(fixed=raw), cache_file=tmp_path / "c.jsonl", cfg=CFG)
    assert out["label"].tolist() == [llm.PARSE_ERROR]
    assert llm.summarise(out)["parse_errors"] == 1


def test_fenced_json_is_parsed():
    got = llm.parse_response('```json\n{"label": "Conspiracy", "confidence": 0.9, "reason": "x"}\n```')
    assert got["label"] == "conspiracy" and got["confidence"] == 0.9


def test_pass_1_and_pass_2_are_cached_separately(tmp_path):
    cache = tmp_path / "c.jsonl"
    df = frame("a b")
    assert llm.prompt_hash("p", "m", 0, 5, 1) != llm.prompt_hash("p", "m", 0, 5, 2)
    llm.classify(df, pass_id=1, client=llm.FakeClient(), cache_file=cache, cfg=CFG)
    second = llm.FakeClient()
    llm.classify(df, pass_id=2, client=second, cache_file=cache, cfg=CFG)
    assert second.calls == 1


def test_at_most_three_variants():
    assert llm.check_variants(["v1", "v2", "v3"]) == ["v1", "v2", "v3"]
    with pytest.raises(ValueError):
        llm.check_variants(["v1", "v2", "v3", "v1"])
    with pytest.raises(ValueError):
        llm.check_variants(["v1", "v2"], max_variants=1)
    assert len(llm.PROMPTS) <= llm.llm_config()["max_prompt_variants"]


def test_truncation_is_identical_and_reported(tmp_path):
    out = llm.classify(frame("a b c d e f g", "a b"), client=llm.FakeClient(), cache_file=tmp_path / "c.jsonl", cfg=CFG)
    assert out["truncated"].tolist() == [True, False]


def test_usage_table_totals(tmp_path):
    cache = tmp_path / "c.jsonl"
    llm.classify(frame("a b", "c d"), client=llm.FakeClient(), cache_file=cache, cfg=CFG)
    table = llm.usage_table(cache, tmp_path)
    assert table["calls"].iloc[0] == 2 and (tmp_path / "llm_usage.csv").exists()


def test_missing_count_reports_exact_gap(tmp_path):
    cache = tmp_path / "c.jsonl"
    df = frame("a b", "c d", "e f")
    assert llm.missing_count(df, "v1", 1, cache_file=cache, cfg=CFG) == 3
    llm.classify(df.iloc[:2], client=llm.FakeClient(), cache_file=cache, cfg=CFG)
    assert llm.missing_count(df, "v1", 1, cache_file=cache, cfg=CFG) == 1
    assert llm.missing_count(df, "v1", 2, cache_file=cache, cfg=CFG) == 3
    llm.classify(df, client=llm.FakeClient(), cache_file=cache, cfg=CFG)
    assert llm.missing_count(df, "v1", 1, cache_file=cache, cfg=CFG) == 0
    assert len(llm.cached_records(df, "v1", 1, cache_file=cache, cfg=CFG)) == 3


def test_duplicate_cache_lines_first_answer_wins_and_dedupe(tmp_path):
    """Two parallel runs may write the same key; numbers must not depend on which line is read."""
    import json

    cache = tmp_path / "c.jsonl"
    base = {"doc_id": "d0", "prompt_hash": "h", "confidence": 0.5, "reason": ""}
    cache.write_text(json.dumps({**base, "label": "conspiracy"}) + "\n" +
                     json.dumps({**base, "label": "mainstream"}) + "\n")
    assert llm.read_cache(cache)[("d0", "h")]["label"] == "conspiracy"
    assert llm.dedupe_cache(cache) == 1
    assert len(cache.read_text().splitlines()) == 1


def test_stage_error_names_exact_missing_counts():
    from fml import pipeline

    err = pipeline._llm_missing_error({"validation v1 pass 1": 7, "test pass 2": 0})
    assert "validation v1 pass 1: 7" in str(err) and "test pass 2" not in str(err)


SENS_CFG = {**CFG, "max_output_tokens": 150, "token_limit_sensitivity": {"max_output_tokens": 400, "n_docs": 4}}


def labelled(n=6):
    df = frame(*[f"word{i} a b" for i in range(n)])
    df["label"] = [i % 2 == 0 for i in range(n)]
    return df


def test_sensitivity_without_key_or_cache_is_not_run_not_guessed(tmp_path, monkeypatch):
    """No key and no pass-3 answers: the result says 'not run' instead of reusing the 150-token answers."""
    monkeypatch.delenv("LLM_CLASSIFIER_API_KEY", raising=False)
    cache = tmp_path / "c.jsonl"
    df = labelled()
    llm.classify(df, "v2", 1, client=llm.FakeClient(), cache_file=cache, cfg=SENS_CFG)
    out = llm.token_limit_sensitivity(df, "v2", cache_file=cache, cfg=SENS_CFG)
    assert out["status"] == "not run" and "LLM_CLASSIFIER_API_KEY" in out["reason"]


def test_sensitivity_has_own_cache_key_and_only_asks_the_subset(tmp_path):
    """Pass 3 must not collide with pass 1 or 2, otherwise the comparison would compare a run with itself."""
    cache = tmp_path / "c.jsonl"
    df = labelled()
    llm.classify(df, "v2", 1, client=llm.FakeClient(), cache_file=cache, cfg=SENS_CFG)
    client = llm.FakeClient()
    out = llm.token_limit_sensitivity(df, "v2", client=client, cache_file=cache, cfg=SENS_CFG)
    assert out["status"] == "run" and out["n_docs"] == 4 and client.calls == 4
    keys = {llm.cache_key("v2", p, SENS_CFG) for p in (1, 2, llm.SENSITIVITY_PASS)}
    assert len(keys) == 3
