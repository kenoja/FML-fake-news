"""LLM classifier: the cache must make reruns free, fail loudly without a key, and never hold text."""
from __future__ import annotations

import pandas as pd
import pytest

from fml import llm

SENTINEL = "zebrafish quantum sentinel sentence that must never reach the cache file"
CFG = {"model": "claude-haiku-5-5", "api_key_env": "LLM_CLASSIFIER_API_KEY", "max_prompt_variants": 3,
       "max_input_words": 5, "max_output_tokens": 1000, "cache_file": "unused"}


def frame(*texts):
    return pd.DataFrame({"doc_id": [f"d{i}" for i in range(len(texts))], "txt": list(texts)})


def test_cache_hit_makes_zero_client_calls(tmp_path):
    cache = tmp_path / "c.jsonl"
    df = frame("a b c", "d e f")
    first = llm.FakeClient()
    llm.classify(df, client=first, cache_file=cache, cfg=CFG)
    assert first.calls == 2
    second = llm.FakeClient()
    llm.classify(df, client=second, cache_file=cache, cfg=CFG)
    assert second.calls == 0


def test_missing_key_and_cache_miss_raises_and_names_the_entry(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_CLASSIFIER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="doc_id d0") as error:
        llm.classify(frame("a b"), cache_file=tmp_path / "c.jsonl", cfg=CFG)
    assert "LLM_CLASSIFIER_API_KEY" in str(error.value)


def test_a_complete_cache_needs_no_key(tmp_path, monkeypatch):
    cache = tmp_path / "c.jsonl"
    df = frame("a b", "c d")
    llm.classify(df, client=llm.FakeClient(), cache_file=cache, cfg=CFG)
    monkeypatch.delenv("LLM_CLASSIFIER_API_KEY", raising=False)
    assert len(llm.classify(df, cache_file=cache, cfg=CFG)) == 2


def test_one_missing_entry_fails_even_when_the_rest_is_cached(tmp_path, monkeypatch):
    cache = tmp_path / "c.jsonl"
    llm.classify(frame("a b"), client=llm.FakeClient(), cache_file=cache, cfg=CFG)
    monkeypatch.delenv("LLM_CLASSIFIER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="doc_id d1"):
        llm.classify(frame("a b", "c d"), cache_file=cache, cfg=CFG)


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
def test_unreadable_answer_is_recorded(tmp_path, raw):
    out = llm.classify(frame("a b"), client=llm.FakeClient(fixed=raw), cache_file=tmp_path / "c.jsonl", cfg=CFG)
    assert out["label"].tolist() == [llm.UNREADABLE]


def test_fenced_json_is_parsed():
    got = llm.parse_response('```json\n{"label": "Conspiracy", "confidence": 0.9, "reason": "x"}\n```')
    assert got["label"] == "conspiracy" and got["confidence"] == 0.9


def test_repeat_run_is_cached_apart_from_the_first_run(tmp_path):
    cache = tmp_path / "c.jsonl"
    df = frame("a b")
    llm.classify(df, pass_id=1, client=llm.FakeClient(), cache_file=cache, cfg=CFG)
    second = llm.FakeClient()
    llm.classify(df, pass_id=2, client=second, cache_file=cache, cfg=CFG)
    assert second.calls == 1


def test_a_higher_output_limit_does_not_reuse_old_answers(tmp_path):
    """Answers cut off at a small limit must be asked again once the limit is raised."""
    cache = tmp_path / "c.jsonl"
    llm.classify(frame("a b"), client=llm.FakeClient(), cache_file=cache, cfg={**CFG, "max_output_tokens": 150})
    again = llm.FakeClient()
    llm.classify(frame("a b"), client=again, cache_file=cache, cfg=CFG)
    assert again.calls == 1


def test_at_most_three_variants():
    assert llm.check_variants(["v1", "v2", "v3"]) == ["v1", "v2", "v3"]
    with pytest.raises(ValueError):
        llm.check_variants(["v1", "v2", "v3", "v1"])
    with pytest.raises(ValueError):
        llm.check_variants(["v1", "v2"], max_variants=1)
    assert len(llm.PROMPTS) <= llm.llm_config()["max_prompt_variants"]


def test_long_articles_are_cut_and_reported(tmp_path):
    out = llm.classify(frame("a b c d e f g", "a b"), client=llm.FakeClient(), cache_file=tmp_path / "c.jsonl", cfg=CFG)
    assert out["truncated"].tolist() == [True, False]


def test_output_limit_is_large_and_no_temperature_is_set():
    assert llm.llm_config()["max_output_tokens"] >= 1000
    assert "temperature" not in llm.llm_config()
