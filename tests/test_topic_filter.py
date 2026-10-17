"""Why: the off-topic filter must only DROP rows. If it changed a kept document's split, or if the pattern
were hard-coded somewhere, results after the filter would no longer be comparable with the first run."""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

import fml.data as data
import fml.pipeline as pipeline
import fml.topic as topic
from fml.split import load_split, split_path


def tiny_corpus() -> pd.DataFrame:
    rows = [("d1", "Attack on 9/11", "text about it", "a"), ("d2", "Cooking", "pasta and sauce", "b"),
            ("d3", "Budget", "the Pentagon announced a budget", "c"), ("d4", "Heart", "symptoms and causes", "d"),
            ("d5", "WTC", "nothing", "e")]
    return pd.DataFrame(rows, columns=["doc_id", "title", "txt", "website"]).assign(label=[True, False, False, False, True])


def test_pattern_is_read_from_config(monkeypatch):
    """Changing the config pattern changes the filter: the pattern is not hard-coded in code."""
    df = tiny_corpus()
    monkeypatch.setattr(data, "project_config", lambda: {"data": {"topic_filter_pattern": "pasta"}})
    assert data.on_topic_mask(df).tolist() == [False, True, False, False, False]
    monkeypatch.setattr(topic, "project_config", lambda: {"data": {"topic_filter_pattern": "pasta|budget"}})
    assert topic.pattern_terms() == ["pasta", "budget"]


def test_term_diagnostic_agrees_with_the_filter():
    """The per-term table is only trustworthy if OR-ing the terms equals the real filter."""
    df = tiny_corpus()
    assert topic.term_matches(df).any(axis=1).tolist() == data.on_topic_mask(df).tolist()
    assert topic.term_matches(df)["pentagon"].tolist() == [False, False, True, False, False]


def test_filtered_set_is_a_subset_and_keeps_rows_unchanged():
    raw = tiny_corpus()
    cleaned = raw.assign(txt=raw["txt"].str.upper())          # cleaning changes text, not membership
    out = pipeline.filter_to_topic(cleaned, raw)
    assert out["doc_id"].tolist() == ["d1", "d3", "d5"]
    assert set(out["doc_id"]) <= set(cleaned["doc_id"])
    pd.testing.assert_frame_equal(out, cleaned[cleaned["doc_id"].isin(out["doc_id"])].reset_index(drop=True))


def test_short_title_respects_excerpt_limit():
    assert len(topic.short_title(" ".join(["w"] * 80)).split()) == topic.MAX_EXCERPT_WORDS
    assert topic.short_title("") == "(no title)"


@pytest.mark.needs_data
def test_real_filter_keeps_split_assignment_and_split_file_untouched():
    before = hashlib.sha256(split_path().read_bytes()).hexdigest()
    all_df = pipeline.cleaned_frame(keep_all=True)
    kept = pipeline.analysis_frame()
    assert set(kept["doc_id"]) < set(all_df["doc_id"])
    split = load_split().set_index("doc_id")["split"]
    assert set(kept["doc_id"]) <= set(split.index)             # every kept doc still has its original split
    assert hashlib.sha256(split_path().read_bytes()).hexdigest() == before
