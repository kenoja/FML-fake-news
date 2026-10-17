"""Why: off-topic articles must be removed during cleaning, before any model, using the pattern from the
config; and removing documents must never touch the frozen split file."""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

import fml.data as data
from fml import clean
from fml.split import load_split, split_path


def tiny_corpus() -> pd.DataFrame:
    rows = [("d1", "Attack on 9/11", "text about it", "a"), ("d2", "Cooking", "pasta and sauce", "b"),
            ("d3", "Budget", "the Pentagon announced a budget", "c"), ("d4", "Heart", "symptoms and causes", "d"),
            ("d5", "WTC", "nothing", "e")]
    return (pd.DataFrame(rows, columns=["doc_id", "title", "txt", "website"])
            .assign(label=[True, False, False, False, True],
                    subcorpus=["conspiracy", "mainstream", "mainstream", "mainstream", "conspiracy"]))


def test_pattern_is_read_from_config(monkeypatch):
    """Changing the config pattern changes the filter: the pattern is not hard-coded in code."""
    monkeypatch.setattr(data, "project_config", lambda: {"data": {"topic_filter_pattern": "pasta"}})
    assert data.on_topic_mask(tiny_corpus()).tolist() == [False, True, False, False, False]


def test_clean_documents_removes_off_topic_documents_and_counts_them():
    raw = tiny_corpus()
    split = pd.DataFrame({"doc_id": raw["doc_id"], "split": ["train"] * 5})
    kept, log, info = clean.clean_documents(raw, split, metadata=[])
    assert kept["doc_id"].tolist() == ["d1", "d3", "d5"]          # cooking and heart documents are gone
    assert (info["before"], info["after_topic"], info["after_duplicates"]) == (5, 3, 3)
    assert info["counts"]["after_topic"] == {"documents": 3, "conspiracy": 2, "mainstream": 1}
    assert log["removed"].tolist() == [0, 2, 0]


@pytest.mark.needs_data
def test_real_cleaning_numbers_and_split_file_untouched():
    before = hashlib.sha256(split_path().read_bytes()).hexdigest()
    kept, _, info = clean.clean_documents()
    assert (info["before"], info["after_topic"]) == (3146, 2608)
    assert info["counts"]["after_topic"] == {"documents": 2608, "conspiracy": 777, "mainstream": 1831}
    assert info["after_duplicates"] == len(kept) <= 2608
    split = load_split().set_index("doc_id")["split"]
    assert set(kept["doc_id"]) <= set(split.index)             # every kept doc still has its original split
    assert hashlib.sha256(split_path().read_bytes()).hexdigest() == before
