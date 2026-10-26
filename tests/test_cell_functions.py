"""Tiny-input checks for the small functions shown as code cells on the site."""
import json

import pandas as pd
import pytest

from fml import clean, llm, pipeline
from fml.features.style import mattr, style_row


def test_mattr_of_a_b_a_b_window_2_is_all_distinct():
    # every 2-word span of "a b a b" holds two different words
    assert mattr(["a", "b", "a", "b"], window=2) == pytest.approx(1.0)


def test_style_row_counts_capitals_marks_and_repetition():
    row = style_row("FBI said no! Why? Said no.", window=2)
    assert len(row) == 6
    assert row[0] == pytest.approx(1 / 6)      # one of six words is in capitals
    assert row[1] > 0 and row[2] > 0           # one "!" and one "?"


def test_step_log_counts_what_each_step_removed():
    frame = pd.DataFrame({"txt": ["a b", "c", "d e f"], "label": [True, False, False]})
    log = clean.step_log([("all", frame), ("fewer", frame.iloc[:1])])
    assert log["removed"].tolist() == [0, 2]
    assert log["remaining"].tolist() == [3, 1]


def test_short_text_info_reports_shortest_article():
    raw = pd.DataFrame({"doc_id": ["x", "y"], "txt": ["one two", "one two three four"]})
    info = clean.short_text_info(raw, raw)
    assert info["shortest_words"] == 2
    assert info["documents_under_threshold"] == 2


def test_check_frozen_config_refuses_unfrozen_or_changed_models():
    with pytest.raises(PermissionError):
        pipeline.check_frozen_config("m", "abc", None)
    with pytest.raises(PermissionError):
        pipeline.check_frozen_config("m", "abc", {"config_hash": "other"})
    pipeline.check_frozen_config("m", "abc", {"config_hash": "abc"})


def test_log_test_access_appends_one_json_line(tmp_path):
    path = tmp_path / "log" / "access.jsonl"
    pipeline.log_test_access(path, "m", "abc", 7)
    entry = json.loads(path.read_text().strip())
    assert (entry["model_id"], entry["config_hash"], entry["n_docs"]) == ("m", "abc", 7)


def test_missing_answers_without_key_fail_loudly(monkeypatch):
    cfg = llm.llm_config()
    monkeypatch.delenv(llm.api_key_env_name(cfg), raising=False)
    with pytest.raises(RuntimeError):
        llm.client_for_missing(["d1"], "v1", "hash", cfg)
