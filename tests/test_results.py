"""The results helpers: one metrics.json that stages fill independently, and fixed class colours."""
import json

import pytest

import matplotlib

matplotlib.use("Agg")

from fml import results  # noqa: E402


def test_write_metrics_merges_and_keeps_required_keys(tmp_path):
    path = tmp_path / "metrics.json"
    results.write_metrics("data", {"n": 1}, path=path)
    results.write_metrics("split", {"train": 2}, path=path)
    got = json.loads(path.read_text())
    assert set(results.METRIC_KEYS) <= set(got)
    # a later stage must not erase an earlier stage's numbers
    assert got["data"] == {"n": 1} and got["split"] == {"train": 2}


def test_metrics_top_level_keys_are_exactly_the_agreed_ones():
    assert results.METRIC_KEYS == ("data", "cleaning", "split", "word_counts", "tfidf", "tfidf_features",
                                   "llm", "comparison", "errors")


def test_top_words_break_ties_by_word():
    """Why: equal weights in a different order would change a table between two identical runs."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from fml.models import top_words

    model = LogisticRegression()
    model.coef_ = np.ones((1, 4))                                     # make every weight equal
    table = top_words(["dd", "bb", "cc", "aa"], model, k=4)
    assert table[table["direction"] == "conspiracy"]["word"].tolist() == ["aa", "bb", "cc", "dd"]
    assert table[table["direction"] == "mainstream"]["word"].tolist() == ["aa", "bb", "cc", "dd"]


def test_write_metrics_rejects_unknown_key(tmp_path):
    with pytest.raises(KeyError):
        results.write_metrics("typo", {}, path=tmp_path / "m.json")


def test_class_colours_are_the_agreed_ones():
    assert results.CONSPIRACY_COLOUR == "#8c3b76"
    assert results.MAINSTREAM_COLOUR == "#2b7a6a"


def test_savefig_writes_png_at_150_dpi(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2, 1))
    ax.bar([0, 1], [1, 2], color=[results.CONSPIRACY_COLOUR, results.MAINSTREAM_COLOUR])
    path = results.savefig(fig, "demo", directory=tmp_path)
    assert path.suffix == ".png" and path.stat().st_size > 0
    assert results.DPI == 150
    from PIL import Image  # installed with matplotlib; PNG metadata records the dpi

    with Image.open(path) as img:
        assert round(img.info["dpi"][0]) == 150


def test_write_table(tmp_path):
    import pandas as pd

    p = results.write_table("t", pd.DataFrame({"a": [1]}), directory=tmp_path)
    assert p.name == "t.csv" and p.read_text().startswith("a")


@pytest.mark.needs_data
def test_table_writing_is_byte_identical_across_runs(tmp_path, monkeypatch):
    """Tables are diffed in git; a rerun that reorders ties would show up as a change nobody made."""
    from fml import pipeline

    dirs = []
    for i in range(2):
        out = tmp_path / f"run{i}"
        monkeypatch.setattr(results, "results_dir", lambda out=out: out)
        pipeline.cleaning_stage()
        dirs.append(out / "tables")
    files = sorted(p.name for p in dirs[0].glob("*.csv"))
    assert files and files == sorted(p.name for p in dirs[1].glob("*.csv"))
    for name in files:
        assert (dirs[0] / name).read_bytes() == (dirs[1] / name).read_bytes(), name
