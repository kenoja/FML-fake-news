"""The data the whole project rests on. If these numbers change, the data or the topic filter
changed, and every result, section and report number must be recomputed — so fail loudly."""
import pytest

from fml.data import load_corpus, summary


@pytest.mark.needs_data
def test_subset_matches_the_published_seed_counts():
    # LOCO paper, table 4: 939 conspiracy + 2,207 mainstream documents for "september 11 attack".
    s = summary(load_corpus())
    assert (s["documents"], s["conspiracy"], s["mainstream"]) == (3146, 939, 2207)
    assert (s["websites_conspiracy"], s["websites_mainstream"]) == (39, 79)


@pytest.mark.needs_data
def test_off_topic_filter_keeps_the_documented_number():
    # requirements/cycle-01.md and config/project.yaml state 2,608 on-topic documents (777 / 1,831).
    s = summary(load_corpus())
    assert (s["on_topic_documents"], s["on_topic_conspiracy"], s["on_topic_mainstream"]) == (2608, 777, 1831)


@pytest.mark.needs_data
def test_label_comes_from_the_website_not_the_text():
    # Every website carries exactly one label; that is why the split must be grouped by website.
    df = load_corpus()
    assert (df.groupby("website")["label"].nunique() == 1).all()
