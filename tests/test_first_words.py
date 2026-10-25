"""All logistic regressions must read the same 3,000-word text as the LLM."""
from fml.data import first_words
from fml.paths import project_config


def test_long_text_is_cut_to_the_limit():
    text = " ".join(f"w{i}" for i in range(5000))
    assert len(first_words(text, 3000).split()) == 3000
    assert first_words(text, 3000).split()[-1] == "w2999"      # the START of the article is kept


def test_short_text_is_unchanged():
    text = "a short  text\nwith odd spacing"
    assert first_words(text, 3000) == text


def test_limit_matches_the_llm_limit():
    assert project_config()["llm_classifier"]["max_input_words"] == 3000


def test_c_grid_is_the_shared_grid():
    assert project_config()["lr"]["c_grid"] == [0.01, 0.1, 1, 10, 100]
    assert project_config()["lr"]["class_weight"] == "balanced"
