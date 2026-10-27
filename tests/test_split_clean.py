"""Why these tests exist: a website in two splits, a re-made split or a leftover site name would each
quietly make the final numbers look better than they are."""
import pandas as pd
import pytest

from fml import clean
from fml.split import SPLITS, SplitExists, assign_websites, make_split, split_summary


def synthetic(n_con=8, n_main=16, per_site=10):
    rows = []
    for k in range(n_con + n_main):
        con = k < n_con
        for j in range(per_site + k % 4):
            rows.append({"doc_id": f"{'C' if con else 'M'}{k}-{j}", "website": f"site{k}.com",
                         "subcorpus": "conspiracy" if con else "mainstream", "txt": f"text {k} {j}"})
    return pd.DataFrame(rows)


def test_no_website_in_two_splits(tmp_path):
    df = synthetic()
    out = make_split(df, tmp_path / "s.csv")
    per_site = df.merge(out, on="doc_id").groupby("website")["split"].nunique()
    assert (per_site == 1).all()          # a shared website would let a model recognise the site
    assert set(out["split"]) == set(SPLITS)
    assert len(out) == len(df)


def test_stratification_roughly_holds(tmp_path):
    # few whole websites -> coarse steps, so the tolerance is wide; the real split is checked below
    df = synthetic(n_con=20, n_main=40)
    out = make_split(df, tmp_path / "s.csv")
    t = split_summary(df, out)
    for c in ("conspiracy", "mainstream"):
        part = t[t["class"] == c]
        share = part.set_index("split")["documents"] / part["documents"].sum()
        assert abs(share["train"] - 0.70) < 0.1 and abs(share["test"] - 0.15) < 0.1
        assert (part["websites"] > 0).all()


def test_make_split_refuses_to_overwrite(tmp_path):
    df = synthetic()
    path = tmp_path / "s.csv"
    make_split(df, path)
    before = path.read_bytes()
    with pytest.raises(SplitExists):
        make_split(df, path)
    assert path.read_bytes() == before     # re-splitting after results exist would invalidate comparisons


def test_split_is_deterministic(tmp_path):
    df = synthetic()
    a = make_split(df, tmp_path / "a.csv")
    b = make_split(df, tmp_path / "b.csv")
    assert a.equals(b)


def test_assign_websites_keeps_sites_whole():
    sizes = pd.Series({f"w{i}": 5 + i for i in range(20)})
    out = assign_websites(sizes, {"train": 0.7, "validation": 0.15, "test": 0.15}, 1)
    assert set(out) == set(sizes.index) and set(out.values()) <= set(SPLITS)


def test_cleaning_removes_planted_url_website_name_and_duplicate():
    base = "The collapse of the towers was studied by engineers for many years and reports followed."
    df = pd.DataFrame({
        "doc_id": ["a", "b", "c"],
        "website": ["truthsite.org"] * 2 + ["newsy.com"],
        "subcorpus": ["conspiracy", "conspiracy", "mainstream"],
        "txt": [base + " Read more at https://truthsite.org/story?id=1 or mail editor@example.org. truthsite.org says so.",
                base.upper() + "  ",                       # same text after normalisation -> exact duplicate
                "A different article about newsy.com coverage of another subject entirely."],
    })
    df.loc[1, "txt"] = df.loc[0, "txt"].upper()
    cleaned, log, _ = clean.clean_corpus(df, metadata=[{"website": "other-site.net"}])
    assert list(cleaned["doc_id"]) == ["a", "c"]                      # duplicate gone (first kept)
    all_text = " ".join(cleaned["txt"]).lower()
    for leftover in ("http", "truthsite", "newsy", "@"):
        assert leftover not in all_text                                # URL, e-mail, website names
    assert log.iloc[0]["documents_before"] == 3 and log.iloc[-1]["documents_after"] == 2


def test_boilerplate_line_repeated_in_three_documents_is_removed():
    rows = [{"doc_id": str(i), "website": "s.com", "subcorpus": "mainstream",
             "txt": f"Unique body number {i} with words.\nSubscribe to our newsletter today"} for i in range(3)]
    rows.append({"doc_id": "x", "website": "o.com", "subcorpus": "mainstream", "txt": "Subscribe to our newsletter today"})
    df = pd.DataFrame(rows)
    out = clean.strip_boilerplate(df, clean.boilerplate_lines(df))
    assert "newsletter" not in " ".join(out.iloc[:3]).lower()
    assert "newsletter" in out.iloc[3].lower()                         # only repeated within ONE website


def test_near_duplicates_are_found():
    a = " ".join(f"word{i}" for i in range(60))
    df = pd.DataFrame({"doc_id": ["a", "b", "c"], "website": ["x", "y", "z"],
                       "txt": [a, a + " extra", " ".join(f"other{i}" for i in range(60))]})
    pairs = clean.find_near_duplicates(df)
    assert len(pairs) == 1 and {pairs.iloc[0]["doc_id_a"], pairs.iloc[0]["doc_id_b"]} == {"a", "b"}


@pytest.mark.needs_data
def test_frozen_split_on_real_data(tmp_path):
    from fml.data import load_corpus
    from fml.split import load_split
    df, sp = load_corpus(), load_split()
    assert make_split(df, tmp_path / "again.csv").equals(sp)     # the code still reproduces the frozen file
    assert len(sp) == 3146 and set(sp["doc_id"]) == set(df["doc_id"])
    assert (df.merge(sp, on="doc_id").groupby("website")["split"].nunique() == 1).all()
    t = split_summary(df, sp)
    assert (t[t["class"] == "conspiracy"]["websites"] >= 3).all()


def pair_table(*pairs):
    return pd.DataFrame(pairs, columns=["doc_id_a", "doc_id_b"]).assign(cosine=1.0, same_website=False)


def test_near_duplicate_across_splits_drops_the_validation_or_test_copy():
    """A copy in train must stay: dropping it would remove training data, while a validation/test copy
    that also sits in train would flatter the model that memorised it."""
    split = pd.Series({"a": "train", "b": "validation", "c": "test", "d": "validation", "e": "test"})
    drops = clean.near_duplicate_drops(pair_table(("a", "b"), ("c", "a"), ("d", "e")), split)
    assert set(drops) == {"b", "c", "e"}                        # validation vs test: the test copy goes


def test_near_duplicate_inside_one_split_keeps_the_lower_doc_id():
    split = pd.Series({"d1": "train", "d2": "train", "d3": "test", "d4": "test"})
    drops = clean.near_duplicate_drops(pair_table(("d2", "d1"), ("d3", "d4")), split)
    assert set(drops) == {"d2", "d4"}                           # the choice must not depend on pair order


def test_site_concentration_counts_the_largest_conspiracy_sites():
    """If a few big sites dominate the conspiracy class, the model may learn those sites instead of the topic."""
    from fml.data_report import site_concentration

    sizes = {"a.com": 5, "b.com": 4, "c.com": 3, "d.com": 2}
    rows = [{"doc_id": f"{s}-{j}", "website": s, "subcorpus": "conspiracy"} for s, n in sizes.items() for j in range(n)]
    rows += [{"doc_id": f"m-{j}", "website": "m.com", "subcorpus": "mainstream"} for j in range(9)]
    df = pd.DataFrame(rows)
    train_sites = ["a.com", "c.com", "m.com"]
    split = pd.DataFrame({"doc_id": df["doc_id"], "split": ["train" if w in train_sites else "test" for w in df["website"]]})
    table, metrics = site_concentration(df, split)
    assert list(table["Website"]) == ["Website 1", "Website 2", "Website 3"]
    assert metrics["top3_sites_share"] == pytest.approx(12 / 14)
    assert metrics["top3_train_articles"] == 8                  # a.com (5) + c.com (3)
    assert "a.com" not in table.to_string()
