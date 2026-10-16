"""Feedback must reach the next cycle complete, sorted and anonymous: usernames never appear."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "collect_feedback", Path(__file__).resolve().parents[1] / "scripts" / "collect_feedback.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

MEMBERS = {"alice-gh": "Member A", "bob-gh": "Member B"}


def discussion(title, comments, thumbs):
    return {"title": title, "category": {"name": "Feedback"},
            "reactions": {"nodes": [{"user": {"login": u}} for u in thumbs]},
            "comments": {"nodes": [{"author": {"login": a}, "body": b, "createdAt": "",
                                    "replies": {"nodes": []}} for a, b in comments]}}


def test_comments_are_sorted_pseudonymised_and_signoffs_counted():
    fb = cf.collect(1, [
        discussion("cycle-01/03-split-by-website", [("alice-gh", "Unclear: why group by website?"),
                                                    ("bob-gh", "**Change:** add an example")], ["alice-gh"]),
        discussion("cycle-01/02-data-and-labels", [], ["alice-gh", "bob-gh"]),
        discussion("cycle-02/02-data-and-labels", [("bob-gh", "other cycle")], []),
    ], MEMBERS)
    items = fb["sections"]["03-split-by-website"]["items"]
    assert [(i["kind"], i["by"]) for i in items] == [("Unclear", "Member A"), ("Change", "Member B")]
    assert "cycle-02" not in str(fb)                               # other cycles are ignored
    md = cf.requirements_markdown(fb)
    assert "alice-gh" not in md and "bob-gh" not in md             # no usernames in the repo
    assert "| 02-data-and-labels | 👍 | 👍 | yes |" in md
    assert "1. **[Unclear]** (Member A) why group by website?" in md
