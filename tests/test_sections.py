"""The walkthrough is the deliverable the team reads. A section with a missing heading, a code cell
that points at a function that was renamed, or an output file that was never written would show up
as a broken box on the site (or not at all), and an overlong quote would put article text in a
public repo. These tests fail on each of those."""
import re
from pathlib import Path

import pytest
import yaml

from sitegen import build as sitegen

REPO = Path(__file__).resolve().parents[1]
SECTIONS = REPO / "content" / "sections"
SLUGS = ["00-how-to-use", "01-question-and-hypotheses", "02-data-and-cleaning", "03-split-by-website",
         "04-word-counts", "05-tf-idf", "06-feature-engineering", "07-llm", "08-results",
         "09-where-they-fail", "10-limitations", "11-conclusion", "12-how-we-used-ai", "99-glossary"]
REWRITTEN = ["04-word-counts", "05-tf-idf"]       # slugs already in the four-part format
NO_CODE = {"00-how-to-use", "99-glossary"}
HEADINGS = ["## Explained simply", "## The code", "## Report", "## Check yourself"]
KEYS = ["title", "part", "hypothesis"]


def problems(sections_dir: Path, repo: Path = REPO, slugs: list[str] = SLUGS,
             max_words: int = 50, max_per_section: int = 6) -> list[str]:
    """Every defect found in the section files, as readable strings (empty list = all good)."""
    out: list[str] = []
    for slug in slugs:
        path = sections_dir / f"{slug}.md"
        if not path.is_file():
            out.append(f"{slug}: file missing")
            continue
        text = path.read_text(encoding="utf-8")
        front = sitegen.FRONT.match(text)
        meta = yaml.safe_load(front.group(1)) if front else {}
        out += [f"{slug}: front matter key '{k}' missing" for k in KEYS if not (meta or {}).get(k)]
        body = text[front.end():] if front else text
        out += [f"{slug}: heading '{h}' missing" for h in HEADINGS
                if not re.search(rf"^{re.escape(h)}\b", body, re.M)]
        cells = sitegen.MARKER.findall(body)
        if slug not in NO_CODE and not any(kind == "cell" for kind, _ in cells):
            out.append(f"{slug}: no code cell")
        for kind, spec in cells:
            try:
                if kind == "cell":
                    sitegen._code_from_ref(spec.strip())
                else:
                    ref = spec.partition("|")[0].strip()
                    file, _, key = ref.partition("#")
                    target = repo / file
                    if not target.is_file():
                        raise FileNotFoundError(file)
                    if target.suffix == ".json" and key:
                        import json
                        sitegen._json_pick(json.loads(target.read_text(encoding="utf-8")), key)
            except Exception as exc:  # noqa: BLE001 - any failure means the reference is broken
                out.append(f"{slug}: {kind} '{spec}' does not resolve ({type(exc).__name__}: {exc})")
        quotes = excerpts(body)
        if len(quotes) > max_per_section:
            out.append(f"{slug}: {len(quotes)} excerpts, limit {max_per_section}")
        out += [f"{slug}: excerpt of {len(q.split())} words, limit {max_words}"
                for q in quotes if len(q.split()) > max_words]
    return out


def excerpts(body: str) -> list[str]:
    """Quoted blocks ('> ...' lines); a block ends at the first line that is not a quote or is empty."""
    blocks, current = [], []
    for line in body.splitlines() + [""]:
        stripped = line.lstrip()
        if stripped.startswith(">") and stripped[1:].strip():
            current.append(stripped[1:].strip())
        else:
            if current:
                blocks.append(" ".join(current))
            current = []
    return blocks


def test_all_sections_exist_and_every_reference_resolves():
    # The same checks the strict site build makes, run in one place with readable messages.
    assert problems(SECTIONS, slugs=REWRITTEN) == []


def test_excerpt_limits_come_from_the_config():
    cfg = yaml.safe_load((REPO / "config" / "project.yaml").read_text(encoding="utf-8"))["excerpts"]
    assert cfg == {"max_words": 50, "max_per_section": 6}


def write(tmp_path, body, slug="01-demo", front="title: T\npart: P\nhypothesis: H\n"):
    d = tmp_path / "sections"
    d.mkdir(exist_ok=True)
    (d / f"{slug}.md").write_text(f"---\n{front}---\n{body}", encoding="utf-8")
    return d


GOOD = ("## Explained simply\nx\n## The code\n<!-- cell: src/fml/split.py#make_split -->\n## Report\nx\n"
        "## Check yourself\nQ: a?\nA: b\n")


def check(d, **kw):
    return problems(d, slugs=["01-demo"], **kw)


def test_fixture_section_that_is_correct_passes(tmp_path):
    assert check(write(tmp_path, GOOD)) == []


def test_a_cell_that_points_at_a_missing_function_fails(tmp_path):
    bad = GOOD.replace("make_split", "no_such_function")
    assert any("does not resolve" in p for p in check(write(tmp_path, bad)))


def test_a_cell_with_a_missing_file_fails(tmp_path):
    bad = GOOD.replace("src/fml/split.py#make_split", "src/fml/nope.py:1-5")
    assert any("does not resolve" in p for p in check(write(tmp_path, bad)))


def test_an_output_file_that_does_not_exist_fails(tmp_path):
    bad = GOOD + "<!-- output: results/tables/never_written.csv -->\n"
    assert any("never_written" in p for p in check(write(tmp_path, bad)))


def test_a_missing_json_key_fails(tmp_path):
    bad = GOOD + "<!-- output: results/metrics.json#no.such.key -->\n"
    assert any("no.such.key" in p for p in check(write(tmp_path, bad)))


def test_a_missing_heading_or_front_matter_key_or_code_cell_fails(tmp_path):
    assert any("heading" in p for p in check(write(tmp_path, GOOD.replace("## Report", "## Notes"))))
    assert any("'hypothesis'" in p for p in check(write(tmp_path, GOOD, front="title: T\npart: P\n")))
    no_cell = GOOD.replace("<!-- cell: src/fml/split.py#make_split -->\n", "")
    assert any("no code cell" in p for p in check(write(tmp_path, no_cell)))


def test_overlong_or_too_many_excerpts_fail(tmp_path):
    long_quote = "> " + " ".join(["word"] * 51) + "\n"
    assert any("limit 50" in p for p in check(write(tmp_path, GOOD + long_quote)))
    many = "".join(f"> short quote {i}\n\n" for i in range(7))
    assert any("limit 6" in p for p in check(write(tmp_path, GOOD + many)))
    assert check(write(tmp_path, GOOD + "> short quote\n")) == []


def test_missing_section_file_fails(tmp_path):
    assert problems(tmp_path / "sections_missing", slugs=["01-demo"]) == ["01-demo: file missing"]


@pytest.mark.parametrize("slug", REWRITTEN)
def test_slug_files_are_present(slug):
    assert (SECTIONS / f"{slug}.md").is_file()
