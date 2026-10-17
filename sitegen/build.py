"""Render content/sections/*.md into a notebook-style site in site/, and assemble the report.

Section files are markdown with a small YAML header and four parts, in this order:

    ---
    title: Splitting by website
    part: Evaluation            # which part of the assignment this explains
    hypothesis: (optional) what we expected before running it
    ---
    ## Explained simply         plain-language explanation and design choices (stands alone)
    ## The code                 code cells and their outputs, with short text in between
    <!-- cell: src/fml/split.py#make_split -->            code cell from a function/class
    <!-- cell: src/fml/split.py:10-42 -->                 code cell from a line range
    <!-- output: results/figures/split_sizes.png | Documents per split -->
    <!-- output: results/tables/split_summary.csv -->
    <!-- output: results/metrics.json#split.sizes -->
    ## Report                   the text that goes into reports/report.md, word for word
    ## Check yourself           Q: question / A: answer pairs, shown with the answer hidden

reports/report.md is assembled from the "## Report" parts (plus content/references.md), so the
site and the report can never disagree. Glossary entries (### Term in 99-glossary.md, optional
"Also: other name, ..." line) are linked automatically where the term first appears in a section.
Code is read from the repository at build time, so the site never shows code that differs from
what actually ran.
"""
from __future__ import annotations

import ast
import csv
import html
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import markdown
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename, TextLexer

REPO = Path(__file__).resolve().parents[1]
CONTENT = REPO / "content"
OUT = REPO / "site"
MARKER = re.compile(r"<!--\s*(cell|output):\s*(.+?)\s*-->")
FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
H2 = re.compile(r"^## +(.+?)\s*$", re.M)
PARTS = {"explained simply": "part-explained", "the code": "part-code", "report": "part-report",
         "check yourself": "part-check"}
NO_LINK_TAGS = {"a", "code", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "script", "style", "figcaption"}
GLOSSARY = "99-glossary"


@dataclass
class Section:
    slug: str
    title: str
    part: str
    hypothesis: str
    html: str
    discussion_term: str
    report_md: str = ""


@dataclass
class BuildResult:
    sections: list[Section] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---- code and output cells ----------------------------------------------------------

def _code_from_ref(ref: str) -> tuple[str, str, int]:
    """Return (source, label, first line number) for 'path#Symbol' or 'path:10-40'."""
    if "#" in ref:
        path, symbol = ref.split("#", 1)
        src = (REPO / path).read_text(encoding="utf-8")
        tree = ast.parse(src)
        node: ast.AST | None = tree
        for name in symbol.split("."):
            node = next((n for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                         and n.name == name), None)
            if node is None:
                raise KeyError(f"{symbol} not found in {path}")
        start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
        lines = src.splitlines()[start - 1: node.end_lineno]
        return "\n".join(lines), f"{symbol} in {path}", start
    path, _, rng = ref.partition(":")
    lines = (REPO / path).read_text(encoding="utf-8").splitlines()
    a, b = (int(x) for x in rng.split("-")) if rng else (1, len(lines))
    return "\n".join(lines[a - 1: b]), f"{path}, lines {a} to {b}", a


def _code_cell(ref: str, n: int) -> str:
    source, label, first = _code_from_ref(ref)
    path = ref.split("#")[0].split(":")[0]
    try:
        lexer = get_lexer_for_filename(path)
    except Exception:
        lexer = TextLexer()
    body = highlight(source, lexer, HtmlFormatter(nowrap=True))
    n_lines = source.count("\n") + 1
    long_cls = " is-long" if n_lines > 28 else ""
    toggle = (f'<button class="expand" type="button" aria-expanded="false">Show all {n_lines} lines</button>'
              if long_cls else "")
    return (f'<figure class="cell code{long_cls}"><div class="prompt">In&nbsp;[{n}]:</div>'
            f'<div class="cell-body"><figcaption><a href="{{repo_url}}/blob/main/{path}#L{first}">{html.escape(label)}</a></figcaption>'
            f'<pre><code>{body}</code></pre>{toggle}</div></figure>')


def _json_pick(data, dotted: str):
    for key in [k for k in dotted.split(".") if k]:
        data = data[key]
    return data


def _render_json(value) -> str:
    if isinstance(value, dict) and all(not isinstance(v, (dict, list)) for v in value.values()):
        rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{_fmt(v)}</td></tr>" for k, v in value.items())
        return f'<table class="kv">{rows}</table>'
    return f"<pre>{html.escape(json.dumps(value, indent=2, ensure_ascii=False))}</pre>"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return html.escape(str(v))


def _output_cell(spec: str, n: int, assets: Path) -> str:
    ref, _, caption = (s.strip() for s in spec.partition("|"))
    path, _, key = ref.partition("#")
    src = REPO / path
    if not src.is_file():
        raise FileNotFoundError(path)
    suffix = src.suffix.lower()
    if suffix in {".png", ".svg", ".jpg", ".jpeg"}:
        assets.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, assets / src.name)
        inner = f'<img src="assets/{src.name}" alt="{html.escape(caption or src.stem)}" loading="lazy">'
    elif suffix == ".csv":
        with src.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))
        head = "".join(f"<th>{html.escape(c)}</th>" for c in rows[0])
        body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows[1:31])
        more = f'<p class="note">{len(rows) - 31} more rows in <code>{path}</code></p>' if len(rows) > 31 else ""
        inner = f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{more}'
    elif suffix == ".json":
        inner = _render_json(_json_pick(json.loads(src.read_text(encoding="utf-8")), key))
    else:
        inner = f"<pre>{html.escape(src.read_text(encoding='utf-8')[:4000])}</pre>"
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return (f'<figure class="cell output"><div class="prompt">Out[{n}]:</div>'
            f'<div class="cell-body">{inner}{cap}</div></figure>')


def _error_cell(kind: str, spec: str, exc: Exception) -> str:
    return (f'<figure class="cell broken"><div class="prompt">!</div><div class="cell-body">'
            f"<p>This {kind} cell could not be built: <code>{html.escape(spec)}</code> — {html.escape(str(exc))}</p>"
            f"</div></figure>")


# ---- check-yourself questions, glossary links, report text ------------------------------

def _inline(text: str) -> str:
    return re.sub(r"^<p>|</p>$", "", markdown.markdown(text.strip()))


def _questions(md: str) -> str:
    """Turn 'Q: ...' / 'A: ...' pairs into <details> blocks so the answer stays hidden until clicked."""
    out, q, a = [], None, None

    def flush():
        if q is not None:
            answer = markdown.markdown(" ".join(a or ["(no answer written yet)"]))
            out.append(f'\n<details class="check"><summary>{_inline(" ".join(q))}</summary>{answer}</details>\n')

    for line in md.splitlines():
        if line.startswith("Q:"):
            flush()
            q, a = [line[2:].strip()], None
        elif line.startswith("A:") and q is not None:
            a = [line[2:].strip()]
        elif line.strip() and q is not None:
            (a if a is not None else q).append(line.strip())
        elif not line.strip() and q is not None and a is not None:
            flush()
            q, a = None, None
        elif q is None:
            out.append(line)
    flush()
    return "\n".join(out)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def glossary_terms(content: Path = CONTENT) -> list[tuple[str, str, str]]:
    """(term, anchor, first sentence) for every '### Term' in the glossary, aliases included."""
    path = content / "sections" / f"{GLOSSARY}.md"
    if not path.is_file():
        return []
    terms = []
    for m in re.finditer(r"^### +(.+?)\s*\n(.*?)(?=^### |\Z)", path.read_text(encoding="utf-8"), re.M | re.S):
        name, body = m.group(1).strip(), m.group(2)
        aliases = re.search(r"^Also:\s*(.+)$", body, re.M)
        definition = re.sub(r"^Also:.*$", "", body, flags=re.M).strip()
        tip = re.split(r"(?<=[.!?])\s", re.sub(r"[*_`]", "", definition), maxsplit=1)[0] if definition else ""
        for t in [name] + ([x.strip() for x in aliases.group(1).split(",")] if aliases else []):
            if t:
                terms.append((t, "g-" + _slugify(name), tip))
    return sorted(terms, key=lambda t: -len(t[0]))  # longest first: "F1 score" before "F1"


def link_terms(page: str, terms: list[tuple[str, str, str]]) -> str:
    """Link the first use of each glossary term in a section's prose (never inside code or links)."""
    done, saved, skip, out = set(), [], 0, []
    for token in re.split(r"(<[^>]+>)", page):
        tag = re.match(r"<(/?)([a-zA-Z0-9]+)", token)
        if tag:
            if tag.group(2).lower() in NO_LINK_TAGS and not token.endswith("/>"):
                skip += -1 if tag.group(1) else 1
            out.append(token)
            continue
        if skip <= 0:
            for term, anchor, tip in terms:
                if anchor in done:
                    continue
                m = re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", token, re.I)
                if m:
                    saved.append(f'<a class="gl" href="#{anchor}" title="{html.escape(tip)}">{m.group(0)}</a>')
                    token = f"{token[:m.start()]}\x00{len(saved) - 1}\x00{token[m.end():]}"
                    done.add(anchor)
        out.append(token)
    return re.sub(r"\x00(\d+)\x00", lambda m: saved[int(m.group(1))], "".join(out))


def _report_output(spec: str) -> str:
    """An output marker inside the report text, as plain markdown (figure, table or values)."""
    ref, _, caption = (x.strip() for x in spec.partition("|"))
    path, _, key = ref.partition("#")
    src = REPO / path
    if src.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg"}:
        return f"![{caption or src.stem}](../{path})"
    if src.suffix.lower() == ".csv":
        with src.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))[:31]
        table = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
        table += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join(table) + (f"\n\n*{caption}*" if caption else "")
    value = _json_pick(json.loads(src.read_text(encoding="utf-8")), key)
    if isinstance(value, dict) and all(not isinstance(v, (dict, list)) for v in value.values()):
        rows = [f"| {k} | {_fmt(v)} |" for k, v in value.items()]
        return "\n".join(["| | |", "|---|---|"] + rows) + (f"\n\n*{caption}*" if caption else "")
    raise ValueError(f"{ref}: only flat key/value JSON can go into the report text")


def _report_markdown(md: str, name: str, warnings: list[str]) -> str:
    def out(match: re.Match) -> str:
        if match.group(1) == "cell":
            warnings.append(f"{name}: code cells belong under '## The code', not in the report text")
            return ""
        try:
            return "\n\n" + _report_output(match.group(2)) + "\n\n"
        except Exception as exc:
            warnings.append(f"{name}: report output {match.group(2)}: {exc}")
            return ""
    return MARKER.sub(out, md).strip()


# ---- sections -----------------------------------------------------------------------

def _split_parts(body: str) -> list[tuple[str, str]]:
    """[(heading or '', text)] in file order, split at '## ' headings."""
    pieces = H2.split(body)
    parts = [("", pieces[0])]
    parts += [(pieces[i], pieces[i + 1]) for i in range(1, len(pieces) - 1, 2)]
    return parts


def render_section(path: Path, counter: list[int], cycle: int, assets: Path, warnings: list[str],
                   terms: list[tuple[str, str, str]] | None = None) -> Section:
    raw = path.read_text(encoding="utf-8")
    m = FRONT.match(raw)
    meta = yaml.safe_load(m.group(1)) if m else {}
    body = raw[m.end():] if m else raw
    blocks: list[str] = []

    def replace(match: re.Match) -> str:
        kind, spec = match.group(1), match.group(2)
        try:
            if kind == "cell":
                counter[0] += 1
                blocks.append(_code_cell(spec, counter[0]))
            else:
                blocks.append(_output_cell(spec, counter[0], assets))
        except Exception as exc:  # show it on the page AND fail the strict build
            warnings.append(f"{path.name}: {kind} {spec}: {exc}")
            blocks.append(_error_cell(kind, spec, exc))
        return f"\n\n@@BLOCK{len(blocks) - 1}@@\n\n"

    extensions = ["tables", "fenced_code", "sane_lists", "attr_list"]
    html_parts, report_md = [], ""
    for heading, text in _split_parts(body):
        key = heading.lower()
        if key == "report":
            report_md = _report_markdown(text, path.name, warnings)
        if key == "check yourself":
            text = _questions(text)
        rendered = markdown.markdown(MARKER.sub(replace, text), extensions=extensions)
        if key in PARTS:
            label = "Report text" if key == "report" else heading
            html_parts.append(f'<div class="part {PARTS[key]}"><h3 class="part-label">{html.escape(label)}</h3>'
                              f"{rendered}</div>")
        elif heading:
            html_parts.append(f"<h2>{html.escape(heading)}</h2>{rendered}")
        else:
            html_parts.append(rendered)
    out = "\n".join(html_parts)
    slug = path.stem
    if slug == GLOSSARY:  # give every entry an anchor so links from other sections land on it
        out = re.sub(r"<h3>(.+?)</h3>", lambda h: f'<h3 id="g-{_slugify(re.sub("<[^>]+>", "", h.group(1)))}">'
                     f"{h.group(1)}</h3>", out)
    elif terms:
        out = link_terms(out, terms)
    for i, block in enumerate(blocks):
        out = out.replace(f"<p>@@BLOCK{i}@@</p>", block).replace(f"@@BLOCK{i}@@", block)
    return Section(slug=slug, title=meta.get("title", slug), part=meta.get("part", ""),
                   hypothesis=meta.get("hypothesis", ""), html=out,
                   discussion_term=f"cycle-{cycle:02d}/{slug}", report_md=report_md)


def assemble_report(sections: list[Section], content: Path = CONTENT, title: str = "") -> str:
    """The report = every section's '## Report' part, in order, plus content/references.md."""
    parts = [f"# {title}"] if title else []
    parts += [f"## {s.title}\n\n{s.report_md}" for s in sections if s.report_md]
    refs = content / "references.md"
    if refs.is_file():
        text = refs.read_text(encoding="utf-8").strip()
        parts.append(text if text.startswith("#") else f"## References\n\n{text}")
    return "\n\n".join(parts) + "\n"


def report_text(content: Path = CONTENT) -> str:
    """Assemble the report without building the site (used by the report-length gate)."""
    import tempfile
    warnings: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        sections = [render_section(p, [0], 0, Path(tmp), warnings)
                    for p in sorted((content / "sections").glob("*.md"))]
    return assemble_report(sections, content) if any(s.report_md for s in sections) else ""


def site_settings() -> dict:
    """config/site.yaml, with giscus ids optionally supplied by the Pages workflow (env vars)."""
    import os
    cfg_path = REPO / "config" / "site.yaml"
    settings = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    g = settings.setdefault("giscus", {})
    for key, env in (("repo_id", "GISCUS_REPO_ID"), ("category_id", "GISCUS_CATEGORY_ID")):
        if os.environ.get(env):
            g[key] = os.environ[env]
    return settings


def build(cycle: int, strict: bool = False, out: Path = OUT, content: Path = CONTENT,
          report_out: Path | None = REPO / "reports" / "report.md") -> BuildResult:
    settings = site_settings()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    assets = out / "assets"
    result = BuildResult()
    counter = [0]
    terms = glossary_terms(content)
    for path in sorted((content / "sections").glob("*.md")):
        result.sections.append(render_section(path, counter, cycle, assets, result.warnings, terms))
    if report_out is not None and any(s.report_md for s in result.sections):
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(assemble_report(result.sections, content, settings.get("report_title", "")),
                              encoding="utf-8")

    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                      autoescape=select_autoescape(["html"]))
    repo_url = settings.get("repo_url", "")
    common = dict(settings=settings, cycle=cycle, repo_url=repo_url,
                  built=datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"),
                  report_available=bool(report_out and report_out.exists()))
    for s in result.sections:
        s.html = s.html.replace("{repo_url}", repo_url)
    (out / "index.html").write_text(env.get_template("notebook.html.j2").render(sections=result.sections, **common),
                                    encoding="utf-8")
    guide = content / "guide.md"
    if guide.is_file():
        guide_html = markdown.markdown(guide.read_text(encoding="utf-8"), extensions=["tables", "sane_lists", "attr_list"])
        (out / "guide.html").write_text(env.get_template("page.html.j2").render(title="How to join in", body=guide_html, **common),
                                        encoding="utf-8")
    (out / ".nojekyll").write_text("")
    for w in result.warnings:
        print(f"warning: {w}")
    if strict and result.warnings:
        raise SystemExit(f"{len(result.warnings)} cell(s) could not be built")
    print(f"built {len(result.sections)} sections into {out}/")
    return result
