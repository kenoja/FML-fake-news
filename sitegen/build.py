"""Render content/sections/*.md into a notebook-style site in site/.

Section files are markdown with a small YAML header:

    ---
    title: Splitting by website
    part: Evaluation            # which part of the assignment this explains
    hypothesis: A random split would overstate accuracy because sites leak into the test set.
    ---
    ## Goal and hypothesis
    ...
    <!-- cell: src/fml/split.py#make_split -->            code cell from a function/class
    <!-- cell: src/fml/split.py:10-42 -->                 code cell from a line range
    <!-- output: results/figures/split_sizes.png | Documents per split -->
    <!-- output: results/tables/split_summary.csv -->
    <!-- output: results/metrics.json#baseline.words_only -->
    ## Explanation ... ## What could go wrong ... ## Technical infrastructure ...

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


@dataclass
class Section:
    slug: str
    title: str
    part: str
    hypothesis: str
    html: str
    discussion_term: str


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


# ---- sections -----------------------------------------------------------------------

def render_section(path: Path, counter: list[int], cycle: int, assets: Path, warnings: list[str]) -> Section:
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

    body = MARKER.sub(replace, body)
    out = markdown.markdown(body, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    for i, block in enumerate(blocks):
        out = out.replace(f"<p>@@BLOCK{i}@@</p>", block).replace(f"@@BLOCK{i}@@", block)
    slug = path.stem
    return Section(slug=slug, title=meta.get("title", slug), part=meta.get("part", ""),
                   hypothesis=meta.get("hypothesis", ""), html=out,
                   discussion_term=f"cycle-{cycle:02d}/{slug}")


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


def build(cycle: int, strict: bool = False, out: Path = OUT, content: Path = CONTENT) -> BuildResult:
    settings = site_settings()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    assets = out / "assets"
    result = BuildResult()
    counter = [0]
    for path in sorted((content / "sections").glob("*.md")):
        result.sections.append(render_section(path, counter, cycle, assets, result.warnings))

    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                      autoescape=select_autoescape(["html"]))
    repo_url = settings.get("repo_url", "")
    common = dict(settings=settings, cycle=cycle, repo_url=repo_url,
                  built=datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"),
                  report_available=(REPO / "reports" / "report.md").exists())
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
