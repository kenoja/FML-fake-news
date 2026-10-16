"""Everything the nodes do to the outside world: Claude Code calls, shell commands, git,
and the deterministic quality gates. Nodes decide *when*; this module knows *how*."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml

REPO = Path(__file__).resolve().parents[1]
USAGE_LIMIT_RE = re.compile(r"usage limit|rate[ _-]?limit|limit reached|out of (extra )?usage|quota|429", re.I)


def load_config(name: str) -> dict:
    return yaml.safe_load((REPO / "config" / f"{name}.yaml").read_text(encoding="utf-8"))


def prompt(name: str, **values) -> str:
    """Load agent/prompts/<name>.md and fill $placeholders (string.Template, so braces stay literal)."""
    text = (REPO / "agent" / "prompts" / f"{name}.md").read_text(encoding="utf-8")
    return Template(text).safe_substitute({k: str(v) for k, v in values.items()})


# --------------------------------------------------------------------------------------
# Claude Code
# --------------------------------------------------------------------------------------

class UsageLimitReached(RuntimeError):
    """The subscription (or credit) limit was hit. The runner pauses and a later run resumes."""


@dataclass
class ClaudeResult:
    text: str
    structured: dict | None
    cost_usd: float
    turns: int


EDIT_TOOLS = ["Read", "Edit", "Write", "Glob", "Grep",
              "Bash(python *)", "Bash(python3 *)", "Bash(pytest *)", "Bash(pip install *)",
              "Bash(fml *)", "Bash(ls *)", "Bash(mkdir *)", "Bash(wc *)", "Bash(head *)"]
READ_ONLY_TOOLS = ["Read", "Glob", "Grep", "Bash(ls *)", "Bash(wc *)"]


def run_claude(task_prompt: str, *, model: str, tools: list[str], json_schema: dict | None = None,
               max_turns: int = 40, timeout_s: int = 900, extra_dirs: list[Path] = ()) -> ClaudeResult:
    """Run one headless Claude Code session in the repo and return its final answer.

    Auth: inside an Anthropic cloud session/routine the CLI uses the account's subscription.
    ANTHROPIC_API_KEY is removed from the child environment on purpose: if it were set, the CLI
    would bill the API key instead of the subscription.
    """
    if os.environ.get("FML_AGENT_FAKE_LLM"):
        from . import fake_llm
        return fake_llm.respond(task_prompt, json_schema)

    cmd = ["claude", "-p", task_prompt, "--model", model, "--output-format", "json",
           "--max-turns", str(max_turns), "--permission-mode", "acceptEdits",
           "--permission-prompts", "none", "--allowedTools", ",".join(tools)]
    for d in extra_dirs:
        cmd += ["--add-dir", str(d)]
    if json_schema:
        cmd += ["--json-schema", json.dumps(json_schema)]
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Claude Code call exceeded {timeout_s}s") from exc

    out = proc.stdout.strip()
    try:
        data = json.loads(out.splitlines()[-1]) if out else {}
    except json.JSONDecodeError:
        data = {}
    text = str(data.get("result", "")) or out or proc.stderr
    if (proc.returncode != 0 or data.get("is_error")) and USAGE_LIMIT_RE.search(text + proc.stderr):
        raise UsageLimitReached(text[-500:])
    if proc.returncode != 0 or data.get("is_error"):
        raise RuntimeError(f"Claude Code failed (exit {proc.returncode}): {(text + proc.stderr)[-1500:]}")
    return ClaudeResult(text=text, structured=data.get("structured_output"),
                        cost_usd=float(data.get("total_cost_usd") or 0.0), turns=int(data.get("num_turns") or 0))


# --------------------------------------------------------------------------------------
# Shell and git
# --------------------------------------------------------------------------------------

def run_cmd(cmd: list[str] | str, timeout_s: int = 1800, cwd: Path = REPO) -> tuple[int, str]:
    """Run a command; return (exit code, last 6,000 characters of combined output)."""
    try:
        proc = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True,
                              text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout_s}s"
    return proc.returncode, (proc.stdout + proc.stderr)[-6000:]


def git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def owner_repo() -> tuple[str, str]:
    url = git("remote", "get-url", "origin")
    m = re.search(r"github\.com[:/]([^/]+)/([^/.]+?)(?:\.git)?$", url)
    if not m:
        raise RuntimeError(f"origin is not a GitHub URL: {url}")
    return m.group(1), m.group(2)


def checkout_cycle_branch(branch: str) -> None:
    """Continue on the cycle branch if it exists on origin, otherwise create it from main."""
    git("fetch", "origin", "--quiet", check=False)
    remote = git("ls-remote", "--heads", "origin", branch, check=False)
    if git("rev-parse", "--abbrev-ref", "HEAD") == branch:
        if remote:
            git("pull", "--ff-only", "origin", branch, check=False)
        return
    if remote:
        git("fetch", "origin", f"{branch}:{branch}", check=False)
        git("checkout", branch)
    else:
        git("checkout", "-b", branch)


def commit_and_push(branch: str, message: str) -> bool:
    git("add", "-A")
    if not git("status", "--porcelain"):
        return False
    git("commit", "-q", "-m", message)
    git("push", "-q", "origin", f"HEAD:{branch}")
    return True


def open_pull_request(branch: str, title: str, body: str) -> str:
    """Open (or find) the PR from the cycle branch to main via the REST API. Returns its URL or a note."""
    owner, repo = owner_repo()
    code, out = run_cmd(["gh", "api", f"repos/{owner}/{repo}/pulls?head={owner}:{branch}&state=open"], 120)
    if code == 0 and out.strip().startswith("[") and json.loads(out) != []:
        return json.loads(out)[0]["html_url"]
    code, out = run_cmd(["gh", "api", f"repos/{owner}/{repo}/pulls", "-X", "POST",
                         "-f", f"title={title}", "-f", f"head={branch}", "-f", "base=main",
                         "-f", f"body={body}"], 120)
    if code == 0:
        try:
            return json.loads(out)["html_url"]
        except (json.JSONDecodeError, KeyError):
            pass
    return f"(could not open the PR automatically: {out[-300:]}) Open it by hand: {branch} -> main"


# --------------------------------------------------------------------------------------
# Hard gates (deterministic). Each returns a list of human-readable failures.
# --------------------------------------------------------------------------------------

REQUIRED_HEADINGS = ["## Goal and hypothesis", "## Explanation", "## What could go wrong",
                     "## Technical infrastructure"]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
KEY_RE = re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}")
PUBLIC_DIRS = ["content", "results", "reports", "docs", "requirements", "site", "src", "tests", "README.md"]
ALLOWED_EMAILS = {"noreply@anthropic.com"}


def tracked_text_files() -> list[Path]:
    files = []
    for entry in PUBLIC_DIRS:
        p = REPO / entry
        candidates = [p] if p.is_file() else (p.rglob("*") if p.is_dir() else [])
        for f in candidates:
            if f.is_file() and f.suffix.lower() in {".md", ".py", ".json", ".jsonl", ".csv", ".html", ".txt", ".yaml", ".yml"}:
                files.append(f)
    return files


def gate_sections() -> list[str]:
    fails = []
    sections = sorted((REPO / "content" / "sections").glob("*.md"))
    if not sections:
        return ["content/sections/ has no section files"]
    for s in sections:
        text = s.read_text(encoding="utf-8")
        missing = [h for h in REQUIRED_HEADINGS if h not in text]
        if missing:
            fails.append(f"{s.relative_to(REPO)} is missing headings: {', '.join(missing)}")
        if "<!-- cell:" not in text and s.stem not in {"00-overview", "99-glossary"}:
            fails.append(f"{s.relative_to(REPO)} has no code cell (<!-- cell: path#symbol -->)")
    return fails


def gate_required_outputs() -> list[str]:
    need = ["results/metrics.json", "reports/report.md", "docs/ai-log.md", "data/splits/split_v1.csv"]
    return [f"missing required output: {p}" for p in need if not (REPO / p).is_file()]


def gate_split_frozen() -> list[str]:
    """The split must never change after it was first created (all comparisons depend on it)."""
    split = REPO / "data" / "splits" / "split_v1.csv"
    stamp = REPO / ".agent_state" / "split_v1.sha256"
    if not split.is_file():
        return []
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    if not stamp.is_file():
        stamp.parent.mkdir(exist_ok=True)
        stamp.write_text(digest)
        return []
    if stamp.read_text().strip() != digest:
        return ["data/splits/split_v1.csv changed after it was frozen — restore it; never re-split after seeing results"]
    return []


def gate_test_access() -> list[str]:
    """Each model may be scored on the test split with one configuration only (no tuning on test)."""
    log = REPO / "results" / "test_access_log.jsonl"
    if not log.is_file():
        return []
    seen: dict[str, str] = {}
    fails = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        model, cfg = entry.get("model_id"), entry.get("config_hash")
        if not model or not cfg:
            fails.append(f"test_access_log entry without model_id/config_hash: {line[:120]}")
        elif model in seen and seen[model] != cfg:
            fails.append(f"model {model} was scored on TEST with two different configurations")
        else:
            seen[model] = cfg
    return fails


def gate_report_length() -> list[str]:
    cfg = load_config("project")["report"]
    path = REPO / cfg["path"]
    if not path.is_file():
        return []
    text = re.sub(r"```.*?```", " ", path.read_text(encoding="utf-8"), flags=re.S)
    body = re.split(r"\n#+\s*References\b", text, flags=re.I)[0]
    n = len(re.findall(r"\b\w[\w'-]*\b", body))
    if not cfg["min_words"] <= n <= cfg["max_words"]:
        return [f"report body has {n} words; target {cfg['min_words']}-{cfg['max_words']}"]
    return []


def gate_privacy() -> list[str]:
    """No e-mail addresses, API keys or listed private terms in anything public.

    Private terms (real names, institution, course, team number) come from the FML_PRIVACY_TERMS
    environment variable — they must never be written into the repo, not even in a blocklist.
    """
    terms = [t.strip() for t in os.environ.get("FML_PRIVACY_TERMS", "").split(",") if t.strip()]
    term_re = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.I) if terms else None
    fails = []
    for f in tracked_text_files():
        text = f.read_text(encoding="utf-8", errors="ignore")
        rel = f.relative_to(REPO)
        for m in EMAIL_RE.findall(text):
            if m.lower() not in ALLOWED_EMAILS and not re.search(r"@example\.(com|org|net)$", m.lower()):
                fails.append(f"{rel}: e-mail address found")
                break
        if KEY_RE.search(text):
            fails.append(f"{rel}: something that looks like an API key")
        if term_re and term_re.search(text):
            fails.append(f"{rel}: contains a private term from FML_PRIVACY_TERMS")
    return fails


def _windows(words: list[str], size: int, step: int = 1):
    for i in range(0, max(len(words) - size + 1, 0), step):
        yield hashlib.blake2b(" ".join(words[i:i + size]).encode(), digest_size=8).digest()


def gate_no_article_text(window: int = 30, stride: int = 10) -> list[str]:
    """No public file may contain a run of 40+ consecutive words copied from an article.

    Article windows are indexed every `stride` words, so any copied run of window+stride-1 words
    is guaranteed to contain one indexed window.
    """
    try:
        from fml.paths import data_dir, project_config
        path = data_dir() / project_config()["data"]["subset_file"]
    except Exception as exc:  # data repo missing: cannot check, say so
        return [f"cannot check for copied article text: {exc}"]
    norm = lambda s: re.findall(r"[a-z0-9']+", s.lower())  # noqa: E731
    index = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            index.update(_windows(norm(json.loads(line).get("txt") or ""), window, stride))
    fails = []
    for f in tracked_text_files():
        if any(part in {"tests"} for part in f.relative_to(REPO).parts):
            continue
        words = norm(f.read_text(encoding="utf-8", errors="ignore"))
        if any(h in index for h in _windows(words, window)):
            fails.append(f"{f.relative_to(REPO)}: contains 40+ consecutive words of an article")
    return fails


PROTECTED = ["agent", "sitegen", "scripts", ".github", "config/rubric.yaml", "config/agent.yaml", "requirements"]


def gate_protected_paths() -> list[str]:
    """The agent may not edit its own harness, the rubric or the requirements (no grading itself)."""
    git("fetch", "origin", "main:refs/remotes/origin/main", "--quiet", check=False)
    if not git("rev-parse", "--verify", "--quiet", "origin/main", check=False):
        return ["cannot compare with origin/main, so protected paths could not be checked"]
    changed = git("diff", "--name-only", "origin/main...HEAD", "--", *PROTECTED, check=False)
    dirty = git("status", "--porcelain", "--", *PROTECTED, check=False)
    paths = sorted(set(changed.split()) | {line[3:] for line in dirty.splitlines()})
    return [f"protected path changed by the agent: {p} (revert it; see AGENTS.md)" for p in paths if p]


def run_gates(tests_passed: bool, pipeline_ok: bool) -> list[str]:
    fails: list[str] = []
    if not tests_passed:
        fails.append("pytest failed (see pipeline_log)")
    if not pipeline_ok:
        fails.append("`fml run-all` failed (see pipeline_log)")
    for gate in (gate_required_outputs, gate_sections, gate_split_frozen, gate_test_access,
                 gate_report_length, gate_privacy, gate_no_article_text, gate_protected_paths):
        fails += gate()
    return fails
