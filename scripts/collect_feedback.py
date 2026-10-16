"""Collect the team's comments and 👍 sign-offs for one cycle and draft the next requirements file.

Deterministic on purpose (no LLM): it copies, sorts and anonymises. The agent interprets the
items in the next cycle; the repo owner edits the draft before merging it.

Run by .github/workflows/collect-feedback.yml, or locally:
    GITHUB_TOKEN=... TEAM_MEMBERS='{"login":"Member A", ...}' python scripts/collect_feedback.py --cycle 1

Writes:
    feedback/cycle-NN.json            all comments for cycle NN, usernames replaced by pseudonyms
    requirements/cycle-(NN+1).md      the draft requirements for the next cycle
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 50, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        title url
        category { name }
        reactions(content: THUMBS_UP, first: 50) { nodes { user { login } } }
        comments(first: 100) { nodes {
          author { login } body createdAt
          replies(first: 50) { nodes { author { login } body createdAt } }
        } }
      }
    }
  }
}"""
KIND_RE = re.compile(r"^\s*\**\s*(unclear|change|other)\s*\**\s*:\s*", re.I)


def graphql(token: str, variables: dict) -> dict:
    req = urllib.request.Request("https://api.github.com/graphql",
                                 data=json.dumps({"query": QUERY, "variables": variables}).encode(),
                                 headers={"Authorization": f"bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    if data.get("errors"):
        raise RuntimeError(data["errors"])
    return data["data"]


def fetch_discussions(owner: str, name: str, token: str, category: str) -> list[dict]:
    out, cursor = [], None
    while True:
        page = graphql(token, {"owner": owner, "name": name, "cursor": cursor})["repository"]["discussions"]
        out += [d for d in page["nodes"] if d["category"]["name"] == category]
        if not page["pageInfo"]["hasNextPage"]:
            return out
        cursor = page["pageInfo"]["endCursor"]


def pseudonym(login: str | None, members: dict[str, str]) -> str:
    if not login:
        return "Unknown"
    return members.get(login, "Guest (not in TEAM_MEMBERS)")


def split_kind(body: str) -> tuple[str, str]:
    m = KIND_RE.match(body)
    return (m.group(1).capitalize(), body[m.end():].strip()) if m else ("Other", body.strip())


def collect(cycle: int, discussions: list[dict], members: dict[str, str]) -> dict:
    prefix = f"cycle-{cycle:02d}/"
    sections = {}
    for d in sorted(discussions, key=lambda d: d["title"]):
        if not d["title"].startswith(prefix):
            continue
        slug = d["title"][len(prefix):]
        items = []
        for c in d["comments"]["nodes"]:
            kind, text = split_kind(c["body"])
            items.append({"kind": kind, "by": pseudonym((c["author"] or {}).get("login"), members), "text": text,
                          "replies": [{"by": pseudonym((r["author"] or {}).get("login"), members), "text": r["body"].strip()}
                                      for r in c["replies"]["nodes"]]})
        signed = sorted({pseudonym((r["user"] or {}).get("login"), members) for r in d["reactions"]["nodes"]})
        sections[slug] = {"items": items, "signed_off_by": signed}
    return {"cycle": cycle, "collected": date.today().isoformat(), "members": sorted(set(members.values())),
            "sections": sections}


def requirements_markdown(fb: dict) -> str:
    cycle, nxt = fb["cycle"], fb["cycle"] + 1
    members = fb["members"]
    lines = [f"# Requirements for cycle {nxt}", "",
             f"Drafted from the team's feedback on cycle {cycle} (collected {fb['collected']}) by "
             "`scripts/collect_feedback.py`. **Edit this file before merging**: delete items the team "
             "rejected, and resolve the conflicts at the end. The agent treats every numbered item as a requirement.", "",
             "## Carried over", "",
             f"All requirements in earlier files (`requirements/cycle-01.md` ... `cycle-{cycle:02d}.md`) still apply "
             "unless an item below changes them. Later items win over earlier ones.", "",
             "## Sign-off status", ""]
    header = "| section | " + " | ".join(members) + " | everyone |"
    lines += [header, "|" + "---|" * (len(members) + 2)]
    all_signed = True
    for slug, sec in fb["sections"].items():
        row = [("👍" if m in sec["signed_off_by"] else "–") for m in members]
        done = all(m in sec["signed_off_by"] for m in members) and not sec["items"]
        all_signed &= done
        lines.append(f"| {slug} | " + " | ".join(row) + f" | {'yes' if done else 'no'} |")
    lines += ["", f"**Cycle finished for everyone:** {'yes' if all_signed and fb['sections'] else 'no'}", "",
              "## Feedback by section", "",
              "For **Unclear** items: answer the question in that section of the walkthrough (and the report, if relevant). "
              "For **Change** items: make the change, or record in `docs/decisions.md` why not. "
              "**Other** items: use judgement; record what you did.", ""]
    n = 0
    for slug, sec in fb["sections"].items():
        if not sec["items"]:
            continue
        lines += [f"### {slug}", ""]
        for item in sec["items"]:
            n += 1
            lines.append(f"{n}. **[{item['kind']}]** ({item['by']}) {item['text']}")
            for r in item["replies"]:
                lines.append(f"   - reply ({r['by']}): {r['text']}")
        lines.append("")
    if n == 0:
        lines += ["No comments were left in this cycle.", ""]
    lines += ["## Conflicts to decide", "",
              "List here any items that contradict each other, with the team's decision. "
              "The agent will not blend contradicting requests.", "", "- none recorded yet", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "kenoja/FML-fake-news"))
    ap.add_argument("--category", default="Feedback")
    args = ap.parse_args()
    owner, name = args.repo.split("/")
    members = json.loads(os.environ.get("TEAM_MEMBERS") or "{}")
    if not members:
        raise SystemExit("TEAM_MEMBERS is empty: set the repository secret (see docs/SETUP.md)")
    fb = collect(args.cycle, fetch_discussions(owner, name, os.environ["GITHUB_TOKEN"], args.category), members)
    (REPO / "feedback").mkdir(exist_ok=True)
    (REPO / "feedback" / f"cycle-{args.cycle:02d}.json").write_text(json.dumps(fb, indent=2, ensure_ascii=False), encoding="utf-8")
    out = REPO / "requirements" / f"cycle-{args.cycle + 1:02d}.md"
    out.write_text(requirements_markdown(fb), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)} with {sum(len(s['items']) for s in fb['sections'].values())} items")


if __name__ == "__main__":
    main()
