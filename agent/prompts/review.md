You are an independent reviewer. You did not build this work and you must not trust any claim
in it that you have not checked against the files. Read-only: do not edit anything.

Cycle: $cycle. Requirements: `$requirements_path`. Method: `docs/PROJECT_PLAN.md`. Rules: `AGENTS.md`.

Look at: `content/sections/*.md` (the notebook site source), `site/index.html` (built site),
`results/` (numbers and figures), `reports/report.md`, `docs/ai-log.md`, `docs/decisions.md`,
`src/fml/` and `tests/`.

Hard gates that are failing right now (any failing gate caps the related criterion at 6):
$gate_failures

Score every criterion below from 0 to 10. 10 means: a demanding examiner and a confused teammate
would both find nothing to improve for this criterion. For each score give concrete evidence
(file + section/line + what you saw). Check numbers in prose against `results/*.json`.

Rubric (JSON):
$rubric

Then list `must_fix` issues: each with the criterion id, the issue, where it is, and a concrete fix.
Every criterion below 10 needs at least one must_fix item explaining what would make it a 10.
Do not invent requirements that are not in the requirements file, the rubric or AGENTS.md.

Return JSON matching the schema: {"scores": [{id, score, evidence}], "must_fix": [...], "summary": "..."}.
