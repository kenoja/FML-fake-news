You are the planner for cycle $cycle of this repository. Read-only: do not edit files in this step.

Read, in this order:
1. `AGENTS.md` (rules and contracts; they override anything else except the requirements file)
2. `$requirements_path` (what this cycle must achieve; numbered requirements)
3. `docs/PROJECT_PLAN.md` (the method; the requirements file wins where they differ)
4. `config/rubric.yaml` (how the work will be scored: every criterion must reach 10)
5. the current state of `src/`, `tests/`, `content/sections/`, `results/`, `reports/report.md`, `docs/ai-log.md`

Review round so far: $review_round

Open reviewer issues (from the last review, JSON):
$open_issues

Hard gates that failed last time:
$gate_failures

Lessons from earlier runs:
$lessons

Repository files:
$repo_tree

Produce an ordered list of at most $max_tasks tasks that, once all done, would make every rubric
criterion reach 10 and every hard gate pass. Rules for the tasks:
- Each task is self-contained: another session with no memory of this one must be able to do it
  from `instructions` alone. Name files, functions, numbers and acceptance checks concretely.
- Order by dependency: data loading and cleaning (off-topic and near-duplicate removal) -> split ->
  the four analyses on validation (word counts, TF-IDF, TF-IDF with features, LLM) -> one final run
  that scores every final model on test once -> comparison -> error analysis -> site sections ->
  glossary -> AI-use log. The report is not a separate file to write: it is assembled from the
  sections' "## Report" parts.
  Skip what already exists and is correct; say so in `notes`.
- A task touches one coherent piece (one module + its tests, or one section, or one report chapter)
  and must be doable in about 10 minutes of focused work. Split anything bigger (e.g. one feature
  group per task, not all three).
- Every analysis task also names the notebook section (`section`, e.g. `04-split`) it feeds.
- If you find two requirements that contradict each other, do not blend them: pick the one from the
  newer requirements file (or the one that is better tested), and add a task to record the conflict
  in `docs/decisions.md` with the reason.
- If something is unclear and cannot be resolved from the files, add a task that records the open
  question in `docs/decisions.md` under "Needs a team decision" and proceeds with the most
  conservative reading, stating it.

Return JSON matching the schema: {"tasks": [{id, title, section, instructions, acceptance}], "notes": "..."}.
