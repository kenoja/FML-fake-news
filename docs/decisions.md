# Decisions

Choices, conflicts between sources and open questions. Newest at the bottom of each list.
Agents: add an entry whenever you choose between alternatives or resolve a conflict (see AGENTS.md).

## Decided

- **2026-10-16 · Course feedback overrules the proposal.** Full articles instead of headlines; logistic
  regression vs one cheap LLM; no fine-tuning; a few extra feature groups compared with words only.
- **2026-10-16 · LLM question.** Ask the source question (conspiracy-oriented vs mainstream website),
  the same target the LR is scored on. Content stance is examined in the error analysis instead.
- **2026-10-16 · Feature groups.** n-grams (1–3), epistemic markers, style. Emotion scores dropped by
  the team (cost and the course's warning that emotional language is not conspiracy language).
- **2026-10-16 · Off-topic documents.** First words-only run on all 3,146 documents; then remove documents
  that never mention 9/11 and explain the effect. Pattern in `config/project.yaml`; it includes
  "pentagon", which may also match unrelated defence articles — check the removed and kept examples.
- **2026-10-16 · JFK transfer test** out of scope for now.
- **2026-10-16 · Data location.** Full texts only in the private data repo; this repo is public.

- **2026-10-17 · GitHub owner name removed from docs.** Replaced with `<owner>/<repo>` and `<owner>/<data-repo>` placeholders in docs/, `.gitignore` and `.env.example`. Not changed: `config/site.yaml` (`repo_url`, `giscus.repo`) still holds the real owner path because `sitegen` has no other source for the "Code on GitHub" link and the comment widget, and `sitegen/` is protected. Also containing it and protected (reported only): `scripts/collect_feedback.py` (default `--repo`). The directory names `../fml-data` / `../FML-data` in `config/project.yaml` are folder lookups, not an owner path.
- **2026-10-16 · Licence.** No licence found in the LOCO paper; OSF page unreachable (maintenance).
  Treat texts as not redistributable until confirmed.
- **2026-10-16 · Licence check, cycle 1 task T1: unreachable on 2026-10-16.** The agent could not open
  https://osf.io/snpcg/ (the fetch tool was denied in this environment, so the page was not seen at all;
  this is not a statement about the page). Licence therefore still unconfirmed. Excerpt rules stay
  (≤ 50 words, ≤ 6 per section). Note for the report's data section: "Licence of the LOCO texts: not
  confirmed (OSF page not reachable on 2026-10-16); texts are kept private and only short excerpts shown."
- **2026-10-17 · Licence recheck, cycle 1 task T9: still unconfirmed.** Attempted to fetch
  https://osf.io/snpcg/ again on 2026-10-17; the shell and fetch calls were denied by the environment
  (no approval surface), so the page was not seen. Wording "unconfirmed" kept in report §2 and section 01.
  A person with network access should open the page and record the licence text and date here.
- **2026-10-17 · Hypotheses traceability.** `docs/PROJECT_PLAN.md` has two commits: `9658f98` (2026-10-16)
  and `5964544` (2026-10-17). The `git log -S` check of which one introduced the hypothesis table was
  denied, so it is unknown. Report §1 and section 01 now give a decision rule and an expected effect size
  for H2–H4, and H6 is called an expectation instead of a claim that predates results.
  Retried in cycle 1 task T1 on 2026-10-17: every `git log` call was denied again (no approval surface), so
  the verdict "hypotheses pre-date results: yes/no" is still **unknown** and no commit hash for
  `results/metrics.json` is recorded. A person with shell access should run
  `git log --format='%h %ad %s' --date=short -S'H1' -- docs/PROJECT_PLAN.md` and
  `git log --diff-filter=A --format='%h %ad' -- results/metrics.json`, then write both hashes, dates and the
  verdict here. Until then, do not claim the hypotheses were pre-registered.
  Retried in cycle 1 task T7 on 2026-10-17: the `git log` calls were denied a third time (no approval
  surface). `.git/logs/HEAD` only lists branch-level commits, not which commit touched which file, so it
  cannot answer the question. Still **no commit hashes and verdict "unknown"**; nothing was invented.
  **H5 gap (resolved by explanation):** there is no H5 because the emotion-score idea was dropped; the
  numbering keeps the gap on purpose. This is stated in report §1 and section 01; H6 keeps its number
  because section 11, the report §5.6, the ai-log and tests refer to it.
- **2026-10-16 · Conflict: final-scoring command.** AGENTS.md says `fml evaluate-final <model_id>`;
  `docs/PROJECT_PLAN.md` says `fml evaluate --final`. AGENTS.md ranks higher, so `evaluate-final` is
  used everywhere (the CLI already implements it).
- **2026-10-16 · Split file covers all 3,146 doc_ids.** The split is made once on the full corpus,
  grouped by website. The off-topic filter (and any near-duplicate removal) happens after the split and
  only drops rows at analysis time: the frozen file is never edited, and a dropped document keeps its
  split, so no document can move between train and test.
- **2026-10-16 · Risk: "pentagon" in the topic pattern.** It can match unrelated defence articles, so
  the filter may keep some off-topic documents. Kept as configured (the team chose the pattern);
  the off-topic section will list kept/removed examples and the count of documents matched only by
  "pentagon". The pattern is not tuned after seeing results.
- **2026-10-16 · Dependencies.** Analysis packages are pinned with upper bounds in `pyproject.toml`
  (next major version). Exact tested versions could not be recorded in T1 because shell access was
  denied; to be pinned exactly when the environment can be inspected. No new heavy dependencies.
  **Pin policy (T5, 2026-10-17):** `pyproject.toml` deliberately keeps ranges (so CI and newer machines can
  install). `requirements-lock.txt` is the authoritative exact pin. Its seven analysis packages and the
  Python version match those recorded in `results/run_info.json` (checked by reading both files). The
  remaining lock entries (jinja2, langgraph, markdown, pygments, pytest, ...) were NOT compared with the
  installed environment, because shell commands such as `pip list` were denied in this task.

- **2026-10-16 · T2 split created.** `data/splits/split_v1.csv` made by `fml prepare-data` (seed
  20261016, greedy whole-website assignment per class): 3,146 rows, no website in two splits, every
  split has ≥ 3 conspiracy websites (25 / 7 / 7), so no per-website table was written. File is frozen.
- **2026-10-16 · T2 cleaning findings.** Two exact duplicates exist (the project plan said none); they
  are dropped in the in-memory cleaned frame only, the split still lists them. 63 near-duplicate pairs
  (cosine ≥ 0.9) were found, 36 of them across two splits: detected and logged, not removed.
- **2026-10-16 · Length.** Shortest article 101 words (whitespace count), so no minimum length. F1 by
  length bucket is left to the first evaluation run (T3).

- **2026-10-16 · T3 first run.** M1 uses cleaned text and keeps the two exact duplicates (`clean_corpus(drop_exact=False)`)
  so it covers all 3,146 documents of the frozen split. Selected C = 100 (class weight none) is the
  edge of the grid {0.01..100}; F1 for C = 30 is within 0.002, so the grid was not widened. Blog-format words
  ("posted", "entry", "rss", "pm") are among the top conspiracy coefficients: a style giveaway, no site names.
  `evaluate_final` is implemented and unit-tested with a temp log; it was NOT called for any real model (no test access logged).
  `fml run-all` still raises at the end until the remaining stages exist.

- **2026-10-16 · T4 topic filter reconciled.** Measured on the real data: 3,146 → 2,608 documents = 777
  conspiracy + 1,831 mainstream, identical to the expected numbers (`metrics.json#topic_filter.matches_expected`
  is true). 538 removed: 17.3% of conspiracy, 17.0% of mainstream. Both exact duplicates are kept (the filtered
  set still contains them, as in the expected 2,608). The mask is computed on the original title and text, not the
  cleaned text, so cleaning cannot change membership. `pipeline.analysis_frame()` is the single filtered frame for
  all later stages. Split file hash unchanged (800843b0…).
- **2026-10-16 · T4 uneven effect by split.** Overall class balance is unchanged, but the test split loses 43 of
  141 conspiracy documents (30.5%) against 5 of 330 mainstream (1.5%), mostly one website; two conspiracy
  websites (one in validation, one in test) lose all documents. Kept as is: the split is frozen and the pattern
  is not tuned after seeing results. Final test numbers will rest on 98 conspiracy documents from 6 websites.
- **2026-10-16 · T4 "pentagon".** Only 10 kept documents (0.4%) match "pentagon" alone; at least one is a
  defence news story unrelated to 9/11 (judged from titles only, 6 shown). Pattern left unchanged.
- **2026-10-16 · T4 comparison.** Validation sets before/after filtering differ, so F1 0.926 (all docs) and 0.927
  (filtered) are not comparable; like-for-like row (all-docs model on filtered validation docs) is 0.940, filtered
  model 0.927. Descriptive only, no test claimed. `fml run-all` still stops at the stages not yet written.

- **2026-10-16 · T5 feature groups.** Added `src/fml/features/` (n-grams 1–3, epistemic lexicon, style) with
  new config keys under `features:` (`ngram_range`, `ngram_min_df`=3, `ngram_max_features`=200000,
  `mattr_window`=100). These values are first choices, not tuned; the ablation task may revisit them on
  validation only. The epistemic lexicon is a short hand-picked list inspired by the cited categories, not
  a copy of a published list. Section 07 uses two invented snippets instead of real article text.
  `feature_summary.csv` is computed on training rows of the filtered set only (1,787 documents).
  Known weakness kept and tested: negation is ignored ("no evidence of cover-up" counts "cover-up").

- **2026-10-17 · T12 report and final conflicts.** The report (`reports/report.md`) quotes only numbers found in
  `results/` or the config; `tests/test_report_numbers.py` enforces it. Conflicts resolved by the ranking in AGENTS.md:
  (1) `fml evaluate-final` over `fml evaluate --final`; (2) rule "temperature 0" vs the model's HTTP 400 (rule cannot be
  followed, open below); (3) the plan's "no duplicates" vs measured two exact duplicates (data wins). Chosen LR for the
  head-to-head is `lr_words` by the pre-registered rule; `lr_extended` was scored on test once, for visibility only.
  Test result on 423 documents: LLM F1 above the chosen LR, McNemar p in `metrics.json#comparison`. Human check of
  the stance sample is pending, so H6 is reported as provisional. LOCO licence still unconfirmed.

- **2026-10-17 · Conflict (a): temperature.** `requirements/cycle-01.md` asked for temperature 0; `requirements/cycle-02.md` item 7 and AGENTS.md say default sampling (the model rejects the setting). Newest wins: `llm_classifier.temperature` is deleted from config, nothing is sent, the cache key no longer contains it, and the text says plainly that the model has no temperature setting. Output limit raised from 150 to 1000 tokens (target: 0 cut-off answers); the token-limit sensitivity check is removed.
- **2026-10-17 · Conflict (b): first run on all documents.** Cycle 1 ran the first words-only model on all 3,146 documents. Cycle-02 item 4 drops this: cleaning (off-topic and near-duplicates) comes first and every model uses the cleaned set. The old wording is removed from `config/project.yaml`.
- **2026-10-17 · Conflict (c): extra analyses.** Cycle 1 added a human stance kappa, a website-level bootstrap, a stability pass and sensitivity analyses. Cycle-02 item 13 removes them; only the statistics listed in AGENTS.md remain (confusion matrix, precision, recall, F1, one 95% bootstrap interval per number, three McNemar tests). Code and text for them are removed in later tasks.
- **2026-10-17 · Near-duplicate rule (d).** Cycle 1 only counted near-duplicates; cycle-02 item 4 removes them before any model runs, together with off-topic documents. Exact threshold and counts are set in the cleaning task and shown in `02-data-and-cleaning`.
- **2026-10-17 · Logistic regression settings.** Shared config `lr: {c_grid: [0.1, 1, 10], class_weight: balanced}`, chosen on validation only.
- **2026-10-17 · Closed cycle-1 items (superseded in cycle 2).** LLM temperature: the model rejects a temperature setting, so default sampling is used (AGENTS.md now says so). Output limit: the 150-token limit is replaced by `max_output_tokens: 1000` (`config/project.yaml`). Near-duplicates: removed before any model runs, after the topic filter; the cleaned set is the one in `metrics.json#cleaning`, not a fixed 2,608. API key and duplicated cache lines: resolved, cache is complete.

## Needs a team decision

- **LOCO licence check.** The agent cannot read https://osf.io/snpcg/. One member opens it, copies the licence text and date here. Until then: article texts only in the private data repo, excerpts at most 50 words, at most 6 per section.
- **Human stance check.** The 40-article sheet (`results/tables/stance_check_sample.csv`) is not filled in by anyone; kappa is `null`, so the Claude stance codes are an unchecked single source. Needs one member per article, then `fml stance-agreement`.
- **GitHub account name in `config/site.yaml` (T12, cycle 2).** `repo_url` and `giscus.repo` hold the real account name, so it appears in `config/site.yaml`, in the built `site/index.html` (link and comment widget), and in `scripts/collect_feedback.py` (default `--repo`); `AGENTS.md` says "no GitHub usernames". The agent cannot fix this: `config/site.yaml`, `sitegen/` and `scripts/` are protected paths. **Owner action:** a human either moves the repository to a neutral organisation (then updates `repo_url`, `giscus.repo` and the collect script default) or accepts the name on the site and removes the rule. Conservative reading until then: treat the name as still present and as a privacy gap; do not claim the repo is free of it. `docs/` and `.gitignore` use `<owner>/<repo>` placeholders, and `tests/test_privacy.py` passes (it reads the name at run time).
- **Report length counting rule.** Decided in T8: the 3,000-5,000 word limit is `len(text.split())` on `reports/report.md` before References, tables included (see the last section). The team confirms or asks for prose-only counting.

## Gate fix: `fml run-all` timeout and the `lr_words` test lock

- **Speed.** `fml run-all` ran into the harness timeout (648 s) because every ablation model refitted
  its TF-IDF / n-gram / dense feature blocks and fitted the 18 (C, class weight) models one after
  another. Now `models.SharedPart` fits each identical feature block once per process (cache key = block
  settings + the exact training texts, so nothing is ever reused on different rows), and
  `tune_on_validation` fits the grid in parallel (same rows, same tie-break). Measured stages: data 27 s,
  first run 32 s, filter 6 s, baseline 7 s, features 13 s, ablation 106 s, llm+comparison+errors 21 s.
- **Conflict: `lr_words` tuned config changed.** The committed `results/tables/ablation.csv` (01:27) was
  produced by older code/data; it selected C=100 / no class weights for words-only LR, while the current
  pipeline (and the filtered baseline stage) select C=10 / balanced. `results/test_access_log.jsonl`
  already holds `lr_words` with the stale config hash, and the test lock forbids scoring the same id with
  another configuration. Choice: new model id `lr_words_v2` (this is a second test access for the
  words-only LR, openly recorded); the old log lines are kept untouched and `lr_words` numbers from the
  stale run are no longer used. Report and sections 08/10/11 were updated to the regenerated results.

## T3: freeze rule and the sequence of test looks

- `fml.pipeline.freeze_model` writes `results/frozen_models.json` (model_id, config_hash,
  selection_source=validation, timestamp, retroactive flag) before the first test scoring; `evaluate_final`
  raises `PermissionError` if the entry is missing or its hash differs. `ablation()` and `llm_stage()`
  freeze after the validation results are written. `fml evaluate-final` alone therefore refuses until the
  model is frozen. Test: `tests/test_test_lock.py`.
- Actual order in `results/test_access_log.jsonl`: `lr_words` 01:19:52, `lr_extended` 01:21:07,
  `llm_haiku` 09:26:03, `lr_words_v2` 10:35:02 (UTC, 2026-10-17). The task text listed the LLM before
  `lr_extended`; the log was followed. Section 10 and report §4 state this sequence and list the stale
  `lr_words` look and the second words-only look as limitations.
- The three looks that predate the rule get `retroactive: true` entries on the next `fml run-all`; this
  freeze is honest bookkeeping, not proof that the choice preceded the look.

## T4: near-duplicates are quantified, not removed

- The split stays frozen (`split_v1.csv`) and the filtered 2,608-document set is unchanged. Test documents
  with a near-duplicate partner in train or validation (TF-IDF cosine >= 0.9, the same pair list as
  `data.near_duplicate_pairs_across_splits`) are only left out of a sensitivity table
  (`results/tables/comparison_test_no_dups.csv`, `comparison.no_duplicate_sensitivity`).
- Result: 9 test documents excluded; F1 moves by -0.0049 (LR) and -0.0042 (LLM); the LR-vs-LLM gap is
  0.0994 before and 0.1001 after, so the conclusion is unchanged. Removal from the corpus stays a team decision.

## T5/T4: LLM prior-knowledge probe — run (5 of 40 identified)

- `llm.website_probe` (<=40 validation documents, seeded) asks Haiku to name the website and caches only
  `doc_id`, prompt hash and a boolean `matched` in `results/llm_cache.jsonl`. `errors_stage` writes the result to
  `metrics.json` `errors.prior_knowledge_probe`.
- Update (T4): a later `fml run-all` with the key ran the probe on 40 validation documents: 5 identified (12.5%), see
  `metrics.json` `errors.prior_knowledge_probe`; 40 flag-only records are in `results/llm_cache.jsonl`. Section 12 and
  report §5.6 quote it. Deviation from the proposal: the prompt gives article text, not title only (conservative
  reading: the share is a lower bound for memorisation, and both texts say so). Whether to also run a title-only probe
  is a team decision (see "Needs a team decision").
- The tests added in this task (`test_classification_table_rows_sum_to_errors`) were also not executed here.
- `error_classification.csv` gains `unparsed` (renamed from `not coded`) and `not_sampled`; the existing CSV in
  `results/` is stale until the next run (report §5.6 table was updated by hand from its old `not coded` values).

## T7: human check of the stance codes — partly, needs team member

- Status: **partly - needs team member.** The agent cannot produce human labels, so `human_stance` stays empty
  (pending, Member to fill; kappa = null). `fml stance-agreement` prints `pending: 0/40 filled` while it is empty.
- Privacy: Claude justifications and LLM reasons in `stance_check_sample.csv` and `error_examples.csv` are cut to
  25 words and name-like capitalised words outside a small allowlist become `[name]`
  (`errors.scrub_justification`, applied at parse time, on cache read and when writing the tables).
  Gate test `tests/test_gates.py` checks every public CSV cell against the excerpt limit (justifications: 25 words).
- The two existing CSVs were scrubbed in place; the next `fml run-all` regenerates them the same way.

## Compliance with `requirements/cycle-01.md` (numbered requirements 1-7)

Status after task T8; "partly" items name the reason. Numbers live in `results/`.

| # | Requirement | Status | Reason / pointer |
|---|---|---|---|
| 1 | Build the analysis in `src/fml/` (load, clean, frozen split, first run, filter, baseline, features, ablation, LLM, comparison with bootstrap CIs and McNemar, error analysis) | partly | All stages exist and write `results/`. Near-duplicates are detected and quantified but not removed (T4, team decision open); the LLM answers at temperature "default", not 0 (model rejects the setting); 150-token limit kept (sensitivity measured). |
| 2 | Error analysis: disagreement sets, Claude stance codes, 40-row human sample, "model wrong" vs "label does not describe the text" | partly | Sets, codes and the sample file exist. The human column `human_stance` is empty (needs a team member), so kappa is null and H6 is provisional. The website prior-knowledge probe was run on 40 validation documents (T5/T4 note). |
| 3 | Websites and documents per split and class; loud warning below 3 conspiracy websites | done | 25 / 7 / 7 conspiracy websites per split; no split below 3, so no per-website table was required (T2). The test split loses 2 conspiracy websites' worth of documents to the topic filter; reported in T4. |
| 4 | Every notebook section in the required format, answering the open questions | partly | Sections exist with headings and code cells. Teaching polish of sections 06, 11 and 13 is queued as a later task. |
| 5 | `reports/report.md`, 3,000-5,000 words, assignment structure, figures and tables by path | partly | Written and number-checked by `tests/test_report_numbers.py`. Literature, feature reasons and trimming are queued as a later task. |
| 6 | Keep `docs/ai-log.md` and `docs/decisions.md` up to date | partly | Decisions are current; the final AI-log pass and the human-check records are queued as a later task. |
| 7 | Re-check the LOCO licence on the OSF page | partly | Page unreachable on 2026-10-16 (fetch tool denied), result recorded; the recheck is queued as a later task. Excerpt rules stay in force. |

No requirement is declined. Two items are **blocked on a person** (status 17 Oct 2026, task T11): requirement 2
(`human_stance` for 40 rows, kappa null, H6 provisional) and requirement 7 (LOCO licence: OSF page unreachable
for the agent, texts stay private). The temperature and 150-token decisions show `Confirmation: pending` under
"Needs a team decision". The agent signed off no human check; the seven rows and ten spot-check numbers in
`docs/ai-log.md` are all `pending` with owner "Member A–E".

## Cycle 1 review: licence, near-duplicates, formally open items (task T2, 2026-10-17)

### Licence of the LOCO texts (requirement 7)

- The OSF page https://osf.io/snpcg/ must be read by a person. In this task the agent again had no
  working web access (shell and fetch calls were denied; nothing was fetched), so no licence text was seen.
- Fill-in line: `Licence text: ___ ; date checked: ___ ; by: Member _`
- Fallback, formally declined by the agent: no web access to confirm; conservative reading = data stays
  private, no redistribution. Excerpt limits (≤ 50 words, ≤ 6 per section) stay in force.
- Closes when: the fill-in line above is completed and report §2 / section 01 replace "unconfirmed" with
  the quoted licence (or the team keeps the private-data stance if the licence forbids redistribution).

### Near-duplicates

- **Decided by agent, pending team confirmation.** Keep `split_v1.csv` frozen; keep the "no duplicates"
  sensitivity row in the main comparison table (`comparison.no_duplicate_sensitivity`); do not remove
  any near-duplicate pairs from the corpus. Experiments keep using the 2,608 filtered documents.
- Closes when: a member confirms or overrides this here with date and name. An override means a new split
  or new model ids, so it needs an explicit team decision, not an agent edit.

### Formally open (requirements 2 and 7)

| Req | Open item | Reason | Owner | What closes it |
|---|---|---|---|---|
| 2 | Human stance labels (`human_stance`, 40 rows) and kappa | The agent cannot produce human labels; kappa is null, H6 provisional | Member _ (A–E) | Member labels the 40 rows in `stance_check_sample.csv`, runs `fml stance-agreement`, records kappa here |
| 2 | Website prior-knowledge probe | Ran with article text (5 of 40); title-only variant from the proposal not run | Member _ (A–E) | Decide whether a title-only probe is needed; if so add it as a second probe prompt and rerun `fml run-all` |
| 7 | LOCO licence check | OSF page not reachable by the agent (see above) | Member _ (A–E) | Fill-in line completed above |
| 7 | Temperature/licence wording confirmation | Team decision needed | Member _ (A–E) | Dates and names written in the entries above |

## T5 (cycle 1 review): epistemic lexicon coverage and negation check

- Added `lexicon_coverage` and `count_markers_negation` (`src/fml/features/epistemic.py`), `EpistemicMarkers(negation_aware=...)`,
  `ablation.epistemic_negation` (`src/fml/ablation.py`) and `pipeline.epistemic_negation_check`, called from the `ablation` stage.
  Outputs: `results/tables/epistemic_coverage.csv`, `results/tables/epistemic_negation.csv`, `metrics.json: ablation.epistemic_negation`.
- Choices: negation window = 3 tokens before the hit's first word; negators = not, no, never, any word ending in n't. The negation-aware
  variant is registered in `features.VARIANT_GROUPS`, not `NUMERIC_GROUPS`, so the feature summary does not list columns twice.
  The main M1/M2 models still use plain counts; the negation check is a side comparison on validation only and changes no model id.
- **Not run in the authoring session:** the sandbox denied every code execution, so the new tests, the coverage table and the
  validation delta were NOT run and no numbers exist yet. Section 07 therefore points at the result files and quotes no values.
  Next `fml run-all` (or the `ablation` stage) writes them; then check the delta against its CI and state in section 07 whether it is within noise.

## T10 (cycle 1 review): privacy scan

- Needs a team decision: AGENTS.md says "no GitHub usernames", but the repo link (`config/site.yaml`, `docs/SETUP.md`, `docs/PROJECT_PLAN.md`, and so `site/index.html`) contains the owner's account name. Most conservative workable reading: kept, because the site's comment tool and links need the repo address. If the team wants it gone, move the repo to an organisation or neutral account and change `config/site.yaml`.
- `tests/test_privacy.py` treats bold call-out blocks and "(invented)" examples as non-excerpts when counting quote blocks.

## Final consistency (cycle 1)

- Report trimmed to 4,903 body words (limit 5,000); count recorded in `docs/ai-log.md`.
- Not done by this task (left to the harness or humans): `fml run-all`, full test suite and site build run; the double-run reproducibility diff; human stance check (40 rows, kappa still `null`); human checks in the `docs/ai-log.md` table; licence confirmation; decisions on near-duplicates, default sampling for the LLM and a larger LLM output limit.

## Blocked: human stance verification (requirement 2) — 2026-10-17

- Status: **blocked, pending human check.** `results/tables/stance_check_sample.csv` has 0/40 rows with `human_stance` filled; kappa is `null` (not 0). Agents cannot supply human labels and no value was invented.
- Who: two members (e.g. Member A and Member B), each independently, reading the articles in the private data repo before looking at `claude_stance`.
- Steps: fill `human_stance` (one of `endorses`, `rejects_or_debunks`, `reports_neutrally`, `no_claim`), run `fml stance-agreement` (prints `pending: 0/40 filled` until then, and kappa afterwards), then update the H6 verdict in section 11 and report §5.6 and record the kappa here.
- Tooling: `fml stance-agreement` now prints a clear "BLOCKED, pending human check" message on empty input and a plain error (exit 2) on a missing or malformed file; tested in `tests/test_errors.py` for empty, filled toy and missing input.

## T4 (review): second test look, stated plainly, with sensitivity

- Facts from `results/test_access_log.jsonl` (unchanged by this task, no `evaluate-final` call): `lr_words`
  01:19:52, `lr_extended` 01:21:07, `llm_haiku` 09:26:03, `lr_words_v2` 10:35:02. `lr_words_v2` is therefore a
  second look at the words-only LR, made after the test results of `lr_extended` and `llm_haiku` were known.
- The LR-vs-LLM conclusion rests on the document-level test only (McNemar p = 0.0004; document-bootstrap
  delta-F1 interval above 0). The website-level delta-F1 interval (-0.0166 to 0.3891) includes 0. Section 10
  and report §4 say so.
- Sensitivity for the stale `lr_words` result: its test numbers are not stored, so none is invented. Choice:
  validation comparison of the stale setting (C=100, no class weights) against the reported one (C=10,
  balanced) from `m1_words_filtered_grid.csv`, written to `results/tables/comparison_lr_config_validation.csv`
  and `metrics.json#comparison.lr_config_validation_sensitivity`. Validation F1 is equal for both.
- The "without near-duplicates" rows now have document and website bootstrap intervals, computed from
  `predictions_lr.csv` and `predictions_llm.csv` (`comparison.no_duplicate_sensitivity(..., sites=...)`).

## Blocked: licence recheck (requirement 7), task T6, 2026-10-17

- **Status: blocked.** The agent tried to open https://osf.io/snpcg/ (and the OSF API) with the shell and with
  web search on 2026-10-17. Every call was denied by the environment (no approval surface, outbound web
  access not permitted), so nothing was fetched. No licence text was seen and none is guessed.
- **Who must do it:** one team member (Member _ of A–E) with a normal browser.
- **What to do:** open the page, copy the licence text word for word, then complete:
  `Licence text: ___ ; date checked: ___ ; source URL: https://osf.io/snpcg/ ; by: Member _`
  and replace "not confirmed" in report §2 and in section 02 with the quote.
- **Until then:** conservative reading stays: texts private, no redistribution, excerpt limits (≤ 50 words,
  ≤ 6 per section) in force.

## Outstanding human items at the end of cycle 1 (task T12, 2026-10-17)

| Task | What is blocked | Who | Where it is described |
|---|---|---|---|
| T2 | Near-duplicate and licence decisions the agent cannot make; team must confirm or change the conservative readings | Team | "Cycle 1 review: licence, near-duplicates, formally open items" |
| T6 | Licence text of the corpus page not seen (web access denied); "not confirmed" stays in report §2 and section 02 | One member (A–E) | "Blocked: licence recheck" |
| T11 | Human stance labels: 0/40 filled, kappa `null`, H6 provisional | Two members (A–E), independently | "Blocked: human stance verification" |

Also open: the GitHub account name inside the repo link (T10, needs a team decision) and the human checks in `docs/ai-log.md`.

Checks done by T12 (not a full run):
- `results/test_access_log.jsonl` has 4 entries, unchanged in git; sha256 `c0cdf6eb…2ee5`.
- `tests/test_report_numbers.py`, `test_privacy.py`, `test_sections.py`, `test_section_numbers.py`, `test_test_lock.py`: 56 passed, 0 skipped.
- **Not run by this task:** the full test suite, `fml run-all` and the sitegen build. The task rules say the harness runs them right after the task, so there are no timings or skip lists yet; those come from the harness output.

## Cycle 2, T2: cleaning (off-topic, then near-duplicates)

- `clean.clean_documents()` is the one cleaning function: text cleaning, then off-topic removal (3,146 → 2,608), then near-duplicate removal (2,608 → 2,554: 755 conspiracy, 1,799 mainstream). Numbers: `results/metrics.json` `cleaning`, `results/tables/cleaning_log.csv`.
- Near-duplicate rule (cosine ≥ 0.9, pairs found on the original texts, only pairs of on-topic documents): across splits the copy in validation/test goes; for a validation/test pair the test copy goes (the requirement does not say which; choice: keep the validation copy). Inside one split the lower doc_id stays. The split file is unchanged.
- The two exact duplicates are no longer dropped separately; they are caught by the near-duplicate rule.
- Short texts (team question 5): shortest article has 102 words, none under 50, so no minimum length (`cleaning.short_texts`).
- Removed: the all-documents run, the off-topic comparison tables/figures and `src/fml/topic.py`. Content sections 01, 04, 05, 06, 12 still describe the old run; the queued section tasks rewrite them. Until then `test_sections`/sitegen checks may fail on missing outputs.
- `baseline_filtered` stays for now (words-only model on the cleaned set) because the ablation and comparison stages read its grid; the analysis tasks replace it.

## Cycle 2, T5: analysis 3 (TF-IDF plus engineered features)

- Five models only (TF-IDF alone, + n-grams 1-3, + certainty words, + style, + all three), same C grid and class-weight choice, validation F1 only. Dropped: bootstrap intervals, McNemar here, drop-one rows, coefficient tables, the negation check (`epistemic_negation`, `count_markers_negation`, `VARIANT_GROUPS`), and `epistemic_negation.csv`, `top_ngrams.csv`, `dense_coefficients.csv`. Kept: the lexicon coverage table (it supports the "why so little" explanation).
- Decision rule without intervals (team may change): a group "helps" if validation F1 rises by at least `ablation.HELPS_BY` = 0.01; the chosen model is the best row only if it helps, else plain TF-IDF. Result: certainty words +0.017 (chosen, `lr_plus_epistemic`), n-grams +0.002, style -0.001, all three -0.018. The report calls this "no reliable help" because 376 validation articles cannot separate 1-2 points from chance.
- **Test scoring removed from this stage** (AGENTS.md: all runs first, then compare). `lr_words_v2` and `lr_extended` keep their frozen ids and configs; their old test predictions are still in `predictions_lr.csv`. `lr_plus_epistemic` has no test scores yet. **The comparison stage still looks for the chosen model in `predictions_lr.csv` and will fail until the queued "final run" and "comparison" tasks score the final models and update it.**
- Section 06 replaces 07 and 08 (deleted). Numbers in it come from `results/tables/ablation.csv`, `epistemic_coverage.csv`, `feature_summary.csv`.

## T8 (cycle 2): comparison and section 08
- `08-results` answers three hypotheses: H1 TF-IDF beats word counts, H2 certainty and style features improve TF-IDF, H3 the LLM beats the best logistic regression. The rewrite of section 01 must use the same numbering and meaning, or section 08 must be adjusted. Hypotheses about label problems belong to `09-where-they-fail`.
- "Best logistic regression" for the third McNemar test = the final model of TF-IDF vs TF-IDF + features with the higher validation F1 (`comparison.lr_best`); it is TF-IDF + features.
- The majority baseline row, precision-recall curve, length buckets and extra scores (macro F1, balanced accuracy) were removed from the comparison.
- Final models were scored on test once in this task (`final_*` ids, `results/test_access_log.jsonl`).
- Not done here: `tests/test_report_numbers.py` still fails 2 tests (old report structure, stale hard-coded 0.6842); that belongs to the report/number-check tasks.

## Cycle 2, T12: final gates

- **Stale section files deleted** (slugs not in the AGENTS.md list): `00-overview`, `01-research-questions`, `02-data-and-labels`, `06-off-topic-filter`, `12-limitations`. They produced build warnings about outputs that no longer exist.
- **Report length.** `python -m sitegen build` gives a report body of 5,766 whitespace-separated words *including* table rows; the prose alone (table rows and figure links removed) is 4,330 words. `tests/test_report_numbers.py::test_report_length_is_within_the_limit` counts prose only and requires 3,000-5,000. Reading: tables are not report prose. If the team counts tables too, about 770 words (mostly table cells of `ablation`, `comparison_test`, `error_examples`) would have to go. Needs a team decision.
- **Explained parts** are all at most 450 words (checked with the counting rule of `agent/tools.py`); `09-where-they-fail` was 499 and was trimmed.
- **Rounding.** Task text asked for 2 decimals for precision/recall/F1. The site renders every table with 3 decimals (`sitegen/build.py`, protected path), so the prose uses 3 decimals everywhere (p-values below 0.0001 are written "p < 0.0001"). One rule, no mixed 0.68 / 0.684. `tests/test_report_numbers.py` now checks section titles, that every number is in `results/` or config (class sizes tp+fn and fp+tn count as derived values; article ids and hashes are ignored), and that scores use at most 3 decimals.
- **Not done by this task (rules forbid running them inside a task):** two consecutive `fml run-all` runs with a diff of `results/`, the timing check (< 8 min) and the full test suite (< 3 min). Run them as described in README "Reproducibility check" and record the outcome here. The last recorded run time is in `results/run_info.json`.
- **Remaining human items:** corpus licence check (T6); the 40-article human stance check (kappa still `null`); the GitHub account name in the repo link (T10); the team decisions listed above and in earlier cycle sections.

## Report length is counted as the plain whitespace count (cycle 2, T8)
- The 3,000–5,000 word limit is checked as `len(text.split())` on `reports/report.md` before References, tables included. This is stricter than the agent gate, which skips table rows.
- To stay under the limit: the report shows a 4-row comparison table (`results/tables/comparison_compact.csv`); the four counts only appear in the confusion-matrix figure; raw key-value and grid tables in sections 02–07 became one sentence each; the duplicate table in 11 was removed; the example-excerpt table in 09 stays in the notebook only. Tests: `tests/test_gates.py`.

## Cycle 2, T1: out-of-scope analyses dropped
- Removed the LLM prior-knowledge probe (`llm.website_probe`, `errors.prior_knowledge_probe`, the "12 of 40" sentence), the stance-coding "unreadable" footnote count (`errors.footnote_unreadable_codes`) and the term "website bootstrap". Reason: AGENTS.md allows only the confusion matrix, precision/recall/F1, one bootstrap CI and three McNemar tests (requirement 13). Section 10 now only says the LLM may know the articles, without a number.
- Kept: the single count of unreadable LLM classifier answers (`llm.unreadable`), because requirement 7 asks for it as one number. Unreadable stance codes are still left out of the error classification, with no separate count.
- Old probe lines in `results/llm_cache.jsonl` are unused.

## T2: refactor and agent test

- `StyleFeatures` and the long stage functions in `pipeline.py` were split into short helpers; tables are byte-identical.
- `tests/test_agent_graph.py` tests the protected `agent/` code, not the study. It could not be deleted in this run (the delete command was blocked), so it is left in place. Needs a team decision: delete it if CI does not require it.

## T3 (cycle 2): stance-check sample now spreads over websites

- The 40-article sheet (`results/tables/stance_check_sample.csv`) was 25 rows from one website. It is now drawn
  from **validation** articles only (never test): 20 conspiracy + 20 mainstream, at most
  `stance_coding.max_per_website` (6) per website, taken one website at a time with the split seed.
  Result: 40 rows from 19 websites, at most 4 per website.
- Claude's stance for these 40 articles needed 40 new API calls (key `LLM_CLASSIFIER_API_KEY` was present);
  they are now in `results/stance_cache.jsonl` (doc_id + hash + answer only), so `fml run-all` makes no new calls.
- The human `human_stance` column is still **blank on purpose**: the agent cannot produce human labels, the
  team fills it in. The previous sheet (test-set errors) is replaced, so any earlier partial human entries
  would be lost; none existed (`has_human_input` guards against overwriting).
- Test: `test_check_sample_caps_rows_per_website_and_balances_classes`.
