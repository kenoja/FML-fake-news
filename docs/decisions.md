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

## Needs a team decision

- **LLM temperature 0 cannot be set (T8). Status: accepted default, awaiting team confirmation.** AGENTS.md
  asks for temperature 0, but `claude-haiku-5-5` answers HTTP 400 ("`temperature` is deprecated for this
  model") and the installed SDK has no such argument. Default chosen: call the model with its default
  sampling and keep `temperature: 0` in config (it stays in the cache key). Why: the rule cannot be
  followed, and sending nothing is the only call the API accepts. Measured effect (second test pass, 423
  documents, `metrics.json#llm.stability`): agreement 0.9504, 21 flipped documents, Cohen's kappa 0.8871;
  the official pass-1 and pass-2 confusion matrices give the same F1. Quoted in section 09 and report §5.4.
  Conservative reading (stated explicitly): the API rejects temperature 0 for this model, default sampling
  is used, and results are cached so reruns are deterministic. Section 09 and the report mirror this.
  Proposed decision wording for the team to confirm: "We accept default sampling for `claude-haiku-5-5`
  because temperature 0 is rejected; the cache makes `fml run-all` deterministic; run-to-run stability is
  reported (agreement, kappa)." Confirmed on (date): ___ ; by: Member _ . Closes when: a member writes the
  date and name here, or the team asks for a different model/setting (which means a new model id on test).

- **Licence of the LOCO texts.** Unconfirmed (see above). Default: texts not redistributable; private
  data repo only; excerpt limits apply. Measured effect: not run: the OSF page was unreachable, so no
  licence text could be read (recheck is queued as a separate task).
- **LLM API key in the cloud environment.** It was unknown whether `LLM_CLASSIFIER_API_KEY` is set when
  the agent runs. Default: assume it is set and fail loudly if not. Measured effect: it was set in the T8 run, so the three validation variants and both test passes were
  produced (numbers: `metrics.json` key `llm`). The committed cache `results/llm_cache.jsonl` is
  complete for those runs, so `fml run-all` needs no key; without key and cache the stage still fails
  loudly with exact missing counts, and never falls back to `ANTHROPIC_API_KEY`.
- **LLM parse errors and the 150-token output limit (T8, revised in T3). Status: accepted default, awaiting team
  confirmation.** Default: keep `max_output_tokens: 150` for the official runs (changing it now would mean a
  new test run after seeing test results). Investigation without new API calls (`llm.parse_error_report`,
  `results/tables/llm_parse_errors.csv`, counts in `metrics.json#llm.parse_error_causes`): every unparsed cache
  record used exactly 150 output tokens, i.e. the whole budget, and no parsed reply used more. The dominant
  (only visible) cause is therefore truncation at the output limit; refusal, hedging and wrong format cannot be
  distinguished because the cache stores no raw replies (inferred, not read). The earlier T8 conclusion "most
  parse errors are NOT caused by the cut-off" is **retracted**: the 400-token sensitivity run (pass 3) reused
  the 150-token client, so its records also show 150 tokens, and `llm.token_limit_sensitivity` compares 150
  with 150. Shares: see `metrics.json` (`llm.variants.v2.parse_errors` / `n_docs` on validation,
  `llm.test.parse_errors` / `n_docs` on test, `llm.token_limit_sensitivity.large` for the 12% subset figure).
  Cheap fix not applied: it needs new API calls (a valid 400-token validation run, ideally storing raw replies)
  and, for test, a new documented run under a new model id. Future work: fix `token_limit_sensitivity` so it
  builds its own client with the larger limit, rerun on validation, and decide with the team.
- **Parallel LLM runs left duplicate lines (T8).** Two interrupted-attempt processes ran the stage at the
  same time, which duplicated cache lines (1,059 dropped; first answer per `doc_id`+`prompt_hash` kept,
  `read_cache` now does the same) and wrote the `llm_haiku` test-access entry twice (the second,
  identical line was removed). Test data was classified once per configuration; the numbers in
  `metrics.json` were regenerated from the deduplicated cache and are reproducible without a key.
- **Near-duplicate removal vs the expected 2,608.** If near-duplicate removal is applied after the
  topic filter, the count will fall below 2,608 and that is the number used downstream. Conservative
  reading (T2 measured 63 near-duplicate pairs, 53 documents would go, 36 pairs cross splits): report 2,608 after the topic filter first, then the count after near-duplicate removal
  separately, and use the smaller set only if the team approves; until then experiments use 2,608.
  Default: experiments use 2,608. Measured effect (T4, `comparison.no_duplicate_sensitivity`, see "T4:
  near-duplicates are quantified, not removed" below): leaving out the 9 test documents with a
  near-duplicate partner moves F1 by -0.0049 (LR) and -0.0042 (LLM); the conclusion is unchanged.

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

No requirement is declined.

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
