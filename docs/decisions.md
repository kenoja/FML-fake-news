# Decisions

One section per cycle, newest last. Every item carries a tag: **[done]** (decided and carried out),
**[declined]** (not done, with the reason) or **[open]** (waiting for a person). Open items are repeated under
"Needs a team decision" at the end. Agents: add an entry whenever you choose between alternatives or
resolve a conflict (see AGENTS.md).

## Cycle 1 (2026-10-16 to 2026-10-17)

- **[done] Course feedback overrules the proposal.** Full articles instead of headlines; logistic regression
  vs one cheap LLM; no fine-tuning; a few extra feature groups compared with words only.
- **[done] LLM question.** The LLM answers the source question (conspiracy-oriented vs mainstream website),
  the same target the logistic regressions are scored on. Content stance is examined in the error analysis.
- **[done] Feature groups.** n-grams (1-3), certainty words (epistemic markers), style. Emotion scores were
  dropped by the team (extra cost, and the course warns that emotional language is not conspiracy language).
- **[done] Data location.** Full texts only in the private data repo; this repo is public.
- **[done] Split.** `data/splits/split_v1.csv` was made once by `fml prepare-data` (seed from config, whole
  websites assigned to one split). It covers all documents of the corpus and is frozen; cleaning only drops
  rows at analysis time, so a dropped document keeps its split and nothing moves between train and test.
- **[done] Conflict: final-scoring command.** AGENTS.md says `fml evaluate-final <model_id>`, the plan says
  `fml evaluate --final`. AGENTS.md ranks higher, so `evaluate-final` is used.
- **[done] Dependencies.** `pyproject.toml` keeps ranges so CI and newer machines can install;
  `requirements-lock.txt` is the exact pin. The seven analysis packages and the Python version in the lock
  match `results/run_info.json`; the other lock entries were not compared with an installed environment.
- **[done] Test lock bookkeeping.** `results/frozen_models.json` records each model before its first test
  scoring; `evaluate_final` refuses a missing or changed entry (`tests/test_test_lock.py`). Three early test
  looks of cycle 1 predate that rule and are marked retroactive; the cycle-2 final models are new ids.
- **[done] Stance-check privacy.** Justifications in the public CSVs are cut to 25 words and name-like words
  are replaced by `[name]` (`errors.scrub_justification`).
- **[declined] Hypotheses "written before the results".** The dating cannot be shown (git history was not
  accessible to the agent), so no text claims it. The hypotheses are described as chosen from the literature
  and fixed before the final test run. There is no H5 on purpose (the emotion idea was dropped).
- **[declined] Licence check by the agent.** The dataset's licence page could not be opened in any
  cycle (web access denied, nothing fetched, nothing guessed). Safe default: texts private, excerpts at most
  50 words and 6 per section. See [open] below.

## Cycle 2 (2026-10-17)

- **[done] Conflicts resolved by the newest source.** (a) Temperature: cycle 1 asked for temperature 0, cycle 2
  and AGENTS.md say default sampling (the model rejects the setting); nothing is sent, the output limit is
  1000 tokens. (b) The first run on all documents is gone: cleaning comes first and every model uses the
  cleaned set. (c) Extra analyses of cycle 1 (stance kappa, website bootstrap, stability, sensitivity
  analyses, prior-knowledge probe) were removed; only the statistics listed in AGENTS.md remain.
  (d) Near-duplicates are now removed before any model runs.
- **[done] Cleaning.** `clean.clean_documents()`: off-topic removal (3,146 to 2,608), then near-duplicate
  removal (cosine 0.9 or higher; 2,608 to 2,554). Which copy of a pair goes is stated once, in the cycle-3 entry
  "R7 near-duplicate choice" (the cycle-2 wording of this rule is dropped). No minimum length (the shortest
  article has 102 words). Numbers: `results/metrics.json#cleaning`.
- **[done] Four analyses.** Word counts, TF-IDF, TF-IDF plus features (five models compared on validation; the
  final model always uses all three groups), and the LLM (at most
  three prompt variants, validation only). Shared logistic-regression settings: C grid and balanced class
  weights from `config/project.yaml`. The "best logistic regression" for the third McNemar test is the
  final model with the higher validation F1.
- **[done] Hypotheses.** H1 TF-IDF beats word counts; H2a/b/c certainty words, style, n-grams help TF-IDF;
  H3 the LLM beats the best logistic regression. H2b and H2c were only tried on validation.
- **[done] Final run.** All four final models were scored on test once each (`final_*` ids,
  `results/test_access_log.jsonl`). Comparisons appear only in `08-results` and later.
- **[done] Stance-check sample.** 40 validation articles (20 + 20, at most 6 per website, 19 websites),
  stance codes cached in `results/stance_cache.jsonl`. The `human_stance` column is blank on purpose.
- **[done] Report counting rule.** The 3,000-5,000 word limit is `len(text.split())` on `reports/report.md`
  before References, tables included (stricter than the agent gate). Tests: `tests/test_gates.py`.
- **[done] Report and sections.** Stale section files were deleted; the licence is not discussed on the site
  or in the report beyond "texts are not redistributed"; 00 has no report part (sitegen would have copied it).
- **[done] Reproducibility (requirement 18).** Two consecutive `fml run-all` runs gave identical tables and
  metrics; both checksum lists are in `docs/reproducibility.md`.
- **[done] Privacy of the account name in the site.** `config/site.yaml` has empty `repo_url` and
  `giscus.repo`, so the built site shows no account name (no "Code on GitHub" link, comments show "not
  connected yet"). `tests/test_privacy.py` scans docs, `config/` and the built site.
- **[done] Requirement 14 (beginner code).** Every function shown as a code cell now has a one-line
  docstring (reasoning moved into short inline comments). Long ones were split into helpers
  (`clean.step_log`, `short_text_info`, `duplicate_info`; `llm.client_for_missing`; `pipeline.check_frozen_config`,
  `score_on_test`, `log_test_access`). Cleaning outputs and `mattr` were checked to be identical before/after;
  `tests/test_cell_functions.py` covers the new helpers. Left as is: `prompt_variants` (30 lines, mostly prompt
  text), `classify` (32) and `errors_stage` (26) are only slightly over ~25 lines.
- **[declined] Protected-path conflict.** `scripts/collect_feedback.py` contains the owner's account name as
  a default `--repo` value. `scripts/` is protected, so the agent did not edit it, and git history cannot be
  rewritten by the agent. Conservative reading: do not claim the repository is free of the name.
- **[declined] Scoring the human stance check by command.** The agreement code was removed as out of scope
  (requirement 13). If members fill the sheet, agreement is computed by hand.
- **[declined] Deleting `tests/test_agent_graph.py`.** It tests the protected `agent/` code, not the study; the
  delete command was blocked, so the file stays.
- **[done] Warning fixed in cycle 3.** The non-flat output marker was removed from the limitations section.

## Needs a team decision

Only items that need people. Each has an owner (Member A-E) in `docs/human-todo.md`.

- **[open] Dataset licence. Owner: repository owner (Member A).** The OSF page of the dataset could not be read
  automatically (maintenance, then blocked for automated access). Open it in a browser and record here: licence text,
  date checked. Until then the safe default applies: texts private, excerpts at most 50 words and 6 per section.
- **[open] Human stance check. Owner: the team.** Each member codes 16 of the 40 articles in
  `results/tables/stance_check_sample.csv` with the offline hand-check file (every article by two members, website
  hidden) and sends the downloaded CSV; the results are added in cycle 4. Until then the error analysis states that
  the stance codes are automatic and that the hand check follows.
- **[done, 2026-10-26] Owner account name.** `scripts/collect_feedback.py` now takes the repository only from
  `GITHUB_REPOSITORY` (set by GitHub Actions); the site settings get it the same way. No account name is stored in files.
- **[done, 2026-10-26] Background and classic method references.** Verified online and added: Dietterich (1998), Salton
  & Buckley (1988), Kaufman et al. (2012), Geirhos et al. (2020), Shu et al. (2017), Swami et al. (2010), Wood &
  Douglas (2013). The LOCO paper's own blind check (85% / 92%) was verified against the paper.
- **[done, 2026-10-26] Website names in the check sheet.** Kept: the domains are public websites, not personal data,
  and the hand-check tool hides them from the coders; the site and report show "Website 1-6" only.
- **[done, 2026-10-26] What the team changed in the AI's output.** Recorded in `11-how-we-used-ai` (report part):
  full articles, the four methods, no emotion features, cleaning before any model, and the fair-comparison rerun.

Closed with a conservative decision (no human needed): report length is a plain whitespace count of `reports/report.md`
before References, tables included; near-duplicates follow the rule above; website names next to labels are shown as
"Website 1-6"; `tests/test_agent_graph.py` stays; R8 limitations wording is met.

## Cycle 3, T4: final2 run
- **Final2 logistic regressions scored once on test** under `final2_word_counts`, `final2_tfidf`, `final2_tfidf_features`
  (settings and C from the new validation run: balanced class weight, shared C grid, first 3,000 words). The old
  `final_word_counts`, `final_tfidf`, `final_tfidf_features` ids stay in `results/test_access_log.jsonl` and
  `results/frozen_models.json` but are no longer built, scored or reported. `final_llm` is unchanged and not scored again.
- **Best logistic regression for the LR-vs-LLM McNemar test and the error analysis**: the one of the three with the highest
  validation F1 (tie: the earlier one). It is `final2_tfidf` in this run.
- **Baseline row** "always guess mainstream" is in `comparison_test.csv` and `comparison_compact.csv`; its accuracy is in
  `metrics.json#comparison.baseline`. It is left out of the two comparison figures (F1 = 0).
- **Stance coding** for the new disagreement articles ran with `LLM_CLASSIFIER_API_KEY` set; new answers were added to
  `results/stance_cache.jsonl`. A later run without the key works from that cache; a missing entry fails loudly.
- The three old `results/tables/predictions_final_{word_counts,tfidf,tfidf_features}.csv`
  files were deleted later (nothing read them). Section prose and
**[superseded]**   `docs/reproducibility.md` still quote earlier numbers and file names; the queued section rewrites must update them
  (`tests/test_report_numbers.py` fails until then).

## Cycle 3, T7 (section 01)
- Added `src/fml/methods_overview.py` (writes `results/tables/methods_overview.csv` at the end of `run_all`); `run_info.json` now also holds machine info and per-stage seconds. LR runtime = stage time (the first stage includes loading data); LLM runtime and cost = averages from the answer cache rows that recorded tokens, at list price from the config.
- References: no new sources were added. The web check of candidate classics (Spärck Jones 1972; Swami et al. 2010; Douglas et al. 2019) was not possible in this run, so the introduction cites only Miani et al. (2022) and Fong et al. (2021), which are already in `content/references.md`. Needs a team decision: add confirmed classic references for TF-IDF, logistic regression and 9/11 conspiracy background.
- **[superseded]** `tests/test_gates.py` report-length gate (≤5000 words) and several number checks in sections 04–08/11 fail from other, queued tasks, not from section 01.

## Cycle 3, T11 (sections 00, 09, 10, 11, docs)
- **[done] D1 slugs.** `09-limitations`, `10-conclusion` and `11-how-we-used-ai` replace the old 10-12 files (old files deleted). Section 00 mentions the new order.
- **[done] D3 hand check.** The 40 `doc_id`s are untouched. The team's hand check follows in cycle 4; the limitations name the error analysis as an estimate that rests on the automatic coding.
- **[done] D10 / H1-H3 wording.** Conclusion gives one answer per hypothesis (H1 narrowly yes, H2 no, H3 yes) and the plain verdict that the hand-made features did not help. Numbers come from `results/metrics.json#comparison` and `#tfidf_features`.
- **[done] R4/R5 process and licence remarks.** Licence status, "None of them is recorded as done" and similar remarks are gone from the site and report; the licence stays here only.
- **[done] Requirement 14 (beginner code).** Marked done at the owner's request in the cycle-3 task list; the final gates and the reviewer re-check the shown cells.
- **[done] R7 report length counting.** Closed with the conservative choice: plain whitespace count of `reports/report.md` before References, tables included (stricter than prose-only). Stated once in `docs/decisions.md`; gate in `tests/test_gates.py`.
- **[done] R7 near-duplicate choice (the only statement of the rule; matches `clean.near_duplicate_drops`).** Closed. Across splits the priority is train, then validation, then test: the copy in the lower-priority split goes (so a validation/test pair loses the test copy; a pair with a training copy loses the other). Inside one split the lower doc_id stays. The limitations section names the 0.9 threshold as a choice.
- **[done] Conflict: model name in the AI log.** `docs/ai-log.md` said Opus 5.5 for the setup and Sonnet 5.5 later. `config/agent.yaml` sets Sonnet 5.5 for planner, implementer and reviewer, so the log now says that; the model of the original claude.ai setup session is not recorded in the repo and is stated as such.
- **[open] Hand check.** Follows in cycle 4 (see "Needs a team decision"); not scored until then.

## Cycle 3, T12 (glossary, references, report length)
- Report was 6,723 words by plain count. Cut to about 4,850: removed the grid tables, the four top-word tables and the error-classification table from the report parts of 04, 05 and 08 (they stay in the code part), and shortened prose in 01-03, 06-11. No result changed.
- `test_website_recall.csv` now has plain column labels (the report had raw model ids, which the gate forbids); `comparison_stage` rerun to refresh it.
- Glossary: Cohen's kappa was already gone; added Shortcut learning (also target leakage) and Per-website recall.
- References added (all well known, cited in the matching Report parts): Hastie et al. 2009, Manning et al. 2008, McNemar 1947, Efron & Tibshirani 1993, Pedregosa et al. 2011.
- **Skipped:** no 9/11-conspiracy background sources were added, because none could be confirmed without a web check (no invented references). Still a team decision.
- **[superseded]** **Skipped:** I did not run `fml run-all` twice or the full test suite in this task (the task rules say the harness does that); only the glossary, gates, privacy, report-numbers, comparison, sections and sitegen tests were run (86 passed).

## Cycle 3, T11 (two identical runs, full test suite)
- **[superseded]** **Skipped:** `fml run-all` twice, the checksum diff and `pytest -q` were NOT run. The command to start them was denied (no approval surface in this session) and the task rules forbid running them inside a task. No determinism check, run time or test count exists for cycle 3. `docs/reproducibility.md` says the same and lists only the committed checksums.
- Needs a team decision / next run: execute the two runs and the test suite by hand (or via the harness) and paste the results into `docs/reproducibility.md`.

## D8 background sources on 9/11 conspiracy content (declined, cycle 3 T3)
- WebSearch was denied in this session (no approval surface), so no extra source could be verified. No reference was added from memory; the 01 introduction keeps only claims supported by Miani et al. (2022) and Fong et al. (2021). Still a team decision: a member can verify one or two sources and add them.
- 01 "Explained simply" trimmed (method table marker removed there; it stays in the Report).

## Cycle 3, T4 (01 introduction)
- **[superseded]** The "platforms and fact-checkers rely on automatic filters" claim was dropped: no published source could be confirmed (no web check available), so no reference was invented. The "inside job / controlled demolition / false flag" framing is cited only to Miani et al. (2022) as general 9/11 material.
- No textbook references were added beyond those already listed (Hastie et al. 2009, Manning et al. 2008, McNemar 1947); Manning et al. is now cited in the 01 introduction for TF-IDF.
- Methods table headers are plain words ("Trained on our data"); the CSV was rewritten from existing results without rerunning the pipeline.

## Cycle 3, T5 (website counts, headers, website names)
- Website counts: 39 conspiracy websites is the count before cleaning; the split tables count what is left after cleaning (25 + 6 + 6 = 37). The code was right; the 02 report now gives both numbers and 03 says "after cleaning".
- Headers of `cleaning_log.csv` and `split_summary.csv` are plain words (set in `pipeline.py`; the CSVs were rewritten from existing results without a pipeline rerun).
- Website names: real domains next to labels in 08 are replaced by "Website 1-6" (alphabetical order of the six test conspiracy websites). `test_website_recall.csv` and `metrics.json#comparison` use the same labels. Conservative choice: naming sites next to a "conspiracy" label could be read as a judgement on real publishers. Domains elsewhere (e.g. error examples) are left to the queued 08 tasks.

## Cycle 3, T11 (privacy follow-ups, outstanding)
- **Outstanding:** `scripts/collect_feedback.py` still has a personal GitHub account name as the default of `--repo`, and `repo_url` is not set. Added as owner task 9 in `docs/human-todo.md`.
- **Why the agent cannot do it:** `scripts/` and `config/site.yaml` are protected paths (AGENTS.md); only the repo owner may edit them. The account name is not repeated here.
- Checked `reports/` and `content/` for names, e-mails and usernames (see the T11 summary); `tests/test_privacy.py` was run.


## Cycle 3, T5 (02 report, references)
- 02 Report: added two sentences on how LOCO was built and rated (keyword retrieval, website-level subcorpus, authors' blind check 85% / 92%), all from `docs/PROJECT_PLAN.md` section 2; licence left out. Library versions copied from `results/run_info.json`. Trimmed the cleaning paragraph.
- **[superseded]** References (D8) **declined**: no 9/11 background source and no extra source for regularised logistic regression could be confirmed from existing files, and no web check is possible here (no invented citations). Hastie et al. (2009) already covers regularised logistic regression. `content/references.md` has no "[open]" markers.

## Cycle 3, owner polish (2026-10-26)

- **Verdict rule.** A hypothesis is supported when the difference points the predicted way and McNemar's test gives
  p < 0.05 (Dietterich, 1998). Overlapping single-model intervals no longer make a verdict "weakly supported"; H1
  (p = 0.029) is reported as supported, with the remark that it would not pass a stricter threshold for three tests.
- **Runtime cell.** `methods_overview.csv` gives runtimes in words ("a few seconds", "about a minute"), so two
  runs of `fml run-all` give identical files. The harness now runs `fml run-all` twice and fails a gate if any
  result file differs (except `run_info.json`).
- **Baseline row.** A score without spread is shown without an interval.
- **MATTR.** Rewritten as a plain sliding window; same values (tests unchanged and passing).
- **Verified state (26 Oct 2026).** In the pinned environment (`requirements-lock.txt`, Python 3.11) the full test suite
  passed with no test skipped, and two consecutive `fml run-all` runs gave byte-identical results
  (`docs/reproducibility.md`). This replaces the earlier entries marked "superseded".
- **Report length.** Measured on the assembled report before References: 4,047 words of prose (tables and figure
  lines excluded, the gate's rule) and 4,869 words by a plain whitespace count with tables. Both are inside
  3,000-5,000; R8 (about 4,500) is met by the prose count.
- **Examples in the report.** The six-row examples table stays on the site (section 08, The code); the report quotes
  two of its rows in prose to save length.
- **Check sheet without website names (26 Oct 2026).** `stance_check_sample.csv` now holds only `doc_id` and
  `human_stance`; the same 40 ids in the same order. Coders never see the website.
- **Hand-check answers.** Each member's downloaded CSV goes to `data/hand_check/stance_check_member_<letter>.csv`
  (ids, letter, stance only). Member A's file is in; the analysis of agreement is added once all five are in.
- **AI-use log vs run records.** The harness's automatic end-of-cycle record moved to `docs/agent-runs.md`;
  `docs/ai-log.md` keeps only the factual Produced / Checked / Learned summary used for the report.
- **Earlier test scorings (seven).** Before the final models, these ids were scored on test and are not reported:
  `lr_words`, `lr_extended`, `llm_haiku`, `lr_words_v2`, `final_word_counts`, `final_tfidf`, `final_tfidf_features`
  (`results/test_access_log.jsonl`). Reported: `final2_word_counts`, `final2_tfidf`, `final2_tfidf_features`, `final_llm`.
- **Declined from the second review (26 Oct 2026).** More excerpt examples in the report: declined to keep the report
  near 4,500 words; the site shows six examples (section 08, The code) and the report names one label problem and one
  model error.
- **Account name.** `scripts/collect_feedback.py` has no account name since 26 Oct 2026 (merged in PR #25).
