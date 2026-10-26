# Agent run records

Written automatically by the agent at the end of each cycle: tasks, gate results and the reviewer's
scores. The AI-use documentation for the report is `docs/ai-log.md`.

## Cycle 3 - agent run (2026-10-26)

**Verdict:** STOPPED BELOW 10/10 (round limit reached) after 4 review round(s).

**Requirements:** `requirements/cycle-03.md`

### What the AI produced (tasks completed)
- T1: Same 3,000-word text and same C grid for all logistic regressions (plan)
- T2: Analysis 3 = TF-IDF plus all three feature groups; ablation table validation-only (plan)
- T3: Remove Cohen's kappa and unused code; plain error-table labels (plan)
- T4: Final2 run: score three logistic regressions on test once, per-website table, trivial baseline (plan)
- T5: Beginner-style code in all shown cells (plan)
- T6: Rename and merge sections to the new slugs (plan)
- T7: Write 01-question-and-hypotheses (summary, introduction, H1–H3, methods table) (plan)
- T8: Rewrite 02-data-and-cleaning and 03-split-by-website (plan)
- T9: Rewrite 04–07 as validation-only method sections (plan)
- T10: Write 08-results-and-why-they-fail (plan)
- T11: Rewrite 09-limitations, 10-conclusion, 11-how-we-used-ai, 00-how-to-use, docs (plan)
- T12: Glossary, references, report length and final gates (plan)
- gate-fix-1: Fix failed hard gates (gate)
- gate-fix-2: Fix failed hard gates (gate)
- T1: Fix false LLM-vs-LR per-website claim in 08 (review)
- T2: One directional wording for H3 and a consistent hypothesis-status rule (review)
- T3: Trim 01 Explained simply under 450 words and handle 9/11 background sources (D8) (review)
- T4: Clarify near-duplicate rule and clue-removal counts in 02 (review)
- T5: Report actual feature column count in 06 and soften causal wording (review)
- T6: Refactor long stage functions and delete stale prediction files (review)
- T7: Restore error table and describe the hand-check sheet in 08 (review)
- T8: Teaching quality in 08 and 01: examples, check questions, glossary (review)
- T9: Trim report to about 4,500 words (review)
- T10: Condense AI-use log and split human to-do (review)
- T11: Privacy and decisions.md cleanup (review)
- T12: Two identical run-all runs and full test suite; update reproducibility doc (review)
- gate-fix-1: Fix failed hard gates (gate)
- T1: Simplify long functions and remove dead ablation code (review)
- T2: Add validation-only ablation without n-grams and TF-IDF-features analysis (review)
- T3: Strip cross-model verdicts from sections 04, 05, 07 (review)
- T4: Rewrite 01 introduction, method table and training environment (review)
- T5: Fix website-count inconsistency and tidy 02/03 report (review)
- T6: Rework section 08: McNemar justification, tests table, validation vs test (review)
- T7: Expand error examples and explain the hand-check sheet (review)
- T8: Qualify LLM validation score and finish 09/10 wording (review)
- T9: Glossary, jargon pass and decisions cleanup (review)
- T10: Update AI-use log and section 11 (review)
- T11: Privacy follow-ups for protected script (review)
- T12: Run pipeline twice, document reproducibility, build report, commit (review)
- gate-fix-1: Fix failed hard gates (gate)
- gate-fix-2: Fix failed hard gates (gate)
- T1: Shorten over-long functions and remove unused summary code (review)
- T2: Section 08: show disagreement examples and per-website sentences (review)
- T3: Section 08/09: validation-vs-test evidence and label-check wording (review)
- T4: One consistent H1 verdict and the D13 methods table in section 01 (review)
- T5: Report text for data, LOCO, tools and references (D8) (review)
- T6: Explain logistic regression and the choice of C in the report (review)
- T7: Features: discuss certainty words and lexicon limits (review)
- T8: Limitations: earlier test scorings and AI-section wording (review)
- T9: Privacy follow-ups (review)
- T10: Trim the report to about 4,500 words (review)
- T11: Reproducibility: two identical runs and full test suite (review)
- gate-fix-1: Fix failed hard gates (gate)

### What was checked
Hard gates still failing:
- none

| criterion | score | evidence |
|---|---|---|
| ai_use | 7 | docs/ai-log.md has a Produced / Checked / Learned block per cycle, mostly factual, and report 'How we used AI' explains the planner/implementer/reviewer setup ( |
| approach | 8 | 01 and the report give the ladder of four methods, an expectation for each, and the alternatives not taken (no other classifiers, no fine-tuning, no emotion fea |
| code_readability | 6 | Most cells are short with plain names. But src/fml/features/style.py#mattr (l.58–74) uses numpy tricks (np.unique inverse, previous_positions, clip arithmetic)  |
| errors | 6 | The report has a per-website recall table, a three-group error table with plain labels, and six excerpt examples. The excerpts are within limits. The label-prob |
| evaluation | 8 | One website-grouped split, training-only fitting, a shared C grid with balanced class weights, the 3,000-word cut for all, and test scored once for each final2_ |
| features | 7 | Each group has literature support (Wang & Manning 2012; Horne & Adali 2017; Marini & Jezek 2024; Islam 2020; Fong 2021) and a six-row ablation table on validati |
| feedback_compliance | 6 | Done: D1 slugs, D4 critical LLM assessment (report l.165), D5 summary/intro split, D6 numbered hypotheses, D10 shared settings and final2 rerun, D11 per-website |
| interpretation | 8 | 09 covers website-vs-text labels, six or fewer websites per split with one site giving 44 of 96 test conspiracy articles, LLM prior knowledge, lexicon limits (4 |
| privacy | 8 | The report and the section grep show no names, e-mails, institution, or course; test websites are anonymised as Website 1–6; excerpts are short. Open items: scr |
| question | 8 | 01 Report: H1–H3 numbered without gaps, each written in full, each answered in 08 and the conclusion (reports/report.md l.27, 207–211, 293–297). Executive summa |
| report | 6 | The report is coherent and covers goal, resources, approach, evaluation, findings, limitations and AI use. By my estimate it is about 4,900 words before Referen |
| reproducibility | 3 | docs/reproducibility.md says outright that for cycle 3 there is no two-run comparison, no new test count and no test-suite run. decisions.md confirms that 'fml  |
| resources | 7 | Report 'Data and cleaning' describes LOCO, the website-level labels, the 85%/92% blind check, 3,146→2,608→2,554, the reason for each step, and pandas/scikit-lea |
| results | 8 | All four models plus the always-mainstream baseline are shown together in 08. Spot-checks match results/tables/comparison_test.csv (e.g. LLM F1 0.795, 0.723–0.8 |
| teaching | 7 | All sections have the four-part format, and the glossary has ~60 entries including MATTR, ablation, bootstrap, exact test and shortcut learning. Weaknesses: 'Ex |

### Open reviewer issues
- [reproducibility] No evidence for cycle 3 that two consecutive `fml run-all` runs are identical; the full test suite was not run; docs/reproducibility.md quotes stale timing (96.5 s vs 108.1 s in run_info.json). (docs/reproducibility.md; docs/decisions.md 'Cycle 3, T11 (two identical runs…)')
- [report] Report is about 4,900 words before References, not about 4,500 (R8), and the introduction has one background source. (reports/report.md (sections 02, 06, 08 are longest))
- [feedback_compliance] D8 not met: no classic method references beyond Hastie, Manning, McNemar, Efron and no published 9/11 background sources; D13 table missing from the 01 'Explained simply'. (content/references.md; content/sections/01-question-and-hypotheses.md)
- [code_readability] `mattr` is vectorised with numpy tricks and is not beginner-readable; ablation_variants has a long docstring; the 06 text says 'five models' while there are six. (src/fml/features/style.py:58-74; src/fml/pipeline.py:323; content/sections/06-feature-engineering.md:26)
- [errors] The label-problem share rests on one automatic coder and the 40-article check sample is from validation, not the test errors; examples don't show why models erred. (reports/report.md l.242–275; results/tables/stance_check_sample.csv)
- [features] The 'why is all-three worse than TF-IDF' question is left as untested speculation, and simple text and report read the last ablation row differently. (content/sections/06-feature-engineering.md l.20, 77)
- [interpretation] The development-use sentence names internal ids and process (`final_*`, `final2_*`). (content/sections/09-limitations.md l.26, 46)
- [ai_use] Log 'Learned' lines read as a diary and member checks are empty. (docs/ai-log.md Cycle 1–3 'Learned'; report 'How we used AI')
- [teaching] Open team items remain; glossary order and some terms in 'Explained simply' are not plain. (docs/decisions.md 'Needs a team decision'; content/sections/99-glossary.md l.217-222; 06 simple text)
- [resources] The LLM and its access are barely described in the report, and the licence check stays [open]. (reports/report.md section 07 and 02; docs/decisions.md)
- [question] The motivation is thin and 'weakly supported' for a significant McNemar result is confusing. (reports/report.md l.11, 28, 207)
- [approach] The reason for dropping emotion features differs between site and report. (content/sections/01-question-and-hypotheses.md l.19 vs reports/report.md l.23)
- [evaluation] Honest sentence is hidden under process detail; hand-check sample does not cover test errors. (09-limitations; stance_check_sample.csv)
- [results] Baseline row shows meaningless CIs '0.000 (0.000 to 0.000)'. (reports/report.md l.179)
- [privacy] The owner account name remains in a protected script; domains sit beside labels in the check sheet. (scripts/collect_feedback.py; results/tables/stance_check_sample.csv)

### Reviewer summary
The cycle-3 content is largely in place. The sections are merged, H1–H3 are numbered and answered, the shared-settings rerun and per-website recall table exist, and the numbers I spot-checked match results/*.json (comparison_test.csv, run_info.json, test log). I could not run Bash, so I did not run the tests or the pipeline. The main gaps are reproducibility (no two-run check and no test run for cycle 3, and a stale timing in docs/reproducibility.md), a report of about 4,900 words against the aim of 4,500, missing classic and 9/11 references (D8), the four-methods table absent from the 01 'Explained simply' (D13), a non-beginner `mattr`, an error analysis that rests on one automatic coder, and process-flavoured wording in limitations and the AI log.

