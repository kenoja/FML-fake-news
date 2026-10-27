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

## Cycle 4 - agent run (2026-10-27)

**Verdict:** STOPPED BELOW 10/10 (round limit reached) after 4 review round(s).

**Requirements:** `requirements/cycle-04.md`

### What the AI produced (tasks completed)
- T1: Site-concentration counts in the data stage (plan)
- T2: Section 02: search-word list, concentration numbers, 37 vs 39, library version (plan)
- T3: Sections 01 and 03: plan changes, evaluation subsection, F1 vs McNemar, multiple tests, risk of one large site (plan)
- T4: Section 06: when the feature model was fixed, why not the best group (plan)
- T5: Section 08: cautious label-problem figure, interval reason, F1/McNemar clause, plain table headers (plan)
- T6: Section 09 and 10 consistency (37 websites, no repetition) (plan)
- T7: Section 11 and docs/ai-log.md: factual learnings and tool roles (plan)
- T8: Beginner-readable style.py with identical results (plan)
- T9: Glossary and cross-check of new terms (plan)
- T10: Report length to at most 4,900 words (plan)
- T11: Full re-run, reproducibility evidence, decisions log and README (plan)
- T12: Final verification pass: tests, gates, privacy, number tie-out (plan)
- gate-fix-1: Fix failed hard gates (gate)
- gate-fix-2: Fix failed hard gates (gate)
- T1: Fix AI-use wording: four cycles, factual learnings, tool mapping (R6) (review)
- T2: Verify 01: Explained-simply length, D3, R1, R4, plan-changes paragraph (review)
- T3: Verify 02/03/09: topic words, top-3 table, 37 vs 39, library version, large-site risk (review)
- T4: Section 06: D2 and R3 wording (review)
- T5: Section 08: summary table, R2 order, D3 clause, interval reason, trim (review)
- T6: Interpretation: quantify 'somewhat too high' in Limitations (review)
- T7: R7: simplify style.py without changing results (review)
- T8: Check teaching gates: glossary, Explained-simply limits, Check-yourself quality (review)
- T9: Privacy and excerpt audit (review)
- T10: Reproducibility: README, pins, identical double run (review)
- T11: Report length and decisions log; approach alternatives check (review)
- T12: Final gate run and number check (review)
- gate-fix-1: Fix failed hard gates (gate)
- T1: Baseline check: gate word count, hash snapshot, privacy grep (review)
- T2: 01: reorder evaluation subsection, gloss jargon, add alternatives costs (review)
- T3: 02/03: search-word count, library version, split table label, why website grouping (review)
- T4: 06: split dense final paragraph, add R3/D2 and ablation sentence (review)
- T5: style.py: beginner-readable code, results unchanged (review)
- T6: 08 Report: reconcile drop, Website 4 wording, error-estimate framing, intervals once (review)
- T7: 09, 10: cross-reference narrow intervals, soften titles inference, test-set sentence (review)
- T8: 11 and ai-log: tools, learnings, what the team changed (review)
- T9: Tables: plain headers (R5) and ablation blank row (review)
- T10: Glossary: terms and links check (review)
- T11: Trim report to <=4,900 words and record decisions (review)
- T12: Final verification: two identical run-all runs, full tests (review)
- gate-fix-1: Fix failed hard gates (gate)
- T1: Measure report length and record the baseline (review)
- T2: Cut repeated hedges in report parts 04, 05, 06, 07, 08 (review)
- T3: Plain executive summary and introduction in 01 (review)
- T4: LLM call and licence wording in 02 / 07 (review)
- T5: Feature section: D2, R3 and softened ablation claim (review)
- T6: Results chapter 08: interval caveat, tentative Website 4, cited numbers, error rule (review)
- T7: Glossary additions and link check (review)
- T8: Code readability: style.py and shown helpers (review)
- T9: AI-use section and ai-log condensed (review)
- T10: README reproducibility instructions (review)
- T11: Privacy and excerpt audit of the whole site (review)
- T12: Final rebuild, determinism and gates (review)

### What was checked
Hard gates still failing:
- none

| criterion | score | evidence |
|---|---|---|
| ai_use | 8 | docs/ai-log.md is factual Produced/Checked/Learned per cycle with no failed-run diary. Report 'How we used AI' covers the tools, steering, checking and learning |
| approach | 9 | Four analyses explained step by step with the methods table (report 27–32). Alternatives not taken are named at line 19: other classifiers, fine-tuning, emotion |
| code_readability | 8 | The cells shown in 06 are small, named functions (count_markers, mattr, per_n_words, run_ablation). I only skimmed the code; I did not open src/fml. The harness |
| errors | 8 | The report gives two real excerpts (short, in the 50-word limit), a split table of model errors versus label problems (37/4/29), the cautious lead figure 21 of  |
| evaluation | 9 | One website-grouped split, train-only fitting, validation selection, test scored once per final model, and one honest sentence on earlier test use (report 301). |
| features | 9 | 06 explains each group with literature (Wang & Manning; Marini & Jezek; Horne & Adali), an ablation table of six rows on validation, and a verdict (H2 not suppo |
| feedback_compliance | 8 | Checked: item 1 (plan changes, report line 19), 2/D3 (line 40), 3 (ten search terms, line 46), 4 (157 and 20.8%, line 60), 5 (37 vs 39, line 60), 7/D2 (line 134 |
| interpretation | 9 | Limitations (report 291–301) covers labels describing websites, few websites, LLM prior knowledge, lexicon and negation limits, and one split. It says plainly t |
| privacy | 9 | The report uses 'Member'-free text, 'Website 1–6' numbering, no names, no institution (the course is called 'the course feedback'), and short excerpts (<50 word |
| question | 9 | report.md: H1–H3 numbered without gaps, each answered in Results (lines 203–207) and Conclusion (305). Separate 'How we judge the methods' subsection (R1) with  |
| report | 8 | The report is coherent and covers goal, resources, approach, evaluation, findings, limitations, AI use and references. The body is below the 5,000 gate accordin |
| reproducibility | 9 | The harness gave run-all in 164 s (<8 min), two runs with identical results over 55 files, 240 tests passing. The seed is in metrics (20261016). docs/reproducib |
| resources | 9 | Data and cleaning (report 44–60) describes LOCO, website labels, the 9/11 subset, the ten search terms, the near-duplicate threshold with its reason, the librar |
| results | 8 | All four test results appear together in one table with CIs plus the baseline, in the Results section, after all runs. Numbers checked against the McNemar CSV,  |
| teaching | 8 | Sections follow the four-part format; the 06 'Explained simply' reads in plain words with analogies (cake/sugar, per-1,000-word example). I could not check the  |

### Open reviewer issues
- [results] The same sentence about the repeated validation run (0.992) appears three times, once as a stray bold stand-alone line. The 'Why the scores are lower…' bold heading is followed by a table with no lead-in. (reports/report.md lines 152, 209–228 (source: content/sections/08-results-and-why-they-fail.md Report part))
- [report] Duplicates and leftover blank structure make the report look assembled rather than written; the report body is close to the 5,000-word gate. (reports/report.md 8, 08 sections)
- [question] The introduction packs TF-IDF, zero-shot and n-grams into one dense paragraph before they are explained. (reports/report.md lines 7, 13)
- [resources] The reason for the 0.9 near-duplicate threshold and the effect on test are brief. (report.md line 46)
- [approach] The switch from headlines to full articles and the course's one-LLM requirement are only implied in a one-line list of non-choices. (report.md line 19)
- [evaluation] The repeat-run agreement figure (0.992) is a statistic outside the permitted set. (report.md 152, 224–228)
- [errors] Only two excerpt examples; the circularity (both-wrong conspiracy-site articles count as label problems by rule) is hard to follow. (report.md 273, 287)
- [ai_use] The 'Learned' line is repeated almost verbatim in cycles 3 and 4 and the report's 'what the team changed' is generic. (docs/ai-log.md Cycle 3/4; report.md 317)
- [teaching] I could not confirm from the sources that every section keeps the plain tone and that all terms are linked; the report-style terms (bootstrap, McNemar) may appear without a plain gloss in Explained simply. (content/sections/08, 01 Explained simply)
- [code_readability] Not verified: shown cells could still have long docstrings or unused leftovers. (src/fml/ablation.py, pipeline.py)
- [reproducibility] Pinned dependencies and README run instructions were not verified here. (pyproject/requirements, README.md)
- [privacy] Not every section/site HTML was scanned for identifiers. (site/index.html)
- [feedback_compliance] Item 1's reasons (full articles over headlines, one LLM per course feedback) are brief, and the Prompt table shows bare 'v1/v2/v3' without saying what each is. (report.md lines 19 and 144–148)

### Reviewer summary
Overall the work is strong and consistent: every number I checked (cleaning counts, McNemar table, test table, ablation) matches the results files. The cycle-4 items are visibly handled: D2 and R3 in 06, R1 and R4 in 01, R2 in 08, 37 versus 39 websites, and the named search terms. The main defects are cosmetic but visible in the report: the 0.992 repeat-run sentence is repeated three times, including once as a stray bold line, and the 'Why the scores are lower' heading has no lead-in. The one statistic beyond the permitted set (0.992) should go. A few explanations (introduction jargon, error-rule circularity, thin AI-log learnings) could be plainer. I could not run commands, so code readability, pinned dependencies and the full privacy scan were only sampled, not verified.
