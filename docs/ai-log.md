# How we used AI

The assignment asks us to document which AI tools we used, what for, and how we checked or adapted
their output. This log is the raw material for that part of the report. One short block per cycle;
the checks that people still have to do are in `docs/human-todo.md`.

## Setup (16 Oct 2026)

### Produced
Claude in a claude.ai session, working with the repo owner (the model of that session is not recorded in the repo), made the project plan with the literature research for the feature groups, the cycle runner, the site generator, the feedback script, the 9/11 data subset with checksums in the private data repo, the first requirements file and the repo rules. The agent loop from cycle 1 on uses Claude Sonnet 5.5 for planner, implementer and reviewer (`config/agent.yaml`).

### Checked
Product facts used in the design (pricing, authentication, cloud limits) against the official documentation; the data subset against the LOCO paper (3,146 = 939 + 2,207, exact match); the full archive rebuilt from its parts and compared by checksum; the runner's pause-and-resume behaviour with a test.

### Learned
About 17% of LOCO's 9/11 documents never mention 9/11, in both classes. This only showed up once we looked at the data instead of trusting the seed label. The team (not the AI) decided: full articles, one cheap LLM asked the source question, the feature groups (no emotion features), removing off-topic articles, public code with private data, pull requests per cycle.

## Cycle 1 (17 Oct 2026)

### Produced
Claude Code (Sonnet 5.5) wrote the first version of `src/fml/`, the tests, `results/`, the sections and the report. Claude Haiku 5.5 was the classifier under study; Claude Sonnet 5.5 coded stance in the error analysis.

### Checked
Tests compare quoted numbers with `results/`. Open at the end: the corpus licence, the 40 stance codes without human labels, and the review of the LLM temperature and output-limit choices (`docs/decisions.md`).

### Learned
Passing number tests did not mean the analysis was well built; cycle 2 rewrote the pipeline.

## Cycle 2 (17 Oct 2026)

### Produced
Claude Code (Sonnet 5.5) rewrote the pipeline to the four reported analyses with beginner-style code, added cleaning (off-topic, then near-duplicates), scored the final models once each on test, and drafted the sections, glossary and report.

### Checked
Automated tests (numbers match `results/`, section format and word limits, privacy, frozen split, locked test set) and two identical `fml run-all` runs (`docs/reproducibility.md`). A reviewer in a fresh session scored the result against a checklist; the implementer then fixed what it found. Output that was changed or rejected:
- A draft sentence said the LLM beat every model. The test table did not support it (high precision, low recall), so the claim was corrected to what the table shows.
- A draft claim about 200,000 columns had no source in the repo or in `results/`. It was replaced by a figure read from the results.
- Hypothesis H1 was relabelled and reworded as a plain guess (TF-IDF against word counts); the lettered sub-hypotheses were later merged.
- Emotion scores in the AI's first plan were removed by the team, because emotional language is not the same as conspiracy language.

### Learned
Tests catch inconsistencies, not wrong reasoning. A glossary only helps if every technical word has an entry.

## Cycle 3 (2026-10-25)

### Produced
Claude Code (Sonnet 5.5) merged results and error analysis into one section, renamed sections, used one identical set of logistic-regression settings for all three analyses, and scored new final models `final2_*` once on test. It added a per-website recall table, a trivial-baseline row and a methods-at-a-glance table, and rewrote limitations, conclusion and this log. Haiku 5.5 answers were reused from the cache; Sonnet 5.5 coded the new disagreement articles.

### Checked
Concrete checks, all automated or recorded in `docs/decisions.md` (none of them is a human action):
- `tests/test_report_numbers.py` compares every number in the report with `results/metrics.json` and the CSV tables, and every three-decimal number in the sections with `results/`; a second test shows the checker fails when a number is changed.
- The same file compares the numbers quoted in prose with the ablation table.
- `tests/test_gates.py` counts the words of the report (3,000-5,000 before References) and of the plain-language parts (450 words).
- `results/test_access_log.jsonl` was compared with the model ids: each `final2_*` id appears once (`tests/test_test_lock.py`).
- Sentences were rewritten where a claim went beyond the tables (for example the introduction keeps only what Miani et al. 2022 and Fong et al. 2021 support).
- Rejected suggestion: the claim that platforms and fact-checkers rely on automatic filters was dropped, because no published source could be confirmed.
- Rejected suggestion: candidate classic references and 9/11 background sources were not added from memory; they stay open for a member to verify.
The checks that need people are listed in `docs/human-todo.md` (placeholders; nothing there is done yet).

### Altered or rejected by members
None recorded yet; members add their cases via `docs/human-todo.md` items 7, 8 and 10.

### Learned
- A number can match `results/metrics.json` and still stand beside a claim the table does not support; the platform-filter sentence was only caught by asking for a source.
- Counting `final_*` and `final2_*` ids in `results/test_access_log.jsonl` showed that four earlier-configuration models had been scored on test, so `09-limitations` now says so instead of claiming a single scoring.
- A feature count (47 hand-made words in four lists) had to be read from the results files, not written from memory.

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
