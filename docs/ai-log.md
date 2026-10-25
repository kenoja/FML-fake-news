# How we used AI

The assignment asks us to document which AI tools we used, what for, and how we checked or adapted
their output. This log is the raw material for that part of the report. An entry is added
after every cycle; team members add what they checked or changed by hand.

## Setup (16 Oct 2026)

- **Tool:** Claude in a claude.ai session, working with the repo owner (the model of that session is not recorded in the repo). The agent loop that ran from cycle 1 on uses Claude Sonnet 5.5 for planner, implementer and reviewer (`config/agent.yaml`).
- **What it produced:** the project plan (`docs/PROJECT_PLAN.md`) including the literature research for
  the feature groups; the cycle runner, the site generator (`sitegen/`), the feedback
  collection script and workflows; the 9/11 data subset and its checksums in the private data repo;
  the first requirements file and this repo's rules (`AGENTS.md`).
- **What was checked:** every product fact used in the design (pricing, authentication, cloud limits)
  against the official documentation; the data subset against the counts in the LOCO paper (3,146 = 939
  + 2,207, matching exactly); the full archive rebuilt from its parts and compared by checksum; the
  runner's pause-and-resume behaviour with a test; a nested Claude Code call inside a cloud session.
- **What the team decided (not the AI):** full articles, one cheap LLM asked the source question,
  feature groups (no emotion features), removing off-topic articles after the first run, public code with
  private data, running in Anthropic's cloud, pull requests per cycle.
- **What we learned:** about 17% of LOCO's 9/11 documents never mention 9/11 at all, in both classes.
  That only showed up once we looked at the data instead of trusting the seed label.


## Cycle 1 (17 Oct 2026)

- **Tool:** Claude Code (Sonnet 5.5) working task by task under the repository rules; Claude Haiku 5.5 as the classifier under study; Claude Sonnet 5.5 for stance coding in the error analysis.
- **What it produced:** the first version of `src/fml/`, the tests, `results/`, the notebook sections and the report.
- **What was checked:** `tests/test_report_numbers.py` and `tests/test_section_numbers.py` compare quoted numbers with `results/`. Cycle 1 ended with open points: the LOCO licence (OSF page unreachable), the 40 stance codes without human labels (kappa `null`), and the review of the LLM temperature and output-limit choices. Details are in `docs/decisions.md`.

## Cycle 2 (17 Oct 2026)

- **Produced:** Claude Code (Sonnet 5.5) rewrote the pipeline to the four reported analyses with beginner-style code, added cleaning (off-topic, then near-duplicates), scored the final models once each on test, and drafted the notebook sections, glossary and report. Claude Haiku 5.5 is the classifier under study, not a helper.
- **Checked:** the automated tests (numbers in text match `results/`, section format and word limits, privacy, frozen split, locked test set) and two identical `fml run-all` runs (`docs/reproducibility.md`). These show that numbers match files, not that they mean the right thing. Still pending for humans: the checks in the table below.
- **Learned:** tests catch inconsistencies, not wrong reasoning; a glossary only helps if every technical word has an entry; the account name in the repo link and the corpus licence stay open (`docs/decisions.md`, "Needs a team decision").

### Human checks (all pending; Members A-E fill in date and what was checked)

| # | Check | Status |
|---|---|---|
| 1 | Spot-check about 10 numbers in the report against `results/metrics.json` after the last `fml run-all` | pending |
| 2 | Fill `human_stance` for the 40 rows of `results/tables/stance_check_sample.csv` | pending |
| 3 | Confirm the LOCO licence at the OSF page and record it in `docs/decisions.md` | pending |
| 4 | Read the report and sections once for article text and personal data (including the repo link) | pending |
| 5 | Decide the open points under "Needs a team decision" in `docs/decisions.md` | pending |
| 6 | Read the glossary and section 11 for plain-language understanding | pending |

## Cycle 2 - agent run (2026-10-17)

**Verdict:** STOPPED BELOW 10/10 (round limit reached) after 4 review round(s).

**Requirements:** `requirements/cycle-02.md`

### What the AI produced (tasks completed)
- T1: Config, conflicts and open items in decisions.md (plan)
- T2: Cleaning: off-topic plus near-duplicates, drop the first all-documents run (plan)
- T3: Remove dropped analyses, make outputs deterministic, beginner-style core code (plan)
- T4: Analyses 1 and 2: word counts and TF-IDF on validation (plan)
- T5: Analysis 3: TF-IDF plus engineered features, one comparison per group (plan)
- T6: Analysis 4: the LLM classifier on validation (plan)
- T7: One final run: score the four final models on test once (plan)
- T8: Comparison: one table, confusion matrices, three McNemar tests (plan)
- T9: Error analysis and the 40-article human check sheet (plan)
- T10: Sections 00, 01, 02, 03 and 10, 11 rewritten for the audience (plan)
- T11: Glossary, AI-use section and ai-log (plan)
- T12: Final gates: report size, number check, reproducibility, README (plan)
- gate-fix-1: Fix failed hard gates (gate)
- gate-fix-2: Fix failed hard gates (gate)
- gate-fix-3: Fix failed hard gates (gate)
- T1: Refactor shown code into beginner-sized functions (review)
- T2: Same small grid for all five feature models (review)
- T3: Reproducibility evidence and version pinning (review)
- T4: Error analysis: examples, blind check sheet, drop unparsed (review)
- T5: Sections 02/03: data source, licence and split table after cleaning (review)
- T6: Sections 01/04/05/06: hypotheses, features verdict, alternatives not taken (review)
- T7: Section 07 and 10: LLM setup, truncation, probe (review)
- T8: Compact results tables and report length <=5,000 words (review)
- T9: References file (review)
- T10: Glossary and Check-yourself questions (review)
- T11: Interpretation, wording of limitations, decisions cleanup (review)
- T12: AI-use log and privacy (GitHub name) (review)
- gate-fix-1: Fix failed hard gates (gate)
- T1: Drop out-of-scope analyses (website bootstrap, prior-knowledge probe) from code and metrics (review)
- T2: Split StyleFeatures and long stage functions into short helpers (review)
- T3: Make stance-check sample cover several websites (review)
- T4: Strengthen number-vs-results test and fix section 06 numbers, validation-only verdict (review)
- T5: Rewrite hypotheses as plain guesses (review)
- T6: Report self-containment: tools, approach alternatives, licence wording, cross-references (review)
- T7: Section 08: split explanations, test result lives here (review)
- T8: Section 09 and 10: excerpts, LLM validation-to-test drop, limitations (review)
- T9: Section 12 and ai-log: facts only, no process talk (review)
- T10: Teaching pass and glossary linking (review)
- T11: Privacy: neutral repo identity and declined-item notes (review)
- T12: Final reproducible run: run-all twice, rebuild, test, commit (review)
- T1: Simplify models.py and pipeline fitting paths (code readability) (review)
- T2: LLM module: named prompts function, cell by name (review)
- T3: Hypotheses honest and tested separately (01, 08, 11) (review)
- T4: Data section: provenance, tools, licence, near-duplicate rule (02, 03) (review)
- T5: Approach, features verdict and example (04, 06) (review)
- T6: Results: consistent decimals, worked example, bootstrap caveat (08) (review)
- T7: Errors: better examples, plain table, hand-back instructions (09) (review)
- T8: Limitations: test-set sentence and LLM drop (10) (review)
- T9: Report length, headline result, section order (review)
- T10: Glossary cleanup (review)
- T11: README, reproducibility evidence, decisions.md, ai-log (review)
- T12: Final run, test-lock check and full verification (review)
- gate-fix-1: Fix failed hard gates (gate)

### What was checked
Hard gates still failing:
- none

| criterion | score | evidence |
|---|---|---|
| ai_use | 7 | 12 and docs/ai-log.md are factual and short, with no failure diary. But the ai-log pairs 'Cycle 2 Produced' with all six human checks 'pending'; the model name  |
| approach | 8 | 04–07 explain each analysis step by step and name the dropped alternatives: no other classifiers, no fine-tuning, no emotion features (report 'Word counts', 06) |
| code_readability | 6 | decisions.md itself records 'Requirement 14 ... partly met ... not confirmed complete ... queued code task (T1)'. Dropped code is still in src: errors.py:293 st |
| errors | 6 | 09 has the error table, four examples with short excerpts, a model-error vs label-problem split, and 40-row stance_check_sample.csv with instructions. Problems: |
| evaluation | 7 | One website-grouped split; final models scored once under final_* ids (test_access_log.jsonl lines 5–8); the test-set sentence is in 10-limitations. Statistics  |
| features | 8 | 06 gives literature reasons per group (Wang & Manning, Marini & Jezek, Fong, Horne & Adali, Pérez-Rosas), a five-row validation table (ablation) and a clear ver |
| feedback_compliance | 6 | Most of the 18 requirements are addressed (structure, four analyses, final run, results table, stance sheet, glossary, run-all reproducibility). Misses: req 5/R |
| interpretation | 8 | 10 covers labels describing websites, few websites, LLM prior knowledge, lexicon limits, style shortcuts, one topic and the test-set sentence, all honestly; it  |
| privacy | 8 | No names, institution or course found in sections or report; excerpts are short (4 in the report table). decisions.md records that scripts/collect_feedback.py c |
| question | 8 | 01 has one question and H1, H2(a–c), H3 with no gaps, plainly phrased, and each is answered in reports/report.md Results and Conclusion. Weaknesses: the odd sen |
| report | 7 | The report is 4,961 words by whole-text count including References, so it is near the cap. It is coherent and covers goal, resources, approach, evaluation, find |
| reproducibility | 8 | docs/reproducibility.md records two identical runs with SHA-256 lists and 98 s runtime (limit 480 s); requirements-lock.txt and seeds exist. Caveats: the eviden |
| resources | 7 | 02 and the report describe LOCO, the website labels, the 9/11 subset (3,146 docs, 39/79 sites), both cleaning steps with reasons, Haiku 5.5 and pandas/scikit-le |
| results | 7 | 08 holds one table of all four models, confusion matrices and the three McNemar tests. comparison_test.csv matches the report (e.g. LLM P 0.985, R 0.667, F1 0.7 |
| teaching | 7 | All 13 sections follow the four-part format, there are analogies (exam, teacher ticking boxes) and the glossary is extensive (≈60 entries incl. TF-IDF, McNemar, |

### Open reviewer issues
- [errors] Cohen's kappa is still explained and coded although requirement 13 removed it; decisions.md says the agreement code was removed, but it exists. (content/sections/09-where-they-fail.md (Explained simply, Check yourself, cells stance_agreement); src/fml/errors.py:293; content/sections/99-glossary.md 'Cohen's kappa')
- [errors] The 09 report error table has raw column names (lr_only_wrong, stance_no_claim, not_sampled) that a non-technical reader cannot read. (reports/report.md Where they fail table; results/tables/error_classification.csv)
- [evaluation] Test-set numbers appear in the method section 06, which should show validation only, and the 06 callout refers to test. (content/sections/06-feature-engineering.md line 8 and line 78 (Report: 'On test its gain (F1 0.610 against 0.582) ... p = 0.332'))
- [results] Process and technical leakage in the 03 report: 'The 7 conspiracy websites ... in metrics.json (per_split) are counted before cleaning, the 6 here after it', plus a duplicated paragraph ('split file was made before cleaning' twice; 'Cleaning runs after the split' contradicts the earlier order). (reports/report.md Splitting by website; content/sections/03-split-by-website.md Report)
- [resources] The licence status is on the site (10-limitations, report), but requirement 17 says the open check is recorded in docs/decisions.md, not on the site; the site should only say where the data come from and that texts are not redistributed. (content/sections/10-limitations.md 'The licence is not confirmed' and Report; reports/report.md Limitations; also 02 libraries)
- [question] The sentence 'the exact date on which they were written down cannot be shown' is an awkward process remark in a hypothesis section, and H2 is split into H2a–c. (content/sections/01-question-and-hypotheses.md (Explained simply line 20, Report); reports/report.md)
- [code_readability] decisions.md itself says requirement 14 is only partly met ('not confirmed complete', 'queued code task T1'); unused kappa code remains. (docs/decisions.md Cycle 2 'Requirement 14'; src/fml/errors.py:293; code cells in 06 (mattr, style_row), 09)
- [feedback_compliance] Several requirements are unmet or partly met (13 kappa, 14 beginner code, 17 licence on site) and seven items are still [open] in decisions.md, including 'Report length counting' and 'Near-duplicate choice' that the owner expects to be resolved up front. (docs/decisions.md 'Needs a team decision'; section 09, 10, 06)
- [report] The report is about 4,960 words with References, leaving no margin under the gate, and it holds process remarks and a repeated paragraph. (reports/report.md)
- [features] The verdict on 'do the extra features help?' is a callout plus table and mixes validation and test; the answer is not one plain sentence. (content/sections/06-feature-engineering.md callout and Report; 11-conclusion)
- [approach] The alternatives we did not take are scattered over several sections. (04 (no other classifiers, fine-tuning), 06 (no emotion features), report 'Word counts')
- [interpretation] Status-like wording ('None of them is recorded as done', 'The licence has not been confirmed') reads as process, and the near-duplicate threshold and the stance-rule dependence are not discussed as limits. (reports/report.md Limitations and How we used AI; content/sections/10-limitations.md)
- [ai_use] docs/ai-log.md lists all six human checks as pending and names the setup tool 'Opus 5.5' versus 'Sonnet 5.5' later; 12 holds only one concrete team change. (docs/ai-log.md Setup/Cycle 2 and the table; content/sections/12-how-we-used-ai.md)
- [teaching] 09's Explained simply introduces kappa and a long 8-step instruction is hidden inside an 'A:' field; the glossary has non-required technical entries while the 450-word limit and link coverage could not be verified here. (content/sections/09-where-they-fail.md; 99-glossary.md)
- [reproducibility] The evidence covers run-all only, with evaluate-final not re-run; results/run_info.json is modified in the working tree; README steps were not verified. (docs/reproducibility.md; results/run_info.json; README.md)
- [privacy] The owner's account name sits in the protected scripts/collect_feedback.py default and the OSF URL is in decisions.md; these are recorded but not resolved. (scripts/collect_feedback.py; docs/decisions.md)

### Reviewer summary
The cycle-2 rewrite is largely in place. All 14 section files follow the four-part format. The report is about 4,960 words and holds all four models' test results together. The numbers I checked match results/tables/comparison_test.csv. The test-access log shows each final_* model scored once. The main faults are leftovers that break specific requirements. Cohen's kappa is still explained and coded although requirement 13 removed it. Test-set numbers appear in the method section 06. The licence status is stated on the site, against requirement 17. Process remarks leak into the report, such as 'metrics.json (per_split)', the hypothesis-date sentence and 'None of them is recorded as done'. decisions.md itself calls the beginner-code requirement (14) only partly met and lists open items. The report also leaves little margin under the 5,000-word cap. I could not run the test suite or run-all, because the shell refused compound commands, so the glossary-link, word-limit and privacy gates are unverified here.

## Cycle 3 (2026-10-25)

- **Tool:** Claude Code (Sonnet 5.5) in the planner, implementer and reviewer loop; Claude Haiku 5.5 as the classifier under study (its answers were reused from the cache, no new classification calls); Claude Sonnet 5.5 for stance coding of the new disagreement articles.
- **Input:** `requirements/cycle-03.md`, drafted from the team's feedback on cycle 2 and edited with the repo owner (decisions D1-D13 and open points R1-R8).
- **Produced:** merged results and error analysis (`08-results-and-why-they-fail`), new section slugs, one identical set of logistic-regression settings for all three analyses, new final models `final2_*` scored once on test, per-website recall table, trivial-baseline row, methods-at-a-glance table, rewritten limitations, conclusion and AI-use sections, updated decisions and reproducibility notes.
- **Checked by tests:** numbers in text against `results/`, section format and word limits, privacy, frozen split, locked test set (`results/test_access_log.jsonl` shows each `final2_*` id once).
- **Human checks:** the 40-article hand check of the stance coding is done by the team with a separate private tool and arrives for cycle 4. The six checks in the cycle-2 table above are unchanged.
