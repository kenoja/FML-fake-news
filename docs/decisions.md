# Decisions

The current design decisions, one entry each, with the reason. Open items wait for a person.
The full history of how each decision was reached is in `docs/archive/decisions-history.md`.
Agents: add an entry whenever you choose between alternatives or resolve a conflict (see AGENTS.md).

## Data and cleaning

- **Data source.** The 9/11 subset of LOCO (Miani, Hills & Bangerter, 2022), full articles only. The texts stay
  in the private data repository; this public repository holds code, results and excerpts of at most 50 words.
- **Off-topic removal.** Articles that never mention the attacks are removed before any model, so that no model
  learns topics instead of kinds of websites (3,146 to 2,608 articles).
- **Near-duplicates.** Pairs with a TF-IDF cosine similarity of at least 0.9 are treated as copies; the copy in
  the earlier set (training, then validation, then test) is kept, inside one set the lower id. The label plays no
  role. A lower threshold would also remove different articles on the same event.
- **Site clues.** Web addresses, e-mail addresses, website names and lines a site repeats in many articles are
  deleted from the texts; no article is removed by this step.

## Split and evaluation

- **One frozen split.** `data/splits/split_v1.csv`, grouped by website and stratified by label, made once with
  the seed in `config/project.yaml`. Cleaning only removes rows from it.
- **Settings on validation only.** All three logistic regressions share one setting: balanced class weights and
  C from [0.01, 0.1, 1, 10, 100], chosen by validation F1. All four methods read the first 3,000 words.
- **The feature model is the planned one.** TF-IDF plus all three groups (n-grams 1-3, certainty words, style);
  the per-group comparison explains the result but does not choose the model.
- **Verdict rule.** A hypothesis is supported when the difference points the predicted way and McNemar's test
  gives p < 0.05 (Dietterich, 1998). The three tests are not corrected for multiple testing; the text says so
  where it matters (H1).
- **Test set.** Every reported model was chosen on validation and scored on test once:
  `final2_word_counts`, `final2_tfidf`, `final2_tfidf_features`, `final_llm`. Seven earlier versions were scored on
  test during development and are not reported: `lr_words`, `lr_extended`, `llm_haiku`, `lr_words_v2`,
  `final_word_counts`, `final_tfidf`, `final_tfidf_features` (`results/test_access_log.jsonl`).
- **Statistics.** Confusion matrix, precision, recall, F1, 95% bootstrap intervals (which treat articles as
  independent and are therefore somewhat too narrow) and three McNemar tests. Nothing more, by design.

## Language model

- **One cheap model.** Claude Haiku 5.5, zero-shot, asked the source question, default sampling (the model has no
  temperature setting), answer limit 1,000 tokens. Three prompts tried on validation; the first of the tied best
  was kept. Answers are cached, so a rerun needs no key.

## Error analysis and hand check

- **Stance coding.** A second model codes what each disputed test article does with conspiracy claims; a fixed rule
  turns label and stance into "label problem" or "model error". The shares are estimates from one automatic coder.
- **Hand check.** 40 validation articles (`results/tables/stance_check_sample.csv`, ids only), each meant to be coded
  blind by two of the five members with the offline hand-check file. Answers are in
  `data/hand_check/stance_check_member_<letter>.csv`; `fml run-all` reads every file there, and when a member sends a
  second file, their first (blind) answer for an article counts (rule set on 27 Oct 2026, before the second file
  arrived).
- **Member B's first file.** It was made under letter A by mistake and holds Member B's blind answers for Member A's
  16 articles (owner's information, 27 Oct 2026). It is stored as Member B. Member B then coded their own 16 articles
  (`stance_check_member_B_2.csv`); for the 8 articles in both files the first answers count. Result: every one of
  the 40 articles has at least two readers (`errors.human_check.status`: complete), and 8 have three.
- **Member B's 8 repeated articles (not in the report).** Member B gave the same stance both times for 1 of the 8
  and the same "endorses or not" answer for 5 of the 8. This shows that stance coding is hard even for one person; it
  is not reported because the repeat was accidental and not planned as a test, and the report has no room left.
- **Agreement measure (closes R1 of cycle 3).** Plain counts and shares: the same stance, and the same answer to
  the one question the error rule asks ("does the article endorse a claim?"), between two members and between a
  member and the automatic coder; plus the rule's verdict per website type. No kappa or other new statistic, to
  keep to the cycle-2 rule of simple statistics. The two automatic answers that were unreadable are skipped.
- **Examples.** The site shows six disagreement examples; the report names one label problem and one model error
  to stay near 4,500 words.

## Report and site

- **Report length.** 3,000 to 5,000 words before References. Measured on 27 Oct 2026: 4,981 words by the team's
  plain count (split at spaces, tables included; checked by `tests/test_gates.py`) and 4,142 words of prose without
  tables. The plain count is the binding one, so later additions must be balanced by cuts.
- **Audience.** Teammates without a technical background: four-part sections, glossary links, hidden answers.
- **No process on the site.** Plans, run records and decision history stay in `docs/`.

## Cycle 4 (final run): how each item was handled

Rule for the cycle: no new models, statistics, prompts, API calls or test-set runs; new numbers are only counts
read from the data or from `results/`.

- **Item 1 (changes from the plan, `01`).** Short paragraph added from the table in `docs/PROJECT_PLAN.md` and the
  course feedback: full articles instead of headlines, one cheap LLM, no transformer; the JFK transfer stays further
  work. About 70 words in the report, the course is not named.
- **Item 2 and D3 (F1 vs McNemar, `01`, `08`).** Verdict rule kept. One sentence in `01` (plain and report) says F1
  looks at the conspiracy class and McNemar's test at both classes; a hypothesis is supported only when both point
  the predicted way. `08` states in one clause that they agree for all three tests (`mcnemar_tests.csv`).
- **Item 3 (search words, `02`).** The eleven topic-filter words and phrases are listed in plain words, taken from
  `data.topic_filter_pattern`; the pattern itself is unchanged.
- **Item 4 (Other: largest conspiracy sites).** `site_concentration` in `src/fml/data_report.py` is called from the
  data stage of `fml run-all`. It writes `results/tables/site_concentration.csv` and the keys `top3_sites_share` and
  `top3_train_articles` under `cleaning` in `metrics.json`. Websites are numbered, not named. "Largest" is counted in
  all conspiracy articles; the training count is the same three sites' articles in the training split.
- **Item 5 (Other: 39 or 37 websites).** 39 is used only for the raw corpus subset, 37 for the analysed data (25
  train, 6 validation, 6 test). Changed in `02` and `09`.
- **Item 6 (`03` watch out).** The risk that one website supplies a large part of a set's conspiracy articles is
  stated without test numbers.
- **Item 7 and D2 (`06`).** The text says the feature model was planned on 16 Oct 2026 before any result, that cycle 2
  used a validation threshold, and that the team returned to the plan on 25 Oct after seeing validation and earlier
  test results (`results/frozen_models.json`), so the choice was not blind. It points to `09`.
- **Item 8 (why intervals may be too narrow, `08`).** One clause in the Report where the intervals are introduced;
  not repeated elsewhere.
- **D1 (hand check).** Kept as is; answer files in `data/hand_check/` untouched; no kappa.
- **R1, R4.** The evaluation paragraph moved to a `### How we judge the methods` subsection at the end of the report
  part of `01`; it holds the D3 sentence and states once that the three McNemar tests are not corrected for multiple
  testing and that only H3 would pass such a correction.
- **R2.** `08` leads with 21 of 41 as the cautious estimate of label problems and gives 49 of 69 as the upper end.
- **R3.** One sentence in `06`: certainty words alone scored highest on validation, but switching to the best single
  row after seeing the table would change H2 to fit the results.
- **R5.** `llm_prompt_variants.csv` now has the headers Prompt, Articles, Precision, Recall, F1; the stray blank row in
  the ablation block of the report was removed.
- **R5 vs rubric (licence) and "more examples": declined** as in "Declined, with the reason" below, per the cycle's
  "Not to do" list.
- **R6.** `11` names which tool did what once, puts "Saving answers" before "What we learned", and the learnings are
  factual (plans and explicit requirements steer the assistant; tests check numbers against files, not the question).
  `docs/ai-log.md` was updated to match.
- **R7.** `src/fml/features/style.py` has one-line docstrings and a plainer capital-letter check; the scikit-learn
  wrapper moved to `src/fml/features/wrappers.py`, out of the cells shown on the site. The results must stay
  byte-identical to the pre-cycle ones except for the new `site_concentration` table and keys and the renamed
  headers of `llm_prompt_variants.csv`; see `docs/reproducibility.md` for the status of that check.
- **R8.** The Anthropic client version from `results/run_info.json` is named in `02`.
- **R9.** Report length is checked by `tests/test_gates.py` (limit 5,000, aim 4,900); cuts were made in repeated
  hedges and explanations.
- **R10.** No "so far" in the hand-check wording.
- **Moving open items.** They stay in this file only.
- **Approach alternatives (T11).** Each of `04`-`07` now names in Explained simply what was not done and why: no other
  classifier or word order (04), no other weighting or word pairs (05), no emotion features and no phrases beyond three
  words (06), no fine-tuning, few-shot examples or several LLMs (07).
- **Report length (T11, final).** Assembled report: 4,894 words before References by the plain count of
  `tests/test_gates.py` (split at spaces, limit 5,000, aim 4,900); it was 5,163 before the cuts. Cuts, all in repeated
  explanations and hedges: the introduction (one citation, the run-time sentence, the classifier remark), the
  executive summary, `08` (repeated caveats, the Website 4 lead-in, the second-estimate remark, the table caption
  clause), `06`, `09` and `11`. Kept: both examples, the Website 4 finding, every hypothesis verdict, the hand-check
  numbers and conclusion. `reports/report.md` was only rebuilt by `python -m sitegen build`, never edited by hand.
  One citation (Swami et al., 2010) is no longer cited in the report text; `content/references.md` was not changed.
- **Choice of F1 and McNemar (D3).** Both are kept: F1 (conspiracy class) is the headline score, McNemar the test on
  both classes; a hypothesis counts as supported only if both point the predicted way. They agree for all three tests.
- **Placement of "intervals" (item 8).** The reason the intervals may be too narrow (articles from one website are
  similar) is given once, in the first paragraph of the Report part of `08`, where the intervals are introduced.
- **Search-term wording (item 3).** The topic filter is described as a plain list of words and phrases with spelling
  variants, not as a regular expression; the filter itself is unchanged.
- **Licence sentence.** The one neutral sentence in `02` stays (see "Declined, with the reason").
- **Nothing new run.** No new models, prompts, API calls or test-set scoring in this cycle's text work; the frozen
  models and `results/test_access_log.jsonl` are unchanged.
- **Declined per "Not to do".** The request for more of the licence discussion (the one neutral sentence in `02`
  stays) and the request to show open items on the site (they stay in this file only).
- **Other items by judgement.** Item 4: three-site concentration counted in the data stage and quoted from `results/`.
  Item 5: 37 websites for the analysed data, 39 only for the raw subset.

## Cycle 5 (final report, 30 Oct 2026)

- **N-gram threshold (O1).** Phrases keep the minimum of 3 training articles, single words 2. Checked on validation
  only (`tfidf_features.ngram_threshold_check`): with a minimum of 2 the planned model fills the 200,000-column cap
  and its validation F1 falls from 0.896 to 0.891. Decided with the owner: no new test run, the report explains the
  difference.
- **Narrative (O3-O6).** Every report chapter now uses the same pattern: method chapters Idea / Setup / Validation
  result / What it learned; data, split, limitations and conclusion use short labelled paragraphs. The test drop is
  attributed to Website 4 as the main single cause: it explains almost all of the LLM's drop (0.971 without it,
  0.987 on validation), and the largest part, not all, of the logistic regressions' drop.
- **Overview table.** The "Reads" column was removed: it was the same in every row and the text says it once.
- **Environment.** The unchosen word-count setting C = 100 gives a validation F1 of 0.791 on the 2-CPU machine used
  for this run and 0.797 in the agent's 4-CPU environment. Reported numbers come from the committed run; no chosen
  setting or test result changes.
- **Report length.** 4,889 words by the plain count (limit 5,000, aim 4,900).
- **Review (cycle 5).** A fresh reviewer session scored 8 to 10 on every criterion. Implemented: separate paragraph
  for what was left out, dates only in the limitations, compact ablation table, plain error headline, uncited
  references (Shu et al., Swami et al.) removed, distinct lesson per cycle in the AI-use log. Not changed: the licence
  stays open until the owner confirms it; README and code were not part of this change.
- **Drop chart.** A new figure (`f1_validation_test.png`) shows validation F1, test F1 and test F1 without Website 4;
  in the report it replaces the validation-vs-test table, which gained a column for the score without Website 4.

- **Numbers (31 Oct).** Every score column in `results/tables/` is written with exactly three decimals by
  `results.three_decimals` (0.480, 1.000); p-values use three decimals too (`p < 0.001` replaces `< 0.0001`). The PDF
  follows APA 7 (owner's choice): `build_pdf.apa_numbers` drops the leading zero of values that cannot exceed 1
  (.721, p = .029); C settings and amounts keep it. The website keeps 0.721, which beginners read more easily. The
  existing tables and `metrics.json` were rewritten with the same code instead of a full rerun, because a rerun in
  this container also redraws the figures with other fonts; the values are unchanged (checked with `git diff`).
- **Website 4 (31 Oct).** The corpus authors flag a conspiracy article as typical when its wording is unusually
  similar to the rest of the conspiracy corpus (more than one standard deviation above the mean). New in
  `pipeline.typical_articles`: none of Website 4's 44 test articles is flagged, against 0.214 of the training
  conspiracy articles, and every model finds all 12 flagged test articles. This supports, but is itself word-based
  and so does not prove, that Website 4's texts read as mainstream. A spot check of five of its articles by the owner
  against the stance coder is open (see below).
- **Limitations (31 Oct).** Stated explicitly: articles are treated as independent although articles from one website
  are alike (intervals too narrow, p-values too small); the corpus authors' own check (15% of conspiracy and 8% of
  mainstream documents did not fit their label); the LLM probably saw the public corpus and websites in training,
  which cannot be checked because the training data are not published. A probe (asking the model to name an
  article's website) was not run: it would need API calls outside the three allowed prompt variants.
- **Tables in the PDF.** Tables no longer break across pages (a `[H]` table instead of a longtable, which repeated its
  header at a page break), and columns get a minimum width so that headers wrap at most once.
- **Report length.** 4,897 words after the additions; repeated sentences were cut in Introduction, Test scores,
  Hypotheses, the LLM chapter and the Conclusion.

## Needs a team decision

- **Website 4 spot check (owner, done 31 Oct; resolved).** Five Website 4 articles that the LLM missed, read in full against the
  stance coder. Same verdict on "endorses a claim, yes or no" in 4 of 5; the owner judged one article (C0623c) to
  endorse a claim where the coder said it reports one neutrally. Stated in the report's limitations: Website 4's share
  of label problems may be slightly overstated.

- **[open] Dataset licence.** Owner: repository owner. The OSF page could not be read automatically and no file in
  the repository states the licence. Record the licence text and date here. Conservative reading until then: texts
  stay private, excerpts of at most 50 words, and the one neutral sentence in `02`.
- **[open] Unused citation.** Swami et al. (2010) remains in `content/references.md` but is no longer cited in the
  report after the length cuts. Decide whether to remove it or cite it again.
- **[open] Reproducibility check (R7).** The byte-identical comparison of `results/` before and after the `style.py`
  change is covered by the final verification (`fml run-all` twice); not repeated in this task.

## Declined, with the reason

- **More excerpts in the report.** Six examples are on the site; adding them to the report would push it past the
  length aim. The report names one label problem and one model error.
- **A window-by-window MATTR.** Recomputing every 100-word window from scratch is easier to read but about a
  hundred times slower over 2,554 articles; the sliding window gives the same values within the time limit.
- **Licence in the report (R5 vs rubric).** R5 kept the licence off the site; the rubric asks for its status. Decided
  with the owner: one neutral sentence in the data section, no further discussion.
- **Website-level confidence intervals, more prompts, a memorisation test for the LLM.** Kept out to keep the
  analysis at the level of a fundamentals course; the limitations section names them.

## Cycle 4 final verification (T12)

- **Environment problem, tests not run.** In the verification session the system Pillow was broken
  (`cannot import name '_imaging'`), so `import matplotlib` failed. Four test files could not be collected
  (`test_ablation.py`, `test_cell_functions.py`, `test_results.py`, `test_test_lock.py`) and four tests in other files
  failed on the same import. They were not run and are not counted as passed.
- **Stale results.** `results/tables/site_concentration.csv` and `metrics.json#cleaning.top3_sites_share` did not
  exist yet, because `fml run-all` had not been run after `fml.data_report` was added. Until it is rerun, the
  number 157 in `02-data-and-cleaning` is not in `results/`, two section/report number tests fail, the section
  reference does not resolve, and the runtime-sentence test fails. Needed: `fml run-all` in a working
  environment, then `python -m sitegen build` and the full suite.
- **Checked and fine.** Gate and sitegen tests pass (20); the report has 4,888 words before References;
  10-conclusion agrees with 08 on H1-H3 (checked by reading, nothing changed).

## Cycle 4 length budget

- **Baseline (T1).** After `python -m sitegen build`, counted as `tests/test_gates.py` does (split at spaces, tables
  included): **5,401 words** in the whole `reports/report.md`, **4,921 words before `## References`**. The gate
  (limit 5,000) passes, but only with 79 words to spare. Target: at most 4,900 words before References, so every
  added sentence is paid for by a cut. (The older count of 4,131 used `agent/tools.py` rules, which drop table rows
  and code fences; it is not the gate's count.)
- **Words per `## Report` part** (raw text of the part, output markers not expanded): 01 573, 02 422, 03 166, 04 348,
  05 193, 06 508, 07 312, 08 905, 09 265, 10 194, 11 290 (sum 4,176; the rest of the 4,921 is headings, tables
  and figure captions pulled in by the output markers). Cuts come mainly from the hedges in 04-08.
- **Open point, declined.** The reviewer's "cluster bootstrap" suggestion is declined because cycle-04 bars new
  statistics; the confidence intervals are instead described as lower bounds on the uncertainty (see T6).
- **Raw-header and blank-row checks (R5)** were run.
- **Results unchanged.** SHA-256 hashes of `results/metrics.json` and `results/tables/*.csv` saved before the
  edits for later comparison; no models, API calls or test runs.
- **Privacy grep.** Searched `content/sections/*.md`, `content/references.md`, `docs/ai-log.md` and
  `reports/report.md` for names, e-mail addresses, institution, course name, team number and GitHub links:
  none found (the only hits were ordinary words such as "rejects" and a publisher name in a citation).

- **Final verification (T12).** Two `fml run-all` runs gave identical hashes for `metrics.json` and all tables; the test log is unchanged (11 lines). Full pytest: 239 passed, 1 failed, 0 skipped (no `needs_data` test skipped); the failure was the missing run-time sentence in section 01, restored ("about three minutes"), and `tests/test_report_numbers.py` then passed (19). The full suite was not re-run after that fix. No T1 hash file could be found, so equality was checked against the committed results (git shows only `run_info.json` modified) and the older checksums in `docs/reproducibility.md`. Privacy grep (e-mail, GitHub links, names, course words) over sections, docs and report: no real hits.

- **T2 hedge cuts.** Repeated caveats removed from the `## Report` parts of 04–08 ("all numbers here are validation numbers", repeated grid/website/cost wording, shortened fairness note in 07 and the "when the model was fixed" paragraph in 06). All numbers, verdicts, the two examples, the Website 4 finding and the hand-check numbers kept. Report body (before References, gate count) went from 4,921 to 4,558 words, leaving 342 words of room under the 4,900 target for T3–T9; re-check in T12.

- **Privacy and excerpt audit (T11).** `tests/test_privacy.py`: 10 passed. Searched `content/`, `reports/`, `docs/`, `README.md` and `results/tables/*.csv` for e-mail addresses, GitHub user links, institution and course words, team numbers and the account owner's name: no real hits (matches were the word "institutions" inside an article excerpt, a publisher name in a citation, the placeholder text in `docs/SETUP.md`/`PROJECT_PLAN.md`, and generic GitHub sign-up/app links). Excerpt limits hold: the six rows of `error_examples.csv` and `error_examples_plain.csv` have 21-26 words each (limit 50), the stance justifications at most 25 words, and the per-section limit of 6 is checked by `test_privacy.py`. Nothing needed fixing; no files changed except this note.

- **Cycle 4 final verification (T12, rerun).** Two further `fml run-all` runs (about 163 s each, LLM stage 0.5 s from cache, so no API calls) gave SHA-256 hashes of `results/metrics.json` and all `results/tables/*.csv` identical to the committed files; only `run_info.json` (timings) changed. `results/test_access_log.jsonl` still has 11 lines (no new test scoring). `python -m sitegen build` regenerated `reports/report.md` (not hand-edited). Plain whitespace count before References (after the fix below): 4,845 words (target at most 4,900, gate 5,000), so no further cuts. Full pytest: 240 passed, 0 skipped. Fixed on the way: the report leaked an internal model id (`final2_tfidf_features`) through a dict table in 08 (removed, the numbers are in the prose) and a bracket in 06 (reworded). Not done: nothing skipped.
