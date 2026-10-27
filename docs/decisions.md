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

## Needs a team decision

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

- **Baseline (T1).** After `python -m sitegen build`, the report has 4,131 words before References, counted as in
  `agent/tools.py: gate_report_length` (code fences and table rows removed). Target: at most 4,900; every added
  sentence is paid for by a cut. (The 4,888 quoted in the T12 note above was an earlier count.)
- **Results unchanged.** SHA-256 hashes of `results/metrics.json` and `results/tables/*.csv` saved before the
  edits for later comparison; no models, API calls or test runs.
- **Privacy grep.** Searched `content/sections/*.md`, `content/references.md`, `docs/ai-log.md` and
  `reports/report.md` for names, e-mail addresses, institution, course name, team number and GitHub links:
  none found (the only hits were ordinary words such as "rejects" and a publisher name in a citation).

- **Final verification (T12).** Two `fml run-all` runs gave identical hashes for `metrics.json` and all tables; the test log is unchanged (11 lines). Full pytest: 239 passed, 1 failed, 0 skipped (no `needs_data` test skipped); the failure was the missing run-time sentence in section 01, restored ("about three minutes"), and `tests/test_report_numbers.py` then passed (19). The full suite was not re-run after that fix. No T1 hash file could be found, so equality was checked against the committed results (git shows only `run_info.json` modified) and the older checksums in `docs/reproducibility.md`. Privacy grep (e-mail, GitHub links, names, course words) over sections, docs and report: no real hits.
