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
  second file, their first (blind) answer for an article counts.
- **Member B's first file.** It was made under letter A by mistake and holds Member B's blind answers for Member A's
  16 articles (owner's information, 27 Oct 2026). It is stored as Member B. Result: 8 articles have three readers,
  and the 8 articles Member B shares with Member C (C015c3, C02348, C04afe, C05997, M09745, M0e3b2, M13d21,
  M1ed07) have one reader until Member B codes them; `errors.human_check.status` in `results/metrics.json` says so.
- **Agreement measure (closes R1 of cycle 3).** Plain counts and shares: the same stance, and the same answer to
  the one question the error rule asks ("does the article endorse a claim?"), between two members and between a
  member and the automatic coder; plus the rule's verdict per website type. No kappa or other new statistic, to
  keep to the cycle-2 rule of simple statistics. The two automatic answers that were unreadable are skipped.
- **Examples.** The site shows six disagreement examples; the report names one label problem and one model error
  to stay near 4,500 words.

## Report and site

- **Report length.** 3,000 to 5,000 words before References. Measured on 27 Oct 2026: 4,987 words by the team's
  plain count (split at spaces, tables included; checked by `tests/test_gates.py`) and 4,148 words of prose without
  tables. The plain count is the binding one, so later additions must be balanced by cuts.
- **Audience.** Teammates without a technical background: four-part sections, glossary links, hidden answers.
- **No process on the site.** Plans, run records and decision history stay in `docs/`.

## Needs a team decision

- **[open] Dataset licence.** Owner: repository owner. The OSF page could not be read automatically. Record the
  licence text and date here. Until then the safe default applies (texts private, short excerpts only).
- **[open] Hand check.** Owner: Member B. Code the 8 articles shared with Member C (list above) with letter B and
  send the file; the owner adds it as `data/hand_check/stance_check_member_B_2.csv`. Until then 32 of the 40
  articles have two or more readers.

## Declined, with the reason

- **More excerpts in the report.** Six examples are on the site; adding them to the report would push it past the
  length aim. The report names one label problem and one model error.
- **A window-by-window MATTR.** Recomputing every 100-word window from scratch is easier to read but about a
  hundred times slower over 2,554 articles; the sliding window gives the same values within the time limit.
- **Licence in the report (R5 vs rubric).** R5 kept the licence off the site; the rubric asks for its status. Decided
  with the owner: one neutral sentence in the data section, no further discussion.
- **Website-level confidence intervals, more prompts, a memorisation test for the LLM.** Kept out to keep the
  analysis at the level of a fundamentals course; the limitations section names them.
