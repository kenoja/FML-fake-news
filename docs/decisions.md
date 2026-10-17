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
- **2026-10-16 · Data location.** Full texts only in the private repo `FML-data`; this repo is public.
- **2026-10-16 · Licence.** No licence found in the LOCO paper; OSF page unreachable (maintenance).
  Treat texts as not redistributable until confirmed.
- **2026-10-16 · Licence check, cycle 1 task T1: unreachable on 2026-10-16.** The agent could not open
  https://osf.io/snpcg/ (the fetch tool was denied in this environment, so the page was not seen at all;
  this is not a statement about the page). Licence therefore still unconfirmed. Excerpt rules stay
  (≤ 50 words, ≤ 6 per section). Note for the report's data section: "Licence of the LOCO texts: not
  confirmed (OSF page not reachable on 2026-10-16); texts are kept private and only short excerpts shown."
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

- **LLM temperature 0 cannot be set (T8).** AGENTS.md asks for temperature 0, but `claude-haiku-5-5`
  answers HTTP 400 ("`temperature` is deprecated for this model") and the installed SDK has no such
  argument. Conflict between the rule and the model; the rule cannot be followed. Conservative choice:
  call the model with default sampling, keep `temperature: 0` in config (it stays in the cache key),
  and measure the consequence with the second test pass (agreement, Cohen's kappa, flipped documents).
  The team should confirm that this is acceptable.

- **Licence of the LOCO texts.** Unconfirmed (see above). Conservative reading chosen: texts not
  redistributable; private data repo only; excerpt limits apply.
- **LLM API key in the cloud environment.** It was unknown whether `LLM_CLASSIFIER_API_KEY` is set when
  the agent runs. In the T8 run it was set, so the three validation variants and both test passes were
  produced (numbers: `metrics.json` key `llm`). The committed cache `results/llm_cache.jsonl` is
  complete for those runs, so `fml run-all` needs no key; without key and cache the stage still fails
  loudly with exact missing counts, and never falls back to `ANTHROPIC_API_KEY`.
- **LLM parse errors from the 150-token output limit (T8).** Many replies (validation: 44 of 398 for the
  chosen prompt v2; see `metrics.json`) were cut off at `max_output_tokens: 150` before the JSON closed
  and are counted as parse errors, scored as "not conspiracy" (lowers recall). Raising the limit now
  would be a re-run after seeing test results, so it was not done. Team decision: accept, or approve a
  new documented run with a larger limit (new model id on test).
- **Parallel LLM runs left duplicate lines (T8).** Two interrupted-attempt processes ran the stage at the
  same time, which duplicated cache lines (1,059 dropped; first answer per `doc_id`+`prompt_hash` kept,
  `read_cache` now does the same) and wrote the `llm_haiku` test-access entry twice (the second,
  identical line was removed). Test data was classified once per configuration; the numbers in
  `metrics.json` were regenerated from the deduplicated cache and are reproducible without a key.
- **Near-duplicate removal vs the expected 2,608.** If near-duplicate removal is applied after the
  topic filter, the count will fall below 2,608 and that is the number used downstream. Conservative
  reading (T2 measured 63 near-duplicate pairs, 53 documents would go, 36 pairs cross splits): report 2,608 after the topic filter first, then the count after near-duplicate removal
  separately, and use the smaller set only if the team approves; until then experiments use 2,608.
