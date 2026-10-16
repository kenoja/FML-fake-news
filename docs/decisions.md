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

## Needs a team decision

- **Licence of the LOCO texts.** Unconfirmed (see above). Conservative reading chosen: texts not
  redistributable; private data repo only; excerpt limits apply.
- **LLM API key in the cloud environment.** It is unknown whether `LLM_CLASSIFIER_API_KEY` is set when
  the agent runs. Conservative reading: assume it is not; the LLM stage must fail loudly without key
  and cache, and never fall back to `ANTHROPIC_API_KEY`. Without a key the LLM results stay missing and
  are reported as missing, not invented.
- **Near-duplicate removal vs the expected 2,608.** If near-duplicate removal is applied after the
  topic filter, the count will fall below 2,608 and that is the number used downstream. Conservative
  reading (T2 measured 63 near-duplicate pairs, 53 documents would go, 36 pairs cross splits): report 2,608 after the topic filter first, then the count after near-duplicate removal
  separately, and use the smaller set only if the team approves; until then experiments use 2,608.
