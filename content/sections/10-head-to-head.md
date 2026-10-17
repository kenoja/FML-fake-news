---
title: Head-to-head on the test set
part: Evaluation
hypothesis: Both approaches beat the always-mainstream baseline on conspiracy F1 (H1); the zero-shot LLM is cautious (high precision, lower recall) while the logistic regression finds more conspiracy pages but also raises more false alarms, and with so few test websites the difference between them is not clearly separable.
---
## Goal and hypothesis

**Read this first (four cautions about the test numbers).**

1. **Wide intervals, website level first.** The test set has 423 documents (98 conspiracy) from 18 websites, but only **6 conspiracy websites**. Documents from one website resemble each other, so the website (cluster) bootstrap interval is the honest one; the document interval is shown second and is too narrow.
2. **The words-only logistic regression was scored on test twice.** `results/test_access_log.jsonl` has `lr_words` at 2026-10-17T01:19:52+00:00 (config hash 4c7ae4b64bb07dc1) and later `lr_words_v2` at 2026-10-17T10:35:02+00:00 (hash 686c1de15756e287), after `lr_extended` (01:21:07) and `llm_haiku` (09:26:03) had been scored. Why: the first configuration was stale (validation tuning no longer picked it), and the lock forbids re-scoring one id with a new configuration, so the corrected model got a new id. **`lr_words_v2` is the reported words-only model; the `lr_words` result is not used for any number.** The log is unchanged, and the second look happened after other test results were known.
3. **Six conspiracy websites.** One of them supplies 44 of the 98 conspiracy test documents, so the test set mostly measures how well each model recognises that site.
4. **36 near-duplicate pairs across splits** (`data.near_duplicate_pairs_across_splits`). Nine test documents have a partner in train or validation. The "without near-duplicates" rows stay in the table below: the F1 changes by -0.005 (logistic regression) and -0.004 (LLM), and the gap stays about 0.1, so the duplicates do not explain the result.

**Plain statement on the second test look and what the conclusion rests on.** `lr_words` was scored on test first (2026-10-17T01:19:52, hash 4c7ae4b64bb07dc1). `lr_words_v2` was scored later (10:35:02, hash 686c1de15756e287), after `lr_extended` and `llm_haiku` had already been scored, so the words-only model got a second test look when other test results were known. The log has not changed (no new line was added for this check). The conclusion "the LLM beats the logistic regression" rests **only on the document-level test**: the McNemar test (p = 0.0004, 32 vs 9 documents that only one model gets right) and the document-bootstrap interval for delta-F1 (+0.0235 to +0.1795). At **website level the interval for the LLM-minus-LR F1 difference includes 0** (+0.0994, interval -0.0166 to +0.3891), so with 6 conspiracy websites the data cannot separate the two models. The without-near-duplicates rows in the table now carry bootstrap intervals computed from the saved predictions only (`results/tables/comparison_test_no_dups.csv`); no new scoring on test was done.

**Sensitivity row for the first `lr_words` configuration.** Its test numbers are not stored anywhere in `results/` (the log keeps only the model id and a hash), and we do not score on test again. We therefore give the **validation** comparison, copied from `results/tables/comparison_lr_config_validation.csv` (`results/metrics.json#comparison.lr_config_validation_sensitivity`):

| Model (validation split) | Setting | Precision | Recall | F1 |
|---|---|---|---|---|
| `lr_words` v1 (stale; scored on test first, number not stored) | C=100, no class weights | 0.965 | 0.867 | 0.914 |
| `lr_words_v2` (reported) | C=10, balanced | 0.914 | 0.914 | 0.914 |
| `llm_haiku` (chosen prompt v2) | zero-shot | 1.0 | 0.906 | 0.951 |

On validation the two logistic regression settings have the same F1 (difference 0.0), so revising the configuration did not move the quality of the words-only model; it only changed the balance of precision and recall.

**Headline comparison on test** (website intervals first, then document intervals; copied from `results/tables/comparison_test.csv`, `results/tables/comparison_test_no_dups.csv` and `results/metrics.json#comparison`):

| Model | F1 | F1 website (cluster) interval | F1 document interval | Precision | Recall | Macro F1 | Balanced accuracy |
|---|---|---|---|---|---|---|---|
| Majority class | 0.0 | 0.0 to 0.0 | 0.0 to 0.0 | 0.0 | 0.0 | 0.434 | 0.5 |
| Logistic regression, words (`lr_words_v2`) | 0.585 | 0.324 to 0.804 | 0.491 to 0.667 | 0.685 | 0.510 | 0.740 | 0.720 |
| Logistic regression, extended (`lr_extended`) | 0.583 | 0.314 to 0.822 | 0.493 to 0.667 | 0.662 | 0.520 | 0.737 | 0.720 |
| Claude Haiku 5.5 (`llm_haiku`) | 0.684 | 0.371 to 0.933 | 0.592 to 0.762 | 0.963 | 0.531 | 0.807 | 0.762 |
| `lr_words_v2`, sensitivity: without near-duplicates (414 documents) | 0.580 | 0.322 to 0.797 | 0.481 to 0.667 | 0.681 | 0.505 | 0.736 | 0.716 |
| `llm_haiku`, sensitivity: without near-duplicates (414 documents) | 0.680 | 0.370 to 0.930 | 0.587 to 0.764 | 0.962 | 0.526 | 0.805 | 0.760 |

<!-- output: results/tables/comparison_test.csv -->

Everything so far was tuned on **validation**. This section is the one place where all approaches meet on the same **test** documents, which were scored exactly once per model by `fml evaluate-final` (the log is `results/test_access_log.jsonl`). The comparison stage reads only the saved prediction files, so it cannot choose or re-score anything on test.

**Sequence of test looks.** The log `results/test_access_log.jsonl` shows this actual order (UTC, copied from the log): `lr_words` at 2026-10-17T01:19:52 (config hash 4c7ae4b64bb07dc1), `lr_extended` at 2026-10-17T01:21:07, `llm_haiku` at 2026-10-17T09:26:03, and `lr_words_v2` at 2026-10-17T10:35:02 (changed configuration, new hash 686c1de15756e287). So `lr_words_v2` is a second test look at the words-only logistic regression, made after the test results of the other models were known. The LR choice rule (M2 only if its F1 gain over M1 has a website-bootstrap interval above 0) is applied to validation rows only (`ablation.chosen_lr`); we state that it was fixed before any test look, but the log cannot prove it, because the first `lr_words` look already used an earlier configuration. Limitation: the stale `lr_words` entry stays in the log and is not used for any number, yet the sequence means the words-only configuration was revised after one test look. From now on `evaluate_final` refuses to score a model unless its validation-chosen configuration is already in `results/frozen_models.json` (model id, config hash, source `validation`, timestamp) with the same hash. Entries for looks that happened before this rule are marked `retroactive` there, so their timestamps are later than the log lines.

We compare four things on the same 423 test documents (numbers in `results/metrics.json`, key `comparison`):

- **majority**: always answers "mainstream" (the most common class in training);
- **lr_words_v2**: logistic regression on word TF-IDF; this is also the "best LR" because validation did not show a clear gain for the extended features (`ablation.chosen_lr`);
- **lr_extended**: the logistic regression with n-grams, epistemic and style features (reported, but not chosen);
- **llm_haiku**: Claude Haiku 5.5 zero-shot, with the prompt chosen on validation, **pass 1** (the official one).

Hypothesis: see the header.

## Explanation

**Open question 3: is the LLM the baseline?** No. Because an LLM does not always answer the same way, we never treat its answer as the truth. **Both** approaches are compared against the **dataset labels** (conspiracy-oriented or mainstream website). The LLM's answers are one more set of predictions that gets scored, exactly like the logistic regression's. The only baseline is the majority rule.

**How stable is the LLM?** We sent the same test documents a second time with the same prompt (pass 2). The two passes agree on the share of documents given in `comparison.llm_stability.agreement`, the number of documents whose answer flipped is `flipped_docs`, and Cohen's kappa (agreement beyond chance) is `cohen_kappa`:

<!-- output: results/metrics.json#comparison.llm_stability -->

Roughly 95% agreement and a kappa near 0.89 mean the answers are mostly stable but not identical, so single-run LLM scores carry a small extra uncertainty. The scored F1 of pass 2 (`llm.test_pass2`) is within the pass-1 interval. Some of the instability comes from replies that could not be parsed (`parse_errors_pass1`/`pass2`): they count as "not conspiracy" and so lower recall.

**Metrics, in plain words.** For the conspiracy class: *precision* is the share of pages called "conspiracy" that really are; *recall* is the share of real conspiracy pages that were found; *F1* balances both. *Macro-F1* averages the F1 of both classes; *balanced accuracy* averages recall on both classes, so a model that ignores the small class cannot look good.

**Confidence intervals by bootstrap** (defined in section 05). We redraw the test set 2,000 times with replacement and recompute the score. The *document* bootstrap redraws articles; the *website* bootstrap redraws whole websites with all their articles, which is more honest because articles from one site resemble each other. Both are in the table, and the website interval is the one to trust.

**McNemar exact test** (first met in section 08). Take only the documents where exactly one of two models is right. If the models were equally good, these "discordant" documents would split about 50/50. The test turns the actual split into a p-value; a small p-value means the models really differ on these documents. A **paired bootstrap** of delta-F1 resamples the *same* documents for both models and reports the interval of the F1 difference; an interval containing 0 means the data cannot separate them.

### Results table

<!-- cell: src/fml/pipeline.py#comparison_stage -->

<!-- cell: src/fml/comparison.py#model_row -->

<!-- output: results/tables/comparison_test.csv -->

<!-- output: results/figures/comparison_f1.png | Conspiracy-class F1 on test with document and website 95% intervals -->

<!-- output: results/figures/comparison_confusion.png | Confusion matrices on test: rows are the true website type -->

**H1 verdict.** The majority baseline finds no conspiracy page, so its conspiracy F1 is 0. Both lr_words_v2 and llm_haiku are above it and the website-bootstrap interval of their F1 stays above 0 (`comparison.h1`: `site_ci_excludes_majority` is true for both). **H1 holds on test.** That is a low bar, however; macro-F1 and balanced accuracy show the real gap (majority 0.50 balanced accuracy).

**Who wins on which class.** The LLM has much higher precision (`winner_by_metric`: llm_haiku for precision, recall and F1) and only 2 false alarms, and it is also right on 99% of mainstream pages versus 93% for lr_words_v2 (`comparison.mainstream_accuracy`). Its recall on conspiracy pages (0.53) is only a little above lr_extended (0.52) and lr_words_v2 (0.51). In short: the LLM is the more careful judge of mainstream pages and has the higher F1; the logistic regression's mistakes are false alarms on mainstream pages (23 for lr_words_v2, 26 for lr_extended) and misses on conspiracy pages. No model finds even 60% of the conspiracy pages.

### Is the difference real?

<!-- output: results/metrics.json#comparison.mcnemar -->

*lr_words_v2 vs llm_haiku:* the LLM is right and the LR wrong on 32 documents, the other way round on 9 (McNemar p = 0.0004). The paired delta-F1 is +0.0994; its document interval is 0.0235 to 0.1795 and its **website interval is -0.0166 to 0.3891**. By documents the LLM is clearly better; by websites the interval includes 0 and is very wide, so the honest statement is "the LLM is better on these documents, but with so few websites we cannot show that the difference would hold for other websites".

*lr_words_v2 vs lr_extended:* 19 vs 17 discordant documents, p = 0.8679, delta-F1 -0.0019 with interval -0.1146 to 0.0972 across websites. No evidence that the hand-made features help, which agrees with the validation result in section 08.

### Where do the models differ? Length, thresholds and websites

<!-- output: results/tables/comparison_length_buckets.csv -->

The length table has only 27 and 32 documents in the two short buckets (5 and 7 conspiracy pages), so it shows no reliable length effect.

<!-- output: results/figures/comparison_pr.png | Precision-recall curves on test for lr_words_v2 and the LLM confidence -->

The precision-recall figure ranks documents by the logistic regression's probability and by the LLM's stated confidence (conspiracy confidence if it answered "conspiracy", 1 - confidence otherwise; unparsable replies get 0). The LLM's confidence is not a calibrated probability and has few distinct values, so its curve is coarse. Average precision is in `comparison.average_precision`: the logistic regression's ranking is the slightly better one even though the LLM's hard decisions score higher F1.

<!-- output: results/tables/per_website_test.csv -->

The per-website table is the most important caution. The test split has only **6 conspiracy websites** (`test_conspiracy_websites`), and one of them (neonnettle.com, 44 pages) is the large majority of the misses for all three models: accuracy there is 0.204 for both logistic regressions and 0.159 for the LLM. One oye.news page counts as a whole website. A different website draw could move every number in the table, which is exactly what the wide website intervals say.

### Sensitivity: test documents with a near-duplicate in train or validation

A *near-duplicate* is a pair of articles whose TF-IDF vectors have cosine similarity of at least 0.9 (the pairs counted in `data.near_duplicate_pairs_across_splits`, 36 of them). A test article that is nearly a copy of a training article is partly "seen", which could flatter a model that memorises vocabulary. To measure this we drop every test document that has such a partner in train or validation and recompute precision, recall and F1 for both models on the rest. The split stays frozen and nothing is removed from the main results; the duplicates are quantified, not removed.

<!-- cell: src/fml/comparison.py#no_duplicate_sensitivity -->

<!-- output: results/tables/comparison_test_no_dups.csv -->
<!-- output: results/tables/comparison_lr_config_validation.csv -->
<!-- output: results/metrics.json#comparison.no_duplicate_sensitivity -->

Nine test documents (one of them conspiracy) are excluded (`n_excluded`), leaving 414. F1 of the logistic regression moves from 0.5848 to 0.5799 (change -0.0049) and F1 of the LLM from 0.6842 to 0.68 (change -0.0042). The gap between them is 0.0994 on all test documents and 0.1001 without the duplicates. Both models lose about the same tiny amount, so near-duplicates do not explain the LLM's lead and do not change the conclusion of this section.

## What could go wrong

- **Few test websites.** 18 websites, only 6 conspiracy. Intervals by website are wide (F1 for the LLM: 0.37 to 0.93). Treat rankings as indications, not as proof.
- **Distribution shift between validation and test.** Prompt and LR settings were chosen on validation (19 websites), where the LLM reached a much higher recall (0.91) than on test (0.53). Different websites behave differently; part of the drop is tuning to validation sites.
- **LLM non-determinism.** `claude-haiku-5-5` rejects `temperature`, so the run uses default sampling. Pass 1 is scored, pass 2 is a sensitivity check (see stability above).
- **Parse errors.** 61 replies were unparsable and counted as "not conspiracy"; this lowers LLM recall and is a formatting issue more than a judgement issue.
- **Prior knowledge.** Haiku may have seen these sites or articles; the logistic regression saw only our training data. The LLM score may not carry over to other topics.
- **Multiple comparisons.** We ran two McNemar tests on test; the second is secondary, and none of this was used to choose a model.

## Technical infrastructure

- Code: `src/fml/comparison.py` (`model_row`, `pairwise`, `per_website_table`, `length_bucket_f1`), `src/fml/pipeline.py#comparison_stage`; statistics from `src/fml/evaluate.py` (`bootstrap_ci`, `paired_delta_ci`, `mcnemar_exact`).
- Inputs: `results/tables/predictions_lr.csv`, `results/tables/predictions_llm.csv` (written by `evaluate_final`); outputs: `results/tables/comparison_test.csv`, `per_website_test.csv`, `comparison_length_buckets.csv`, figures `comparison_f1.png`, `comparison_confusion.png`, `comparison_pr.png`, and `metrics.json#comparison`.
- Tests: `tests/test_comparison.py` recomputes the McNemar counts and p-values by hand on a synthetic example and from the saved prediction files.
