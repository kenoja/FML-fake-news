---
title: Do the extra features help?
part: Model building
hypothesis: Each feature group (n-grams H2, epistemic markers H3, style H4) adds a small gain over plain words on validation, but with only 19 validation websites most gains will sit inside the confidence interval.
---
## Goal and hypothesis

The course feedback asked whether hand-made features help the logistic regression. We test three groups, one at a time, against the words-only model **M1**:

- **H2**: n-grams (word pairs and triples) help.
- **H3**: epistemic markers (hedges, boosters, hidden-truth and debunking words) help.
- **H4**: style features (ALL-CAPS share, punctuation, sentence length, repetition) help.

**M2** (`lr_extended`) has all three groups. We also drop one group at a time from M2 ("drop-one") to see what each group adds when the others are present. Everything uses the off-topic-filtered set and the frozen split (sections 03 and 06). We expected small gains at best.

## Explanation

**Protocol.** Each model gets the same tuning: the regularisation strength C (how strongly the model is pulled towards small weights; small C means simpler models) and `class_weight` (`none` or `balanced`, which counts the rarer conspiracy class more) are chosen on the validation split. The vectorizers are fitted once per feature setting on the training split only, not inside the C loop.

**Metrics.** Conspiracy-class *precision* (of the documents called conspiracy, how many are), *recall* (of the real conspiracy documents, how many were found) and *F1* (their harmonic mean), plus macro-F1 (F1 averaged over both classes) and balanced accuracy (average of recall in both classes).

**Confidence intervals.** A **95% confidence interval** (CI) is a range that would contain the true value in about 95 of 100 repeats of the study. We use the **bootstrap**: resample the validation set many times (2,000) and read off the 2.5% and 97.5% points. The *document* bootstrap resamples articles; the *website* bootstrap resamples whole websites, which is the honest one because articles from one site are not independent. Only 19 validation websites exist, so website CIs are wide.

**Comparing with M1.** For each model we resample the *same* documents and compute the **difference in F1 against M1** (a paired bootstrap), so shared luck cancels. The **McNemar test** looks only at documents where exactly one of the two models is right and asks whether the split between "only M1 right" and "only the new model right" differs from 50/50; a small p-value means the models really differ.

**Verdict rule** (`src/fml/ablation.py#verdict`): *helps* only if the whole CI of the F1 difference lies above 0, *hurts* if it lies below 0, otherwise *no clear gain*. We use the website-bootstrap CI for the verdict. We never call an improvement inside the noise a gain.

<!-- cell: src/fml/ablation.py#verdict -->
<!-- cell: src/fml/ablation.py#run_ablation -->
<!-- cell: src/fml/pipeline.py#ablation_variants -->

### Validation results

<!-- output: results/tables/ablation.csv -->
<!-- output: results/figures/ablation_f1.png | Validation conspiracy-F1 with 95% website-bootstrap CIs; bars are neutral, the chosen model is highlighted -->
<!-- output: results/metrics.json#ablation.verdicts -->

### Verdicts

All numbers below are in `results/tables/ablation.csv` (columns `delta_f1_vs_m1*`, `mcnemar_p_m1`, `contribution_f1_vs_m2*`). M1 has validation F1 0.9268.

| Hypothesis | Added to M1: ΔF1 [website CI] | McNemar p | Dropped from M2: ΔF1 [website CI] | Verdict |
|---|---|---|---|---|
| H2 n-grams | -0.0031 [-0.0234, 0.0159] | 1.0 | -0.0146 [-0.0395, 0.0302] | no clear gain |
| H3 epistemic | +0.0060 [-0.0278, 0.0469] | 1.0 | -0.0044 [-0.0318, 0.0324] | no clear gain |
| H4 style | -0.0098 [-0.0561, 0.0271] | 0.5488 | -0.0089 [-0.0398, 0.0213] | no clear gain |
| M2 (all three) | -0.0244 [-0.0629, 0.0123] | 0.2379 | – | no clear gain |

Every interval contains 0, so **none of H2, H3 or H4 is supported**, and none is refuted either: the data cannot tell a gain of a few points from no gain. The point estimate for epistemic markers (+0.006) is the only positive one and is far inside the noise. Style and the combined model M2 have negative point estimates, also inside the noise; we do not claim they hurt.

**Model choice (validation only).** The rule is: take M2 only if its website-CI for ΔF1 against M1 lies fully above 0. It does not, so **M1 (`lr_words`) is the chosen logistic regression** (`ablation.chosen_lr` in `metrics.json`). Both M1 and M2 were then scored once on the locked test split with `fml evaluate-final`, because the comparison with the LLM needs a reported number for both; the choice was already fixed on validation.

### What the models learned

<!-- output: results/tables/top_ngrams.csv -->
<!-- output: results/tables/dense_coefficients.csv -->

`top_ngrams.csv` lists the highest-weighted n-grams per class. Several top terms look like web page furniture or website habits (for example "was posted"), not belief; this is the risk that a website split exposes. `dense_coefficients.csv` shows standardised coefficients (per one standard deviation of the feature): the hidden-truth rate is the largest, then the ALL-CAPS share, matching the descriptive table in section 07. Large coefficients are not proof of usefulness; the ablation above says they add nothing measurable beyond words.

### Locked test scores

<!-- output: results/tables/test_lr.csv -->
<!-- output: results/metrics.json#ablation.test -->

Each model was scored on test exactly once (`results/test_access_log.jsonl` has one line each for `lr_words` and `lr_extended`). The test conspiracy-F1 of M1 is 0.5605 (website CI 0.3183 to 0.7925) and of M2 is 0.5829 (website CI 0.3137 to 0.8221), compared with about 0.93 on validation. The drop is large and the two test intervals overlap almost completely, so the test set does not separate M1 from M2 either. Per-document predictions (ids, labels, predictions, probabilities; no text) are in `results/tables/predictions_lr.csv` for the head-to-head and error analysis.

**Literature.** Our groups follow Wang and Manning (2012) for n-grams, Islam et al. (2020), Marini and Jezek (2024) and Rashkin et al. (2017) for epistemic markers, and Horne and Adali (2017) and Perez-Rosas et al. (2018) for style. The latter found style features strong inside one domain but weak across domains, which fits a flat result on unseen websites.

## What could go wrong

- **Few websites.** 19 validation and 18 test websites. A website CI of about ±0.07 F1 hides any real gain smaller than that.
- **Wide CIs, repeated looks.** We compare eight models on one validation set; some differences would look "significant" by chance. We therefore require the CI, not a point estimate, and still count nothing as helping.
- **Weak lexicon.** The word lists ignore negation and are short and hand-picked (section 07), so a null result for H3 is partly a result about our list, not about epistemic language.
- **Emotional topic.** 9/11 reporting is emotional in mainstream outlets too, so style and certainty cues separate sources less than they would for other topics.
- **Tuning on validation** makes validation scores slightly optimistic for every model; the test gap is partly this and partly new websites.
- **Website names in features.** Top n-grams can encode a website; the split by website limits but cannot remove this.

## Technical infrastructure

- `src/fml/ablation.py`: `run_ablation` (fit, tune on validation, bootstrap CIs, paired difference, McNemar), `verdict`, `group_verdicts`, `choose_between`, `coefficient_tables`.
- `src/fml/pipeline.py`: `ablation_variants`, stage `ablation`, `evaluate_final` (test lock and log), `plot_ablation`.
- Command: `fml run-all` (stage `ablation`); `fml evaluate-final lr_words` and `fml evaluate-final lr_extended` for the test scores.
- Outputs: `results/tables/ablation.csv`, `top_ngrams.csv`, `dense_coefficients.csv`, `test_lr.csv`, `predictions_lr.csv`, `results/figures/ablation_f1.png`, `metrics.json` keys `ablation` and `baseline`.
- Tests: `tests/test_ablation.py` (a planted informative feature is found; an uninformative one is not called a gain; identical numbers on rerun; one test-log entry per model).
