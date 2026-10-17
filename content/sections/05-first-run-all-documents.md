---
title: First run on all documents
part: Evaluation
hypothesis: H1 - a words-only logistic regression clearly beats the baselines on unseen websites, though the high score may partly reflect how the sites write rather than what they argue.
---
## Goal and hypothesis

Before cleaning up the topic we run M1 (words only, 1-word TF-IDF, L2 logistic regression) on **all 3,146 documents**, including those that never mention 9/11. Hypothesis H1: it beats guessing by a wide margin on validation websites it has never seen. We report validation only. The test split stays locked.

## Explanation

**Baselines first.** A baseline is a trivial model that tells us what a score means. *Majority class*: always answer "mainstream" (the larger class). *Stratified random*: guess "conspiracy" at random, as often as it occurs in training. A real model must beat both.

**Open question 4: what are precision, recall and F1?** We count validation articles (471, of which 140 come from conspiracy websites). The table of right and wrong calls is the **confusion matrix**:

<!-- output: results/figures/baseline_all_docs_confusion.png | Confusion matrix of M1 on validation -->

Reading it with our numbers (`results/metrics.json#baseline_all_docs.confusion`): 125 conspiracy articles were found (true positives), 15 were missed (false negatives), 5 mainstream articles were wrongly called conspiracy (false positives), 326 mainstream were correct (true negatives).

- **Precision** answers: "when the model says conspiracy, how often is it right?" = true positives / (true positives + false positives) = 125 / (125 + 5) = 0.962.
- **Recall** answers: "of all real conspiracy-site articles, how many did it find?" = 125 / (125 + 15) = 0.893.
- **F1** is the harmonic mean of the two, 2 × precision × recall / (precision + recall) = 0.926. The harmonic mean is pulled towards the smaller number, so F1 is high only if *both* are high. A model that says "conspiracy" for everything has recall 1 but low precision, and a low F1.

We report the **conspiracy class** because it is the minority and the one we want to find. **Macro-F1** averages the F1 of both classes, and **balanced accuracy** averages recall of both classes, so the big mainstream class cannot hide a failure. Plain accuracy is not used as the headline: the majority baseline would already look good on it.

<!-- output: results/metrics.json#baseline_all_docs.validation -->
<!-- output: results/metrics.json#baseline_all_docs.baselines -->

**Confidence interval (CI).** Our validation set is one sample. A 95% CI is a range of plausible values for the true score. We compute it by **bootstrap**: draw 2,000 new validation sets by sampling with replacement, recompute the metric each time, and take the middle 95%. We do it twice. *Document bootstrap* resamples articles. *Website bootstrap* resamples whole websites, because articles of one site are alike and the validation set has only 20 websites (7 conspiracy). The website interval is the more honest one. For F1: 0.926, document CI 0.892 to 0.957, website CI 0.791 to 0.957 (`validation` and `validation_website_bootstrap` in `metrics.json`).

<!-- output: results/figures/baseline_all_docs_pr.png | Precision-recall curve of M1 on validation -->

The curve shows precision against recall as we move the decision threshold. Its area (average precision) is 0.970 here.

**Baselines in numbers.** Majority class: F1 0.000 (it never finds a conspiracy article), macro-F1 0.413, balanced accuracy 0.500. Stratified random: F1 0.297, balanced accuracy 0.500. M1: F1 0.926, balanced accuracy 0.939.

**Honest reading of H1.** H1 holds on validation: M1 is far above both baselines, even the lower end of the website-bootstrap interval (F1 0.791) is far above 0.297. But we should not read this as "the model understands conspiracy content". Three cautions: (1) the label is about the **website**, so the model can use how a site writes, and near-copies of stories exist across splits; (2) this run includes documents that never mention 9/11, which are easy to tell apart by topic alone; (3) one validation set with 7 conspiracy websites is a small sample. Documents under 250 words are few, so the by-length table (`m1_words_all_docs_length_buckets.csv`) is too small to claim a length effect.

**What the model uses.** The highest-weighted words (`m1_words_all_docs_top_coefficients.csv`):

<!-- output: results/tables/m1_words_all_docs_top_coefficients.csv -->

No website name remains among the top 25 words per class (`top_coefficient_note` in `metrics.json`): cleaning removed them. But some top words are **formatting traces of blog-style sites**, such as "posted", "entry", "rss", "pm", "am", "below", "here", "youtube", not claims about 9/11. That is a style giveaway, not a name giveaway, and it is a reason to read the score with care. Mainstream-leaning words ("tower", "killed", "counterterrorism", "symptoms") look like news reporting. The off-topic filter and the feature experiments in the next sections build on this.

## What could go wrong

- *Style giveaways* (listed above) can inflate the score without the model reading arguments.
- *Small validation set*: wide website-level interval; one site can move F1 noticeably.
- *Tuning on validation*: the selected C is on the edge of the grid, and validation numbers are slightly optimistic because we picked the best of 18 settings on them. Test is untouched.
- *Rounding*: numbers here are rounded to three decimals from `results/metrics.json`.

## Technical infrastructure

- `src/fml/pipeline.py#baseline_all_docs` (stage), `src/fml/evaluate.py` (`classification_metrics`, `bootstrap_ci`, `majority_baseline`, `stratified_random_baseline`, `pr_curve_data`).
- Cleaned text (URLs, site names and repeated boilerplate removed) is built in memory by `pipeline.cleaned_frame`. Both exact duplicates are kept here so the run covers all 3,146 documents.
- Outputs: `results/metrics.json#baseline_all_docs`, `results/tables/m1_words_all_docs_*.csv`, `results/figures/baseline_all_docs_*.png`.
- Run: `fml run-all` (currently runs this stage and stops with a message about the stages not written yet).

<!-- cell: src/fml/pipeline.py#baseline_all_docs -->
<!-- cell: src/fml/evaluate.py#bootstrap_ci -->
