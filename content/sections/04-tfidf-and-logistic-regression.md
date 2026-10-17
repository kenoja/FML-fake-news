---
title: TF-IDF and logistic regression
part: Methods
hypothesis: Counting words with TF-IDF weights and fitting a regularised logistic regression is enough to separate conspiracy websites from mainstream ones better than guessing, and it must be tuned on validation only.
---
## Goal and hypothesis

We build the first model, **M1**: it only sees words (single words, no extra features). We expect it to beat guessing clearly. We also set up the tools every later model uses: one pipeline object that learns only from the training split, a tuning loop that only looks at validation, and an evaluation toolkit.

## Explanation

**Open question 2: "the model gets labelled texts, assigns numerical importance to words, and from those statistics learns to recognise conspiracy-site articles. Is this right?"** Almost. Two different things get numbers, and only one of them is learned from the labels. Step by step:

1. **Labelled texts go in.** Each training article has a label: its website is conspiracy-oriented (1) or mainstream (0). The label describes the *website*, not what the article argues.
2. **Words are counted.** For every article we count how often each word occurs. No labels are used here.
3. **TF-IDF turns counts into weights.** *TF-IDF* (term frequency times inverse document frequency) gives a word a high number when it is frequent in this article but rare across the training articles. A word like "the" appears everywhere, so it gets a low weight. A word like "controlled" in one article gets a higher one. These numbers are **fixed formulas of word counts**. Nothing is learned from labels in this step, and the importance is not yet "conspiracy" or "mainstream". The vocabulary and the "rare across articles" part are computed on the training split only.
4. **Logistic regression learns coefficients.** Now the labels are used. For every word the model learns one **coefficient**: positive pushes the prediction towards "conspiracy site", negative towards "mainstream". An article's score is the sum of (TF-IDF weight × coefficient) over its words, turned into a probability between 0 and 1. Training searches for the coefficients that fit the training labels best.
5. **Prediction.** For a new article, the same fixed TF-IDF recipe produces its weights, the learned coefficients are applied, and the article is called conspiracy if the probability is above 0.5.

So the correction is: *the importance of a word to the **label** (the coefficient) is learned; the TF-IDF number is a fixed recipe applied to word counts*. "Statistics" is fair for step 3, "learns" is correct for step 4.

**Regularisation** is a penalty on large coefficients. Without it, a model with tens of thousands of words can memorise the training articles by giving rare words big coefficients. We use the common **L2** penalty (it shrinks all coefficients smoothly). **C** is the knob that sets how strong the penalty is, and it works backwards: a *small* C means a *strong* penalty (simple model), a *large* C a weak penalty (flexible model). **Class weight** tells the model how much a mistake on each class costs. With `none` every article counts the same, so the larger mainstream class dominates. With `balanced` a conspiracy article counts more, in proportion to how rare it is.

**Tuning.** We try nine values of C (0.01 up to 100) with both class-weight settings, 18 combinations. Each is fitted on the training split and scored on the validation split; the best conspiracy-class F1 wins. Every combination tried is written to the grid table, not only the winner. The test split is never read by this code: the loader function used for tuning raises an error if asked for it.

**Why one pipeline.** TF-IDF vocabulary and weights, and later the scaler for numeric features, are learned quantities. They live inside one scikit-learn `Pipeline`, so calling `fit` on the training rows is the only place anything is learned. If we built the vocabulary on all articles first, words that only occur in validation or test would shape the model, which is **leakage** and makes scores look better than they are. A test fits TF-IDF on train+validation on purpose and checks that our leak detector fails.

<!-- cell: src/fml/models.py#build_pipeline -->
<!-- cell: src/fml/models.py#tune_on_validation -->
<!-- output: results/tables/m1_words_all_docs_grid.csv -->

Reading the grid: small C (strong penalty) predicts almost nothing as conspiracy (recall near 0). Scores climb with C and flatten around C = 30 to 100. The best row is the one with `selected` true (see `results/metrics.json#baseline_all_docs.selected`). It sits at the edge of the grid, so we cannot rule out that a larger C would be slightly better. The gain from 30 to 100 is tiny, so we did not widen the grid after looking.

## What could go wrong

- *Leakage*: any learned quantity fitted on more than train. Tested.
- *Reading coefficients as proof*: a large coefficient means "useful for separating these websites in this data", not "this word signals conspiracy thinking". The coefficient table is examined in the next section.
- *Best value at the grid edge* (see above) and *tuning on a small validation set*: validation has few conspiracy websites, so the choice is noisy.
- *Class weight*: `balanced` raises recall at the cost of precision. We choose by F1, which balances both.

## Technical infrastructure

- `src/fml/models.py`: `build_pipeline`, `split_frames` (refuses `test`), `tune_on_validation`, `top_coefficients`.
- `src/fml/evaluate.py`: metrics, baselines, bootstrap intervals, McNemar test (explained when first used).
- Settings (seed, split) in `config/project.yaml`; the grid is in `models.C_GRID` and `models.CLASS_WEIGHTS`.
- Tests: `tests/test_leakage.py` (vocabulary from train only; a deliberately leaky function is caught), `tests/test_test_lock.py` (tuning ignores poisoned test rows).
