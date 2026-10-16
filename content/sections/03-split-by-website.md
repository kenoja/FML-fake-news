---
title: Splitting by website
part: Evaluation
hypothesis: Keeping every website entirely inside one split gives a lower but honest score than a random split, and our split has enough conspiracy websites in each part to be usable.
---
## Goal and hypothesis

We divide the articles into **train** (the model learns from it), **validation** (we compare options on it) and **test** (looked at once, at the end). The split is **grouped by website**: all articles of one website go to the same part. We expect this to give lower, honest scores than a random split, and we check that each part holds enough conspiracy websites (at least 3).

## Explanation

**Open question 1: what does "grouped by website" mean, is the training data fed per website?** No, the model is not fed one website at a time. Training is ordinary: the model sees all training articles together, mixed. "Grouped" only describes how we decide **which articles are in the training set at all**. The training set should be diverse (many websites), but the real rule is: *a website is never in two parts*.

A small example with three websites (invented names, ten articles each):

| website | label | random split | grouped split |
|---|---|---|---|
| site-A | conspiracy | 7 train, 1 validation, 2 test | all 10 in train |
| site-B | mainstream | 6 train, 2 validation, 2 test | all 10 in validation |
| site-C | mainstream | 8 train, 1 validation, 1 test | all 10 in test |

With the random split the model sees eight articles of site-C in training, with the site's typical wording, layout and topics, then is tested on two more from the same site. It can score well by **recognising the site**. With the grouped split, site-C is completely new at test time, so a good score means the model learned something about the texts that carries over to unseen websites. That is the situation we care about. The cost: fewer, coarser units to divide, so the part sizes only approximate 70/15/15.

**How we did it.** Websites, not articles, are assigned. Within each class (conspiracy, mainstream), websites are placed from largest to smallest into the part whose share of its target (70% / 15% / 15% of documents) would be lowest after adding the site. This is **stratified by label**: each class is divided separately, so every part has the corpus's class mix (about 30% conspiracy). A fixed random seed from `config/project.yaml` breaks ties. The result is stored once as `data/splits/split_v1.csv` (`doc_id`, `split`) and covers all 3,146 documents. The function refuses to overwrite an existing file, and every model uses the same file. The off-topic filter and any duplicate removal later only drop rows. A dropped document keeps its split.

<!-- cell: src/fml/split.py#assign_websites -->
<!-- cell: src/fml/split.py#make_split -->
<!-- output: results/tables/split_summary.csv -->
<!-- output: results/figures/split_sizes.png | Documents and websites per split and class -->
<!-- output: results/metrics.json#split -->

Reading the table: `websites` and `documents` per split and class; the split sizes are close to 70/15/15. Every part has at least 3 conspiracy websites (the minimum we set), so the warning flag in `metrics.json#split.few_conspiracy_websites_flag` is false and no per-website table was needed. Validation and test each have only a handful of conspiracy websites, so a single site can still move the score. Later confidence intervals therefore also resample whole websites.

A **grouped split** is not a **confidence interval**: the interval (a range of plausible values for the true score, given how few documents and sites we have) comes later, in the head-to-head section.

## What could go wrong

- *Few conspiracy websites in test*: a score then depends on a few sites. Reported, not hidden.
- *Near-copies across splits*: different websites sometimes publish the same story (see the cleaning log in the previous section). Grouping by website cannot stop this.
- *Class share per part drifts* because websites are indivisible. The table shows the real shares.
- *Re-splitting after seeing results* would be a form of tuning on test. The code refuses to overwrite the file, and a test checks that.

## Technical infrastructure

- `src/fml/split.py`: `make_split`, `load_split`, `assign_websites`, `split_summary`, `check_conspiracy_websites`.
- File: `data/splits/split_v1.csv`; settings (ratios, seed, minimum) in `config/project.yaml`.
- Command: `fml prepare-data` creates the split only if the file is missing, otherwise reads it.
- Tests in `tests/test_split_clean.py`: no website in two parts, rough stratification, refusal to overwrite, determinism, and (with the data repo) that the code reproduces the frozen file.
