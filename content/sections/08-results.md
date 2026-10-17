---
title: Results
part: Evaluation
hypothesis: Counting words is weaker than weighting them, extra features help a little, and the LLM beats every model we trained ourselves.
---
## Explained simply

Each of the four final models takes the final exam once, on test articles nobody has looked at. All four get the same articles, so the comparison is fair. We explain the scores with one real case: the LLM on the test set.

**Part 1: what did the model get right?**

The test set has 96 conspiracy articles and 314 mainstream ones. The LLM's answers fill a **confusion matrix**, a small table with four counts:

- 64 conspiracy articles caught,
- 32 conspiracy articles missed,
- 1 mainstream article wrongly flagged (a false alarm),
- 313 mainstream articles correctly left alone.

**Precision** asks: when the model says "conspiracy", how often is it right? It flagged 65 articles and 64 were right: 64 / 65 = 0.985. Like a smoke alarm: how often is there really a fire when it rings?

**Recall** asks: of all real conspiracy articles, how many did it find? 64 of 96: 0.667. For the alarm: of all real fires, how many did it catch?

**F1** blends the two into one number that is low if either is low: 0.795. We always look at the conspiracy class, because that is the group we want to find.

**Part 2: is the difference real?**

Our test set is only a sample, so every score could be a bit different with other articles. A **confidence interval** shows how much. We get it with the **bootstrap**: draw 410 articles from the test set at random, putting each back after drawing, score the model, and repeat many times. The range holding the middle bulk of the scores is the interval. For the LLM's recall it is 0.570 to 0.759: the true value could easily be a bit lower or higher.

One warning: this method treats every article as independent. But articles from one website resemble each other, and websites are the real unit of dependence. With few websites, the intervals are therefore too narrow, that is, optimistic.

The **McNemar test** asks whether two models really differ. It looks only at articles where exactly one of them is right. If one model wins almost all of these, the difference is clear; if the wins are split evenly, it may be luck. The **p-value** says how likely a split this lopsided is if both models were equally good. A small one is called *statistically significant*; it does not say the difference is large.

We fixed three comparisons in advance, because testing many pairs would sooner or later give a "significant" result by luck.

## The code

Each model gets one row of counts, the three scores and bootstrap intervals. The report shows a compact version of this table; the four counts stay in the confusion-matrix figure.

<!-- cell: src/fml/comparison.py#model_row -->

The McNemar test counts who is right where the other is wrong.

<!-- cell: src/fml/comparison.py#mcnemar_pair -->

The stage reads the saved test predictions, builds the table and the figures, and runs the three tests.

<!-- cell: src/fml/pipeline.py#align_predictions -->

<!-- cell: src/fml/pipeline.py#run_mcnemar_pairs -->

<!-- cell: src/fml/pipeline.py#comparison_stage -->

<!-- output: results/tables/comparison_compact.csv | Precision, recall and F1 of the conspiracy class on the test articles, with 95% confidence intervals -->
<!-- output: results/figures/comparison_confusion.png | Confusion matrix of each model on the test articles -->
<!-- output: results/figures/comparison_f1.png | Conspiracy F1 with 95% confidence interval -->

## Report

All four final models were scored once on the same 410 test articles (96 from conspiracy-oriented websites). The table gives precision, recall and F1 for the conspiracy class with 95% bootstrap confidence intervals (2,000 resamples of documents). The first figure shows the confusion matrices and the second the F1 scores with their intervals.

<!-- output: results/tables/comparison_compact.csv | Precision, recall and F1 of the conspiracy class on the test articles, with 95% confidence intervals -->

<!-- output: results/figures/comparison_confusion.png | Confusion matrices on the test set (rows: true website type, columns: predicted) -->

<!-- output: results/figures/comparison_f1.png | Conspiracy F1 on the test set with 95% confidence intervals -->

The LLM has the highest F1 (0.795, interval 0.723 to 0.857), followed by TF-IDF with features (0.610), TF-IDF (0.582) and word counts (0.559). The three logistic regressions have overlapping intervals. Their precision is moderate (0.602 to 0.735) and their recall is about one half (0.500 to 0.521). The LLM is almost never wrong when it says "conspiracy" (precision 0.985, one false alarm among 314 mainstream articles) but still misses a third of the conspiracy articles (recall 0.667).

Three McNemar tests (exact, two-sided) compare pairs of models. Word counts against TF-IDF: 9 articles are right only for word counts and 19 only for TF-IDF (p = 0.087). TF-IDF against TF-IDF with features: 6 against 11 (p = 0.332). The best logistic regression on validation (TF-IDF with features) against the LLM: 3 against 34 (p < 0.0001).

**H1: TF-IDF weighting beats plain word counts.** Not confirmed. TF-IDF has a higher F1 (0.582 against 0.559), but the intervals overlap almost completely and the McNemar test is not significant (p = 0.087). The direction fits the hypothesis; the evidence is too weak to claim it.

**H2: Adding hand-made features improves the TF-IDF model.** Each kind was tried alone on validation (TF-IDF alone: F1 0.914). **H2a, certainty words:** the only group above our threshold (F1 0.930), so it went to the test set. There F1 rises from 0.582 to 0.610, mainly through fewer false alarms (precision 0.735 against 0.696), but the intervals overlap and the difference is within chance (p = 0.332). Not confirmed. **H2b, style features:** not supported; validation F1 falls to 0.894, so this model was not scored on test. **H2c, phrases (n-grams):** not supported; validation F1 is 0.916, no clear gain, so it was not scored on test either.

**H3: The LLM beats the best logistic regression.** Confirmed. F1 is 0.795 against 0.610, the intervals do not overlap, and the test is clearly significant (p < 0.0001). The gain comes from precision, not from recall.

**How to read these verdicts.** The intervals resample documents, but websites are the real unit of dependence: articles from one website resemble each other and the test set contains few websites, so the intervals are optimistic, that is, narrower than they should be. "Not confirmed" means that the data do not show the effect, not that it is absent. Only the gap to the LLM is large enough to be clearly visible.

**What the numbers say about the models.** The logistic regressions behave alike: they find about half of the conspiracy articles and raise a moderate number of false alarms. Of the 314 mainstream test articles, word counts flag 33, TF-IDF flags 21 and TF-IDF with features flags 18, so the features mainly make the model more careful rather than more sensitive. The LLM hardly ever flags a mainstream article, but it lets one third of the conspiracy articles pass. Why each model misses the articles it misses is taken up in the section on where the models fail.

**Summary.** The ranking by F1 is LLM, TF-IDF with features, TF-IDF, word counts. Only the first place is clearly separated from the rest.

## Check yourself
Q: What would happen to precision and recall if a model flagged every single article as conspiracy?
A: Recall would be perfect, because no conspiracy article is missed. Precision would be poor, because most flagged articles are mainstream. This is why we need both numbers.

Q: Why can the LLM have a precision of 0.985 but a recall of only 0.667?
A: It almost never calls a mainstream article conspiracy (1 false alarm), so what it flags is nearly always right. But it misses 32 of the 96 conspiracy articles, so it finds only two thirds of them.

Q: Why do we show a confidence interval and not only the F1 number?
A: The test set is just one sample of articles. The interval shows how much the score could change with other articles, so we do not over-read small differences.

Q: TF-IDF with features has a higher F1 than TF-IDF alone, yet the McNemar test gives p = 0.332. What does that mean?
A: A split as uneven as 11 against 6 happens quite often by chance when two models are equally good. The extra features may help, but this test cannot show it.

Q: Why would a reader who wants to catch every conspiracy article still be unhappy with a model that has high precision?
A: High precision only says that its alarms are right. If recall is low, many conspiracy articles still pass as mainstream, as with the LLM, which lets about one in three through.
