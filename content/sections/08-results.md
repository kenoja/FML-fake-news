---
title: Results
part: Evaluation
hypothesis: Counting words is weaker than weighting them, extra features help a little, and the LLM beats every model we trained ourselves.
---
## Explained simply

**One exam, four students.** Up to now every choice was made on the validation articles. Now each of the four final models takes the final exam once, on test articles nobody has looked at. All four get the same 410 articles, so the comparison is fair.

**Confusion matrix.** A small table that counts the four things that can happen. A conspiracy article can be caught or missed. A mainstream article can be left alone or wrongly flagged. Everything else is calculated from these four counts.

**Precision, recall and F1.** Think of a smoke alarm. *Precision* asks: when it rings, how often is there a fire? *Recall* asks: of all real fires, how many did it catch? Our own numbers show why both matter. The LLM flagged 65 articles and 64 were really conspiracy, so almost every alarm was right. But it caught only two thirds of the conspiracy articles. The word-count model caught about half, yet it also raised many false alarms on mainstream articles. *F1* is one number that blends precision and recall. It is low if either one is low. We always look at the conspiracy class, because that is the group we want to find.

**Confidence interval and bootstrap.** Our test set is only a sample. With other articles the scores would be a bit different. To see how much, we use the *bootstrap*: draw 410 articles from the test set with replacement (an article can appear twice), score the model, and repeat 2,000 times. The range that holds the middle 95% of these scores is the *95% confidence interval*. A wide interval means "do not trust small differences". If two intervals barely overlap, the models probably differ.

**McNemar test and p-value.** Do two models really differ, or did one just get lucky? The McNemar test looks only at articles where exactly one of the two models is right. If model A wins 3 of these and model B wins 34, that is very lopsided. If it were 19 against 9, that is less clear. The *p-value* answers: if the two models were equally good, how likely is a split at least this lopsided? A small p-value (below 0.05 by custom) means "unlikely to be luck", which we call *statistically significant*. It does not say the difference is large or important.

**Why these choices?** Only three comparisons are tested, fixed before looking at results, because testing many pairs would eventually produce a "significant" result by luck. We compare the best logistic regression with the LLM, because that is the main question of the study.

## The code

Each model gets one row of counts, the three scores and bootstrap intervals. The report shows a compact version of this table; the four counts stay in the confusion-matrix figure.

<!-- cell: src/fml/comparison.py#model_row -->

The McNemar test counts who is right where the other is wrong.

<!-- cell: src/fml/comparison.py#mcnemar_pair -->

The stage reads the saved test predictions, builds the table and the figures, and runs the three tests.

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

**H2: Adding certainty and style features improves the TF-IDF model.** Not confirmed on test. F1 rises from 0.582 to 0.610, mainly through fewer false alarms (precision 0.735 against 0.696), but the difference is within chance (p = 0.332).

**H3: The LLM beats the best logistic regression.** Confirmed. F1 is 0.795 against 0.610, the intervals do not overlap, and the test is clearly significant (p < 0.0001). The gain comes from precision, not from recall.

**How to read these verdicts.** The test set contains few websites, and articles from one website resemble each other, so the intervals are probably still somewhat optimistic. A "not confirmed" verdict means that the data do not show the effect; it does not show that the effect is absent. With 410 articles, differences of two or three F1 points between the logistic regressions cannot be told apart from chance. Only the gap to the LLM is large enough to be clearly visible.

**What the numbers say about the models.** The logistic regressions behave alike: they find about half of the conspiracy articles and raise a moderate number of false alarms. Of the 314 mainstream test articles, word counts flag 33, TF-IDF flags 21 and TF-IDF with features flags 18, so the features mainly make the model more careful rather than more sensitive. The LLM works differently. It reads the article as a whole and hardly ever flags a mainstream article, but it also lets one third of the conspiracy articles pass. It therefore has the better balance, though it is not a perfect detector either. The question why each model misses the articles it misses is taken up in the next section, together with the question of how many errors are caused by labels that describe the website rather than the article.

**Summary.** On the test set the ranking by F1 is LLM, TF-IDF with features, TF-IDF, word counts. Only the first place is clearly separated from the rest. Better weighting and extra features move the logistic regression in the expected direction, but the gain is small compared with the uncertainty of the estimate.

## Check yourself
Q: A model flags 10 articles as conspiracy and 8 of them really are. It misses 12 other conspiracy articles. What are its precision and recall?
A: Precision is 8 of 10, so 0.8. Recall is 8 of the 20 real conspiracy articles (8 caught plus 12 missed), so 0.4.

Q: Why do we show a confidence interval and not only the F1 number?
A: The test set is just one sample of articles. The interval shows how much the score could change with other articles, so we do not over-read small differences.

Q: TF-IDF with features has a higher F1 than TF-IDF alone, yet the McNemar test gives p = 0.332. What does that mean?
A: A split as uneven as 11 against 6 happens quite often by chance when two models are equally good. The extra features may help, but this test cannot show it.

Q: The LLM has precision 0.985 but recall 0.667. Describe in your own words what kind of mistake it makes.
A: It is very careful: when it says "conspiracy" it is nearly always right, but it lets about one in three conspiracy articles pass as mainstream.
