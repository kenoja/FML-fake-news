---
title: Results and why they fail
part: Evaluation
hypothesis: The zero-shot LLM reaches a higher conspiracy-class F1 on the test set than the best logistic regression, and the McNemar test gives p < 0.05.
---
## Explained simply

**How to read the table.** Each of the four final models takes the final exam once, on test articles nobody used before. For each model the table shows three scores for the conspiracy class: **precision** (when it says "conspiracy", how often is it right?), **recall** (of all real conspiracy articles, how many did it find?) and **F1**, one number that is low if either of the other two is low. Next to each score is a **confidence interval**. Like a poll that says "40%, give or take 3 points", it shows how far a score may be off.

**Why F1 and not accuracy.** Most of the articles are mainstream. A lazy "model" that always answers "mainstream" is right about three times in four, which sounds decent and finds no conspiracy article at all. That is the baseline row of the table: its F1 is zero.

**A worked case.** The test set has 96 conspiracy articles and 314 mainstream ones. The LLM caught 64 of the 96 (recall 0.667) and wrongly flagged 1 mainstream article, so 64 of its 65 alarms were right (precision 0.985); its F1 is 0.795. Intervals are a bit optimistic.

**Why McNemar for the comparisons.** Two judges rate the same cases. Where they agree, we learn nothing about who is better. Only the cases where they disagree count. If one judge is right in 20 of 22 disagreements, that judge is clearly better; if it is 11 against 11, it may be luck. McNemar's test does this with our models, on the same articles: it counts the articles where only one of the two is right. A confidence interval for each model alone cannot compare a pair, because it says nothing about who wins on the same articles.

**Why validation scores are higher.** Models were tuned on validation articles and then scored on test articles from other websites, and there are fewer conspiracy websites in the test set. The per-website table shows where the test scores are lost: one large site in the test set is hard for every model.

**Model errors and label problems.** The label belongs to the website, not to the article. A conspiracy site can print an ordinary news item, and then the label describes the text badly. To tell the two apart, a second language model read each disputed article and ticked what it does with conspiracy claims: endorses, debunks, reports neutrally, or makes no claim. A conspiracy-site article without an endorsed claim, or a mainstream article that endorses one, counts as a label problem; everything else is a model error. Because a machine did the ticking, the team checks 40 articles by hand.

## The code

Each model gets one row of counts, the three scores and bootstrap intervals. The baseline "always guess mainstream" is one more row.

<!-- cell: src/fml/comparison.py#model_row -->

The McNemar test counts who is right where the other is wrong.

<!-- cell: src/fml/comparison.py#mcnemar_pair -->

The stage reads the saved test predictions, builds the table and the figures, and runs the three tests.

<!-- cell: src/fml/pipeline.py#align_predictions -->

<!-- cell: src/fml/pipeline.py#run_mcnemar_pairs -->

<!-- cell: src/fml/pipeline.py#write_website_recall -->

<!-- cell: src/fml/pipeline.py#comparison_stage -->

<!-- output: results/tables/comparison_compact.csv | Precision, recall and F1 of the conspiracy class on the test articles, with 95% confidence intervals -->
<!-- output: results/figures/comparison_confusion.png | Confusion matrix of each model on the test articles -->
<!-- output: results/tables/mcnemar_tests.csv | The three McNemar tests: articles where only one model is right, and the p-value -->
<!-- output: results/figures/validation_vs_test_confusion.png | Confusion matrices of the best logistic regression and the LLM on validation (top) and test (bottom) -->
<!-- output: results/metrics.json#comparison.baseline | The baseline that always guesses mainstream -->
<!-- output: results/metrics.json#comparison.mcnemar.words_vs_tfidf | McNemar test: word counts against TF-IDF -->
<!-- output: results/metrics.json#comparison.mcnemar.tfidf_vs_features | McNemar test: TF-IDF against TF-IDF with features -->
<!-- output: results/metrics.json#comparison.mcnemar.best_lr_vs_llm | McNemar test: best logistic regression against the LLM -->

To see where the test scores are lost, this cell counts, for every conspiracy website in the test set, how many of its articles each model found.

<!-- cell: src/fml/pipeline.py#conspiracy_website_recall -->

<!-- output: results/tables/test_website_recall.csv | Recall of each model on the conspiracy articles of each test website -->
<!-- output: results/metrics.json#comparison.f1_without_large_site | F1 of each model without the large site -->

The same count for the validation set, for the LLM only, shows that validation held no site as hard as the large test site.

<!-- cell: src/fml/comparison.py#validation_website_table -->

<!-- output: results/tables/validation_website_recall.csv | Recall of the LLM on the conspiracy articles of each validation website -->

Reading the table, finding by finding.

- The site with one article (Website 5) is too small to tell us anything.
- The large site, Website 4, supplies 44 of the 96 conspiracy test articles and every model finds about a third of them or fewer, so this one site pulls every test score down.
- On the five sites with more than one article the LLM finds at least as many articles as every logistic regression, and more on 4 of the 5; the one tie is Website 6.

Now the errors. First we find, for each test article, who is wrong.

<!-- cell: src/fml/errors.py#disagreement_sets -->

The stance coding sends each article to the second model once and stores only the id and the answer, never the text.

<!-- cell: src/fml/errors.py#code_stances -->

The rule turns the label and the stance into a verdict.

<!-- cell: src/fml/errors.py#classify_error -->

The examples are picked with a seeded shuffle among articles that have a clean excerpt.

<!-- cell: src/fml/errors.py#pick_examples -->

The stage puts it all together and writes the tables.

<!-- cell: src/fml/pipeline.py#label_problem_values -->

<!-- cell: src/fml/pipeline.py#errors_values -->

<!-- cell: src/fml/pipeline.py#errors_stage -->

<!-- output: results/tables/error_classification.csv -->
<!-- output: results/tables/error_counts_plain.csv -->

<!-- output: results/tables/error_examples_plain.csv | Examples where the models disagree -->

Each row is one test article with a short excerpt. The first two rows show that the two models make different mistakes: each is sometimes right where the other is wrong. The rows where both models are wrong, or where the label does not fit the text, show articles that carry little or no conspiracy claim, so the website label describes them badly.
<!-- output: results/metrics.json#errors.label_problem_share -->

## Report

All four final models were scored once on the same 410 test articles (96 from conspiracy-oriented websites). The table gives precision, recall and F1 for the conspiracy class with 95% bootstrap confidence intervals (Efron and Tibshirani, 1993; 2,000 resamples of documents). The baseline row always answers "mainstream"; its accuracy is 0.766, which is why accuracy is not used.

<!-- output: results/tables/comparison_compact.csv | Precision, recall and F1 of the conspiracy class on the test articles, with 95% confidence intervals -->

<!-- output: results/figures/comparison_confusion.png | Confusion matrices on the test set (rows: true website type, columns: predicted) -->

The LLM has the highest F1 (0.795, interval 0.723 to 0.857), followed by TF-IDF (0.598), word counts (0.567) and TF-IDF with features (0.560). The LLM is almost never wrong when it says "conspiracy" (precision 0.985) but misses a third of the conspiracy articles (recall 0.667).

Each pair of models is compared with an exact two-sided McNemar test (McNemar, 1947) on the articles where only one of the two is right, as set out in section 1.

<!-- output: results/tables/mcnemar_tests.csv | The three McNemar tests: articles where only one model is right, and the p-value -->

**H1: TF-IDF weighting beats plain word counts.** Supported: 7 articles are right only for word counts and 19 only for TF-IDF (p = 0.029). TF-IDF has the higher F1 (0.598 against 0.567), mainly through fewer false alarms (precision 0.721 against 0.607). It is the smallest of the three effects and would not pass a stricter threshold that corrects for running three tests (0.05 divided by three), so the support is modest.

**H2: Adding the three feature groups improves the TF-IDF model.** Not supported. On validation the model with the extra features scored an F1 of 0.896 against 0.918 for TF-IDF alone, and on test it is lower again (0.560 against 0.598); 22 articles are right only for TF-IDF and 14 only with the features, a difference that chance explains (p = 0.243).

**H3: The LLM beats the best logistic regression.** Supported. F1 is 0.795 against 0.598, the intervals do not overlap, and 4 articles are right only for the best logistic regression (TF-IDF) against 37 only for the LLM (p < 0.0001).

**Why the scores are lower on test than on validation.** Validation F1 was 0.865 for word counts, 0.918 for TF-IDF, 0.896 for TF-IDF with features and 0.987 for the LLM. The test set has fewer and different websites, so the models face sites they never saw. The table gives the recall of every model on each test website.

<!-- output: results/figures/validation_vs_test_confusion.png | Confusion matrices on validation (top) and test (bottom); rows: true website type, columns: predicted -->

<!-- output: results/tables/test_website_recall.csv | Recall of each model on the conspiracy articles of each test website -->

Three findings stand out.

- The large site, Website 4, supplies 44 of the 96 conspiracy test articles, and all four models find at most about a third of its articles (recall 0.204 to 0.318).
- Without those 44 articles (366 left, 52 from conspiracy sites) the F1 is 0.640 for word counts, 0.721 for TF-IDF, 0.661 for TF-IDF with features and 0.971 for the LLM; on the other sites the logistic regressions find 0.579 to 1.000 of the articles.
- On every site with more than one article, the LLM finds at least as many articles as each logistic regression.

**Where the models fail, and why.** The labels describe websites, not articles. We examined the test articles on which the best logistic regression or the LLM disagrees with the label. A second model (Claude Sonnet 5.5) coded the stance of each towards conspiracy claims about 9/11, without being told the website. A fixed rule counts a conspiracy-site article without an endorsed claim, or a mainstream article with one, as a label problem; the rest are model errors.

Of the 70 wrongly classified articles, 69 have a readable coding: 49 are label problems and 20 are model errors. One article was left out because the second model's answer was not in the required form. The table gives the three groups with the share of coded articles that are label problems.

<!-- output: results/tables/error_counts_plain.csv | Test articles wrongly classified, by who is wrong, and the share that are label problems -->

Two test articles show the pattern. A conspiracy-site article that both models called mainstream begins "A military judge has set the trial date for five men accused of masterminding the 9/11 attacks": a plain court report without a conspiracy claim, so the label rather than the models is at fault. A mainstream article opening "Not only is there a smoking gun in the 28 pages" was called conspiracy by the logistic regression and mainstream by the LLM; the stance coder found no conspiracy claim in it, so here the trained model reacted to wording that only sounds like conspiracy writing.

The label-problem share rests on automatic coding by a model of the same family as the classifier under test, so it is an estimate. To measure how far it can be trusted, two team members each code 40 validation articles blind (website and automatic answer hidden; validation keeps the test set untouched); their agreement with the automatic coding follows this report.

## Check yourself
Q: A model that always answers "mainstream" is right most of the time. Why do we still not use accuracy?
A: Because most articles are mainstream, the lazy rule scores high accuracy while finding no conspiracy article. F1 for the conspiracy class is zero for it, so it shows the failure.

Q: Why can the LLM have a precision of 0.985 but a recall of only 0.667?
A: It almost never calls a mainstream article conspiracy (1 false alarm), so what it flags is nearly always right. But it misses 32 of the 96 conspiracy articles, so it finds only two thirds of them.

Q: Why does McNemar's test look only at articles where one model is right and the other wrong?
A: Both models saw the same articles. Where they agree, there is nothing to compare, so the evidence about a difference sits only in the disagreements.

Q: A site supplies 44 of 96 conspiracy test articles, and a model finds only 14 of them. What happens to the model's overall recall?
A: Those 44 articles make up almost half of all conspiracy articles, so at most 52 other articles remain to be found. Even if the model found all of them, recall would stay below (52 + 14) / 96 = 0.69, so one missed site drags the whole score down.

Q: Why is a conspiracy-site article with no conspiracy claim a label problem and not a model error?
A: A model can only judge the text it is given. If the text contains no conspiracy claim, no model could tell from the text that the website is conspiracy-oriented, so blaming the model would be unfair; the label is what fits badly.
