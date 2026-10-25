---
title: Results and why they fail
part: Evaluation
hypothesis: Weighting words beats counting them, the extra features help a little, and the LLM beats every model we trained; some "mistakes" are label problems, not model problems.
---
## Explained simply

**How to read the table.** Each of the four final models takes the final exam once, on test articles nobody used before. All four get the same articles, so the comparison is fair. For each model the table shows three scores for the conspiracy class: **precision** (when it says "conspiracy", how often is it right?), **recall** (of all real conspiracy articles, how many did it find?) and **F1**, one number that is low if either of the other two is low. Next to each score is a **confidence interval**: the range the score could easily take with other articles.

**Why F1 and not accuracy.** Most of the articles are mainstream. A lazy "model" that always answers "mainstream" is right about three times in four, which sounds decent and finds no conspiracy article at all. That is the baseline row of the table: its F1 is zero. Accuracy rewards it, F1 does not.

**A worked case.** The test set has 96 conspiracy articles and 314 mainstream ones. The LLM caught 64 of the 96 (recall 0.667) and wrongly flagged 1 mainstream article, so 64 of its 65 alarms were right (precision 0.985); its F1 is 0.795. The intervals treat every article as independent, but articles from one website resemble each other, so they are optimistic, that is, too narrow.

**Why McNemar for the comparisons.** Two models were scored on the same articles, so we compare them article by article. Where both are right, or both wrong, they tell us nothing. Only the articles where exactly one model is right count. If one model wins almost all of those, the difference is clear; if the wins split evenly, it may be luck.

**Why validation scores are higher.** Models were tuned on validation articles and then scored on test articles from other websites. The per-website table shows where the test scores are lost: one large site in the test set is hard for every model.

**Model errors and label problems.** The label belongs to the website, not to the article. A conspiracy site can print an ordinary news item. If a model "misses" it, the label describes the text badly. To tell the two apart, a second language model read each disputed article and ticked what it does with conspiracy claims: endorses, debunks, reports neutrally, or makes no claim. A conspiracy-site article without an endorsed claim, or a mainstream article that endorses one, counts as a label problem; everything else is a model error. The share rests on this automatic coding only; the team's hand check of 40 articles follows.

## The code

Each model gets one row of counts, the three scores and bootstrap intervals. The baseline "always guess mainstream" is one more row.

<!-- cell: src/fml/comparison.py#model_row -->

The McNemar test counts who is right where the other is wrong.

<!-- cell: src/fml/comparison.py#mcnemar_pair -->

The stage reads the saved test predictions, builds the table and the figures, and runs the three tests.

<!-- cell: src/fml/pipeline.py#align_predictions -->

<!-- cell: src/fml/pipeline.py#run_mcnemar_pairs -->

<!-- cell: src/fml/pipeline.py#comparison_stage -->

<!-- output: results/tables/comparison_compact.csv | Precision, recall and F1 of the conspiracy class on the test articles, with 95% confidence intervals -->
<!-- output: results/figures/comparison_confusion.png | Confusion matrix of each model on the test articles -->
<!-- output: results/metrics.json#comparison.baseline | The baseline that always guesses mainstream -->
<!-- output: results/metrics.json#comparison.mcnemar.words_vs_tfidf | McNemar test: word counts against TF-IDF -->
<!-- output: results/metrics.json#comparison.mcnemar.tfidf_vs_features | McNemar test: TF-IDF against TF-IDF with features -->
<!-- output: results/metrics.json#comparison.mcnemar.best_lr_vs_llm | McNemar test: best logistic regression against the LLM -->

To see where the test scores are lost, this cell counts, for every conspiracy website in the test set, how many of its articles each model found.

<!-- cell: src/fml/pipeline.py#conspiracy_website_recall -->

<!-- output: results/tables/test_website_recall.csv | Recall of each model on the conspiracy articles of each test website -->

Reading the table, finding by finding. The site with one article cannot tell us anything. The large site neonnettle.com supplies 44 of the 96 conspiracy test articles, and every model finds only about a third of them or fewer, so this one site pulls every test score down. On the other sites the models find most articles, so the drop from validation to test is mostly the work of this site and of the change of websites. The LLM finds more articles than every logistic regression on each site with more than one article.

Now the errors. First we find, for each test article, who is wrong.

<!-- cell: src/fml/errors.py#disagreement_sets -->

The stance coding sends each article to the second model once and stores only the id and the answer, never the text.

<!-- cell: src/fml/errors.py#code_stances -->

The rule turns the label and the stance into a verdict.

<!-- cell: src/fml/errors.py#classify_error -->

The examples are picked with a seeded shuffle among articles that have a clean excerpt.

<!-- cell: src/fml/errors.py#pick_examples -->

The stage puts it all together and writes the tables.

<!-- cell: src/fml/pipeline.py#errors_stage -->

<!-- output: results/tables/error_classification.csv -->
<!-- output: results/tables/error_examples_plain.csv -->
<!-- output: results/metrics.json#errors.label_problem_share -->

## Report

All four final models were scored once on the same 410 test articles (96 from conspiracy-oriented websites). The table gives precision, recall and F1 for the conspiracy class with 95% bootstrap confidence intervals (2,000 resamples of documents). The baseline row always answers "mainstream"; its accuracy is 0.766, which shows why accuracy is not used: it rewards a rule that finds no conspiracy article.

<!-- output: results/tables/comparison_compact.csv | Precision, recall and F1 of the conspiracy class on the test articles, with 95% confidence intervals -->

<!-- output: results/figures/comparison_confusion.png | Confusion matrices on the test set (rows: true website type, columns: predicted) -->

The LLM has the highest F1 (0.795, interval 0.723 to 0.857), followed by TF-IDF (0.598), word counts (0.567) and TF-IDF with features (0.560). The three logistic regressions have overlapping intervals and find about one half of the conspiracy articles (recall 0.490 to 0.531). The LLM is almost never wrong when it says "conspiracy" (precision 0.985, one false alarm among 314 mainstream articles) but still misses a third of the conspiracy articles (recall 0.667).

Three exact two-sided McNemar tests compare pairs of models on the same articles. Word counts against TF-IDF: 7 articles are right only for word counts and 19 only for TF-IDF (p = 0.029). TF-IDF against TF-IDF with features: 22 against 14 (p = 0.243). The best logistic regression on validation (TF-IDF) against the LLM: 4 against 37 (p < 0.0001).

**H1: TF-IDF weighting beats plain word counts.** Confirmed, narrowly. TF-IDF has the higher F1 (0.598 against 0.567), mainly through fewer false alarms (precision 0.721 against 0.607), and the McNemar test is significant (p = 0.029), although the confidence intervals overlap.

**H2: Adding the three feature groups improves the TF-IDF model.** Not confirmed. On validation the model with the extra features scored an F1 of 0.896 against 0.918 for TF-IDF alone, and on test it is lower again (0.560 against 0.598); the difference is within chance (p = 0.243).

**H3: The LLM differs from the best logistic regression.** Confirmed: the LLM is better. F1 is 0.795 against 0.598, the intervals do not overlap, and the test is clearly significant (p < 0.0001). The gain comes from precision (0.985 against 0.721) and from recall (0.667 against 0.510).

"Not confirmed" means that the data do not show an effect, not that there is none. The intervals resample documents, but websites are the real unit of dependence, so they are optimistic.

**Why the scores are lower on test than on validation.** Validation F1 was 0.865 for word counts, 0.918 for TF-IDF, 0.896 for TF-IDF with features and 0.987 for the LLM. The table gives recall per conspiracy website on the test set.

<!-- output: results/tables/test_website_recall.csv | Recall of each model on the conspiracy articles of each test website -->

The test set holds six conspiracy websites, and one of them, neonnettle.com, supplies 44 of the 96 conspiracy articles. All four models find at most about a third of its articles (recall 0.204 to 0.318), while on the other sites the logistic regressions find 0.579 to 1.000 of them. One large, hard site therefore lowers every test score, and since no website appears in more than one split, the test websites are new to every model.

**Where the models fail, and why.** The labels describe websites, not articles. We examined the test articles on which the best logistic regression or the LLM disagrees with the label: missed only by the logistic regression, only by the LLM, or by both. A second model (Claude Sonnet 5.5) coded the stance of each towards conspiracy claims about 9/11, without being told the website. A rule fixed in advance counts a conspiracy-site article that does not endorse a claim, or a mainstream article that endorses one, as a label problem; all other cases are model errors.

<!-- output: results/tables/error_classification.csv -->

Of the 69 articles with a readable coding, 49 are label problems and 20 are model errors. Almost all label problems are conspiracy-site articles that make no claim about 9/11 at all, such as ordinary news, so a model that reads only the text has little to find. Mainstream articles called conspiracy are mostly model errors. The 29 articles missed by both models are all conspiracy-site articles. Four examples show these cases.

<!-- output: results/tables/error_examples_plain.csv -->

The label-problem share rests on the automatic coding alone; the team's hand check of 40 articles follows.

## Check yourself
Q: A model that always answers "mainstream" is right most of the time. Why do we still not use accuracy?
A: Because most articles are mainstream, the lazy rule scores high accuracy while finding no conspiracy article. F1 for the conspiracy class is zero for it, so it shows the failure.

Q: Why can the LLM have a precision of 0.985 but a recall of only 0.667?
A: It almost never calls a mainstream article conspiracy (1 false alarm), so what it flags is nearly always right. But it misses 32 of the 96 conspiracy articles, so it finds only two thirds of them.

Q: Why does McNemar's test look only at articles where one model is right and the other wrong?
A: Both models saw the same articles. Where they agree, there is nothing to compare, so the evidence about a difference sits only in the disagreements.

Q: Why are scores on validation higher than on test?
A: The test conspiracy articles come from other websites, and one large site is hard for every model. That site pulls the test scores down.

Q: Why is a conspiracy-site article with no conspiracy claim a label problem and not a model error?
A: Because our rule says so: the text does not look like conspiracy content, so the website label describes it badly. This is a definition we chose, not a finding.
