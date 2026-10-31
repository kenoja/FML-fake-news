---
title: Results and why they fail
part: Evaluation
hypothesis: The LLM beats the three trained models on unseen websites, and many errors are label problems rather than model mistakes.
---
## Explained simply

**How to read the table.** Each final model takes the exam once, on unseen test articles. The table shows three scores for the conspiracy class: **precision** (when it says "conspiracy", how often is it right?), **recall** (of all real conspiracy articles, how many did it find?) and **F1**, one number that is low if either of the other two is low. Each score has a **confidence interval**. Like a poll's "give or take 3 points", it shows how far a score may be off.

**Why F1 and not accuracy.** Most articles are mainstream. A lazy "model" that always answers "mainstream" is right three times in four yet finds no conspiracy article: the baseline row, with F1 zero.

**A worked case.** The test set has 96 conspiracy articles and 314 mainstream ones. The LLM caught 64 of the 96 (recall 0.667) and wrongly flagged 1 mainstream article, so 64 of its 65 alarms were right (precision 0.985); its F1 is 0.795. The intervals are optimistic: articles from one website are alike.

**Why McNemar.** Two judges rate the same cases: only disagreements show who is better. Right in 20 of 22 disagreements: better. 11 against 11: maybe luck. McNemar's test counts exactly these cases.

**Why validation scores are higher.** Models were tuned on validation articles, then scored on test articles from other websites. The per-website table shows the loss: one large test site is hard for every model.

**Model errors and label problems.** The label belongs to the website, not to the article. A conspiracy site can print an ordinary news item; then the label fits the text badly. To tell the two apart, a second language model read each disputed article and recorded its **stance**, that is, what the article does with conspiracy claims: endorses them, debunks them, reports them neutrally, or makes no such claim. The rule has three steps. (1) Look at the website label. (2) Look at the stance. (3) If the label does not fit the stance, it is a label problem; otherwise the model erred. Example (invented): a conspiracy site prints "A judge set the trial date for five men." The stance is "reports neutrally", which does not fit "conspiracy site", so a model that calls it mainstream is not really wrong. The main error type is this: a conspiracy-site article called mainstream. Because a machine did the ticking, we checked 40 other articles by hand, blind to the website and the machine's answer. Two members picked the same stance in 22 of 56 pairs (2 of 5 answers); on the one question that decides the verdict, does the article endorse a claim, they agreed in 43 of 56 pairs (3 of 4).

## The code

Each model gets one row of counts, the three scores and bootstrap intervals. The baseline "always guess mainstream" is one more row.

<!-- cell: src/fml/comparison.py#model_row -->

The McNemar test counts who is right where the other is wrong.

<!-- cell: src/fml/comparison.py#mcnemar_pair -->

The pipeline applies these two functions to the saved test predictions of all four models.

<!-- output: results/tables/comparison_compact.csv | Precision, recall and F1 of the conspiracy class on the test articles, with 95% confidence intervals -->
<!-- output: results/figures/comparison_confusion.png | Confusion matrix of each model on the test articles -->
<!-- output: results/tables/mcnemar_tests.csv | The three McNemar tests: articles where only one model is right, and the p-value -->
<!-- output: results/figures/validation_vs_test_confusion.png | Confusion matrices of the best logistic regression and the LLM on validation (top) and test (bottom) -->
<!-- output: results/tables/validation_vs_test.csv | F1 on validation, on test, and on test without the largest conspiracy website -->
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

<!-- output: results/tables/error_classification.csv -->
<!-- output: results/tables/error_counts_plain.csv -->

<!-- output: results/tables/error_examples_plain.csv | Examples where the models disagree -->

Each row is one test article with a short excerpt. The first two rows show that the two models make different mistakes: each is sometimes right where the other is wrong. The rows where both models are wrong, or where the label does not fit the text, show articles that carry little or no conspiracy claim, so the website label describes them badly.
<!-- output: results/metrics.json#errors.label_problem_share -->

Last, the team's hand check. The answer files of all members are read together; a member who sends a second file keeps their first, blind answer.

<!-- cell: src/fml/errors.py#read_hand_check -->

Every pair of members who read the same article is compared, each pair once.

<!-- cell: src/fml/errors.py#member_pairs -->

The rule only asks "does the article endorse a claim?", so agreement is counted twice: on the exact stance, and on that one question.

<!-- cell: src/fml/errors.py#agreement_row -->
<!-- output: results/tables/hand_check_agreement.csv | How often two members, or a member and the automatic coder, gave the same answer -->

Finally the rule is applied to each member answer and to the coder's answer for the same article.

<!-- cell: src/fml/errors.py#verdicts_by_site_type -->
<!-- output: results/tables/hand_check_verdicts.csv | Label problems found with the members' answers and with the coder's answers, per kind of website -->
<!-- output: results/metrics.json#errors.human_check | How complete the hand check is -->

## Report

### Test scores

All four final models were scored once on the same 410 test articles (96 conspiracy) from websites that no model had seen. The 95% bootstrap intervals (Efron & Tibshirani, 1993) are too narrow for the reason given under Limitations. Always answering "mainstream" reaches an accuracy of 0.766 and finds no conspiracy article.

<!-- output: results/tables/comparison_compact.csv | Precision, recall and F1 of the conspiracy class on the test articles, with 95% confidence intervals -->

The LLM leads clearly. All four models mainly miss conspiracy articles: the logistic regressions find 51, 49 and 47 of the 96, the LLM 64. They differ in false alarms: 33 for word counts, 19 for TF-IDF, 25 for TF-IDF + features and 1 for the LLM, whose precision is 0.985.

<!-- output: results/figures/comparison_confusion.png | Confusion matrices on the test set (rows: true website type, columns: predicted) -->

### Hypotheses

Each pair is compared with an exact two-sided McNemar test (McNemar, 1947) on the articles where only one of the two models is right.

<!-- output: results/tables/mcnemar_tests.csv | The three McNemar tests: articles where only one model is right, and the p-value -->

**H1: TF-IDF beats word counts.** Supported, narrowly: 7 articles are right only for word counts and 19 only for TF-IDF (p = 0.029), and TF-IDF has the higher F1 (0.598 against 0.567), mainly through fewer false alarms.

**H2: The three feature groups improve TF-IDF.** Not supported: 22 articles are right only for TF-IDF and 14 only with the features (p = 0.243), and F1 is lower with the features (0.560 against 0.598), as on validation.

**H3: The LLM beats the best logistic regression.** Supported: 4 articles are right only for TF-IDF and 37 only for the LLM (p < 0.001); F1 is 0.795 against 0.598 and the intervals do not overlap.

### Why test scores are lower: Website 4

Every model scores far lower on test than on validation:

<!-- output: results/figures/f1_validation_test.png | F1 on validation, on test, and on test without Website 4 -->

The main cause is one website. Website 4 supplies 44 of the 96 conspiracy test articles, and all four models find at most a third of them (recall 0.205 to 0.318), while the LLM finds at least 0.947 of the articles on every other test website with more than one article.

<!-- output: results/tables/test_website_recall.csv | Recall of each model on the conspiracy articles of each test website (Website 4 holds 44 of the 96) -->

Without Website 4 the F1 is 0.640 for word counts, 0.721 for TF-IDF, 0.661 for TF-IDF + features and 0.971 for the LLM. For the LLM, Website 4 explains almost the whole drop (0.971 against 0.987 on validation). For the logistic regressions it is the largest single cause, but they stay well below their validation scores, which fits their partial reliance on how the training websites write. Judging by its titles, Website 4 publishes short partisan political news that mentions 9/11; among its 30 articles that the LLM missed, the stance coder found a conspiracy claim in none (27 make no claim, 2 report claims neutrally, 1 rejects them). The corpus authors' own measure agrees: they flag a conspiracy article as typical when its wording is unusually close to the rest of the conspiracy corpus (Miani et al., 2022). None of Website 4's 44 articles is flagged, against 21.4% of the training conspiracy articles, and all four models found all 12 flagged test articles. Its texts read as mainstream; only its website rating says conspiracy.

### Why the models fail: label problems

For every test article that the best logistic regression or the LLM gets wrong, a second model (Claude Sonnet 5.5; Anthropic, 2026d) coded its stance towards conspiracy claims without seeing the website. A fixed rule calls an error a label problem when label and stance do not fit (a conspiracy-site article without an endorsed claim, or a mainstream article with one); otherwise it is a model error. Of the 70 wrongly classified articles, 69 have a readable coding. In short, about half of the errors that only one model makes are label problems (21 of 41, our main estimate); the upper end is 49 of 69, because the 28 articles both models missed are conspiracy-site articles that the rule counts as label problems whenever no claim is endorsed.

<!-- output: results/tables/error_counts_plain.csv | Test articles wrongly classified, by who is wrong, and the share that are label problems -->

Two examples show the pattern. A conspiracy-site article that both models called mainstream begins "A military judge has set the trial date for five men accused of masterminding the 9/11 attacks": a plain court report, a label problem. A mainstream article opening "Not only is there a smoking gun in the 28 pages" was called conspiracy by the logistic regression and mainstream by the LLM; the stance coder found no conspiracy claim in it, a model error caused by wording that only sounds like conspiracy writing.

To check the coder, the team coded 40 validation articles blind, each read by at least two members. Members often chose different stances (22 of 56 pairs agree), mostly "reports neutrally" against "makes no claim", which the rule treats alike; on whether an article endorses a claim, the question that decides the verdict, two members agreed in 43 of 56 pairs and a member and the coder in 74 of 84. On conspiracy-site articles the coder found more label problems than the members (21 against 17 of 40), so the estimates above are, if anything, somewhat too high.

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

Q: In the hand check, one member says an article "reports neutrally" and another says it "makes no claim". Does that change the verdict?
A: No. The rule only asks whether the article endorses a conspiracy claim. Both answers say it does not, so both lead to the same verdict; that is why agreement is also counted on this one question.
