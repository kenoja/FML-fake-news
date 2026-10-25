---
title: Question and hypotheses
part: Introduction
hypothesis: The zero-shot LLM reaches a higher conspiracy-class F1 on the test set than the best logistic regression, and the McNemar test gives p < 0.05.
---
## Explained simply

**The question.** Can a computer tell from the text of a 9/11 article whether it was published on a conspiracy-oriented website or on a mainstream website?

**The four ways.** All four see the same articles, cut to the same length. Analyses 1 to 3 are all **logistic regression**, a method that learns from examples which words go with which kind of website. They differ only in what we feed it:
1. **Word counts**: how often each word appears.
2. **TF-IDF**: the same, but rare and telling words count more than everyday words.
3. **TF-IDF plus extra features**: we add phrases of two or three words, words of certainty or doubt, and style clues such as capital letters and exclamation marks.

Analysis 4 is different: a ready-made AI system (an **LLM**) that has never seen our training articles. It reads each article and answers. It is not trained by us.

**Why these four.** They form a ladder from simple to rich. Word counts are the plainest baseline, TF-IDF is the standard small improvement, the extra features test our own ideas about how conspiracy writing sounds, and the LLM shows what a very different, untrained approach can do.

**What we did not do.** We did not try other classifiers, because comparing many methods would hide which *input* matters. We did not fine-tune an LLM, which needs more data and money than we have. We did not use emotion features, which simple word lists measure too unreliably.

**What is a hypothesis?** A clear guess that the results can prove wrong. We have three:
- **H1.** TF-IDF beats plain word counts, because telling words carry more information.
- **H2.** Adding the three feature groups (phrases, certainty words, style) improves TF-IDF.
- **H3.** The zero-shot LLM reaches a higher conspiracy-class F1 on the test set than the best logistic regression, and the McNemar test gives p < 0.05.

**How we judge "better".** Most articles come from mainstream websites, so a model that always says "mainstream" is right most of the time and finds nothing. We use the F1 score for the conspiracy class instead.

**A warning about the label.** The label belongs to the website, not to the single article. A calm news item on a conspiracy website still counts as "conspiracy".

## The code

This section only sets the question. The first function builds the "at a glance" table from the files the pipeline writes. The second is the F1 score we use to judge every method.

<!-- cell: src/fml/methods_overview.py#methods_table -->
<!-- output: results/tables/methods_overview.csv -->
<!-- cell: src/fml/evaluate.py#f1_score -->

## Report

### Executive summary

We asked whether the text of a full 9/11 article reveals if it was published on a conspiracy-oriented or a mainstream website. Three logistic regressions (on word counts, on TF-IDF weights, and on TF-IDF plus three hand-made feature groups) and one low-cost language model, Claude Haiku 5.5, were compared on 410 held-out test articles. The language model reached the highest F1 score for the conspiracy class, 0.795 (95% confidence interval 0.723 to 0.857), against 0.567 for word counts, 0.598 for TF-IDF and 0.560 for TF-IDF with features. H1 (weighting words helps) was weakly supported, H2 (hand-made features help) was not confirmed, and H3 (the language model beats the best logistic regression) was confirmed. All models found fewer than seven in ten conspiracy articles on the test websites, so none is reliable enough for automatic flagging.

### Introduction

The 9/11 attacks are a lasting subject of online conspiracy content, (Miani, Hills & Bangerter, 2022). This study asks whether the text of a full article about the 9/11 attacks reveals if it was published on a conspiracy-oriented or on a mainstream website, using machine-learning methods for text classification. The articles come from the LOCO corpus, and the label describes the website, not the single article.

We compare four methods. Analyses 1 to 3 are logistic regression on different inputs: word counts, TF-IDF weights, and TF-IDF weights plus three hand-made feature groups (word sequences of up to three words, certainty and hedging words, and style measures). Analysis 4 is one low-cost large language model (LLM) used without training on our data. One simple classifier with changing inputs shows which input matters; other classifiers, fine-tuning and emotion features were left out.

Table 1 gives the four methods at a glance.
<!-- output: results/tables/methods_overview.csv -->

We test three hypotheses. **H1:** TF-IDF weights give a higher conspiracy-class F1 than plain word counts, because distinctive words carry more information than common ones. **H2:** adding the three feature groups (word sequences, certainty words, style) to TF-IDF raises the F1, because they capture how conspiracy texts argue. **H3:** The zero-shot LLM reaches a higher conspiracy-class F1 on the test set than the best logistic regression, and the McNemar test gives p < 0.05.
All methods are judged by precision, recall and F1 for the conspiracy class, because most articles are mainstream and the share of correct answers would reward a model that never finds a conspiracy article. A hypothesis is called confirmed only if the difference points in the predicted direction, the McNemar test gives p < 0.05 and the 95% confidence intervals do not overlap. If the test is significant but the intervals overlap, it is called weakly supported; otherwise it is not confirmed.

## Check yourself

Q: Which of the four methods are trained on our articles, and which is not?
A: The three logistic regressions learn from our training articles; they differ only in their input. The LLM is used as it comes, with no training on our data.

Q: Why do we use only one kind of classifier for analyses 1 to 3?
A: If the classifier stayed the same and only the input changed, any difference in results can be traced to the input.

Q: Why can a model with high accuracy still be useless here?
A: Most articles are mainstream, so always answering "mainstream" is right most of the time while finding no conspiracy article.

Q: A calm news item sits on a conspiracy website. Why is it still labelled "conspiracy"?
A: We did not read and label each article; we labelled the website. So a calm article inherits its site's label, and a model that calls it "mainstream" may be reading the text correctly. This is why some "errors" are really label problems.
