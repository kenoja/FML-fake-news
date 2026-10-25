---
title: Question and hypotheses
part: Introduction
hypothesis: Weighting words helps, our three feature groups help, and the language model differs from the best model we trained.
---
## Explained simply

**The question.** Can a computer tell from the text of a 9/11 article whether it was published on a conspiracy-oriented website or on a mainstream website?

**The four ways.** All four see the same articles, cut to the same length. Analyses 1 to 3 are all **logistic regression**, a method that learns from examples which words go with which kind of website. They differ only in what we feed it:
1. **Word counts**: how often each word appears.
2. **TF-IDF**: the same, but rare and telling words count more than everyday words.
3. **TF-IDF plus extra features**: we add phrases of two or three words, words of certainty or doubt, and style clues such as capital letters and exclamation marks.

Analysis 4 is different: a ready-made AI system (an **LLM**) that has never seen our training articles. It reads each article and answers. It is not trained by us.

**Why these four.** They form a ladder from simple to rich. Word counts are the plainest baseline. TF-IDF is the standard small improvement. The extra features test our own ideas about how conspiracy writing sounds. The LLM shows what a very different, untrained approach can do.

**What we did not do, and why.** We did not try other classifiers (such as support vector machines or neural networks), because comparing many methods would hide which *input* matters; one simple method with different inputs answers our question. We did not fine-tune an LLM, because that needs far more data, money and care than this course project allows. We did not use emotion features, because the team judged them too unreliable to measure with simple word lists.

**The four methods at a glance.**

<!-- output: results/tables/methods_overview.csv -->

Everything except the LLM runs on an ordinary laptop in about a minute or two. Only the LLM needs an internet connection and a paid service.

**What is a hypothesis?** A clear guess that the results can prove wrong. We have three:
- **H1.** TF-IDF beats plain word counts, because telling words carry more information than everyday ones.
- **H2.** Adding the three feature groups (phrases, certainty words, style) improves TF-IDF, because they capture how conspiracy texts argue.
- **H3.** The LLM does about as well as the best logistic regression, or differs from it; we have no firm direction.

**How we judge "better".** Most articles come from mainstream websites, so a model that always says "mainstream" is right most of the time and finds nothing. We use the F1 score for the conspiracy class instead.

**A warning about the label.** The label belongs to the website, not to the single article. A calm news item on a conspiracy website still counts as "conspiracy".

## The code

This section only sets the question. The first function builds the "at a glance" table from the files the pipeline writes. The second is the F1 score we use to judge every method.

<!-- cell: src/fml/methods_overview.py#methods_table -->
<!-- output: results/tables/methods_overview.csv -->
<!-- cell: src/fml/evaluate.py#f1_score -->

## Report

### Executive summary

We asked whether the text of a full 9/11 article reveals if it was published on a conspiracy-oriented or a mainstream website. Three logistic regressions (on word counts, on TF-IDF weights, and on TF-IDF plus three hand-made feature groups) and one low-cost language model, Claude Haiku 5.5, were compared on 410 held-out test articles. The language model reached the highest F1 score for the conspiracy class, 0.795 (95% confidence interval 0.723 to 0.857), against 0.567 for word counts, 0.598 for TF-IDF and 0.560 for TF-IDF with features. Weighting words helped slightly (H1), the hand-made features did not (H2), and the language model clearly differed from the best logistic regression (H3). All models found fewer than seven in ten conspiracy articles on the test websites, so none is reliable enough for automatic flagging.

### Introduction

The 9/11 attacks are a lasting subject of online conspiracy content, and conspiracy-oriented websites publish far more text than can be checked by hand (Miani, Hills & Bangerter, 2022). Earlier work has shown that conspiracy language differs measurably from ordinary language (Fong et al., 2021). This study asks whether the text of a full article about the 9/11 attacks reveals if it was published on a conspiracy-oriented or on a mainstream website, using machine-learning methods for text classification. The articles come from the LOCO corpus, and the label describes the website, not the single article.

We compare four methods. Analyses 1 to 3 are logistic regression on different inputs: word counts, TF-IDF weights, and TF-IDF weights plus three hand-made feature groups (word sequences of up to three words, certainty and hedging words, and style measures). Analysis 4 is one low-cost large language model (LLM) used without any training on our data. A single simple classifier with changing inputs shows which input matters. Other classifiers, fine-tuning an LLM and emotion features were left out for reasons of scope and reliability.

Table 1 gives the four methods at a glance.

<!-- output: results/tables/methods_overview.csv -->

All three logistic regressions ran on a laptop-class machine (see the run information in the results folder; 4 CPUs, no graphics card) in about a minute each. Only the LLM needs an internet connection and a paid interface; its cost is shown in the table.

We test three hypotheses. **H1:** TF-IDF weights give a higher conspiracy-class F1 than plain word counts, because distinctive words carry more information than common ones. **H2:** adding the three feature groups (word sequences, certainty words, style) to TF-IDF raises the F1, because they capture how conspiracy texts argue. **H3:** the LLM performs about as well as the best logistic regression, or differs from it; no direction is assumed.
All methods are judged by precision, recall and F1 for the conspiracy class, because most articles are mainstream and the share of correct answers would reward a model that never finds a conspiracy article. A hypothesis counts as supported only if the confidence interval and the significance test agree with the direction of the difference.

## Check yourself

Q: Which of the four methods are trained on our articles, and which is not?
A: The three logistic regressions learn from our training articles; they differ only in their input. The LLM is used as it comes, with no training on our data.

Q: Why do we use only one kind of classifier for analyses 1 to 3?
A: If the classifier stayed the same and only the input changed, any difference in results can be traced to the input.

Q: Why can a model with high accuracy still be useless here?
A: Most articles are mainstream, so always answering "mainstream" is right most of the time while finding no conspiracy article.

Q: A calm news item sits on a conspiracy website. Why is it still labelled "conspiracy"?
A: The label describes the website that published the article, not the content of the single article.
