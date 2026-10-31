---
title: Question and hypotheses
part: Introduction
hypothesis: Richer inputs help the logistic regression, and the LLM beats all three.
---
## Explained simply

**The question.** Can a computer tell from the text of a 9/11 article whether it was published on a conspiracy-oriented website or on a mainstream website?

**The four ways.** All four see the same articles, cut to the same length. Analyses 1 to 3 are all **logistic regression**, a method that learns from examples which words go with which kind of website. They differ only in what we feed it:
1. **Word counts**: how often each word appears.
2. **TF-IDF**: the same, but rare and telling words count more than everyday words.
3. **TF-IDF plus extra features**: we add phrases of two or three words, words of certainty or doubt, and style clues such as capital letters and exclamation marks.

Analysis 4 is a ready-made AI system (an **LLM**) that has never seen our training articles.

**Why these four.** They form a ladder from simple to rich, ending with a very different, untrained approach.

**What we did not do.** We did not try other classifiers, because comparing many methods would hide which *input* matters. We did not fine-tune an LLM; that needs more data and money. We did not use emotion features: fear and anger are normal in any report on an attack.

**The four at a glance:**

<!-- output: results/tables/methods_overview.csv | The four methods at a glance -->

**Hypotheses** are guesses the results can prove wrong. We have three:
- **H1.** TF-IDF beats plain word counts, because telling words carry more information.
- **H2.** Adding the three feature groups (phrases, certainty words, style) improves TF-IDF.
- **H3.** The LLM, used without any training, beats the best logistic regression, because it has read a great deal of text.

**How we judge "better".** Most articles are mainstream, so always saying "mainstream" is right most of the time. We use the F1 score instead. To decide whether one model really beats another, we use the McNemar test. F1 looks at the conspiracy class only. The McNemar test counts articles of both classes that exactly one of the two models gets right. A hypothesis is supported only when both point the way we predicted: the F1 is higher and p is below 0.05. We ran three tests without correcting for that; only H3 would survive the stricter limit of 0.05 divided by three.

**What we changed from the plan.** The proposal planned headlines, a test on another event (the JFK assassination), several LLMs and perhaps a transformer. Following the course feedback we use full articles and one inexpensive LLM; the JFK test is left for later work.

**A warning.** The label belongs to the website: a calm news item on a conspiracy website counts as "conspiracy".

## The code

This section only sets the question. The first function builds the "at a glance" table from the files the pipeline writes. The second is the F1 score we use to judge every method.

<!-- cell: src/fml/methods_overview.py#methods_table -->
<!-- output: results/tables/methods_overview.csv -->
<!-- cell: src/fml/evaluate.py#f1_score -->

## Report

### Executive summary

We asked whether the text of a full 9/11 article reveals whether it was published on a conspiracy-oriented or a mainstream website. Four methods were compared in a demanding but fair test: every website appears in only one of the training, validation and test sets. On validation all four looked excellent (F1 from 0.865 to 0.987). On the 410 test articles from unseen websites all four dropped sharply, above all because of one website, Website 4, which supplies 44 of the 96 conspiracy test articles and whose texts read like ordinary political news. The language model Claude Haiku 5.5 stayed best (F1 0.795 against 0.598 for TF-IDF): weighting words helped narrowly (H1 supported), hand-made features did not help (H2 not supported) and the language model won (H3 supported). The main lesson concerns the data: with website labels and few conspiracy websites, an honest test reaches the limits of the data set before those of any method.

### Introduction

The attacks of 11 September 2001 are a persistent subject of conspiracy theories: claims that they were an "inside job", a "controlled demolition" or a "false flag" are actively argued for online (Wood & Douglas, 2013). In a 2006 survey, 36 percent of Americans found it credible that federal officials helped the attacks happen or deliberately failed to stop them (Stempel et al., 2007). This study asks whether the text of a full 9/11 article reveals if it was published on a conspiracy-oriented or a mainstream website, using machine-learning methods for text classification.

We compare four methods that form a ladder from simple to rich: logistic regression on word counts, the plainest baseline; logistic regression on TF-IDF weights, the standard step beyond counting (Manning et al., 2008; Salton & Buckley, 1988); logistic regression on TF-IDF plus three groups of hand-made features; and a low-cost LLM used without training (zero-shot). The four method chapters follow the same pattern: the idea and what we expected, the setup, the result on validation, and what the model learned. We test three hypotheses:

- **H1.** TF-IDF reaches a higher F1 than word counts, because distinctive words carry more information than common ones.
- **H2.** TF-IDF plus the three feature groups reaches a higher F1 than TF-IDF alone, because these features capture how conspiracy texts argue.
- **H3.** The LLM reaches a higher F1 than the best logistic regression, because it brings broad knowledge of how texts are written.

Compared with the original proposal, we classify full articles instead of headlines, because a headline may mention a conspiracy without showing whether the article supports it, and we use one inexpensive language model without fine-tuning, as the course feedback advised. All methods read the same first 3,000 words of each article, and everything except the language model runs on an ordinary computer in about four minutes.

What we left out, and why: other classifiers, because the question is about the input; emotion features, because fear and anger are normal in any report on an attack; and a transfer test on other topics, to keep one topic.

<!-- output: results/tables/methods_overview.csv | The four methods at a glance -->

### How we judge the methods

Most articles are mainstream, so accuracy would reward a model that never finds a conspiracy article. All methods are therefore judged by precision (how many flagged articles are conspiracy), recall (how many conspiracy articles are found) and their combination F1, all for the conspiracy class. Two models are compared with McNemar's test, which counts the articles that exactly one of them gets right, from both classes (Dietterich, 1998). A hypothesis is supported only when both point the predicted way: F1 is higher and p < 0.05. The three tests are not corrected for multiple testing; with a correction for three tests (0.05/3) only H3 would still pass.

## Check yourself

Q: Which of the four methods are trained on our articles, and which is not?
A: The three logistic regressions learn from our training articles; they differ only in their input. The LLM is used as it comes, with no training on our data.

Q: Why do we use only one kind of classifier for analyses 1 to 3?
A: If the classifier stayed the same and only the input changed, any difference in results can be traced to the input.

Q: Why can a model with high accuracy still be useless here?
A: Most articles are mainstream, so always answering "mainstream" is right most of the time while finding no conspiracy article.

Q: A calm news item sits on a conspiracy website. Why is it still labelled "conspiracy"?
A: We did not read and label each article; we labelled the website. So a calm article inherits its site's label, and a model that calls it "mainstream" may be reading the text correctly. This is why some "errors" are really label problems.

Q: What makes "TF-IDF reaches a higher F1 than word counts" a testable hypothesis, and "TF-IDF is better" not?
A: The first names a measurable score and a direction, so the results can show it to be wrong. "Better" says neither what is measured nor how we would know.
