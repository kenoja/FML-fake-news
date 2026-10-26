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

Analysis 4 is different: a ready-made AI system (an **LLM**) that has never seen our training articles. It reads each article and answers. It is not trained by us.

**Why these four.** They form a ladder from simple to rich. Word counts are the plainest baseline, TF-IDF is the standard small improvement, the extra features test our own ideas about how conspiracy writing sounds, and the LLM shows what a very different, untrained approach can do.

**What we did not do.** We did not try other classifiers, because comparing many methods would hide which *input* matters. We did not fine-tune an LLM, which needs more data and money than we have. We did not use emotion features, because fear and anger are normal in any report on an attack, so they say little about conspiracy.

**The four at a glance.** What each method reads, whether we train it, its settings, how long it runs and what it costs:

<!-- output: results/tables/methods_overview.csv | The four methods at a glance -->

**What is a hypothesis?** A clear guess that the results can prove wrong. We have three:
- **H1.** TF-IDF beats plain word counts, because telling words carry more information.
- **H2.** Adding the three feature groups (phrases, certainty words, style) improves TF-IDF.
- **H3.** The LLM, used without any training, beats the best logistic regression, because it has read a great deal of text.

**How we judge "better".** Most articles come from mainstream websites, so a model that always says "mainstream" is right most of the time and finds nothing. We use the F1 score for the conspiracy class instead. To decide whether one model really beats another, we look at the articles where exactly one of the two is right (the McNemar test). If that test gives p below 0.05 and the better model is the one we predicted, the hypothesis is supported.

**A warning.** The label belongs to the website, not the article: a calm news item on a conspiracy website counts as "conspiracy".

## The code

This section only sets the question. The first function builds the "at a glance" table from the files the pipeline writes. The second is the F1 score we use to judge every method.

<!-- cell: src/fml/methods_overview.py#methods_table -->
<!-- output: results/tables/methods_overview.csv -->
<!-- cell: src/fml/evaluate.py#f1_score -->

## Report

### Executive summary

We asked whether the text of a full 9/11 article reveals if it comes from a conspiracy-oriented or a mainstream website, comparing three logistic regressions and one low-cost language model, Claude Haiku 5.5, on 410 held-out test articles. The language model reached the highest F1 score for the conspiracy class, 0.795 (95% confidence interval 0.723 to 0.857), against 0.567 for word counts, 0.598 for TF-IDF and 0.560 for TF-IDF with features. H1 (weighting words helps) was supported, H2 (hand-made features help) was not supported, and H3 (the language model beats the best logistic regression) was supported. All models found fewer than seven in ten conspiracy articles on the test websites, so none is reliable enough for automatic flagging.

### Introduction

The attacks of 11 September 2001 are among the most persistent subjects of conspiracy theories: claims that they were an "inside job", a "controlled demolition" or a "false flag" are actively argued for online (Wood & Douglas, 2013) and have been studied as a widely held belief (Swami, Chamorro-Premuzic & Furnham, 2010). In a 2006 survey, 36 percent of Americans found it credible that federal officials helped the attacks happen or deliberately failed to stop them (Stempel, Hargrove & Stempel, 2007). Automatic detection of misleading web content is an active research field (Shu et al., 2017). This study asks whether the text of a full 9/11 article reveals if it was published on a conspiracy-oriented or a mainstream website, using machine-learning methods for text classification.

We compare four methods that form a ladder from simple to rich; the expectation for each is our hypothesis.

1. **Word counts with logistic regression**, the plainest baseline. We expect it to beat always guessing "mainstream", because the two kinds of website use different words.
2. **TF-IDF with logistic regression**, the standard step beyond counting (Salton & Buckley, 1988; Manning, Raghavan & Schütze, 2008). **H1:** it reaches a higher conspiracy-class F1 than word counts, because distinctive words carry more information than common ones.
3. **TF-IDF plus three feature groups** (word sequences, certainty words, style). **H2:** it reaches a higher F1 than TF-IDF alone, because these features capture how conspiracy texts argue.
4. **A low-cost LLM used without training** (zero-shot). **H3:** it reaches a higher F1 than the best logistic regression, because it brings broad knowledge of how texts are written.

We used logistic regression for every trained model because it is simple, well understood and its weights can be read as word lists (Hastie, Tibshirani & Friedman, 2009). We tried no other classifiers, because the question is about the input, not about the best algorithm. We did not fine-tune a language model, which needs more data and money than a small study has, and we left out emotion features, because fear and anger are normal in any report on an attack and so say little about conspiracy content. The whole pipeline, without the language model, runs on an ordinary computer (two to four processor cores, Python 3.11) in about two minutes, about one of which is the feature model; only the language model needs an internet connection and a paid service.

All methods are judged by precision, recall and F1 for the conspiracy class, because most articles are mainstream and the share of correct answers would reward a model that never finds a conspiracy article. Two models are compared with McNemar's test, which uses only the articles on which exactly one of the two is right; it is the recommended test for two classifiers scored on the same test set (Dietterich, 1998). Separate confidence intervals can overlap even when such a paired difference is real, so they describe uncertainty but do not decide. A hypothesis is supported if the difference points in the predicted direction and McNemar's test gives p < 0.05, and not supported otherwise.

The four methods at a glance:

<!-- output: results/tables/methods_overview.csv | The four methods at a glance -->

## Check yourself

Q: Which of the four methods are trained on our articles, and which is not?
A: The three logistic regressions learn from our training articles; they differ only in their input. The LLM is used as it comes, with no training on our data.

Q: Why do we use only one kind of classifier for analyses 1 to 3?
A: If the classifier stayed the same and only the input changed, any difference in results can be traced to the input.

Q: Why can a model with high accuracy still be useless here?
A: Most articles are mainstream, so always answering "mainstream" is right most of the time while finding no conspiracy article.

Q: A calm news item sits on a conspiracy website. Why is it still labelled "conspiracy"?
A: We did not read and label each article; we labelled the website. So a calm article inherits its site's label, and a model that calls it "mainstream" may be reading the text correctly. This is why some "errors" are really label problems.
