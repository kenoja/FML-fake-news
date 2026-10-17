---
title: Question and hypotheses
part: Introduction
hypothesis: Weighting words helps, our own features add little, and the language model beats the models we trained.
---
## Explained simply

**The question.** Can a computer tell from the text of a 9/11 article whether it was published on a conspiracy-oriented website or on a mainstream website, and which of four ways of doing this works best?

**Why this question.** Conspiracy websites publish too much to check by hand, so we need to know how well computer methods work and where they fail.

**The four ways.** All four see the same articles.
1. **Word counts.** The computer counts how often each word appears and learns which words go with which kind of website.
2. **TF-IDF.** The same idea, but rare and telling words count more than everyday words.
3. **TF-IDF plus extra features.** We add clues that we made ourselves: words that express certainty or doubt, and features of writing style such as capital letters and exclamation marks.
4. **A language model.** A ready-made AI system (an LLM) that has never seen our training articles. It reads each article and gives its answer.

The first three use a method called logistic regression, which we train ourselves. The fourth is used as it comes.

**What is a hypothesis?** A hypothesis is a guess that we write down before we look at the results, so that the results can prove it wrong. We have three. The results say for each one whether the data support it. If they do not clearly, we say so. That is also an answer.

- **H1.** Weighting words by how telling they are (TF-IDF) improves on plain word counts.
- **H2.** Adding features about certainty and writing style improves the TF-IDF model. Only the certainty words passed our validation threshold, so the final tested model contains certainty words only, not the style features.
- **H3.** The language model does better than the best of the three models we trained.

**Why these three?** They follow the order of the study: first the choice of how to represent words (H1), then whether our own ideas add anything (H2), then the comparison with a very different approach (H3). Each is answered in the results section and again in the conclusion.

**How we judge "better".** About seven in ten articles come from mainstream websites. A model that always says "mainstream" would be right 70% of the time and would find no conspiracy article at all. So we do not use the share of correct answers. We use the F1 score for the conspiracy class, which rewards finding conspiracy articles without raising false alarms. The glossary explains it.

**A warning about the label.** The label belongs to the website, not to the single article. A calm news item on a conspiracy website counts as "conspiracy".
## The code

This section only sets the question. The one piece of code that belongs here is the F1 score that we use to judge every method. It combines how many of the flagged articles are right with how many conspiracy articles we find.

<!-- cell: src/fml/evaluate.py#f1_score -->

The real analysis code starts in the next sections.

## Report

This study asks whether the text of a full article about the 9/11 attacks reveals if it was published on a conspiracy-oriented or on a mainstream website, and how four methods compare on this task. The methods are logistic regression on word counts, logistic regression on TF-IDF weights, logistic regression on TF-IDF weights plus hand-made features, and one low-cost large language model (LLM) used without any training on our data. The following hypotheses were formulated before the final evaluation.

H1. Weighting words by their distinctiveness (TF-IDF) improves on plain word counts.

H2. Adding features about certainty and writing style improves the TF-IDF model. The final tested model contains certainty words only, because only they passed the validation threshold (metrics.json, tfidf_features).

H3. The LLM does better than the best logistic regression.

All methods are judged by precision, recall and F1 score for the conspiracy class, because the classes are unbalanced: about seven in ten articles come from mainstream websites, so the share of correct answers would reward a model that never finds a conspiracy article. A hypothesis counts as supported only if the confidence interval and the significance test agree with the direction of the difference. Results are reported in the results section, and the conclusion returns to each hypothesis.

## Check yourself

Q: Why do we write down hypotheses before looking at the results?
A: So that the results can show a guess to be wrong. Guesses made after seeing the results would always seem to fit.

Q: What is the difference between H1 and H2?
A: H1 changes how words are counted (plain counts versus TF-IDF weights). H2 adds new clues that are not words (certainty and style features) on top of TF-IDF.

Q: Why do we not simply report the share of correct answers?
A: Because most articles are mainstream, so always answering "mainstream" would already be right about 70% of the time while finding no conspiracy article.

Q: What does the label in this study describe?
A: The website that published the article, not the content of the single article.
