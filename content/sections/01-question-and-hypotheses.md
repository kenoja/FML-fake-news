---
title: Question and hypotheses
part: Introduction
hypothesis: Weighting words helps, our own features help, and the language model beats the models we trained.
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

**What is a hypothesis?** A hypothesis is a clear guess that the results can prove wrong. Ours were chosen from what earlier studies suggest and were fixed before the final test run; we cannot show the exact date on which they were written down. We have three, and the second has three parts. Later the results say for each one whether the data support it. If they do not clearly, we say so. That is also an answer.

- **H1.** Weighting words by how telling they are (TF-IDF) improves on plain word counts.
- **H2.** Adding our own features improves the TF-IDF model. We test three kinds of features one by one:
  - **H2a.** Certainty words (such as "obviously" or "perhaps") improve it.
  - **H2b.** Style features (such as capital letters and exclamation marks) improve it.
  - **H2c.** Phrases of two or three words in a row (n-grams) improve it.
- **H3.** The language model does better than the best of the three models we trained.

**Why these three?** They follow the order of the study: first the choice of how to represent words (H1), then whether our own ideas add anything (H2a to H2c), then the comparison with a very different approach (H3). Each is answered later, in the results and the conclusion.

**How we judge "better".** Most articles come from mainstream websites. A model that always says "mainstream" would be right most of the time and would find no conspiracy article at all. So we do not use the share of correct answers. We use the F1 score for the conspiracy class, which rewards finding conspiracy articles without raising false alarms. The glossary explains it.

**A warning about the label.** The label belongs to the website, not to the single article. A calm news item on a conspiracy website counts as "conspiracy".
## The code

This section only sets the question. The one piece of code that belongs here is the F1 score that we use to judge every method. It combines how many of the flagged articles are right with how many conspiracy articles we find.

<!-- cell: src/fml/evaluate.py#f1_score -->

The real analysis code starts in the next sections.

## Report

This study asks whether the text of a full article about the 9/11 attacks reveals if it was published on a conspiracy-oriented or on a mainstream website, and how four methods compare on this task. The methods are logistic regression on word counts, logistic regression on TF-IDF weights, logistic regression on TF-IDF weights plus hand-made features, and one low-cost large language model (LLM) used without any training on our data. The hypotheses were chosen from the literature and fixed before the final test run; the exact date on which they were written down cannot be shown.

H1. Weighting words by their distinctiveness (TF-IDF) improves on plain word counts.

H2. Adding hand-made features improves the TF-IDF model. This is tested separately for three kinds of features: H2a, certainty words; H2b, style features; H2c, phrases of two or three words (n-grams).

H3. The LLM does better than the best logistic regression.

All methods are judged by precision, recall and F1 score for the conspiracy class, because the classes are unbalanced: most articles come from mainstream websites, so the share of correct answers would reward a model that never finds a conspiracy article. A hypothesis counts as supported only if the confidence interval and the significance test agree with the direction of the difference. Results are reported in the results section, and the conclusion returns to each hypothesis.

## Check yourself

Q: Why should a hypothesis be fixed before the final test run and not adjusted afterwards?
A: A guess adjusted after seeing the results would always seem to fit, so the results could no longer show it to be wrong.

Q: Why do we test H2a, H2b and H2c separately and not all features together?
A: Together, a helpful group could be hidden by a harmful one. Separately we see which kind of feature helps and which does not.

Q: Why can a model with high accuracy still be useless here?
A: Most articles are mainstream, so always answering "mainstream" is right most of the time while finding no conspiracy article.

Q: A calm news item sits on a conspiracy website. Why is it still labelled "conspiracy"?
A: The label describes the website that published the article, not the content of the single article.
