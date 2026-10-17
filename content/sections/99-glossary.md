---
title: Glossary
part: Reference
hypothesis: A short list of plain-language definitions lets every teammate follow the other sections without prior machine-learning knowledge.
---
## Goal and hypothesis

Every technical word in this walkthrough is defined here in one or two sentences. We expect that teammates will look words up here instead of skipping the paragraph.

## Explanation

**TF-IDF.** A way to turn a text into numbers. *Term frequency* counts how often a word appears in the article; *inverse document frequency* down-weights words that appear in almost every article ("the"). Words that are frequent here but rare elsewhere get high numbers.

**n-gram.** A run of n words in a row. Unigrams are single words, bigrams two ("inside job"), trigrams three ("no evidence of").

**Logistic regression.** A simple model that gives each feature (a word, a number) a weight, adds the weighted values up, and turns the sum into a probability between 0 and 1. We call an article "conspiracy" when the probability is above 0.5.

**Regularisation (C).** A penalty that keeps the weights small so the model does not memorise the training articles. In scikit-learn, **C** is the *inverse* strength: small C means strong penalty, large C means weak penalty. We choose C on the validation split.

**Class weight.** Makes mistakes on the rarer class (conspiracy) count more during training. "Balanced" weights each class by the inverse of its frequency; "none" treats all documents equally.

**Precision.** Of the articles the model called conspiracy, the share that really come from conspiracy sites.

**Recall.** Of the articles from conspiracy sites, the share the model found.

**F1.** One number that combines precision and recall (their harmonic mean). It is low if either one is low. We report F1 for the conspiracy class.

**Macro-F1.** The F1 of each class, averaged with equal weight, so the larger mainstream class does not dominate.

**Balanced accuracy.** The average of recall on the conspiracy class and recall on the mainstream class. A model that always says "mainstream" gets 0.5.

**Bootstrap.** Re-drawing the test set many times with replacement and recomputing a metric each time, to see how much it wobbles. We draw either documents or whole websites.

**Confidence interval.** The range in which the true value probably lies (here: the middle 95% of the bootstrap values). A wide interval means the number is uncertain.

**McNemar test.** A test for two models scored on the same documents. It looks only at the documents where exactly one model is right and asks whether one model wins such cases more often than chance.

**Grouped split.** A train/validation/test split where all articles of one website stay together in one part. The model is tested on websites it has never seen.

**Stratified.** The split keeps the share of each label about the same in every part, as far as the grouping allows.

**Leakage.** Information from the test (or validation) data getting into training, for example fitting the vocabulary on all articles. It makes results look better than they are.

**Zero-shot.** Asking a language model to do a task with instructions only and no training examples.

**Temperature.** A setting that controls how random a language model's answers are. 0 means "pick the most likely answer". Our classifier model does not accept the setting (see `docs/decisions.md`).

**Prompt variant.** One wording of the instruction we give the LLM. We try at most three on validation and report all.

**Stance.** How an article treats a claim: endorses it, rejects or debunks it, reports it neutrally, or makes no such claim.

**Label noise.** Labels that do not describe the text. Here the label is the website, so a conspiracy-site article without any conspiracy claim is "noisy".

**Lexicon.** A word list, for example words that express doubt. Lexicon features count how often listed words occur; they usually ignore negation.

**Type-token ratio.** Number of different words divided by the number of words in a text. A measure of vocabulary variety; we use a moving-window version so it does not depend on article length.

**Negation-aware count.** A lexicon count that skips a listed word when a negating word ("no", "not", "never", "without") comes just before it. "No evidence of cover-up" then does not count "cover-up". Our plain lexicon counts ignore negation, which is a known weakness.

**Cohen's kappa.** A measure of how much two raters agree beyond what chance would give. 1 means perfect agreement, 0 means no more than chance. We use it for the LLM's two passes (stability) and, once people fill the 40-row sample, for Claude's stance codes against a human's. Until then it is `null`, not 0.

**Parse error (unreadable answer).** An LLM reply that is not valid JSON with a label, often because the 150-token output limit cut it off. We count it as "mainstream" in the main score, which lowers recall, and report how many there were.

**Cluster bootstrap (website bootstrap).** A bootstrap that redraws whole websites instead of single documents. Articles from one site resemble each other, so the document bootstrap is too optimistic; the website interval is wider and is the one to trust when only a few conspiracy websites are in the test set.

## What could go wrong

- Using a word from the glossary loosely in another section. If a definition here and a section disagree, the section should be fixed.
- Definitions that are short may hide details. The sections give the details where the term is used.

## Technical infrastructure

This section has no code. Terms are used in `src/fml/` (for example `src/fml/evaluate.py` for the metrics) and explained at first use in sections 03 to 11.
