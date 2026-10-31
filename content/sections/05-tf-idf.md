---
title: TF-IDF
part: Methods
hypothesis: Giving rare, telling words more weight than everyday words improves on plain word counts.
---
## Explained simply

**The problem with plain counts.** Words like "the", "said" or "this" occur in almost every article, so a count of them says little. A word like "demolition" occurs in few articles, so when it does appear it tells you much more. Plain counts treat both the same way.

**The fix: TF-IDF.** TF-IDF turns each count into a weight made of two parts. The first part (**term frequency**) asks how often the word appears in this article. The second part (**inverse document frequency**) asks how rare the word is across all articles: the fewer articles use it, the higher the score. Multiply the two and you get a high weight for words that are frequent here but rare elsewhere. Think of a library: if every book contains the word "chapter", it cannot help you find one book. A word only a few books contain is a good search term.

**What stays the same.** Everything else is identical to the word-count analysis, so that any difference comes from the weighting alone:
- The same articles, the same split, the same list of words (words in at least two training articles).
- The same model, logistic regression, with class weight "balanced" so the smaller conspiracy group is not ignored.
- The same five values (0.01, 0.1, 1, 10, 100) of the regularisation strength (C), and the one with the best validation F1 for the conspiracy class wins. F1 balances precision (how many of the articles called conspiracy really are) and recall (how many of the conspiracy articles are found).

**Why we do it this way.** The rarity part (IDF) is learned from the articles, so it must be computed on the training articles only. If validation or test articles shaped it, the model would have peeked at the exam. We also use a gentle version of the term frequency (the logarithm of the count), so that a word repeated fifty times does not count fifty times as much as a word used once. And TF-IDF scales every article to the same length, so long articles do not win just because they have more words. We try no other weighting and no word pairs here, so that the weighting is the only change.

**Our expectation.** Section 01 says what we expected from rare, telling words. This section only reports the TF-IDF model on the validation articles; the methods are compared in the results section, using the test articles.

**What to look out for.** Better scores do not mean the model understands the articles. It still only sees words. Strong words can still reflect how a website is built rather than what it argues.

## The code

The vectorizer is the only thing that changes compared with the word-count analysis. It is fitted on the training articles only.

<!-- cell: src/fml/models.py#tfidf_vectorizer -->

The same helpers as before choose the setting and fit the model.

<!-- cell: src/fml/models.py#tune_on_validation -->

The analysis function is shared, so both analyses run in exactly the same way.

<!-- cell: src/fml/pipeline.py#run_logistic_analysis -->

<!-- output: results/tables/tfidf_grid.csv -->
<!-- output: results/figures/tfidf_validation_confusion.png | TF-IDF: validation confusion matrix -->
<!-- output: results/tables/tfidf_top_conspiracy_words.csv -->
<!-- output: results/tables/tfidf_top_mainstream_words.csv -->
<!-- output: results/metrics.json#tfidf.validation -->

## Report

**Idea.** TF-IDF keeps the bag of words but weights each count by how rare the word is across the training articles (Manning et al., 2008; Salton & Buckley, 1988): everyday words such as "said" count little, rare and telling words count more. H1 expects this to beat plain counts.

**Setup.** As for word counts. In addition, counts are damped with a logarithm, each article vector is scaled to unit length, and the rarity weights are computed on the training articles only.

**Validation result.** C = 10 scored best: precision 0.930, recall 0.906 and F1 0.918. The model found the same 106 of 117 conspiracy articles as word counts but raised only 8 false alarms instead of 22, so the whole gain is in precision. A larger C suits TF-IDF because its values are small numbers that need larger weights.

<!-- output: results/figures/tfidf_validation_confusion.png | TF-IDF: validation confusion matrix -->

**What it learned.** The strongest words barely change: "posted", "truth" and "article" point to conspiracy websites, "killed", "tower" and "york" to mainstream ones. TF-IDF reduces noise rather than finding new signals.

## Check yourself

Q: Why does "the" get a low TF-IDF weight although it is very frequent in every article?
A: The second part of TF-IDF measures rarity across articles. "The" occurs in nearly all of them, so it cannot help to tell articles apart and its weight stays low.

Q: Why is the rarity of words computed on the training articles only?
A: If validation or test articles shaped the weights, information from the exam would leak into the model and the scores would look better than they are.

Q: Why does a higher TF-IDF score on validation not prove that the model understands the articles better?
A: It only shows that the weighted words separate these websites better. The model still sees only words, not meaning.
