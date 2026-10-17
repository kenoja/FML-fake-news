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
- The same three values (0.1, 1, 10) of the cautiousness setting (regularisation), and the one with the best validation F1 for the conspiracy class wins.

**Why we do it this way.** The rarity part (IDF) is learned from the articles, so it must be computed on the training articles only. If validation or test articles shaped it, the model would have peeked at the exam. We also use a gentle version of the term frequency (the logarithm of the count), so that a word repeated fifty times does not count fifty times as much as a word used once. And TF-IDF scales every article to the same length, so long articles do not win just because they have more words.

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
<!-- output: results/tables/tfidf_top_words.csv -->
<!-- output: results/metrics.json#tfidf.validation -->

## Report

The second analysis keeps everything from the word-count analysis (articles, split, vocabulary, balanced logistic regression, grid for C) and replaces raw counts by TF-IDF weights. Term frequencies were damped with a logarithm, article vectors were scaled to unit length, and the inverse document frequencies were computed on the training articles only. C was again chosen from {0.1, 1, 10} by the validation F1 score of the conspiracy class.

The largest value, C = 10, scored best, with a precision of 0.922, a recall of 0.906 and an F1 score of 0.914 on validation. Compared with word counts (F1 0.839), the weighting raised all three numbers. Unlike with raw counts, larger C helped: because TF-IDF vectors have a fixed length, the model can use larger weights without being dominated by long articles. The chosen value is at the edge of the grid, so a still larger C might do slightly better. We kept the grid as planned.

The strongest words resemble those of the word-count model. "Posted", "truth" and "article" point to conspiracy websites, and "killed", "tower" and "response" point to mainstream websites. Part of this signal therefore reflects how websites write and present their articles. TF-IDF improves the use of words, but it does not remove this limitation. Whether the improvement holds on unseen articles from other websites is only decided by the final comparison.

<!-- output: results/figures/tfidf_validation_confusion.png | TF-IDF: validation confusion matrix -->

## Check yourself

Q: Why does "the" get a low TF-IDF weight although it is very frequent in every article?
A: The second part of TF-IDF measures rarity across articles. "The" occurs in nearly all of them, so it cannot help to tell articles apart and its weight stays low.

Q: Why is the rarity of words computed on the training articles only?
A: If validation or test articles shaped the weights, information from the exam would leak into the model and the scores would look better than they are.

Q: Why does a higher TF-IDF score on validation not prove that the model understands the articles better?
A: It only shows that the weighted words separate these websites better. The model still sees only words, not meaning.
