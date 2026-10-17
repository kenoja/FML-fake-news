# Conspiracy or mainstream? Classifying 9/11 articles with logistic regression and an LLM

## Word counts

The first analysis represents each article as a bag of words: a vector with the number of times each word occurs, ignoring order. Words were counted on the training articles only, and words occurring in fewer than two training articles were dropped. A logistic regression with balanced class weights was fitted on these counts. The regularisation strength C was chosen from {0.1, 1, 10} by the F1 score of the conspiracy class on the validation set. The test set was not used.

The smallest value, C = 0.1, scored best. On validation this model reached a precision of 0.832, a recall of 0.846 and an F1 score of 0.839 (see the table below). Larger values of C scored lower, which is the typical sign of overfitting: with raw counts, the model starts to rely on rare words that only describe the training articles. The best value lies at the edge of the grid, so a smaller C could be slightly better. We did not widen the grid after seeing the results.



| | |
|---|---|
| precision | 0.832 |
| recall | 0.846 |
| f1 | 0.839 |



The words with the largest weights are informative, but not all of them are about content. Several of the strongest conspiracy words, for example "posted", "am" and "article", look like habits of how a website presents its articles, not like conspiracy thinking. Strong mainstream words such as "killed", "tower" and "anniversary" fit factual news reporting. The model therefore partly recognises websites and not only arguments, which matters for how the results are read in later sections.



![Word counts: validation confusion matrix](../results/figures/word_counts_validation_confusion.png)

## TF-IDF

The second analysis keeps everything from the word-count analysis (articles, split, vocabulary, balanced logistic regression, grid for C) and replaces raw counts by TF-IDF weights. Term frequencies were damped with a logarithm, article vectors were scaled to unit length, and the inverse document frequencies were computed on the training articles only. C was again chosen from {0.1, 1, 10} by the validation F1 score of the conspiracy class.

The largest value, C = 10, scored best, with a precision of 0.922, a recall of 0.906 and an F1 score of 0.914 on validation. Compared with word counts (F1 0.839), the weighting raised all three numbers. Unlike with raw counts, larger C helped: because TF-IDF vectors have a fixed length, the model can use larger weights without being dominated by long articles. The chosen value is at the edge of the grid, so a still larger C might do slightly better. We kept the grid as planned.



| | |
|---|---|
| precision | 0.922 |
| recall | 0.906 |
| f1 | 0.914 |



The strongest words resemble those of the word-count model. "Posted", "truth" and "article" point to conspiracy websites, and "killed", "tower" and "response" point to mainstream websites. Part of this signal therefore reflects how websites write and present their articles. TF-IDF improves the use of words, but it does not remove this limitation. Whether the improvement holds on unseen articles from other websites is only decided by the final comparison.



![TF-IDF: validation confusion matrix](../results/figures/tfidf_validation_confusion.png)
