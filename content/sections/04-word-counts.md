---
title: Word counts
part: Methods
---
## Explained simply

**The idea.** Imagine sorting letters into two piles by looking only at which words appear in them. You never read for meaning. You just notice that letters with "hurry" and "secret" mostly come from one sender. That is our first analysis. We count how often each word appears in each article and let a simple model learn which words go with which kind of website. Ignoring word order and only counting is called a **bag of words**. We start here because it is the simplest possible approach: if it already works, fancier methods have to beat it.

**How does logistic regression learn from words?** (Team question 2.) Every word gets a **weight**, a number that starts at zero. The model reads a training article, adds up the weights of its words, and turns the total into a probability that the article comes from a conspiracy website. If it guesses wrong, it nudges the weights of the words in that article towards the true answer. After many rounds, words found mostly in conspiracy articles get positive weights, and words found mostly in mainstream articles get negative weights. A new article is scored by adding up its weights. So the model really does learn from labelled texts. The counting is only its input.

A high weight means "useful for telling these websites apart", not "proves conspiracy thinking". A word like "posted" may simply reflect how a website is built.

**Why these choices?**
- **Only training articles build the word list.** Words that appear only in validation or test articles must not influence the model. Otherwise it would have peeked at the exam.
- **Words in at least two articles.** Words used once are mostly typos and names, and they let the model memorise single articles.
- **Class weight "balanced".** There are fewer conspiracy articles than mainstream ones, so each conspiracy article counts more. Otherwise the model could ignore the smaller group.
- **A short list of cautiousness settings, chosen on validation.** The **regularisation** setting (called C in the code) controls how freely the model may use large weights. A cautious setting keeps weights small. A free setting lets the model follow the training data closely, which can lead to **overfitting** (memorising instead of learning). We try three values (0.1, 1, 10) and keep the one with the best validation F1 for the conspiracy class. Validation is a separate set of articles, so the choice is honest. The test set stays untouched.

## The code

First the vocabulary is counted. It is fitted on the training articles only.

<!-- cell: src/fml/models.py#word_counts_vectorizer -->

One function fits a single logistic regression.

<!-- cell: src/fml/models.py#fit_logistic -->

This helper fits one model per cautiousness setting and keeps the best on validation.

<!-- cell: src/fml/models.py#choose_c -->

This function runs the whole analysis and writes the results.

<!-- cell: src/fml/pipeline.py#run_logistic_analysis -->

Finally we list the words with the largest weights in each direction.

<!-- cell: src/fml/models.py#top_words -->

<!-- output: results/tables/word_counts_grid.csv -->
<!-- output: results/figures/word_counts_validation_confusion.png | Word counts: validation confusion matrix -->
<!-- output: results/tables/word_counts_top_words.csv -->
<!-- output: results/metrics.json#word_counts.validation -->

## Report

The first analysis represents each article as a bag of words: a vector with the number of times each word occurs, ignoring order. Words were counted on the training articles only, and words occurring in fewer than two training articles were dropped. A logistic regression with balanced class weights was fitted on these counts. The regularisation strength C was chosen from {0.1, 1, 10} by the F1 score of the conspiracy class on the validation set. The test set was not used.

The smallest value, C = 0.1, scored best. On validation this model reached a precision of 0.832, a recall of 0.846 and an F1 score of 0.839 (see the table below). Larger values of C scored lower, which is the typical sign of overfitting: with raw counts, the model starts to rely on rare words that only describe the training articles. The best value lies at the edge of the grid, so a smaller C could be slightly better. We did not widen the grid after seeing the results.

The words with the largest weights are informative, but not all of them are about content. Several of the strongest conspiracy words, for example "posted", "am" and "article", look like habits of how a website presents its articles, not like conspiracy thinking. Strong mainstream words such as "killed", "tower" and "anniversary" fit factual news reporting. The model therefore partly recognises websites and not only arguments, which matters for how the results are read in later sections.

<!-- output: results/figures/word_counts_validation_confusion.png | Word counts: validation confusion matrix -->

## Check yourself

Q: Why does a high weight for the word "posted" not show that conspiracy websites think differently?
A: A weight only says the word helps separate these websites in this data. "Posted" may just be how some websites label their articles, which says nothing about their ideas.

Q: What would go wrong if the word list were built from all articles, including validation and test?
A: Words from the exam articles would shape the model before it is tested, so the scores would look better than they really are.

Q: The model does not read meaning. How can it still sort articles?
A: During training it learned which words usually come with which kind of website, and it adds up those learned weights for a new article.
