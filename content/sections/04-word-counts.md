---
title: Word counts
part: Methods
hypothesis: We expect plain word counts to tell the two kinds of website apart better than always guessing mainstream, because the two kinds tend to use different words.
---
## Explained simply

**The idea.** Imagine sorting letters into two piles by looking only at which words appear in them. You never read for meaning. You just notice that letters with "hurry" and "secret" mostly come from one sender. That is our first analysis. We count how often each word appears in each article and let a simple model learn which words go with which kind of website. Ignoring word order and only counting is called a **bag of words**. It is the simplest approach and our starting point.
**How does logistic regression learn from words?** Every word gets a **weight**, a number that starts at zero. The model reads a training article, adds up the weights of its words and turns the total into a probability that the article comes from a conspiracy website. If it guesses wrong, it nudges the weights of that article's words towards the true answer. After many rounds, words found mostly in conspiracy articles get positive weights and words found mostly in mainstream articles get negative weights. A high weight means "useful for telling these websites apart", not "proves conspiracy thinking".

**Regularisation strength (C).** This setting decides how freely the model may use large weights. A small value keeps the weights small and cautious. A large value lets the model follow the training articles very closely, which can lead to **overfitting**: memorising instead of learning. Oddly, small C means strong caution. We try five values (0.01, 0.1, 1, 10, 100), the same for all three logistic regressions.

**How do we pick the value? Precision, recall and F1.** Most articles come from mainstream websites, so "accuracy" would flatter a model that rarely says "conspiracy". We use three numbers about the conspiracy class instead:
- **Precision**: of the articles the model calls conspiracy, how many really are. It punishes false alarms.
- **Recall**: of the really conspiracy articles, how many the model finds. It punishes misses.
- **F1**: one number that balances the two. It is high only if both are high.

The value of C with the best F1 on the validation articles wins. The test articles stay untouched.

**Other choices, and why.**
- Only training articles build the word list, so no word from the exam shapes the model.
- Words in at least two articles only. Words used once are mostly typos and names.
- Class weight "balanced": there are fewer conspiracy articles, so each counts more.
- We use no other classifier and no word order: the question is about the input, and logistic regression keeps the weights readable.

**Our expectation.** Section 01 states what we expected. Here we report validation results only.

## The code

First the vocabulary is counted. It is fitted on the training articles only.

<!-- cell: src/fml/models.py#word_counts_vectorizer -->

One function fits a single logistic regression. Its argument C is the regularisation strength.

<!-- cell: src/fml/models.py#fit_logistic -->

This function fits one model per regularisation strength (C) and keeps the one with the best validation F1.

<!-- cell: src/fml/models.py#tune_on_validation -->

This function runs the whole analysis and writes the results.

<!-- cell: src/fml/pipeline.py#run_logistic_analysis -->

Finally we list the words with the largest weights in each direction, in two tables.

<!-- cell: src/fml/models.py#top_words -->

<!-- output: results/tables/word_counts_grid.csv -->
<!-- output: results/figures/word_counts_validation_confusion.png | Word counts: validation confusion matrix -->
<!-- output: results/tables/word_counts_top_conspiracy_words.csv -->
<!-- output: results/tables/word_counts_top_mainstream_words.csv -->
<!-- output: results/metrics.json#word_counts.validation -->

## Report

**Idea.** The first method is the plainest baseline. Each article becomes a bag of words: the number of times each word occurs, ignoring order. Logistic regression (Hastie, Tibshirani and Friedman, 2009) gives each word a weight, adds the weights of an article's words and turns the sum into a probability that the article comes from a conspiracy website; above 0.5 it predicts conspiracy. We expected it to beat always guessing mainstream, because the two kinds of website use different words.

**Setup.** The same setup holds for all three logistic regressions. The vocabulary and every other learned quantity come from the training articles only, and a word must occur in at least two of them (23,635 words). Class weights are balanced, because conspiracy articles are the smaller group. The regularisation strength C, which pulls the weights towards zero and so keeps the model from memorising single articles (Hastie, Tibshirani and Friedman, 2009, ch. 4), was chosen from {0.01, 0.1, 1, 10, 100} by validation F1.

**Validation result.** The strongest regularisation, C = 0.01, scored best: precision 0.828, recall 0.906 and F1 0.865. The model found 106 of the 117 conspiracy articles and called 22 of the 259 mainstream articles conspiracy. Weaker regularisation scored lower (F1 0.791 at C = 100), the typical sign of overfitting.

<!-- output: results/figures/word_counts_validation_confusion.png | Word counts: validation confusion matrix -->

**What it learned.** The strongest words are partly about content ("truth", "cia") and partly about presentation: "posted" and "am" sound like blog posts, "says" and "killed" like news reports. The model therefore partly learns how the training websites write, a first hint of why new websites will be harder.

## Check yourself

Q: Why does a high weight for the word "posted" not show that conspiracy websites think differently?
A: A weight only says the word helps separate these websites in this data. "Posted" may just be how some websites label their articles, which says nothing about their ideas.

Q: What would go wrong if the word list were built from all articles, including validation and test?
A: Words from the exam articles would shape the model before it is tested, so the scores would look better than they really are.

Q: A model finds 9 of 10 conspiracy articles but also calls many mainstream articles conspiracy. Which of precision and recall is low?
A: Precision is low, because many of the articles it calls conspiracy are not. Recall is high, because it finds nearly all real ones.

Q: Why is the regularisation strength (C) chosen on validation and not on the training articles?
A: On the training articles a free model always looks best because it memorises them. Validation articles are unseen, so they show which setting really generalises.
