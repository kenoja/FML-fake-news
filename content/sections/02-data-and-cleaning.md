---
title: Data and cleaning
part: Data
hypothesis: Removing articles that are not about 9/11 and articles that appear twice makes the data a fairer test, and even the shortest articles are long enough to classify.
---
## Explained simply

**Where the articles come from.** We use LOCO, a collection of web pages from conspiracy-oriented and mainstream websites. LOCO is a **corpus**, which simply means a large collection of texts. We use the part about the 9/11 attacks and only the full articles, 3,146 of them. LOCO (Language of Conspiracy corpus) was compiled by its authors (Miani, Hills & Bangerter, 2022), and the compilers rated each website as conspiracy-oriented or mainstream. We used Python, scikit-learn and Claude Haiku 5.5. The texts are used for study only and are not redistributed until the licence is confirmed: they stay in a private store, and this page shows only counts, tables and a few short quotes.

**What the label means.** The corpus compilers rated each website as conspiracy-oriented or mainstream, and every article inherits the rating of its website. So the label describes the website, not the text. A polite note from a sender known for chain letters still counts as chain-letter mail, so a "wrong" model answer is sometimes a sensible reading.

**Why we clean first.** We clean once, before any model sees an article, so all four analyses use the same articles.

**Step 1: remove articles that are not about 9/11.** The collection was gathered with a broad search, and some pages barely mention the attacks. A model that sees a cooking blog from one website and a news report from another learns cooking versus news, not the two kinds of websites. We keep only articles that mention the attacks, using a fixed list of search words.

**Step 2: remove near-duplicates.** Websites copy each other, so the same story can appear twice with tiny changes. Imagine an exam where some questions also appeared on the practice sheet: a good score would prove little. If one copy is in training and the other in the test set, the model is partly tested on something it has seen. We drop one copy of each pair: the one in validation or test if they are in different parts, otherwise the one with the higher identification number.

**Step 3: remove clues that give the website away.** We delete web addresses, e-mail addresses, website names and lines that a website repeats in many articles, such as menus and "share this" banners. Otherwise a model could learn "contains the site name" instead of anything about the writing.

**Team question 5: could very short texts be impossible to classify?** The shortest article has about 100 words, and the middle one is several times longer. These are real articles, not snippets, so we set no minimum length.

**What to look out for.** Cleaning removes the obvious site clues, not all of them.

## The code

This function runs the three cleaning steps in order and counts what each one removes.

<!-- cell: src/fml/clean.py#clean_documents -->

This one decides which copy of a near-duplicate pair is dropped.

<!-- cell: src/fml/clean.py#near_duplicate_drops -->

This one measures how long the articles are.

<!-- cell: src/fml/data_report.py#length_table -->

<!-- output: results/tables/cleaning_log.csv | Articles before and after each cleaning step -->
<!-- output: results/figures/cleaning_before_after.png | Articles per class before and after cleaning -->
<!-- output: results/metrics.json#cleaning.near_duplicates_dropped_by_split -->
<!-- output: results/tables/length_quantiles.csv | Words per article (q5 means 5% of articles are shorter) -->
<!-- output: results/figures/length_distribution.png | Words per article by class (log scale) -->

## Report

The study uses the 9/11 subset of the LOCO corpus (Miani, Hills & Bangerter, 2022), a collection of web pages from websites rated as conspiracy-oriented or mainstream. We use the full articles of this subset, 3,146 documents from 39 conspiracy and 79 mainstream websites. Each document carries the label of its website. The label therefore describes the publisher and not the single text: a neutral news item on a conspiracy-oriented website is labelled conspiracy, and a mainstream article that discusses a conspiracy theory is labelled mainstream. The texts are not redistributed and are used for study only; they are kept in a private data store, and this report shows only counts and short excerpts. The work was done in Python with pandas and scikit-learn, and the language-model classifier is Claude Haiku 5.5, used through the Anthropic API.

Cleaning was applied before any model was fitted, and all four analyses use the same cleaned documents. First, documents that do not mention the attacks in their title or text were removed, because a model could otherwise learn the difference between topics instead of the difference between kinds of websites. Second, near-duplicates were removed: pairs of documents with a cosine similarity of their TF-IDF vectors of at least 0.9. Such copies are common between websites and would let a model meet a near copy of a test article during training. For a pair across two parts of the data, the copy in validation or test was dropped; inside one part, the copy with the lower identifier was kept. Third, web addresses, e-mail addresses, website names and lines repeated across many documents of one website were deleted from the texts. The table shows the number of documents before and after each of the first two steps.

<!-- output: results/tables/cleaning_log.csv | Documents before and after each cleaning step -->

After cleaning, 2,554 documents remain: 755 from conspiracy-oriented and 1,799 from mainstream websites. Roughly three in ten articles are conspiracy, so the classes are unbalanced. Document length was checked to decide whether very short texts need special treatment. After cleaning, the shortest document has 102 words and the median is 622 words, so no minimum length was applied.

## Check yourself

Q: What does the label "conspiracy" mean in our data?
A: That the article was published on a website rated as conspiracy-oriented. It says nothing certain about the content of the single article.

Q: Why do we remove articles that are not about 9/11?
A: Otherwise a model could learn to tell topics apart instead of telling the two kinds of websites apart.

Q: Why is it a problem if the same story is in the training and the test articles?
A: The model is then partly tested on something it has already seen, so the score looks better than it is.

Q: Do very short texts need special treatment here? Why or why not?
A: No. The shortest article has about 100 words, so all texts are real articles and none is a snippet.
