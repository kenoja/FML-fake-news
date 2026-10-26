---
title: Data and cleaning
part: Data
hypothesis: Removing articles that are not about 9/11 and articles that appear twice makes the data a fairer test, and even the shortest articles are long enough to classify.
---
## Explained simply

**Where the articles come from.** We use LOCO, a collection of web pages from conspiracy-oriented and mainstream websites. LOCO is a **corpus**, which simply means a large collection of texts. It was compiled by Miani, Hills & Bangerter (2022), who rated each website as conspiracy-oriented or mainstream. We use the part about the 9/11 attacks and only the full articles, 3,146 of them.
**What the label means, and its risk.** Every article inherits the rating of its website, so the label describes the website, not the text. A "wrong" model answer is sometimes a sensible reading of the text.

**Why cleaning matters.** A model learns whatever separates the two groups most easily, not what we hope it learns. This is **shortcut learning**: a pupil who passes by recognising the teacher's handwriting on the answer key has learned nothing about the subject. If a website's name or a copied article lets the model solve the task, the score looks good and means little. So we clean once, before any model runs, and all four analyses use the same articles.

**Step 1: remove articles that are not about 9/11.** Some pages barely mention the attacks. A model that sees a cooking blog from one website and a news report from another learns cooking versus news, not the two kinds of websites. We keep only articles that mention the attacks, using a fixed list of search words.

**Step 2: remove near-duplicates.** Websites copy each other, so the same story can appear twice. Imagine an exam where some questions also appeared on the practice sheet: a good score would prove little. We call two articles near-duplicates when their word patterns are very alike (a **cosine similarity** of at least 0.9, where 1 means identical). We drop one copy of each pair. If the copies sit in different parts of the split, the copy used for training stays; if both sit in one part, the one with the lower id stays. The label plays no role.

**Step 3: remove clues that give the website away.** We delete web addresses, e-mail addresses, website names and lines a website repeats in many articles, such as menus and "share this" banners. This step changes texts but never removes an article.
**Team question 5: could very short texts be impossible to classify?** The shortest article has about 100 words, and the middle one is several times longer. These are real articles, so we set no minimum length.

**What to look out for.** Cleaning removes the obvious site clues, not all of them. A site's typical style can still act as a shortcut.

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
<!-- output: results/metrics.json#cleaning.clue_removal -->
<!-- output: results/tables/length_quantiles.csv | Words per article (q5 means 5% of articles are shorter) -->
<!-- output: results/figures/length_distribution.png | Words per article by class (log scale) -->

## Report

The study uses the 9/11 subset of the LOCO corpus (Miani, Hills & Bangerter, 2022), a collection of web pages from websites rated as conspiracy-oriented or mainstream. Its authors retrieved pages from many websites with keyword searches on conspiracy-related topics and assigned each website to a conspiracy or a mainstream subcorpus; in their own blind check of 60 documents per subcorpus, 85% of the conspiracy and 92% of the mainstream documents were correctly labelled. We use the full articles of this subset, 3,146 documents from 39 conspiracy and 79 mainstream websites. After cleaning, the 2,554 remaining documents come from 37 conspiracy and 79 mainstream websites (39 conspiracy websites before cleaning). Each document carries the label of its website. The label therefore describes the publisher and not the single text: a neutral news item on a conspiracy-oriented website is labelled conspiracy, and a mainstream article that discusses a conspiracy theory is labelled mainstream. The work used Python 3.11.17 with pandas 3.1.0 and scikit-learn 1.9.1 (Pedregosa et al., 2011), numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2, anthropic 1.17.0 and pyyaml 6.0.1.

Cleaning matters because a classifier takes the easiest signal that separates the classes. This shortcut learning, a form of target leakage, arises when website names, boilerplate, topic differences or copies of an article reveal the label without any property of the writing. All cleaning was therefore applied before any model was fitted, and all four analyses use the same cleaned documents. First, documents that do not mention the attacks in their title or text were removed (3,146 to 2,608), because a model could otherwise learn topics instead of kinds of websites. Second, near-duplicates were removed (2,608 to 2,554): pairs of documents with a cosine similarity of their TF-IDF vectors of at least 0.9. Otherwise a model could meet a near copy of a test article during training. The rule uses the split, not the label: when the copies lie in different sets, the training copy is kept (the validation copy rather than the test copy), and when both lie in one set, the copy with the lower document id is kept. Third, web addresses, e-mail addresses, website names and lines repeated across many documents of one website were deleted from the texts. This step edited the texts of 2,498 documents and removed 23,390 words, but removed 0 documents. The table shows the number of documents before and after each of the first two steps.

<!-- output: results/tables/cleaning_log.csv | Documents before and after each cleaning step -->

After cleaning, 2,554 documents remain: 755 from conspiracy-oriented and 1,799 from mainstream websites. About three in ten are conspiracy, so the classes are unbalanced.

## Check yourself

Q: What does the label "conspiracy" mean in our data?
A: That the article was published on a website rated as conspiracy-oriented. It says nothing certain about the content of the single article.

Q: Why do we remove articles that are not about 9/11?
A: Otherwise a model could learn to tell topics apart instead of telling the two kinds of websites apart.

Q: Why is it a problem if the same story is in the training and the test articles?
A: The model is then partly tested on something it has already seen, so the score looks better than it is.

Q: Do very short texts need special treatment here? Why or why not?
A: No. The shortest article has about 100 words, so all texts are real articles and none is a snippet.

Q: What is shortcut learning, and how can a website's name cause it?
A: The model solves the task with a clue that gives the answer away instead of with the writing. If articles contain their site's name, the model can learn the name and not the difference between the two kinds of websites.
