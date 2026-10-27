---
title: Data and cleaning
part: Data
hypothesis: Removing articles that are not about 9/11 and articles that appear twice makes the data a fairer test, and even the shortest articles are long enough to classify.
---
## Explained simply

**Where the articles come from.** We use LOCO, a collection of web pages from conspiracy-oriented and mainstream websites. LOCO is a **corpus**, which simply means a large collection of texts. It was compiled by Miani, Hills & Bangerter (2022), who rated each website as conspiracy-oriented or mainstream. We use the 9/11 part, full articles only: 3,146.

**What the label means, and its risk.** Every article inherits the rating of its website, so the label describes the website, not the text. A "wrong" answer is sometimes a sensible reading.

**Why cleaning matters.** A model learns whatever separates the two groups most easily, not what we hope it learns. This is **shortcut learning**: a pupil who passes by recognising the teacher's handwriting on the answer key has learned nothing about the subject. If a website's name or a copied article lets the model solve the task, the score looks good and means little. So we clean once, before any model runs, and all four analyses use the same articles.

**Step 1: remove articles that are not about 9/11.** Some pages barely mention the attacks. A model that sees a cooking blog and a news report learns cooking versus news, not the two kinds of websites. We keep an article if it contains at least one of these search words: 9/11 (also 9-11), nine eleven, September 11 (Sept 11, 11 September), Twin Towers, World Trade Center, WTC, Pentagon, al-Qaeda (several spellings), bin Laden, Ground Zero. This removed 538 articles.

**Step 2: remove near-duplicates.** Websites copy each other, so the same story can appear twice. Imagine an exam where some questions also appeared on the practice sheet: a good score would prove little. We call two articles near-duplicates when their word patterns are very alike (a **cosine similarity** of at least 0.9, where 1 means identical). We drop one copy of each pair. If the copies sit in different parts of the split, the copy used for training stays; if both sit in one part, the one with the lower id stays. The label plays no role.

**Step 3: remove clues that give the website away.** We delete web addresses, e-mail addresses, website names and lines a website repeats in many articles, such as menus and "share this" banners. This step changes texts but never removes an article.

**Team question 5: could very short texts be impossible to classify?** The shortest article has about 100 words, and the middle one is several times longer. These are real articles, so we set no minimum.

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
<!-- output: results/tables/site_concentration.csv | The three largest conspiracy websites (names hidden): articles, share of all conspiracy articles, articles in train -->
<!-- output: results/metrics.json#cleaning.top3_sites_share -->
<!-- output: results/tables/length_quantiles.csv | Words per article (q5 means 5% of articles are shorter) -->
<!-- output: results/figures/length_distribution.png | Words per article by class (log scale) -->

## Report

The study uses the 9/11 subset of the LOCO corpus (Miani, Hills & Bangerter, 2022), a collection of web pages from websites rated as conspiracy-oriented or mainstream. In their own blind check of 60 documents per subcorpus, 85% of the conspiracy and 92% of the mainstream documents were correctly labelled. We use the full articles of this subset, 3,146 documents from 39 conspiracy and 79 mainstream websites. Each document carries the label of its website, so the label describes the publisher and not the single text. The texts belong to their publishers, and no licence terms for the corpus could be confirmed, so the texts are kept in a private data store and this report shows only counts and short excerpts. The work used Python 3.11.17 with pandas, scikit-learn 1.9.1 (Pedregosa et al., 2011) and the Anthropic client library 1.17.0.

A classifier takes the easiest signal that separates the classes; this shortcut learning (Geirhos et al., 2020), a form of target leakage (Kaufman et al., 2012), arises when site names, boilerplate, topics or copies reveal the label. All cleaning was therefore applied before any model was fitted. First, documents were kept only if their title or text contains at least one of eleven fixed search words or phrases (9/11, nine eleven, September 11, Twin Towers, World Trade Center, WTC, Pentagon, al-Qaeda, bin Laden, Ground Zero, with common spellings), so 538 were removed (3,146 to 2,608), because a model could otherwise learn topics instead of kinds of websites. Second, near-duplicates were removed (2,608 to 2,554), so that no model meets a near copy of a test article during training: pairs with a cosine similarity of their TF-IDF vectors of at least 0.9 (a lower threshold would also remove different articles on the same event). The copy in the earlier set (training, then validation, then test) is kept. Third, web addresses, e-mail addresses, website names and lines repeated across many documents of one website were deleted from the texts. This step edited 2,498 texts but removed no document.

<!-- output: results/tables/cleaning_log.csv | Documents before and after each cleaning step -->

The raw subset has 39 conspiracy websites; after cleaning, 2,554 documents from 37 conspiracy and 79 mainstream websites remain (25, 6 and 6 conspiracy websites in training, validation and test). They hold 755 conspiracy and 1,799 mainstream articles, so the classes are unbalanced. The three largest conspiracy websites supply 157 training articles, 20.8% of all 755 conspiracy articles. The shortest document has 101 words, so every document is a real article and no minimum length was needed.

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
