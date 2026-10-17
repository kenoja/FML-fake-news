---
title: Limitations
part: Interpretation
hypothesis: Our results are real but narrow - the model mostly learns what separates these websites, not what makes a text a conspiracy text, and one split with few websites cannot tell the two apart with certainty.
---
## Goal and hypothesis

This section lists what our numbers do **not** show. We expected the main weaknesses to be about the labels (they describe a website, not an article), about how few websites we have, and about shortcuts a model can use without understanding anything. We also answer open question 6 here.

## Explanation

### Open question 6: can the model recognise the website without being told its name?

Yes, in principle it can, and that is the main threat to our claim. A model that learns "this is the typical look of site X" would score well on documents from site X but would learn nothing about conspiracy writing. Two protections work against this, and they protect against **different** things.

**1. The grouped split protects against unseen-website evaluation being too easy.** *Grouped split* means every website lands in exactly one of train, validation or test (`websites_in_more_than_one_split` in `metrics.json` is 0). If articles of one site were spread over train and test, a model could memorise that site's style on train and be rewarded for it on test. With the grouped split, the test sites are strangers to the model, which is the situation a real user faces. The price: results depend on *which* sites ended up in test (see below).

**2. Removing website names, URLs and repeated boilerplate protects against the cheapest shortcuts.** Before any model sees a text, `clean_corpus` strips the site's own name, URLs and e-mail-like strings, and any line that appears in at least 3 documents of one website (menus, "Subscribe to our newsletter", copyright lines). The cleaning log shows how much was removed at each step:

<!-- cell: src/fml/clean.py#clean_corpus -->
<!-- output: results/tables/cleaning_log.csv | What the cleaning removed, step by step -->

Together the two steps stop a model from saying "the text contains `neonnettle.com`, so conspiracy" or from recognising a site by its footer. They do **not** remove everything that identifies a site type. What remains:

- **House style.** Blog-format words such as "posted", "rss" and "am" are among the strongest conspiracy coefficients of the words-only model (see the list below). They say "this is a blog post", not "this is a conspiracy claim". `docs/decisions.md` (T3) records this as a style giveaway.
- **Bylines, dateline and sentence templates** that a site repeats but that are not exact repeated lines, so the boilerplate rule misses them.
- **Vocabulary specific to a site type.** Conspiracy sites say "mainstream", "truth", "official"; news agencies say "counterterrorism", "Manhattan". That is partly the genuine signal we want, and partly a topic-and-genre signal. We cannot separate the two with these data.

<!-- output: results/tables/m1_words_filtered_top_coefficients.csv | Words-only model: the 25 strongest words per direction -->

**Test evidence.** (a) We scanned the 25 strongest conspiracy and 25 strongest mainstream words of both words-only models (all documents and filtered) for the names of the test websites and of the other well-known outlets in the data: none appears. So the name removal worked on the words that matter most. This is a scan of the top 25 only; a site token with a smaller weight would not be seen. (b) Accuracy varies a lot between websites in the test set, which is what a style-dependent model looks like:

<!-- output: results/tables/per_website_test.csv | Test accuracy per website -->

One conspiracy site (`neonnettle.com`, 44 test documents) is classified correctly by the logistic regression for only 16% of its articles, while the other five conspiracy sites reach 53% to 100%. A model that had merely memorised "conspiracy sites" would not fail so unevenly; one that picked up the style of the training sites would. We read this as: the model learned **some** transferable signal and a good deal of site-type style. We have no experiment that isolates the two (for example a model trained on style features alone), so this is an interpretation, not a measurement.

### Other limitations

- **The label is the website, not the content.** An article is "conspiracy" because its publisher is. Error analysis (section 11) shows many conspiracy-site articles contain no conspiracy claim, and some mainstream articles discuss such claims. Scores therefore measure "recognise the source", not "detect conspiracy content". This is what we asked both models.
- **Label noise.** Mompelat et al. (2022) annotated LOCO documents and found that a large share of the conspiracy subcorpus contains no conspiracy language (about 62% and 41% for two other topics). We expect the noise mainly on the **conspiracy side**: mainstream sites rarely publish conspiracy text, while conspiracy sites publish everything. Our 9/11 numbers from section 11 point the same way. Noise caps recall for every model and we cannot correct it without human coding.
- **Few websites, wide intervals.** The test split has only 6 conspiracy websites (18 websites in all, 423 documents after the filter). Intervals that resample websites instead of documents are much wider than document-level ones, for example the F1 of the LLM:

<!-- output: results/tables/comparison_test.csv | Test metrics with document and website bootstrap intervals -->

Differences that look clear at document level (the McNemar test on documents is significant) are far less certain when the websites are the unit. A different split could change the ranking of the models.
- **The off-topic filter and the "pentagon" rule.** The filter keeps documents that match one of several terms, including "pentagon". Only 10 kept documents match "pentagon" alone and at least one is an unrelated defence story (`metrics.json#topic_filter.pentagon_only`). The filter also removes unevenly across splits (test lost far more conspiracy than mainstream documents, mostly from one website; two conspiracy websites lost all their documents). We did not tune the pattern after seeing results.

<!-- output: results/metrics.json#topic_filter.removal_rate_by_split | Share of documents removed per split -->

- **LLM prior knowledge.** Claude Haiku was trained on web text. It may have seen these very articles or websites, and it may know that a site is conspiracy-oriented from its name, tone or reputation. This could inflate its score in a way the logistic regression cannot match, and we cannot measure it. A second concern is the 150-token output limit: parse errors (replies cut off) count as "mainstream" and lower recall (61 on the test pass); see section 9. Temperature 0 could not be set for this model (decisions.md), so we measured stability with a second pass instead (section 10).
- **Lexicon negation weakness.** Our epistemic features count words from a hand-made list. They ignore negation: "no evidence of a cover-up" counts "cover-up" like "it was a cover-up" (Chen 2026). The n-gram features partly compensate, but the weakness stays and is tested on purpose in `tests/test_features.py`.
- **Licence status.** We found no licence for the LOCO texts and could not open the OSF page. We therefore keep texts private and show only short excerpts. If a licence turns out to be stricter, the public repo needs review.
- **One seed, one split, one corpus.** All numbers come from one split (`split_v1`, seed from config) and one random seed. We did not repeat the experiment with other splits, so we cannot say how much the results depend on this particular assignment of websites to splits.
- **No transfer test.** The JFK transfer test is out of scope for now. We cannot say whether anything learned here works on another topic.
- **Full articles only, first 3,000 words for the LLM.** Longer articles are cut for the LLM; the logistic regression sees the full cleaned text.

## What could go wrong

- Reading the F1 numbers as "how good the model is at spotting conspiracy theories". They show how well each approach recovers the website type under these conditions.
- Assuming that the name removal catches everything: it only catches exact strings and repeated lines.
- Reading small differences between models as real when the website-level intervals overlap heavily.
- Trusting numbers in this section after a rerun without checking: they are copied from `results/` and must be refreshed when `fml run-all` changes them.

## Technical infrastructure

- Grouped split: `src/fml/split.py#make_split` (website = group, stratified by label, frozen `data/splits/split_v1.csv`).
- Name, URL and boilerplate removal: `src/fml/clean.py` (`strip_website_names`, `strip_urls_and_emails`, `strip_boilerplate`).
- Leakage tests: `tests/test_leakage.py`; split tests: `tests/test_split_clean.py`.
- Numbers: `results/metrics.json` (`split`, `topic_filter`, `comparison`, `errors`), `results/tables/per_website_test.csv`, `results/tables/m1_words_filtered_top_coefficients.csv`.

<!-- cell: src/fml/split.py#make_split -->
<!-- cell: src/fml/clean.py#strip_website_names -->
