---
title: TF-IDF plus extra features
part: Methods
hypothesis: Adding features about certainty and writing style improves the TF-IDF model.
---
## Explained simply

**The idea.** TF-IDF only sees single words. Do a few hand-made clues help on top? A clue is called a **feature**: one number that describes an article. We tried three families, each backed by earlier research.

1. **Phrases (n-grams of one to three words).** "Inside job" or "no evidence of" mean more than their single words (Wang and Manning, 2012).
2. **Certainty and doubt words.** Writers show how sure they are through their words, as in "I know" against "I guess". Our short lists hold doubt words, called hedges ("perhaps"), certainty words, called boosters ("clearly"), hidden-truth words ("cover-up") and debunking words ("no evidence").
3. **Style.** Share of CAPITAL words, exclamation and question marks, quotation marks, sentence length and repetition (Horne and Adali, 2017; Pérez-Rosas et al., 2018).

**Counting per 1,000 words.** An invented text of 100 words with 3 certainty words scores 30; a text of 200 words with the same 3 scores 15. A long text does not look more certain just because it is long. Every number is then put on one common scale, learned on training articles only.

**Why MATTR for repetition?** MATTR measures how much a text repeats itself. The simple share of different words falls as a text grows, so it would measure length. MATTR (moving-average type-token ratio) takes every window of 100 words, computes the share of different words in it, and averages over all windows.

**The comparison table (an "ablation").** An ablation means switching one ingredient on or off and watching what happens to the score, like baking the same cake with and without sugar. We compare six models on the same articles and split: TF-IDF alone, TF-IDF plus each family, TF-IDF plus all three, and TF-IDF plus certainty words and style but without the phrases. Each row has its own best regularisation strength (C). The last column is the change in validation F1 against TF-IDF alone: plus means better, minus worse.

**Why all three together do worse.** Style lowers the score even on its own; a habit such as many capital letters may belong to one website, so it does not carry over to new websites. The phrases add very many columns (146,172 for 1,768 training articles), which probably lets the model memorise single articles. Certainty words are the only family that helps on its own. With so few validation articles, small differences can be luck, so these are tendencies, not proof.

**Why the final model is not the best row.** The model with all three families was planned on 16 October, before any result. In cycle 2 the assistant instead kept the best family by a validation rule, and earlier versions were scored on the test set. On 25 October, after seeing the validation results and those earlier test runs, the team returned to the planned model. So the choice was made after seeing results, not blind; the honest sentence in the limitations section says the same. Certainty words alone scored highest on validation, but our question is about adding all three families, as planned. Switching to the best row after seeing the table would change the question to fit the results.

## The code

The six models differ only in which feature families are switched on.

<!-- cell: src/fml/pipeline.py#ablation_variants -->

Phrases come from a TF-IDF step that reads word runs of length one to three.

<!-- cell: src/fml/features/ngrams.py#build_ngram_vectorizer -->

Counts become counts per 1,000 words, so length does not matter.

<!-- cell: src/fml/features/_text.py#per_n_words -->

Certainty words are counted with this function.

<!-- cell: src/fml/features/epistemic.py#count_markers -->

Style features are computed from the text alone.

<!-- cell: src/fml/features/style.py#style_row -->

Repetition is measured with MATTR: a window of 100 words slides along the text.

<!-- cell: src/fml/features/style.py#mattr -->

Each model is tuned on validation, then the rows are compared with plain TF-IDF. Two rows (certainty words, style) already leave out the phrases; the last variant combines them.

<!-- cell: src/fml/ablation.py#run_ablation -->

<!-- output: results/tables/ablation.csv -->
<!-- output: results/figures/ablation_f1.png | Validation F1 of the six models -->
<!-- output: results/metrics.json#tfidf_features.validation -->
<!-- output: results/tables/epistemic_coverage.csv -->
<!-- output: results/tables/feature_summary.csv -->

## Report

The third analysis tests H2 by adding three groups of hand-made features to TF-IDF. N-grams of one to three words capture phrases such as "inside job"; word and bigram features are a strong baseline for text classification (Wang and Manning, 2012). Epistemic markers are counts per 1,000 words of four short hand-made lists: hedges (doubt words such as "perhaps"), boosters (certainty words such as "clearly"), hidden-truth phrases and debunking words, following work that ties conspiracy discourse to certainty and concealment (Marini and Jezek, 2024; Islam et al., 2020; but see Fong et al., 2021). Style features are capitals, punctuation, quotation marks, sentence length and repetition, the last measured as the share of different words in windows of 100 words (MATTR) (Horne and Adali, 2017); this evidence comes from fake-news data, not conspiracy websites, and such features generalise poorly across domains (Pérez-Rosas et al., 2018).

The numeric groups were standardised with training statistics only. In an ablation, which switches one ingredient on or off, six logistic regressions with the same settings were compared on validation: TF-IDF alone, with each group, with all three (the planned final model), and with certainty words and style but without n-grams.

| Model | Validation F1 | Change against TF-IDF alone |
|---|---|---|
| TF-IDF alone | 0.918 | reference |
| + n-grams (1-3) | 0.920 | +0.002 |
| + certainty words | 0.930 | +0.013 |
| + style | 0.909 | -0.009 |
| + all three groups | 0.896 | -0.022 |
| + certainty words and style (no n-grams) | 0.913 | -0.005 |

<!-- output: results/figures/ablation_f1.png | Validation F1 of the six models -->

Style lowers validation F1 on its own (0.909). Certainty words and style without n-grams reach 0.913, but all three groups only 0.896: the n-grams add 146,172 columns for 1,768 training articles, which probably makes it easy to fit single articles, and style habits such as capitals may describe single websites rather than conspiracy writing. These are likely explanations that fit the table, not tested causes: with 376 validation articles, one hundredth of F1 is about one or two of the 117 conspiracy articles, so no single group differs from TF-IDF alone by more than chance would allow, and the full combination chose the largest regularisation value in the grid (each row chooses its own value on validation). Certainty words are the only group that helps alone (0.930), but H2 is about adding the three groups together, as planned; switching to the best row after seeing the table would change the hypothesis to fit the results, so the planned model was kept and H2 is judged on it. That model was planned on 16 October before any result; in cycle 2 the best group was kept by a validation threshold and earlier versions were scored on the test set (`results/test_access_log.jsonl`); on 25 October, after the validation results (0.896 against 0.918) and those test runs were known, the team returned to the plan (`final2_tfidf_features`, selected on validation). The choice was therefore not blind, as `09-limitations` states.

## Check yourself

Q: Why can a new feature fail to improve a model that already uses TF-IDF?
A: The information may already be in the words. If "cover-up" matters, TF-IDF has it as a word, so a count of the same word adds almost nothing new.

Q: Why do we count certainty words per 1,000 words and not just count them?
A: A long article contains more words of every kind. Counting per 1,000 words makes a short and a long article comparable.

Q: Why does MATTR use windows of 100 words instead of the whole text?
A: The share of different words falls as a text gets longer, so over the whole text it would mostly measure length. Equal windows compare equal-sized pieces.

Q: Why can adding more features make a model worse?
A: With many more columns than articles the model can memorise single articles. Overlapping columns also make the weights unstable, and style habits of single websites do not carry over to new ones.

Q: What would go wrong if the common scale for the style numbers were worked out from validation or test articles too?
A: Information about those articles would leak into training and the scores would look better than they are.
