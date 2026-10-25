---
title: TF-IDF plus extra features
part: Methods
hypothesis: Adding features about certainty and writing style improves the TF-IDF model.
---
## Explained simply

**The idea.** TF-IDF only sees single words. Do a few hand-made clues help on top? A clue is called a **feature**: one number that describes an article. We tried three families, each backed by earlier research.

1. **Phrases (n-grams of one to three words).** "Inside job" or "no evidence of" mean more than their single words (Wang and Manning, 2012).
2. **Certainty and doubt words.** Writers show how sure they are through their words, as in "I know" against "I guess". Our short lists hold doubt words ("perhaps"), certainty words ("clearly"), hidden-truth words ("cover-up") and debunking words ("no evidence"). Evidence is mixed (Fong et al., 2021).
3. **Style.** Share of CAPITAL words, exclamation and question marks, quotation marks, sentence length and repetition (Horne and Adali, 2017; Perez-Rosas et al., 2018).

**Counting per 1,000 words.** An invented text of 100 words with 3 certainty words scores 30 per 1,000 words. A text of 200 words with the same 3 words scores 15. A long text does not look more certain just because it is long. We then put every number on one common scale, learned on the training articles only, like converting miles and kilometres to one unit.

**Why MATTR for repetition?** The simple idea is the share of different words in a text. But long texts must repeat words, so that share falls as a text grows, and it would measure length instead of style. MATTR (moving-average type-token ratio) fixes this: take every window of 100 words, compute the share of different words in it, and average over all windows. Every text is judged in pieces of the same size.

**The comparison table (an "ablation").** We compare five models on the same articles and split: TF-IDF alone, TF-IDF plus each family, and TF-IDF plus all three. Each row is a full model with its own best regularisation strength (C). The last column is the change in validation F1 against TF-IDF alone: plus means better, minus worse. The table explains what each family contributes; it is not a model choice. The final model is TF-IDF plus all three, as planned.

**Our verdict.** The three families together do not help: the table shows a lower F1 than TF-IDF alone. Hypothesis H2 expected an improvement, and on validation it is not supported.

**Why more clues can make it worse.** Suppose the model must learn from 1,768 articles but gets hundreds of thousands of phrase columns. It can memorise single articles. Many columns also repeat each other, which makes weights unstable. And a style habit may belong to one website, not to a belief, so it fails on new websites.

## The code

The five models differ only in which feature families are switched on.

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

Each model is tuned on validation, then the rows are compared with plain TF-IDF.

<!-- cell: src/fml/ablation.py#run_ablation -->

<!-- output: results/tables/ablation.csv -->
<!-- output: results/figures/ablation_f1.png | Validation F1 of the five models -->
<!-- output: results/metrics.json#tfidf_features.validation -->
<!-- output: results/tables/epistemic_coverage.csv -->
<!-- output: results/tables/feature_summary.csv -->

## Report

The third analysis asks whether hand-made features improve the TF-IDF model (hypothesis H2: adding the three feature groups improves TF-IDF). We added three groups, each motivated by earlier work. N-grams of one to three words capture phrases such as "inside job"; word and bigram features are a strong baseline for text classification (Wang and Manning, 2012). Epistemic markers are counts per 1,000 words of four short hand-made lists (hedges, boosters, hidden-truth phrases, debunking words), following work that ties conspiracy discourse to certainty and concealment (Marini and Jezek, 2024; Islam et al., 2020), although evidence on certainty is mixed (Fong et al., 2021). Style features are capitals, punctuation, quotation marks, sentence length and repetition (Horne and Adali, 2017); they generalise poorly across domains (Perez-Rosas et al., 2018).

The numeric groups were standardised with training statistics only. We compared five logistic regressions on the same split: TF-IDF alone, with each group, and with all three, each with C chosen from the same grid by validation F1 and balanced class weights. The table is an ablation, not a model selection: the final model is TF-IDF plus all three groups, as planned. All numbers in this section are validation numbers.

| Model | Validation F1 | Change against TF-IDF alone |
|---|---|---|
| TF-IDF alone | 0.918 | reference |
| + n-grams (1-3) | 0.920 | +0.002 |
| + certainty words | 0.930 | +0.013 |
| + style | 0.909 | -0.009 |
| + all three groups | 0.896 | -0.022 |

<!-- output: results/figures/ablation_f1.png | Validation F1 of the five models -->

The verdict on H2 is no: the three feature groups together lower the validation F1 from 0.918 to 0.896. With 376 validation articles, differences of one or two hundredths can be chance, so even the certainty gain is only a weak hint.

Why can the full combination be worse than TF-IDF alone? The table cannot prove a cause, and we did not test any of the following three explanations. First, the full model has far more columns than articles (146,172 columns against 1,768 training articles), so it could fit single articles. Second, the groups overlap with words TF-IDF already holds. Third, style habits may belong to single websites, which would fit style alone scoring below TF-IDF alone.

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
