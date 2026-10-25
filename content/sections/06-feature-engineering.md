---
title: TF-IDF plus extra features
part: Methods
hypothesis: Adding features about certainty and writing style improves the TF-IDF model.
---
## Explained simply

> **Do the extra features help?** On validation only certainty words pass; on test we cannot show a gain beyond noise. See section 08 for the test numbers.

**A tiny example.** Take an invented text of 100 words that contains 3 certainty words such as "obviously". Its certainty score is 3.0 per 100 words. A text of 200 words with the same 3 words scores 1.5, so a long text does not look more certain just because it is long.

**The idea.** TF-IDF only sees single words. The course feedback asked whether a few hand-made clues help on top. A clue is called a **feature**: one number that describes an article. We tried three families of clues, each backed by earlier research.

1. **Phrases (n-grams of one to three words).** "Inside job" or "no evidence of" mean more than their single words (Wang and Manning, 2012).
2. **Certainty and doubt words.** Writers show how sure they are through their words, as in "I know" against "I guess". Our short lists hold doubt words ("perhaps"), certainty words ("clearly"), hidden-truth words ("cover-up") and debunking words ("no evidence"). Research links conspiracy writing to certainty and hidden truths, but the evidence is mixed (Marini and Jezek, 2024; Fong et al., 2021).
3. **Style.** Share of CAPITAL words, exclamation and question marks, quotation marks, sentence length and repetition (Horne and Adali, 2017; Perez-Rosas et al., 2018).

**A fair test.** We compare five models on the same articles and split: TF-IDF alone, TF-IDF plus each family, and TF-IDF plus all three. Each family is built from the training articles only. Numbers such as "capital words per article" are put on one common scale (this is called **standardising**, like converting miles and kilometres to one unit) with a scale learned on the training articles only. We judge by F1 on the validation articles.

**How we read it.** We set a minimum gain before the test: a family counts as helping only if F1 rises by at least 0.01. A smaller change could be luck, because one or two articles can move F1 that much. Otherwise we keep plain TF-IDF.

**Why extra clues can add little.** The words already carry the information: if "cover-up" is telling, TF-IDF has it. Our short lists ignore meaning, so "there was no cover-up" still counts "cover-up". And a style habit can belong to one website, not to a belief.

We added no emotion scores: emotional wording is normal in reports on a terror attack.

## The code

The five models differ only in which feature families are switched on.

<!-- cell: src/fml/pipeline.py#ablation_variants -->

Phrases come from a TF-IDF step that reads word runs of length one to three.

<!-- cell: src/fml/features/ngrams.py#build_ngram_vectorizer -->

Certainty words are counted per 1,000 words, so a long article does not look more certain just because it is long.

<!-- cell: src/fml/features/epistemic.py#count_markers -->

Style features are computed from the text alone.

<!-- cell: src/fml/features/style.py#style_row -->

Repetition is measured with a sliding window over the words.

<!-- cell: src/fml/features/style.py#mattr -->

Each model is tuned on validation, then the rows are compared with plain TF-IDF.

<!-- cell: src/fml/ablation.py#run_ablation -->

A family counts as helping only with a gain of at least 0.01.

<!-- cell: src/fml/ablation.py#verdict -->

<!-- output: results/tables/ablation.csv -->
<!-- output: results/figures/ablation_f1.png | Validation F1 of the five models -->
<!-- output: results/metrics.json#tfidf_features.validation -->
<!-- output: results/tables/epistemic_coverage.csv -->
<!-- output: results/tables/feature_summary.csv -->

## Report

The third analysis asks whether hand-made features improve the TF-IDF model. We added three groups, each motivated by earlier work. N-grams of one to three words capture phrases such as "inside job"; word and bigram features are a strong baseline for text classification (Wang and Manning, 2012). Epistemic markers are counts per 1,000 words of four short hand-made lists (hedges, boosters, hidden-truth phrases, debunking words), following work that ties conspiracy discourse to certainty and concealment (Marini and Jezek, 2024; Islam et al., 2020), although evidence on certainty is mixed (Fong et al., 2021). Style features are capitals, punctuation, quotation marks, sentence length and repetition; fake-news bodies use more capitals and repetition (Horne and Adali, 2017), but such features generalise poorly across domains (Perez-Rosas et al., 2018).

We compared five logistic regressions on the same cleaned articles and split: TF-IDF alone, with each group, and with all three. Each feature matrix was built once on the training articles, numeric groups were standardised with training statistics only, and every model chose the regularisation strength (C) and the class weight from the same grid by validation F1 of the conspiracy class. A group passes the threshold if it raises validation F1 by at least 0.01; otherwise plain TF-IDF is kept. With 376 validation articles, smaller differences cannot be told from chance.

The table lists the validation F1 of every model and its change against TF-IDF alone. Only the certainty words reach our threshold.

| Model | Validation F1 | Change against TF-IDF alone |
|---|---|---|
| TF-IDF alone | 0.914 | reference |
| + n-grams (1-3) | 0.916 | +0.002 |
| + certainty words | 0.930 | +0.017 |
| + style | 0.894 | -0.020 |
| + all three groups | 0.887 | -0.027 |

The final tested model therefore contains the certainty words alone. On test its gain (F1 0.610 against 0.582) was within noise (p = 0.332). Style and the full combination made the model worse, probably because many weak clues add noise.

<!-- output: results/figures/ablation_f1.png | Validation F1 of the five models -->

Three reasons explain why the groups add so little. First, the information is already in the words: "cover-up" is a TF-IDF feature as soon as it is a word. Second, the lists reach only part of the data: the hidden-truth list occurs in 45% of training conspiracy articles against 3% of mainstream ones, but is absent from more than half of the conspiracy articles. Third, the lists count words and ignore meaning, so an article that denies a cover-up still scores as hidden-truth vocabulary.

## Check yourself

Q: Why can a new feature fail to improve a model that already uses TF-IDF?
A: The information may already be in the words. If "cover-up" matters, TF-IDF has it as a word, so a count of the same word adds almost nothing new.

Q: What would go wrong if we accepted a feature group for any gain at all, however small?
A: With a few hundred validation articles, one or two articles can move F1 by that much by chance. A rule with a margin stops us from calling luck an improvement.

Q: What would go wrong if the common scale for the style numbers were worked out from validation or test articles too?
A: If the scale used validation or test articles, information about them would leak into training and the scores would look better than they are.

Q: Why does a clear difference in hidden-truth words between the two kinds of website not guarantee that they help the classifier?
A: A difference in averages only describes the data. The words may already be in TF-IDF, and the list covers only some articles.
