---
title: TF-IDF plus extra features
part: Methods
hypothesis: Adding features about certainty and writing style improves the TF-IDF model.
---
## Explained simply

**The idea.** TF-IDF only sees single words. The course feedback asked whether a few hand-made clues help on top. A clue is called a **feature**: one number that describes an article. We tried three families of clues, each backed by earlier research.

1. **Phrases (n-grams of one to three words).** "Inside job" or "no evidence of" mean more than "inside", "job" and "evidence" alone. Earlier work found that words plus word pairs make a strong baseline for sorting texts (Wang and Manning, 2012).
2. **Certainty and doubt words (epistemic markers).** "Epistemic" means "about knowing". We made short lists of doubt words ("perhaps"), certainty words ("clearly"), hidden-truth words ("cover-up") and debunking words ("no evidence"). Research links conspiracy writing to certainty and to claims of a hidden truth, but the evidence is mixed (Marini and Jezek, 2024; Fong et al., 2021). The lists are ours, short, and count words only.
3. **Style.** Share of CAPITAL words, exclamation and question marks, quotation marks, sentence length and how much a text repeats itself. Fake-news texts were found to shout and repeat more (Horne and Adali, 2017), but such clues travelled badly between sources (Perez-Rosas et al., 2018).

**A fair test.** We compare five models, all with the same articles, the same split and the same list of regularisation values: TF-IDF alone, TF-IDF plus each family, and TF-IDF plus all three. Each family is built once from the training articles. Numbers such as "capital words per article" are **standardised** first (put on one common scale), and that scale is also learned on the training articles only. We judge by F1 on the validation articles, never on test.

**How we read the result.** A family counts as "above threshold" only if F1 rises by at least 0.01. A smaller change could be luck: validation holds a few hundred articles, and one or two articles can move F1 by that much. We keep the plain TF-IDF model unless a family is clearly above it.

**Why extra clues can add little.**
- The words already carry the information. If "cover-up" is a telling clue, TF-IDF has it already as a word.
- Our short lists miss articles that say the same thing differently, and ignore meaning: "there was no cover-up" still counts "cover-up".
- A style habit can belong to one website, not to a belief.

**What we did not do.** We added no emotion scores: they need a further pre-trained model, and emotional wording is normal in reports on a terror attack.

**Verdict.** On validation only the certainty words passed our 0.01 threshold; on test the gain (0.582 to 0.610) is within chance, so the features do not help reliably.

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
<!-- output: results/metrics.json#tfidf_features.chosen_lr -->
<!-- output: results/tables/epistemic_coverage.csv -->
<!-- output: results/tables/feature_summary.csv -->

## Report

The third analysis asks whether hand-made features improve the TF-IDF model. We added three groups, each motivated by earlier work. N-grams of one to three words capture phrases such as "inside job"; word and bigram features are a strong baseline for text classification (Wang and Manning, 2012). Epistemic markers are counts per 1,000 words of four short hand-made lists (hedges, boosters, hidden-truth phrases, debunking words), following work that ties conspiracy discourse to certainty and concealment (Marini and Jezek, 2024; Islam et al., 2020), although evidence on certainty is mixed (Fong et al., 2021). Style features are capitals, punctuation, quotation marks, sentence length and repetition; fake-news bodies use more capitals and repetition (Horne and Adali, 2017), but such features generalise poorly across domains (Perez-Rosas et al., 2018).

We compared five logistic regressions on the same cleaned articles and split: TF-IDF alone, with each group, and with all three. Each feature matrix was built once on the training articles, numeric groups were standardised with training statistics only, and every model chose the regularisation strength (C) and the class weight from the same grid by validation F1 of the conspiracy class. A group counts as above threshold if it raises validation F1 by at least 0.01; otherwise plain TF-IDF is kept. With 376 validation articles, smaller differences cannot be told from chance.

Validation F1 was 0.914 for TF-IDF alone and 0.916 with n-grams, a gain of 0.002. Adding the certainty words gave 0.930, a gain of 0.017, the only change above our threshold. Style features gave 0.913, a change of -0.001. All three groups together scored 0.896, 0.018 below plain TF-IDF. Only the certainty words passed the threshold, so the final tested model contains certainty words only. On validation only the certainty words passed our 0.01 threshold; on test the gain (0.582 to 0.610) is within chance (McNemar p = 0.332), so the features do not help reliably. The certainty words gave a small gain that may be luck, and combining everything made the model worse, probably because many weak clues add noise. TF-IDF with certainty words is the only model above the threshold and is chosen on validation, but only the later test comparison can show whether its small edge holds on new websites.

<!-- output: results/figures/ablation_f1.png | Validation F1 of the five models -->

Three reasons explain why the groups add so little. First, the information is already in the words: "cover-up" is a TF-IDF feature as soon as it is a word, and trigrams largely repeat single words while adding many rare columns. Second, the lists reach only part of the data. The hidden-truth list has 11 entries and occurs in 45% of training conspiracy articles against 3% of mainstream ones, a sharp clue that is absent from more than half of the conspiracy articles; the debunking list appears in only 28% of them. Third, the lists count words and ignore meaning, so an article that denies a cover-up still scores as hidden-truth vocabulary, and style habits can belong to one website rather than to a belief.

## Check yourself

Q: Why can a new feature fail to improve a model that already uses TF-IDF?
A: The information may already be in the words. If "cover-up" matters, TF-IDF has it as a word, so a count of the same word adds almost nothing new.

Q: Why do we keep plain TF-IDF unless a group raises F1 by at least 0.01?
A: With a few hundred validation articles, one or two articles can move F1 by that much by chance. A rule with a margin stops us from calling luck an improvement.

Q: Why are the numbers for the style features put on one scale using the training articles only?
A: If the scale used validation or test articles, information about them would leak into training and the scores would look better than they are.

Q: Hidden-truth words differ a lot between the classes in training. Does that guarantee they help the classifier?
A: No. A difference in averages only describes the data. The words may already be in TF-IDF, and the list covers only some articles.
