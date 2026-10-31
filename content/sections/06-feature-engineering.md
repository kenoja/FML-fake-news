---
title: TF-IDF plus extra features
part: Methods
hypothesis: Adding features about certainty and writing style improves the TF-IDF model.
---
## Explained simply

**The idea.** TF-IDF only sees single words. Do a few hand-made clues help on top? A clue is a **feature**: one number that describes an article. We tried three families.

1. **Phrases (n-grams of one to three words).** "Inside job" or "no evidence of" mean more than their single words (Wang & Manning, 2012).
2. **Certainty and doubt words.** Words show how sure a writer is: "I know" against "I guess". Our short lists hold doubt words, called hedges ("perhaps"), certainty words, called boosters ("clearly"), hidden-truth words ("cover-up") and debunking words ("no evidence").
3. **Style.** Share of CAPITAL words, exclamation and question marks, quotation marks, sentence length and repetition (Horne & Adalı, 2017; Pérez-Rosas et al., 2018).

**Counting per 1,000 words.** A long text does not look more certain just because it is long: 3 certainty words in 100 words score 30, in 200 words 15. Every number is then put on one common scale, learned on training articles only.

**Why MATTR for repetition?** The simple share of different words falls as a text grows, so it would measure length. MATTR (moving-average type-token ratio) averages that share over every window of 100 words.

**The comparison table (an "ablation").** An ablation switches one ingredient on or off, like baking a cake with and without sugar. We compare six models on the same split: TF-IDF alone, plus each family, plus all three, and plus certainty words and style without phrases. Each row has its own best regularisation strength (C). The last column is the change in validation F1 against TF-IDF alone.

**Why all three together do worse.** Style lowers the score even on its own; a habit such as many capital letters may belong to one website and not carry over. The phrases add very many columns (146,172 for 1,768 training articles), probably letting the model memorise articles. That is also why a phrase must appear in at least three training articles, a word in two: with two, the phrases fill our cap of 200,000 columns and the score drops further. Certainty words are the only family that helps on its own. Small differences can be luck.

**Why the final model is not the best row.** We planned the model with all three families before any result, and our question is about adding all three. Certainty words alone scored highest, but picking the best row after seeing the table is like changing your bet after the race: the question would bend to fit the result. (The limitations say when this choice was made.)

## The code

The six models differ only in which feature families are switched on.

<!-- cell: src/fml/pipeline.py#ablation_variants -->

Phrases come from a TF-IDF step that reads word runs of length one to three.

<!-- cell: src/fml/features/ngrams.py#build_ngram_vectorizer -->

Words are found after curly apostrophes are made straight.

<!-- cell: src/fml/features/_text.py#normalise_quotes -->

<!-- cell: src/fml/features/_text.py#words -->

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

**Idea.** The third method tests H2: whether hand-made features add what single words miss. Three groups were planned before any result. N-grams are phrases of one to three words such as "no evidence of", a strong baseline for text classification (Wang & Manning, 2012). Epistemic markers are counts per 1,000 words of four short lists of doubt, certainty, hidden-truth and debunking words, following work that ties conspiracy discourse to certainty and concealment (Islam et al., 2020; Marini & Jezek, 2024; but see Fong et al., 2021). Style features are capitals, punctuation, quotation marks, sentence length and word variety (MATTR), which separated fake from real news in earlier work (Horne & Adalı, 2017) but transfer poorly across domains (Pérez-Rosas et al., 2018).

**Setup.** As for TF-IDF; the numeric groups are standardised with training statistics only. Phrases must occur in at least three training articles, single words in at least two, because phrases are far more numerous: with two, the planned model fills the cap of 200,000 phrase columns and its validation F1 falls to 0.891. To explain the result, an ablation compares six models on validation: TF-IDF alone, with each group, with all three, and with certainty words and style but no n-grams.

| Model | Validation F1 |
|---|---|
| TF-IDF alone | 0.918 |
| + n-grams (1-3) | 0.920 |
| + certainty words | 0.930 |
| + style | 0.909 |
| + all three groups | 0.896 |
| + certainty words and style (no n-grams) | 0.913 |

<!-- output: results/figures/ablation_f1.png | Validation F1 of the six models -->

**Validation result.** The planned model with all three groups scored 0.896, below TF-IDF alone (0.918); certainty words alone scored highest (0.930). A difference of one or two hundredths is one or two of the 117 conspiracy articles. H2 concerns the planned combination, and switching to the best single row after seeing the table would bend the hypothesis to the results, so the planned model was kept.

**What it learned: why the features could work, and why they did not.** The features do separate the classes in the training data: hidden-truth phrases such as *cover-up* appear in 44% of conspiracy and 3% of mainstream articles. Four likely reasons, not tested causes, explain why they added nothing. First, too many variables for too few examples: the full model has 146,172 columns for 1,768 training articles, so it can fit single articles, and it chose the weakest regularisation in the grid (C = 100). Second, overlap: TF-IDF already holds most list words, and "truth" is its second-strongest conspiracy word. Third, style habits such as capitals may describe single websites rather than conspiracy writing, and do not carry over to new websites. Fourth, word lists ignore context: debunking words appear in 26% of conspiracy and 10% of mainstream articles, because conspiracy texts argue against debunkers.

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
