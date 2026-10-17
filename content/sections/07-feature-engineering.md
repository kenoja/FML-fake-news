---
title: Feature engineering
part: Model building
hypothesis: Phrases (n-grams), certainty and hidden-truth words (epistemic markers) and style cues each carry some signal about the kind of website, but the hidden-truth words and ALL-CAPS share will separate the classes most clearly in training data.
---
> **Summary in 3 lines**
> We built three feature groups: n-grams, epistemic markers, style cues.
> In training data, hidden-truth words differ most between classes (Cohen's d 0.85).
> Careful: this only describes; word lists ignore negation, style can be a website habit.

## Goal and hypothesis

The words-only model (M1) sees which words occur. The course feedback suggested adding "a few other things" and checking whether they help. We chose three **feature groups** (a feature is one number that describes a document; a group is a family of related numbers):

1. **N-grams**: TF-IDF over single words, pairs and triples of words.
2. **Epistemic markers**: how often a text uses doubt words, certainty words, "hidden truth" phrases and debunking words.
3. **Style**: capital letters, punctuation, quotation marks, sentence length and repetition.

Hypothesis: each group adds a little beyond plain words, the hidden-truth phrases and the ALL-CAPS share differ most between the classes, and sentence length differs least. This section builds and describes the features. Whether they *improve the classifier* is decided later by the ablation (adding one group at a time) in section 08, so nothing here claims an improvement.

## Explanation

All feature code takes **only the cleaned article text**. A feature can therefore never read the website name or the label. The numeric groups are followed by a `StandardScaler`, which subtracts the mean and divides by the spread so that a feature in "words per sentence" and one in "percent" are on one scale. The scaler, like the TF-IDF vocabulary, is fitted on the training split only (see section 03 for the frozen split, section 04 for TF-IDF).

### Group 1: n-grams (`src/fml/features/ngrams.py`)

<!-- cell: src/fml/features/ngrams.py#build_ngram_vectorizer -->

An **n-gram** is n words in a row: "inside job" is a 2-gram, "no evidence of" a 3-gram. **TF-IDF** gives a term a high weight if it is frequent in this document and rare across all documents. Single words cannot see "no evidence of"; trigrams can. This is the course feedback's own example. Settings (`ngram_range`, `ngram_min_df`, `ngram_max_features`) are in `config/project.yaml` under `features:`. Requiring an n-gram to occur in at least `ngram_min_df` training documents keeps the vocabulary from exploding with one-off triples.

*Literature reason.* Wang and Manning (2012) showed that simple word and bigram features with a linear model are a strong baseline for text classification, so n-grams are the natural first extension. Rashkin et al. (2017) found that word choice (for example superlatives and subjective words) separates reliable from unreliable news.

### Group 2: epistemic markers (`src/fml/features/epistemic.py`)

<!-- cell: src/fml/features/epistemic.py:27-41 -->
<!-- cell: src/fml/features/epistemic.py#EpistemicMarkers -->

"Epistemic" means "about knowledge and certainty". The module holds a small, documented word list (the constant `LEXICON`) in four categories and returns, per category, the number of hits **per 1,000 words** (`features.per_n_words`). Normalising by length matters: a 3,000-word article would otherwise always have three times the hits of a 1,000-word one. A test doubles a text by concatenation and checks that the rates do not change.

| Category | Examples | Idea |
|---|---|---|
| hedges | perhaps, allegedly, possibly, reportedly | the writer signals doubt or distance |
| boosters | undeniable, clearly, obviously, undoubtedly | the writer signals certainty |
| hidden truth | cover-up, inside job, wake up, mainstream media, they don't want you to know | the text claims something is being concealed |
| debunking | debunked, no evidence, conspiracy theory | the text argues against a claim |

*Literature reason.* Hedges and boosters follow the distinction used by Islam et al. (2020) for unreliable news. Marini and Jezek (2024) describe conspiracy discourse through evidentiality and certainty markers and through vocabulary that claims a hidden truth. Rashkin et al. (2017) found that unreliable news used more hedges and subjective words. The evidence is **mixed**: Fong et al. (2021) report that certainty was lower for conspiracy influencers but higher for their followers. So the certainty features are a hypothesis to test, not a fact. Our word list is a short hand-picked subset, not a copy of any published list.

#### How much of the data the word list covers, and does negation handling matter?

Two questions about the list: how many training documents contain at least one of its words (**coverage**), and does ignoring negation hurt? `lexicon_coverage` counts, per category and class, the exact list size (`list_size`) and the share of **training** documents with at least one hit (`share_plain`). The negation-aware count treats a hit as *negated* when "not", "no", "never" or a word ending in "n't" stands within 3 tokens before it ("not proven" is negated), and drops it from the count. A window of 3 tokens is crude (it ignores punctuation and clause boundaries). Nothing is learned; this only counts.

<!-- cell: src/fml/features/epistemic.py#count_markers_negation -->
<!-- cell: src/fml/features/epistemic.py#lexicon_coverage -->
<!-- output: results/tables/epistemic_coverage.csv -->

To see whether negation handling changes the classifier, `ablation.epistemic_negation` compares **validation** F1 of "words + epistemic" with plain and with negation-aware counts: same C grid and class weights, scaler fitted on the training split, test split never read. The verdict uses the paired website-bootstrap confidence interval (a range that would contain the true difference in 95 of 100 repeated samples of websites): if the interval contains 0 the difference is within noise and we do not call it an improvement. Read the list sizes, coverage shares, F1 values, delta and interval from the table above and from:

<!-- output: results/metrics.json#ablation.epistemic_negation -->
<!-- output: results/tables/epistemic_negation.csv -->

### Group 3: style (`src/fml/features/style.py`)

<!-- cell: src/fml/features/style.py#StyleFeatures -->
<!-- cell: src/fml/features/style.py#mattr -->

Six numbers: the share of ALL-CAPS words (words of at least two letters, so "I" and "A" do not count), exclamation marks and question marks per 1,000 words, quotation marks per 1,000 words, mean sentence length in words, and the **moving-average type-token ratio** (MATTR). A type-token ratio is the number of different words divided by the number of words; it measures repetition. Plain ratios fall with text length, so MATTR averages the ratio over every 100-word window (`features.mattr_window`) and is comparable between long and short articles.

*Literature reason.* Horne and Adali (2017) found that fake-news bodies used more all-caps words, fewer quotes and more repetition. Perez-Rosas et al. (2018) found punctuation and readability features strong inside one domain but poor across domains, which is why our website-held-out split is the right test for them.

### Why these groups, and what we expected (as in the report, section 5.3)

*N-grams:* simple word and bigram features with a linear model are a strong text-classification baseline (Wang and Manning, 2012), so we expected a small gain, the largest of the three (H2). *Epistemic markers:* conspiracy discourse is marked by evidentiality, certainty and hidden-truth vocabulary (Marini and Jezek, 2024); we expected more hidden-truth words on conspiracy websites, while the direction for certainty is mixed. *Style:* fake-news bodies use more ALL-CAPS words, fewer quotes and more repetition (Horne and Adali, 2017), but punctuation and readability features generalise poorly across domains (Perez-Rosas et al., 2018), so we expected a small gain at best (H4). **Untuned settings:** `ngram_min_df` (3 training documents) and `ngram_max_features` (200,000) were fixed in `config/project.yaml` before the ablation and never varied.

### What the features look like

The table below is from `results/tables/feature_examples.csv`. The two snippets are **invented** sentences (not article text) written to show what each column reacts to. Snippet A is full of hidden-truth vocabulary and capitals; snippet B uses a hedge and a debunking phrase:

> A (invented): "WAKE UP! The mainstream media will never tell you the hidden truth. Clearly it was an inside job, and the cover-up goes on. Why do they hide it?"
>
> B (invented): "Investigators reported that there is no evidence of a cover-up. The claim has been debunked, although some details are possibly still disputed."

<!-- output: results/tables/feature_examples.csv -->

Note that B still gets a nonzero **hidden-truth** value because it contains "cover-up", although the sentence *denies* a cover-up. That is the negation weakness described below.

### What the features look like in the training data

For each numeric feature, `results/tables/feature_summary.csv` gives the mean per class **on the training split only** (and the number of training documents used) and the **standardised difference** (Cohen's d: the difference of the two class means divided by the pooled standard deviation; about 0.2 is small, 0.5 medium, 0.8 large; positive means higher for conspiracy-site articles). It is descriptive: it does not prove that a feature helps a classifier.

<!-- output: results/tables/feature_summary.csv -->
<!-- output: results/figures/style_features_by_class.png | Style features by class, training documents only -->

Reading the table: the hidden-truth rate and the ALL-CAPS share show the largest differences between the classes, boosters and exclamation marks are smaller, and sentence length and quotation density are close to zero. These are averages over many websites; one website with an unusual habit can drive a mean, and the classes differ in the websites they contain, so a difference here may be a website habit and not a conspiracy habit. We only describe the training split on purpose, so that nothing about validation or test documents shapes how we describe or choose features.

### What we did not do

These were considered and left out deliberately:

- **Emotion scores (the proposal's group G4: anger, fear, disgust, sadness).** Dropped by team decision, for two reasons recorded in `docs/decisions.md`: cost (it needs a pre-trained emotion model run over every full article), and the course feedback's warning that emotional language is not automatically conspiracy language, especially when reporting on something like 9/11, where mainstream coverage is emotional too. We therefore expected weak separation for a high price. Not trying it means we cannot say whether emotion would have helped.
- **LIWC** (a commercial word-category dictionary). It needs a paid licence, so it is not reproducible for the team or for readers.
- **Topic models.** They largely overlap with the n-gram words and are harder to explain to teammates.
- **Fine-tuning a transformer.** The course feedback says it is not needed; the comparison is one logistic regression against one zero-shot LLM.

## What could go wrong

- **Negation and scope (the known lexicon weakness; Chen 2026).** A word list counts words, not meaning. "There is no evidence of cover-up" still counts "cover-up" as hidden-truth vocabulary, as the test `test_negation_caveat_known_weakness` shows on purpose. Mainstream articles that *debunk* a conspiracy theory use the same vocabulary as articles that promote it. N-grams ("no evidence of") help a little, but the error analysis should look for this.
- **Word lists are short and ours.** They miss synonyms and spelling variants and reflect our guesses. Terms like "myth" or "maybe" are ambiguous. Different lists give different numbers.
- **Style can be a website habit.** A blog format or a house style (many ALL-CAPS headlines, many quotes) identifies the website, not the belief. Perez-Rosas et al. (2018) saw such features fail across domains. The split by website tests exactly this.
- **Cleaning decides what is counted.** Boilerplate removal, URLs and the sentence splitter (full stops, `!`, `?` and line breaks) change sentence length; abbreviations such as "U.S." create extra sentence ends.
- **Mixed evidence.** Certainty markers may point the opposite way from what we expect (Fong et al. 2021). A result against our hypothesis is a result.
- **Descriptive is not predictive.** A big standardised difference in training data can disappear on new websites; only the validation and test scores say.

## Technical infrastructure

- `src/fml/features/__init__.py`: `scaled(group)` returns a numeric group followed by a `StandardScaler`; `NUMERIC_GROUPS` lists them.
- `src/fml/features/ngrams.py#build_ngram_vectorizer`, `epistemic.py#EpistemicMarkers`, `style.py#StyleFeatures`: scikit-learn transformers with `fit`/`transform` that take a column of cleaned text. The two numeric transformers learn nothing themselves; the scaler after them does.
- `src/fml/features/summary.py#feature_summary`, `features_stage`: the table, examples and figure, computed from the training rows of `pipeline.analysis_frame()` (the off-topic-filtered set). The stage runs inside `fml run-all` as stage `features`.
- Settings (`per_n_words`, `ngram_range`, `ngram_min_df`, `ngram_max_features`, `mattr_window`): `config/project.yaml`, block `features`.
- Outputs: `results/tables/feature_summary.csv`, `results/tables/feature_examples.csv`, `results/figures/style_features_by_class.png`.
- Tests: `tests/test_features.py` (length normalisation, ALL-CAPS ratio, lexicon counts, empty-text safety, the negation caveat, scaler and vocabulary fitted on train only, summary uses training rows only).

<!-- cell: src/fml/features/summary.py#feature_summary -->
