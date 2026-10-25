# Conspiracy or mainstream? Classifying 9/11 articles with logistic regression and an LLM

## Question and hypotheses

### Executive summary

We asked whether the text of a full 9/11 article reveals if it was published on a conspiracy-oriented or a mainstream website. Three logistic regressions (on word counts, on TF-IDF weights, and on TF-IDF plus three hand-made feature groups) and one low-cost language model, Claude Haiku 5.5, were compared on 410 held-out test articles. The language model reached the highest F1 score for the conspiracy class, 0.795 (95% confidence interval 0.723 to 0.857), against 0.567 for word counts, 0.598 for TF-IDF and 0.560 for TF-IDF with features. Weighting words helped slightly (H1), the hand-made features did not (H2), and the language model clearly differed from the best logistic regression (H3). All models found fewer than seven in ten conspiracy articles on the test websites, so none is reliable enough for automatic flagging.

### Introduction

The 9/11 attacks are a lasting subject of online conspiracy content, and conspiracy-oriented websites publish far more text than can be checked by hand (Miani, Hills & Bangerter, 2022). Automatic filters are therefore an obvious idea, and earlier work has shown that conspiracy language differs measurably from ordinary language (Fong et al., 2021). This study asks whether the text of a full article about the 9/11 attacks reveals if it was published on a conspiracy-oriented or on a mainstream website, using machine-learning methods for text classification. The articles come from the LOCO corpus, and the label describes the website, not the single article.

We compare four methods. Analyses 1 to 3 are logistic regression on different inputs: word counts, TF-IDF weights, and TF-IDF weights plus three hand-made feature groups (word sequences of up to three words, certainty and hedging words, and style measures). Analysis 4 is one low-cost large language model (LLM) used without any training on our data. The set is deliberately small: a single simple classifier with changing inputs shows which input matters, whereas many classifiers would blur it. Other classifiers, fine-tuning an LLM and emotion features were left out for reasons of scope and reliability.

Table 1 gives the four methods at a glance.



| method | input | reads | trained_on_our_data | settings | runtime | cost |
|---|---|---|---|---|---|---|
| 1. Word counts + logistic regression | word counts | first 3,000 words of each article | yes | C from [0.01, 0.1, 1, 10, 100], class weight balanced, chosen on validation | 26 seconds | free (runs locally) |
| 2. TF-IDF + logistic regression | TF-IDF weights | first 3,000 words of each article | yes | C from [0.01, 0.1, 1, 10, 100], class weight balanced, chosen on validation | 2 seconds | free (runs locally) |
| 3. TF-IDF + features + logistic regression | TF-IDF weights, n-grams, certainty words, style | first 3,000 words of each article | yes | C from [0.01, 0.1, 1, 10, 100], class weight balanced, chosen on validation | 57 seconds | free (runs locally) |
| 4. LLM (claude-haiku-5-5) | the article text itself, with a written question | first 3,000 words of each article | no | prompt v1 of 3, default sampling | about 1.0 seconds per article | about $0.26 per 1,000 articles |



All three logistic regressions ran on a laptop-class machine (see the run information in the results folder; 4 CPUs, no graphics card) in about a minute each. Only the LLM needs an internet connection and a paid interface; its cost is shown in the table.

We test three hypotheses. **H1:** TF-IDF weights give a higher conspiracy-class F1 than plain word counts, because distinctive words carry more information than common ones. **H2:** adding the three feature groups (word sequences, certainty words, style) to TF-IDF raises the F1, because they capture how conspiracy texts argue. **H3:** the LLM performs about as well as the best logistic regression, or differs from it; no direction is assumed. We also expect plain word counts to beat the trivial rule "always guess mainstream", because the two kinds of website tend to use different words; this is a sanity check, not a hypothesis.

All methods are judged by precision, recall and F1 for the conspiracy class, because most articles are mainstream and the share of correct answers would reward a model that never finds a conspiracy article. A hypothesis counts as supported only if the confidence interval and the significance test agree with the direction of the difference.

## Data and cleaning

The study uses the 9/11 subset of the LOCO corpus (Miani, Hills & Bangerter, 2022), a collection of web pages from websites rated as conspiracy-oriented or mainstream. We use the full articles of this subset, 3,146 documents from 39 conspiracy and 79 mainstream websites. Each document carries the label of its website. The label therefore describes the publisher and not the single text: a neutral news item on a conspiracy-oriented website is labelled conspiracy, and a mainstream article that discusses a conspiracy theory is labelled mainstream. The texts are not redistributed; they are kept in a private data store, and this report shows only counts and short excerpts. The work was done in Python with pandas and scikit-learn, and the language-model classifier is Claude Haiku 5.5, used through the Anthropic API.

Cleaning matters because a classifier takes the easiest signal that separates the classes, whether or not it is the intended one. This shortcut learning, a form of target leakage, arises when website names, repeated boilerplate, topic differences or copies of the same article reveal the label without any property of the text itself. All cleaning was therefore applied before any model was fitted, and all four analyses use the same cleaned documents. First, documents that do not mention the attacks in their title or text were removed (3,146 to 2,608), because a model could otherwise learn the difference between topics instead of the difference between kinds of websites. Second, near-duplicates were removed (2,608 to 2,554): pairs of documents with a cosine similarity of their TF-IDF vectors of at least 0.9. Such copies are common between websites and would let a model meet a near copy of a test article during training. For a pair across the training set and another set, the copy in the other set was dropped; for a pair of validation and test documents, the validation copy was kept, so that the test set is changed as little as possible. Inside one set, the copy with the lower identifier was kept. Third, web addresses, e-mail addresses, website names and lines repeated across many documents of one website were deleted from the texts. This reduces but does not remove website-specific style, which remains a limitation. The table shows the number of documents before and after each of the first two steps.



| step | removed | remaining | removed_conspiracy | removed_mainstream | remaining_conspiracy | remaining_mainstream |
|---|---|---|---|---|---|---|
| all documents | 0 | 3146 | 0 | 0 | 939 | 2207 |
| off-topic articles removed | 538 | 2608 | 162 | 376 | 777 | 1831 |
| near-duplicates removed | 54 | 2554 | 22 | 32 | 755 | 1799 |

*Documents before and after each cleaning step*



After cleaning, 2,554 documents remain: 755 from conspiracy-oriented and 1,799 from mainstream websites. Roughly three in ten articles are conspiracy, so the classes are unbalanced. Document length was checked to decide whether very short texts need special treatment. After cleaning, the shortest document has 102 words and the median is 622 words, so no minimum length was applied.

## Splitting by website

The documents were divided into a training set, a validation set and a test set with target shares of 70%, 15% and 15%. The division was made by website: all documents of one website were assigned to the same set, so that no website appears in more than one set. This prevents a model from being rewarded for recognising the layout, vocabulary or topics of a website it has already seen, and it mimics the practical situation of classifying articles from unknown websites. Website names are removed from the texts for a different reason: the grouped split stops a model from memorising a website, whereas removing the names stops it from using the name as a shortcut to the label. The division was stratified by label: conspiracy and mainstream websites were assigned separately, from the largest to the smallest website, to the set that was furthest short of its target share. A fixed random seed resolved ties. The assignment was made once and stored in a file that was never changed. The table shows the numbers of websites and documents in each set of the final data.



| split | class | websites | documents | share_of_documents |
|---|---|---|---|---|
| train | conspiracy | 25 | 542 | 0.212 |
| train | mainstream | 54 | 1226 | 0.48 |
| validation | conspiracy | 6 | 117 | 0.046 |
| validation | mainstream | 13 | 259 | 0.101 |
| test | conspiracy | 6 | 96 | 0.038 |
| test | mainstream | 12 | 314 | 0.123 |

*Websites and documents per set and class*



Because websites cannot be divided, the shares only approximate the targets. The training, validation and test sets contain 25, 6 and 6 conspiracy websites (37 in total) and 54, 13 and 12 mainstream websites (79 in total); no website is in more than one set. Validation and test sets contain only a few conspiracy websites, so scores depend to some extent on which websites ended up there. All model settings were chosen on the validation set, and the test set was used only for the final comparison.

## Word counts

The first analysis represents each article as a bag of words: a vector with the number of times each word occurs, ignoring order. Words were counted on the training articles only, and words occurring in fewer than two training articles were dropped. The model is a logistic regression with balanced class weights (section 1 gives the reasons for this choice and for leaving out other classifiers). Its regularisation strength (C) was chosen from {0.01, 0.1, 1, 10, 100} by the F1 score of the conspiracy class on the validation set. Precision is the share of articles predicted as conspiracy that are conspiracy; recall is the share of conspiracy articles that are found; F1 is their harmonic mean. All numbers in this section are validation numbers.

The smallest value, C = 0.01, scored best, with a precision of 0.828, a recall of 0.906 and an F1 score of 0.865. Larger values scored lower (F1 0.797 at C = 100), the typical sign of overfitting: with raw counts the model starts to rely on rare words that only describe the training articles. The best value lies at the edge of the grid, so a still smaller C might be slightly better; we did not widen the grid after seeing the results. Of the 117 conspiracy articles in validation the model found 106, and of the 259 mainstream articles it called 22 conspiracy.



| C | precision | recall | f1 | chosen |
|---|---|---|---|---|
| 0.01 | 0.828 | 0.906 | 0.865 | True |
| 0.1 | 0.821 | 0.863 | 0.842 | False |
| 1.0 | 0.811 | 0.846 | 0.828 | False |
| 10.0 | 0.797 | 0.803 | 0.8 | False |
| 100.0 | 0.807 | 0.786 | 0.797 | False |





![Word counts: validation confusion matrix](../results/figures/word_counts_validation_confusion.png)



The words with the largest weights are informative, but not all are about content. Words such as "posted", "am" and "article" among the strongest conspiracy words look like habits of how a website presents its articles. Mainstream words such as "killed", "says" and "anniversary" fit factual news reporting. The model therefore partly recognises websites and not only arguments.



| word | weight |
|---|---|
| this | 0.172 |
| posted | 0.159 |
| truth | 0.157 |
| am | 0.142 |
| former | 0.139 |
| judge | 0.123 |
| 11 | 0.113 |
| cia | 0.112 |
| fbi | 0.106 |
| being | 0.101 |
| all | 0.099 |
| article | 0.096 |
| clinton | 0.096 |
| months | 0.094 |
| false | 0.09 |
| weapons | 0.089 |
| media | 0.085 |
| israeli | 0.085 |
| up | 0.084 |
| 2019 | 0.084 |
| wtc | 0.083 |
| reported | 0.083 |
| fact | 0.082 |
| resolution | 0.081 |
| entry | 0.079 |





| word | weight |
|---|---|
| after | -0.114 |
| killed | -0.114 |
| says | -0.101 |
| response | -0.1 |
| life | -0.095 |
| when | -0.094 |
| anniversary | -0.092 |
| but | -0.09 |
| tower | -0.09 |
| whether | -0.089 |
| omar | -0.089 |
| among | -0.088 |
| security | -0.083 |
| nearly | -0.083 |
| much | -0.081 |
| where | -0.079 |
| york | -0.078 |
| september | -0.076 |
| attacks | -0.075 |
| said | -0.074 |
| 000 | -0.073 |
| airlines | -0.072 |
| talks | -0.072 |
| first | -0.071 |
| long | -0.07 |



Hypothesis H1 expects TF-IDF to beat these counts. This section provides the reference value for that comparison, an F1 score of 0.865; the comparison is made in the next section.

## TF-IDF

The second analysis keeps everything from the word-count analysis (articles, split, vocabulary, balanced logistic regression, grid for the regularisation strength C) and replaces raw counts by TF-IDF weights. Term frequencies were damped with a logarithm, article vectors were scaled to unit length, and the inverse document frequencies were computed on the training articles only. C was again chosen from {0.01, 0.1, 1, 10, 100} by the validation F1 score of the conspiracy class. All numbers in this section are validation numbers.

The value C = 10 scored best, with a precision of 0.930, a recall of 0.906 and an F1 score of 0.918. Compared with word counts (F1 0.865), recall stayed the same and precision rose from 0.828 to 0.930: the number of mainstream articles wrongly called conspiracy fell from 22 to 8. Unlike with raw counts, larger C helped: because TF-IDF vectors have a fixed length, the model can use larger weights without being dominated by long articles. The chosen value lies inside the grid, and C = 100 scored almost the same (F1 0.909).



| C | precision | recall | f1 | chosen |
|---|---|---|---|---|
| 0.01 | 0.665 | 0.915 | 0.77 | False |
| 0.1 | 0.746 | 0.88 | 0.808 | False |
| 1.0 | 0.897 | 0.889 | 0.893 | False |
| 10.0 | 0.93 | 0.906 | 0.918 | True |
| 100.0 | 0.921 | 0.897 | 0.909 | False |





![TF-IDF: validation confusion matrix](../results/figures/tfidf_validation_confusion.png)



Hypothesis H1 expected TF-IDF to beat plain word counts, because rare, telling words carry more information than everyday ones. On validation the expectation holds. The gain comes from precision and not from recall, which fits the idea: down-weighting everyday words removes noise that made the count model call mainstream articles conspiracy. Two cautions apply. The validation set holds only 376 articles from a few websites, so a difference of this size is not yet a proof; the final comparison on unseen websites decides whether H1 is kept. And the weights still describe words, not arguments.

The strongest words resemble those of the word-count model. "Posted", "truth" and "article" point to conspiracy websites; "killed", "tower" and "york" point to mainstream websites. Part of this signal therefore reflects how websites write and present their articles, and TF-IDF does not remove this limitation.



| word | weight |
|---|---|
| posted | 5.467 |
| truth | 4.904 |
| this | 4.115 |
| am | 3.815 |
| article | 3.735 |
| false | 3.678 |
| fbi | 3.576 |
| media | 3.292 |
| fact | 3.197 |
| official | 3.099 |
| video | 2.965 |
| government | 2.928 |
| former | 2.871 |
| wtc | 2.831 |
| here | 2.808 |
| mainstream | 2.643 |
| cia | 2.525 |
| planet | 2.508 |
| is | 2.477 |
| trillion | 2.473 |
| question | 2.448 |
| entry | 2.439 |
| israeli | 2.416 |
| israel | 2.371 |
| rss | 2.364 |





| word | weight |
|---|---|
| killed | -3.083 |
| tower | -2.626 |
| york | -2.508 |
| nearly | -2.478 |
| response | -2.473 |
| 000 | -2.447 |
| often | -2.373 |
| life | -2.369 |
| qaeda | -2.331 |
| city | -2.082 |
| whether | -2.026 |
| counterterrorism | -2.024 |
| risk | -1.948 |
| manhattan | -1.858 |
| range | -1.855 |
| twitter | -1.852 |
| al | -1.841 |
| spread | -1.816 |
| long | -1.785 |
| after | -1.782 |
| crashed | -1.779 |
| ll | -1.768 |
| number | -1.714 |
| stewart | -1.69 |
| focus | -1.684 |

## TF-IDF plus extra features

The third analysis asks whether hand-made features improve the TF-IDF model (hypothesis H2: adding the three feature groups improves TF-IDF). We added three groups, each motivated by earlier work. N-grams of one to three words capture phrases such as "inside job"; word and bigram features are a strong baseline for text classification (Wang and Manning, 2012). Epistemic markers are counts per 1,000 words of four short hand-made lists (hedges, boosters, hidden-truth phrases, debunking words), following work that ties conspiracy discourse to certainty and concealment (Marini and Jezek, 2024; Islam et al., 2020), although evidence on certainty is mixed (Fong et al., 2021). Style features are capitals, punctuation, quotation marks, sentence length and repetition; fake-news bodies use more capitals and repetition (Horne and Adali, 2017), but such features generalise poorly across domains (Perez-Rosas et al., 2018). Repetition is measured with the moving-average type-token ratio (MATTR) over windows of 100 words, because the plain type-token ratio falls with text length and would measure length rather than style.

Counts are expressed per 1,000 words so that article length does not matter, and the numeric groups were standardised with training statistics only. We compared five logistic regressions on the same cleaned articles and split: TF-IDF alone, with each group, and with all three. Every model chose its regularisation strength (C) from the same grid by validation F1 of the conspiracy class, with balanced class weights. The table below is an ablation: it explains the contribution of each group and is not a model selection, because the model used in the final comparison is TF-IDF plus all three groups, as planned. All numbers in this section are validation numbers.

| Model | Validation F1 | Change against TF-IDF alone |
|---|---|---|
| TF-IDF alone | 0.918 | reference |
| + n-grams (1-3) | 0.920 | +0.002 |
| + certainty words | 0.930 | +0.013 |
| + style | 0.909 | -0.009 |
| + all three groups | 0.896 | -0.022 |



![Validation F1 of the five models](../results/figures/ablation_f1.png)



The verdict on H2 is no: the three feature groups together lower the validation F1 from 0.918 to 0.896. Phrases changed almost nothing, certainty words alone raised F1 by 0.013, and style alone lowered it by 0.009. With 376 validation articles, differences of one or two hundredths can be chance, so even the certainty gain is only a weak hint.

Why can the full combination be worse than TF-IDF alone? The table cannot prove a cause, but it fits three explanations. First, the models have far more columns than articles: the phrase vocabulary is capped at 200,000 columns while only 1,768 articles are available for training, so the model can fit single articles. The best C for the full model is the largest value in the grid, which also means that it was allowed to follow the training data freely. Second, the groups overlap: a phrase column repeats its words, and the certainty lists count words that TF-IDF already holds, so many columns carry the same information and the weights become unstable. Third, style habits may belong to single websites. Style alone scored below TF-IDF alone, which fits the idea that these numbers describe how particular websites write and do not transfer to new websites; we did not test this directly.

The hand-made word lists also reach only part of the data: the hidden-truth list occurs in 0.439 of training conspiracy articles against 0.031 of mainstream ones, but is absent from more than half of the conspiracy articles. The lists count words and ignore meaning, so an article that denies a cover-up still scores as hidden-truth vocabulary.

## The LLM classifier

The fourth analysis uses a language model instead of a trained classifier. Claude Haiku 5.5 received each article without any examples (zero-shot) and answered the source question: would the article appear on a conspiracy-oriented or on a mainstream website? This is the same target as in the three logistic regression analyses, and the answers are scored against the same dataset labels.

Articles were cut to their first 3,000 words, as for the other methods, and the model could write up to 1,000 tokens. The model has no temperature setting, so default sampling was used. Three prompts were tried on the validation articles only: the bare question (v1), the question with short definitions of both website types (v2), and a step-by-step version that warns against judging by topic alone (v3). The prompt with the highest F1 score for the conspiracy class was chosen. Every answer was saved with the article id and a prompt hash, so repeated runs need no new requests.



| variant | documents | precision | recall | f1 |
|---|---|---|---|---|
| v1 | 376 | 1.0 | 0.974 | 0.987 |
| v2 | 376 | 1.0 | 0.966 | 0.983 |
| v3 | 376 | 1.0 | 0.974 | 0.987 |



All three prompts reached almost the same F1 score on the 376 validation articles, and none called a mainstream article conspiracy-oriented. The bare question and the step-by-step prompt tied at 0.987, and the short definitions reached 0.983. With a tie, the first prompt, v1, was kept. The choice of prompt mattered little. The chosen prompt reached a precision of 1.000 and a recall of 0.974: it missed 3 of the 117 conspiracy articles. Answers that could not be read would have counted as wrong; there were none (0), so the output limit was large enough. Because sampling cannot be fixed, the validation run of the chosen prompt was repeated once; the share of articles with the same answer both times was 0.992. In total, 30 validation articles were longer than 3,000 words and were cut.



![LLM: validation confusion matrix of the chosen prompt](../results/figures/llm_validation_confusion.png)



Hypothesis H3 expected the language model to do about as well as the best trained model, or to differ from it. On validation it differs: its F1 of 0.987 lies well above the best logistic regression on the same articles (0.918 for TF-IDF). This is not yet a verdict, because the validation websites are not the test websites and the final comparison decides.

Scoring the LLM against website labels is not a fully fair comparison with the trained models, for four reasons. First, the LLM never saw our labels or our training websites, so it cannot overfit to them; this favours it as an independent reference but also means it was never taught what our labels mean. Second, the labels describe websites, not texts. A conspiracy website prints plain news and a mainstream website sometimes prints conspiracy-friendly pieces, so a "wrong" answer may be a correct reading of the text. Third, the model may know these websites or this 9/11 material from its own training and recognise a site from its style, headings or bylines; a correct answer can then be memory rather than reading, and the logistic regressions have no such prior knowledge. Fourth, all methods see only the first 3,000 words, so the end of long articles is hidden from all of them. The score therefore shows how well the model reproduces our website labels, not how well it understands conspiracy writing.

Cost and runtime also differ. The logistic regressions run on a laptop-class machine in about a minute each and cost nothing, whereas the LLM needs an internet connection and a paid interface; the runtime per article and the price per 1,000 articles are in the methods table in section 1.

## Results and why they fail

All four final models were scored once on the same 410 test articles (96 from conspiracy-oriented websites). The table gives precision, recall and F1 for the conspiracy class with 95% bootstrap confidence intervals (2,000 resamples of documents). The baseline row always answers "mainstream"; its accuracy is 0.766, which shows why accuracy is not used: it rewards a rule that finds no conspiracy article.



| Model | Precision | Recall | F1 |
|---|---|---|---|
| Word counts | 0.607 (0.500 to 0.714) | 0.531 (0.434 to 0.631) | 0.567 (0.478 to 0.650) |
| TF-IDF | 0.721 (0.614 to 0.824) | 0.510 (0.413 to 0.615) | 0.598 (0.508 to 0.682) |
| TF-IDF + features | 0.653 (0.542 to 0.767) | 0.490 (0.394 to 0.590) | 0.560 (0.469 to 0.643) |
| LLM | 0.985 (0.949 to 1.000) | 0.667 (0.570 to 0.759) | 0.795 (0.723 to 0.857) |
| Always guess mainstream | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |

*Precision, recall and F1 of the conspiracy class on the test articles, with 95% confidence intervals*





![Confusion matrices on the test set (rows: true website type, columns: predicted)](../results/figures/comparison_confusion.png)



The LLM has the highest F1 (0.795, interval 0.723 to 0.857), followed by TF-IDF (0.598), word counts (0.567) and TF-IDF with features (0.560). The three logistic regressions have overlapping intervals and find about one half of the conspiracy articles (recall 0.490 to 0.531). The LLM is almost never wrong when it says "conspiracy" (precision 0.985, one false alarm among 314 mainstream articles) but still misses a third of the conspiracy articles (recall 0.667).

Three exact two-sided McNemar tests compare pairs of models on the same articles. Word counts against TF-IDF: 7 articles are right only for word counts and 19 only for TF-IDF (p = 0.029). TF-IDF against TF-IDF with features: 22 against 14 (p = 0.243). The best logistic regression on validation (TF-IDF) against the LLM: 4 against 37 (p < 0.0001).

**H1: TF-IDF weighting beats plain word counts.** Confirmed, narrowly. TF-IDF has the higher F1 (0.598 against 0.567), mainly through fewer false alarms (precision 0.721 against 0.607), and the McNemar test is significant (p = 0.029), although the confidence intervals overlap.

**H2: Adding the three feature groups improves the TF-IDF model.** Not confirmed. On validation the model with the extra features scored an F1 of 0.896 against 0.918 for TF-IDF alone, and on test it is lower again (0.560 against 0.598); the difference is within chance (p = 0.243).

**H3: The LLM differs from the best logistic regression.** Confirmed: the LLM is better. F1 is 0.795 against 0.598, the intervals do not overlap, and the test is clearly significant (p < 0.0001). The gain comes from precision (0.985 against 0.721) and from recall (0.667 against 0.510).

"Not confirmed" means that the data do not show an effect, not that there is none. The intervals resample documents, but websites are the real unit of dependence, so they are optimistic.

**Why the scores are lower on test than on validation.** Validation F1 was 0.865 for word counts, 0.918 for TF-IDF, 0.896 for TF-IDF with features and 0.987 for the LLM. The table gives recall per conspiracy website on the test set.



| website | test_articles | recall_final2_word_counts | recall_final2_tfidf | recall_final2_tfidf_features | recall_final_llm |
|---|---|---|---|---|---|
| conspiracydailyupdate.com | 19 | 0.684 | 0.579 | 0.684 | 0.947 |
| escapeallthesethings.com | 6 | 0.833 | 0.833 | 0.833 | 1.0 |
| needtoknow.news | 19 | 0.79 | 0.842 | 0.737 | 0.947 |
| neonnettle.com | 44 | 0.25 | 0.204 | 0.204 | 0.318 |
| oye.news | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| thelibertybeacon.com | 7 | 0.857 | 1.0 | 0.714 | 1.0 |

*Recall of each model on the conspiracy articles of each test website*



The test set holds six conspiracy websites, and one of them, neonnettle.com, supplies 44 of the 96 conspiracy articles. All four models find at most about a third of its articles (recall 0.204 to 0.318), while on the other sites the logistic regressions find 0.579 to 1.000 of them. One large, hard site therefore lowers every test score, and since no website appears in more than one split, the test websites are new to every model.

**Where the models fail, and why.** The labels describe websites, not articles. We examined the test articles on which the best logistic regression or the LLM disagrees with the label: missed only by the logistic regression, only by the LLM, or by both. A second model (Claude Sonnet 5.5) coded the stance of each towards conspiracy claims about 9/11, without being told the website. A rule fixed in advance counts a conspiracy-site article that does not endorse a claim, or a mainstream article that endorses one, as a label problem; all other cases are model errors.



| Group | Error type | Errors | Coded | Model wrong | Label does not describe the text | Not coded | Article endorses a conspiracy claim | Article debunks claims | Article reports claims neutrally | Article makes no claim about 9/11 |
|---|---|---|---|---|---|---|---|---|---|---|
| Only the logistic regression is wrong | Mainstream article called conspiracy | 19 | 19 | 18 | 1 | 0 | 1 | 3 | 4 | 11 |
| Only the logistic regression is wrong | Conspiracy-site article called mainstream | 18 | 18 | 1 | 17 | 0 | 1 | 0 | 0 | 17 |
| Only the logistic regression is wrong | All errors | 37 | 37 | 19 | 18 | 0 | 2 | 3 | 4 | 28 |
| Only the LLM is wrong | Mainstream article called conspiracy | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| Only the LLM is wrong | Conspiracy-site article called mainstream | 3 | 3 | 0 | 3 | 0 | 0 | 0 | 1 | 2 |
| Only the LLM is wrong | All errors | 4 | 4 | 1 | 3 | 0 | 0 | 0 | 1 | 3 |
| Both are wrong | Mainstream article called conspiracy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Both are wrong | Conspiracy-site article called mainstream | 29 | 28 | 0 | 28 | 1 | 0 | 1 | 1 | 26 |
| Both are wrong | All errors | 29 | 28 | 0 | 28 | 1 | 0 | 1 | 1 | 26 |
| All errors | Mainstream article called conspiracy | 20 | 20 | 19 | 1 | 0 | 1 | 3 | 4 | 12 |
| All errors | Conspiracy-site article called mainstream | 50 | 49 | 1 | 48 | 1 | 1 | 1 | 2 | 45 |
| All errors | All errors | 70 | 69 | 20 | 49 | 1 | 2 | 4 | 6 | 57 |



Of the 69 articles with a readable coding, 49 are label problems and 20 are model errors. Almost all label problems are conspiracy-site articles that make no claim about 9/11 at all, such as ordinary news, so a model that reads only the text has little to find. Mainstream articles called conspiracy are mostly model errors. The 29 articles missed by both models are all conspiracy-site articles. Four examples show these cases.



| Case | Website type | Logistic regression said | Claude said | What the article does | Excerpt |
|---|---|---|---|---|---|
| Only Claude was right | mainstream | conspiracy | mainstream | makes no claim | New York time, paced to display as they were broadcast at the corresponding time on ... |
| Only the logistic regression was right | conspiracy | conspiracy | mainstream | reports neutrally | The vacuum left after the invasion is widely believed to have spawned terror group ISIS. ... |
| Mainstream article called conspiracy | mainstream | conspiracy | mainstream | debunks claims | government investigations, whose reports are all available online, have concluded that there is no evidence ... |
| Conspiracy-site article called mainstream | conspiracy | mainstream | mainstream | makes no claim | An organization with links to terrorism, that's backed by radical ... |



The label-problem share rests on the automatic coding alone; the team's hand check of 40 articles follows.

## Limitations

Several limitations qualify the results. First, the label describes the website that published a document, not its content. Conspiracy-oriented websites publish many neutral articles, and mainstream websites sometimes discuss conspiracy theories, so a prediction that disagrees with the label is not necessarily an error of the model. The scores measure how well each method recovers the type of website, not whether a text contains conspiracy claims. This label noise limits the recall of every method.

Second, the number of websites is small: the validation and test sets each contain only a handful of conspiracy websites. Documents from one website resemble each other, so the confidence intervals, which resample documents, are probably somewhat too narrow, and a different assignment of websites to sets could change the ranking of close methods. The size of this effect is visible in the LLM, whose F1 score on validation was 0.987 and on test 0.795, almost all of it lost in recall (0.974 on validation, 0.667 on test). We did not test the reasons. We also repeated the validation run of the LLM once and report the share of identical answers.



**Share of identical answers in a repeated validation run:** 0.992



Third, the LLM was trained on large amounts of web text and may know the articles or the reputation of their websites. This cannot be excluded, so part of its performance may rest on prior knowledge that the trained models do not have. Moreover, the LLM saw only the first 3,000 words of each article, and 30 validation articles were longer and were cut, so it may have missed later passages that the logistic regressions, which read the full text, could use.

Fourth, the certainty and doubt features are counts of words from a hand-made list and ignore negation: "no evidence of a cover-up" counts like "a cover-up". Fifth, although names, addresses and repeated lines of the websites were removed, the typical style of a website can remain and can serve as a shortcut for the models. The licence of the corpus has not been confirmed, so the texts are used for study only and are not redistributed. Finally, all results concern one topic, one corpus and one split, so we cannot say whether they transfer. The test set was used during development; every final model was chosen on validation and scored on test once.

## Conclusion

The study compared four methods for deciding whether a full 9/11 article comes from a conspiracy-oriented or a mainstream website. The results are summarised per hypothesis. All numbers are F1 scores of the conspiracy class on the 410 test articles.

H1 (TF-IDF improves on word counts) is not confirmed. TF-IDF reached an F1 score of 0.582 and plain word counts 0.559. The direction agrees with the hypothesis, but the intervals overlap almost completely and the McNemar test is not significant (p = 0.087).

H2a (certainty words improve the TF-IDF model) is not confirmed. On validation they lifted F1 from 0.914 to 0.930, but on test F1 rose only from 0.582 to 0.610, mostly through fewer false alarms, and the difference is within chance (p = 0.332).

H2b (style features improve it) is not supported: validation F1 fell to 0.894.

H2c (phrases of two or three words improve it) is not supported: validation F1 was 0.916 against 0.914, no clear gain.

H3 (the LLM beats the best logistic regression) is confirmed. The LLM reached an F1 score of 0.795 against 0.610 for TF-IDF with features, the intervals do not overlap, and the difference is significant (p < 0.0001). The gain comes from precision (0.985) and not from recall (0.667): the LLM almost never labels a mainstream article as conspiracy but misses one third of the conspiracy articles.

In sum, the methods that we trained ranked as hypothesised, but their differences are small relative to the uncertainty, and only the advantage of the LLM is clear. Because the labels describe websites and the number of websites is small, the findings describe how well each method recovers the type of website, and further work should test other topics, more websites and the human check of the label problems.

## How we used AI

**What the AI produced.** The AI assistant Claude drafted the project plan, wrote the analysis code and the tests, produced the tables and figures, and drafted the text of the course site and of this report. A separate language model, Claude Haiku 5.5, classified the articles as the fourth method under study. A second model, Claude Sonnet 5.5, coded how the articles in which the methods disagree treat conspiracy claims, for the error analysis.

**What the team checked or changed.** The team fixed the research question, full articles as input, the four methods, the feature groups, and the rule that each final model is scored on the test articles once. For example, the team removed emotion scores from the feature list of the plan the AI drafted, because they add cost and emotional language is not the same as conspiracy language. The article counts were compared with those published for the corpus. Tests were written that fail when a rule is broken, such as fitting on training articles only, a changed split, or a number in the report that is missing from the result files. Human checks are pending: reading the text, spot-checking numbers, and coding 40 articles by hand. None of them is recorded as done.

**What we learned about the work.** Tests confirm that numbers match files, not that they mean the right thing, so a person must still judge the reasoning. The stance codes produced by a language model need a human check before they are relied on. Saving the language model answers makes every figure reproducible without new cost.

## References

- Fong, A., Roozenbeek, J., Goldwert, D., Rathje, S., & van der Linden, S. (2021). The language of conspiracy: A psychological analysis of speech used by conspiracy theorists and their followers on Twitter. *Group Processes & Intergroup Relations, 24*(4). https://journals.sagepub.com/doi/10.1177/1368430220987596
- Horne, B., & Adalı, S. (2017). This just in: Fake news packs a lot in title, uses simpler, repetitive content in text body, more similar to satire than real news. *ICWSM workshops*. https://arxiv.org/abs/1703.09398
- Islam, J., Xiao, L., & Mercer, R. (2020). A lexicon-based approach for detecting hedges in informal text. *LREC*. https://aclanthology.org/2020.lrec-1.380
- Marini, C., & Jezek, E. (2024). What to annotate: Retrieving lexical markers of conspiracy discourse from an Italian-English corpus of Telegram data. *ISA-20 Workshop*. https://aclanthology.org/2024.isa-1.6
- Miani, A., Hills, T., & Bangerter, A. (2022). LOCO: The 88-million-word language of conspiracy corpus. *Behavior Research Methods, 54*, 1794–1817. https://doi.org/10.3758/s13428-021-01698-z
- Pérez-Rosas, V., Kleinberg, B., Lefevre, A., & Mihalcea, R. (2018). Automatic detection of fake news. *COLING*. https://arxiv.org/abs/1708.07104
- Wang, S., & Manning, C. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. *ACL*. https://aclanthology.org/P12-2018
