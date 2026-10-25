# Conspiracy or mainstream? Classifying 9/11 articles with logistic regression and an LLM

## Question and hypotheses

### Executive summary

We asked whether the text of a full 9/11 article reveals if it was published on a conspiracy-oriented or a mainstream website. Three logistic regressions (on word counts, on TF-IDF weights, and on TF-IDF plus three hand-made feature groups) and one low-cost language model, Claude Haiku 5.5, were compared on 410 held-out test articles. The language model reached the highest F1 score for the conspiracy class, 0.795 (95% confidence interval 0.723 to 0.857), against 0.567 for word counts, 0.598 for TF-IDF and 0.560 for TF-IDF with features. H1 (weighting words helps) was weakly supported, H2 (hand-made features help) was not confirmed, and H3 (the language model beats the best logistic regression) was confirmed. All models found fewer than seven in ten conspiracy articles on the test websites, so none is reliable enough for automatic flagging.

### Introduction

The 9/11 attacks are a lasting subject of online conspiracy content, (Miani, Hills & Bangerter, 2022). This study asks whether the text of a full article about the 9/11 attacks reveals if it was published on a conspiracy-oriented or on a mainstream website, using machine-learning methods for text classification. The articles come from the LOCO corpus, and the label describes the website, not the single article.

We compare four methods. Analyses 1 to 3 are logistic regression on different inputs: word counts, TF-IDF weights, and TF-IDF weights plus three hand-made feature groups (word sequences of up to three words, certainty and hedging words, and style measures). Analysis 4 is one low-cost large language model (LLM) used without training on our data. One simple classifier with changing inputs shows which input matters; other classifiers, fine-tuning and emotion features were left out.

Table 1 gives the four methods at a glance.


| method | input | reads | trained_on_our_data | settings | runtime | cost |
|---|---|---|---|---|---|---|
| 1. Word counts + logistic regression | word counts | first 3,000 words of each article | yes | C from [0.01, 0.1, 1, 10, 100], class weight balanced, chosen on validation | 3 seconds | free (runs locally) |
| 2. TF-IDF + logistic regression | TF-IDF weights | first 3,000 words of each article | yes | C from [0.01, 0.1, 1, 10, 100], class weight balanced, chosen on validation | 2 seconds | free (runs locally) |
| 3. TF-IDF + features + logistic regression | TF-IDF weights, n-grams, certainty words, style | first 3,000 words of each article | yes | C from [0.01, 0.1, 1, 10, 100], class weight balanced, chosen on validation | 58 seconds | free (runs locally) |
| 4. LLM (claude-haiku-5-5) | the article text itself, with a written question | first 3,000 words of each article | no | prompt v1 of 3, default sampling | about 1.0 seconds per article | about $0.26 per 1,000 articles |



We test three hypotheses. **H1:** TF-IDF weights give a higher conspiracy-class F1 than plain word counts, because distinctive words carry more information than common ones. **H2:** adding the three feature groups (word sequences, certainty words, style) to TF-IDF raises the F1, because they capture how conspiracy texts argue. **H3:** The zero-shot LLM reaches a higher conspiracy-class F1 on the test set than the best logistic regression, and the McNemar test gives p < 0.05.
All methods are judged by precision, recall and F1 for the conspiracy class, because most articles are mainstream and the share of correct answers would reward a model that never finds a conspiracy article. A hypothesis is called confirmed only if the difference points in the predicted direction, the McNemar test gives p < 0.05 and the 95% confidence intervals do not overlap. If the test is significant but the intervals overlap, it is called weakly supported; otherwise it is not confirmed.

## Data and cleaning

The study uses the 9/11 subset of the LOCO corpus (Miani, Hills & Bangerter, 2022), a collection of web pages from websites rated as conspiracy-oriented or mainstream. We use the full articles of this subset, 3,146 documents from 39 conspiracy and 79 mainstream websites. Each document carries the label of its website. The label therefore describes the publisher and not the single text: a neutral news item on a conspiracy-oriented website is labelled conspiracy, and a mainstream article that discusses a conspiracy theory is labelled mainstream. The work used Python with pandas and scikit-learn (Pedregosa et al., 2011).

Cleaning matters because a classifier takes the easiest signal that separates the classes, whether or not it is the intended one. This shortcut learning, a form of target leakage, arises when website names, repeated boilerplate, topic differences or copies of the same article reveal the label without any property of the text itself. All cleaning was therefore applied before any model was fitted, and all four analyses use the same cleaned documents. First, documents that do not mention the attacks in their title or text were removed (3,146 to 2,608), because a model could otherwise learn the difference between topics instead of the difference between kinds of websites. Second, near-duplicates were removed (2,608 to 2,554): pairs of documents with a cosine similarity of their TF-IDF vectors of at least 0.9. Such copies would let a model meet a near copy of a test article during training. The rule depends on the split and not on the label: when the two copies lie in different sets, the copy in the training set is kept (and the validation copy rather than the test copy), and when both lie in the same set, the copy with the lower document id is kept. Third, web addresses, e-mail addresses, website names and lines repeated across many documents of one website were deleted from the texts. This step edited the texts of 2,498 documents and removed 23,390 words, but removed 0 documents. The table shows the number of documents before and after each of the first two steps.



| step | removed | remaining | removed_conspiracy | removed_mainstream | remaining_conspiracy | remaining_mainstream |
|---|---|---|---|---|---|---|
| all documents | 0 | 3146 | 0 | 0 | 939 | 2207 |
| off-topic articles removed | 538 | 2608 | 162 | 376 | 777 | 1831 |
| near-duplicates removed | 54 | 2554 | 22 | 32 | 755 | 1799 |

*Documents before and after each cleaning step*



After cleaning, 2,554 documents remain: 755 from conspiracy-oriented and 1,799 from mainstream websites. Roughly three in ten articles are conspiracy, so the classes are unbalanced.

## Splitting by website

The documents were divided into a training set, a validation set and a test set with target shares of 70%, 15% and 15%. The division was made by website: all documents of one website were assigned to the same set, so that no website appears in more than one set. This stops a model from being rewarded for recognising a website it has seen, as when classifying unknown websites. The division was stratified by label: conspiracy and mainstream websites were assigned separately, from the largest to the smallest website, to the set that was furthest short of its target share, with a fixed random seed for ties. The assignment was made once and never changed.



| split | class | websites | documents | share_of_documents |
|---|---|---|---|---|
| train | conspiracy | 25 | 542 | 0.212 |
| train | mainstream | 54 | 1226 | 0.48 |
| validation | conspiracy | 6 | 117 | 0.046 |
| validation | mainstream | 13 | 259 | 0.101 |
| test | conspiracy | 6 | 96 | 0.038 |
| test | mainstream | 12 | 314 | 0.123 |

*Websites and documents per set and class*



Because websites cannot be divided, the shares only approximate the targets. The training, validation and test sets contain 25, 6 and 6 conspiracy websites (37 in total) and 54, 13 and 12 mainstream websites (79 in total); no website is in more than one set. All model settings were chosen on the validation set, and the test set was used only for the final comparison.

## Word counts

The first analysis represents each article as a bag of words: a vector with the number of times each word occurs, ignoring order. Words were counted on the training articles only, and words occurring in fewer than two training articles were dropped. The model is a logistic regression (Hastie, Tibshirani and Friedman, 2009) with balanced class weights (section 1 gives the reasons for this choice and for leaving out other classifiers). Its regularisation strength (C) was chosen from {0.01, 0.1, 1, 10, 100} by the F1 score of the conspiracy class on the validation set. All numbers in this section are validation numbers.

The smallest value, C = 0.01, scored best, with a precision of 0.828, a recall of 0.906 and an F1 score of 0.865. Larger values scored lower (F1 0.797 at C = 100), the typical sign of overfitting: with raw counts the model starts to rely on rare words that only describe the training articles. Of the 117 conspiracy articles in validation the model found 106, and of the 259 mainstream articles it called 22 conspiracy.



![Word counts: validation confusion matrix](../results/figures/word_counts_validation_confusion.png)



The words with the largest weights are informative, but not all are about content. Words such as "posted", "am" and "article" among the strongest conspiracy words look like habits of how a website presents its articles. Mainstream words such as "killed", "says" and "anniversary" fit factual news reporting. The model therefore partly recognises websites and not only arguments.

## TF-IDF

The second analysis keeps everything from the word-count analysis (articles, split, vocabulary, balanced logistic regression, grid for the regularisation strength C) and replaces raw counts by TF-IDF weights (Manning, Raghavan and Schütze, 2008). Term frequencies were damped with a logarithm, article vectors were scaled to unit length, and the inverse document frequencies were computed on the training articles only. C was again chosen from {0.01, 0.1, 1, 10, 100} by the validation F1 score of the conspiracy class. All numbers in this section are validation numbers.

The value C = 10 scored best, with a precision of 0.930, a recall of 0.906 and an F1 score of 0.918. Compared with word counts (F1 0.865), recall stayed the same and precision rose from 0.828 to 0.930: the number of mainstream articles wrongly called conspiracy fell from 22 to 8. Unlike with raw counts, larger C helped: because TF-IDF vectors have a fixed length, the model can use larger weights without being dominated by long articles. The chosen value lies inside the grid, and C = 100 scored almost the same (F1 0.909).



![TF-IDF: validation confusion matrix](../results/figures/tfidf_validation_confusion.png)



Hypothesis H1 expected TF-IDF to beat plain word counts, because rare, telling words carry more information than everyday ones. On validation the expectation holds. The gain comes from precision and not from recall, which fits the idea: down-weighting everyday words removes noise that made the count model call mainstream articles conspiracy. The final comparison on unseen websites decides whether H1 is kept.

The strongest words resemble those of the word-count model. "Posted", "truth" and "article" point to conspiracy websites; "killed", "tower" and "york" point to mainstream websites. Part of this signal reflects how websites present their articles.

## TF-IDF plus extra features

The third analysis asks whether hand-made features improve the TF-IDF model (hypothesis H2: adding the three feature groups improves TF-IDF). We added three groups, each motivated by earlier work. N-grams of one to three words capture phrases such as "inside job"; word and bigram features are a strong baseline for text classification (Wang and Manning, 2012). Epistemic markers are counts per 1,000 words of four short hand-made lists (hedges, boosters, hidden-truth phrases, debunking words), following work that ties conspiracy discourse to certainty and concealment (Marini and Jezek, 2024; Islam et al., 2020), although evidence on certainty is mixed (Fong et al., 2021). Style features are capitals, punctuation, quotation marks, sentence length and repetition (Horne and Adali, 2017); they generalise poorly across domains (Perez-Rosas et al., 2018).

The numeric groups were standardised with training statistics only. We compared five logistic regressions on the same split: TF-IDF alone, with each group, and with all three, each with C chosen from the same grid by validation F1 and balanced class weights. The table is an ablation, not a model selection: the final model is TF-IDF plus all three groups, as planned. All numbers in this section are validation numbers.

| Model | Validation F1 | Change against TF-IDF alone |
|---|---|---|
| TF-IDF alone | 0.918 | reference |
| + n-grams (1-3) | 0.920 | +0.002 |
| + certainty words | 0.930 | +0.013 |
| + style | 0.909 | -0.009 |
| + all three groups | 0.896 | -0.022 |



![Validation F1 of the five models](../results/figures/ablation_f1.png)



The verdict on H2 is no: the three feature groups together lower the validation F1 from 0.918 to 0.896. With 376 validation articles, differences of one or two hundredths can be chance, so even the certainty gain is only a weak hint.

Why can the full combination be worse than TF-IDF alone? The table cannot prove a cause, and we did not test any of the following three explanations. First, the full model has far more columns than articles (146,172 columns against 1,768 training articles), so it could fit single articles. Second, the groups overlap with words TF-IDF already holds. Third, style habits may belong to single websites, which would fit style alone scoring below TF-IDF alone.

## The LLM classifier

The fourth analysis uses a language model instead of a trained classifier. Claude Haiku 5.5 received each article without any examples (zero-shot) and answered the source question: would the article appear on a conspiracy-oriented or on a mainstream website? The answers are scored against the same labels as for the logistic regressions.

Articles were cut to their first 3,000 words, as for the other methods, and the model could write up to 1,000 tokens. The model has no temperature setting, so default sampling was used. Three prompts were tried on the validation articles only: the bare question (v1), the question with short definitions of both website types (v2), and a step-by-step version that warns against judging by topic alone (v3). The prompt with the highest conspiracy-class F1 was chosen. Answers were saved, so repeated runs need no new requests.



| variant | documents | precision | recall | f1 |
|---|---|---|---|---|
| v1 | 376 | 1.0 | 0.974 | 0.987 |
| v2 | 376 | 1.0 | 0.966 | 0.983 |
| v3 | 376 | 1.0 | 0.974 | 0.987 |



All three prompts reached almost the same F1 score on the 376 validation articles, and none called a mainstream article conspiracy-oriented. The bare question and the step-by-step prompt tied at 0.987, and the short definitions reached 0.983. With a tie, the first prompt, v1, was kept. The chosen prompt reached a precision of 1.000 and a recall of 0.974: it missed 3 of the 117 conspiracy articles. There were no unreadable answers. A repeated validation run gave the same answer for 0.992 of the articles.



![LLM: validation confusion matrix of the chosen prompt](../results/figures/llm_validation_confusion.png)



Hypothesis H3 states: The zero-shot LLM reaches a higher conspiracy-class F1 on the test set than the best logistic regression, and the McNemar test gives p < 0.05. On validation it is already ahead: its F1 of 0.987 lies well above the best logistic regression on the same articles (0.918 for TF-IDF). The final comparison decides.

Scoring the LLM against website labels is not a fully fair comparison with the trained models, for four reasons. First, the LLM never saw our labels or training websites, so it cannot overfit to them, but it was never taught what our labels mean. Second, the labels describe websites, not texts, so a "wrong" answer may be a correct reading of the text. Third, the model may recognise a site from its style or bylines through its own training; a correct answer can then be memory rather than reading, which the logistic regressions cannot use. Fourth, all methods see only the first 3,000 words.

## Results and why they fail

All four final models were scored once on the same 410 test articles (96 from conspiracy-oriented websites). The table gives precision, recall and F1 for the conspiracy class with 95% bootstrap confidence intervals (Efron and Tibshirani, 1993; 2,000 resamples of documents). The baseline row always answers "mainstream"; its accuracy is 0.766, which is why accuracy is not used.



| Model | Precision | Recall | F1 |
|---|---|---|---|
| Word counts | 0.607 (0.500 to 0.714) | 0.531 (0.434 to 0.631) | 0.567 (0.478 to 0.650) |
| TF-IDF | 0.721 (0.614 to 0.824) | 0.510 (0.413 to 0.615) | 0.598 (0.508 to 0.682) |
| TF-IDF + features | 0.653 (0.542 to 0.767) | 0.490 (0.394 to 0.590) | 0.560 (0.469 to 0.643) |
| LLM | 0.985 (0.949 to 1.000) | 0.667 (0.570 to 0.759) | 0.795 (0.723 to 0.857) |
| Always guess mainstream | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |

*Precision, recall and F1 of the conspiracy class on the test articles, with 95% confidence intervals*





![Confusion matrices on the test set (rows: true website type, columns: predicted)](../results/figures/comparison_confusion.png)



The LLM has the highest F1 (0.795, interval 0.723 to 0.857), followed by TF-IDF (0.598), word counts (0.567) and TF-IDF with features (0.560). The three logistic regressions have overlapping intervals and find about half of the conspiracy articles. The LLM is almost never wrong when it says "conspiracy" (precision 0.985, one false alarm among 314 mainstream articles) but still misses a third of the conspiracy articles (recall 0.667).

Three exact two-sided McNemar tests (McNemar, 1947) compare pairs of models on the same articles.

**H1: TF-IDF weighting beats plain word counts.** Weakly supported: the test is significant, but the intervals overlap. TF-IDF has the higher F1 (0.598 against 0.567), mainly through fewer false alarms (precision 0.721 against 0.607). Of the articles on which they differ, 7 are right only for word counts and 19 only for TF-IDF (p = 0.029), although the confidence intervals overlap.

**H2: Adding the three feature groups improves the TF-IDF model.** Not confirmed. On validation the model with the extra features scored an F1 of 0.896 against 0.918 for TF-IDF alone, and on test it is lower again (0.560 against 0.598); 22 articles are right only for TF-IDF and 14 only with the features, within chance (p = 0.243).

**H3: The zero-shot LLM reaches a higher conspiracy-class F1 on the test set than the best logistic regression, and the McNemar test gives p < 0.05.** Confirmed: the LLM is better. F1 is 0.795 against 0.598, the intervals do not overlap, and 4 articles are right only for the best logistic regression (TF-IDF) against 37 only for the LLM (p < 0.0001).

**Why the scores are lower on test than on validation.** Validation F1 was 0.865 for word counts, 0.918 for TF-IDF, 0.896 for TF-IDF with features and 0.987 for the LLM. The table gives recall per conspiracy website on the test set.



| Website | Test articles | Recall: Word counts | Recall: TF-IDF | Recall: TF-IDF + features | Recall: LLM |
|---|---|---|---|---|---|
| conspiracydailyupdate.com | 19 | 0.684 | 0.579 | 0.684 | 0.947 |
| escapeallthesethings.com | 6 | 0.833 | 0.833 | 0.833 | 1.0 |
| needtoknow.news | 19 | 0.79 | 0.842 | 0.737 | 0.947 |
| neonnettle.com | 44 | 0.25 | 0.204 | 0.204 | 0.318 |
| oye.news | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| thelibertybeacon.com | 7 | 0.857 | 1.0 | 0.714 | 1.0 |

*Recall of each model on the conspiracy articles of each test website*



The test set holds six conspiracy websites, and one of them, neonnettle.com, supplies 44 of the 96 conspiracy articles. All four models find at most about a third of its articles (recall 0.204 to 0.318), while on the other sites the logistic regressions find 0.579 to 1.000 of them. Among the five sites with more than one test article, the LLM finds at least as many articles as every logistic regression, and more on four of them. When the 44 neonnettle.com articles are left out of the test set (366 articles, 52 of them from conspiracy sites), the F1 is 0.640 for word counts, 0.721 for TF-IDF, 0.661 for TF-IDF with features and 0.971 for the LLM. This is an observation, not a tested explanation.

**Where the models fail, and why.** The labels describe websites, not articles. We examined the test articles on which the best logistic regression or the LLM disagrees with the label. A second model (Claude Sonnet 5.5) coded the stance of each towards conspiracy claims about 9/11, without being told the website. A fixed rule counts a conspiracy-site article that does not endorse a claim, or a mainstream article that endorses one, as a label problem; all other cases are model errors.

Of the 69 articles with a readable coding, 49 are label problems and 20 are model errors. Almost all label problems are conspiracy-site articles that make no claim about 9/11; mainstream articles called conspiracy are mostly model errors. The 29 articles missed by both models are all conspiracy-site articles. The table gives the three groups with the share of coded articles that are label problems.



| Group | Articles | Coded | Label problems | Label-problem share |
|---|---|---|---|---|
| Only the logistic regression is wrong | 37 | 37 | 18 | 0.486 |
| Only the LLM is wrong | 4 | 4 | 3 | 0.75 |
| Both are wrong | 29 | 28 | 28 | 1.0 |

*Test articles wrongly classified, by who is wrong, and the share that are label problems*



Four examples show these cases.



| Case | Website type | Logistic regression said | Claude said | What the article does | Excerpt |
|---|---|---|---|---|---|
| Only Claude was right | mainstream | conspiracy | mainstream | makes no claim | New York time, paced to display as they were broadcast at the corresponding time on ... |
| Only the logistic regression was right | conspiracy | conspiracy | mainstream | reports neutrally | The vacuum left after the invasion is widely believed to have spawned terror group ISIS. ... |
| Mainstream article called conspiracy | mainstream | conspiracy | mainstream | debunks claims | government investigations, whose reports are all available online, have concluded that there is no evidence ... |
| Conspiracy-site article called mainstream | conspiracy | mainstream | mainstream | makes no claim | An organization with links to terrorism, that's backed by radical ... |



The label-problem share rests on the automatic coding alone. For a hand check, the file stance_check_sample.csv lists 40 validation articles, with the second model's answer kept in a separate file so the team reads blind. The sheet has not been filled in yet, so no agreement is reported.

## Limitations

Several limitations qualify the results. First, the label describes the website that published a document, not its content. Conspiracy-oriented websites publish many neutral articles, and mainstream websites sometimes discuss conspiracy theories, so a prediction that disagrees with the label is not necessarily an error of the model. The scores measure how well each method recovers the type of website, not whether a text contains conspiracy claims. This label noise limits the recall of every method.

Second, the number of websites is small. The validation and test sets each contain only six conspiracy websites, and in the test set one of them supplies 44 of the 96 conspiracy articles. Documents from one website resemble each other, so the confidence intervals are probably too narrow.

Third, the LLM was trained on large amounts of web text and may know the articles or the reputation of their websites, which the trained models cannot use. All four methods read only the first 3,000 words of each article, and 30 validation articles were longer and were cut.

Fourth, some choices are thresholds that could be set differently, such as the cosine similarity of 0.9 for copies, which removed 54 articles. The split between label problems and model errors is an estimate. The certainty and doubt features ignore negation, and website style can serve as a shortcut.

Finally, all results concern one topic, one corpus and one split, so we cannot say whether they transfer to other topics or websites. The test set was used during development; every final model was chosen on validation and scored on test once.

## Conclusion

The study compared four methods for deciding whether a full 9/11 article comes from a conspiracy-oriented or a mainstream website. All numbers are conspiracy-class F1 scores on the 410 test articles.

H1 (TF-IDF improves on word counts) is weakly supported: F1 0.598 against 0.567 (p = 0.029), but the confidence intervals overlap.

H2 (adding the three feature groups improves the TF-IDF model) is not confirmed: on test the F1 was 0.560 against 0.598, within chance (p = 0.243). The hand-made features did not help.

H3 (The zero-shot LLM reaches a higher conspiracy-class F1 on the test set than the best logistic regression, and the McNemar test gives p < 0.05) is confirmed: F1 0.795 against 0.598 (p < 0.0001). It misses one third of the conspiracy articles.

In sum, only the advantage of the LLM is clear, and no method finds enough conspiracy articles to be used alone for automatic flagging. Further work should test other topics and more websites, and should add the human check of the label problems.

## How we used AI

**Setup.** The analysis was produced with Claude Code, an AI coding assistant that reads and edits the files of a project. A planner split the team's requirements into small tasks, an implementer carried out one task at a time and ran the tests, and a reviewer, started in a fresh session, scored the result against a checklist. The website and this report are generated from the project files, so the code shown is the code that ran.

**What the AI produced.** The assistant drafted the plan, the code, the tests, the tables, figures and text. Claude Sonnet 5.5 coded how disputed articles treat conspiracy claims.

**What the team checked or changed.** The team fixed the research question, the four methods, the feature groups and the rule that each final model is scored on test once, and removed emotion scores from the AI's draft plan. Tests fail when a rule is broken, such as fitting on training articles only. A hand check of 40 articles, which tests the automatic stance coding, is done by the team and follows.

**What we learned.** Tests confirm that numbers match files, not that they mean the right thing, so a person must judge the reasoning.

## References

- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.
- Fong, A., Roozenbeek, J., Goldwert, D., Rathje, S., & van der Linden, S. (2021). The language of conspiracy: A psychological analysis of speech used by conspiracy theorists and their followers on Twitter. *Group Processes & Intergroup Relations, 24*(4). https://journals.sagepub.com/doi/10.1177/1368430220987596
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2nd ed.). Springer.
- Horne, B., & Adalı, S. (2017). This just in: Fake news packs a lot in title, uses simpler, repetitive content in text body, more similar to satire than real news. *ICWSM workshops*. https://arxiv.org/abs/1703.09398
- Islam, J., Xiao, L., & Mercer, R. (2020). A lexicon-based approach for detecting hedges in informal text. *LREC*. https://aclanthology.org/2020.lrec-1.380
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press.
- Marini, C., & Jezek, E. (2024). What to annotate: Retrieving lexical markers of conspiracy discourse from an Italian-English corpus of Telegram data. *ISA-20 Workshop*. https://aclanthology.org/2024.isa-1.6
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157.
- Miani, A., Hills, T., & Bangerter, A. (2022). LOCO: The 88-million-word language of conspiracy corpus. *Behavior Research Methods, 54*, 1794–1817. https://doi.org/10.3758/s13428-021-01698-z
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Pérez-Rosas, V., Kleinberg, B., Lefevre, A., & Mihalcea, R. (2018). Automatic detection of fake news. *COLING*. https://arxiv.org/abs/1708.07104
- Wang, S., & Manning, C. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. *ACL*. https://aclanthology.org/P12-2018
