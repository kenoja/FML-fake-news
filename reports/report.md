# Conspiracy or mainstream? Classifying 9/11 articles with logistic regression and an LLM

## Question and hypotheses

This study asks whether the text of a full article about the 9/11 attacks reveals if it was published on a conspiracy-oriented or on a mainstream website, and how four methods compare on this task. The methods are logistic regression on word counts, logistic regression on TF-IDF weights, logistic regression on TF-IDF weights plus hand-made features, and one low-cost large language model (LLM) used without any training on our data. The following hypotheses were formulated before the final evaluation.

H1. Weighting words by their distinctiveness (TF-IDF) improves on plain word counts.

H2. Adding features about certainty and writing style improves the TF-IDF model.

H3. The LLM does better than the best logistic regression.

All methods are judged by precision, recall and F1 score for the conspiracy class, because the classes are unbalanced: most articles come from mainstream websites, so the share of correct answers would reward a model that never finds a conspiracy article. A hypothesis counts as supported only if the confidence interval and the significance test agree with the direction of the difference. Results are reported in the results section, and the conclusion returns to each hypothesis.

## Data and cleaning

The study uses the 9/11 subset of the LOCO corpus (Miani, Hills & Bangerter, 2022), a collection of web pages from websites rated as conspiracy-oriented or mainstream. We use the full articles of this subset, 3,146 documents from 39 conspiracy and 79 mainstream websites. Each document carries the label of its website. The label therefore describes the publisher and not the single text: a neutral news item on a conspiracy-oriented website is labelled conspiracy, and a mainstream article that discusses a conspiracy theory is labelled mainstream. The texts are not redistributed and are used for study only; they are kept in a private data store, and this report shows only counts and short excerpts. The work was done in Python with pandas and scikit-learn, and the language-model classifier is Claude Haiku 5.5, used through the Anthropic API.

Cleaning was applied before any model was fitted, and all four analyses use the same cleaned documents. First, documents that do not mention the attacks in their title or text were removed, because a model could otherwise learn the difference between topics instead of the difference between kinds of websites. Second, near-duplicates were removed: pairs of documents with a cosine similarity of their TF-IDF vectors of at least 0.9. Such copies are common between websites and would let a model meet a near copy of a test article during training. For a pair across two parts of the data, the copy in validation or test was dropped; inside one part, the copy with the lower identifier was kept. Third, web addresses, e-mail addresses, website names and lines repeated across many documents of one website were deleted from the texts. The table shows the number of documents before and after each of the first two steps.



| step | removed | remaining | removed_conspiracy | removed_mainstream | remaining_conspiracy | remaining_mainstream |
|---|---|---|---|---|---|---|
| all documents | 0 | 3146 | 0 | 0 | 939 | 2207 |
| off-topic articles removed | 538 | 2608 | 162 | 376 | 777 | 1831 |
| near-duplicates removed | 54 | 2554 | 22 | 32 | 755 | 1799 |

*Documents before and after each cleaning step*



After cleaning, 2,554 documents remain: 755 from conspiracy-oriented and 1,799 from mainstream websites. Roughly three in ten articles are conspiracy, so the classes are unbalanced. Document length was checked to decide whether very short texts need special treatment. After cleaning, the shortest document has 102 words and the median is 622 words, so no minimum length was applied.

## Splitting by website

The documents were divided into a training set, a validation set and a test set with target shares of 70%, 15% and 15%. The division was made by website: all documents of one website were assigned to the same set, so that no website appears in more than one set. This prevents a model from being rewarded for recognising the layout, vocabulary or topics of a website it has already seen, and it mimics the practical situation of classifying articles from unknown websites. The division was stratified by label: conspiracy and mainstream websites were assigned separately, from the largest to the smallest website, to the set that was furthest short of its target share. A fixed random seed resolved ties. The assignment was made once, before any model was fitted, and stored in a file that was never changed; the cleaning steps only remove documents from it. The split file was made before cleaning, on all 3,146 documents; cleaning then removed documents from it without moving any to another set. The table shows the numbers of websites and documents that remain in each set after cleaning, and the second table the totals per set.



| split | class | websites | documents | share_of_documents |
|---|---|---|---|---|
| train | conspiracy | 25 | 542 | 0.212 |
| train | mainstream | 54 | 1226 | 0.48 |
| validation | conspiracy | 6 | 117 | 0.046 |
| validation | mainstream | 13 | 259 | 0.101 |
| test | conspiracy | 6 | 96 | 0.038 |
| test | mainstream | 12 | 314 | 0.123 |

*Websites and documents per set and class (after cleaning)*



Because websites cannot be divided, the shares only approximate the targets. Each set contains at least three conspiracy websites, and the number of websites in more than one set is zero. Validation and test sets still contain only a few conspiracy websites, so scores depend to some extent on which websites ended up there. All model settings were chosen on the validation set, and the test set was used only for the final comparison.

## Word counts

The first analysis represents each article as a bag of words: a vector with the number of times each word occurs, ignoring order. Words were counted on the training articles only, and words occurring in fewer than two training articles were dropped. A logistic regression with balanced class weights was fitted on these counts. The regularisation strength C was chosen from {0.1, 1, 10} by the F1 score of the conspiracy class on the validation set. The test set was not used.

The smallest value, C = 0.1, scored best. On validation this model reached a precision of 0.832, a recall of 0.846 and an F1 score of 0.839. Larger values of C scored lower, which is the typical sign of overfitting: with raw counts, the model starts to rely on rare words that only describe the training articles. The best value lies at the edge of the grid, so a smaller C could be slightly better. We did not widen the grid after seeing the results.

The words with the largest weights are informative, but not all of them are about content. Several of the strongest conspiracy words, for example "posted", "am" and "article", look like habits of how a website presents its articles, not like conspiracy thinking. Strong mainstream words such as "killed", "tower" and "anniversary" fit factual news reporting. The model therefore partly recognises websites and not only arguments, which matters for how the results are read in the results section.

What we did not do and why. We used logistic regression for all three trained models because it is simple, its weights can be read as word lists, and it fits the wide, mostly empty tables that text produces. We did not fine-tune a language model: that costs more data and money than a small project has, and it would blur the comparison. For the same reason we tried no other classifiers, such as support vector machines or tree ensembles; the question was how the features and the language model differ, not which algorithm is best. We also left out emotion features, which lie outside the question and rest on word lists that are unreliable for a topic where fear and anger are normal.



![Word counts: validation confusion matrix](../results/figures/word_counts_validation_confusion.png)

## TF-IDF

The second analysis keeps everything from the word-count analysis (articles, split, vocabulary, balanced logistic regression, grid for C) and replaces raw counts by TF-IDF weights. Term frequencies were damped with a logarithm, article vectors were scaled to unit length, and the inverse document frequencies were computed on the training articles only. C was again chosen from {0.1, 1, 10} by the validation F1 score of the conspiracy class.

The largest value, C = 10, scored best, with a precision of 0.922, a recall of 0.906 and an F1 score of 0.914 on validation. Compared with word counts (F1 0.839), the weighting raised all three numbers. Unlike with raw counts, larger C helped: because TF-IDF vectors have a fixed length, the model can use larger weights without being dominated by long articles. The chosen value is at the edge of the grid, so a still larger C might do slightly better. We kept the grid as planned.

The strongest words resemble those of the word-count model. "Posted", "truth" and "article" point to conspiracy websites, and "killed", "tower" and "response" point to mainstream websites. Part of this signal therefore reflects how websites write and present their articles. TF-IDF improves the use of words, but it does not remove this limitation. Whether the improvement holds on unseen articles from other websites is only decided by the final comparison.



![TF-IDF: validation confusion matrix](../results/figures/tfidf_validation_confusion.png)

## TF-IDF plus extra features

The third analysis asks whether hand-made features improve the TF-IDF model. We added three groups, each motivated by earlier work. N-grams of one to three words capture phrases such as "inside job"; word and bigram features are a strong baseline for text classification (Wang and Manning, 2012). Epistemic markers are counts per 1,000 words of four short hand-made lists (hedges, boosters, hidden-truth phrases, debunking words), following work that ties conspiracy discourse to certainty and concealment (Marini and Jezek, 2024; Islam et al., 2020), although evidence on certainty is mixed (Fong et al., 2021). Style features are capitals, punctuation, quotation marks, sentence length and repetition; fake-news bodies use more capitals and repetition (Horne and Adali, 2017), but such features generalise poorly across domains (Perez-Rosas et al., 2018). We added no emotion scores, as explained earlier: they lie outside the question and their word lists are unreliable here.

We compared five logistic regressions on the same cleaned articles and split: TF-IDF alone, with each group, and with all three. Each feature matrix was built once on the training articles, numeric groups were standardised with training statistics only, and every model chose the regularisation strength (C) and the class weight from the same grid by validation F1 of the conspiracy class. A group passes the threshold if it raises validation F1 by at least 0.01; otherwise plain TF-IDF is kept. With 376 validation articles, smaller differences cannot be told from chance.

The table lists the validation F1 of every model and its change against TF-IDF alone. Only the certainty words reach our threshold.

| Model | Validation F1 | Change against TF-IDF alone |
|---|---|---|
| TF-IDF alone | 0.914 | reference |
| + n-grams (1-3) | 0.916 | +0.002 |
| + certainty words | 0.930 | +0.017 |
| + style | 0.894 | -0.020 |
| + all three groups | 0.887 | -0.027 |

The final tested model therefore contains the certainty words alone; whether it helps on test is answered in the results section. Style and the full combination made the model worse, probably because many weak clues add noise.



![Validation F1 of the five models](../results/figures/ablation_f1.png)



Three reasons explain why the groups add so little. First, the information is already in the words: "cover-up" is a TF-IDF feature as soon as it is a word. Second, the lists reach only part of the data: the hidden-truth list has 11 entries and occurs in 45% of training conspiracy articles against 3% of mainstream ones, but it is absent from more than half of the conspiracy articles. Third, the lists count words and ignore meaning, so an article that denies a cover-up still scores as hidden-truth vocabulary.

## The LLM classifier

The fourth analysis uses a language model instead of a trained classifier. Claude Haiku 5.5 received each article without any examples (zero-shot) and answered the source question: would the article appear on a conspiracy-oriented or on a mainstream website? This is the same target as in the three logistic regression analyses, and the answers are scored against the same dataset labels. The model's output is never treated as ground truth, so the LLM and the logistic regressions are compared on equal terms.

Articles were cut to their first 3,000 words, and the model could write up to 1,000 tokens. The model has no temperature setting, so default sampling was used. Three prompts were tried on the validation articles only: the bare question, the question with short definitions of both website types, and a step-by-step version that warns against judging by topic alone. The prompt with the highest F1 score for the conspiracy class was chosen. Every answer was saved with the article id and a prompt hash, so repeated runs need no new requests.

All three prompts reached almost the same F1 score on the 376 validation articles, and none called a mainstream article conspiracy-oriented. The bare question (v1) and the step-by-step prompt (v3) tied at an F1 score of 0.987, and the short definitions (v2) reached 0.983. With a tie, the first prompt, v1, was kept. Differences this small are within what chance could produce, so the choice of prompt mattered little. The chosen prompt reached a precision of 1.0 and a recall of 0.974: it missed 3 of the 117 conspiracy articles. Answers that could not be read would have been counted as wrong. There were none (0), so the output limit was large enough.

Because sampling cannot be fixed, the validation run of the chosen prompt was repeated once. The share of articles that received the same answer both times was 0.992, which means 99.2% of the answers stayed the same. This share is the only stability measure we report. In total, 30 validation articles were longer than the 3,000-word limit and were cut.



![LLM: validation confusion matrix of the chosen prompt](../results/figures/llm_validation_confusion.png)



These numbers describe validation articles only. Whether the model keeps this level on unseen websites is decided in the final comparison. Two cautions apply. The model may have met these websites or this 9/11 material during its own training (see the limitations discussed later), so a correct answer can be memory rather than reading. And the question is hard from one article alone, because mainstream sites sometimes print conspiracy-friendly pieces and conspiracy sites print plain news.

## Results

All four final models were scored once on the same 410 test articles (96 from conspiracy-oriented websites). The table gives precision, recall and F1 for the conspiracy class with 95% bootstrap confidence intervals (2,000 resamples of documents). The first figure shows the confusion matrices and the second the F1 scores with their intervals.



| Model | Precision | Recall | F1 |
|---|---|---|---|
| Word counts | 0.60 (0.49 to 0.71) | 0.52 (0.43 to 0.62) | 0.56 (0.47 to 0.64) |
| TF-IDF | 0.70 (0.59 to 0.80) | 0.50 (0.40 to 0.60) | 0.58 (0.49 to 0.66) |
| TF-IDF + features | 0.74 (0.63 to 0.84) | 0.52 (0.42 to 0.62) | 0.61 (0.52 to 0.69) |
| LLM | 0.98 (0.95 to 1.00) | 0.67 (0.57 to 0.76) | 0.80 (0.72 to 0.86) |

*Precision, recall and F1 of the conspiracy class on the test articles, with 95% confidence intervals*





![Confusion matrices on the test set (rows: true website type, columns: predicted)](../results/figures/comparison_confusion.png)





![Conspiracy F1 on the test set with 95% confidence intervals](../results/figures/comparison_f1.png)



The LLM has the highest F1 (0.795, interval 0.723 to 0.857), followed by TF-IDF with features (0.610), TF-IDF (0.582) and word counts (0.559). The three logistic regressions have overlapping intervals. Their precision is moderate (0.602 to 0.735) and their recall is about one half (0.500 to 0.521). The LLM is almost never wrong when it says "conspiracy" (precision 0.985, one false alarm among 314 mainstream articles) but still misses a third of the conspiracy articles (recall 0.667).

Three McNemar tests (exact, two-sided) compare pairs of models. Word counts against TF-IDF: 9 articles are right only for word counts and 19 only for TF-IDF (p = 0.087). TF-IDF against TF-IDF with features: 6 against 11 (p = 0.332). The best logistic regression on validation (TF-IDF with features) against the LLM: 3 against 34 (p < 0.0001).

**H1: TF-IDF weighting beats plain word counts.** Not confirmed. TF-IDF has a higher F1 (0.582 against 0.559), but the intervals overlap almost completely and the McNemar test is not significant (p = 0.087). The direction fits the hypothesis; the evidence is too weak to claim it.

**H2: Adding certainty and style features improves the TF-IDF model.** Not confirmed on test. F1 rises from 0.582 to 0.610, mainly through fewer false alarms (precision 0.735 against 0.696), but the difference is within chance (p = 0.332).

**Do the hand-made features help on test?** Slightly, and not provably. The model with the certainty features, the only group that passed the validation threshold, scores a little higher than TF-IDF alone. The intervals overlap and the McNemar test does not separate the two models (p = 0.332), so we cannot say that the features help.

**H3: The LLM beats the best logistic regression.** Confirmed. F1 is 0.795 against 0.610, the intervals do not overlap, and the test is clearly significant (p < 0.0001). The gain comes from precision, not from recall.

**How to read these verdicts.** The test set contains few websites, and articles from one website resemble each other, so the intervals are probably still somewhat optimistic. "Not confirmed" means that the data do not show the effect, not that it is absent. Only the gap to the LLM is large enough to be clearly visible.

**What the numbers say about the models.** The logistic regressions behave alike: they find about half of the conspiracy articles and raise a moderate number of false alarms. Of the 314 mainstream test articles, word counts flag 33, TF-IDF flags 21 and TF-IDF with features flags 18, so the features mainly make the model more careful rather than more sensitive. The LLM hardly ever flags a mainstream article, but it lets one third of the conspiracy articles pass. Why each model misses the articles it misses is taken up in the section on where the models fail.

**Summary.** The ranking by F1 is LLM, TF-IDF with features, TF-IDF, word counts. Only the first place is clearly separated from the rest.

## Where they fail

This is an estimate from one unchecked source: the stance codes come from a single language model and no person has checked them. We examined the test articles on which the best logistic regression and the LLM disagree with the label. The groups are articles missed only by the logistic regression, only by the LLM, and by both. A false positive is a mainstream article called conspiracy, a false negative the reverse. The counts per group are given in the first table of this part.

The labels describe websites, not articles. To separate model errors from label problems, a second model (Claude Sonnet 5.5) coded the stance of every disagreement article towards conspiracy claims about 9/11: endorses, rejects or debunks, reports neutrally, or no claim. The question did not mention the website. A fixed rule, written before the results were read, turns label and stance into a verdict. A conspiracy-site article that does not endorse a claim, or a mainstream article that endorses one, is counted as a label problem. All other cases are model errors. A mainstream article that debunks a theory counts as a model error, because it is still mainstream content.



| set | error_type | errors_total | coded | model wrong | label does not describe the text | not_sampled | stance_endorses | stance_rejects_or_debunks | stance_reports_neutrally | stance_no_claim |
|---|---|---|---|---|---|---|---|---|---|---|
| lr_only_wrong | false positive | 18 | 17 | 16 | 1 | 1 | 1 | 2 | 3 | 11 |
| lr_only_wrong | false negative | 16 | 16 | 1 | 15 | 0 | 1 | 0 | 0 | 15 |
| lr_only_wrong | all | 34 | 33 | 17 | 16 | 1 | 2 | 2 | 3 | 26 |
| llm_only_wrong | false positive | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| llm_only_wrong | false negative | 2 | 2 | 0 | 2 | 0 | 0 | 1 | 1 | 0 |
| llm_only_wrong | all | 3 | 3 | 1 | 2 | 0 | 0 | 1 | 1 | 1 |
| both_wrong | false positive | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| both_wrong | false negative | 30 | 29 | 0 | 29 | 1 | 0 | 0 | 1 | 28 |
| both_wrong | all | 30 | 29 | 0 | 29 | 1 | 0 | 0 | 1 | 28 |
| all | false positive | 19 | 18 | 17 | 1 | 1 | 1 | 2 | 3 | 12 |
| all | false negative | 48 | 47 | 1 | 46 | 1 | 1 | 1 | 2 | 43 |
| all | all | 67 | 65 | 18 | 47 | 2 | 2 | 3 | 5 | 55 |



By this rule most of the errors are label problems, especially missed conspiracy-site articles. Many of them are ordinary news or commentary with no conspiracy claim, so a model that reads only the text has little to find. Mistakes on mainstream articles are mostly model errors: the models react to the topic words and not to the stance of the article. These shares depend on the automatic coding. Of the 67 coded articles, 2 received an answer that could not be read and are left out of the shares.



| cell | doc_id | website | true_label | excerpt |
|---|---|---|---|---|
| lr_only_wrong | C0007c | conspiracydailyupdate.com | conspiracy | Lt Col Tony Shaffer (retired) is highly experienced intelligence officer, and recipient of the Bronze Star, with 30 years of field and operational experience. Tony ... |
| llm_only_wrong | C024ca | neonnettle.com | conspiracy | Soleimani’s Daughter Warns US Soldiers Await Death as Revenge Against 'Crazy Trump' The daughter of slain Iranian terrorist Gen. Qasem Soleimani has issued a threat ... |
| both_wrong | C00459 | neonnettle.com | conspiracy | Ilhan Omar Gets New 2020 Challenger: Iraq War Hero and Veteran Police Officer Radical Democrat Rep. Ilhan Omar (D-MN) has a new challenger for her ... |
| correct_mainstream | M07849 | foxnews.com | mainstream | The former mayor of New York City, Rudy Giuliani, appeared on "Fox & Friends" Wednesday to commemorate the anniversary of the 9/11 terror attacks, and ... |
| correct_conspiracy | C0013c | needtoknow.news | conspiracy | Click on headlines to see full articles If original sources are missing, click on Cached. Star indicates article worth printing for future reference. Camera indicates ... |



The 40 articles of the human check come from 19 websites, 6 conspiracy-oriented and 13 mainstream, with at most 4 articles from any one website.

The automatic stance coding is not verified: no person has checked it yet, and the team plans a human check of 40 of these articles. The label-problem share (47 of 65 readable codings) is therefore an estimate from one unchecked source.

## Limitations

Several limitations qualify the results. First, the label describes the website that published a document, not its content. Conspiracy-oriented websites publish many neutral articles, and mainstream websites sometimes discuss conspiracy theories, so a prediction that disagrees with the label is not necessarily an error of the model. The scores measure how well each method recovers the type of website, not whether a text contains conspiracy claims. This label noise limits the recall of every method and can only be reduced by manual coding of the texts.

Second, the number of websites is small: the validation and test sets each contain only a handful of conspiracy websites. Documents from one website resemble each other, so the confidence intervals, which resample documents, are probably somewhat too narrow, and a different assignment of websites to sets could change the ranking of close methods. The size of this effect is visible in the LLM, whose F1 score on validation was 0.987 and on test 0.795, almost all of it lost in recall (0.974 on validation, 0.667 on test). We did not test the reasons; likely ones are different websites in the two sets and label noise. We also repeated the validation run of the LLM once and report the share of identical answers.



Third, the LLM was trained on large amounts of web text and may know the articles or the reputation of their websites. This cannot be excluded. A part of its performance may therefore rest on prior knowledge that the trained models do not have. Moreover, the LLM saw only the first 3,000 words of each article, and 30 validation articles were longer and were cut, so it may have missed later passages that the logistic regressions, which read the full text, could use.

Fourth, the certainty and doubt features are counts of words from a hand-made list and ignore negation: "no evidence of a cover-up" counts like "a cover-up". Fifth, although names, addresses and repeated lines of the websites were removed, the typical style of a website can remain and can serve as a shortcut for the models. Finally, all results concern one topic, one corpus and one split, so we cannot say whether they transfer. The test set was used during development; every final model was chosen on validation and scored on test once.

## Conclusion

The study compared four methods for deciding whether a full 9/11 article comes from a conspiracy-oriented or a mainstream website. The results are summarised per hypothesis. All numbers refer to the test set of 410 articles and to the F1 score of the conspiracy class, with 95% confidence intervals as given in the results section.

H1 (TF-IDF improves on word counts) is not confirmed. TF-IDF reached an F1 score of 0.582 and plain word counts 0.559. The direction agrees with the hypothesis, but the intervals overlap almost completely and the McNemar test is not significant (p = 0.087).

H2 (certainty and style features improve the TF-IDF model) is not confirmed on test. F1 rose from 0.582 to 0.610, mostly through fewer false alarms, but the difference is within chance (p = 0.332). On validation, the certainty words were the only feature group with a gain of 0.01 or more; n-grams and style features gave no clear gain, and using all three groups together lowered the score.

H3 (the LLM beats the best logistic regression) is confirmed. The LLM reached an F1 score of 0.795 against 0.610 for TF-IDF with features, the intervals do not overlap, and the difference is significant (p < 0.0001). The gain comes from precision (0.985) and not from recall (0.667): the LLM almost never labels a mainstream article as conspiracy but misses one third of the conspiracy articles.

We did not fine-tune a language model or use other classifiers or emotion features, for the reasons given with the word-count analysis. In sum, the methods that we trained ranked as hypothesised, but their differences are small relative to the uncertainty, and only the advantage of the LLM is clear. Because the labels describe websites and the number of websites is small, the findings describe how well each method recovers the type of website, and further work should test other topics, more websites and the human check of the label problems.

## How we used AI

**What the AI produced.** The AI assistant Claude drafted the project plan, wrote the analysis code and the tests, produced the tables and figures, and drafted the text of the course site and of this report. A separate language model, Claude Haiku 5.5, classified the articles as the fourth method under study. A second model, Claude Sonnet 5.5, coded how the articles in which the methods disagree treat conspiracy claims, for the error analysis.

**What the team checked or changed.** The team fixed the research question, full articles as input, the four methods, the feature groups, and the rule that each final model is scored on the test articles once. The article counts were compared with those published for the corpus. Tests were written that fail when a rule is broken, such as fitting on training articles only, a changed split, or a number in the report that is missing from the result files. Human checks are pending: reading the text, spot-checking numbers, and coding 40 articles by hand. None of them is recorded as done.

**What we learned about the work.** Tests confirm that numbers match files, not that they mean the right thing, so a person must still judge the reasoning. The stance codes produced by a language model need a human check before they are relied on. Saving the language model answers makes every figure reproducible without new cost.

## References

- Fong, A., Roozenbeek, J., Goldwert, D., Rathje, S., & van der Linden, S. (2021). The language of conspiracy: A psychological analysis of speech used by conspiracy theorists and their followers on Twitter. *Group Processes & Intergroup Relations, 24*(4). https://journals.sagepub.com/doi/10.1177/1368430220987596
- Horne, B., & Adalı, S. (2017). This just in: Fake news packs a lot in title, uses simpler, repetitive content in text body, more similar to satire than real news. *ICWSM workshops*. https://arxiv.org/abs/1703.09398
- Islam, J., Xiao, L., & Mercer, R. (2020). A lexicon-based approach for detecting hedges in informal text. *LREC*. https://aclanthology.org/2020.lrec-1.380
- Marini, C., & Jezek, E. (2024). What to annotate: Retrieving lexical markers of conspiracy discourse from an Italian-English corpus of Telegram data. *ISA-20 Workshop*. https://aclanthology.org/2024.isa-1.6
- Miani, A., Hills, T., & Bangerter, A. (2022). LOCO: The 88-million-word language of conspiracy corpus. *Behavior Research Methods, 54*, 1794–1817. https://doi.org/10.3758/s13428-021-01698-z
- Pérez-Rosas, V., Kleinberg, B., Lefevre, A., & Mihalcea, R. (2018). Automatic detection of fake news. *COLING*. https://arxiv.org/abs/1708.07104
- Wang, S., & Manning, C. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. *ACL*. https://aclanthology.org/P12-2018
