# Conspiracy or mainstream? Classifying 9/11 articles with logistic regression and an LLM

## Question and hypotheses

### Executive summary

We asked whether the text of a full 9/11 article reveals if it comes from a conspiracy-oriented or a mainstream website, comparing three logistic regressions and one low-cost language model, Claude Haiku 5.5, on 410 held-out test articles. The language model reached the highest F1 score for the conspiracy class, 0.795 (95% confidence interval 0.723 to 0.857), against 0.567 for word counts, 0.598 for TF-IDF and 0.560 for TF-IDF with features. H1 (weighting words helps) was supported, H2 (hand-made features help) was not supported, and H3 (the language model beats the best logistic regression) was supported. All models found fewer than seven in ten conspiracy articles on the test websites, so none is reliable enough for automatic flagging.

### Introduction

The attacks of 11 September 2001 are among the most persistent subjects of conspiracy theories: claims that they were an "inside job", a "controlled demolition" or a "false flag" are actively argued for online (Wood & Douglas, 2013) and have been studied as a widely held belief (Swami, Chamorro-Premuzic & Furnham, 2010). In a 2006 survey, 36 percent of Americans found it credible that federal officials helped the attacks happen or deliberately failed to stop them (Stempel, Hargrove & Stempel, 2007). Automatic detection of misleading web content is an active research field (Shu et al., 2017). This study asks whether the text of a full 9/11 article reveals if it was published on a conspiracy-oriented or a mainstream website, using machine-learning methods for text classification.

We compare four methods that form a ladder from simple to rich; the expectation for each is our hypothesis.

1. **Word counts with logistic regression**, the plainest baseline. We expect it to beat always guessing "mainstream", because the two kinds of website use different words.
2. **TF-IDF with logistic regression**, the standard step beyond counting (Salton & Buckley, 1988; Manning, Raghavan & Schütze, 2008). **H1:** it reaches a higher conspiracy-class F1 than word counts, because distinctive words carry more information than common ones.
3. **TF-IDF plus three feature groups** (word sequences, certainty words, style). **H2:** it reaches a higher F1 than TF-IDF alone, because these features capture how conspiracy texts argue.
4. **A low-cost LLM used without training** (zero-shot). **H3:** it reaches a higher F1 than the best logistic regression, because it brings broad knowledge of how texts are written.

We used logistic regression for every trained model because it is simple, well understood and its weights can be read as word lists (Hastie, Tibshirani & Friedman, 2009). We tried no other classifiers, because the question is about the input, not about the best algorithm. We did not fine-tune a language model, which needs more data and money than a small study has, and we left out emotion features, because fear and anger are normal in any report on an attack and so say little about conspiracy content. The whole pipeline, without the language model, runs on an ordinary computer (two to four processor cores, Python 3.11) in about two minutes, about one of which is the feature model; only the language model needs an internet connection and a paid service.

All methods are judged by precision, recall and F1 for the conspiracy class, because most articles are mainstream and the share of correct answers would reward a model that never finds a conspiracy article. Two models are compared with McNemar's test, which uses only the articles on which exactly one of the two is right; it is the recommended test for two classifiers scored on the same test set (Dietterich, 1998). Separate confidence intervals can overlap even when such a paired difference is real, so they describe uncertainty but do not decide. A hypothesis is supported if the difference points in the predicted direction and McNemar's test gives p < 0.05, and not supported otherwise.

The four methods at a glance:



| Method | Input | Reads | Trained on our data | Settings | Runtime | Cost |
|---|---|---|---|---|---|---|
| 1. Word counts + logistic regression | word counts | first 3,000 words of each article | yes | C from [0.01, 0.1, 1, 10, 100], class weight balanced, chosen on validation | a few seconds | free (runs locally) |
| 2. TF-IDF + logistic regression | TF-IDF weights | first 3,000 words of each article | yes | C from [0.01, 0.1, 1, 10, 100], class weight balanced, chosen on validation | a few seconds | free (runs locally) |
| 3. TF-IDF + features + logistic regression | TF-IDF weights, n-grams, certainty words, style | first 3,000 words of each article | yes | C from [0.01, 0.1, 1, 10, 100], class weight balanced, chosen on validation | about a minute | free (runs locally) |
| 4. LLM (claude-haiku-5-5) | the article text itself, with a written question | first 3,000 words of each article | no | prompt v1 of 3, default sampling | about 1.0 seconds per article | about $0.26 per 1,000 articles |

*The four methods at a glance*

## Data and cleaning

The study uses the 9/11 subset of the LOCO corpus (Miani, Hills & Bangerter, 2022), a collection of web pages from websites rated as conspiracy-oriented or mainstream. In their own blind check of 60 documents per subcorpus, 85% of the conspiracy and 92% of the mainstream documents were correctly labelled. We use the full articles of this subset, 3,146 documents from 39 conspiracy and 79 mainstream websites. Each document carries the label of its website, so the label describes the publisher and not the single text. The texts are kept in a private data store and are not redistributed; this report shows only counts and short excerpts. The work used Python with pandas and scikit-learn (Pedregosa et al., 2011).

A classifier takes the easiest signal that separates the classes. This shortcut learning (Geirhos et al., 2020), a form of target leakage (Kaufman et al., 2012), arises when website names, boilerplate, topic differences or copies of an article reveal the label without any property of the writing. All cleaning was therefore applied before any model was fitted, and all four analyses use the same cleaned documents. First, documents that do not mention the attacks in their title or text were removed (3,146 to 2,608), because a model could otherwise learn topics instead of kinds of websites. Second, near-duplicates were removed (2,608 to 2,554): pairs of documents with a cosine similarity of their TF-IDF vectors of at least 0.9. Otherwise a model could meet a near copy of a test article during training. The copy in the earlier set (training, then validation, then test) is kept, never chosen by label. Third, web addresses, e-mail addresses, website names and lines repeated across many documents of one website were deleted from the texts. This step edited 2,498 texts but removed no document.



| Step | Removed | Remaining | Conspiracy removed | Mainstream removed | Conspiracy remaining | Mainstream remaining |
|---|---|---|---|---|---|---|
| all documents | 0 | 3146 | 0 | 0 | 939 | 2207 |
| off-topic articles removed | 538 | 2608 | 162 | 376 | 777 | 1831 |
| near-duplicates removed | 54 | 2554 | 22 | 32 | 755 | 1799 |

*Documents before and after each cleaning step*



After cleaning, 2,554 documents from 37 conspiracy and 79 mainstream websites remain: 755 conspiracy and 1,799 mainstream, so the classes are unbalanced. The shortest document has 101 words, so every document is a real article and no minimum length was needed.

## Splitting by website

The documents were divided into a training set, a validation set and a test set with target shares of 70%, 15% and 15%. The division was made by website: all documents of one website were assigned to the same set, so that no website appears in more than one set. This stops a model from being rewarded for recognising a website it has seen, as when classifying unknown websites. The division was stratified by label: conspiracy and mainstream websites were assigned separately, from the largest to the smallest website, to the set that was furthest short of its target share, with a fixed random seed for ties. The assignment was made once and never changed.



| Set | Class | Websites | Documents | Share of documents |
|---|---|---|---|---|
| train | conspiracy | 25 | 542 | 0.212 |
| train | mainstream | 54 | 1226 | 0.48 |
| validation | conspiracy | 6 | 117 | 0.046 |
| validation | mainstream | 13 | 259 | 0.101 |
| test | conspiracy | 6 | 96 | 0.038 |
| test | mainstream | 12 | 314 | 0.123 |

*Websites and documents per set and class*



Because websites cannot be divided, the shares only approximate the targets. The training, validation and test sets contain, after cleaning, 25, 6 and 6 conspiracy websites (37 in total) and 54, 13 and 12 mainstream websites (79 in total); no website is in more than one set. All model settings were chosen on the validation set, and the test set was used only for the final comparison.

## Word counts

The first analysis represents each article as a bag of words: a vector with the number of times each word occurs, ignoring order. Words were counted on the training articles only, and words occurring in fewer than two training articles were dropped. Logistic regression (Hastie, Tibshirani and Friedman, 2009) gives each word a weight, adds the weights of the words in an article and turns the sum into a probability that the article comes from a conspiracy website. Class weights were balanced because conspiracy articles are the smaller group, and each article was cut to its first 3,000 words, as for all methods. The regularisation strength C, which pulls the weights towards zero, was chosen from {0.01, 0.1, 1, 10, 100} by the validation F1 score of the conspiracy class. All numbers here are validation numbers.

The smallest value, C = 0.01, scored best, with a precision of 0.828, a recall of 0.906 and an F1 score of 0.865. Larger values scored lower (F1 0.797 at C = 100), the typical sign of overfitting: with raw counts the model starts to rely on rare words that only describe the training articles. The best value lies at the lower edge of the grid; the grid was fixed before the run and was not widened afterwards, so an even more cautious model might score slightly higher. Of the 117 conspiracy articles in validation the model found 106, and of the 259 mainstream articles it called 22 conspiracy.



![Word counts: validation confusion matrix](../results/figures/word_counts_validation_confusion.png)



The words with the largest weights are informative, but not all are about content. Words such as "posted", "am" and "article" among the strongest conspiracy words look like habits of how a website presents its articles. Mainstream words such as "killed", "says" and "anniversary" fit factual news reporting. The model therefore partly recognises websites and not only arguments.

## TF-IDF

The second analysis keeps everything from the word-count analysis (articles, split, vocabulary, balanced logistic regression, with its weights and the choice of C by validation F1 as explained in section 4) and replaces raw counts by TF-IDF weights (Salton and Buckley, 1988; Manning, Raghavan and Schütze, 2008). Term frequencies were damped with a logarithm, article vectors were scaled to unit length, and the inverse document frequencies were computed on the training articles only. C was again chosen from {0.01, 0.1, 1, 10, 100} by the validation F1 score of the conspiracy class. All numbers here are validation numbers.

The value C = 10 scored best, with a precision of 0.930, a recall of 0.906 and an F1 score of 0.918. The model called 8 mainstream articles conspiracy. Larger C helped here: because TF-IDF vectors have a fixed length, the model can use larger weights without being dominated by long articles. The chosen value lies inside the grid, and C = 100 scored almost the same (F1 0.909).



![TF-IDF: validation confusion matrix](../results/figures/tfidf_validation_confusion.png)



TF-IDF down-weights words that occur in nearly every article, damps repeated words, and scales each article to the same length. Everything else was kept identical to the word-count analysis so that only the weighting changes.

The strongest words resemble those of the word-count model. "Posted", "truth" and "article" point to conspiracy websites; "killed", "tower" and "york" point to mainstream websites. Part of this signal reflects how websites present their articles.

## TF-IDF plus extra features

The third analysis tests H2 by adding three groups of hand-made features to TF-IDF. N-grams of one to three words capture phrases such as "inside job"; word and bigram features are a strong baseline for text classification (Wang and Manning, 2012). Epistemic markers are counts per 1,000 words of four short hand-made lists: hedges (doubt words such as "perhaps"), boosters (certainty words such as "clearly"), hidden-truth phrases and debunking words, following work that ties conspiracy discourse to certainty and concealment (Marini and Jezek, 2024; Islam et al., 2020; but see Fong et al., 2021). Style features are capitals, punctuation, quotation marks, sentence length and repetition, the last measured by MATTR (moving-average type-token ratio: the share of different words, averaged over windows of 100 words) (Horne and Adali, 2017); they generalise poorly across domains (Pérez-Rosas et al., 2018).

The numeric groups were standardised with training statistics only. In an ablation, which switches one ingredient on or off, six logistic regressions with the same settings were compared on validation: TF-IDF alone, with each group, with all three (the planned final model), and with certainty words and style but without n-grams.

| Model | Validation F1 | Change against TF-IDF alone |
|---|---|---|
| TF-IDF alone | 0.918 | reference |
| + n-grams (1-3) | 0.920 | +0.002 |
| + certainty words | 0.930 | +0.013 |
| + style | 0.909 | -0.009 |
| + all three groups | 0.896 | -0.022 |
| + certainty words and style (no n-grams) | 0.913 | -0.005 |



![Validation F1 of the six models](../results/figures/ablation_f1.png)



The table also shows why the full combination falls below TF-IDF alone. Style lowers validation F1 on its own (0.909). Certainty words and style without n-grams reach 0.913, but all three groups only 0.896: the n-grams add 146,172 columns for 1,768 training articles, which probably makes it easy to fit single articles, and style habits such as capitals may describe single websites rather than conspiracy writing. These are likely explanations that fit the table, not tested causes. Certainty words are the only group that helps alone (0.930). The planned model is the full combination, so its result decides H2. With 376 validation articles, differences of one or two hundredths can be chance, so these are tendencies.

## The LLM classifier

The fourth analysis uses a language model instead of a trained classifier. Claude Haiku 5.5 is the lowest-cost model of Anthropic's Claude 5.5 family; it was called through Anthropic's API in October 2026, one request per article, and had to answer in a fixed short format (label, confidence, one-line reason) of at most 1,000 tokens. The cost, about $0.26 per 1,000 articles, follows from the counted tokens at the list price ($0.10 and $0.50 per million input and output tokens). It received each article without any examples (zero-shot) and answered the source question: would the article appear on a conspiracy-oriented or on a mainstream website? The answers are scored against the same labels as for the logistic regressions.

Articles were cut to their first 3,000 words, as for the other methods; the model has no temperature setting, so default sampling was used. Three prompts were tried on validation only: the bare question (v1), the question with short definitions of both website types (v2), and a step-by-step version that warns against judging by topic (v3).



| variant | documents | precision | recall | f1 |
|---|---|---|---|---|
| v1 | 376 | 1.0 | 0.974 | 0.987 |
| v2 | 376 | 1.0 | 0.966 | 0.983 |
| v3 | 376 | 1.0 | 0.974 | 0.987 |



All three prompts reached almost the same F1 score on the 376 validation articles, and none called a mainstream article conspiracy-oriented. The bare question and the step-by-step prompt tied at 0.987, and the short definitions reached 0.983. With a tie, the first prompt, v1, was kept. The chosen prompt reached a precision of 1.000 and a recall of 0.974: it missed 3 of the 117 conspiracy articles. There were no unreadable answers. A repeated validation run gave the same answer for 0.992 of the articles.



![LLM: validation confusion matrix of the chosen prompt](../results/figures/llm_validation_confusion.png)




Scoring the LLM against website labels is not a fully fair comparison with the trained models. It never saw our labels, so it cannot overfit to them, but it was never taught what they mean. The labels describe websites, not texts, so a "wrong" answer may be a correct reading. And the model may recognise a site from its style or bylines through its own training, which the logistic regressions cannot use.

## Results and why they fail

All four final models were scored once on the same 410 test articles (96 from conspiracy-oriented websites). The table gives precision, recall and F1 for the conspiracy class with 95% bootstrap confidence intervals (Efron and Tibshirani, 1993; 2,000 resamples of documents). The baseline row always answers "mainstream": its accuracy is 0.766, but it finds no conspiracy article, which is why accuracy is not used. All three logistic regressions clearly beat this baseline, as expected for the word-count model in section 1.



| Model | Precision | Recall | F1 |
|---|---|---|---|
| Word counts | 0.607 (0.500 to 0.714) | 0.531 (0.434 to 0.631) | 0.567 (0.478 to 0.650) |
| TF-IDF | 0.721 (0.614 to 0.824) | 0.510 (0.413 to 0.615) | 0.598 (0.508 to 0.682) |
| TF-IDF + features | 0.653 (0.542 to 0.767) | 0.490 (0.394 to 0.590) | 0.560 (0.469 to 0.643) |
| LLM | 0.985 (0.949 to 1.000) | 0.667 (0.570 to 0.759) | 0.795 (0.723 to 0.857) |
| Always guess mainstream | 0.000 | 0.000 | 0.000 |

*Precision, recall and F1 of the conspiracy class on the test articles, with 95% confidence intervals (the baseline has no spread, so no interval)*





![Confusion matrices on the test set (rows: true website type, columns: predicted)](../results/figures/comparison_confusion.png)



The LLM has the highest F1 (0.795, interval 0.723 to 0.857), followed by TF-IDF (0.598), word counts (0.567) and TF-IDF with features (0.560). The LLM is almost never wrong when it says "conspiracy" (precision 0.985) but misses a third of the conspiracy articles (recall 0.667).

Each pair of models is compared with an exact two-sided McNemar test (McNemar, 1947) on the articles where only one of the two is right, as set out in section 1.



| First model | Second model | Right only for the first | Right only for the second | p-value |
|---|---|---|---|---|
| Word counts | TF-IDF | 7 | 19 | 0.029 |
| TF-IDF | TF-IDF + features | 22 | 14 | 0.243 |
| TF-IDF | LLM | 4 | 37 | < 0.0001 |

*The three McNemar tests: articles where only one model is right, and the p-value*



**H1: TF-IDF weighting beats plain word counts.** Supported: 7 articles are right only for word counts and 19 only for TF-IDF (p = 0.029). TF-IDF has the higher F1 (0.598 against 0.567), mainly through fewer false alarms (precision 0.721 against 0.607). It is the smallest of the three effects and would not pass a stricter threshold that corrects for running three tests (0.05 divided by three), so the support is modest.

**H2: Adding the three feature groups improves the TF-IDF model.** Not supported. On validation the model with the extra features scored an F1 of 0.896 against 0.918 for TF-IDF alone, and on test it is lower again (0.560 against 0.598); 22 articles are right only for TF-IDF and 14 only with the features, a difference that chance explains (p = 0.243).

**H3: The LLM beats the best logistic regression.** Supported. F1 is 0.795 against 0.598, the intervals do not overlap, and 4 articles are right only for the best logistic regression (TF-IDF) against 37 only for the LLM (p < 0.0001).

**Why the scores are lower on test than on validation.** Validation F1 was 0.865 for word counts, 0.918 for TF-IDF, 0.896 for TF-IDF with features and 0.987 for the LLM. The likely reason is the websites: the test set has few and different conspiracy websites that no model saw. The table tests this reading. Each row is one conspiracy website in the test set; a recall of 0.5 means that a model found half of that website's articles.



![Confusion matrices on validation (top) and test (bottom); rows: true website type, columns: predicted](../results/figures/validation_vs_test_confusion.png)





| Website | Test articles | Recall: Word counts | Recall: TF-IDF | Recall: TF-IDF + features | Recall: LLM |
|---|---|---|---|---|---|
| Website 1 | 19 | 0.684 | 0.579 | 0.684 | 0.947 |
| Website 2 | 6 | 0.833 | 0.833 | 0.833 | 1.0 |
| Website 3 | 19 | 0.79 | 0.842 | 0.737 | 0.947 |
| Website 4 | 44 | 0.25 | 0.204 | 0.204 | 0.318 |
| Website 5 | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| Website 6 | 7 | 0.857 | 1.0 | 0.714 | 1.0 |

*Recall of each model on the conspiracy articles of each test website*



One site, Website 4, supplies 44 of the 96 conspiracy test articles, and all four models find at most about a third of them (recall 0.204 to 0.318). Without those 44 articles the F1 is 0.640 for word counts, 0.721 for TF-IDF, 0.661 for TF-IDF with features and 0.971 for the LLM. On every site with more than one article the LLM finds at least as many articles as each logistic regression.

**Where the models fail, and why.** We examined the test articles on which the best logistic regression or the LLM disagrees with the label. A second model (Claude Sonnet 5.5) coded the stance of each towards conspiracy claims about 9/11, without being told the website. A fixed rule counts a conspiracy-site article without an endorsed claim, or a mainstream article with one, as a label problem; the rest are model errors.

Of the 70 wrongly classified articles, 69 have a readable coding: 49 are label problems and 20 are model errors. One article was left out because the second model's answer was not in the required form. The table gives the three groups with the share of coded articles that are label problems.



| Group | Articles | Coded | Label problems | Label-problem share |
|---|---|---|---|---|
| Only the logistic regression is wrong | 37 | 37 | 18 | 0.486 |
| Only the LLM is wrong | 4 | 4 | 3 | 0.75 |
| Both are wrong | 29 | 28 | 28 | 1.0 |

*Test articles wrongly classified, by who is wrong, and the share that are label problems*



These shares are estimates from one automatic coder. That all 28 coded articles missed by both models count as label problems follows partly from the rule itself: a conspiracy-site article without a conspiracy claim is a label problem by definition.

Two test articles show the pattern. A conspiracy-site article that both models called mainstream begins "A military judge has set the trial date for five men accused of masterminding the 9/11 attacks": a plain court report without a conspiracy claim, so this is a label problem. A mainstream article opening "Not only is there a smoking gun in the 28 pages" was called conspiracy by the logistic regression and mainstream by the LLM; the stance coder found no conspiracy claim in it, so this is a model error: the trained model reacted to wording that only sounds like conspiracy writing.

The label-problem share rests on automatic coding by a model of the same family as the classifier under test, so it is an estimate. Because coder and classifier come from the same model family, they may share blind spots, which would turn some of the LLM's errors into apparent label problems. To measure how far the coding can be trusted, the team codes 40 validation articles blind, each article by two of the five members (website and automatic answer hidden; validation keeps the test set untouched); their agreement with the automatic coding will be added after the team's check.

## Limitations

First, the label describes the publishing website, not the content, so a prediction that disagrees with it is not necessarily an error. The scores measure how well each method recovers the type of website, not whether a text contains conspiracy claims.

Second, the number of websites is small. The validation and test sets each contain only six conspiracy websites, and in the test set one of them supplies 44 of the 96 conspiracy articles. The confidence intervals resample single articles as if they were independent, but articles from one website resemble each other, so the intervals are probably too narrow.

Third, the LLM was trained on large amounts of web text and may know the articles or the reputation of their websites, which the trained models cannot use. The stance coder behind the label-problem share is also a Claude model and may share its blind spots, so that share is a rough estimate. All four methods read only the first 3,000 words of each article, and 30 validation articles were longer and were cut.

Fourth, some choices are thresholds that could be set differently, such as the cosine similarity of 0.9 for copies, which removed 54 articles. The certainty and doubt features ignore negation and rest on four short word lists (47 entries in total). Website style can serve as a shortcut.

Finally, all results concern one topic, one corpus and one split, so we cannot say whether they transfer to other topics or websites. The test set was also used during development: seven earlier model versions, none of them reported here, were scored on it. Every final model reported here was chosen on validation alone and scored on the test set once.

## Conclusion

The study compared four methods for deciding whether a full 9/11 article comes from a conspiracy-oriented or a mainstream website. All numbers are conspiracy-class F1 scores on the 410 test articles.

H1 (TF-IDF improves on word counts) is supported, modestly: F1 0.598 against 0.567 (p = 0.029); with a stricter threshold for three tests it would not count.

H2 (adding the three feature groups improves the TF-IDF model) is not supported: on test the F1 was 0.560 against 0.598, within chance (p = 0.243). Certainty words alone gave the best validation F1 (0.930), but the planned full combination did not help.

H3 (the zero-shot LLM beats the best logistic regression) is supported: F1 0.795 against 0.598 (p < 0.0001). It misses one third of the conspiracy articles.

In sum, only the advantage of the LLM is clear, and no method finds enough conspiracy articles to be used alone for automatic flagging. All three logistic regressions beat the baseline that always answers mainstream. Further work should test other topics and more websites; the results of the team's hand check of the automatic stance coding will be added once it is complete.

## How we used AI

**Tools and set-up.** The code, tests, tables, figures and drafts of this report and of the course website were produced with Claude Code, an AI coding assistant, running as a loop of three roles of Claude Sonnet 5.5: a planner that turned the team's written requirements into small tasks, an implementer that carried out one task at a time and ran the tests, and a reviewer that, in a fresh session, scored the result against a fixed checklist. A separate Claude session helped the project owner to set up this loop and the website. Claude Haiku 5.5 (the fourth classifier) and the stance coder are part of the study, not tools.

**How the team steered the work.** The work ran in three cycles. After each cycle team members read the website, which shows the code that ran next to its output, and their comments became the next cycle's requirements. Decisions that changed the study came from the team: the use of full articles, the four methods, leaving out emotion features, removing off-topic articles and near-duplicates before any model, and, after the second cycle, giving all three logistic regressions identical settings and the same text as the language model.

**How the output was checked.** Automated tests fail when a rule is broken, for example when a model learns from more than the training articles, the split changes, the test set is used outside the final scoring, or a number in this report is missing from the result files. The whole analysis was run twice with identical results. Judgements that tests cannot make, such as whether the automatic stance coding is right, are checked by people through the hand check described in the error analysis.

**What we learned.** The AI was fast and thorough at code and tables, but it followed its instructions literally: it reported an unplanned feature selection and an untested explanation until the team asked for a fair, planned comparison and for evidence. A test shows that a number matches a file, not that it answers the right question, so the reasoning stays the team's responsibility.

## References

- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923.
- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.
- Fong, A., Roozenbeek, J., Goldwert, D., Rathje, S., & van der Linden, S. (2021). The language of conspiracy: A psychological analysis of speech used by conspiracy theorists and their followers on Twitter. *Group Processes & Intergroup Relations, 24*(4). https://journals.sagepub.com/doi/10.1177/1368430220987596
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence, 2*, 665–673.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2nd ed.). Springer.
- Horne, B., & Adalı, S. (2017). This just in: Fake news packs a lot in title, uses simpler, repetitive content in text body, more similar to satire than real news. *ICWSM workshops*. https://arxiv.org/abs/1703.09398
- Islam, J., Xiao, L., & Mercer, R. (2020). A lexicon-based approach for detecting hedges in informal text. *LREC*. https://aclanthology.org/2020.lrec-1.380
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4). https://doi.org/10.1145/2382577.2382579
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press.
- Marini, C., & Jezek, E. (2024). What to annotate: Retrieving lexical markers of conspiracy discourse from an Italian-English corpus of Telegram data. *ISA-20 Workshop*. https://aclanthology.org/2024.isa-1.6
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157.
- Miani, A., Hills, T., & Bangerter, A. (2022). LOCO: The 88-million-word language of conspiracy corpus. *Behavior Research Methods, 54*, 1794–1817. https://doi.org/10.3758/s13428-021-01698-z
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Pérez-Rosas, V., Kleinberg, B., Lefevre, A., & Mihalcea, R. (2018). Automatic detection of fake news. *COLING*. https://arxiv.org/abs/1708.07104
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523.
- Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake news detection on social media: A data mining perspective. *ACM SIGKDD Explorations Newsletter, 19*(1), 22–36. https://arxiv.org/abs/1708.01967
- Stempel, C., Hargrove, T., & Stempel, G. H., III (2007). Media use, social structure, and belief in 9/11 conspiracy theories. *Journalism & Mass Communication Quarterly, 84*(2). https://doi.org/10.1177/107769900708400210
- Swami, V., Chamorro-Premuzic, T., & Furnham, A. (2010). Unanswered questions: A preliminary investigation of personality and individual difference predictors of 9/11 conspiracist beliefs. *Applied Cognitive Psychology, 24*(6), 749–761. https://doi.org/10.1002/acp.1583
- Wang, S., & Manning, C. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. *ACL*. https://aclanthology.org/P12-2018
- Wood, M. J., & Douglas, K. M. (2013). "What about building 7?" A social psychological study of online discussion of 9/11 conspiracy theories. *Frontiers in Psychology, 4*, 409. https://doi.org/10.3389/fpsyg.2013.00409
