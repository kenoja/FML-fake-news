---
title: Limitations
part: Interpretation
hypothesis: Our results are real but narrow, because the labels describe websites, we have few websites, and the language model may know more than it should.
---
## Explained simply

**The label describes the website, not the article.** An article is "conspiracy" because its website is. A calm news item on a conspiracy website still gets the label, and a mainstream article that rejects a theory is "mainstream". So a model that disagrees with the label is not always wrong. We measure how well a method recovers the kind of website.

**Few websites, and one large one.** We have only 39 conspiracy and 79 mainstream websites. The validation and test sets hold only six conspiracy websites each, and in the test set one site supplies almost half of the conspiracy articles. So we give a range (a confidence interval) with every score.

**The language model may know too much.** The LLM was trained on a huge amount of web text. It may have seen these articles or may know a website's reputation from its name or tone. Our own models cannot use such knowledge. Its very high validation score was reached on websites without a hard case like the large test site, so it should not be taken at face value.

**The cleaning rules are choices.** We treat two articles as copies when their word weights are more than 90 percent alike. A stricter or looser line would remove a few more or fewer articles. Of a pair across validation and test we kept the validation copy, so no test article repeats one from the other sets.

**The error analysis rests on a second AI.** Whether a disputed article is a "label problem" or a "model error" depends on a fixed rule and on the stance that a second language model read from the text. The second reader is itself a Claude model, so the share is a rough estimate. A different rule, or a different reader, would give a different share.

**Hand-made word lists are crude.** They count words and ignore context: "no evidence of a cover-up" counts like "it was a cover-up".

**Style can give a website away.** Even after we delete names, web addresses and repeated menu lines, a typical habit such as writing "posted by" can remain, and a model can use it as a shortcut. Splitting by website cannot remove style signals that similar sites share.

**Long articles are cut.** All four methods read only the first 3,000 words of an article. On the validation set, 30 articles were longer and lost their ends.

**One topic, one collection.** Everything here concerns 9/11 articles from one collection. We cannot say whether the results carry over to other topics.

**The test set.** The test set was used during development; every final model was chosen on validation and scored on test once.

## The code

This section describes limits of the study, so it has no analysis of its own. The function below is the one that guarantees a website is never in two parts of the split.

<!-- cell: src/fml/split.py#make_split -->

<!-- output: results/tables/split_summary.csv | Documents and websites in each part of the split -->

## Report

Several limitations qualify the results. First, the label describes the website that published a document, not its content. Conspiracy-oriented websites publish many neutral articles, and mainstream websites sometimes discuss conspiracy theories, so a prediction that disagrees with the label is not necessarily an error of the model. The scores measure how well each method recovers the type of website, not whether a text contains conspiracy claims. This label noise limits the recall of every method.

Second, the number of websites is small. The validation and test sets each contain only six conspiracy websites, and in the test set one of them supplies 44 of the 96 conspiracy articles. Documents from one website resemble each other, so the confidence intervals are probably too narrow.

Third, the LLM was trained on large amounts of web text and may know the articles or the reputation of their websites, which the trained models cannot use. Its validation F1 of 0.987 was reached on validation websites where it found 0.905 to 1.000 of the conspiracy articles, with no site as hard as the large test site (recall 0.318), and the stance coder behind the label-problem share is also a Claude model, so that share is a rough estimate. All four methods read only the first 3,000 words of each article, and 30 validation articles were longer and were cut.

Fourth, some choices are thresholds that could be set differently, such as the cosine similarity of 0.9 for copies, which removed 54 articles. The split between label problems and model errors is an estimate. The certainty and doubt features ignore negation, and website style can serve as a shortcut.

Finally, all results concern one topic, one corpus and one split, so we cannot say whether they transfer to other topics or websites. The test set was used during development; every final model was chosen on validation and scored on test once.

## Check yourself

Q: Why is a model that disagrees with the label not always wrong?
A: The label describes the website. A neutral article on a conspiracy website has the label "conspiracy", and a mainstream article that discusses a theory has the label "mainstream", although the text may suggest otherwise.

Q: Why can the good score of the LLM be misleading?
A: It was trained on web text and may know the articles or the reputation of their websites, which our own models cannot use.

Q: Why do few websites make the results less certain?
A: Articles from one website resemble each other, so a single unusual website can move a score, and another division of websites could give other numbers.

Q: Name one way a model can recognise a website even though we deleted its name.
A: Through its habits of style, for example typical phrases such as "posted by" or the way a site formats its articles.

Q: Why is the share of "label problems" only an estimate?
A: It depends on a rule we chose and on a second AI reading each article, so another rule or reader would give another share.
