---
title: Limitations
part: Interpretation
hypothesis: Our results are real but narrow, because the labels describe websites, we have few websites, and the language model may know more than it should.
---
## Explained simply

**The label describes the website, not the article.** An article is "conspiracy" because its website is. A calm news item on a conspiracy website still gets the label, and a mainstream article that rejects a theory is "mainstream". We measure how well a method recovers the kind of website.

**Few websites, and one large one.** After cleaning we have only 37 conspiracy and 79 mainstream websites. The validation and test sets hold six conspiracy websites each, and in the test set one site supplies almost half of the conspiracy articles. So every score comes with a range (a confidence interval).

**The language model may know too much.** The LLM was trained on a huge amount of web text. It may have seen these articles or know a website's reputation. Our own models cannot. Its very high validation score should not be taken at face value.

**The cleaning rules are choices.** We treat two articles as copies when their word weights are more than 90 percent alike. Of a pair across validation and test we kept the validation copy, so no test article repeats one from the other sets.

**The error analysis rests on a second AI.** Whether a disputed article is a "label problem" or a "model error" depends on a fixed rule and on the stance that a second language model read from the text, so the share is a rough estimate.

**Hand-made word lists are crude.** The four lists hold only 47 words and phrases in total, so they miss many ways of sounding certain. They ignore context.

**Style can give a website away.** Even after cleaning, a habit such as writing "posted by" can remain, and a model can use it as a shortcut.

**Long articles are cut.** All four methods read only the first 3,000 words of an article. On the validation set, 30 articles were longer and lost their ends.

**One topic, one collection.** Everything here concerns 9/11 articles from one collection. We cannot say whether the results carry over to other topics.

**The test set.** While the method was being built, earlier versions of the models were scored on the test set. That is like peeking at old exam papers: it can make later choices a little too lucky. To limit this, every final model was chosen on the validation set alone and took the test once.

## The code

This section describes limits of the study, so it has no analysis of its own. The function below is the one that guarantees a website is never in two parts of the split.

<!-- cell: src/fml/split.py#make_split -->

<!-- output: results/tables/split_summary.csv | Documents and websites in each part of the split -->

## Report

**Labels describe websites.** Every article inherits its website's rating, and in the corpus authors' blind check of 60 documents per class, 15% of the conspiracy and 8% of the mainstream documents did not fit their label (Miani et al., 2022). Our scores therefore measure how well a method recovers the type of website, not whether a text contains conspiracy claims; the error analysis and the hand check estimate how many errors are label problems but cannot remove them.

**Few websites, and articles that are not independent.** After cleaning there are 37 conspiracy websites, only 6 each in validation and test. The bootstrap intervals and McNemar's test treat every article as an independent observation, but articles from one website are alike, so the intervals are too narrow and the p-values too small: H1 (p = 0.029) is fragile, while the gap behind H3 is large enough to stand. All results rest on one split.

**The data set and its cleaning.** The corpus was collected by keyword search: the topic filter and the 0.9 copy threshold are judgement calls, habits such as "posted" survive the removal of site clues, and publication dates were not checked. About 30% of our articles are conspiracy articles; where they are rarer, as on most of the web, the same models would raise more false alarms per correct flag. Our reading of Website 4 rests on article titles and on the stance coder, not on a full reading of its texts.

**The language model.** Anthropic does not publish its training data, so we cannot check whether the LLM saw these articles or knows the reputation of their websites; because the corpus and the websites are public and older than the model, it probably did. That it missed 30 of the 44 articles of Website 4, a conspiracy-rated website, suggests that it judged the text rather than recalling the source. Its answers are reproducible only through the stored answers (0.992 alike in a repeated run). The stance coder is also a Claude model and may share its blind spots.

**Use of the test set.** Seven earlier model versions, none reported here, were scored on the test set during development, and on 25 October the team returned to the feature model planned on 16 October after seeing validation results and those test runs. Every reported model was chosen on validation alone and scored on test once.

**Scope.** One topic, one English corpus, the first 3,000 words of each article (30 validation articles were cut), and one kind of trained classifier.

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
