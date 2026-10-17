---
title: Limitations
part: Interpretation
hypothesis: Our results are real but narrow, because the labels describe websites, we have few websites, and the language model may know more than it should.
---
## Explained simply

**Why this section matters.** Naming the limits is part of honest work, and it is where your criticism is most welcome.

**The label describes the website, not the article.** An article is "conspiracy" because its website is. A calm news item on a conspiracy website still gets the label, and a mainstream article that explains and rejects a theory is "mainstream". So when a model disagrees with the label, the model is not always wrong. What we measure is how well a method recovers the kind of website, which is not the same as detecting conspiracy thinking in a text.

**Few websites.** We have only 39 conspiracy and 79 mainstream websites, and the validation and test sets contain only a handful of conspiracy websites each. Articles from one website resemble each other, so one unusual website can move a score. That is why we give a range (a confidence interval) with every score and do not read small differences as real.

**The language model may know too much.** The LLM was trained on a huge amount of web text. It may have seen these articles or may know a website's reputation from its name or tone. A human reader or our own models cannot use such knowledge. We checked a small sample on validation and the model named the true website for 30% of it. That does not prove it never uses such knowledge, so its good score may be partly due to prior knowledge.

**Hand-made word lists are crude.** They count words and ignore context: "no evidence of a cover-up" counts like "it was a cover-up".

**Can a model recognise a website without its name?** (Team question 6.) Yes, in part. Even after we delete names, web addresses and repeated menu lines, the typical style of a website can remain, for example the habit of writing "posted by". A model can use such habits as a shortcut. The split by website protects against the worst case, since test websites are new to the model, but it cannot remove style signals that sites of the same kind share. We cannot fully separate them from real differences in content.

**Long articles are cut.** The language model reads only the first 3,000 words of an article. On the validation set, 30 articles were longer and lost their ends, so the model may have missed something that the other models saw.

**One topic, one collection.** Everything here concerns 9/11 articles from one collection. We cannot say whether the results carry over to other topics.

## The code

This section describes limits of the study, so it has no code of its own. The tests that protect the split by website are in the repository. The function below is the one that guarantees a website is never in two parts.

<!-- cell: src/fml/split.py#make_split -->

<!-- output: results/metrics.json#llm.repeat_agreement -->
<!-- output: results/metrics.json#llm.truncated_docs -->
<!-- output: results/metrics.json#errors.prior_knowledge_probe.identified -->
<!-- output: results/metrics.json#errors.prior_knowledge_probe.n -->

## Report

Several limitations qualify the results. First, the label describes the website that published a document, not its content. Conspiracy-oriented websites publish many neutral articles, and mainstream websites sometimes discuss conspiracy theories, so a prediction that disagrees with the label is not necessarily an error of the model. The scores measure how well each method recovers the type of website, not whether a text contains conspiracy claims. This label noise limits the recall of every method and can only be reduced by manual coding of the texts.

Second, the number of websites is small: the validation and test sets each contain only a handful of conspiracy websites. Documents from one website resemble each other, so the confidence intervals, which resample documents, are probably somewhat too narrow, and a different assignment of websites to sets could change the ranking of close methods. The size of this effect is visible in the LLM, whose F1 score on validation was 0.987 and on test 0.795. We also repeated the validation run of the LLM once and report the share of identical answers below.

<!-- output: results/metrics.json#llm.repeat_agreement | Share of identical answers in a repeated validation run -->

Third, the LLM was trained on large amounts of web text and may know the articles or the reputation of their websites. This cannot be excluded. In a probe, the model named the true website for 12 of 40 validation articles (30%), so a part of its performance may rest on prior knowledge that the trained models do not have. Moreover, the LLM saw only the first 3,000 words of each article, and 30 validation articles were longer and were cut, so it may have missed later passages that the logistic regressions, which read the full text, could use.

Fourth, the certainty and doubt features are counts of words from a hand-made list and ignore negation: "no evidence of a cover-up" counts like "a cover-up". Fifth, although names, addresses and repeated lines of the websites were removed, the typical style of a website can remain and can serve as a shortcut for the models. The split by website prevents a model from profiting from sites it has seen, but not from style habits that websites of the same kind share. Finally, all results concern one topic, one corpus and one split, so we cannot say whether they transfer. The test set was used during development; every final model was chosen on validation and scored on test once.

## Check yourself

Q: Why is a model that disagrees with the label not always wrong?
A: The label describes the website. A neutral article on a conspiracy website has the label "conspiracy", and a mainstream article that discusses a theory has the label "mainstream", although the text may suggest otherwise.

Q: Why can the good score of the LLM be misleading?
A: It was trained on web text and may know the articles or the reputation of their websites, which our own models cannot use.

Q: Why do few websites make the results less certain?
A: Articles from one website resemble each other, so a single unusual website can move a score, and another division of websites could give other numbers.

Q: Name one way a model can recognise a website even though we deleted its name.
A: Through its habits of style, for example typical phrases such as "posted by" or the way a site formats its articles.
