---
title: Conclusion
part: Interpretation
hypothesis: The language model is clearly better than our trained models, while better word weighting and extra features give only small, uncertain gains.
---
## Explained simply

**What we asked.** Can a computer tell from the text of a 9/11 article whether it comes from a conspiracy-oriented or a mainstream website, and which of four ways works best?

**What we found, in everyday words.** The three models we trained ourselves find about half of the conspiracy articles and raise a moderate number of false alarms. Each small improvement, better word weighting and then extra features, moved the score slightly in the hoped-for direction, but the differences are so small that chance could explain them. The language model, which we did not train at all, was clearly the best. It almost never calls a mainstream article "conspiracy", but it still misses a third of the conspiracy articles.

**Why the answer is not a simple "it works".** The label describes the website, and we have few websites. So a score tells us how well a method recovers the kind of website, and we cannot be sure how it would do on new websites or other topics.

**Two things we did not do, and why.**
- **We did not fine-tune a language model.** Fine-tuning means training an AI system further on our articles. It needs much more data, time and money than a fundamentals project has, and it would be harder to explain. Using one cheap model without training makes the comparison simple and fair.
- **We did not use emotion features.** We considered scores for how emotional a text sounds. We dropped them because they need another AI model, which costs extra effort, and because emotional language is not the same as conspiracy language: serious news about 9/11 is full of fear and anger.

**What you can take away.** A better-looking number is not always a better method. Always ask how large the difference is compared with the uncertainty, and what the label really measures.

## The code

The conclusion has no analysis of its own. This function scores one final model on the test set, once, and records that it did.

<!-- cell: src/fml/pipeline.py#evaluate_final -->

<!-- output: results/tables/comparison_test.csv | Test scores of the four final models -->

## Report

The study compared four methods for deciding whether a full 9/11 article comes from a conspiracy-oriented or a mainstream website. The results are summarised per hypothesis. All numbers refer to the test set of 410 articles and to the F1 score of the conspiracy class, with 95% confidence intervals in the table.

<!-- output: results/tables/comparison_test.csv | Precision, recall and F1 on the test set -->

H1 (TF-IDF improves on word counts) is not confirmed. TF-IDF reached an F1 score of 0.582 and plain word counts 0.559. The direction agrees with the hypothesis, but the intervals overlap almost completely and the McNemar test is not significant (p = 0.087).

H2 (certainty and style features improve the TF-IDF model) is not confirmed on test. F1 rose from 0.582 to 0.610, mostly through fewer false alarms, but the difference is within chance (p = 0.332). On validation, the certainty words were the only feature group with a gain of 0.01 or more; n-grams and style features gave no clear gain, and using all three groups together lowered the score.

H3 (the LLM beats the best logistic regression) is confirmed. The LLM reached an F1 score of 0.795 against 0.610 for TF-IDF with features, the intervals do not overlap, and the difference is significant (p < 0.0001). The gain comes from precision (0.985) and not from recall (0.667): the LLM almost never labels a mainstream article as conspiracy but misses one third of the conspiracy articles.

Two alternatives were not pursued. Fine-tuning a language model on the training articles would need more data, computing cost and explanation than the scope of this project allows, and a zero-shot model gives a simple and fair comparison. Emotion scores were dropped because they require a further pre-trained model and because emotional language is not specific to conspiracy writing: reporting on a serious event is emotional by nature. In sum, the methods that we trained ranked as hypothesised, but their differences are small relative to the uncertainty, and only the advantage of the LLM is clear. Because the labels describe websites and the number of websites is small, the findings describe how well each method recovers the type of website, and further work should test other topics, more websites and the human check of the label problems.

## Check yourself

Q: Which hypotheses were confirmed and which were not?
A: Only H3 (the LLM beats the best trained model) was confirmed. H1 and H2 moved in the expected direction, but the differences could be due to chance.

Q: What does "not confirmed" mean here?
A: The data do not show the effect clearly. It does not prove that the effect is absent.

Q: The LLM has very high precision but lower recall. What does that mean for a user?
A: When it says "conspiracy" it is almost always right, but it lets about a third of the conspiracy articles pass.

Q: Why did we not fine-tune a language model?
A: It needs much more data, time and money than this project has, and a model used as it comes gives a simpler and fairer comparison.
