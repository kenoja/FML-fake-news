---
title: Conclusion
part: Interpretation
hypothesis: The zero-shot LLM reaches a higher conspiracy-class F1 on the test set than the best logistic regression, and the McNemar test gives p < 0.05.
---
## Explained simply

**What we asked.** Can a computer tell from the text of a 9/11 article whether it comes from a conspiracy-oriented or a mainstream website, and which of four ways works best?

**One plain answer per hypothesis.**
- **H1, weighting words (TF-IDF) beats counting them: supported, but only just.** TF-IDF raised the score slightly, mainly because it raised fewer false alarms. The test says the gain is probably not luck; it is the smallest of the three effects.
- **H2, the three feature groups improve TF-IDF: no.** Phrases, certainty words and style clues did not help. The model with all three did slightly worse than TF-IDF alone, on the validation articles and on the test articles. The difference on test is small enough for chance.
- **H3, the zero-shot LLM reaches a higher conspiracy-class F1 on the test set than the best logistic regression, and the McNemar test gives p < 0.05: supported.** It almost never calls a mainstream article "conspiracy", but it still misses about a third of the conspiracy articles.

**The verdict on our hand-made features.** They did not help. The section on feature engineering looks at why.

**Why the answer is not a simple "it works".** The label describes the website, and we have few websites. So a score tells us how well a method recovers the kind of website, and we cannot be sure how it would do on new websites or other topics. None of the four finds enough conspiracy articles to be trusted on its own.

**Two things we did not do, and why.**
- **We did not fine-tune a language model.** Fine-tuning means training an AI system further on our articles. It needs much more data, time and money than a fundamentals project has, and it would be harder to explain.
- **We did not use emotion features.** Emotional language is not the same as conspiracy language: serious news about 9/11 is full of fear and anger.

**What you can take away.** A better-looking number is not always a better method. Always ask how large the difference is compared with the uncertainty, and what the label really measures.

## The code

The conclusion has no analysis of its own. This function scores one final model on the test set, once, and records that it did.

<!-- cell: src/fml/pipeline.py#evaluate_final -->

## Report

The study compared four methods for deciding whether a full 9/11 article comes from a conspiracy-oriented or a mainstream website. All three logistic regressions beat the baseline that always answers mainstream. H1 (TF-IDF beats word counts) was supported, narrowly; H2 (the three feature groups help) was not supported; H3 (the LLM beats the best logistic regression) was supported, with an F1 of 0.795 against 0.598, although it still misses one third of the conspiracy articles.

The results allow a clear statement about this corpus and its website labels: a cheap language model recovers the type of website far better than word statistics, and hand-made features did not close the gap. They do not show that any method recognises conspiracy content, because the labels describe websites, and they rest on six conspiracy websites in the test set. No method finds enough conspiracy articles to be used alone for automatic flagging. Further work should test other topics and more websites; the results of the team's hand check of the automatic stance coding will be added once it is complete.

## Check yourself

Q: Which hypotheses were supported and which were not?
A: H3 (the LLM beats the best trained model) was supported. H1 (TF-IDF beats word counts) was supported, but only just: the test is significant, and it is the smallest effect. H2 (the three feature groups improve TF-IDF) was not supported.

Q: What does "not supported" mean here?
A: The data do not show the effect clearly. It does not prove that the effect is absent.

Q: The LLM has very high precision but lower recall. What does that mean for a user?
A: When it says "conspiracy" it is almost always right, but it lets about a third of the conspiracy articles pass.

Q: Why did we not fine-tune a language model?
A: It needs much more data, time and money than this project has, and a model used as it comes gives a simpler and fairer comparison.

Q: Did the hand-made features help?
A: No. The model with all three groups scored slightly lower than TF-IDF alone, on validation and on test.
