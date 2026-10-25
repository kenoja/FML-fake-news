---
title: Conclusion
part: Interpretation
hypothesis: The language model is clearly better than our trained models, better word weighting helps a little, and the hand-made features do not help.
---
## Explained simply

**What we asked.** Can a computer tell from the text of a 9/11 article whether it comes from a conspiracy-oriented or a mainstream website, and which of four ways works best?

**One plain answer per hypothesis.**
- **H1, weighting words (TF-IDF) beats counting them: yes, a little.** TF-IDF raised the score slightly, mainly because it raised fewer false alarms. The gain is small but not just luck.
- **H2, the three feature groups improve TF-IDF: no.** Phrases, certainty words and style clues did not help. The model with all three did slightly worse than TF-IDF alone, on the validation articles and on the test articles. The difference on test is small enough for chance.
- **H3, the language model differs from the best model we trained: yes, it is clearly better.** It almost never calls a mainstream article "conspiracy", but it still misses about a third of the conspiracy articles.

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

The study compared four methods for deciding whether a full 9/11 article comes from a conspiracy-oriented or a mainstream website. The results are summarised per hypothesis. All numbers are F1 scores of the conspiracy class on the 410 test articles.

H1 (TF-IDF improves on word counts) is confirmed, narrowly: F1 0.598 against 0.567 (p = 0.029).

H2 (adding the three feature groups improves the TF-IDF model) is not confirmed: on test the F1 was 0.560 against 0.598, within chance (p = 0.243). The hand-made features did not help.

H3 (the LLM differs from the best logistic regression) is confirmed, and the LLM is better: F1 0.795 against 0.598 (p < 0.0001). It almost never labels a mainstream article as conspiracy but misses one third of the conspiracy articles.

In sum, only the advantage of the LLM is clear, and no method finds enough conspiracy articles to be used alone for automatic flagging. Because the labels describe websites and the number of websites is small, the findings describe how well each method recovers the type of website. Further work should test other topics and more websites, and should add the human check of the label problems.

## Check yourself

Q: Which hypotheses were confirmed and which were not?
A: H1 (TF-IDF beats word counts) was confirmed narrowly, and H3 (the LLM differs from the best trained model, being better) clearly. H2 (the three feature groups improve TF-IDF) was not confirmed.

Q: What does "not confirmed" mean here?
A: The data do not show the effect clearly. It does not prove that the effect is absent.

Q: The LLM has very high precision but lower recall. What does that mean for a user?
A: When it says "conspiracy" it is almost always right, but it lets about a third of the conspiracy articles pass.

Q: Why did we not fine-tune a language model?
A: It needs much more data, time and money than this project has, and a model used as it comes gives a simpler and fairer comparison.

Q: Did the hand-made features help?
A: No. The model with all three groups scored slightly lower than TF-IDF alone, on validation and on test.
