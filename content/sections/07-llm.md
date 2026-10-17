---
title: The LLM classifier
part: Methods
hypothesis: A cheap language model, given no training at all, can tell conspiracy-oriented websites from mainstream ones about as well as the models we trained.
---
## Explained simply

**What an LLM is.** A large language model (LLM) is a program that has read a huge amount of text and can answer questions in writing. We use one cheap model, Claude Haiku 5.5, and ask it the question the other analyses answer: does this article come from a conspiracy-oriented or a mainstream website?

**Zero-shot.** "Zero-shot" means we show the model no solved examples. We do not train it. We only give it an instruction and the article, like handing a stranger an article and asking "which kind of website printed this?" The instruction is called the **prompt**.

**Why the prompt matters.** Small changes in wording can change the answers, so we try three prompts: the bare question, the question with a short description of both website types, and the question with step-by-step advice and a warning not to judge by topic alone. We try them on the validation articles only and keep the one with the best F1 score. The test articles play no part in this choice.

**Why no fine-tuning.** Fine-tuning means training the model further on our articles. That is costly, and we want to know what a cheap, ready-made model can do without help from our data.

**Tokens.** A token is a small piece of text, about three quarters of a word. Limits on models are counted in tokens. We cut every article to its first 3,000 words, the same for all articles. The model may write up to 1,000 tokens, so no answer is cut off in the middle.

**Why default sampling.** A model picks among likely next words with a little randomness. Some models have a setting called temperature to turn the randomness down. This model has no temperature setting, so we use its default behaviour. That is why we repeat the validation run once: it shows how often the answers stay the same.

**Why the website question.** The dataset labels describe the website that published an article, not what the article claims. A mainstream site can print a conspiracy story, and a conspiracy site can print plain news. So we ask about the website. The LLM and the logistic regressions are both judged against the same dataset labels, and the LLM's answer is never treated as the truth.

**Answers we cannot read.** The model must answer in a fixed format. An answer that breaks it counts as wrong, and we report how many there were as one number.

**Saved answers.** Every answer is saved with the article's id and the prompt, never the article text, so a rerun needs no new questions. If an answer is missing and there is no key, the code stops with an error.

## The code

The three prompts. All ask for the same short answer format.

<!-- cell: src/fml/llm.py:30-59 -->

Reading the model's answer. Anything that does not fit the format becomes "unreadable".

<!-- cell: src/fml/llm.py#parse_response -->

Classifying articles: saved answers first, the model only for answers that are missing.

<!-- cell: src/fml/llm.py#classify -->

Scoring one prompt against the dataset labels on the validation articles.

<!-- cell: src/fml/pipeline.py#llm_scores -->

The analysis: try the three prompts, keep the best by F1, repeat it once.

<!-- cell: src/fml/pipeline.py#llm_stage -->

<!-- output: results/tables/llm_prompt_variants.csv -->
<!-- output: results/figures/llm_validation_confusion.png | LLM: validation confusion matrix of the chosen prompt -->
<!-- output: results/metrics.json#llm.chosen_prompt -->
<!-- output: results/metrics.json#llm.validation -->
<!-- output: results/metrics.json#llm.unreadable -->
<!-- output: results/metrics.json#llm.repeat_agreement -->

## Report

The fourth analysis uses a language model instead of a trained classifier. Claude Haiku 5.5 received each article without any examples (zero-shot) and answered the source question: would the article appear on a conspiracy-oriented or on a mainstream website? This is the same target as in the three logistic regression analyses, and the answers are scored against the same dataset labels. The model's output is never treated as ground truth, so the LLM and the logistic regressions are compared on equal terms.

Articles were cut to their first 3,000 words, and the model could write up to 1,000 tokens. The model has no temperature setting, so default sampling was used. Three prompts were tried on the validation articles only: the bare question, the question with short definitions of both website types, and a step-by-step version that warns against judging by topic alone. The prompt with the highest F1 score for the conspiracy class was chosen.

<!-- output: results/tables/llm_prompt_variants.csv -->

All three prompts reached almost the same F1 score on the 376 validation articles, and none called a mainstream article conspiracy-oriented. The bare question (v1) and the step-by-step prompt (v3) tied at an F1 score of 0.987, and the short definitions (v2) reached 0.983. With a tie, the first prompt, v1, was kept. Differences this small are within what chance could produce, so the choice of prompt mattered little. The chosen prompt reached a precision of 1.0 and a recall of 0.974: it missed 3 of the 117 conspiracy articles. Answers that could not be read would have been counted as wrong. There were none (0), so the output limit was large enough.

<!-- output: results/metrics.json#llm.validation -->
<!-- output: results/metrics.json#llm.unreadable -->

Because sampling cannot be fixed, the validation run of the chosen prompt was repeated once. The share of articles that received the same answer both times was 0.992, which means 99.2% of the answers stayed the same. This share is the only stability measure we report. Thirty validation articles were longer than the word limit and were cut.

<!-- output: results/metrics.json#llm.repeat_agreement -->

<!-- output: results/figures/llm_validation_confusion.png | LLM: validation confusion matrix of the chosen prompt -->

These numbers describe validation articles only. Whether the model keeps this level on unseen websites is decided in the final comparison. Two cautions apply. The model may have met these websites or this 9/11 material during its own training, so a correct answer can be memory rather than reading. And the question is hard from one article alone, because mainstream sites sometimes print conspiracy-friendly pieces and conspiracy sites print plain news.

## Check yourself

Q: What does "zero-shot" mean?
A: The model gets no solved examples and no training from us. It only receives an instruction and the article.

Q: Why do we ask about the website and not about whether the article believes a conspiracy theory?
A: The dataset labels describe the website. To compare the LLM and the logistic regressions fairly, both must be judged against those same labels.

Q: Why do we repeat the validation run once?
A: The model has no temperature setting, so its answers can vary. The share of equal answers shows how much.

Q: The LLM was not trained on our articles. Does that make its score trustworthy?
A: Not fully. It may have seen the websites in its own training, so its score may partly reflect memory.
