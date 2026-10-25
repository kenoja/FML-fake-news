---
title: The LLM classifier
part: Methods
hypothesis: We expect a cheap language model with no training to do roughly as well as the trained models, because it has read a great deal of text and can judge how an article is written.
---
## Explained simply

**What an LLM is.** A large language model (LLM) is a program that has read a huge amount of text and can answer questions in writing. We use one cheap model, Claude Haiku 5.5, and ask it the question the other analyses answer: does this article come from a conspiracy-oriented or a mainstream website? Section 01 explains why it is the fourth method.

**Zero-shot.** "Zero-shot" means we show the model no solved examples and do not train it. It gets an instruction and the article, like a stranger asked "which kind of website printed this?" The instruction is the **prompt**.

**Three prompts.** Small changes in wording can change the answers, so we try three: the bare question, the question with a short description of both website types, and step-by-step advice with a warning not to judge by topic alone. We try them on the validation articles only, report all three, and keep the best by F1.

**Tokens and limits.** A token is a small piece of text, about three quarters of a word. Every article is cut to its first 3,000 words, the same for all four methods. The model may write up to 1,000 tokens, so no answer is cut off. This model has no temperature setting, so we use default sampling and repeat the validation run once to see how often the answers stay the same.

**Answers we cannot read, and saved answers.** An answer that breaks the fixed format counts as wrong, and we report how many there were. Every answer is saved with the article's id and the prompt, never the text, so a rerun needs no new questions.

**Is the comparison fair?** Not entirely. The LLM never saw our labels or our training websites, which is fair in one way: it cannot memorise our sites. But the labels describe websites, not texts. A mainstream site can print a conspiracy story. The LLM may also know these websites from its own training and recognise their style, a kind of prior knowledge the logistic regressions lack. And the cut-off at 3,000 words hides the end of long articles. So its score measures agreement with our website labels, not understanding.

**Cost and runtime.** The LLM costs money for every article and needs an internet connection. The logistic regressions run free on a laptop in about a minute. Section 01 lists the figures.

**Our expectation (H3).** We expected the LLM to do about as well as the trained models, or to differ from them. The validation numbers say it differs, but the final comparison decides.

## The code

The three prompts. All ask for the same short answer format.

<!-- cell: src/fml/llm.py#prompt_variants -->

Reading the model's answer. Anything that does not fit the format becomes "unreadable".

<!-- cell: src/fml/llm.py#parse_response -->

Classifying articles: saved answers first, the model only for answers that are missing.

<!-- cell: src/fml/llm.py#classify -->

Scoring one prompt against the dataset labels on the validation articles.

<!-- cell: src/fml/pipeline.py#llm_scores -->

The analysis: try the three prompts, keep the best by F1, repeat it once.

<!-- cell: src/fml/pipeline.py#llm_stage -->

<!-- output: results/tables/llm_prompt_variants.csv -->
<!-- output: results/tables/methods_overview.csv -->
<!-- output: results/figures/llm_validation_confusion.png | LLM: validation confusion matrix of the chosen prompt -->
<!-- output: results/metrics.json#llm.chosen_prompt -->
<!-- output: results/metrics.json#llm.validation -->
<!-- output: results/metrics.json#llm.unreadable -->
<!-- output: results/metrics.json#llm.repeat_agreement -->

## Report

The fourth analysis uses a language model instead of a trained classifier. Claude Haiku 5.5 received each article without any examples (zero-shot) and answered the source question: would the article appear on a conspiracy-oriented or on a mainstream website? This is the same target as in the three logistic regression analyses, and the answers are scored against the same dataset labels.

Articles were cut to their first 3,000 words, as for the other methods, and the model could write up to 1,000 tokens. The model has no temperature setting, so default sampling was used. Three prompts were tried on the validation articles only: the bare question (v1), the question with short definitions of both website types (v2), and a step-by-step version that warns against judging by topic alone (v3). The prompt with the highest F1 score for the conspiracy class was chosen. Every answer was saved with the article id and a prompt hash, so repeated runs need no new requests.

<!-- output: results/tables/llm_prompt_variants.csv -->

All three prompts reached almost the same F1 score on the 376 validation articles, and none called a mainstream article conspiracy-oriented. The bare question and the step-by-step prompt tied at 0.987, and the short definitions reached 0.983. With a tie, the first prompt, v1, was kept. The choice of prompt mattered little. The chosen prompt reached a precision of 1.000 and a recall of 0.974: it missed 3 of the 117 conspiracy articles. Answers that could not be read would have counted as wrong; there were none (0), so the output limit was large enough. Because sampling cannot be fixed, the validation run of the chosen prompt was repeated once; the share of articles with the same answer both times was 0.992. In total, 30 validation articles were longer than 3,000 words and were cut.

<!-- output: results/figures/llm_validation_confusion.png | LLM: validation confusion matrix of the chosen prompt -->

Hypothesis H3 expected the language model to do about as well as the best trained model, or to differ from it. On validation it differs: its F1 of 0.987 lies well above the best logistic regression on the same articles (0.918 for TF-IDF). This is not yet a verdict, because the validation websites are not the test websites and the final comparison decides.

Scoring the LLM against website labels is not a fully fair comparison with the trained models, for four reasons. First, the LLM never saw our labels or our training websites, so it cannot overfit to them; this favours it as an independent reference but also means it was never taught what our labels mean. Second, the labels describe websites, not texts. A conspiracy website prints plain news and a mainstream website sometimes prints conspiracy-friendly pieces, so a "wrong" answer may be a correct reading of the text. Third, the model may know these websites or this 9/11 material from its own training and recognise a site from its style, headings or bylines; a correct answer can then be memory rather than reading, and the logistic regressions have no such prior knowledge. Fourth, all methods see only the first 3,000 words, so the end of long articles is hidden from all of them. The score therefore shows how well the model reproduces our website labels, not how well it understands conspiracy writing.

Cost and runtime also differ. The logistic regressions run on a laptop-class machine in about a minute each and cost nothing, whereas the LLM needs an internet connection and a paid interface; the runtime per article and the price per 1,000 articles are in the methods table in section 1.

## Check yourself

Q: Why is a zero-shot model a fair test of what a cheap, ready-made model can do?
A: It gets no solved examples and no training from us, only an instruction and the article, so its score shows what it can do alone.

Q: Why do we ask about the website and not about whether the article believes a conspiracy theory?
A: The dataset labels describe the website. To compare the LLM and the logistic regressions fairly, both must be judged against those same labels.

Q: Why do we repeat the validation run once?
A: The model has no temperature setting, so its answers can vary. The share of equal answers shows how much.

Q: The LLM was not trained on our articles. Does that make its score trustworthy?
A: Not fully. It may have seen the websites in its own training, so its score may partly reflect memory.

Q: Why is scoring the LLM against website labels not a fully fair comparison?
A: The labels describe websites, not texts, and the LLM may know those websites already. Its score shows agreement with the labels, not understanding of the articles.
