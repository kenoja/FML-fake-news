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

**Answers we cannot read, and saved answers.** An answer that breaks the fixed format counts as wrong, and we report how many there were. Answers are saved by article id and prompt, never the text, so a rerun needs no new questions.

**Is the comparison fair?** Not entirely. The LLM never saw our labels or our training websites, which is fair in one way: it cannot memorise our sites. But the labels describe websites, not texts. A mainstream site can print a conspiracy story. The LLM may also know these websites from its own training and recognise their style, a kind of prior knowledge the logistic regressions lack. And the cut-off at 3,000 words hides the end of long articles. So its score measures agreement with our website labels, not understanding.

**Cost.** The LLM costs money per article and needs internet; the logistic regressions run free on a laptop. Section 01 lists the figures.

**Why no training and no examples.** Fine-tuning needs many labelled articles and would end the "cheap, ready-made" idea. Solved examples (few-shot) from our websites would teach it our labels. **Our expectation** is in section 01; the comparison is in the results section.

## The code

The three prompts. All ask for the same short answer format.

<!-- cell: src/fml/llm.py#prompt_variants -->

Reading the model's answer. Anything that does not fit the format becomes "unreadable".

<!-- cell: src/fml/llm.py#parse_response -->

Asking the model about one article and saving its answer.

<!-- cell: src/fml/llm.py#ask_model -->

Asking about all articles that have no saved answer yet, several at a time.

<!-- cell: src/fml/llm.py#fetch_missing -->

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

The fourth analysis uses a language model instead of a trained classifier. Claude Haiku 5.5 is the lowest-cost model of Anthropic's Claude 5.5 family, chosen because the course asked for one inexpensive model rather than a comparison of language models; it was called through Anthropic's API in October 2026, one request per article, and had to answer in a fixed short format (label, confidence, one-line reason) of at most 1,000 tokens. The cost, about $0.26 per 1,000 articles, follows from the counted tokens at the list price ($0.10 and $0.50 per million input and output tokens). It received each article without examples (zero-shot) and answered whether it would appear on a conspiracy-oriented or a mainstream website; the answers are scored against the same labels.

Articles were cut to their first 3,000 words, as for the other methods; the model has no temperature setting, so default sampling was used. Three prompts were tried on validation only: the bare question (v1), the question with short definitions of both website types (v2), and a step-by-step version that warns against judging by topic (v3).

<!-- output: results/tables/llm_prompt_variants.csv -->

All three prompts reached almost the same validation F1 (0.987, 0.983 and 0.987); with a tie, the first prompt, v1, was kept. The chosen prompt reached a precision of 1.000 and a recall of 0.974: it missed 3 of the 117 conspiracy articles. There were no unreadable answers. A repeated validation run gave the same answer for 0.992 of the articles.

<!-- output: results/figures/llm_validation_confusion.png | LLM: validation confusion matrix of the chosen prompt -->


Scoring the LLM against website labels is not a fully fair comparison with the trained models. It never saw our labels, so it cannot overfit to them, but it was never taught what they mean. The labels describe websites, not texts, so a "wrong" answer may be a correct reading. And the model may recognise a site from its style or bylines through its own training, which the logistic regressions cannot use.


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
