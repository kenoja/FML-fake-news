---
title: The LLM classifier
part: Model building
hypothesis: A cheap zero-shot LLM recognises the tone of conspiracy-oriented websites without any training, but it reacts to the 9/11 topic and to confident wording, so it will not clearly beat the logistic regression on websites it has never seen.
---
## Goal and hypothesis

We compare our own logistic regression with one cheap language model, **Claude Haiku 5.5**, used **zero-shot**: it gets no training examples, only an instruction and the article. It answers the **source question**, the same target the logistic regression is scored on: would this article come from a conspiracy-oriented website or from a mainstream website? (Whether the article itself *endorses* a conspiracy theory is a different question, examined in the error analysis.)

Hypothesis: see the header. Numbers (validation scores of the three prompts, the chosen prompt, the stability pass, token use) are filled in once the model has been run; this draft only describes the setup.

## Explanation

**Zero-shot.** The model is not trained by us. We write an instruction (a **prompt**) and send it together with the article. Because small wording changes can change answers, we try **at most three prompt variants**, on the **validation** split only, and report all of them:

- **v1**, minimal: just the source question.
- **v2**, with a short definition of a "conspiracy-oriented website" and a "mainstream news website".
- **v3**, with step-by-step guidance and the instruction not to decide by topic alone (both site types write about 9/11).

All variants require strict JSON: `{"label": "conspiracy|mainstream", "confidence": 0-1, "reason": "<=25 words"}`. The **confidence** is the model's own statement, not a calibrated probability.

**Same text as the logistic regression.** The model receives exactly the cleaned text (URLs, website names and repeated boilerplate removed) that the logistic regression receives, taken from the same function. Both are scored against the dataset labels. The LLM's answer is never treated as ground truth.

**One word limit for everybody.** Every article is cut to its first `llm_classifier.max_input_words` words (3,000, set in `config/project.yaml`). The limit is the same for all documents; the number of documents that were cut is counted and reported (`truncated` in the result).

**Temperature 0** makes the model pick its most likely answer each time. This reduces, but does not remove, differences between runs, so a **second pass** over the test documents measures how often the answer changes. Pass 1 and pass 2 are cached separately.

**Cache.** Every answer is stored in `results/llm_cache.jsonl`, one line per call: `doc_id`, `prompt_hash`, label, confidence, reason, pass, token counts and time. The `prompt_hash` covers the prompt text, model, temperature, word limit and pass number, so changing any of them cannot reuse an old answer. The cache never contains article text. `fml run-all` therefore reruns without any API call.

**Parse errors.** If the reply is not valid JSON with an allowed label, it is recorded as `parse_error`, counted and reported. It is never silently turned into a label.

**Key and cost.** The API key comes only from the environment variable `LLM_CLASSIFIER_API_KEY`. If a document is missing from the cache and there is no key, the code raises an error instead of guessing. Token totals are written to `results/tables/llm_usage.csv`.

<!-- cell: src/fml/llm.py#PROMPTS -->

<!-- cell: src/fml/llm.py#classify -->

<!-- cell: src/fml/llm.py#parse_response -->

Prompt variants, validation scores, parse-error count and truncated-document count are added after the run:

<!-- output: results/tables/llm_usage.csv -->

## What could go wrong

- **Prior knowledge.** The model has seen 9/11 conspiracy material in training. That is part of what we compare, but it means the result says little about other topics.
- **Topic instead of style.** The model may answer "conspiracy" whenever an article discusses conspiracy theories, even a mainstream debunking. Prompt v3 asks it not to; we test whether that helps.
- **Prompt tuning on validation.** With three variants and few validation websites, the best prompt may be partly luck. We report all three.
- **Unstable answers.** Even at temperature 0 the same input can give different answers; the second pass quantifies this.
- **Truncation.** Cutting at the word limit removes the end of long articles for the LLM but not for the logistic regression's counts if limits differ; we report how many documents are cut.
- **Cost and privacy.** Article text is sent to an external API; it is never written to this repository or to the cache.

## Technical infrastructure

- Code: `src/fml/llm.py` (`PROMPTS`, `classify`, `parse_response`, `make_client`, `FakeClient`, `usage_table`).
- Settings: `config/project.yaml` under `llm_classifier` (model, temperature, key variable name, number of variants, word limit, cache file).
- Cache: `results/llm_cache.jsonl`; usage totals: `results/tables/llm_usage.csv`.
- Tests: `tests/test_llm.py` run offline with `FakeClient` (cache hit means zero calls, missing key plus cache miss raises, the cache never holds article text, parse errors are counted, the two passes hash differently, at most three variants).
