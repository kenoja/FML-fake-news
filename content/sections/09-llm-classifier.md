---
title: The LLM classifier
part: Model building
hypothesis: A cheap zero-shot LLM recognises the tone of conspiracy-oriented websites without any training, but it reacts to the 9/11 topic and to confident wording, so it will not clearly beat the logistic regression on websites it has never seen.
---
## Goal and hypothesis

We compare our own logistic regression with one cheap language model, **Claude Haiku 5.5**, used **zero-shot**: it gets no training examples, only an instruction and the article. It answers the **source question**, the same target the logistic regression is scored on: would this article come from a conspiracy-oriented website or from a mainstream website? (Whether the article itself *endorses* a conspiracy theory is a different question, examined in the error analysis.)

Hypothesis: see the header. All numbers below come from `results/metrics.json` (key `llm`) and `results/tables/`; they were produced by one real run with an API key. A rerun of `fml run-all` reads the committed cache and needs neither key nor network.

## Explanation

**Zero-shot.** The model is not trained by us. We write an instruction (a **prompt**) and send it together with the article. Because small wording changes can change answers, we try **at most three prompt variants**, on the **validation** split only, and report all of them:

- **v1**, minimal: just the source question.
- **v2**, with a short definition of a "conspiracy-oriented website" and a "mainstream news website".
- **v3**, with step-by-step guidance and the instruction not to decide by topic alone (both site types write about 9/11).

All variants require strict JSON: `{"label": "conspiracy|mainstream", "confidence": 0-1, "reason": "<=25 words"}`. The **confidence** is the model's own statement, not a calibrated probability.

**Same text as the logistic regression.** The model receives exactly the cleaned text (URLs, website names and repeated boilerplate removed) that the logistic regression receives, taken from the same function. Both are scored against the dataset labels. The LLM's answer is never treated as ground truth.

**One word limit for everybody.** Every article is cut to its first `llm_classifier.max_input_words` words (3,000, set in `config/project.yaml`). The limit is the same for all documents; the number of documents that were cut is counted and reported (`truncated` in the result).

**Temperature 0** would make the model pick its most likely answer each time. `claude-haiku-5-5` rejects the temperature setting (HTTP 400), so we could not send it; the model runs with its default sampling (see "Needs a team decision" in `docs/decisions.md`). That makes the **second pass** over the test documents more important: it measures how often the answer changes when the same prompt is sent again. Pass 1 and pass 2 are cached separately.

**Cache.** Every answer is stored in `results/llm_cache.jsonl`, one line per call: `doc_id`, `prompt_hash`, label, confidence, reason, pass, token counts and time. The `prompt_hash` covers the prompt text, model, temperature, word limit and pass number, so changing any of them cannot reuse an old answer. The cache never contains article text. `fml run-all` therefore reruns without any API call.

**Parse errors.** If the reply is not valid JSON with an allowed label, it is recorded as `parse_error`, counted and reported. It is never silently turned into a label.

**Key and cost.** The API key comes only from the environment variable `LLM_CLASSIFIER_API_KEY`. If a document is missing from the cache and there is no key, the code raises an error instead of guessing. Token totals are written to `results/tables/llm_usage.csv`.

The three prompts (the shared answer format is in `_ANSWER_FORMAT`):

<!-- cell: src/fml/llm.py:30-60 -->

How an article is classified, cached and parsed:

<!-- cell: src/fml/llm.py#classify -->

<!-- cell: src/fml/llm.py#parse_response -->

### Step 1: choose the prompt on validation only

The rule was fixed before any test document was sent: **take the prompt with the highest conspiracy-F1 on the validation documents; if tied, take the one with fewer input tokens.** F1 is the balance of precision (of the articles called "conspiracy", how many really are) and recall (of the real conspiracy articles, how many were found). The rule is stored in `llm.selection_rule`.

<!-- cell: src/fml/pipeline.py#llm_validation -->

<!-- output: results/tables/llm_prompt_variants.csv -->

The chosen prompt is stored in `llm.chosen`:

<!-- output: results/metrics.json#llm.chosen -->

Per-variant details (confusion counts, intervals by document and by website, parse errors, truncated documents, tokens, cost):

<!-- output: results/metrics.json#llm.variants -->

Reading the table: all three prompts almost never call a mainstream article "conspiracy" (precision is 1.0 on validation); they differ in how many conspiracy articles they find (recall). The intervals by website are much wider than those by document, because validation contains only a few websites. Prompt v3, which tells the model not to decide by topic, did *worse* than v2; the "think step by step" wording also produced the most unparseable answers (see below).

### Step 2: one test scoring

`fml evaluate-final llm_haiku` classifies the filtered test documents once with the chosen prompt (pass 1). That is the **official LLM result**, scored against the dataset labels, never against another model's answer. It is logged once in `results/test_access_log.jsonl`.

<!-- cell: src/fml/pipeline.py#llm_stage -->

<!-- output: results/metrics.json#llm.test -->

Per-document predictions (`doc_id`, label, prediction, confidence, pass; no text) are in `results/tables/predictions_llm.csv`.

### Stability: does the same prompt give the same answer twice?

A second pass (same prompt, same documents) is reported separately as a sensitivity check. **Agreement** is the share of documents with the same label in both passes. **Cohen's kappa** is agreement corrected for the agreement expected by chance (1 = identical, 0 = no better than chance). A **flipped** document got a different label in the two passes.

<!-- output: results/metrics.json#llm.stability -->

<!-- output: results/metrics.json#llm.test_pass2 -->

### Parse errors, tokens and cost

A reply that is not the requested JSON is a **parse error**. Almost all of them are replies cut off at the output limit of 150 tokens (the model started explaining before it answered). A parse error is never turned into a label by hand; for scoring it counts as "not conspiracy", so it lowers recall, and the count is reported next to every score. The test recall is therefore partly a formatting problem, not only a judgement problem.

<!-- output: results/tables/llm_usage.csv -->

Cost is computed from the token counts and the list prices in `config/project.yaml` (`cost_usd` in the tables above).

## What could go wrong

- **Prior knowledge.** The model has seen 9/11 conspiracy material in training. That is part of what we compare, but it means the result says little about other topics.
- **Topic instead of style.** The model may answer "conspiracy" whenever an article discusses conspiracy theories, even a mainstream debunking. Prompt v3 asks it not to; we test whether that helps.
- **Prompt tuning on validation.** With three variants and few validation websites, the best prompt may be partly luck. We report all three.
- **Unstable answers.** Temperature could not be set to 0, so the same input can give different answers; the second pass quantifies this.
- **The model may know the sites.** Haiku may have seen these websites or these exact articles in training, so a correct answer can be memory rather than reading. We cannot test this; a good score on 9/11 does not show it would work on unseen topics.
- **The source question is hard from text alone.** The model sees one article, not the website, its design or its other pages. Mainstream sites sometimes publish conspiracy-friendly pieces and conspiracy sites publish plain news; those articles are close to unanswerable from the text.
- **Parse errors lower recall.** Replies cut off at 150 output tokens count as "not conspiracy" (see above). A larger limit might raise recall, but the cache key does not include the limit and changing it now would mean a new run after seeing test results, so we did not.
- **Test-set size.** The test split has few websites; the website-level intervals are wide, so small differences to the logistic regression must not be over-read.
- **Truncation.** Cutting at the word limit removes the end of long articles for the LLM but not for the logistic regression's counts if limits differ; we report how many documents are cut.
- **Cost and privacy.** Article text is sent to an external API; it is never written to this repository or to the cache.

## Technical infrastructure

- Code: `src/fml/llm.py` (`PROMPTS`, `classify`, `parse_response`, `make_client`, `FakeClient`, `usage_table`).
- Settings: `config/project.yaml` under `llm_classifier` (model, temperature, key variable name, number of variants, word limit, cache file).
- Cache: `results/llm_cache.jsonl`; usage totals: `results/tables/llm_usage.csv`.
- Tests: `tests/test_llm.py` run offline with `FakeClient` (cache hit means zero calls, missing key plus cache miss raises, the cache never holds article text, parse errors are counted, the two passes hash differently, at most three variants).
