# Reproducibility evidence

`fml run-all` rebuilds `results/` from the committed LLM and stance caches (no API calls). The three
logistic regressions and the LLM were scored on test once each under the ids `final2_word_counts`,
`final2_tfidf`, `final2_tfidf_features` and `final_llm` with `fml evaluate-final <model_id>`;
`results/test_access_log.jsonl` lists each id once with its configuration hash. The older `final_*`
logistic-regression ids stay in the log but are no longer built or reported.

Runtime of the last `fml run-all` (from `results/run_info.json`): 96.5 s on 4 CPUs
(limit 480 s). Stage seconds: word counts 25.6, TF-IDF 1.7,
TF-IDF plus features 57.3, LLM 0.5.

Check it yourself: run `fml run-all` twice, run `sha256sum results/metrics.json results/tables/*` after each run
and compare. Only `results/run_info.json` (timing) may differ.

## Status of the two-run comparison

**Not yet recorded for the cycle-3 results.** The task that wrote this file did not run `fml run-all` (the
harness runs it afterwards, and a task must not run it). The list below is the SHA-256 of the committed
`results/metrics.json` and `results/tables/*` at the time of writing. The final-gates task must run
`fml run-all` twice, paste both lists here and state whether they are identical; until it does, this file
claims no identical runs for cycle 3. (The cycle-2 evidence of two identical runs is in the git history.)

## Committed files

```
b0c15941bfceed2a9da14b070e582de5d029489a3d3ed92551969b2d54881e7d  results/metrics.json
304a4e43680922bb009dd569eddf5abe669ac1200fb45f8bdb1a9978d5fd6868  results/tables/ablation.csv
4c7a86ef2c277caf935674c6660754dd455c42dab08d7d3d2fe2011a028ce9b8  results/tables/cleaning_log.csv
6c72859c81318e72c1ffbeafd452c4e84e4cab23d135eb336774fcfee1575b56  results/tables/comparison_compact.csv
0cf0b2030bdfd22c44b924a54e0ce60a7554a15f966e85a018d01f60f68dbdb1  results/tables/comparison_test.csv
ae8430c767f4fcc70bdf0739466f2f7bef54278fb129b33c4d9a3af95869d772  results/tables/epistemic_coverage.csv
61363bf16483c6cd321125650154da4f7da69ab17fdf91b4f228aa785684cf15  results/tables/error_classification.csv
61783f13b71bdfc141f859ce19279f39ff645a213657574997c3f4be8baf8465  results/tables/error_examples.csv
7b8e283c4d2865cd6cd3018f0cf77e6d86f7b2f13fdc871f7b5f132272300b1d  results/tables/error_examples_plain.csv
b62e90654323db4f99328a43e4a294d517712a5c8c42626c9dfbaa85497e6172  results/tables/feature_examples.csv
f6ed47d341135e9b2560b9f469857a1d754723cd52871114e5d9d7090f5487e8  results/tables/feature_summary.csv
c167da1729952ff530f6b29393aed4d69a75bb94a334749690511bb6fb080c38  results/tables/length_quantiles.csv
425a56cebe245ddfb8d0aab02f5f0f12b2bba84fc909f94ab43db892d9ff49e0  results/tables/llm_prompt_variants.csv
2c0aef8063142d31525092de727e915245561ba9edb8b37a9d7c0462e2073288  results/tables/methods_overview.csv
4ae8f0c16e3f9d7598a4604f6fce47077f865654a11a9578f3cf9915ad1a8b71  results/tables/predictions_final2_tfidf.csv
b5b78ba5dcf2829cc817486bee39b76b67c06a330b25d8c18a169861fd0c6d8a  results/tables/predictions_final2_tfidf_features.csv
24f115335e8f36d64654a868f97254ad8d8fbca9c8ef4427b2819e5a232cad27  results/tables/predictions_final2_word_counts.csv
6f839d6ca3f28372c0597ba4359679475af4ed0583cbc1dd7a1eb9811d28ac71  results/tables/predictions_final_llm.csv
500139ab75cd6de7d565da299df7ccc7decf1b9a4a0f84e646dc5599001d2109  results/tables/predictions_llm.csv
a0ceea0e311c49c951e85bd28bcc1cbfa0b12a9c79c75836782b2004a90f0acc  results/tables/predictions_lr.csv
01a3ced785ff95cdd63983ee565043071701bf5318871e155a4722ada8d032f8  results/tables/split_summary.csv
63358373351362b66d5be6ad89cb08386f1ba9ece0e25911b0b881d28aa95580  results/tables/stance_check_sample.csv
8a64939f66801b72bc7204fef199c12b4f35fb6d5c8fef98db17754fcae65e2d  results/tables/stance_claude_answers.csv
9a97acb1d2b76d3bc361476180d8ae2ca8f2ba4839afbef5deabca326a4a9f66  results/tables/test_lr.csv
40039728477b8981e11ed43fec05f5c1cd10132ae6e0dde04e36dfd3a9446c52  results/tables/test_website_recall.csv
33f4dfa71012304dd4b3c9646c25b561f0dcd900e6ae36eb382f878de7e678bf  results/tables/tfidf_grid.csv
f71ce780f41a03424077f30ea3db1077dace694edf69454f63f5f60a08439e53  results/tables/tfidf_top_conspiracy_words.csv
08f80e1da47877479f299f5463c4eaf3d54c5f3e659f9e56336cbc7b6ba3b653  results/tables/tfidf_top_mainstream_words.csv
f02d2f31fdc83ab08628f50daa11487bbaa64c20a86ebdd6c43437b6afdbfe05  results/tables/word_counts_grid.csv
c0df3ab5b9a61c14fc676f5e9904ca867cadbec7f87e44b8385a1b870deef091  results/tables/word_counts_top_conspiracy_words.csv
004faaa13ec9f2a91b1881308b2c590a5d7aba80b942f2e476e74e9871dd5584  results/tables/word_counts_top_mainstream_words.csv
```
