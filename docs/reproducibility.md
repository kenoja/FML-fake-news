# Reproducibility evidence

`fml run-all` rebuilds `results/` from the committed LLM and stance caches (no API calls). The three
logistic regressions (`final2_word_counts`, `final2_tfidf`, `final2_tfidf_features`) and the LLM (`final_llm`)
were each scored on test once with `fml evaluate-final <model_id>`; `results/test_access_log.jsonl` lists every
scoring with its configuration hash.

## Cycle 4 re-run (27 Oct 2026): not yet recorded

The two-run hash comparison for cycle 4 could not be run in the session that wrote this note (running
`fml run-all` was not permitted there), so no new hashes or timing are recorded. The check below is still to be
done; the checksums further down are the pre-cycle ones and will differ only in the new `site_concentration`
table, its `cleaning` keys in `metrics.json` and the renamed headers of `llm_prompt_variants.csv`.

## Two identical runs (26 Oct 2026, before cycle 4)

In the pinned environment of `requirements-lock.txt` (Python 3.11.17, numpy 2.4.6,
scikit-learn 1.9.1), `fml run-all` was run twice on 2 CPUs (about
106 seconds each). All 52 result files except `run_info.json` (the run's own
timing) were byte-identical between the two runs, and `python -m pytest -q` passed with no test skipped.
The agent harness now repeats this check after every pipeline run and fails a gate if any file differs.

On a different Python or numpy version one word weight in `word_counts_top_conspiracy_words.csv` can round
differently in the third decimal (0.098 or 0.099); it is not quoted anywhere in the text.

Check it yourself: run `fml run-all` twice and compare `sha256sum results/metrics.json results/tables/*`.

## Checksums of the committed results

```
2aaf5ce5c7596c6c4cd935fa4587aabddabf536b7aee6325e31aa305a87be171  results/frozen_models.json
95ecf467630ef6daab63162932bc6739ec6d18177c375705bb5e345f01517d95  results/metrics.json
11fb3374205d25f2b000e5ee0d5cc1a583f0b1f79abd74639f83ef97ef99026b  results/tables/ablation.csv
5e826c9ad1fe2fd2a2df8c9ba6c9008968bf2c70b48e1aaa20ba10787a154ae7  results/tables/cleaning_log.csv
a5f3722c9718774946779f692a1c410d59d7b2bc75d0d383f7a1ccdd7b1f217a  results/tables/comparison_compact.csv
0cf0b2030bdfd22c44b924a54e0ce60a7554a15f966e85a018d01f60f68dbdb1  results/tables/comparison_test.csv
ae8430c767f4fcc70bdf0739466f2f7bef54278fb129b33c4d9a3af95869d772  results/tables/epistemic_coverage.csv
61363bf16483c6cd321125650154da4f7da69ab17fdf91b4f228aa785684cf15  results/tables/error_classification.csv
113bf3dab9c8ae92ace70059a6da747301d67c11087ed17886a2a854824ee6c7  results/tables/error_counts_plain.csv
94568ae832a406d2225a26a42b9803dc98286eec45f6f7677673472b5c21fe63  results/tables/error_examples.csv
da1e6a2ecec40daaef2d5e57f43ffa85efe78c1c3d818e25e563311235fbd137  results/tables/error_examples_plain.csv
b62e90654323db4f99328a43e4a294d517712a5c8c42626c9dfbaa85497e6172  results/tables/feature_examples.csv
a8a3202e73f0c20cc087b6e56613f6904e1a44fb978e3f23f163756d3d7cc208  results/tables/feature_summary.csv
007961024d5370995af3700de1ef3d419102a0bd92b9896775915e44b46d9e89  results/tables/large_site_missed_stances.csv
c167da1729952ff530f6b29393aed4d69a75bb94a334749690511bb6fb080c38  results/tables/length_quantiles.csv
425a56cebe245ddfb8d0aab02f5f0f12b2bba84fc909f94ab43db892d9ff49e0  results/tables/llm_prompt_variants.csv
3f531803fa0cd8184caadb8974d5929f1ba179f9bb9c8b268ae5461cde6eb0b6  results/tables/mcnemar_tests.csv
f4d847e2ae414839f9a8fecb81b5fd6c7b8aae7c3464f51a95d87ea4325b0b12  results/tables/methods_overview.csv
4ae8f0c16e3f9d7598a4604f6fce47077f865654a11a9578f3cf9915ad1a8b71  results/tables/predictions_final2_tfidf.csv
b5b78ba5dcf2829cc817486bee39b76b67c06a330b25d8c18a169861fd0c6d8a  results/tables/predictions_final2_tfidf_features.csv
24f115335e8f36d64654a868f97254ad8d8fbca9c8ef4427b2819e5a232cad27  results/tables/predictions_final2_word_counts.csv
6f839d6ca3f28372c0597ba4359679475af4ed0583cbc1dd7a1eb9811d28ac71  results/tables/predictions_final_llm.csv
500139ab75cd6de7d565da299df7ccc7decf1b9a4a0f84e646dc5599001d2109  results/tables/predictions_llm.csv
a0ceea0e311c49c951e85bd28bcc1cbfa0b12a9c79c75836782b2004a90f0acc  results/tables/predictions_lr.csv
cca8da6623f44c28928029f24efeb36e85a3e30ea6144f69bcb1b7872df5deaa  results/tables/split_summary.csv
ef8d21318fd203e04d2a8726962d2f5c8162b7c76147fcc4bb4157b7a8c3b4f2  results/tables/stance_check_layout.csv
829ea7877946cdf6af352a477d6318eabd5f9e5894b2eb9f853d752b06d8dac9  results/tables/stance_check_sample.csv
8a64939f66801b72bc7204fef199c12b4f35fb6d5c8fef98db17754fcae65e2d  results/tables/stance_claude_answers.csv
9a97acb1d2b76d3bc361476180d8ae2ca8f2ba4839afbef5deabca326a4a9f66  results/tables/test_lr.csv
564dcadd40513b6e562418b3485291fb2cefd74121e83a44b8d72e7f9b15fe61  results/tables/test_website_recall.csv
33f4dfa71012304dd4b3c9646c25b561f0dcd900e6ae36eb382f878de7e678bf  results/tables/tfidf_grid.csv
f71ce780f41a03424077f30ea3db1077dace694edf69454f63f5f60a08439e53  results/tables/tfidf_top_conspiracy_words.csv
08f80e1da47877479f299f5463c4eaf3d54c5f3e659f9e56336cbc7b6ba3b653  results/tables/tfidf_top_mainstream_words.csv
a63a0a332c529b78f88f47df0741ec6f75615bc1ef2b51c699f0639e8ab9e809  results/tables/validation_vs_test.csv
8b717fb166e438fa99f78b01d0296ae959df0300ada64a7df87ccf95f5deaba2  results/tables/validation_website_recall.csv
f02d2f31fdc83ab08628f50daa11487bbaa64c20a86ebdd6c43437b6afdbfe05  results/tables/word_counts_grid.csv
90c217953be5d8bde71ccc4e0b3d42102118e89343c84de6c9675134baec5746  results/tables/word_counts_top_conspiracy_words.csv
004faaa13ec9f2a91b1881308b2c590a5d7aba80b942f2e476e74e9871dd5584  results/tables/word_counts_top_mainstream_words.csv
```
