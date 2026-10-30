# Requirements for cycle 5 (final report)

Written with the repo owner on 2026-10-30 after the presentation. All earlier files still apply unless an item
below changes them; where they differ, this file wins. No new models, prompts, API calls or test-set runs.

## Owner's requirements

- **O1. N-gram threshold, explained.** Phrases (n-grams) keep the minimum of 3 training articles; single words keep 2.
  The report states both and why: phrases are far more numerous than words, and with a minimum of 2 the planned
  model fills the 200,000-column cap and scores lower on validation (`tfidf_features.ngram_threshold_check` in
  `results/metrics.json`, validation only, no test run). Decided with the owner: no new test run.
- **O2. Feature limits in the feature chapter.** Explain why the features could work (they separate the classes
  clearly in the training data) and why they did not: far more columns than training articles, overlap with words
  TF-IDF already has, style that may describe single websites, word lists that ignore context.
- **O3. One story, told plainly.** The report reads as one narrative: a fair test on new websites, a ladder of four
  methods, validation results that looked excellent, and a test that dropped sharply. The drop is attributed
  clearly to Website 4 as the main single cause, with the honest qualification from the per-website table and the
  F1 without Website 4 (for the LLM it explains almost all of the drop, for the logistic regressions the largest part).
- **O4. Splitting by website is standard practice.** Say that we kept the grouped split although it is demanding,
  because holding out whole sources is the accepted way to test generalisation, and that it exposed the limits of
  the data set (few conspiracy websites, one dominant test website). That is why test scores are far below
  validation scores.
- **O5. A weak score, a useful lesson.** Present the gap between validation and test as the study's main lesson
  (an honest evaluation shows what the data can and cannot support), not as a failure to hide.
- **O6. Same structure and wording in every chapter.** Method chapters share one pattern (idea and expectation,
  setup, validation result, what the model learned and what follows); other chapters follow one consistent pattern
  too. One name per concept throughout (word counts, TF-IDF, TF-IDF + features, LLM; validation, test; Website 4).
- **O7. Limitations stated critically,** including that McNemar's test, like the bootstrap intervals, treats
  articles as independent, and that the share of conspiracy articles in our data differs from the web.
- **O8. Length.** At most 4,900 words by the plain count (`tests/test_gates.py`), tables included.
