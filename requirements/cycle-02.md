# Requirements for cycle 2

Cycle 2 reshapes the cycle-1 work for its real audience before the team starts giving feedback.
Sources: the repo owner's review of cycle 1 (quoted below) and the decisions taken with the owner on
17 Oct 2026. Requirements are cumulative: `requirements/cycle-01.md` still applies **where this file
does not change it; where they differ, this file wins** (record each such conflict once in
`docs/decisions.md`). The new section format, slugs, statistics and code style are in `AGENTS.md`.

## Owner's feedback on cycle 1 (verbatim)

> - scope: the idea is that my teammates without any knowledge can go through the document, learn the underlying concepts and then assess how well they understand it and what they have to criticize about the text
> - audience: noone of my teammates has any technical background extend the glossary and use simpler wording, focussed on actually explaining concepts rather than technical complexity. use cross links for the respective words, so they can easily jump to the explanations.
> - do not write about agent fails during the article runs in the text and limitations, that document should contain the full assignment. if there are mistakes or uncertainties they should be resolved before
> - rephrase the hypotheses to match the avademic level and scope for this, in terms of understanding but keep it content wise on high level. the phrasing is really unclear for someone without machine learning background. also remove the agent plan stuff and so on, that is only relevant for me not for my colleagues. fix the counting.
> - LLM API key: the API key is in the claude cloud session as explained. if it did not work
> - the analysis and tests are way to technical and detailed. reminder: this is a 2.5 ECTS credit fundamentals of machine learning course for non-technical background.
> - remove off topic during cleaning and explain why we do this
> - simplify the analysis for the three different tests: words, TF-IDF and LLM
> - I really like the section on feature engineering. please try to simplify the coding style so it is easier to read for beginners.
> - do not add big commenting texts in the code but rather small in line comments, easy to understand
> - do all runs first and then analyse all three and compare after
> - discuss changes with me, lets come up with the final structure and make it easy to use for my teammates.
> - technical infrastructure is not needed in the current style. waht was meant with it was to explain the code and decisions.
> - reduce the size of it the whole thing should be max 5000 words.
> - make sure that the "report text" of each section is separated from the explanations of the section. so the low-technical knowledge explanation of the section with design choices which were amde are standing alone independently from the analysis or respective part of the final report.

## Decisions taken with the owner (17 Oct 2026)

- **Four analyses**, all logistic regression except the last: (1) word counts, (2) TF-IDF,
  (3) TF-IDF plus the engineered features, (4) the LLM.
- **5,000 words** is the cap for the **report text** (all `## Report` parts together, 3,000–5,000).
  Explanations, code, questions and glossary come on top but must stay short (see AGENTS.md).
- **Test set**: one clean final run scores every final model once; `10-limitations` contains one
  honest sentence that the test set was used during development.
- **Human check of 40 articles**: prepared now, done by the team in cycle 2's feedback period.
- **The LLM key works** (`LLM_CLASSIFIER_API_KEY` is set in the cloud environment; cycle 1 used it).
- **Order inside each section**: Explained simply → The code → Report → Check yourself.

## Requirements

1. **New structure.** Rewrite `content/sections/` into the slugs and the four-part format of
   `AGENTS.md` ("Section format"). Delete every section file whose slug is not in that list. Reuse
   correct cycle-1 content, but rewrite it for the audience. `00-how-to-use` explains how to read the
   page (the four parts, glossary links, hidden answers) and how to give feedback; it is not report text.
2. **Audience.** Write for teammates with no technical background (2.5-ECTS fundamentals course):
   short sentences, everyday examples, concepts before details. Remove detail that a fundamentals
   course does not need. "Explained simply" explains the step and every design choice with its reason
   and stands alone without the report text.
3. **Question and hypotheses** (`01-question-and-hypotheses`). One research question and 3–5
   hypotheses, numbered H1, H2, ... without gaps, phrased at academic level but understandable without
   machine-learning background, high-level in content (e.g. "Adding features about certainty and writing
   style improves the TF-IDF model"). Each hypothesis is answered in `08-results` and `11-conclusion`.
   Remove all agent-plan material (task ids, "T5", "round", "provisional") from sections and report.
4. **Cleaning** (`02-data-and-cleaning`). Off-topic removal and near-duplicate removal happen during
   cleaning, before any model. When a near-duplicate pair crosses splits, drop the copy in validation or
   test; inside one split keep the lower `doc_id`. Show before/after counts and explain why we remove
   each kind. Drop the first run on all 3,146 documents from code, `results/` and text. Keep the answer to
   team question 5 (very short texts) here. The split file stays unchanged (removing documents from it
   is not re-splitting).
5. **The four analyses, one section each** (`04-word-counts`, `05-tf-idf`, `06-feature-engineering`,
   `07-llm`). Each says what the analysis is, how it was run and which settings were chosen on
   validation, with no test-set numbers. Logistic regression settings are the same small grid for all
   three (e.g. C in {0.1, 1, 10}, chosen by validation F1; class weight "balanced", explained in one
   sentence). Team question 2 (how logistic regression learns from words) is answered in `04-word-counts`.
6. **Feature engineering** (`06-feature-engineering`). The owner likes this section: keep its content
   and literature reasons. Show each feature group (n-grams 1–3, certainty/uncertainty words, style)
   with one simple validation comparison (TF-IDF alone vs plus each group vs plus all), and simplify
   its code for beginners.
7. **LLM** (`07-llm`). Default sampling settings (state plainly that the model has no temperature
   setting to fix). Raise the output limit so no answer is cut off: the target is 0 unreadable answers;
   any that remain are counted as wrong and reported as one number. At most 3 prompt variants on
   validation; the best one is used. Answer team question 3 here: both approaches are compared with the
   dataset labels, and the LLM's stability is shown by repeating the validation run once (share of equal
   answers, nothing more). Cache everything so `fml run-all` makes no API calls.
8. **One final run.** After all settings are fixed, score every final model on test exactly once with
   `fml evaluate-final` under new model ids (e.g. `final_word_counts`, `final_tfidf`,
   `final_tfidf_features`, `final_llm`). Earlier log entries stay in `results/test_access_log.jsonl`.
9. **Results** (`08-results`). One table with all four models (precision, recall, F1 with 95%
   confidence intervals), their confusion matrices, and the three McNemar tests from AGENTS.md. Explain
   precision, recall and F1 with our own numbers (team question 4). Answer each hypothesis.
10. **Where they fail** (`09-where-they-fail`). Real disagreement examples (excerpts within limits),
    and the split between model errors and label problems. Describe the automatic stance coding in plain
    words. Prepare `results/tables/stance_check_sample.csv` (40 articles) with an empty column for the
    team and a short instruction in "Check yourself" on how a teammate fills it in. The report text says
    that this human check is done by the team; it does not present the automatic coding as verified.
11. **Limitations** (`10-limitations`) are limitations of the study only (labels describe websites,
    few websites, LLM prior knowledge, lexicon limits, team question 6) plus the one test-set sentence.
    Nothing about agent runs, failed steps, retries or tooling problems.
12. **AI use** (`12-how-we-used-ai`): factual and short: what the AI produced, what was checked or
    changed, what we learned. No diary of failures. `docs/ai-log.md` gets this cycle's entry.
13. **Simpler analysis.** Only the statistics listed in AGENTS.md. Remove website-level bootstrap,
    kappa, stability passes beyond requirement 7, sensitivity analyses, parse-error analysis and
    near-duplicate sensitivity from code, tests, `results/` and text.
14. **Beginner code.** Rewrite the code shown in cells in the style of AGENTS.md ("Code conventions"):
    short functions, plain names, small inline comments, no long docstrings. Tests stay and still encode
    the rules (fit on train only, frozen split, locked test set, privacy); the suite runs in under 3
    minutes.
15. **Glossary** (`99-glossary`). Every technical term used anywhere on the site has an entry in the
    format of AGENTS.md, with an everyday example. At least: classifier, label, feature, word count,
    bag of words, TF-IDF, n-gram, certainty and uncertainty words (epistemic markers), style feature,
    logistic regression, weight, probability, training/validation/test set, grouped split, overfitting,
    regularisation, class weight, LLM, zero-shot, prompt, token, precision, recall, F1 score, confusion
    matrix, confidence interval, bootstrap, McNemar test, p-value, statistical significance,
    near-duplicate, corpus.
16. **Size.** Report text 3,000–5,000 words (gate on the assembled report). "Explained simply" at most
    450 words per section (gate). Glossary entries at most 80 words.
17. **Licence.** The OSF page could not be read automatically; the owner checks it. Until then keep
    the safe default (texts only in the private data repo, short excerpts) and say in `02-data-and-cleaning`
    where the data comes from and that the texts are not redistributed. Record the open check in
    `docs/decisions.md`, not on the site.
18. **Reproducible.** `fml run-all` gives byte-identical `results/` tables when run twice (cycle 1 had
    three tables that changed between runs: the lists of strongest words and the words in removed
    titles; sort ties deterministically).

## Open team questions from cycle 1, new homes

1 → `03-split-by-website`, 2 → `04-word-counts`, 3 → `07-llm`, 4 → `08-results`,
5 → `02-data-and-cleaning`, 6 → `10-limitations`.
