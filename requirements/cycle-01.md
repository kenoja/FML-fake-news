# Requirements for cycle 1

The first cycle builds the whole project from scratch. Sources: the team's project proposal, the
course feedback on it (quoted below; it overrules the proposal), and team decisions taken on
16 Oct 2026. Method details and literature: `docs/PROJECT_PLAN.md`.

## Course feedback (verbatim content, identifying details removed)

> This is an interesting and suitable project. I would suggest using the full articles rather than just headlines. A headline may mention a conspiracy without telling us whether the article supports it, rejects it, or simply discusses it.
>
> Stick with logistic regression versus an LLM. You do not need to fine-tune a transformer or train and compare several models. Pick one inexpensive API model, or a local model through Ollama, and give both approaches the same texts.
>
> For logistic regression, you could extend the current bag-of-words features with a few other things:
> - Bigrams and trigrams, combinations of two or three consecutive words, such as "inside job" or "no evidence of". These can also be represented using TF-IDF.
> - Relative frequencies of certainty or uncertainty expressions, such as "undeniable", "perhaps", or "allegedly". Count these per 1,000 words, for example, so longer articles do not automatically get higher values.
> - Style features, such as the proportion of words in ALL CAPS, question marks or exclamation marks per 1,000 words, average sentence length, or repetition.
> - Sentiment or emotion scores from another model, such as negativity, anger, or fear. You can use these scores as additional numerical inputs to logistic regression.
>
> You do not need to include all of these. Choose a few and compare the word-only model with an extended version. Do the extra features help? Negative or emotional language is not automatically conspiracy language, especially when reporting on something like 9/11. With 2000-3000 documents in typical news-length, a logistic regression model should be comfortably trainable locally on an average laptop - even without GPU.
>
> One issue - as far as I understood your proposal - to keep in mind is that your labels describe the source website, not necessarily the content of each article. A mainstream article discussing conspiracy theories could therefore receive a "conspiracy" prediction for understandable reasons. Look at actual disagreements and consider whether the prediction is implausible, or whether the dataset label does not really describe the text.
>
> Give your plan to a coding agent and ask it to implement a first version. [...] Then run it and work backwards to understand the features, predictions, and errors. Keep your website-based split consistent across comparisons, and calculate the vocabulary and TF-IDF weights using only the training data.
>
> As discussed in class, describe what actually happened: what AI produced, what you checked or changed, and what you learned. You do not need to focus on why the LLM wins or what you could do to get your model to the same accuracy. Finding out where the approaches work or fail, and how you can add engineer additional features for your logistic regression is already a useful result.

## Team decisions (16 Oct 2026)

- The LLM is Claude Haiku 5.5 (`claude-haiku-5-5`), the cheapest Anthropic model, asked the **source
  question**: would this article come from a conspiracy-oriented or a mainstream website?
- Feature groups for the extended model: **n-grams (1–3), epistemic markers, style**. Emotion scores are
  **not** used (record the reason in the "what we did not do" part).
- **Off-topic documents**: run the first words-only model on all 3,146 documents, then remove documents
  that never mention 9/11 (pattern in `config/project.yaml`), and explain *after* the first run why they
  were removed, with the before/after numbers. All later experiments use the filtered set.
- The JFK transfer test is **out of scope** for now.
- Everything is written in English. No personal data anywhere (see AGENTS.md).

## Open questions from the team (answer each one in the section named)

These came up when the team read the proposal. Answer them in plain language where they belong.

1. *(03-split-by-website)* What does "split grouped by website" mean in practice? Is the training data
   fed per website, or does it just need to be diverse? Explain with a small example.
2. *(04-tfidf-and-logistic-regression)* Is this right: the model gets labelled texts, assigns numerical
   importance to words, and from those statistics learns to recognise conspiracy-site articles?
   Confirm or correct, step by step.
3. *(10-head-to-head)* LLMs do not always give the same answer, so the LLM should not be treated as the
   baseline. Both approaches are compared against the dataset labels, right? Show how stable the LLM's
   answers are.
4. *(05-first-run-all-documents)* What is an F1 score? Explain precision, recall and F1 with our own
   numbers.
5. *(02-data-and-labels)* Could very short texts be impossible to classify? With full articles, show the
   length distribution and say whether a minimum length is needed.
6. *(12-limitations)* Could the model learn to recognise news websites even though the website name is
   not given to it? Explain what the grouped split and the removal of site names protect against.

## Requirements

1. Build the analysis in `src/fml/` following `AGENTS.md` (contracts, hard rules) and the method in
   `docs/PROJECT_PLAN.md`: load data, clean (drop duplicates and near-duplicates, strip website names,
   URLs and repeated boilerplate), create and freeze the website-grouped split, words-only TF-IDF +
   logistic regression on all documents, off-topic filter, words-only baseline on the filtered set,
   the three feature groups and the ablation, the LLM classifier (validation for prompt choice, one
   final test run, a second test pass for answer stability), comparison with bootstrap confidence
   intervals (documents and websites) and a McNemar test, and the error analysis.
2. Error analysis: build the disagreement sets (LR-only wrong, LLM-only wrong, both wrong); code the
   stance of a sample of disagreements (endorses / rejects or debunks / reports neutrally / no claim)
   with Claude and a short justification each; select a random sample of 40 for **human verification**
   and write it to `results/tables/stance_check_sample.csv` with an empty `human_stance` column for the
   team to fill in (the walkthrough explains how). Classify each disagreement as "model wrong" or
   "label does not describe the text".
3. Report the number of websites and documents per split and class; if a split has fewer than 3
   conspiracy websites, say so prominently and add per-website results.
4. Write every notebook section listed in `AGENTS.md` in the required format, answering the open
   questions above where named.
5. Write `reports/report.md` (3,000–5,000 words) following the assignment's structure: what we wanted to
   find out; data, models and resources; approach (and what we did not do, with reasons); evaluation;
   findings; interpretation and limitations; use of AI tools (what was produced, checked, changed,
   learned). Include figures and tables by path.
6. Keep `docs/ai-log.md` and `docs/decisions.md` up to date.
7. Re-check the LOCO licence on https://osf.io/snpcg/ if the site is reachable, and record the result in
   `docs/decisions.md` and the report's data section. Until a licence is confirmed, the rules on excerpts
   in `AGENTS.md` apply.
