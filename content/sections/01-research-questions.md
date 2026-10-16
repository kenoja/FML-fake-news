---
title: Research questions and hypotheses
part: Introduction
hypothesis: Both approaches beat always guessing "mainstream", words and short phrases carry most of the signal, and many "errors" are really labels that describe the website rather than the article.
---
## Goal and hypothesis

We compare a logistic regression we train ourselves with one cheap LLM (Claude Haiku 5.5, zero-shot, no fine-tuning). Both read **full** 9/11 articles from the LOCO corpus and decide: does this article come from a conspiracy-oriented website or a mainstream one?

**Research questions**

- **RQ1.** How do (a) a words-only logistic regression, (b) a logistic regression with extra features and (c) a zero-shot LLM compare on the *same* test set, where whole websites are held out?
- **RQ2.** Do hand-made features improve the logistic regression over words alone? Which feature groups help and which do not?
- **RQ3.** Where does each approach fail? For each disagreement with the dataset label: is the prediction implausible, or does the label not describe the text?

**Hypotheses** (written before any result exists)

- **H1.** Both approaches beat the majority-class baseline (always "mainstream") on conspiracy-class F1.
- **H2.** Adding 2- and 3-word phrases helps more than any single non-word feature group, because phrases like "no evidence of" or "inside job" carry stance that single words lose.
- **H3.** Epistemic markers (certainty, hedging, "hidden truth" vocabulary) add signal beyond n-grams.
- **H4.** Style features (ALL CAPS, exclamation marks, question marks, quotation density, sentence length, repetition) add signal, but less than the word-based features.
- **H6.** Many "false positives" (mainstream articles predicted as conspiracy) discuss or debunk conspiracy theories, and many "false negatives" (conspiracy-site articles predicted as mainstream) contain no conspiracy claim at all, which is label noise.

There is no H5: the emotion-score idea was dropped by the team, so the numbering keeps a gap on purpose.

## Explanation

**Majority-class baseline**: a "model" that always answers the most common label. About 70% of our articles are mainstream, so it scores about 70% accuracy while finding zero conspiracy articles. That is why we judge by **F1 on the conspiracy class** (the balance of precision, "how many flagged articles really are conspiracy", and recall, "how many conspiracy articles did we catch").

**Why the label is tricky**: the label belongs to the *website*. A mainstream article that debunks a theory may use conspiracy vocabulary; a conspiracy site may publish a plain news item. RQ3 and H6 look at exactly that.

**Open points that affect this question** (all in `docs/decisions.md`, under "Needs a team decision"):

- *Licence of the texts* is not confirmed (the OSF page could not be reached on 2026-10-16). Until it is, texts stay in the private repo and we show only short excerpts.
- *LLM API key*: it is not known whether `LLM_CLASSIFIER_API_KEY` is available in the cloud environment. If not, the LLM results will be reported as missing, never invented.
- *Near-duplicates*: removing them might change the expected 2,608 documents after the off-topic filter. We keep 2,608 until the team decides.

Two decided points: the final test command is `fml evaluate-final <model_id>` (not the older `fml evaluate --final`), and the split file covers all 3,146 documents, so filtering later never moves a document between train and test.

## What could go wrong

- Reading the hypotheses as results. They are expectations; later sections report whether they held, including when they did not.
- Tuning on the test set. Only `fml evaluate-final` may score on test, and every use is logged.
- Judging the LLM against a label it cannot know: it sees one article, not the website.
- A small number of conspiracy websites means a few sites can dominate a split, so results can swing. We report this wherever it applies.

## Technical infrastructure

Every number in this project is written through one helper, so prose can quote files in `results/` instead of typing values:

<!-- cell: src/fml/results.py#write_metrics -->

- `src/fml/results.py`: `write_metrics`, `write_table`, `savefig` (150 dpi; conspiracy `#8c3b76`, mainstream `#2b7a6a`).
- `src/fml/pipeline.py`: `run_all()` and `evaluate_final(model_id)`, behind `fml run-all` and `fml evaluate-final`. Both raise `NotImplementedError` until later steps fill them in.
- `results/metrics.json`: top-level keys `data`, `split`, `baseline_all_docs`, `topic_filter`, `baseline`, `ablation`, `llm`, `comparison`, `errors`.
- `docs/decisions.md`: conflicts and open questions.
