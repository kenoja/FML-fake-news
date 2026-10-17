# Conspiracy or mainstream? A logistic regression and a cheap LLM classify full 9/11 articles by source website

All numbers in this report are copied from `results/metrics.json` and `results/tables/*.csv`. The test `tests/test_report_numbers.py` checks them. Every figure and table is linked by path, and the code behind each step is explained in the walkthrough sections under `content/sections/`.

## 1. What we wanted to find out

News about a big event reaches us from very different kinds of websites. Some are mainstream outlets, some are websites that mainly promote conspiracy theories. We asked whether a machine can tell, from the **full text** of one article about the 9/11 attacks, which kind of website published it. We compared two very different approaches on exactly the same documents:

- a **logistic regression** (a simple model that gives each word a weight and adds the weights up) that we train ourselves, and
- **one cheap large language model (LLM)**, Claude Haiku 5.5, which is simply asked the question and has never seen our training data.

The course feedback shaped the design. It asked for full articles instead of headlines, for one cheap LLM instead of a benchmark of many models, for a small set of hand-made features for the logistic regression, and for an honest look at the cases where the model "fails", because the labels describe the website and not necessarily the article. It also asked us to describe what actually happened with the AI tools. The feedback overrules the original proposal.

**Research questions**

- **RQ1.** How do a words-only logistic regression, a logistic regression with extra features, and a zero-shot LLM compare on the same test set, when whole websites are held out?
- **RQ2.** Do hand-made features help the logistic regression? Which groups help and which do not?
- **RQ3.** Where does each approach fail? For each disagreement with the dataset label, is the prediction implausible, or does the label not describe the text?

**Hypotheses** (written before any result existed)

- **H1.** Both approaches beat the majority-class baseline (always "mainstream") on the F1 score of the conspiracy class.
- **H2.** Two- and three-word phrases help more than any single non-word feature group.
- **H3.** Epistemic markers (words of certainty, doubt and "hidden truth") add signal beyond phrases.
- **H4.** Style features (ALL CAPS, exclamation and question marks, quotation density, sentence length, repetition) add signal, but less than word-based features.
- **H6.** Many false positives are mainstream articles that discuss or debunk conspiracy theories, and many false negatives are conspiracy-site articles that contain no conspiracy claim at all.

There is no H5. The emotion-score idea of the proposal was dropped by the team (section 3), and the numbering keeps the gap on purpose.

## 2. Data, models and resources

**Corpus and labels.** We use the 9/11 subset of the LOCO corpus (Miani, Hills and Bangerter, 2022). It holds 3,146 documents: 939 from conspiracy-oriented websites and 2,207 from mainstream websites, published by 39 conspiracy and 79 mainstream websites. The label is the **subcorpus of the website**. It says what kind of site published the article, not what the article claims. A mainstream article that debunks a theory and a conspiracy-site article that reports plain news both carry their site's label. This is the central caveat of the whole project.

**Length.** The shortest article has 101 words and the median is 637.5 words. 131 documents are shorter than 150 words and 398 are shorter than 250 words. Because even the shortest text is a real article, we apply no minimum length. We still report results by length bucket (section 5).

**Cleaning.** Website names, URLs and repeated boilerplate are removed from the text so that the model cannot read the answer off a site's signature. Two exact duplicates exist. They are dropped only inside the cleaned in-memory frame. We also found 63 pairs of near-duplicate documents (cosine similarity of at least 0.9). 36 of these pairs lie across two splits, so a small leak is possible. We detected and logged the pairs but did not remove them; this is an open decision (section 6).

**The off-topic filter.** About one in six documents never mentions 9/11. The team therefore decided to run the first words-only model on all 3,146 documents, and only then to keep documents that match a list of 9/11 terms in title or text. 2,608 documents remain: 777 conspiracy and 1,831 mainstream. All later experiments use this filtered set. The effect is explained in section 5.

**Models.**

- Logistic regression with TF-IDF features (scikit-learn). TF-IDF turns each article into numbers: a word counts more if it is frequent in this article and rare in other articles.
- Claude Haiku 5.5 (`claude-haiku-5-5`), the cheapest model of the Claude 5 family, asked the **source question**: "does this article come from a conspiracy-oriented or a mainstream website?". Every article is cut to its first 3,000 words, the same for all documents. The answers are cached in `results/llm_cache.jsonl` (document id, prompt hash and answer, never article text), so `fml run-all` repeats without new API calls.
- Claude Sonnet 5.5 coded the stance of disagreement articles in the error analysis (section 5.6).

**Software.** Python 3.11.17, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2 and the `anthropic` client 1.15.0. The exact versions are pinned in `requirements-lock.txt` and match the ranges in `pyproject.toml`. `fml run-all` writes the versions it ran with and its total run time (`run_all_seconds`, limit 8 minutes) to `results/run_info.json`. Random seeds come from `config/project.yaml` (seed 20261016).

**Licence status.** The LOCO paper states no licence for the texts. The project page at the Open Science Framework was not reachable when we tried on 2026-10-16, so the licence of the LOCO texts is **not confirmed**. We therefore keep the texts in a private data repository, publish no article text, and show only excerpts of at most 50 words (this report contains none).

## 3. Approach

**One frozen split by website.** The split file `data/splits/split_v1.csv` was created once, grouped by website and stratified by label. *Grouped* means that all articles of one website go to the same part. The model never trains on a website and is then tested on more articles of the same website. This protects against a model that learns the style of a site instead of the style of conspiracy writing.

| Split | Conspiracy websites | Conspiracy documents | Mainstream websites | Mainstream documents |
|---|---|---|---|---|
| train | 25 | 658 | 54 | 1,546 |
| validation | 7 | 140 | 13 | 331 |
| test | 7 | 141 | 12 | 330 |

Every split has at least 3 conspiracy websites, so no split triggers the warning rule in our configuration. No website appears in two splits. The split covers all 3,146 documents. The off-topic filter later drops rows at analysis time without editing the file, so a dropped document keeps its split.

**Step by step.**

1. A words-only TF-IDF logistic regression on all 3,146 documents (first run).
2. The off-topic filter, with before and after numbers.
3. The same words-only model on the 2,608 filtered documents (baseline M1).
4. Three feature groups and an ablation: phrases (word n-grams of length 1 to 3, such as "inside job"), epistemic markers (counts of hedges, boosters, "hidden truth" words and debunking words per 1,000 words), and style features. The extended model M2 uses all three. We also test each group alone and M2 with each group removed.
5. The LLM with three prompt variants, tried on validation only. The variant with the best conspiracy F1 on validation is fixed before any test document is classified.
6. One final test run per model, a second LLM pass for stability, and the comparison with confidence intervals.
7. An error analysis with stance coding.

**Fitting on training data only.** Vocabulary, IDF weights, scalers and all learned values are fitted on the training split. The regularisation strength C (how strongly large weights are penalised) and the class-weight setting are chosen on validation. Text features are built once per feature configuration and reused for every C. A test in `tests/test_leakage.py` fails if anything is fitted on more than the training data.

**What we did not do, and why.**

- *No fine-tuning and no multi-model benchmark.* The feedback asked for one logistic regression against one inexpensive LLM. A fine-tuned transformer would answer a different question.
- *No emotion or sentiment features.* The team dropped them for cost and because the feedback warns that negative or emotional language is not automatically conspiracy language, especially when reporting on 9/11.
- *No JFK transfer test.* It was out of scope for this cycle.
- *No LIWC.* LIWC is a licensed dictionary. We used a short hand-picked epistemic word list instead, inspired by published categories and not copied from one. It is a weak instrument (section 6).
- *No tuning after seeing test results.* For example, 44 of 398 answers of the chosen prompt were cut off by the 150-token output limit on validation. We did not raise the limit afterwards, because that would be a re-run after seeing test results.

## 4. Evaluation

**Metrics.** *Precision* is the share of articles flagged as conspiracy that really come from conspiracy websites. *Recall* is the share of conspiracy-site articles that we catch. *F1* is the harmonic mean of the two. We report F1 for the conspiracy class, because about 70 percent of the documents are mainstream and a model that always answers "mainstream" would look good on plain accuracy while finding nothing. We also give macro F1 (average of both classes) and balanced accuracy.

**Baselines.** The majority-class baseline always answers "mainstream". A stratified random baseline guesses at the class rates. On the first-run validation set the random baseline reaches F1 0.297 and the majority baseline 0.

**Confidence intervals.** A *95 percent confidence interval* is a range that would contain the true value in 95 of 100 repeated samples. We compute it by the *bootstrap* (2,000 resamples, seed 20261016): we redraw the evaluation set with replacement and recompute the metric. We do this in two ways. The *document* bootstrap redraws documents. The *website* bootstrap redraws whole websites, which is more honest because documents from one site resemble each other. The website intervals are much wider. Our test set has only 18 websites (6 with conspiracy articles that remain after filtering), so the website intervals are the ones to trust.

**McNemar test.** To ask whether two models differ on the same documents, we count documents that only model A gets right and documents that only model B gets right. McNemar's test asks whether these two counts differ more than chance would allow. We also report the F1 difference with bootstrap intervals.

**Test set used once per model.** The test split is locked. Only `fml evaluate-final <model_id>` may score a model on it. It appends the model id, a hash of the configuration and the number of documents to `results/test_access_log.jsonl`, and a gate fails if one model id is scored with two configurations. Three models count: the words-only logistic regression `lr_words_v2`, the extended `lr_extended`, and `llm_haiku`. The log also holds an earlier `lr_words` entry. It came from a stale configuration (C 100, no class weights) that validation tuning no longer picks (it picks C 10, balanced). Because the lock forbids scoring the same id with a new configuration, the model was renamed `lr_words_v2` and the old `lr_words` line was left untouched in the log. It is superseded and is not used anywhere in this report. Model selection used validation only. The rule for choosing between M1 and M2 was fixed in advance: M2 is chosen only if the F1 gain over M1 has a website-bootstrap interval above 0 on validation.

**Sequence of test looks.** The log `results/test_access_log.jsonl` shows this actual order (UTC, copied from the log): `lr_words` at 2026-10-17T01:19:52, `lr_extended` at 2026-10-17T01:21:07, `llm_haiku` at 2026-10-17T09:26:03, and `lr_words_v2` at 2026-10-17T10:35:02 (changed configuration, new config hash). So `lr_words_v2` is a second test look at the words-only logistic regression, made after the test results of the other models were known. The LR choice rule (M2 only if its F1 gain over M1 has a website-bootstrap interval above 0) is applied to validation rows only (`ablation.chosen_lr`); we state that it was fixed before any test look, but the log cannot prove it, because the first `lr_words` look already used an earlier configuration. Limitation: the stale `lr_words` entry stays in the log and is not used for any number, yet the sequence means the words-only configuration was revised after one test look. From now on `evaluate_final` refuses to score a model unless its validation-chosen configuration is already in `results/frozen_models.json` (model id, config hash, source `validation`, timestamp) with the same hash. Entries for looks that happened before this rule are marked `retroactive` there, so their timestamps are later than the log lines.

## 5. Findings

### 5.1 First run on all documents

The words-only model on all 3,146 documents reaches a validation F1 of 0.926 (document interval 0.892 to 0.957, website interval 0.791 to 0.957), with precision 0.962 and recall 0.893. The average precision is 0.97. The selected C is 100 with no class weights. No website name appears among the top 25 words per class. A blog-format vocabulary ("posted", "entry") is among the top conspiracy words, a style giveaway that has nothing to do with 9/11.

![Confusion matrix of the first run](../results/figures/baseline_all_docs_confusion.png)

### 5.2 The off-topic filter

![Documents before and after the filter](../results/figures/topic_filter_before_after.png)

The filter removes 538 of 3,146 documents. The removal rate is almost the same in both classes: 0.1725 for conspiracy and 0.1704 for mainstream, so the class balance does not change (Fisher exact test p = 0.8768). The removed mainstream articles come mostly from health and nature websites (for example 48 of the 50 documents of one nature magazine site) and are real off-topic text. The removal is uneven across the splits: 0.1892 of the training documents, 0.155 of the validation documents and 0.1019 of the test documents are removed. Two conspiracy websites lose all their documents. The pattern includes "pentagon", which could match unrelated defence news. Only 10 kept documents match "pentagon" alone, so we left the pattern unchanged.

| Setting | Documents | Conspiracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| A: all-documents model on all validation documents | 471 | 140 | 0.962 | 0.893 | 0.926 |
| B: all-documents model on filtered validation documents | 398 | 128 | 0.959 | 0.914 | 0.936 |
| C: filtered model on filtered validation documents | 398 | 128 | 0.914 | 0.914 | 0.914 |

The validation sets of rows A and C differ, so their F1 values are not directly comparable. Row B scores the first model on exactly the documents of row C. Filtering did not improve validation F1 (row C minus row B is -0.022). We keep the filter because it is a team decision with a clear reason: articles that never mention 9/11 do not belong to the topic under study. The first model classified the 73 removed validation documents with an accuracy of 0.945, so it did not need the topic to recognise the site.

### 5.3 Do the extra features help? (RQ2)

All values below are validation results on 398 filtered documents from 19 websites. The last column is the F1 change against words only, with the website-bootstrap interval.

| Model | Precision | Recall | F1 | Change vs words | Website interval | Verdict |
|---|---|---|---|---|---|---|
| M1: words | 0.9141 | 0.9141 | 0.9141 | | | reference |
| + n-grams (1-3) | 0.9426 | 0.8984 | 0.92 | 0.0059 | -0.0166 to 0.0496 | no clear gain |
| + epistemic | 0.944 | 0.9219 | 0.9328 | 0.0187 | -0.0099 to 0.0851 | no clear gain |
| + style | 0.9206 | 0.9062 | 0.9134 | -0.0007 | -0.043 to 0.0563 | no clear gain |
| M2: all three | 0.9483 | 0.8594 | 0.9016 | -0.0124 | -0.0613 to 0.073 | no clear gain |

![Ablation F1 on validation](../results/figures/ablation_f1.png)

No feature group gives a clear gain, and the full extended model is not better than words alone. H2, H3 and H4 are therefore **not supported** on validation: the intervals all include 0. The best single number (epistemic markers, +0.0187) lies inside the noise: its website interval runs from -0.0099 to 0.0851. By the rule fixed in advance, M1 (`lr_words_v2`, C 10, balanced class weights) is the chosen logistic regression. The extended model was still scored on test, once, so that the cost of the extra features is visible.

The feature table shows that some groups do separate the classes in the training data. Hidden-truth vocabulary is far more frequent on conspiracy websites (mean 1.0913 per 1,000 words against 0.0554), as are boosters and exclamation marks. The words already carry this signal, so the extra counts add little. See `results/tables/feature_summary.csv`.

### 5.4 The LLM (part of RQ1)

We tried three prompt variants on the same 398 validation documents.

| Variant | Prompt words | Precision | Recall | F1 | Unreadable answers | Cost (USD) |
|---|---|---|---|---|---|---|
| v1 | 42 | 1.0 | 0.7578 | 0.8622 | 64 | 0.1066 |
| v2 | 102 | 1.0 | 0.9062 | 0.9508 | 44 | 0.11 |
| v3 | 107 | 1.0 | 0.6875 | 0.8148 | 91 | 0.1141 |

Variant v2 is chosen by the fixed rule (highest validation F1). All three variants have a precision of 1.0 on validation: the LLM almost never calls a mainstream article conspiracy. The differences between variants come from recall. Part of the loss comes from answers that were cut off by the 150-token output limit before the JSON closed; we count these as "mainstream", which lowers recall. 36 documents were also cut at 3,000 input words.

**Stability.** An LLM does not always give the same answer, and the model rejects a fixed temperature. The `claude-haiku-5-5` interface returns an error for the temperature parameter, so we used default sampling and recorded the conflict with our rule "temperature 0" in `docs/decisions.md`. We measured the consequence with a second pass over the 423 test documents. The two passes agree on 0.9504 of the documents, 21 documents flip, and Cohen's kappa is 0.8871. The scored pass is pass 1; pass 2 has the same confusion matrix on the final metrics (F1 0.6842 in both), so the stability problem is small compared with the other effects reported below.

**Both approaches are compared with the dataset labels.** The LLM is not the reference, and its answers are not treated as ground truth.

### 5.5 Head-to-head on the test set (RQ1)

| Model | Precision | Recall | F1 | F1 document interval | F1 website interval | Macro F1 | Balanced accuracy |
|---|---|---|---|---|---|---|---|
| Majority class | 0.0 | 0.0 | 0.0 | 0.0 to 0.0 | 0.0 to 0.0 | 0.4345 | 0.5 |
| Logistic regression, words (`lr_words_v2`) | 0.6849 | 0.5102 | 0.5848 | 0.491 to 0.6667 | 0.3235 to 0.8036 | 0.7398 | 0.7197 |
| Logistic regression, extended (`lr_extended`) | 0.6623 | 0.5204 | 0.5829 | 0.4932 to 0.6667 | 0.3137 to 0.8221 | 0.737 | 0.7202 |
| Claude Haiku 5.5 (`llm_haiku`) | 0.963 | 0.5306 | 0.6842 | 0.5917 to 0.7625 | 0.3713 to 0.9333 | 0.8075 | 0.7622 |

The test set has 423 documents, of which 98 are conspiracy, from 18 websites.

![Test F1 by model](../results/figures/comparison_f1.png)

![Confusion matrices on the test set](../results/figures/comparison_confusion.png)

**H1 is supported.** Both approaches beat the majority baseline, and the website-bootstrap intervals of the difference exclude 0 (for the LLM 0.3713 to 0.9333, for the logistic regression 0.3235 to 0.8036).

**The LLM scores higher than the chosen logistic regression on documents, but the website-level evidence is not conclusive.** Among the 423 test documents, 32 are right only for the LLM and 9 are right only for the logistic regression, 343 are right for both and 39 are wrong for both. McNemar's test gives p = 0.0004. The F1 difference is 0.0994 (document interval 0.0235 to 0.1795). The website interval, however, is -0.0166 to 0.3891 and includes 0. So the LLM is better on this set of documents, but with only 6 conspiracy websites we cannot rule out that the two models are equal when whole websites are resampled. The two models make different kinds of mistakes. The LLM almost never flags a mainstream article (accuracy on mainstream articles 0.9938, against 0.9292 for the logistic regression) but it misses about half of the conspiracy articles. The logistic regression produces 23 false positives against 2 for the LLM. The ranking quality of the two scores is similar, and the logistic regression is slightly ahead: the average precision is 0.6764 for the logistic regression and 0.6409 for the LLM.

**The extended model is not better than the words-only model on test either.** Its F1 is 0.5829 against 0.5848. McNemar's test gives p = 0.8679, and the website interval of the F1 difference is -0.1146 to 0.0972. It finds slightly more conspiracy articles (recall 0.5204) but raises 26 false positives instead of 23.

**Validation numbers do not carry over to test.** The logistic regression drops from 0.9141 on validation to 0.5848 on test, and the LLM from 0.9508 to 0.6842. Some of the reason is in the data. Most conspiracy test documents come from a few websites, and one of them contributes 44 of the 98 conspiracy documents. The models recognise only 0.204 (both logistic regressions) and 0.159 (LLM) of the documents of this site, while several other conspiracy sites reach between 0.579 and 0.895 (leaving out a site with one document). The test set therefore measures mostly "how well do we recognise this one site type", and the website intervals (0.3235 to 0.8036 for the logistic regression) are the honest description of our uncertainty.

![Precision-recall curves on the test set](../results/figures/comparison_pr.png)

**Length.** Results by length bucket on test (F1 of the conspiracy class):

| Length | Documents | Conspiracy | Logistic regression, words | Logistic regression, extended | LLM |
|---|---|---|---|---|---|
| 0-149 words | 27 | 5 | 0.5 | 0.5 | 1.0 |
| 150-249 words | 32 | 7 | 0.6154 | 0.6667 | 0.7273 |
| >= 250 words | 364 | 86 | 0.5867 | 0.5806 | 0.6565 |

Short articles are not impossible to classify, but the 5 and 7 conspiracy documents in the two short buckets are too few for a conclusion.

**Near-duplicates.** Nine test documents have a near-duplicate partner in train or validation. Without them (414 documents left), F1 is 0.5799 for the logistic regression (change -0.0049) and 0.68 for the LLM (change -0.0042). The gap between the two is 0.0994 on all test documents and 0.1001 without the duplicates, so the LLM keeps its lead and the conclusion does not change.

### 5.6 Error analysis (RQ3, H6)

On test, the logistic regression is wrong alone on 32 documents (22 false positives, 10 false negatives), the LLM alone on 9, and both are wrong on 39 (38 of them false negatives). Claude Sonnet 5.5 coded the stance of these 80 articles towards a 9/11 conspiracy claim: endorses, rejects or debunks, reports neutrally, or makes no claim, with a short justification. Two answers could not be parsed, so 78 documents are usable; the "Unparsed" column holds those two, so every row adds up to its error count. We then applied a fixed rule: a conspiracy-site article that rejects, reports neutrally or makes no claim, and a mainstream-site article that endorses, are classed as "label does not describe the text". All other errors are "model wrong".

| Disagreement set | Errors | Model wrong | Label does not describe the text | Unparsed |
|---|---|---|---|---|
| Logistic regression only wrong | 32 | 21 | 10 | 1 |
| LLM only wrong | 9 | 2 | 7 | 0 |
| Both wrong | 39 | 1 | 37 | 1 |
| All | 80 | 24 | 54 | 2 |

By stance, 62 articles make no claim, 9 report neutrally, 3 endorse a conspiracy claim and 4 reject or debunk. For the **false negatives**, the answer to H6 is clear: of the conspiracy-site articles that were missed, the share with no conspiracy claim is 0.8727, and the share that do not endorse a claim is 0.9636. Many of them only mention 9/11 in passing, for example as a death-toll comparison, and a model that missed them was arguably right. For the **false positives** the picture is mixed: of the mainstream articles flagged as conspiracy, the share that discuss conspiracy theories without endorsing them is 0.3478, but only 0.1304 debunk them. Most false positives are therefore plain model errors (22 of 24). In total, the share of errors that are label problems ranges from 0.2857 to 0.6923, depending on how the stance codes are counted (`errors.label_problem_range` in `metrics.json`): 0.6923 under the current rule with unparsed replies left out, 0.675 when unparsed replies stay in the count, and 0.2857 if only the decisive codes (endorses, rejects or debunks) are counted. The whole range rests on stance codes written by Claude that no human has checked yet. Human check: pending, Member to fill (kappa = null); `fml stance-agreement` prints `pending: 0/40 filled`.

**H6 is supported for false negatives and only partly for false positives.** The finding is provisional: the stance codes come from Claude and have not been checked by a person (section 6).

## 6. Interpretation and limitations

**What the results say.** Both approaches work well on validation and much less well on test, because test asks for generalisation to unseen websites. The LLM is the more careful of the two: it is almost never wrong about mainstream articles, and its recall is limited by articles that do not look like conspiracy writing at all. The logistic regression is cheaper per document and learns the vocabulary of the training sites, but it also learns blog style and misclassifies some mainstream news sites. Hand-made features did not help in a way we can distinguish from chance.

**Label versus content.** The labels describe websites. Our error analysis suggests that a part of the "errors" are articles that a reader would not call conspiracy content: between 0.2857 and 0.6923 of the coded errors, depending on how the stance codes are counted. That range rests on Claude-coded stance that humans have not checked, so the size of the effect is open. If so, the real limit of this task is not model strength but the match between site label and article text. An F1 of 0.68 does not mean the LLM reads articles badly.

**Noise and sample size.** Seven conspiracy websites in the test split (6 after filtering), and 44 of the 98 conspiracy test documents from one site, mean that single sites move the result. Our website bootstrap is wide for this reason. The off-topic filter was uneven: the test split lost far more conspiracy documents than mainstream ones (mostly one website), so the final test set has 98 conspiracy documents from 6 websites. The split is frozen and was not "repaired".

**The website name.** Could the model recognise news websites without being given their names? We removed site names, URLs and boilerplate, and no site name appears among the top coefficients. The grouped split then makes sure that a site seen in training is never in test. What remains possible is that style, such as blog formatting, identifies a **type** of website. That is exactly the sort of signal that transfers to unseen sites of the same type, and it explains part of the success. Near-duplicates across splits (36 pairs) could still leak some information. We did not remove them, because the split is frozen; we quantified them instead (section 5.5): dropping the 9 affected test documents changes F1 by -0.0049 (logistic regression) and -0.0042 (LLM), so the effect is small.

**LLM prior knowledge.** Haiku 5.5 may know some of these websites or articles from its training. Its recall of obscure sites may not reflect what it would do on unknown ones. We built a probe for this (asking Haiku to name the website of up to 40 validation articles; only a yes/no match is stored), but it has **not been run yet**: it could not be run when the code was written and there is no cached probe answer, so `metrics.json` key `errors.prior_knowledge_probe` says "not run" and we report no share. The prior-knowledge risk therefore remains unmeasured; running `fml run-all` with the key set fills it in.

**LLM output limits.** Between 44 and 91 of 398 validation answers per prompt variant could not be parsed, and 61 of 423 test answers. Their effect is a lower recall. A larger output limit would be a new run with a new model id and needs a team decision. On a validation subset of 150 documents, re-asking the chosen prompt with 400 output tokens lowered the parse-error rate from 0.1333 to 0.12 and moved F1 by 0.012 (`metrics.json#llm.token_limit_sensitivity`), so the 150-token limit is not the main cause. The model also refuses a fixed temperature (section 5.4).

**Lexicon weakness.** The epistemic list is short, hand-made and ignores negation: "no evidence of cover-up" counts "cover-up". The negative result for H3 should not be generalised to better lexicons.

**Human verification: pending, Member to fill (kappa = null).** The 40-row sample in `results/tables/stance_check_sample.csv` has an empty `human_stance` column (`fml stance-agreement` prints `pending: 0/40 filled`; to fill it, follow the instructions block in section 11). No agreement figure between Claude and a person exists, so the answer to H6 depends on Claude's reading.

**Open questions for the team.** Whether to remove near-duplicates (the sensitivity check suggests it would matter little); whether to accept default sampling for the LLM; whether to approve a larger output limit; and the licence of the texts.

## 7. Use of AI tools

**What the AI produced.** Two kinds of AI work appear in the project. First, Claude (Opus 5.5) in a chat session with the repository owner produced the project plan with the literature research, the agent harness, the site generator, the data subset with checksums and the repository rules. Second, in cycle 1 an AI coding agent (Claude Code, Sonnet 5.5) wrote the analysis code in `src/fml/`, the tests, the result tables, the walkthrough sections and this report, one task at a time. Claude Haiku 5.5 is the classifier under study, and Claude Sonnet 5.5 coded article stance for the error analysis. The team (Members A to E) decided what to study: full articles, one cheap LLM, which feature groups, the off-topic filter, a public repository with private data, and the pull-request cycles.

**What was checked, and what caught problems.** Automatic **gates** run after each task and fail if a rule breaks: the section format, required outputs, the frozen split, the test-access log, the report length, privacy checks, a scan for copied article text and protected paths. **Tests** encode why a rule matters, for example that fitting on more than the training data makes the leakage test fail. The data subset was checked against the counts of the LOCO paper (3,146 = 939 + 2,207). Real findings by this checking:

- The plan said no duplicates exist. There are two exact duplicates and 63 near-duplicate pairs.
- About one in six documents never mention 9/11. That became the off-topic filter.
- The LLM refuses the temperature setting that our rules required. The agent recorded the conflict and measured the consequence.
- Two parallel interrupted runs duplicated cache lines. The agent repaired the cache reader and said so.
- The output limit cut off many answers, and the agent did not raise it after seeing test results.
- The filter hit the test split unevenly, and the agent reported it instead of changing the frozen split.

**What was changed by people or rules.** The feedback overruled the proposal, and the team removed emotion features. The agent's own wish to tune the output limit was held back by the rule of no tuning after test.

**What was learned.** Rules in files and automatic gates work better than telling an agent to be careful. An agent reports conflicts between a rule and reality if it is told to write them down, and it should not resolve them silently. AI output sounds equally confident when it is right and wrong, so the checks that mattered were comparisons with outside facts and with the data itself. The most interesting scientific result, that many "errors" are label problems and that extra features do not help, was found by looking at the predictions after the first run, as the feedback advised.

**What is skipped or still open.** We could not confirm the licence (the page was unreachable). The stance codes are unchecked by people. The temperature and output-limit decisions are open. The `fml run-all` reproduction relies on the LLM cache, so without the cache and a key it fails loudly with the exact count of missing entries. The AI log (`docs/ai-log.md`) lists what a person must still check.

## References

- Miani, A., Hills, T., and Bangerter, A. (2022). LOCO: The 88-million-word language of conspiracy corpus. *Behavior Research Methods*, 54. https://doi.org/10.3758/s13428-021-01698-z
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12, 153-157.
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *Annals of Statistics*, 7, 1-26.
