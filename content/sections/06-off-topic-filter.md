---
title: Removing off-topic documents
part: Data preparation and evaluation
hypothesis: Documents that never mention 9/11 are easy to separate by topic alone, so removing them lowers the words-only score a little and makes the remaining task more honest; the filter removes both website types about equally.
---
## Goal and hypothesis

The corpus was collected with 9/11 as a search seed, but some documents never mention it (a medical dictionary entry, a cooking page). We remove them with the **topic filter**: a document stays only if its title or text matches the pattern in `config/project.yaml` (`data.topic_filter_pattern`: "9/11", "september 11", "world trade center", "pentagon", "ground zero", "wtc" and a few more, ignoring upper/lower case). The pattern is a team decision and was not changed after seeing results.

Hypothesis: (1) off-topic documents make the task artificially easy or noisy, so the score after filtering changes little or drops slightly; (2) the filter hits conspiracy and mainstream websites at a similar rate. We did **not** expect it to raise the score.

## Explanation

**Why filter after the first run, not before.** In the previous section the words-only model (M1) was trained on all 3,146 documents. We filter *after* that run so you can see the effect with your own eyes: same split, same model, same settings, once with and once without the off-topic documents. If we had filtered first, nobody could tell whether the filter helped, hurt or did nothing.

**What the filter does to the data.** Numbers are in `results/metrics.json#topic_filter` and the table below (before/after by class, split and website):

<!-- output: results/tables/topic_filter_counts.csv -->

- 3,146 documents before, **2,608 after**: 777 conspiracy and 1,831 mainstream. This matches the numbers the team expected (`matches_expected` is true), so nothing needed reconciling.
- 538 documents were removed (17.1%). The removal rate is 17.3% for conspiracy-site articles and 17.0% for mainstream-site articles. A Fisher exact test (a test of whether two shares differ more than chance would explain) gives p = 0.88, so overall the filter does **not** favour one class. The class balance barely moves: 29.9% conspiracy before, 29.8% after.
- Websites: 118 before, 116 after. Two conspiracy websites lose every document (one in validation, one in test). 39 conspiracy websites become 37 and 79 mainstream stay 79.

<!-- output: results/figures/topic_filter_before_after.png | Documents before and after the filter (left) and share removed per split and class (right) -->

**But the overall equality hides uneven spots.** Look at the right-hand panel and at the split-by-class rows of the table:

- Removal is concentrated in a few websites. A few mainstream sites that mostly publish health or general-knowledge pages lose almost everything (one science-and-history site 48 of 50 documents, one consumer-health site 45 of 55). On the conspiracy side, one blog-style site in the **test** split loses 30 of its 36 documents. See `results/tables/topic_filter_by_website.csv`.
- That one website makes the test split lose 30.5% of its conspiracy documents (43 of 141) but only 1.5% of its mainstream ones (5 of 330). In validation it is the other way round: 8.6% conspiracy against 18.4% mainstream. In train it is 16.3% against 20.1%. So the class mix per split shifts after filtering, even though the overall class mix does not. The share of all removed documents falls mostly on train (77.5%), then validation (13.6%) and test (8.9%) (`share_of_removed_by_split`). We report these counts only; no model has been scored on test.
- The Fisher test treats each article as independent, but articles from one website are alike, so the p-value is optimistic. Read "no overall difference" as "no evidence of a systematic difference", not as proof.

**What the removed documents looked like.** The most common words in removed titles (`results/tables/topic_filter_removed_title_words.csv`) are on the mainstream side health and reference words ("symptoms", "causes", "definition", "heart", "treatment") plus dictionary and encyclopedia pages. On the conspiracy side they are mostly titles of recurring show or newsletter formats of a few sites, and generic "attack" or "alert" headlines that do not mention 9/11 in the text. A random spot check of six removed documents (`results/tables/topic_filter_spot_check.csv`, titles only, cut to 50 words) shows the same picture: a history piece on tanks, a mental-health opinion, a dictionary entry, a cancer study, an explainer on a people group and a health-service cyber-attack notice. None of the six is about 9/11, so the filter does what we want on this sample. Six is a spot check, not a measurement.

**Which pattern term lets documents in.** Most kept documents match several terms. `results/tables/topic_filter_terms.csv` counts, per term, the documents it matches and the documents it matches *alone*:

<!-- output: results/tables/topic_filter_terms.csv -->

"9/11" alone admits 204 documents, "september 11" alone 156, "sept 11" 25, "11 september" 21, "9-11" 17. "pentagon" alone admits **10** documents (0.4% of the kept set, 7 of them conspiracy-site articles); "ground zero" alone admits 1 and "wtc" none. So the worry that "pentagon" would let in many defence stories is small in size here. But it is real in kind: the listed six titles for these ten (titles only, `topic_filter_spot_check.csv`) include a mainstream news story about US troops sent abroad because of a regional threat, which is a defence story and not a 9/11 story, and several conspiracy-site pieces on geopolitics that may mention the Pentagon in passing. Four of the ten mention "pentagon" only once (median 2 mentions). We looked at titles only, so we cannot say how many of the ten discuss 9/11 in some other wording. The additional check "contains attack, terror, hijack or plane" is true for all ten, but "plane" also matches "planet", so it is a weak signal. Conclusion: "pentagon" admits a handful of probably off-topic documents; it is too few to change any result visibly, and we keep the pattern as configured instead of tuning it after seeing results.

**Words-only model on the filtered set.** M1 was retrained with the same settings grid and the same frozen split on the 1,787 training and 398 validation documents that remain (128 conspiracy, 19 websites). Validation results (`metrics.json#baseline`):

<!-- output: results/metrics.json#baseline.validation -->
<!-- output: results/tables/topic_filter_f1_comparison.csv -->

Three rows answer different questions:

- **A** is the first run: the all-documents model on all 471 validation documents, F1 0.926 (website-level CI 0.784 to 0.958).
- **C** is the filtered model on the 398 filtered validation documents, F1 0.927 (document CI 0.891 to 0.959, website CI 0.796 to 0.963). Precision 0.966, recall 0.891.
- **B** is the like-for-like row: the *all-documents* model scored on exactly the same 398 documents as C, F1 0.940.

**A and C cannot be compared directly**, because their validation sets contain different documents; the equal-looking F1 values (0.926 and 0.927) are a coincidence of two different samples. The fair comparison is B against C: training on the filtered set gave F1 0.927 against 0.940, a difference of -0.013, far inside the intervals. We do not claim that filtering helps or hurts and we ran no significance test, because B and C are not independent samples of a stable effect on 19 websites. What can be said: the first-run score was not an artefact of off-topic documents. The all-documents model got 93.2% of the 73 removed validation documents right (12 of them conspiracy), about as good as on-topic documents (95.5% for C), so these documents were **not** noticeably easier.

The selected regularisation strength is again the grid edge (C = 100), so as in the first run we note that the model prefers very weak regularisation.

## What could go wrong

- *The filter is itself a design choice.* A keyword list decides what "about 9/11" means. A different list gives a different dataset. We fixed the list before looking at results.
- *It can remove relevant context.* An article may discuss the event without any listed phrase ("the attacks", "two planes"), or a conspiracy-site article may argue about related events. Such a document is removed, and the model never sees how sites write about it.
- *It can shift class balance in a split.* Overall the shares barely move, but the test split loses nearly a third of its conspiracy documents, so conspiracy recall on test will rest on 98 articles from 6 websites, and validation on 128 from 6.
- *It can admit off-topic documents.* "pentagon" alone lets in about ten probably unrelated defence or politics documents.
- *Counting words is not reading.* Spot checks use titles only and six documents each; they illustrate, they do not prove.
- *Small validation set.* The 19 validation websites keep every interval wide, especially the website-level one.

## Technical infrastructure

- `src/fml/data.py#on_topic_mask` applies the configured pattern to the original title and text. `src/fml/topic.py` splits the pattern into its terms for the diagnostic and cuts titles to the excerpt limit.
- `src/fml/pipeline.py#topic_filter` is the stage that counts and diagnoses; `pipeline.analysis_frame()` is the **one function** that returns the filtered, cleaned documents. Every later experiment must use it, so no stage can return to the unfiltered corpus by accident. `pipeline.baseline_filtered` retrains M1 on that frame with the same split.
- The split file is read only. Removed documents keep their entry, and `topic_filter` stores the file hash in `metrics.json#topic_filter.split_file_sha256`.
- Outputs: `results/metrics.json#topic_filter` and `#baseline`; `results/tables/topic_filter_*.csv`, `m1_words_filtered_*.csv`; `results/figures/topic_filter_before_after.png`, `baseline_filtered_confusion.png`.
- Tests: `tests/test_topic_filter.py` (pattern read from config, terms agree with the filter, subset and split unchanged).
- Run: `fml run-all`.

<!-- cell: src/fml/data.py#on_topic_mask -->
<!-- cell: src/fml/pipeline.py#filter_to_topic -->
<!-- cell: src/fml/pipeline.py#topic_filter -->
<!-- cell: src/fml/pipeline.py#baseline_filtered -->
