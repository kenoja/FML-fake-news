---
title: Data, labels and cleaning
part: Data
hypothesis: With full articles even the shortest text is long enough to classify, so no minimum length is needed, and cleaning changes the texts only a little.
---
## Goal and hypothesis

We use the 9/11 part of the LOCO corpus: 3,146 full articles, 939 from conspiracy-oriented websites and 2,207 from mainstream ones. This section explains what the label means, how we clean the texts before any model sees them, and answers one open question from the team: **could very short texts be impossible to classify, and do we need a minimum length?** Our expectation: no, because these are full articles and the shortest has about 100 words.

## Explanation

**The label describes the website, not the article.** LOCO collected pages from websites that were rated as conspiracy-oriented or mainstream by an external media-rating source, then kept pages related to a seed topic (here "september 11 attack"). Every article inherits its website's label. So "conspiracy" means "published on a conspiracy-oriented site". A calm news item on such a site is labelled conspiracy, and a mainstream article that debunks a theory is labelled mainstream (or the reverse, if it uses the theory's vocabulary and sits on a conspiracy site). This is **label noise** at article level. Mompelat and colleagues, who studied LOCO, point to this noise: the website-level label is a practical shortcut and not a judgement of each text. We treat it as a known limit and look at actual disagreements in the error analysis.

**Licence status.** We found no licence in the corpus paper, and the OSF page could not be opened on 2026-10-16 or 2026-10-17 (access blocked; the recheck is a blocked item that a team member must do, see `docs/decisions.md`). Until a licence is confirmed we keep the texts in a private repository, never copy them here, and show at most short excerpts.

**Cleaning.** Models should read the article, not recognise where it came from. The steps, in order, with counts in the table below:

1. *Exact duplicates*: texts that are identical after lower-casing and removing punctuation and spacing are reduced to one copy (the first). A "hash" is a short fingerprint of a text; equal fingerprints mean equal texts.
2. *Near-duplicates*: pairs of texts whose **TF-IDF cosine similarity** is at least 0.9. TF-IDF turns a text into numbers that weigh words by how distinctive they are; cosine similarity measures how similar two such number lists are (1 = same words in the same proportions). We **detect and log** these pairs but do not remove them yet: removing them would change the 2,608 documents the team fixed for the off-topic filter, so it is a team decision (open in `docs/decisions.md`).
3. *URLs and e-mail-like strings* are deleted.
4. *Website names* (domains from the `website` field and from the metadata file, and the name stem when it is at least 6 letters) are deleted.
5. *Boilerplate lines*: lines that appear in at least 3 documents of the same website (menus, "share this", subscription banners) are deleted.

**Why fitting TF-IDF on the whole corpus here is not leakage.** *Leakage* means information from the test data reaches the model during training. The TF-IDF in step 2 is used only to ask "is this a copy of that?". It never becomes a model input, never sees labels, and no model learns from it. Every model fits its own vocabulary and TF-IDF weights on the training split only.

The cleaned texts exist only in memory while the code runs. Only counts are written down.

<!-- cell: src/fml/clean.py#clean_corpus -->
<!-- cell: src/fml/clean.py#find_near_duplicates -->
<!-- output: results/tables/cleaning_log.csv -->

Reading the log: the exact-duplicate step finds the duplicates, and the near-duplicate step lists pairs without removing them. Most near-duplicate pairs are between different websites (we have not inspected why; re-posts are the likely cause), and many of those pairs lie across two splits, so a model can meet a near copy of a test article in training. That makes the test slightly optimistic, and it is a limit we report (counts: `results/metrics.json#data`).

<!-- output: results/metrics.json#data -->

**Open question 5 – short texts and minimum length.** The table gives words per article by class (quantiles: "q5" means 5% of articles are shorter than this value). The figure shows the distribution on a log axis.

<!-- cell: src/fml/data_report.py#length_statement -->
<!-- output: results/tables/length_quantiles.csv -->
<!-- output: results/figures/length_distribution.png | Words per article by class (log scale) -->

Answer from the numbers: the shortest article has more than twice the 50 words we would call a snippet, so **no minimum length is applied**. The check is data-driven and rerun with the data (`results/metrics.json#data.length`). Short articles are still harder in principle (fewer words, noisier counts), so the first evaluation also reports F1 by length bucket instead of assuming it makes no difference. Conspiracy-site articles are longer than mainstream ones (compare the medians), so length alone carries some label signal. That is one reason the later feature groups normalise counts per 1,000 words.

## What could go wrong

- Treating the label as truth about the text: it is truth about the website only.
- Over-cleaning: removing a word that is also a normal word. Site-name stems shorter than 6 letters are left alone for this reason, so a few short site names may remain in the text.
- Under-cleaning: names written in other forms (for example spaced out) or in images are not caught, so a model may still pick up some site signal. The grouped split (next section) is the main protection.
- Boilerplate lines that are repeated within a site but differ slightly are not caught.
- Counting words by spaces gives slightly different numbers than the corpus's own word count.

## Technical infrastructure

- `src/fml/clean.py`: `clean_corpus`, `drop_exact_duplicates`, `find_near_duplicates`, `website_names`, `strip_urls_and_emails`, `strip_website_names`, `boilerplate_lines`.
- `src/fml/data_report.py`: `run_data_stage` (writes the tables, figures and `metrics.json#data`), `length_table`, `length_statement`.
- Command: `fml prepare-data`.
- Tests: `tests/test_split_clean.py` plants a URL, an e-mail, a website name and a duplicate in synthetic texts and checks they are removed.
