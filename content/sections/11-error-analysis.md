---
title: Error analysis
part: Evaluation
hypothesis: H6 - many false positives are mainstream articles that discuss or debunk conspiracy theories, and many false negatives are conspiracy-site articles that contain no conspiracy claim, so part of the "error" is the label, not the model.
---
**In five lines**

1. We compared the LR and the LLM on the test articles where they disagree, and asked Claude how each article treats conspiracy claims.
2. Conspiracy-site misses are mostly articles with no conspiracy claim; mainstream false positives are mostly plain model errors.
3. The share of errors that are label problems is a range, not one number: see `errors.label_problem_range` below.
4. The range rests on stance codes written by Claude and not yet checked by humans, so the H6 verdict is provisional.
5. The long detail (rule, worked examples, how to check the codes) follows in the sub-headings below; read it only if you need it.

## Goal and hypothesis

The labels say which **website** published an article, not what the article says. A mainstream outlet can run a piece that debunks 9/11 theories; a conspiracy site can run a plain news item. If the model "fails" on such an article, is the model wrong, or does the label simply not describe the text? This section counts both cases on the **test** documents (hypothesis H6, see the header). Everything here is read from the saved prediction files of `fml evaluate-final`; nothing is scored or selected on test.

**Human verification: pending, Member to fill (kappa = null).** To fill: open `results/tables/stance_check_sample.csv`, read each article in the private data repo, write one of `endorses`, `rejects_or_debunks`, `reports_neutrally`, `no_claim` into `human_stance`, then run `fml stance-agreement` (it shows `pending: 0/40 filled` until you do). Justifications in the public CSVs are cut to 25 words and name-like words are replaced by `[name]` (`errors.scrub_justification`). The stance codes below were produced by Claude (Sonnet 5.5). A teammate has not yet checked any of them, so no agreement figure (Cohen's kappa) exists yet and none is reported. The H6 verdict is therefore **provisional**: it rests on Claude's reading of the articles.

## Explanation

**Disagreement sets.** For every test document we compare the best logistic regression (`lr_words_v2`) and the LLM (Haiku, pass 1) with the true label:

- `lr_only_wrong`: the LR is wrong, the LLM is right;
- `llm_only_wrong`: the LLM is wrong, the LR is right (replies that could not be parsed count as "mainstream", so some of these are formatting failures);
- `both_wrong`: both are wrong.

A *false positive* is a mainstream article called conspiracy; a *false negative* is a conspiracy-site article called mainstream. Counts by set and class are in `metrics.json` under `errors.counts`:

<!-- output: results/metrics.json#errors.counts -->

**Stance coding.** Claude Sonnet 5.5 read each disagreement article (the first 3,000 words, the same cleaned text the other models saw) and answered one question that does *not* mention the website: how does the article treat conspiracy claims about 9/11?

- `endorses`: presents such a claim as true or likely true;
- `rejects_or_debunks`: discusses such claims in order to refute them;
- `reports_neutrally`: describes the claims or the debate without taking a side;
- `no_claim`: makes and discusses no such claim.

It also gave a justification of at most 25 words. We coded **all** 80 disagreement documents (the cap in `config/project.yaml` is 40 for the two smaller sets, and none exceeds it). Answers are cached in `results/stance_cache.jsonl` (document id, prompt hash and answer only, never article text), so reruns need no key. Without a key and without a complete cache the stage fails loudly.

**The rule: "model wrong" or "label does not describe the text".** Stated before looking at results:

| True label (website) | Stance | Verdict |
|---|---|---|
| conspiracy | endorses | model wrong |
| conspiracy | rejects_or_debunks, reports_neutrally, no_claim | label does not describe the text |
| mainstream | endorses | label does not describe the text |
| mainstream | rejects_or_debunks, reports_neutrally, no_claim | model wrong |

The last row is deliberately strict: a mainstream article that debunks theories *is* mainstream content, so calling it "conspiracy" is a model mistake, even though we can explain it (the model reacted to the topic). Documents whose stance reply could not be parsed are counted in the "unparsed" column and left out of the verdicts, so each row of the table adds up to its error count.

### Results

<!-- output: results/tables/error_classification.csv -->

<!-- output: results/metrics.json#errors.h6 -->

Read the `h6` block as `hit` out of `n` (with `share`):

- **False negatives (conspiracy sites missed).** `fn_conspiracy_no_claim`: 48 of 55 coded documents (87%) contain no conspiracy claim at all according to Claude. Another 5 only report or reject. Only 2 endorse a claim and are really "missed" conspiracy content. Many of these pages are show announcements, politics or health news that merely mention 9/11. The label (conspiracy *website*) is true, but a text-only model has nothing to find.
- **False positives (mainstream articles flagged).** `fp_mainstream_not_endorsing_but_discusses`: 8 of 23 (35%) discuss conspiracy claims, 3 of them to debunk them (13%) and 5 neutrally. Most others were classified `no_claim`, so the model flagged them for other reasons (topic words such as "Saudi", "FBI", "government").
- Overall, between 0.2857 and 0.6923 of the coded errors are "label does not describe the text", depending on how the stance codes are counted (see "How sure is the label-problem share?" below).

**Verdict on H6: supported for false negatives, only partly for false positives.** The false-negative half is strong (87% have no claim), but see the cautions below. For false positives, a minority debunk or discuss conspiracy theories (35%), and only 13% debunk; the large majority of FP errors are plain model mistakes (22 of 24 by the rule). Because Claude's stance codes are unchecked, we call this provisional.

The two models err differently. In `lr_only_wrong` the LR's false positives are mostly "model wrong" (20 of 22), while in `both_wrong` almost everything (37 of 39) is a conspiracy-site article without a conspiracy claim: neither approach can be expected to get these right from the text.

### How sure is the label-problem share?

One number would look more certain than it is, so we compute it three ways (`label_problem_range`): (a) the current rule with every coded error in the count, (b) counting only the decisive stance codes (endorses, rejects or debunks), and (c) the current rule leaving out unparsed replies. The range is the minimum and maximum of the three. It rests on Claude-coded stance that humans have not checked.

<!-- output: results/tables/label_problem_range.csv -->

<!-- output: results/metrics.json#errors.label_problem_range -->

Variant (b) is low because most "label problem" verdicts come from the vague codes `no_claim` and `reports_neutrally`, which that variant does not count. Whether those vague codes deserve the verdict is what the human check must decide.

### Worked examples

`results/tables/error_examples.csv` has one row per case (no text). It lists the top LR contributions (coefficient x TF-IDF weight for that document; positive pushes towards conspiracy) and the LLM's reason. Excerpts are cut to a few words.

<!-- output: results/tables/error_examples.csv -->

**1. Mainstream article that debunks (false positive explained).** `M07f55` (politico.eu), LR-only error, stance `rejects_or_debunks`:

> Mysterious terrorist attacks prompt public panic, allowing a cynical government to trample on the constitution. Would that be Russia after the 1999 apartment block bombings [...]

The LR's strongest conspiracy words are "official", "russia", "russian", "is", "story"; "theories" and "theorists" pull towards mainstream but not enough. The LLM answered correctly ("weighs both sides, scrutinizes conspiracy theories skeptically"). The LR reacts to the vocabulary of the topic, not to the article's stance. By the rule: **model wrong**.

**2. Conspiracy site, no claim.** `C0051b` (neonnettle.com), LR-only error, stance `no_claim`:

> A nine-foot-tall statue honoring Saudi Arabia has been placed on the grounds of the World Trade Center that is wrapped in the nation's emerald flag

The article is outrage news about a statue, not a theory about 9/11. The LR called it mainstream ("qaeda", "killed" pull that way); with no theory in the text there is little for a word model to find. The LLM called it conspiracy from tone ("sensationalist framing ... typical of alarm-driven outlets"), a judgement about the site's style that happened to match the label. By the rule: **label does not describe the text**.

**3. Conspiracy article that endorses, missed by the LR.** `C0245a` (conspiracydailyupdate.com), stance `endorses`. The words "mossad", "israeli", "intelligence", "government" push towards conspiracy, but the LR still said mainstream. By the rule: **model wrong**; the LLM ("claims Mossad secretly ran the 9/11 hijackers ... classic cover-up narrative") was right.

**4. Mainstream site that endorses.** `M1b4b7` (foxnews.com), stance `endorses`:

> The FBI has documents that could help prove Saudi officials were involved in the 9/11 terror attacks, an attorney representing several families [...] told Tucker Carlson Wednesday.

"fbi", "saudi", "government", "ruling" are strong conspiracy words for the LR. The article relays claims of an FBI cover-up, so a conspiracy call is understandable. By the rule: **label does not describe the text**. (The LLM's reply for this document could not be parsed.)

**5. Opinion piece on a conspiracy site, LR-only error.** `C00a99` (thelibertybeacon.com), stance `no_claim`:

> Donald Trump Didn't Lie About Where He Was after the 9/11 Attacks, But Hillary Clinton Did

This is partisan election commentary. Words such as "truth", "clinton" and "lie" push the LR towards conspiracy; yet the LLM itself noted "not overtly conspiratorial". It is another **label does not describe the text** case, here caught by tone rather than claim.

### How to fill in `human_stance` (walkthrough for teammates)

The file `results/tables/stance_check_sample.csv` has 40 disagreement documents (seeded random sample, spread over the three sets). Its columns are `doc_id, website, true_label, lr_pred, llm_pred, claude_stance, claude_justification, human_stance, human_note`. `human_stance` and `human_note` are **empty on purpose**. To check Claude:

1. **Find the article.** The texts are in the private data repository (found by `fml.paths.data_dir()`), file `loco_911/loco_911.jsonl.gz`: one JSON object per line with the field `doc_id`. In Python: open it with `gzip.open(..., "rt")`, read line by line and stop at the line whose `doc_id` matches. Read the field `txt`. Do **not** copy the text anywhere in this public repository.
2. **Read first, then look.** Read at least the first 500 words **before** reading the `claude_stance` column, so that you are not influenced by it. (If you cannot avoid seeing it, say so in `human_note`.)
3. **Pick exactly one value** for `human_stance`, written exactly like this: `endorses`, `rejects_or_debunks`, `reports_neutrally`, `no_claim`. The question: how does the article treat conspiracy claims about 9/11 (inside job, cover-up, the official account is a lie)?
4. **Unclear cases.** If the article only mentions 9/11 in passing: `no_claim`. If it describes claims but you cannot tell whether it sides with them: `reports_neutrally`. If it endorses a claim not about 9/11 itself (for example about Saudi involvement or an FBI cover-up), choose by whether it presents the claim as true; write your reasoning in `human_note`. If you truly cannot decide, leave `human_stance` empty and say why in `human_note`; empty rows are left out of the agreement.
5. **No article text in `human_note`**: write your reason in your own words (at most 25 words).
6. **Agreement.** Run `fml stance-agreement` (or `python -m fml.cli stance-agreement path/to/file.csv`). It prints how many rows are filled, the share of agreement and **Cohen's kappa** (agreement between two raters beyond what chance alone would give: 1 means perfect, 0 means no better than chance). While `human_stance` is empty it prints `pending: 0/40 filled` and kappa = null. Rerunning `fml run-all` never overwrites a file that already contains human entries.

Once at least some rows are filled, the report quotes the kappa from this command; until then the stance codes stay "unverified".

## What could go wrong

- **Claude's stance codes are unchecked.** Sonnet is a second model with its own mistakes; the "label problem" counts are only as good as its reading. The human sample exists to measure this; until it is filled, treat the percentages as estimates.
- **The stance question and the label are different things.** `no_claim` includes articles about other conspiracy topics (for example 2020 Benghazi or Templar history); they may well be typical conspiracy-site content. "Label does not describe the text" means *about 9/11 conspiracy claims*, not "this site is not conspiratorial".
- **Few websites, big clusters.** One test website (neonnettle.com) supplies many of the false negatives (see section 10), so the shares are driven by a few sites.
- **The rule is a convention.** A different but reasonable rule (for example treating a debunking mainstream article as a label problem) would change the counts; the stance columns in `error_classification.csv` let readers re-count.
- **Parse errors.** Two stance replies could not be parsed (not coded), and some LLM classifier replies were unparsable (counted as "mainstream" in `llm_only_wrong`).
- **Prompt sensitivity.** The first stance prompt let the model reason before answering and produced 21 unparsable replies; the final prompt demands JSON only and allows more tokens. The cache holds only the final prompt.

## Technical infrastructure

- `src/fml/errors.py`: disagreement sets, selection for coding, stratified human sample, stance cache and client, the rule, the label-problem range, kappa.
- `src/fml/pipeline.py#errors_stage`: runs it all (also part of `fml run-all`) and writes `results/metrics.json` (key `errors`), `results/tables/error_classification.csv`, `error_examples.csv` and `stance_check_sample.csv`. The LR is refitted on the **training** split only, just to compute per-document contributions.
- Settings: `config/project.yaml`, block `stance_coding` (model `claude-sonnet-5-5`, key from `LLM_CLASSIFIER_API_KEY`, cache path, cap, sample size).
- Commands: `fml run-all`, `fml stance-agreement [csv]`.
- Tests: `tests/test_errors.py` (set counts by hand, rule table, seeded caps, honest kappa, no overwrite of human work, excerpt limits).

<!-- cell: src/fml/errors.py#disagreement_sets -->

<!-- cell: src/fml/errors.py#classify_error -->

<!-- cell: src/fml/errors.py#human_agreement -->

<!-- cell: src/fml/pipeline.py#errors_stage -->
