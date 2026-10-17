---
title: Where they fail
part: Evaluation
hypothesis: Some of the "mistakes" are not mistakes of the model but of the label, because the label describes the website and not the article.
---
## Explained simply

**Two kinds of wrong.** Every article in our data carries the label of the website it came from. That label is a good guess, but it is not always true for the single article. A mainstream paper can print a piece that debunks a 9/11 theory; a conspiracy website can print an ordinary news item. If a model calls the first one "conspiracy" or the second one "mainstream", is the model wrong, or is the label misleading? We want to know how often each happens.

**Where we look.** We take the test articles where the best logistic regression and the LLM do not agree with the label. We sort them into three groups: only the logistic regression is wrong, only the LLM is wrong, or both are wrong.

**Stance coding.** To judge the article and not the website, we ask a simple question that never mentions the website: *what does this article do with conspiracy claims about 9/11?* There are four possible answers:

- it **endorses** such a claim (presents it as true);
- it **rejects or debunks** such claims;
- it **reports** them **neutrally**, without taking a side;
- it contains **no claim** at all (for example ordinary news that only mentions the date).

Think of a teacher reading an essay and ticking one of four boxes. We call this "stance coding". A second language model (Claude Sonnet 5.5) does the ticking for us, because reading them all by hand would take too long.

**Turning ticks into a verdict.** We fixed a simple rule before looking at the results. If the website is a conspiracy site, but the article debunks, reports neutrally or has no claim, then the text does not look like conspiracy content, so we call it a *label problem*. The same is true for a mainstream site whose article endorses a claim. In every other case the text fits its label and the model simply missed it: a *model error*. A mainstream article that debunks a theory is still mainstream content, so calling it "conspiracy" is a model error.

**Why we do not trust the ticks yet.** The automatic coding is a second model with its own mistakes. It has not been checked by people, so none of the numbers from it count as verified. For that reason we prepared a sheet with 40 articles. Our team fills in its own tick for each article, and only then can we say how far the automatic coding can be trusted.

**Why we show examples.** Numbers hide what a mistake looks like, so we show a few short excerpts.

## The code

First we find, for each test article, who is wrong. The groups come from comparing both predictions with the label.

<!-- cell: src/fml/errors.py#disagreement_sets -->

The stance coding sends each article to the second model once and stores only the id and the answer, never the text. A second run reads the stored answers.

<!-- cell: src/fml/errors.py#code_stances -->

The rule turns the label and the stance into a verdict.

<!-- cell: src/fml/errors.py#classify_error -->

The check sheet for the team lists the article id and website with an empty column, and has no Claude answer; the answers go to a separate file. It is never overwritten once someone has started to fill it.

<!-- cell: src/fml/errors.py#write_check_sample -->

The examples are picked by a fixed rule: the first article in each cell.

<!-- cell: src/fml/errors.py#pick_examples -->

The stage puts it all together and writes the tables.

<!-- cell: src/fml/pipeline.py#errors_stage -->

<!-- output: results/metrics.json#errors.counts -->
<!-- output: results/tables/error_classification.csv -->
<!-- output: results/tables/error_examples.csv -->
<!-- output: results/tables/stance_check_sample.csv -->
<!-- output: results/tables/stance_claude_answers.csv -->

## Report

We examined the test articles on which the best logistic regression and the LLM disagree with the label. The groups are articles missed only by the logistic regression, only by the LLM, and by both. A false positive is a mainstream article called conspiracy, a false negative the reverse. The counts per group are in the first table below.

The labels describe websites, not articles. To separate model errors from label problems, a second model (Claude Sonnet 5.5) coded the stance of every disagreement article towards conspiracy claims about 9/11: endorses, rejects or debunks, reports neutrally, or no claim. The question did not mention the website. A fixed rule, written before the results were read, turns label and stance into a verdict. A conspiracy-site article that does not endorse a claim, or a mainstream article that endorses one, is counted as a label problem. All other cases are model errors. A mainstream article that debunks a theory counts as a model error, because it is still mainstream content.

<!-- output: results/tables/error_classification.csv -->

By this rule most of the errors are label problems, especially missed conspiracy-site articles. Many of them are ordinary news or commentary with no conspiracy claim, so a model that reads only the text has little to find. Mistakes on mainstream articles are mostly model errors: the models react to the topic words and not to the stance of the article. These shares depend on the automatic coding. Of the 67 coded articles, 2 received an answer that could not be read and are left out of the shares.

Five short excerpts, one per cell, are shown in the table below: an article missed only by the logistic regression, one missed only by the LLM, one missed by both, and for contrast one mainstream and one conspiracy article that both models classified correctly. Each excerpt is at most 25 words.

The table gives identifiers, websites and excerpts, never full texts.

<!-- output: results/tables/error_examples.csv -->

The automatic stance coding is not presented as verified. It was produced by a language model and no person has checked it yet. The team therefore carries out a human check on a sample of 40 of these articles. Each team member reads the article first, without seeing the automatic answer (it is kept in a separate file), and records one of the four stances in an empty column. The agreement between the team and the automatic coding will show how far the label-problem shares can be trusted. The label-problem share (47 of 65 readable codings) comes from one unchecked source, the Claude stance coding, so it is an estimate. Until this check is done, the split between model errors and label problems is an estimate from one unchecked source and should be read as such.

## Check yourself

Q: Why can a model be "wrong" on an article even though the model is not at fault?
A: The label comes from the website, not from the article. A conspiracy site can publish an ordinary news item, so a model that reads only the text has nothing to find and the label does not describe the text.

Q: Why do we call the automatic stance coding unverified?
A: A second language model made the ticks and no person has compared them with their own reading yet. Until the team has done this, we do not know how often the automatic coding is wrong.

Q: How do I fill in `results/tables/stance_check_sample.csv`? Follow these steps.
A: (1) Open the file: it has the columns `doc_id`, `website` and an empty `human_stance`; Claude's answers are deliberately not in this file. (2) Take the first `doc_id` and read the whole article in the private data repository. (3) Do not look for Claude's answers (`results/tables/stance_claude_answers.csv`) until all 40 rows are filled. (4) Ask: what does the article do with conspiracy claims about 9/11? (5) Write exactly one of these four values in `human_stance`: `endorses`, `rejects_or_debunks`, `reports_neutrally`, `no_claim`. (6) If the article only mentions 9/11 in passing, write `no_claim`. (7) Repeat for all 40 rows, save, and never paste article text into the file or the repository.
