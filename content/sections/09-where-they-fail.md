---
title: Where they fail
part: Evaluation
hypothesis: Some of the "mistakes" are not mistakes of the model but of the label, because the label describes the website and not the article.
---
## Explained simply

**Two kinds of wrong.** Every article carries the label of its website. That is a good guess, but not always true for a single article. A mainstream paper can print a piece that debunks a 9/11 theory; a conspiracy website can print an ordinary news item. If a model gets such an article "wrong", is the model at fault, or the label?

**Where we look.** We take the test articles where the best logistic regression or the LLM disagrees with the label, in three groups: only the logistic regression is wrong, only the LLM, or both.

**Stance coding.** To judge the article and not the website, we ask one question that never mentions the website: *what does this article do with conspiracy claims about 9/11?* It **endorses** one, **rejects or debunks** them, **reports** them **neutrally**, or contains **no claim** (for example ordinary news). Think of a teacher ticking one of four boxes per essay. A second language model (Claude Sonnet 5.5) does the ticking, because reading everything by hand would take too long.

**Turning ticks into a verdict.** We fixed a simple rule before looking at results. A conspiracy-site article that debunks, reports neutrally or has no claim does not look like conspiracy content: a *label problem*. So does a mainstream article that endorses a claim. Every other case is a *model error*, even a mainstream article that debunks a theory, because it is still mainstream content. One consequence: a conspiracy-site article with no conspiracy claim is a label problem **by definition**, so part of the label-problem share follows from the rule itself.

**Why we do not trust the ticks yet.** The coding is a second model with its own mistakes, and no person has checked it. We prepared a sheet of 40 articles. Each teammate ticks their own stance and returns the sheet (see the last question below). We then measure agreement with Claude using *Cohen's kappa*: 1 is perfect agreement, 0 is no better than chance, since raters who often tick the same box agree sometimes by luck.

**Why we show examples.** Numbers hide what a mistake looks like. A fixed random seed picks one article per case, with a clean passage (no menu text, no named people): logistic regression wrong and Claude right, the reverse, one mainstream article called conspiracy, one conspiracy-site article called mainstream.


## The code

First we find, for each test article, who is wrong. The groups come from comparing both predictions with the label.

<!-- cell: src/fml/errors.py#disagreement_sets -->

The stance coding sends each article to the second model once and stores only the id and the answer, never the text. A second run reads the stored answers.

<!-- cell: src/fml/errors.py#code_stances -->

The rule turns the label and the stance into a verdict.

<!-- cell: src/fml/errors.py#classify_error -->

The check sheet for the team lists the article id and website with an empty column, and has no Claude answer; the answers go to a separate file. It is never overwritten once someone has started to fill it.

<!-- cell: src/fml/errors.py#write_check_sample -->

The examples are picked with a seeded shuffle among articles that have a clean excerpt.

<!-- cell: src/fml/errors.py#find_excerpt -->

<!-- cell: src/fml/errors.py#pick_examples -->

Once the team has returned the sheet, the agreement is computed by this function.

<!-- cell: src/fml/errors.py#stance_agreement -->

The stage puts it all together and writes the tables.

<!-- cell: src/fml/pipeline.py#code_disagreements -->

<!-- cell: src/fml/pipeline.py#pick_examples -->

<!-- cell: src/fml/pipeline.py#errors_stage -->

<!-- output: results/metrics.json#errors.counts -->
<!-- output: results/tables/error_classification.csv -->
<!-- output: results/tables/error_examples.csv -->
<!-- output: results/tables/error_examples_plain.csv -->
<!-- output: results/tables/stance_check_sample.csv -->
<!-- output: results/tables/stance_claude_answers.csv -->

## Report

We examined the test articles on which the best logistic regression and the LLM disagree with the label. The groups are articles missed only by the logistic regression, only by the LLM, and by both. A false positive is a mainstream article called conspiracy, a false negative the reverse. The counts per group are given in the first table of this part. The stance codes come from a single language model and no person has checked them yet.

The labels describe websites, not articles. To separate model errors from label problems, a second model (Claude Sonnet 5.5) coded the stance of every disagreement article towards conspiracy claims about 9/11: endorses, rejects or debunks, reports neutrally, or no claim. The question did not mention the website. A fixed rule, written before the results were read, turns label and stance into a verdict. A conspiracy-site article that does not endorse a claim, or a mainstream article that endorses one, is counted as a label problem. All other cases are model errors. A mainstream article that debunks a theory counts as a model error, because it is still mainstream content. By this rule, a conspiracy-site article that makes no conspiracy claim is a label problem by definition, so the share of label problems follows partly from the rule.

<!-- output: results/tables/error_classification.csv -->

By this rule most of the errors are label problems, especially missed conspiracy-site articles. Many of them are ordinary news or commentary with no conspiracy claim, so a model that reads only the text has little to find. Mistakes on mainstream articles are mostly model errors, since the rule counts a mainstream article that does not endorse a claim as correctly labelled. We did not test why the models call these articles conspiracy. These shares depend on the automatic coding. Of the 67 articles with an error, 65 received a readable stance and 2 an answer that could not be read; the shares use the 65.

Four examples show what these cases look like.

<!-- output: results/tables/error_examples_plain.csv -->

The 40 articles of the human check come from 19 websites, 6 conspiracy-oriented and 13 mainstream, with at most 4 articles from any one website.

The automatic stance coding is not verified: no person has checked it yet, and the team will check 40 articles by hand. Each teammate returns the file `results/tables/stance_check_sample.csv` with the column `human_stance` filled in, through a pull request or by email to the project lead, and agreement with Claude's stance is computed as Cohen's kappa. The label-problem share (47 of 65 readable codings) is therefore an estimate from one unchecked source.

## Check yourself

Q: Why can a model be "wrong" on an article even though the model is not at fault?
A: The label comes from the website, not from the article. A conspiracy site can publish an ordinary news item, so a model that reads only the text has nothing to find and the label does not describe the text.

Q: Why do we call the automatic stance coding unverified?
A: A second language model made the ticks and no person has compared them with their own reading yet. Until the team has done this, we do not know how often the automatic coding is wrong.

Q: How do I return the filled sheet, and how is it scored?
A: Send `results/tables/stance_check_sample.csv` with the `human_stance` column filled in, either as a pull request or by email to the project lead. The project lead compares it with Claude's stance and computes Cohen's kappa: 1 is perfect agreement, 0 is no better than chance.

Q: Why is a conspiracy-site article with no conspiracy claim a label problem and not a model error?
A: Because the rule says so: the text does not look like conspiracy content, so the label (the website) describes it badly. This is a definition we chose, not a finding.

Q: How do I fill in `results/tables/stance_check_sample.csv`? Follow these steps.
A: (1) Open the file: it has the columns `doc_id`, `website` and an empty `human_stance`; Claude's answers are deliberately not in this file. (2) Take the first `doc_id` and read the whole article in the private data repository. (3) Do not look for Claude's answers (`results/tables/stance_claude_answers.csv`) until all 40 rows are filled. (4) Ask: what does the article do with conspiracy claims about 9/11? (5) Write exactly one of these four values in `human_stance`: `endorses`, `rejects_or_debunks`, `reports_neutrally`, `no_claim`. (6) If the article only mentions 9/11 in passing, write `no_claim`. (7) Repeat for all 40 rows, save, and never paste article text into the file or the repository. (8) Return the file by pull request or by email to the project lead.
