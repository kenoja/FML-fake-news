---
title: Where they fail
part: Evaluation
hypothesis: Some of the "mistakes" are not mistakes of the model but of the label, because the label describes the website and not the article.
---
## Explained simply

**Two kinds of wrong.** Every article in our data carries the label of the website it came from. That label is a good guess, but it is not always true for the single article. A mainstream newspaper can print a piece that explains why a 9/11 theory is nonsense. A conspiracy website can print an ordinary news item about a statue. If a model calls the first one "conspiracy" or the second one "mainstream", is the model wrong, or is the label misleading? We want to know how often each happens.

**Where we look.** We take the test articles where the best logistic regression and the LLM do not agree with the label. We sort them into three groups: only the logistic regression is wrong, only the LLM is wrong, or both are wrong. Looking at what these articles actually say is more useful than looking at one more score.

**Stance coding.** To judge the article and not the website, we ask a simple question that never mentions the website: *what does this article do with conspiracy claims about 9/11?* There are four possible answers:

- it **endorses** such a claim (presents it as true);
- it **rejects or debunks** such claims;
- it **reports** them **neutrally**, without taking a side;
- it contains **no claim** at all (for example ordinary news that only mentions the date).

Think of a teacher reading an essay and ticking one of four boxes. The tick says nothing about who wrote it. We call this ticking "stance coding". A second language model (Claude Sonnet 5.5) does the ticking for us, because reading all disagreement articles by hand would take too long.

**Turning ticks into a verdict.** We fixed a simple rule before looking at the results. If the website is a conspiracy site, but the article debunks, reports neutrally or has no claim, then the text does not look like conspiracy content, so we call it a *label problem*. The same is true for a mainstream site whose article endorses a claim. In every other case the text fits its label and the model simply missed it: a *model error*. One case is strict on purpose: a mainstream article that debunks a theory is still mainstream content, so calling it "conspiracy" is a model error, even if we can understand how the model was fooled by the topic words.

**Why we do not trust the ticks yet.** The automatic coding is a second model with its own mistakes. It has not been checked by people, so none of the numbers from it count as verified. For that reason we prepared a sheet with 40 articles. Our team fills in its own tick for each article, and only then can we say how far the automatic coding can be trusted.

**Why we show examples.** Numbers hide what a mistake looks like. A few short excerpts show how a model can be fooled, for example by a long article that spends one sentence on the topic.

## The code

First we find, for each test article, who is wrong. The groups come from comparing both predictions with the label.

<!-- cell: src/fml/errors.py#disagreement_sets -->

The stance coding sends each article to the second model once and stores only the id and the answer, never the text. A second run reads the stored answers.

<!-- cell: src/fml/errors.py#code_stances -->

The rule turns the label and the stance into a verdict.

<!-- cell: src/fml/errors.py#classify_error -->

The check sheet for the team is written with an empty column. It is never overwritten once someone has started to fill it.

<!-- cell: src/fml/errors.py#write_check_sample -->

The stage puts it all together and writes the tables.

<!-- cell: src/fml/pipeline.py#errors_stage -->

<!-- output: results/metrics.json#errors.counts -->
<!-- output: results/tables/error_classification.csv -->
<!-- output: results/tables/error_examples.csv -->
<!-- output: results/tables/stance_check_sample.csv -->

## Report

We examined the test articles on which the best logistic regression and the LLM disagree with the label. The groups are articles missed only by the logistic regression, only by the LLM, and by both. A false positive is a mainstream article called conspiracy, a false negative the reverse. The counts per group are in the first table below.

The labels describe websites, not articles. To separate model errors from label problems, a second model (Claude Sonnet 5.5) coded the stance of every disagreement article towards conspiracy claims about 9/11: endorses, rejects or debunks, reports neutrally, or no claim. The question did not mention the website. A fixed rule, written before the results were read, turns label and stance into a verdict. A conspiracy-site article that does not endorse a claim, or a mainstream article that endorses one, is counted as a label problem. All other cases are model errors. A mainstream article that debunks a theory counts as a model error, because it is still mainstream content.

<!-- output: results/tables/error_classification.csv -->

By this rule most of the errors are label problems, especially missed conspiracy-site articles. Many of them are ordinary news or commentary with no conspiracy claim, so a model that reads only the text has little to find. Mistakes on mainstream articles are mostly model errors: the models react to the topic words and not to the stance of the article. These shares depend on the automatic coding.

Two short excerpts illustrate the pattern. A mainstream article that criticises conspiracy theories was flagged by the logistic regression:

> Mysterious terrorist attacks prompt public panic, allowing a cynical government to trample on the constitution. Would that be Russia after the 1999 apartment block bombings [...]

A conspiracy-site article that the logistic regression missed, although it presents a claim as fact, shows the opposite case:

> British intelligence reported in February 2002 that the Israeli Mossad ran the Arab hijacker cells that were later blamed [...]

The table of all examples, with identifiers and no full text, is available in the results folder.

<!-- output: results/tables/error_examples.csv -->

The automatic stance coding is not presented as verified. It was produced by a language model and no person has checked it yet. The team therefore carries out a human check on a sample of 40 of these articles. Each team member reads the article first and records one of the four stances in an empty column, without seeing the automatic answer. The agreement between the team and the automatic coding will show how far the label-problem shares can be trusted. Until this check is done, the split between model errors and label problems is an estimate from one unchecked source and should be read as such.

## Check yourself

Q: Why can a model be "wrong" on an article even though the model is not at fault?
A: The label comes from the website, not from the article. A conspiracy site can publish an ordinary news item, so a model that reads only the text has nothing to find and the label does not describe the text.

Q: Why do we call the automatic stance coding unverified?
A: A second language model made the ticks and no person has compared them with their own reading yet. Until the team has done this, we do not know how often the automatic coding is wrong.

Q: How do I fill in `results/tables/stance_check_sample.csv`? Follow these steps.
A: (1) Open the file: it has the columns `doc_id`, `claude_stance` and an empty `human_stance`. (2) Take the first `doc_id` and read the whole article in the private data repository. (3) Only after reading it, look at `claude_stance`; do not let it decide for you. (4) Ask: what does the article do with conspiracy claims about 9/11? (5) Write exactly one of these four values in `human_stance`: `endorses`, `rejects_or_debunks`, `reports_neutrally`, `no_claim`. (6) If the article only mentions 9/11 in passing, write `no_claim`. (7) Repeat for all 40 rows, save, and never paste article text into the file or the repository.
