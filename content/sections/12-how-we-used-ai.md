---
title: How we used AI
part: Use of AI tools
hypothesis: An AI assistant can write most of the code quickly, but a person still has to check the numbers and the reasoning.
---
## Explained simply

**Why this section exists.** The assignment asks us to say which AI tools we used, what they produced, and what we checked or changed. Think of a calculator in a maths exam: it saves work, but you still decide what to calculate and whether the answer makes sense.

**Which AI tools.** Two kinds of AI appear in this project.

1. An AI assistant (Claude) helped us build the study: it drafted the project plan, wrote the code, produced the tables and figures, and drafted the text of this site and the report.
2. A language model (Claude Haiku 5.5) is part of the study itself. It is one of the four ways we sort articles, so it is the thing being tested, not a helper.

**What the AI did not decide.** The team decided the question, the use of full articles, the four ways of sorting, the feature groups, and the rule that the final test articles are scored only once.

**How output was checked.** Small test programs fail when a rule is broken, for example if the split of articles changes or if a number in the report is not found in the result files. The article counts were compared with the counts published for the corpus. Some checks need a person, such as whether an explanation is understandable. These are still open.

**What a test program cannot do.** It can confirm that a number matches a file. It cannot confirm that the number answers the right question. That judgement stays with people.

**Saving answers.** The language model costs money for every answer, so each answer is saved the first time. Running the study again reads the saved answers instead of asking again. The saved file holds only article ids and answers, never article text.

## The code

One short piece of code shows how saved answers stay trustworthy. The key joins the prompt wording, the model and the settings into one fingerprint. If any of these changes, the old answer is not reused.

<!-- cell: src/fml/llm.py#cache_key -->

## Report

**What the AI produced.** The AI assistant Claude drafted the project plan, wrote the analysis code and the tests, produced the tables and figures, and drafted the text of the course site and of this report. A separate language model, Claude Haiku 5.5, classified the articles as the fourth method under study. A second model, Claude Sonnet 5.5, coded how the articles in which the methods disagree treat conspiracy claims, for the error analysis.

**What the team checked or changed.** The team fixed the research question, full articles as input, the four methods, the feature groups, and the rule that each final model is scored on the test articles once. The article counts were compared with those published for the corpus. Tests were written that fail when a rule is broken, such as fitting on training articles only, a changed split, or a number in the report that is missing from the result files. Human checks are pending: reading the text, spot-checking numbers, and coding 40 articles by hand. None of them is recorded as done.

**What we learned about the work.** Tests confirm that numbers match files, not that they mean the right thing, so a person must still judge the reasoning. The stance codes produced by a language model need a human check before they are relied on. Saving the language model answers makes every figure reproducible without new cost.

## Check yourself

Q: Why is the language model in this study not described as a helper?
A: Because it is one of the four classifiers being tested. Its answers are results to be judged, not help with the work.

Q: Why is a test that says "every number matches a result file" not enough to trust the conclusions?
A: A number can match its file and still measure the wrong thing. A person must judge whether the conclusion makes sense.

Q: What would happen if the language model answers were not saved?
A: Every rerun would ask again, cost money, and could give slightly different answers, so the numbers might change.

Q: What would happen if a stance code from the language model were used without a human check?
A: Errors in the code would pass unseen into the error analysis, and we could not say how far to trust it.
