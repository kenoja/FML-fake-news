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

**How output was checked.** Three kinds of checks worked together.

- *Automatic checks.* Small programs that fail loudly when a rule is broken. For example, they fail if the split of articles changes, if private names appear in public text, or if a number in the report is not found in the result files.
- *Comparing with outside facts.* The article counts were compared with the counts published for the corpus, and every figure in the text was copied from the result files rather than typed.
- *Reading by people.* Some checks cannot be automatic, such as whether an explanation is understandable or whether a result makes sense. These are listed as open for the team in the AI log.

**What an automatic check cannot do.** It can confirm that a number matches a file. It cannot confirm that the number answers the right question. That judgement stays with people.

**Saving answers.** The language model costs money for every answer, so each answer is saved the first time. Running the study again reads the saved answers instead of asking again. The saved file holds only article ids and answers, never article text.

## The code

One short piece of code shows how saved answers stay trustworthy. The key joins the prompt wording, the model and the settings into one fingerprint. If any of these changes, the old answer is not reused.

<!-- cell: src/fml/llm.py#cache_key -->

## Report

**Use of AI tools.** We used an AI assistant (Claude, in the form of a coding agent) to draft the project plan, write the analysis code and tests, produce tables and figures, and draft the notebook text and this report. A separate language model, Claude Haiku 5.5, is the fourth classifier under study and a second model (Claude Sonnet 5.5) coded how disagreement articles treat conspiracy claims for the error analysis. The log of what was produced, checked and changed is kept in the repository.

The study design was fixed by the team and not by the assistant: the research question, full articles as input, the four ways of sorting, the feature groups, and the rule that each final model is scored on the test articles once. The assistant followed written rules in the repository and was bound by automatic checks, among them a lock on the test set, a frozen split, a privacy check on public text and a comparison of every number in the report with the result files.

We checked the output in three ways. First, the article counts were compared with those published for the corpus. Second, automatic tests were written to fail whenever a rule such as fitting on training articles only is broken. Third, the team reads the text for clarity and spot-checks numbers against the result files; these human checks are recorded in the log as the team's own entries.

We learned that written rules plus automatic checks keep assistant-written work reliable, that automatic checks verify consistency rather than meaning, and that the stance codes produced by a language model need a human check before they are relied on. The language model answers are cached so that every figure can be reproduced without new cost.

## Check yourself

Q: Why is the language model in this study not described as a helper?
A: Because it is one of the four classifiers being tested. Its answers are results to be judged, not help with the work.

Q: An automatic check confirms that every number in the report matches a result file. What does it not confirm?
A: It does not confirm that the number measures the right thing or that the conclusion drawn from it makes sense. A person must judge that.

Q: Why do we save every answer of the language model?
A: Each answer costs money and can vary, so saving it makes a rerun free and gives the same numbers again.
