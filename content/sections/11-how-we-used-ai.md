---
title: How we used AI
part: Use of AI tools
hypothesis: An AI assistant can write most of the code quickly, but people still have to set the goals, read the results and judge the reasoning.
---
## Explained simply

**Why this section exists.** The assignment asks which AI tools we used, what they produced, and what we checked. Like a calculator in a maths exam, AI saves work, but you still decide what to calculate and whether the answer makes sense.

**The setup.** We worked with Claude Code, a programme that lets an AI assistant read and change the files of a project. A small loop of three roles drives it:
1. a **planner** reads our requirements and splits them into small tasks,
2. an **implementer** does one task at a time, writes the code and the text, and runs the tests,
3. a **reviewer**, started fresh so that it has not seen the work being made, scores the result against a checklist and lists what is still wrong.

The loop repeats until the reviewer is satisfied or a round limit is reached. This website is generated from the project files, so the code you see is the code that ran.

**Several rounds.** After each cycle the team read the website and wrote feedback under each section; the feedback became the next cycle's requirements.
**What the AI produced.** The assistant (Claude) wrote the plan, the code, the tests, the tables and figures, and drafted the text of this site and the report. Separately, Claude Haiku 5.5 is one of the four ways we sort articles: it is being tested, not a helper. Claude Sonnet 5.5 read the disputed articles for the error analysis.

**What the team did.** People set the question, the four ways of sorting, the feature groups and the rule that final models are scored on test once, and wrote the feedback. They removed emotion scores from the first plan, because emotional language is not the same as conspiracy language. They compared the article counts with those published for the corpus.

**How output was checked.** The agent ran small test programs that fail when a rule is broken, for example if the split changes or a number in the report is missing from the result files. Other checks need a person, such as whether the automatic stance coding is right: the team's hand check of 40 articles is planned for the next cycle.

**What we learned.** A test can confirm that a number matches a file, but not that the number answers the right question; that judgement stays with people. Saving the language model's answers makes every figure repeatable without new cost.

**Saving answers.** The language model costs money for every answer, so each answer is saved once and reused. The saved file holds only article ids and answers, never article text.

## The code

One short piece of code shows how saved answers stay trustworthy. The key joins the prompt wording, the model and the settings into one fingerprint. If any of these changes, the old answer is not reused.

<!-- cell: src/fml/llm.py#cache_key -->

## Report

**Setup.** The analysis was produced with Claude Code, an AI coding assistant, in three Claude Sonnet 5.5 roles: a planner splits the team's requirements into tasks, an implementer writes code and text and runs the tests, and a reviewer in a fresh session scores the result against a checklist.

**Cycles.** Work ran in three cycles. After each cycle the team read the website and wrote feedback, which became the next cycle's requirements. A log in the repository records what the AI produced and what was checked.

**Checks.** The agent checked every number in the report against the result files with automated tests (`test_report_numbers`); further tests cover length limits, the split, training-only fitting and the locked test set. Members' own checks are recorded in the log only where they entered them; so far none are recorded. The team's hand check of 40 articles for the automatic stance coding is planned for the next cycle. Claude Haiku 5.5 is one of the four classifiers under study, not a helper.

## Check yourself

Q: Why is the language model in this study not described as a helper?
A: Because it is one of the four classifiers being tested. Its answers are results to be judged, not help with the work.

Q: Why is a test that says "every number matches a result file" not enough to trust the conclusions?
A: A number can match its file and still measure the wrong thing. A person must judge whether the conclusion makes sense.

Q: What would happen if the language model answers were not saved?
A: Every rerun would ask again, cost money, and could give slightly different answers, so the numbers might change.

Q: Why does a reviewer that starts in a fresh session help?
A: It has not seen the work being made, so it judges the result as a newcomer would, instead of defending choices it already knows.

Q: What would happen if a stance code from the language model were used without a human check?
A: Errors in the code would pass unseen into the error analysis, and we could not say how far to trust it.
