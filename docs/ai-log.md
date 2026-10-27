# How we used AI

The assignment asks which AI tools we used, what for, and how we checked or changed their output. This log is the
source for that part of the report: one block per cycle, facts only. The agent's own run records are in
`docs/archive/agent-runs.md`; checks that people still have to do are in `docs/human-todo.md`.

## Setup (16 Oct 2026)

**Produced.** A Claude session working with the repository owner drafted the project plan (including the literature
search behind the feature groups), the agent loop, the site generator, the feedback collection, the data subset in
the private repository and the first requirements file.

**Checked.** The data subset against the LOCO paper (3,146 = 939 + 2,207 documents); the full corpus archive by
checksum; the loop's pause-and-resume behaviour with automated tests.

**Changed by the team.** Full articles instead of headlines; one cheap language model asked the source question; the
feature groups without emotion scores; public code with private data; every cycle ends in a pull request that the
owner reviews.

**Learned.** About 17% of the 9/11 documents never mention 9/11. Looking at the data, not only at the corpus labels,
is necessary before any model.

## Cycle 1 (17 Oct 2026)

**Produced.** The agent loop (Claude Code with Claude Sonnet 5.5 as planner, implementer and reviewer) wrote the
first analysis, tests, results, site sections and report.

**Checked.** Automated tests compare every number in the text with the result files; a separate reviewer session
scored the work against a checklist.

**Changed by the team.** The owner rejected the first version as too technical for the team and restructured the
site into four parts per section (plain explanation, code, report text, questions).

**Learned.** Tests that check numbers against files do not show whether the design suits the reader or the question.

## Cycle 2 (17 Oct 2026)

**Produced.** Four analyses with beginner-style code, cleaning before any model, one final test run, a glossary and a
report assembled from the sections.

**Checked.** Automated tests for numbers, section format, word limits, privacy, the frozen split and the locked test
set; a reviewer session, which caught a claim that the language model beat every model (corrected to what the table
shows: high precision, low recall) and a figure without a source (replaced by one read from the results).

**Changed by the team.** The owner set the four analyses (word counts, TF-IDF, TF-IDF with features, LLM), the
5,000-word cap for the report text, one honest sentence about earlier use of the test set, and that the hand check is
done by the team.

**Learned.** A glossary helps only if every technical word has an entry, so the site now links terms automatically.

## Cycle 3 (25 to 26 Oct 2026)

**Produced.** Merged results and error analysis, numbered hypotheses, a methods overview, a per-website table, a
baseline row and a rerun of the three logistic regressions with identical settings.

**Checked.** The owner and a second session looked up each new reference on its publisher or DOI page (26 Oct 2026), ran the whole analysis twice with
identical results and the full test suite with no test skipped (`docs/reproducibility.md`), and corrected three
sentences whose numbers did not match the result files.

**Changed by the team.** Two members commented on 11 of the 13 sections (18 comments), which became the cycle's
requirements. After reviewing the method, the owner required the planned feature model instead of the agent's
best-group selection, identical settings and text for all methods, and an explanation of the drop from validation to
test, which led to the finding about the large test website.

**Learned.** The agent follows instructions literally: it chose the best feature group by a threshold until the team
asked for the planned comparison. Evidence from runs made in code must be handed to a reviewer that cannot run code,
and the tests that guard the agent's own process must be protected from that agent.

## Hand check (25 to 27 Oct 2026)

**Produced.** A Claude session built the offline hand-check file (articles shown without website or automatic answer)
and the code that compares the members' answers with each other and with the automatic stance coder.

**Checked.** Every answer file against the assignment of articles to members; the comparison code with tests that fail
when an answer is counted twice or a row does not fit the sheet.

**Changed by the team.** 16 articles per member instead of 40, every article meant for two members; the five members
did the coding themselves.

**Learned.** Members often chose different stances but mostly agreed, with each other and with the automatic coder,
on whether an article endorses a claim, the only question the error rule uses.
