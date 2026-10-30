# How we used AI

The assignment asks which AI tools we used, what for, and how we checked their output. This log is the source for that
part of the report: one block per cycle, facts only. The agent's own run records are in
`docs/archive/agent-runs.md`; checks that people still have to do are in `docs/human-todo.md`.

## Cycle 1

**Produced.** Claude Code with Claude Sonnet 5.5 drafted the project plan, the agent loop, the site generator and the
first analysis, tests, results, site sections and report text.

**Checked.** Number tests tie every figure in the prose to the results files. The data subset was compared with the
LOCO paper (3,146 = 939 + 2,207 documents) and the corpus archive by checksum. The owner read the first site and
rejected it as too technical.

**Learned.** About 17% of the 9/11 documents never mention 9/11, so the data must be looked at before any model.
Tests that check numbers against files do not show whether the design suits the reader.

## Cycle 2

**Produced.** Four analyses with beginner-style code, cleaning before any model, one final test run, a glossary and a
report assembled from the sections.

**Checked.** Number tests, section-format and word-limit gates, privacy tests (excerpts of at most 50 words), the frozen
split and the locked test set (`tests/test_test_lock.py`). A reviewer session scored the work against a checklist.

**Learned.** A glossary helps only if every technical word has an entry, so the site links terms automatically.

## Cycle 3

**Produced.** Merged results and error analysis, numbered hypotheses, a methods overview, a per-website table, a
baseline row and a rerun of the three logistic regressions with identical settings.

**Checked.** Each new reference was looked up on its publisher or DOI page. The whole analysis was run twice with
identical results and the full test suite ran with no test skipped (`docs/reproducibility.md`). Three sentences whose
numbers did not match the results files were corrected. Two members commented on the sections; their comments became
the cycle's requirements.

**Learned.** Written plans and explicit requirements steer the assistant. The tests that guard the agent's own process
must be protected from that agent.

## Cycle 4

**Produced.** Claude Code with Claude Sonnet 5.5 revised code, tests, sections and report against the requirements;
Claude Haiku 5.5 remained the classifier under test; Claude Sonnet 5.5 coded article stance. A Claude session also built
the offline hand-check file and the code that compares the members' answers.

**Checked.** The report word count against its gate (at most 4,900 words). Hashes of `results/metrics.json` and the
tables before and after code edits, to show that the results did not change. The five members did the hand check
themselves, 16 articles each, every article read by two members; the comparison code has tests that fail when an answer
is counted twice. No new models, statistics or API calls were added.

**Learned.** Human judgement is noisy too: members often chose different stances but mostly agreed on whether an
article endorses a claim, the only question the error rule uses.

## Cycle 5: final report (30 Oct 2026)

**Produced.** A Claude session working with the project owner restructured the report into one narrative with the
same pattern in every chapter, added a validation-only check of the n-gram threshold, and built the LaTeX version.

**Checked.** All tests and gates, two identical runs of the analysis, and a fresh reviewer session.

**Changed by the team.** The owner set the narrative (Website 4 as the main cause of the test drop, the split by
website as standard practice, the gap between validation and test as the main lesson) and kept the n-gram threshold
of 3 instead of a new test run.

**Learned.** An honest test can make the headline weaker and the study more useful.

