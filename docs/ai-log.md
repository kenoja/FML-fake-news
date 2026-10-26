# How we used AI

The assignment asks us to document which AI tools we used, what for, and how we checked or adapted
their output. This log is the raw material for that part of the report. One short block per cycle;
the checks that people still have to do are in `docs/human-todo.md`.

## Setup (16 Oct 2026)

### Produced
Claude in a claude.ai session, working with the repo owner (the model of that session is not recorded in the repo), made the project plan with the literature research for the feature groups, the cycle runner, the site generator, the feedback script, the 9/11 data subset with checksums in the private data repo, the first requirements file and the repo rules. The agent loop from cycle 1 on uses Claude Sonnet 5.5 for planner, implementer and reviewer (`config/agent.yaml`).

### Checked
Product facts used in the design (pricing, authentication, cloud limits) against the official documentation; the data subset against the LOCO paper (3,146 = 939 + 2,207, exact match); the full archive rebuilt from its parts and compared by checksum; the runner's pause-and-resume behaviour with a test.

### Learned
About 17% of LOCO's 9/11 documents never mention 9/11, in both classes; this was found by looking at the data, not at the seed label. The team decided: full articles, one cheap LLM asked the source question, the feature groups (no emotion features), removing off-topic articles, public code with private data, pull requests per cycle.

## Cycle 1 (17 Oct 2026)

### Produced
Claude Code (Sonnet 5.5) wrote the first version of `src/fml/`, the tests, `results/`, the sections and the report. Claude Haiku 5.5 was the classifier under study; Claude Sonnet 5.5 coded stance in the error analysis.

### Checked
Tests compare quoted numbers with `results/`. Open at the end: the corpus licence, the 40 stance codes without human labels, and the review of the LLM temperature and output-limit choices (`docs/decisions.md`).

### Learned
Tests that check numbers against files do not show whether the analysis is well designed; the design was reworked in cycle 2.

## Cycle 2 (17 Oct 2026)

### Produced
Claude Code (Sonnet 5.5) rewrote the pipeline to the four reported analyses with beginner-style code, added cleaning (off-topic, then near-duplicates), scored the final models once each on test, and drafted the sections, glossary and report.

### Checked
Automated tests (numbers match `results/`, section format and word limits, privacy, frozen split, locked test set) and two identical `fml run-all` runs (`docs/reproducibility.md`). A reviewer in a fresh session scored the result against a checklist; the implementer then fixed what it found. Output that was changed or rejected:
- A draft sentence said the LLM beat every model. The test table did not support it (high precision, low recall), so the claim was corrected to what the table shows.
- A draft claim about 200,000 columns had no source in the repo or in `results/`. It was replaced by a figure read from the results.
- Hypothesis H1 was relabelled and reworded as a plain guess (TF-IDF against word counts); the lettered sub-hypotheses were later merged.
- Emotion scores in the AI's first plan were removed by the team, because emotional language is not the same as conspiracy language.

### Learned
Automated checks find inconsistencies, not wrong reasoning. A glossary helps only if every technical word has an entry, so the site links them automatically.

## Cycle 3 (2026-10-25)

### Produced
Claude Code (Sonnet 5.5) merged results and error analysis into one section, renamed sections, used one identical set of logistic-regression settings for all three analyses, and scored new final models `final2_*` once on test. It added a per-website recall table, a trivial-baseline row and a methods-at-a-glance table, and rewrote limitations, conclusion and this log. Haiku 5.5 answers were reused from the cache; Sonnet 5.5 coded the new disagreement articles.

### Checked
Concrete checks, all automated or recorded in `docs/decisions.md` (none of them is a human action):
- `tests/test_report_numbers.py` compares every number in the report with `results/metrics.json` and the CSV tables, and every three-decimal number in the sections with `results/`; a second test shows the checker fails when a number is changed.
- The same file compares the numbers quoted in prose with the ablation table.
- `tests/test_gates.py` counts the words of the report (3,000-5,000 before References) and of the plain-language parts (450 words).
- `results/test_access_log.jsonl` was compared with the model ids: each `final2_*` id appears once (`tests/test_test_lock.py`).
- Sentences were rewritten where a claim went beyond the tables (for example the introduction keeps only what Miani et al. 2022 and Fong et al. 2021 support).
- Rejected suggestion: the claim that platforms and fact-checkers rely on automatic filters was dropped, because no published source could be confirmed.
- Rejected suggestion: candidate classic references and 9/11 background sources were not added from memory; they stay open for a member to verify.
The checks that need people are listed in `docs/human-todo.md` (placeholders; nothing there is done yet).

### Altered or rejected by members
- Member A (repo owner) and Member B commented on 11 of the 13 sections of the cycle-2 site; their 18 comments became
  `requirements/cycle-03.md` (for example: merge results and error analysis, write the hypotheses out in full, keep
  all three prompts, explain the code snippet by snippet).
- After reviewing the method, Member A added decisions D10-D13: the feature model is the planned full combination
  (the AI had kept only the best group by a threshold), all three logistic regressions share one setting, all
  methods read the same text, and a per-website table must explain the drop from validation to test.

### Learned
- A number can match `results/metrics.json` and still stand beside a claim the table does not support; the platform-filter sentence was only caught by asking for a source.
- Counting `final_*` and `final2_*` ids in `results/test_access_log.jsonl` showed that four earlier-configuration models had been scored on test, so `09-limitations` now says so instead of claiming a single scoring.
- A feature count (47 hand-made words in four lists) had to be read from the results files, not written from memory.

## Cycle 3 - owner review and polish (2026-10-26)

### Produced
A separate Claude session, working with the repo owner, edited the cycle-3 output directly: the verdict rule (a
hypothesis counts as supported when McNemar's test gives p < 0.05), the introduction with verified background
sources, the explanation of why all three feature groups together score lower (from the comparison table), a
readable rewrite of the repetition measure (same values), shorter report text, and the harness change that runs
`fml run-all` twice and gives the reviewer the test and pipeline results.

### Checked
Seven new references were looked up online before they were added; the LOCO paper's own label check (85% / 92%)
was checked against the paper. `fml run-all` was run twice in the pinned environment with identical results, and
the full test suite passed with no test skipped (`docs/reproducibility.md`). The number tests caught three edits
that did not match `results/`; those sentences were rewritten.

### Learned
The agent's reviewer could not run commands, so it scored reproducibility low for lack of evidence; the harness now
passes it the results of runs made in code. The agent had also deleted the tests of its own harness; those tests
are now protected.
