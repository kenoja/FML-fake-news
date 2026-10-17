# How we used AI

The assignment asks us to document which AI tools we used, what for, and how we checked or adapted
their output. This log is the raw material for that part of the report. An entry is added
after every cycle; team members add what they checked or changed by hand.

## Setup (16 Oct 2026)

- **Tool:** Claude (Opus 5.5) in a claude.ai session, working with the repo owner.
- **What it produced:** the project plan (`docs/PROJECT_PLAN.md`) including the literature research for
  the feature groups; the cycle runner, the site generator (`sitegen/`), the feedback
  collection script and workflows; the 9/11 data subset and its checksums in the private data repo;
  the first requirements file and this repo's rules (`AGENTS.md`).
- **What was checked:** every product fact used in the design (pricing, authentication, cloud limits)
  against the official documentation; the data subset against the counts in the LOCO paper (3,146 = 939
  + 2,207, matching exactly); the full archive rebuilt from its parts and compared by checksum; the
  runner's pause-and-resume behaviour with a test; a nested Claude Code call inside a cloud session.
- **What the team decided (not the AI):** full articles, one cheap LLM asked the source question,
  feature groups (no emotion features), removing off-topic articles after the first run, public code with
  private data, running in Anthropic's cloud, pull requests per cycle.
- **What we learned:** about 17% of LOCO's 9/11 documents never mention 9/11 at all, in both classes.
  That only showed up once we looked at the data instead of trusting the seed label.


## Cycle 1 (17 Oct 2026)

- **Tool:** Claude Code (Sonnet 5.5) working task by task under the repository rules; Claude Haiku 5.5 as the classifier under study; Claude Sonnet 5.5 for stance coding in the error analysis.
- **What it produced:** the first version of `src/fml/`, the tests, `results/`, the notebook sections and the report.
- **What was checked:** `tests/test_report_numbers.py` and `tests/test_section_numbers.py` compare quoted numbers with `results/`. Cycle 1 ended with open points: the LOCO licence (OSF page unreachable), the 40 stance codes without human labels (kappa `null`), and the review of the LLM temperature and output-limit choices. Details are in `docs/decisions.md`.

## Cycle 2 (17 Oct 2026)

- **Tool:** Claude Code (Sonnet 5.5), task by task under the repository rules; Claude Haiku 5.5 is the classifier under study.
- **Code rewrite:** the pipeline was reduced to the four reported analyses (word counts, TF-IDF, TF-IDF plus features, LLM) with beginner-style code; analyses no longer reported were deleted.
- **Cleaning:** one function, `clean.clean_documents()`: off-topic removal (3,146 to 2,608), then near-duplicate removal (2,608 to 2,554). The split file was not changed. Numbers: `results/metrics.json#cleaning`.
- **Final run:** settings were chosen on validation for all four analyses; the final models were then scored once each on test with new `final_*` ids (`results/test_access_log.jsonl`).
- **Report assembly:** every section has the four parts; the report is assembled from the `## Report` parts and `content/references.md`. The glossary has one entry per term. Section 12 was rewritten as three plain parts: what the AI produced, what the team checked, what we learned.
- **What was checked:** the tests for report numbers, section numbers, gates and privacy, and the section-format and word limits. They show that numbers match files, not that they mean the right thing.
- **Not recorded as done:** a two-run comparison of `results/` after consecutive full runs has no recorded outcome. The last run time is in `results/run_info.json`.
- **Privacy:** the account name in the repo link (`config/site.yaml`) is still open; see "Needs a team decision" in `docs/decisions.md`.
- **What we learned:** tests catch inconsistencies, not wrong reasoning; a glossary only helps if every technical word has an entry.
- **Task T9 (this entry):** section 12 and this log were reduced to facts. Humans still must verify the items in the table below, in particular the report-number spot-check, the 40 hand-coded stance rows and the plain-language reading of section 12.

### Human checks (all pending; Members A-E fill in date and what was checked)

| # | Check | Status |
|---|---|---|
| 1 | Spot-check about 10 numbers in the report against `results/metrics.json` after the last `fml run-all` | pending |
| 2 | Fill `human_stance` for the 40 rows of `results/tables/stance_check_sample.csv`, then run `fml stance-agreement` | pending |
| 3 | Confirm the LOCO licence at the OSF page and record it in `docs/decisions.md` | pending |
| 4 | Read the report and sections once for article text and personal data (including the repo link) | pending |
| 5 | Decide the open points under "Needs a team decision" in `docs/decisions.md` | pending |
| 6 | Read the glossary and section 12 for plain-language understanding | pending |
