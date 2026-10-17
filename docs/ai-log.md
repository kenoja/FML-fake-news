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

- **Produced:** Claude Code (Sonnet 5.5) rewrote the pipeline to the four reported analyses with beginner-style code, added cleaning (off-topic, then near-duplicates), scored the final models once each on test, and drafted the notebook sections, glossary and report. Claude Haiku 5.5 is the classifier under study, not a helper.
- **Checked:** the automated tests (numbers in text match `results/`, section format and word limits, privacy, frozen split, locked test set) and two identical `fml run-all` runs (`docs/reproducibility.md`). These show that numbers match files, not that they mean the right thing. Still pending for humans: the checks in the table below.
- **Learned:** tests catch inconsistencies, not wrong reasoning; a glossary only helps if every technical word has an entry; the account name in the repo link and the corpus licence stay open (`docs/decisions.md`, "Needs a team decision").

### Human checks (all pending; Members A-E fill in date and what was checked)

| # | Check | Status |
|---|---|---|
| 1 | Spot-check about 10 numbers in the report against `results/metrics.json` after the last `fml run-all` | pending |
| 2 | Fill `human_stance` for the 40 rows of `results/tables/stance_check_sample.csv` | pending |
| 3 | Confirm the LOCO licence at the OSF page and record it in `docs/decisions.md` | pending |
| 4 | Read the report and sections once for article text and personal data (including the repo link) | pending |
| 5 | Decide the open points under "Needs a team decision" in `docs/decisions.md` | pending |
| 6 | Read the glossary and section 12 for plain-language understanding | pending |
