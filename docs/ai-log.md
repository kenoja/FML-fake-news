# How we used AI

The assignment asks us to document which AI tools we used, what for, and how we checked or adapted
their output. This log is the raw material for that part of the report. The agent appends an entry
after every cycle; team members add what they checked or changed by hand.

## Setup (16 Oct 2026)

- **Tool:** Claude (Opus 5.5) in a claude.ai session, working with the repo owner.
- **What it produced:** the project plan (`docs/PROJECT_PLAN.md`) including the literature research for
  the feature groups; the agent harness (`agent/`), the site generator (`sitegen/`), the feedback
  collection script and workflows; the 9/11 data subset and its checksums in the private data repo;
  the first requirements file and this repo's rules (`AGENTS.md`).
- **What was checked:** every product fact used in the design (pricing, authentication, cloud limits)
  against the official documentation; the data subset against the counts in the LOCO paper (3,146 = 939
  + 2,207, matching exactly); the full archive rebuilt from its parts and compared by checksum; the
  agent graph's pause-and-resume behaviour with a test; a nested Claude Code call inside a cloud session.
- **What the team decided (not the AI):** full articles, one cheap LLM asked the source question,
  feature groups (no emotion features), removing off-topic articles after the first run, public code with
  private data, running in Anthropic's cloud, pull requests per cycle.
- **What we learned:** about 17% of LOCO's 9/11 documents never mention 9/11 at all, in both classes.
  That only showed up once we looked at the data instead of trusting the seed label.


## Cycle 1 (17 Oct 2026)

- **Tool:** Claude Code (Sonnet 5.5) working task by task under `AGENTS.md`; Claude Haiku 5.5 as the classifier under study; Claude Sonnet 5.5 for stance coding in the error analysis.
- **What it produced:** the first version of `src/fml/`, the tests, `results/`, the notebook sections and the report.
- **What was checked:** `tests/test_report_numbers.py` and `tests/test_section_numbers.py` compare quoted numbers with `results/`. Cycle 1 ended with open points: the LOCO licence (OSF page unreachable), the 40 stance codes without human labels (kappa `null`), and the review of the LLM temperature and output-limit choices. Details are in `docs/decisions.md`.
- **Superseded:** the cycle 1 results and text were replaced in cycle 2 (new cleaning, new analyses, new final run). The task lists, reviewer tables and a retracted explanation of the LLM parse errors were removed from this log; nothing in them is relied on any more.

## Cycle 2 (17 Oct 2026)

- **Tool:** the same coding agent (Claude Code, Sonnet 5.5), task by task under `AGENTS.md`; Claude Haiku 5.5 is the classifier under study. The agent loop commits after each step.
- **Code rewrite:** the pipeline was reduced to the four reported analyses (word counts, TF-IDF, TF-IDF plus features, LLM) with beginner-style code; analyses no longer reported were deleted (all-documents run, drop-one ablation, coefficient tables, extra scores).
- **Cleaning:** one function, `clean.clean_documents()`: off-topic removal (3,146 to 2,608), then near-duplicate removal (2,608 to 2,554). The split file was not changed. Numbers: `results/metrics.json#cleaning`.
- **Final run:** settings were chosen on validation for all four analyses; the final models were then scored once each on test with new `final_*` ids (`results/test_access_log.jsonl`). The old `lr_words_v2` and `lr_extended` ids are no longer used for the comparison.
- **Report assembly:** every section has the four parts; `python -m sitegen build` assembles `reports/report.md` from the `## Report` parts and `content/references.md`. The glossary was rewritten (one entry per term, at most 80 words, everyday example). Section 12 was cut to a short factual summary.
- **What was checked (ran by the agent):** `tests/test_report_numbers.py` (every number in the report is found in `results/` or the config; length gate counts `len(text.split())` before References), `tests/test_section_numbers.py`, `tests/test_gates.py`, `tests/test_privacy.py`, and the section-format and explained-part word limits. These checks show that numbers match files, not that they mean the right thing.
- **Not recorded as done:** a two-run diff of `results/` after consecutive `fml run-all` runs is not recorded with an outcome anywhere in the repository (a task may not run `fml run-all`; the harness does). The last run time is in `results/run_info.json`. No claim of byte-identical reruns is made here.
- **Privacy:** the account name in the repo link (`config/site.yaml`) could not be removed by the agent (protected path); see "Needs a team decision" in `docs/decisions.md`.
- **What we learned:** automatic checks catch inconsistencies, not wrong reasoning; written rules plus gates keep agent work reliable; a glossary only helps if every technical word has an entry.

### Human checks (all pending; Members A-E fill in date and what was checked)

| # | Check | Status |
|---|---|---|
| 1 | Spot-check about 10 numbers in the report against `results/metrics.json` after the last `fml run-all` | pending |
| 2 | Fill `human_stance` for the 40 rows of `results/tables/stance_check_sample.csv`, then run `fml stance-agreement` | pending |
| 3 | Confirm the LOCO licence at the OSF page and record it in `docs/decisions.md` | pending |
| 4 | Read the report and sections once for article text and personal data (including the repo link) | pending |
| 5 | Decide the open points under "Needs a team decision" in `docs/decisions.md` | pending |
| 6 | Read the glossary and section 12 for plain-language understanding | pending |
