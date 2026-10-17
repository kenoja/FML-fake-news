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

- **Tool:** an AI coding agent (Claude Code, Sonnet 5.5) working task by task under `AGENTS.md`; Claude Haiku 5.5
  as the classifier under study; Claude Sonnet 5.5 for stance coding in the error analysis.
- **What it produced:** everything in `src/fml/` (data, cleaning, split, TF-IDF logistic regression, topic filter,
  feature groups, ablation, LLM classifier, comparison, error analysis, `fml run-all`), the tests, `results/`,
  the notebook sections 01-13 and 99, `reports/report.md`, the README update and this entry.
- **What automatic checks caught:** two exact duplicates and 63 near-duplicate pairs the plan did not expect;
  the off-topic share (538 of 3,146 documents) and its uneven effect on the test split; the model refusing
  `temperature` (rule vs reality, recorded, not hidden); duplicated cache lines from parallel interrupted runs;
  answers cut off by the 150-token output limit (not raised after seeing test results). `tests/test_report_numbers.py`
  compares every decimal and large integer in the report with `results/` and the config.
- **What the agent could not do or did not check:** it did not run `fml run-all` twice inside the report task
  (the harness runs it afterwards), so byte-identical `metrics.json` across two runs is checked by the harness,
  not claimed here. The LOCO licence could not be confirmed (page unreachable).
- **What humans must still check:**
  1. Fill `human_stance` in `results/tables/stance_check_sample.csv` (40 rows). Until then the H6 verdict is provisional.
  2. Confirm the licence of the LOCO texts at https://osf.io/snpcg/ and record it in `docs/decisions.md`.
  3. Review the open points in `docs/decisions.md` ("Needs a team decision"): temperature, output limit, near-duplicates.
  4. API key state: `LLM_CLASSIFIER_API_KEY` was set during the LLM task; `fml run-all` needs it only if the cache
     `results/llm_cache.jsonl` is missing entries.
  5. Spot-check several report numbers against `results/metrics.json`, and read the report once for article text
     or personal data. Members A-E add here what they checked.
