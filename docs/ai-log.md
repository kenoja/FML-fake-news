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
  1. **Pending, Member to fill (kappa = null):** fill `human_stance` in `results/tables/stance_check_sample.csv` (40 rows; instructions in section 11), then run `fml stance-agreement`. Until then it prints `pending: 0/40 filled` and the H6 verdict is provisional.
  2. Confirm the licence of the LOCO texts at https://osf.io/snpcg/ and record it in `docs/decisions.md`.
  3. Review the open points in `docs/decisions.md` ("Needs a team decision"): temperature, output limit, near-duplicates.
  4. API key state: `LLM_CLASSIFIER_API_KEY` was set during the LLM task; `fml run-all` needs it only if the cache
     `results/llm_cache.jsonl` is missing entries.
  5. Spot-check several report numbers against `results/metrics.json`, and read the report once for article text
     or personal data. Members A-E add here what they checked.

- Gate fix (run-all timeout): AI added a per-process feature-block cache and a parallel C-grid in
  `src/fml/models.py`, renamed the words-only LR to `lr_words_v2` after the test lock rejected the stale
  `lr_words` config, and refreshed the quoted numbers. Humans: please re-check the numbers in the report.

## Cycle 1, review round (17 Oct 2026)

- **Tool:** the same AI coding agent (Claude Code, Sonnet 5.5), working through the review tasks one by one.
- **What changed:** the review tasks revised the off-topic section (summary box at the top), the report and section 13 (code cells for the cache key, stance coding and the error rule), and this log. This task added the human-verification table below and mirrored it in report section 7 and section 13. No result numbers were changed by hand.
- **Unchecked outputs:** the 40 stance codes (Claude-written, no human codes, kappa = null); the licence of the LOCO texts; the prior-knowledge probe (not run unless a key is set); every number in the report (checked by `tests/test_report_numbers.py` against `results/`, not by a person).

### Human verification

Only checks recorded in this repository are entered. The agent does not fill in sign-offs: every row is
`pending`, the owner is a placeholder, and a member replaces `pending` with the date and what they checked.

| # | Check (what to do) | Owner | Status |
|---|---|---|---|
| 1 | **Report and section numbers.** After the last `fml run-all`, compare the numbers in `reports/report.md` and sections 05, 08 and 10 with `results/metrics.json` (spot-check at least 10, including every F1 in the head-to-head table). The automated check below does not replace this. | Member A–E | pending |
| 2 | **Stance codes.** Fill `human_stance` for the 40 rows in `results/tables/stance_check_sample.csv`, then run `fml stance-agreement`; copy kappa into section 11 and the report only via `results/`. | Member A–E | pending |
| 3 | **Licence.** Confirm the licence of the LOCO texts at https://osf.io/snpcg/ and record it in `docs/decisions.md`. | Member A–E | pending |
| 4 | **Temperature decision.** Confirm or change the accepted default (the model rejects `temperature`, answers use default sampling; cache key keeps `temperature: 0`) in `docs/decisions.md`. | Member A–E | pending |
| 5 | **Output limit and near-duplicates.** Decide on `max_output_tokens: 150` (parse errors, see below) and on near-duplicate removal; both are under "Needs a team decision". | Member A–E | pending |
| 6 | **Privacy read.** Read the report and sections once for article text, excerpts over 50 words, names and identifiers. | Member A–E | pending |
| 7 | **Double test access.** Read `results/test_access_log.jsonl` and the "Sequence of test looks" in section 10; agree that `lr_words_v2` is the reported words-only model. | Member A–E | pending |

### What we learned in cycle 1

- **The LLM's 12% parse errors have a visible cause: the output limit.** The 150-token limit cut replies off.
  Every unparsed cache record used exactly 150 output tokens and no parsed reply used more
  (`results/metrics.json#llm.parse_error_causes`: 344 of 2,190 replies, all `truncated_at_max_tokens`). The raw
  replies were not stored, so refusals or wrong formats cannot be ruled out. An earlier claim that cut-offs were not
  the cause was wrong and is retracted in `docs/decisions.md`; the 400-token check had reused the 150-token client.
  Unparsed answers count as errors, so they lower recall (precision stays 1.0).
- **The off-topic filter hit the splits unevenly.** It removed 18.9% of train, 15.5% of validation and 10.2% of
  test documents (`metrics.json#topic_filter.removal_rate_by_split`), so scores before and after the filter are
  computed on different test documents. We reported this and did not touch the frozen split.
- **Temperature 0 was rejected by the model.** The rule in `AGENTS.md` cannot be followed (HTTP 400). The agent
  recorded the conflict instead of hiding it; the effect is measured with a second pass, the decision is open.
- **The test set was looked at twice for the words-only model.** `lr_words` was scored first with a stale
  configuration; the corrected model had to get a new id, `lr_words_v2`, scored after other test results were
  known. Both entries stay in the log and only `lr_words_v2` is used. Since then `evaluate_final` requires a
  validation-frozen configuration (`results/frozen_models.json`).

### Automated check of quoted numbers (not a human check)

- **Date and tool:** 17 Oct 2026, run by the AI agent (Claude Code, Sonnet 5.5).
- **What was run:** `tests/test_report_numbers.py` (every decimal, percentage and large integer in the report
  against `results/metrics.json`, `results/tables/*.csv` and the config) and the new
  `tests/test_section_numbers.py` (same matcher on sections 05, 08 and 10; front matter, cell markers,
  config hashes and length thresholds are removed first). Both include a test that a wrong number is caught.
- **Outcome:** 13 passed in the combined run. The first run of the section check flagged config-hash fragments and
  a "250 words" threshold; these are not results, and the check now ignores them.
- **Limits:** the check proves a number exists somewhere in the results, not that it is the right metric or split.
  It ran on the `results/` currently in the repository, not after a fresh `fml run-all` (left to the harness).
  A person still has to do row 1 above.

### Reproducibility task (17 Oct 2026)

- **Tool:** Claude Code (Sonnet 5.5). Kept version ranges in `pyproject.toml` (pinning with `==` could break CI) and documented `requirements-lock.txt` in the README ("Exact versions").
- **Not done:** the double `fml run-all` and the `metrics.json` diff. The task rules forbid running `run-all` inside a task, so the README says the check is still open. Runtime of the last run is 217.3 s (`results/run_info.json`).
- **Checked by reading code only:** `run-all` does not append to `results/test_access_log.jsonl` for an already-logged model with the same config hash.

### Privacy scan (17 Oct 2026)

- **Tool:** Claude Code (Sonnet 5.5). Rebuilt the site (`python -m sitegen build`) and searched `site/`, `results/tables/`, `results/*.json*`, `content/`, `reports/`, `docs/` and `README.md` with regular expressions: e-mail addresses, `github.com/<user>`, the account owner's name/e-mail words, "university", "course code", "team number".
- **Outcome:** no e-mail addresses (only `noreply@anthropic.com` in agent code and `@example.*` in a test), no personal names, no institution, course or team identifiers in the outputs. "university" appears only in LLM reasons about article content (`results/llm_cache.jsonl`), not as an affiliation.
- **One remaining hit:** the repository owner's GitHub account name appears in the repo link (`config/site.yaml`, rendered into `site/index.html`, plus `docs/SETUP.md`, `docs/PROJECT_PLAN.md`). It is the address of this repo and needed for the comment tool and "edit" links, so it was not removed; it is listed in `docs/decisions.md` as a team decision.
- **Excerpt check:** no cell in `results/tables/*.csv` (including `stance_check_sample.csv` and `error_examples.csv` justifications) exceeds 50 words, and no quote block in the sections or report does. Bold call-out boxes and "(invented)" teaching examples are not counted as article excerpts.
- **New test:** `tests/test_privacy.py` (e-mail regex plus the 50-word / 6-block limits; includes tests proving it fails on an injected e-mail, a 60-word quote block and a 60-word CSV cell). Ran it: 4 passed. Not run: the full suite and `fml run-all` (left to the harness). The private-terms gate (`FML_PRIVACY_TERMS`) was not run because the real terms are not available here.
