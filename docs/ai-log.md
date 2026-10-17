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

| # | Check (what to do) | How to check | Owner | Status |
|---|---|---|---|---|
| 1 | **Report and section numbers.** Spot-check the 10 numbers below (and every F1 in the head-to-head table) after the last `fml run-all`. The automated check does not replace this. | Protocol below: open `results/metrics.json`, follow the path, compare with the number in the report or section, tick the box. | Member A–E | pending |
| 2 | **Stance codes.** Fill `human_stance` for the 40 rows of `results/tables/stance_check_sample.csv`. | Read each article in the private data repo before looking at `claude_stance`; run `fml stance-agreement` (prints `pending: 0/40 filled` until done); copy kappa into section 11 and the report only via `results/`. | Member A–E | pending |
| 3 | **Licence.** Confirm the licence of the LOCO texts and record it. | Open https://osf.io/snpcg/ in a browser, copy the licence text word for word into the fill-in line in `docs/decisions.md`, replace "unconfirmed" in report §2 and section 02. | Member A–E | pending |
| 4 | **Temperature decision.** Confirm or change the accepted default (default sampling; cache key keeps `temperature: 0`). | Read the entry "LLM temperature 0 cannot be set" in `docs/decisions.md`; fill its `Confirmation:` line with name and date. | Member A–E | pending |
| 5 | **Output limit and near-duplicates.** Decide on `max_output_tokens: 150` and on near-duplicate removal. | Read both entries under "Needs a team decision" in `docs/decisions.md`; fill the `Confirmation:` line / confirm in the near-duplicate entry. | Member A–E | pending |
| 6 | **Privacy read.** Read the report and sections once for article text, excerpts over 50 words, names and identifiers. | Read `reports/report.md` and `content/sections/*.md`; flag anything that is article text or identifies a person or institution. | Member A–E | pending |
| 7 | **Double test access.** Agree that `lr_words_v2` is the reported words-only model. | Read `results/test_access_log.jsonl` (four entries) and "Sequence of test looks" in section 10. | Member A–E | pending |

Outstanding human checks: all seven rows above (none done). This is what report §7 and section 13 refer to.

#### 10-number spot-check protocol (row 1)

Values on 17 Oct 2026 `results/`; if a later `fml run-all` changed them, compare against the new file. Tick when the number in the report/section equals the number at the path. Paths are keys in `results/metrics.json`; a number in a list is addressed by its index.

| # | Number (current value) | Path in `metrics.json` | Likely place (search for the number) | Ticked by / date |
|---|---|---|---|---|
| 1 | Documents in the corpus (3,146) | `data.documents` | report §2, section 02 | pending (Member __, date __) |
| 2 | Documents after the topic filter (2,608) | `topic_filter.documents_after` | report §3, section 06 | pending (Member __, date __) |
| 3 | Test removal rate of the filter (0.1019) | `topic_filter.removal_rate_by_split.test` | section 06, report §3 | pending (Member __, date __) |
| 4 | Words-only LR validation F1, all documents (0.926) | `baseline_all_docs.validation.f1.estimate` | section 05 | pending (Member __, date __) |
| 5 | Words-only LR validation F1, filtered (0.914) | `baseline.validation.f1.estimate` | section 06, report §4 | pending (Member __, date __) |
| 6 | Test F1 of `lr_words_v2` (0.5848) | `comparison.table.1.f1` | section 10, report §4 | pending (Member __, date __) |
| 7 | Test F1 of `lr_extended` (0.5829) | `comparison.table.2.f1` | section 08, 10, report §4 | pending (Member __, date __) |
| 8 | Test F1 of `llm_haiku` (0.6842) | `comparison.table.3.f1` | section 10, report §4 | pending (Member __, date __) |
| 9 | McNemar p, best LR vs LLM (0.0004) | `comparison.mcnemar.lr_best_vs_llm.mcnemar_p` | section 10, report §4 | pending (Member __, date __) |
| 10 | LLM unparsed replies (344 of 2,190) | `llm.parse_error_causes.parse_errors` (and `.replies`) | section 09, report §5 | pending (Member __, date __) |

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

### Final consistency task (17 Oct 2026)

- **Tool:** Claude Code (Sonnet 5.5).
- **Word count (body): 4,903** words of `reports/report.md` before the References heading (whitespace-separated, tables and markup included). It was 5,524 before trimming; the cuts were repeated detail, not results.
- **Glossary:** added negation-aware count, Cohen's kappa, parse error and cluster bootstrap to `content/sections/99-glossary.md`.
- **Not run here:** `fml run-all`, the full test suite and `python -m sitegen`. The task rules forbid running `run-all` and the whole suite inside a task; the harness runs them afterwards. Skips (`needs_data`) therefore cannot be reported by this task.

### Cycle 1, round 3: human-check status (T11, 17 Oct 2026)

- **Tool:** Claude Code (Sonnet 5.5).
- **What changed:** the human-verification table now has a "How to check" column and the 10-number spot-check protocol above; `docs/decisions.md` got `Confirmation: pending` lines for the temperature and 150-token decisions; section 13, report §7 and the compliance statement mirror the status.
- **Not done, on purpose:** no human check was performed or signed off. All seven rows and all ten spot-check ticks are `pending`; owners are the placeholder "Member A–E". The 10 current values were read from `results/metrics.json` by the agent only to fill in the list, which is not a check.
- **Not run:** `fml run-all` and the full suite (left to the harness).

## Cycle 1 - agent run (2026-10-17)

**Verdict:** STOPPED BELOW 10/10 (round limit reached) after 4 review round(s).

**Requirements:** `requirements/cycle-01.md`

### What the AI produced (tasks completed)
- T1: Scaffold, decisions log, licence check, research-questions section (plan)
- T2: Cleaning, length distribution, frozen website-grouped split (plan)
- T3: Evaluation toolkit and first words-only run on ALL documents (plan)
- T4: Off-topic filter, before/after numbers and filtered words-only baseline (plan)
- T5: Feature groups: n-grams, epistemic markers, style (plan)
- T6: Ablation of feature groups and final LR test scoring (plan)
- T7: LLM classifier module with cache and fake client (plan)
- T8: Run the LLM: validation prompt choice, one test run, stability pass (plan)
- T9: Head-to-head comparison and statistics (plan)
- T10: Error analysis with stance coding and human-check sample (plan)
- T11: Remaining sections: limitations, AI use, glossary, overview polish; section audit (plan)
- T12: Report, AI-use log, README and full reproduction check (plan)
- gate-fix-1: Fix failed hard gates (gate)
- T1: Deterministic tables + byte-identical rerun test (review)
- T2: Pin dependencies, lock file, runtime (review)
- T3: Test-access sequence limitation and freeze rule (review)
- T4: Near-duplicate sensitivity analysis (review)
- T5: LLM prior-knowledge probe and unparsed column (review)
- T6: Label-problem share sensitivity range (review)
- T7: Human check hook and privacy gate for justifications (review)
- T8: Resolve open design choices with measured effect (review)
- T9: Licence recheck and hypotheses traceability (review)
- T10: Report literature, feature reasons, trim (review)
- T11: Teaching polish for sections 06, 11, 13 (review)
- T12: AI log, human-check records, final run (review)
- T1: Hypothesis traceability via git history (review)
- T2: Record open human items and temperature/licence/near-duplicate decisions (review)
- T3: Investigate LLM parse errors and store raw-reply sample (review)
- T4: State prior-knowledge probe as untested (review)
- T5: Quantify epistemic lexicon coverage and negation effect (validation only) (review)
- T6: Real excerpts and honest stance-check status in section 11 (review)
- T7: Make double test access and site-level CIs prominent (review)
- T8: Teaching: summary boxes in sections 06, 07, 11 (review)
- T9: Reproducibility: pin versions, double run, README note (review)
- T10: Privacy scan of all outputs plus excerpt-length check (review)
- T11: AI-log: what we learned, human verification table, report §7 (review)
- T12: Final consistency: run-all, report length, tests, site (review)
- gate-fix-1: Fix failed hard gates (gate)
- T1: Remove GitHub owner names and verify excerpt limits (review)
- T2: Human stance verification: tooling and honest blocked status (review)
- T3: Mark label-problem shares as provisional everywhere (review)
- T4: Head-to-head transparency: second test look and sensitivity (review)
- T5: Reproducibility: double run and version pinning (review)
- T6: Licence recheck of LOCO on OSF (review)
- T7: Hypotheses traceability and H5 gap (review)
- T8: Epistemic lexicon justification (review)
- T9: Plain-language summaries for long sections (review)
- T10: Trim report below 5,000 words (review)
- T11: Human-check rows and open team decisions: honest status (review)
- T12: Final consistency run (review)
- gate-fix-1: Fix failed hard gates (gate)

### What was checked
Hard gates still failing:
- none

| criterion | score | evidence |
|---|---|---|
| ai_use | 7 | docs/ai-log.md (149 lines), section 13 (1,507 words) and report §7 describe what the AI produced and what the gates changed. The ai-log human-check rows are sti |
| approach | 8 | Report §3 and sections 07/09/12 walk through the pipeline, including the 'not done' list (no fine-tuning, no emotion features). Choices and conflicts are logged |
| errors | 5 | Disagreement sets and 80 Claude-coded stances exist, but the human verification is not done: all 40 human_stance cells are blank, kappa is null, and H6 is expli |
| evaluation | 7 | One frozen website split; split table shows 0 websites in more than one split and 7/7/25 conspiracy websites; document and website bootstrap intervals; McNemar  |
| features | 8 | Report §5.3 and section 08 give a per-group ablation with website-bootstrap intervals; all include 0 and the verdict is an honest 'no clear gain'. Each group is |
| feedback_compliance | 6 | Requirements 1–6 are addressed, but requirement 2 is only partly met (the human verification is blank) and requirement 7 (licence) is blocked. Requirement 1 ask |
| interpretation | 8 | Report §6 and section 12 cover label-vs-content, few test websites, the single dominant site, off-topic filtering and LLM prior knowledge. The label-problem sha |
| privacy | 9 | Grep of content, reports, docs, results/tables and site for e-mail, GitHub URLs and the user name found nothing. Excerpts use doc_ids and scrubbed Claude justif |
| question | 8 | 01-research-questions.md (947 words) and report §1 define RQ1–RQ3 and H1–H6; report ties each finding to an RQ/H. Not every section was individually checked for |
| report | 8 | report.md has the seven required parts plus References. It has 5,051 words in total. The count before References was not obtained because that command was block |
| reproducibility | 7 | requirements-lock.txt, pyproject.toml, README (81 lines), seeds in config and results/run_info.json exist. decisions.md line 303 says the double-run reproducibi |
| resources | 7 | Data counts match metrics.json (3,146 docs; 939/2,207; 39/79 websites; 2,608 after filter). Licence is still unconfirmed: decisions.md lines 21-31 and 327-344 s |
| results | 8 | Report numbers I checked match metrics.json: split counts, 0.926/0.962/0.893 for the first run, 0.8768 Fisher p, 0.9504 LLM agreement. Figures and tables exist. |
| teaching | 7 | Section files have code cells (1–8 each, none in 00/99) and outputs, in the required format. Open questions Q1–Q6 are answered in their sections per requirement |

### Open reviewer issues
- [errors] Human stance verification has not been done: 40 blank human_stance cells, kappa null, H6 provisional, label-problem share spans 0.286–0.692. (results/tables/stance_check_sample.csv; report §5.6; section 11)
- [resources] LOCO licence is still unconfirmed because the OSF page was unreachable on three attempts. (docs/decisions.md lines 21-31, 327-344; report §2; section 02)
- [feedback_compliance] Requirements 2 and 7 are only partly met (see the errors and resources items). The temperature-0 conflict for the LLM is documented but unresolved. (requirements/cycle-01.md items 2 and 7; config/project.yaml temperature: 0)
- [evaluation] The test set was looked at four times, including a stale lr_words and a second words-only look (lr_words_v2). The freeze entries are retroactive. 36 near-duplicate pairs cross splits. One site holds 44 of the 98 conspiracy test documents. (results/test_access_log.jsonl; decisions.md 'T3' and 'T4')
- [results] No automated check ties every number in prose to results/*.json. Some rounding is inconsistent (0.68 vs 0.684, 0.92 vs 0.920). (report §5; sections 08–11)
- [report] Report length before References is not confirmed; the whole file is 5,051 words. (reports/report.md)
- [ai_use] The human-check rows in the ai-log table are unfilled, so the team's own checks are missing. (docs/ai-log.md)
- [reproducibility] The double-run reproducibility diff was not done; the LLM stage is reproducible only through the cache. (decisions.md line 303; README)
- [teaching] Sections 07, 10 and 11 are very long for a mixed-background reader, and the sections have not been checked for a hypothesis in every front matter. (content/sections/07, 10, 11)
- [question] Hypotheses per section were not all checked, and H6's status depends on the pending human check. (content/sections/*.md front matter)
- [approach] 'What we did not do' rejects alternatives; the sections do not consistently name them with reasons next to each decision. (report §3; sections 07, 09)
- [interpretation] The label-versus-content conclusion rests on unverified Claude codes; lexicon weaknesses are discussed only briefly. (report §6; section 12)
- [features] The verdict is an honest null, but the explanation of why the lexicon and style groups add nothing is thin. (report §5.3; section 08)
- [privacy] Excerpt limits (at most 6 per section, 50 words) were not exhaustively checked; the stance justifications carry placeholders like [name] that show scrubbing happened. (content/sections, report)

### Reviewer summary
Overall this is a solid, honest cycle-1 package, with limits I should name. I read the requirements, the first part of metrics.json, report §5.1–§6, decisions.md, the test-access log and word counts. I did not check every section file, figure or number. Numbers I spot-checked match metrics.json. The main gaps are the blank human stance verification (kappa null, H6 provisional) and the still-unconfirmed LOCO licence. The test set was looked at four times, including a stale lr_words and the second words-only look. The LLM ran at default sampling because the API refused temperature 0. The test split has few conspiracy websites, and one site holds 44 of 98 conspiracy test documents. The report is 5,051 words in total; I could not confirm the count before References. Privacy looks clean on the checks I ran. Every criterion below 10 has a must_fix item; the biggest gains are the human stance check, the licence check, an automated number check and the reproducibility diff.

## Cycle 2 (17 Oct 2026)

- **Tool:** the same AI coding agent (Claude Code, Sonnet 5.5) working task by task under `AGENTS.md`; Claude Haiku 5.5 is the classifier under study.
- **What it produced:** the restructured notebook sections, the rewritten glossary (one `###` entry per technical term, each with an everyday example), the new section `12-how-we-used-ai` (the old section 13 was removed) and this entry.
- **What was checked:** every glossary entry is at most 80 words and the list in requirement 15 is covered; a script compared technical words used in the sections with the glossary; `python -m sitegen build` links the first use of each term; section 12 follows the four-part format and its word limits.
- **What was changed:** the AI-use section was cut to a short factual summary (what the AI produced, what was checked, what we learned) and no longer lists problems met while building.
- **What we learned:** automatic checks show that numbers match files, not that they mean the right thing; written rules plus automatic gates keep agent work reliable; a glossary only helps if every technical word on the site has an entry.
- **Human checks for cycle 2:** to be filled by team (glossary wording understandable without technical background; section 12 accurate; numbers spot-checked against `results/metrics.json`; privacy read). The human-check table of cycle 1 above is unchanged and still `pending`.
