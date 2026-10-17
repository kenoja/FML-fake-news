# Decisions

One section per cycle, newest last. Every item carries a tag: **[done]** (decided and carried out),
**[declined]** (not done, with the reason) or **[open]** (waiting for a person). Open items are repeated under
"Needs a team decision" at the end. Agents: add an entry whenever you choose between alternatives or
resolve a conflict (see AGENTS.md).

## Cycle 1 (2026-10-16 to 2026-10-17)

- **[done] Course feedback overrules the proposal.** Full articles instead of headlines; logistic regression
  vs one cheap LLM; no fine-tuning; a few extra feature groups compared with words only.
- **[done] LLM question.** The LLM answers the source question (conspiracy-oriented vs mainstream website),
  the same target the logistic regressions are scored on. Content stance is examined in the error analysis.
- **[done] Feature groups.** n-grams (1-3), certainty words (epistemic markers), style. Emotion scores were
  dropped by the team (extra cost, and the course warns that emotional language is not conspiracy language).
- **[done] Data location.** Full texts only in the private data repo; this repo is public.
- **[done] Split.** `data/splits/split_v1.csv` was made once by `fml prepare-data` (seed from config, whole
  websites assigned to one split). It covers all documents of the corpus and is frozen; cleaning only drops
  rows at analysis time, so a dropped document keeps its split and nothing moves between train and test.
- **[done] Conflict: final-scoring command.** AGENTS.md says `fml evaluate-final <model_id>`, the plan says
  `fml evaluate --final`. AGENTS.md ranks higher, so `evaluate-final` is used.
- **[done] Dependencies.** `pyproject.toml` keeps ranges so CI and newer machines can install;
  `requirements-lock.txt` is the exact pin. The seven analysis packages and the Python version in the lock
  match `results/run_info.json`; the other lock entries were not compared with an installed environment.
- **[done] Test lock bookkeeping.** `results/frozen_models.json` records each model before its first test
  scoring; `evaluate_final` refuses a missing or changed entry (`tests/test_test_lock.py`). Three early test
  looks of cycle 1 predate that rule and are marked retroactive; the cycle-2 final models are new ids.
- **[done] Stance-check privacy.** Justifications in the public CSVs are cut to 25 words and name-like words
  are replaced by `[name]` (`errors.scrub_justification`).
- **[declined] Hypotheses "written before the results".** The dating cannot be shown (git history was not
  accessible to the agent), so no text claims it. The hypotheses are described as chosen from the literature
  and fixed before the final test run. There is no H5 on purpose (the emotion idea was dropped).
- **[declined] Licence check by the agent.** The OSF page https://osf.io/snpcg/ could not be opened in any
  cycle (web access denied, nothing fetched, nothing guessed). Safe default: texts private, excerpts at most
  50 words and 6 per section. See [open] below.

## Cycle 2 (2026-10-17)

- **[done] Conflicts resolved by the newest source.** (a) Temperature: cycle 1 asked for temperature 0, cycle 2
  and AGENTS.md say default sampling (the model rejects the setting); nothing is sent, the output limit is
  1000 tokens. (b) The first run on all documents is gone: cleaning comes first and every model uses the
  cleaned set. (c) Extra analyses of cycle 1 (stance kappa, website bootstrap, stability, sensitivity
  analyses, prior-knowledge probe) were removed; only the statistics listed in AGENTS.md remain.
  (d) Near-duplicates are now removed before any model runs.
- **[done] Cleaning.** `clean.clean_documents()`: off-topic removal (3,146 to 2,608), then near-duplicate
  removal (cosine 0.9 or higher; 2,608 to 2,554). Across splits the validation/test copy goes; for a
  validation/test pair the test copy goes; inside one split the lower doc_id stays. No minimum length (the
  shortest article has 102 words). Numbers: `results/metrics.json#cleaning`.
- **[done] Four analyses.** Word counts, TF-IDF, TF-IDF plus features (five models, chosen on validation with
  one rule: a group helps only if validation F1 rises by at least `ablation.HELPS_BY`), and the LLM (at most
  three prompt variants, validation only). Shared logistic-regression settings: C grid and balanced class
  weights from `config/project.yaml`. The "best logistic regression" for the third McNemar test is the
  final model with the higher validation F1.
- **[done] Hypotheses.** H1 TF-IDF beats word counts; H2a/b/c certainty words, style, n-grams help TF-IDF;
  H3 the LLM beats the best logistic regression. H2b and H2c were only tried on validation.
- **[done] Final run.** All four final models were scored on test once each (`final_*` ids,
  `results/test_access_log.jsonl`). Comparisons appear only in `08-results` and later.
- **[done] Stance-check sample.** 40 validation articles (20 + 20, at most 6 per website, 19 websites),
  stance codes cached in `results/stance_cache.jsonl`. The `human_stance` column is blank on purpose.
- **[done] Report counting rule.** The 3,000-5,000 word limit is `len(text.split())` on `reports/report.md`
  before References, tables included (stricter than the agent gate). Tests: `tests/test_gates.py`.
- **[done] Report and sections.** Stale section files were deleted; the licence is not discussed on the site
  or in the report beyond "texts are not redistributed"; 00 has no report part (sitegen would have copied it).
- **[done] Reproducibility (requirement 18).** Two consecutive `fml run-all` runs gave identical tables and
  metrics; both checksum lists are in `docs/reproducibility.md`.
- **[done] Privacy of the account name in the site.** `config/site.yaml` has empty `repo_url` and
  `giscus.repo`, so the built site shows no account name (no "Code on GitHub" link, comments show "not
  connected yet"). `tests/test_privacy.py` scans docs, `config/` and the built site.
- **[partly met] Requirement 14 (beginner code).** The pipeline stages and `StyleFeatures` were split into short
  helpers and tables stayed byte-identical, but the full rewrite of every code cell shown on the site in the
  AGENTS.md style is not confirmed complete. It stays "partly met" until the queued code task (T1) is done.
- **[declined] Protected-path conflict.** `scripts/collect_feedback.py` contains the owner's account name as
  a default `--repo` value. `scripts/` is protected, so the agent did not edit it, and git history cannot be
  rewritten by the agent. Conservative reading: do not claim the repository is free of the name.
- **[declined] Scoring the human stance check by command.** The agreement code was removed as out of scope
  (requirement 13). If members fill the sheet, agreement is computed by hand.
- **[declined] Deleting `tests/test_agent_graph.py`.** It tests the protected `agent/` code, not the study; the
  delete command was blocked, so the file stays.
- **[open] Warning left.** `10-limitations.md` has a non-flat `llm.repeat_agreement` output marker in its report
  part; sitegen drops it with a warning.

## Needs a team decision

- **Owner account name (protected path).** The owner must replace the name in `scripts/collect_feedback.py` or
  move the repository to a neutral organisation, then set `repo_url` and `giscus.repo` in `config/site.yaml`.
  Until then pass `--repo` or set `GITHUB_REPOSITORY`. Git history cannot be rewritten by the agent. To make
  `tests/test_privacy.py` search for the name itself, set `FML_PRIVACY_TERMS=<account name>`.
- **LOCO licence.** One member opens https://osf.io/snpcg/ and records here: licence text, date checked,
  Member letter. Until then the safe default above applies.
- **Human stance check.** Two members label the 40 rows of `results/tables/stance_check_sample.csv`
  independently, before looking at `claude_stance`. Until then the Claude stance codes are an unchecked single
  source and the stance-related hypothesis stays provisional.
- **What the team altered or rejected in AI output.** Only one concrete case is recorded (emotion scores
  removed from the AI-drafted plan, cycle 1). Further cases are **not yet recorded**: members add them here and
  in `docs/ai-log.md`.
- **Report length counting.** Confirm the plain whitespace count (tables included) or ask for prose-only
  counting.
- **Near-duplicate choice.** Confirm keeping the validation copy of a validation/test pair.
- **Test file for the agent.** Delete `tests/test_agent_graph.py` if CI does not need it.
