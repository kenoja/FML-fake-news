# AGENTS.md

Rules for every AI agent working in this repository (Claude Code reads this through `CLAUDE.md`).
Humans: this is also the shortest accurate description of how the project is put together.

## Mission

Compare a logistic regression we train ourselves with one cheap LLM (Claude Haiku 5.5, zero-shot) on
classifying **full** 9/11 articles from the LOCO corpus as coming from conspiracy-oriented or mainstream
websites. Find out where each approach works and fails, whether hand-made features help the logistic
regression, and explain every step so that five teammates with mixed backgrounds understand it.

## Sources of truth, in order

1. The **latest** `requirements/cycle-NN.md`. Requirements are cumulative: earlier files still apply
   unless a later item changes them. Later wins.
2. The course feedback quoted in `requirements/cycle-01.md` (it overrules the original proposal).
3. `docs/PROJECT_PLAN.md` (method and reasoning).
4. `config/project.yaml` (all settings: paths, split, model, limits). Never hard-code these.

If two of these contradict, do not blend them. Pick the higher one, do it, and record the conflict and
your choice in `docs/decisions.md`. If you cannot resolve something from the files, record it under
"Needs a team decision" in `docs/decisions.md`, take the most conservative reading, and say so.

## Hard rules (a failed rule fails the run)

- **Data stays private.** Article texts live in the private data repo (found via `fml.paths.data_dir()`).
  Never copy article text into this public repo. Excerpts in sections/report: at most 50 words each,
  at most 6 per section (`config/project.yaml: excerpts`). Caches store `doc_id` + hashes, never text.
- **No personal data.** No real names of students, staff or annotators, no institution, course name,
  course code or team number, no e-mail addresses, no GitHub usernames (use "Member A–E"). Published
  authors in citations are fine.
- **One split, frozen.** `data/splits/split_v1.csv` (doc_id → train/validation/test) is created once
  by `src/fml/split.py`: grouped by website, stratified by label, seed from config. Never regenerate,
  edit or "rebalance" it after results exist. Every model uses it.
- **Fit on train only.** Vocabulary, IDF weights, scalers and every learned quantity are fitted on
  the training split only. Model selection uses validation (or grouped CV inside train), never test.
- **Test set is locked.** Only `fml evaluate-final <model_id>` may score a model on test. It appends
  `{"model_id", "config_hash", "at", "n_docs"}` to `results/test_access_log.jsonl`. The same model id
  may never be scored on test with a different configuration.
- **The LLM classifier**: model from `config/project.yaml` (`claude-haiku-5-5`), temperature 0, the
  API key from the env var `LLM_CLASSIFIER_API_KEY` (never `ANTHROPIC_API_KEY`). It answers the **source
  question** (conspiracy-oriented vs mainstream website). At most 3 prompt variants, all tried on
  validation only, all reported. Responses cached in `results/llm_cache.jsonl` keyed by doc_id and
  prompt hash, so `fml run-all` reruns without new API calls. No key and no cache entry → fail loudly.
- **Off-topic filter, in this order.** The first words-only model runs on all 3,146 documents.
  Then documents not matching `data.topic_filter_pattern` are removed (expected 2,608 remain:
  777 conspiracy, 1,831 mainstream), and a section shows the before/after numbers and explains why.
  All later experiments use the filtered set and the same split.
- **Protected paths.** Do not edit `agent/`, `sitegen/`, `scripts/`, `.github/`, `config/rubric.yaml`,
  `config/agent.yaml` or `requirements/`. They define how you are run and graded.
- **Fail loud.** "Done" is wrong if anything was skipped; "tests pass" is wrong if any were skipped.
  Write what you could not do into your final summary and into `docs/decisions.md`.

## Contracts (what other parts rely on)

| What | Where / how |
|---|---|
| Reproduce everything | `fml run-all` → writes `results/` (calls `fml.pipeline.run_all()`) |
| Final test scoring | `fml evaluate-final <model_id>` (calls `fml.pipeline.evaluate_final`) |
| All numbers | `results/metrics.json` (top-level keys: `data`, `split`, `baseline_all_docs`, `topic_filter`, `baseline`, `ablation`, `llm`, `comparison`, `errors`), plus CSV tables in `results/tables/` |
| Figures | `results/figures/*.png`, matplotlib, 150 dpi; conspiracy `#8c3b76`, mainstream `#2b7a6a` everywhere |
| Notebook sections | `content/sections/NN-slug.md` (format below) |
| Report | `reports/report.md`, 3,000–5,000 words before References |
| AI-use log | `docs/ai-log.md` (the agent appends after every cycle; humans add what they checked) |
| Decisions | `docs/decisions.md` (choices, conflicts, open questions) |

Numbers in prose are copied from `results/` files, never typed from memory. When results change,
update every place that quotes them.

## Performance

The whole pipeline must stay fast enough to check often: `fml run-all` must finish in **under 8
minutes** on a 2-CPU machine, and the full test suite in under 3 minutes.
- Build text features (TF-IDF vocabulary, n-grams, lexicon counts, style features) **once per feature
  configuration** on the training split and reuse the matrices for every regularisation value and
  class-weight setting. Do not refit vectorizers inside the C loop.
- Use a solver that suits large sparse inputs (e.g. `liblinear` or `saga`) and a sensible `max_iter`.
- Cache expensive intermediate results under `results/cache/` (git-ignored) keyed by a config hash.
- Slow tests on the real data belong in at most a few `needs_data` tests; unit tests use tiny inputs.

## Section format

Planned sections (create what is missing, keep slugs stable because feedback threads use them):
`00-overview` (exists), `01-research-questions`, `02-data-and-labels`, `03-split-by-website`,
`04-tfidf-and-logistic-regression`, `05-first-run-all-documents`, `06-off-topic-filter`,
`07-feature-engineering`, `08-do-the-features-help`, `09-llm-classifier`, `10-head-to-head`,
`11-error-analysis`, `12-limitations`, `13-how-we-used-ai`, `99-glossary`.

Each file:

```markdown
---
title: Splitting by website
part: Evaluation                 # which part of the assignment this explains
hypothesis: One sentence: what we expected before running it.
---
## Goal and hypothesis
## Explanation                    (plain language; define every technical term on first use)
## What could go wrong
## Technical infrastructure       (files, functions, commands)
```

Put code and outputs between the text where they belong, notebook-style:

```markdown
<!-- cell: src/fml/split.py#make_split -->                  real code, by function/class name
<!-- cell: src/fml/features/style.py:10-42 -->              or by line range
<!-- output: results/figures/split_sizes.png | Documents per split and label -->
<!-- output: results/tables/split_summary.csv -->
<!-- output: results/metrics.json#baseline.validation -->
```

Every section except `00-overview` and `99-glossary` needs at least one code cell. Write for a
teammate who knows little ML: short sentences, concrete examples, define TF-IDF, F1, precision,
recall, regularisation, grouped split, confidence interval, McNemar test the first time they appear,
and answer every **Unclear** item from the requirements in the section it is about.

## Code conventions

- Python ≥ 3.11, type hints, small functions; docstrings say *why*, not only what.
- pandas + scikit-learn; `random_state` from config everywhere.
- Tests in `tests/` encode why a rule matters and must fail when the rule breaks (e.g. a test that
  fits on all data must make the leakage test fail). Mark tests that need the data repo with
  `@pytest.mark.needs_data` (they skip when it is missing, e.g. in CI).
- No Jupyter notebooks: the site *is* the notebook.

## How the agent loop runs (for humans)

`python -m agent run` advances the current cycle for about 20 minutes, checkpoints after every step
(`.agent_state/`), commits and pushes to `claude/cycle-NN`, and prints `STATUS: CONTINUE | DONE | PAUSED
| FAILED`. Run it again to continue; it resumes from the last finished step. When the cycle is done it
opens a pull request to `main`. Limits are in `config/agent.yaml`. Setup: `docs/SETUP.md`.
