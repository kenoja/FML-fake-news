# AGENTS.md

Rules for every AI agent working in this repository (Claude Code reads this through `CLAUDE.md`).
Humans: this is also the shortest accurate description of how the project is put together.

## Mission

Classify **full** 9/11 articles from the LOCO corpus as coming from conspiracy-oriented or mainstream
websites, in four ways: logistic regression on (1) word counts, (2) TF-IDF weights, (3) TF-IDF plus
engineered features, and (4) one cheap LLM (Claude Haiku 5.5, zero-shot). Find out where each works and
fails and whether the hand-made features help.

**Audience.** Five teammates with no technical background, in a 2.5-ECTS fundamentals course. They read
the site to learn the concepts, check their own understanding and criticise the text. Plain words beat
technical completeness everywhere: keep the analysis simple, and explain concepts rather than details.

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
- **The LLM classifier**: model from `config/project.yaml` (`claude-haiku-5-5`), its default sampling
  settings (the model rejects a temperature setting), an output limit large enough that answers are
  never cut off, the API key from the env var `LLM_CLASSIFIER_API_KEY` (never `ANTHROPIC_API_KEY`). It answers the **source
  question** (conspiracy-oriented vs mainstream website). At most 3 prompt variants, all tried on
  validation only, all reported. Responses cached in `results/llm_cache.jsonl` keyed by doc_id and
  prompt hash, so `fml run-all` reruns without new API calls. No key and no cache entry → fail loudly.
- **Cleaning comes first.** Before any model runs, cleaning removes off-topic documents (not matching
  `data.topic_filter_pattern`: 3,146 → 2,608, i.e. 777 conspiracy, 1,831 mainstream) and near-duplicate
  articles. Section `02-data-and-cleaning` shows the before/after numbers and explains why. Every model
  uses the cleaned set and the same split.
- **All runs first, then compare.** Settings are chosen on validation for all four analyses; then one
  final run scores every final model on test exactly once (new model ids, see the test-set rule).
  Comparisons and test numbers appear only in `08-results-and-why-they-fail` and later.
- **Protected paths.** Do not edit `agent/`, `sitegen/`, `scripts/`, `.github/`, `config/rubric.yaml`,
  `config/agent.yaml`, `config/site.yaml`, `requirements/` or the harness tests
  (`tests/test_agent_graph.py`, `test_gates.py`, `test_sitegen.py`, `test_collect_feedback.py`). They define how you are run and graded.
- **Fail loud.** "Done" is wrong if anything was skipped; "tests pass" is wrong if any were skipped.
  Write what you could not do into your final summary and into `docs/decisions.md`.

## Contracts (what other parts rely on)

| What | Where / how |
|---|---|
| Reproduce everything | `fml run-all` → writes `results/` (calls `fml.pipeline.run_all()`) |
| Final test scoring | `fml evaluate-final <model_id>` (calls `fml.pipeline.evaluate_final`) |
| All numbers | `results/metrics.json` (top-level keys: `data`, `cleaning`, `split`, `word_counts`, `tfidf`, `tfidf_features`, `llm`, `comparison`, `errors`), plus CSV tables in `results/tables/` |
| Figures | `results/figures/*.png`, matplotlib, 150 dpi; conspiracy `#8c3b76`, mainstream `#2b7a6a` everywhere |
| Notebook sections | `content/sections/NN-slug.md` (format below) |
| Report | `reports/report.md` is **assembled automatically** by `python -m sitegen build` from every section's `## Report` part plus `content/references.md`. Never edit it by hand. 3,000–5,000 words before References (gate). |
| AI-use log | `docs/ai-log.md`: factual Produced / Checked / Learned per cycle for the report (no run diary); humans add what they checked. The harness writes its own run records to `docs/archive/agent-runs.md`. |
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

Sections, in this order (slugs are fixed because feedback threads use them; delete section files
whose slug is not in this list):
`00-how-to-use`, `01-question-and-hypotheses`, `02-data-and-cleaning`, `03-split-by-website`,
`04-word-counts`, `05-tf-idf`, `06-feature-engineering`, `07-llm`, `08-results-and-why-they-fail`,
`09-limitations`, `10-conclusion`, `11-how-we-used-ai`, `99-glossary`.

Every section except `00-how-to-use` and `99-glossary` has exactly these four parts, in this order:

```markdown
---
title: Splitting by website
part: Evaluation                 # which part of the assignment this belongs to
hypothesis: (optional) one plain sentence: what we expected
---
## Explained simply
Plain-language explanation of the ideas in this step and of every design choice we made (and why),
with an everyday example. It must stand on its own: someone who reads only this part understands the
step. At most 450 words (gate). No numbers that need the code to understand.

## The code
Short sentences between cells that say what each cell does, then the cells:
<!-- cell: src/fml/split.py#make_split -->                  real code, by function/class name
<!-- output: results/figures/split_sizes.png | Documents per split and label -->
<!-- output: results/tables/split_summary.csv -->
<!-- output: results/metrics.json#split.sizes -->

## Report
The text of the final report for this part, written as report prose (academic, concise). It may
contain output markers for figures and tables (flat JSON only), never code cells. All report parts
together are the whole report: 3,000–5,000 words, coherent when read without the rest of the site.

## Check yourself
Q: A question that tests understanding, not memory?
A: The answer, in one to three sentences.
(at least two pairs; the answers are hidden until clicked)
```

**Glossary (`99-glossary`).** One entry per technical term, as `### Term`, an optional line
`Also: other name, other name`, then a definition in plain words with an everyday example (at most 80
words). The site links the first use of each term in every section to its entry automatically, so
every technical word used anywhere must have an entry.

**What stays off the site.** Plans, task lists, decision logs, "Member __ confirm" lines, agent errors
and retries belong in `docs/` only. The site and the report describe the study, not how the agent
built it. The test set was used during development: `09-limitations` says so in one honest sentence
and states that every final model was chosen on validation and scored on test once.

**Statistics, and nothing more.** Confusion matrix, precision, recall, F1 (conspiracy class), one 95%
bootstrap confidence interval per number, and McNemar tests for three pairs (word counts vs TF-IDF,
TF-IDF vs TF-IDF with features, best logistic regression vs LLM). Define each in the glossary.

## Code conventions

- Python ≥ 3.11. **Write for a beginner who reads the code on the site**: short functions, plain
  variable names, one step per line, short inline comments in plain English (`# keep only training
  articles`). At most a one-line docstring; no long comment blocks. Remove code for analyses that are no
  longer reported instead of leaving it unused.
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
