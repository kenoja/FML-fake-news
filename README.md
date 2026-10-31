# Conspiracy or mainstream? Logistic regression vs. an LLM on 9/11 articles

A student machine-learning project. We classify full articles about the 9/11 attacks from the
[LOCO corpus](https://doi.org/10.3758/s13428-021-01698-z) by the kind of website that published them:
three logistic regressions we train ourselves (word counts, TF-IDF, TF-IDF plus hand-made features) and
one LLM (Claude Haiku 5.5) that never saw our training data. We study where each works and fails.

## What is where

- `content/` the walkthrough sections (code, outputs, plain explanations); [how to join in](content/guide.md)
- `src/fml/` the analysis; `results/` numbers, tables and figures; `reports/report.md` the report
- `docs/PROJECT_PLAN.md` method and reasoning; `docs/decisions.md` choices and open questions
- `docs/ai-log.md` what the AI did and what we checked; [`docs/reproducibility.md`](docs/reproducibility.md) evidence that two runs give identical results
- `AGENTS.md` the rules for the AI assistant

The article texts are not in this repository (private data repo; corpus: Miani, Hills & Bangerter, 2022).

## How to run it

Python 3.11 or newer (`requires-python = ">=3.11"` in `pyproject.toml`). Run these commands in this order:

```bash
pip install -r requirements-lock.txt && pip install -e . --no-deps    # 1. install the pinned versions
export FML_DATA_DIR=/path/to/data-repo     # private data repo with the article texts
export LLM_CLASSIFIER_API_KEY=...          # 2. only if results/llm_cache.jsonl or results/stance_cache.jsonl
                                           #    is missing an entry; with the caches no key is needed
fml run-all                                # 3. rebuilds results/ (no API calls with the caches)
python -m pytest -q                        # 4. tests; those marked needs_data skip without the data repo
python -m sitegen build                    # 5. rebuilds reports/report.md and the site from content/
PYTHONPATH=src python reports/latex/build_pdf.py   # 6. reports/report.pdf (needs XeLaTeX, pandoc, the Inter font)
```

Expected runtimes: `fml run-all` must finish in under 8 minutes on a 2-CPU machine (measured: about
3.5 to 4 minutes on a 2-CPU machine, see `results/run_info.json`); the full test suite must finish in
under 3 minutes.

The exact package versions are pinned in `requirements-lock.txt`. How we checked that two runs give
identical results is in [`docs/reproducibility.md`](docs/reproducibility.md).

`fml evaluate-final <model_id>` is the only way to score a model on the locked test split (it is logged).
The final models are `final2_word_counts`, `final2_tfidf`, `final2_tfidf_features` and `final_llm`.
All random choices use the seed in `config/project.yaml`.

## Versions

`requirements-lock.txt` is the exact pin (`pyproject.toml` only gives the ranges that may work). The
committed results were made with Python 3.11.17, numpy 2.4.6, pandas 3.1.0, scipy 1.17.1,
scikit-learn 1.9.1, matplotlib 3.11.2, anthropic 1.17.0 and pyyaml 6.0.1 (`results/run_info.json`).

## Check that it reproduces

Run `fml run-all` twice and compare `sha256sum results/metrics.json results/tables/*`. Only
`results/run_info.json` (timing) may differ. Checksums and runtime are in `docs/reproducibility.md`.
