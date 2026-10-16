# Conspiracy or mainstream? Logistic regression vs. an LLM on 9/11 articles

A student machine-learning project. We classify full articles about the 9/11 attacks from the
[LOCO corpus](https://doi.org/10.3758/s13428-021-01698-z) by the kind of website that published them,
once with a logistic regression we train ourselves and once with an LLM (Claude Haiku) that never saw
our training data, and study where each approach works and fails.

- **Walkthrough for the team:** the GitHub Pages site of this repo (built from `content/`), with code,
  outputs and explanations per step, and a feedback box under every section.
  [How to join in](content/guide.md).
- **Method and reasoning:** [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)
- **Rules for the AI agent that builds the analysis:** [`AGENTS.md`](AGENTS.md)
- **What the AI did and what we checked:** [`docs/ai-log.md`](docs/ai-log.md)

## How it is built

The analysis is written and improved in cycles by a LangGraph agent that runs in Anthropic's cloud
whenever the repo owner starts it. Each cycle ends with a pull request. The team reads the walkthrough, leaves feedback per
section, and that feedback becomes the next cycle's requirements (`requirements/`).

```
agent/       LangGraph graph (agent.py), state, nodes, tools, memory, prompts
src/fml/     the analysis (data, split, features, models, LLM classifier, evaluation)
content/     walkthrough sections (markdown with code/output cells)
sitegen/     turns content/ into the notebook-style site
results/     numbers and figures produced by `fml run-all`
reports/     the report draft
config/      project settings, agent limits, the 10/10 rubric
```

## Run it yourself

```bash
pip install -e ".[dev]"
export FML_DATA_DIR=/path/to/FML-data          # the private data repo
fml data-summary                               # check the data is found
fml run-all                                    # reproduce every result
python -m sitegen build && open site/index.html
python -m agent run --no-git                   # one agent step locally (needs Claude Code)
```

The article texts are not in this repository; they come from the LOCO corpus (Miani, Hills &
Bangerter, 2022), available at https://osf.io/snpcg/.
