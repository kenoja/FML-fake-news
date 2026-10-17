You are implementing one task in cycle $cycle of this repository. Follow `AGENTS.md` strictly
(it is loaded automatically). The requirements for this cycle are in `$requirements_path`.

Your task (JSON):
$task

Still queued after this one (do NOT do these now; just avoid conflicting with them):
$remaining

How to work:
0. An earlier attempt at this task may have been interrupted (time limit). Files can already contain
   part of the work: check what exists, keep what is correct, and finish the rest. Work in small,
   complete increments and run the tests as you go, so an interruption never leaves broken code.
1. Read before you write: the files you will change, their callers and tests, and the relevant
   part of `docs/PROJECT_PLAN.md`.
2. Make the smallest change that fully meets `acceptance`. Match the existing code style.
3. Tests encode WHY a rule matters (e.g. "TF-IDF must be fitted on train only, otherwise test
   vocabulary leaks into training"), and must fail if that rule is broken.
4. Run what you changed, quickly: the test files you touched (`python -m pytest -q tests/test_x.py`)
   and the one stage or function you changed. Prefix every command with `timeout 300` (5 minutes).
   Do NOT run `fml run-all` or the whole test suite inside a task: the harness runs both right
   after the task, and a task that runs out of time is lost work. If a stage is too slow to check in
   5 minutes, making it faster is part of the task (see AGENTS.md, "Performance").
   Do not claim success for anything you did not run.
5. If the task produces numbers, write them to `results/` (JSON/CSV) and never type numbers into
   prose by hand: sections and the report reference values from `results/`.
6. If you touch a notebook section in `content/sections/`, keep all required headings and real
   code cells (see AGENTS.md, "Section format").
7. Never put article text into the repo beyond the excerpt limits; never print secrets.

Finish with a short plain-text summary: what you changed (files), what you ran and its outcome,
and anything you could not do (be explicit; "done" is wrong if anything was skipped).
