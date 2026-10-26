# Human to-do

Checks that people must do; the AI cannot do them. Members A-E fill in the date and what was checked.
The stance hand check (item 2) is done with a separate private tool and arrives in cycle 4.

| # | Check | Owner | Status |
|---|---|---|---|
| 1 | Spot-check about 10 numbers in the report against `results/metrics.json` after the last `fml run-all` | Member A | pending |
| 2 | Fill `human_stance` for the 40 rows of `results/tables/stance_check_sample.csv` | Member D | pending |
| 3 | Confirm the LOCO licence at the dataset page and record it in `docs/decisions.md` | Member B | pending |
| 4 | Read the report and sections once for article text and personal data (including the repo link) | Member A | pending |
| 5 | Verify one or two 9/11 background and method references and add them to `content/references.md` (see "Needs a team decision") | Member C | pending |
| 6 | Read the glossary and section 11 for plain-language understanding | Member E | pending |
| 7 | Collect cases where the team altered or rejected AI output and add them to `docs/ai-log.md` | Member E | pending |
| 8 | Add your own checks of AI output for the Cycle 3 block of `docs/ai-log.md` (what you read, compared or changed, with date). Placeholder: nothing recorded yet | Members A-E | pending |
| 10 | Send the changes you made to AI output (what you altered or rejected, with date) so they can be entered under "Altered or rejected by members" in the Cycle 3 block of `docs/ai-log.md`. Section 11 says none are recorded yet; do not invent entries | Members A-E | pending |
| 9 | Owner task: in `scripts/collect_feedback.py` replace the default account name in the `--repo` default (line ~141) with a neutral value or require the env var, and set `repo_url` in the site config. The report and site must not carry a personal account name | Repo owner | pending |
