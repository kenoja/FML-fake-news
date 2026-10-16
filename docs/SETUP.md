# Setup (repo owner, once)

About 20 minutes. Do the steps in order. Nothing here needs a local install.

## 1. Merge the setup pull request

Open the pull request "Agent setup for cycle 1" in `FML-fake-news`, read it, and merge it into `main`.
The agent compares its work against `main`, so it must contain this setup before the first run.

## 2. GitHub settings for `FML-fake-news`

1. **Pages:** Settings → Pages → *Build and deployment* → Source: **GitHub Actions**.
2. **Discussions:** Settings → General → Features → tick **Discussions**.
   Then open the new Discussions tab → the pencil next to *Categories* → **New category**:
   name `Feedback`, format **Announcement** (only you and the comment tool can start threads,
   so every thread matches a section).
3. **Comment tool:** install the giscus app on this repo only: https://github.com/apps/giscus → Install
   → *Only select repositories* → `FML-fake-news`.
4. **Pull requests from workflows:** Settings → Actions → General → *Workflow permissions* →
   tick **Allow GitHub Actions to create and approve pull requests** (needed by "Collect feedback").
5. **Team mapping (secret):** once teammates have GitHub accounts, Settings → Secrets and variables →
   Actions → **New repository secret**, name `TEAM_MEMBERS`, value like
   `{"their-username-1": "Member A", "their-username-2": "Member B", ...}` (include yourself).
   It stays secret; files only ever show "Member A–E".
6. After the first push to `main`, the **Publish walkthrough** workflow runs. The site address is
   shown under Settings → Pages. Share it and `guide.html` with the team.

## 3. Cloud environment (claude.ai)

Go to https://claude.ai/code, open the environment picker (cloud icon) → **Add environment**
(or edit one), name it `fml`:

- **Network access:** Custom → tick *Also include default list of common package managers* →
  allowed domains: `osf.io`, `api.osf.io` (only for re-checking the dataset licence).
  `api.anthropic.com`, PyPI and GitHub are already reachable.
- **Environment variables** (only your sessions in this environment can read them):

  ```
  LLM_CLASSIFIER_API_KEY=<your key from platform.claude.com>
  FML_PRIVACY_TERMS=<comma-separated real names, institution, course name/code, team number>
  BASH_DEFAULT_TIMEOUT_MS=1800000
  BASH_MAX_TIMEOUT_MS=1800000
  ```
  The key is an environment variable, not a "network secret", because network secrets are never
  attached to requests to `api.anthropic.com`. Do **not** call it `ANTHROPIC_API_KEY`.
- **Setup script:** leave empty. `python -m agent run` installs what it needs on first start.

## 4. The agent routine (started by hand)

The agent never runs on its own. You start it whenever you want it to work.

https://claude.ai/code/routines → **New routine**:

- **Name:** `FML agent`
- **Model:** Sonnet 5.5
- **Repositories:** `kenoja/FML-fake-news` and `kenoja/FML-data`
- **Environment:** `fml`
- **Connectors:** remove all (the agent does not need any)
- **Trigger:** **API** only, and do **not** generate a token. No schedule. Without a token nobody can
  start it remotely, so the only way to start it is the **Run now** button.
- **Prompt** (copy exactly):

```text
You run the agent for this project. Do not edit any files yourself.

1. In the FML-fake-news repository, run `python -m agent run` with a command timeout of
   25 minutes. It works for up to 20 minutes, saves progress, pushes to the branch
   claude/cycle-NN, and ends with a line "STATUS: <word>".
2. If that word is CONTINUE, run the same command again. Keep repeating.
3. Stop when the word is DONE, PAUSED or FAILED.
4. End with a short summary: the last STATUS line and the contents of .agent_state/last_run.md.
```

**To start work:** open the routine → **Run now**. Each run continues where the last one stopped, so
start it as often as you like. When a cycle is already finished, a run ends right away with `DONE`.

## 5. Each cycle

1. The agent finishes a cycle by opening a pull request `claude/cycle-NN → main` with its
   scores and open issues. Read the summary, look at the changes, merge.
2. The walkthrough updates automatically. Tell the team the feedback period has started.
3. At the end of the period: Actions → **Collect feedback** → *Run workflow* → enter the cycle number.
   It opens a pull request with `requirements/cycle-(NN+1).md`.
4. Edit that file (remove rejected items, decide conflicts), merge. Then select **Run now** on the
   routine to start the new cycle.

## Useful controls

- **Limits** (time per run, review rounds, ...): `config/agent.yaml`.
- **Status of the last run:** `.agent_state/last_run.md` on the `claude/cycle-NN` branch.
- **Start a cycle over:** delete `.agent_state/cycle-NN.sqlite` on its branch (or run
  `python -m agent reset-cycle NN` in a cloud session) and run again.
- **Stop a run:** open its session at claude.ai/code and stop it. Everything up to the last finished
  step is kept.
