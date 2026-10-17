# Project plan: Logistic regression vs. an LLM for classifying 9/11 conspiracy articles

v0.3 · 16 Oct 2026 · approved as a starting point; updated with the team's decisions
This file will live in the repo as `docs/PROJECT_PLAN.md`. It is the agent's master spec. Where the original proposal and the course feedback disagree, **the course feedback wins**.

---

## 0. Fixed decisions and where they come from

| Decision | Source |
|---|---|
| Classify **full article texts**, not headlines | Course feedback: a headline may mention a conspiracy without showing whether the article supports, rejects or merely discusses it |
| Compare **logistic regression (LR)** with **one cheap LLM** only. No transformer fine-tuning, no multi-model benchmark | Course feedback |
| The LLM is **Claude Haiku 5.5** (`claude-haiku-5-5`), the cheapest current Anthropic model | Team decision; pricing page |
| Two LR models: **words-only** vs **extended features**, plus an ablation over a *few* feature groups | Course feedback |
| One fixed **website-grouped split** for every comparison; vocabulary and TF-IDF fitted **on the training set only** | Course feedback, proposal |
| Error analysis treats the label as a statement about the **website**, not the text | Course feedback; Mompelat et al. 2022 |
| The report documents **what the AI produced, what we checked or changed, and what we learned** | Course feedback; assignment ("Use of LLMs and other AI tools") |
| JFK transfer test: **parked** (not in scope for now) | Team decision |
| Everything in English | Team decision |
| **Public** analysis repo (`<owner>/<repo>`) + **private** data repo (`<owner>/<data-repo>`). No names, institutions, course codes or other personal data anywhere public | Team decision |
| The agent runs in **Anthropic's cloud** as a routine started by hand (uses the cloud-session credit); GitHub Actions only publishes the site and collects feedback | Team decision |
| Off-topic documents (no mention of 9/11) are removed **after** the first words-only run, with an explanation | Team decision |
| LLM is asked the **source question** (conspiracy-oriented vs mainstream website), the same target LR is scored on | Team decision |
| Emotion features (G4) are **out** | Team decision |
| Agent work arrives as a **pull request** per cycle, merged by the repo owner | Team decision |

## 1. Research questions and hypotheses

- **RQ1.** On full 9/11 articles from LOCO, how do (a) a words-only LR, (b) an extended-feature LR and (c) a zero-shot LLM compare on the *same* website-held-out test set?
- **RQ2.** Do engineered features improve LR over words alone? Which feature groups help, and which don't?
- **RQ3.** Where does each approach fail? For each disagreement with the dataset label: is the *prediction* implausible, or does the *label* not describe the text?

Hypotheses (each one becomes a notebook section with its test):

| ID | Hypothesis | Test |
|---|---|---|
| H1 | Both approaches beat the majority-class baseline on conspiracy-class F1 | Section 6 |
| H2 | Adding 2- and 3-word n-grams helps more than any single non-lexical group, because phrases like "no evidence of" or "inside job" carry stance that single words lose | Ablation |
| H3 | Epistemic markers (certainty, hedging, "hidden truth" vocabulary) add signal beyond n-grams | Ablation |
| H4 | Style features (ALL CAPS, `!`, `?`, quotation density, sentence length, repetition) add signal, but less than lexical features | Ablation |
| H6 | Many "false positives" on mainstream articles discuss or debunk conspiracy theories, and many "false negatives" on conspiracy sites contain no conspiracy claim (label noise) | Error analysis |

## 2. Data

- **Corpus:** LOCO, the Language of Conspiracy corpus (Miani, Hills & Bangerter, 2022). Fields used: `doc_id`, `website`, `seeds`, `subcorpus`, `title`, `txt`, `txt_nwords`.
- **Where it lives:** the private repo `<owner>/<data-repo>` holds the full LOCO download (compressed, split into parts under 100 MB, with checksums) and the extracted 9/11 subset `loco_911/loco_911.jsonl.gz`. The public repo never contains article texts.
- **Subset (checked 16 Oct 2026):** 3,146 documents whose `seeds` contain `september.11.attack`: 939 conspiracy and 2,207 mainstream, from 39 conspiracy and 79 mainstream websites. No empty texts, no exact duplicate texts; median length 661 words (min 101, 90th percentile 2,489).
- **Label:** `subcorpus` (conspiracy / mainstream). This is **source-based**: it says which kind of website published the text.
  - The authors' own blind check of 60 documents per subcorpus found 85% of conspiracy and 92% of mainstream documents correctly labelled.
  - A later annotation study found that many conspiracy-subcorpus documents contain no conspiracy language at all. For two other topics the rates were about 62% and 41% (Mompelat et al. 2022).
  - We therefore *expect* label noise, mainly on the conspiracy side.
- **Off-topic documents:** about 17% of documents in *both* classes never mention 9/11 (e.g. health articles about depression, dictionary pages); LOCO's keyword retrieval brought them in. Team decision: the first words-only model runs on all 3,146 documents; then documents not matching the 9/11 pattern in `config/project.yaml` are removed (2,608 remain: 777 conspiracy, 1,831 mainstream), and the walkthrough explains why with the before/after numbers.
- **Cleaning** (each step logged with counts before and after):
  1. Remove near-duplicates (normalised-text hash, plus a cosine-similarity check for near-identical texts across splits).
  2. Remove **source giveaways**: website names, URLs, bylines and boilerplate lines that repeat across one site's documents. Otherwise the model learns "which website", not "which language".
  3. Report the length distribution; no minimum length is needed with full articles (shortest: 101 words), but this is shown, not assumed.
- **Licence:** the LOCO paper calls the corpus freely available but names no licence; the OSF page could not be checked on 16 Oct (maintenance). Rule: no article texts in the public repo, excerpts of at most 50 words in public outputs. The agent re-checks the OSF page and records the result.

## 3. Train / validation / test split

- **Grouped by website and stratified by label**, roughly 70/15/15 of documents, with a fixed random seed. Stored as `data/splits/split_v1.csv` (`doc_id → split`) and reused by *every* model.
- **Why grouped:** if one website appears in both training and test data, a model can score highly just by recognising that site's style. In one benchmark, AUC fell from 0.97 to 0.72 once test domains were unseen during training (Bozarth & Budak 2020).
- **Websites available:** 39 conspiracy and 79 mainstream sites, so roughly 6 conspiracy sites land in the test split. That is enough to run, but results per website vary, so confidence intervals also resample whole websites.
- **Risk:** a split could still end up with very few conspiracy websites. The agent reports websites and documents per split and class. If any split holds fewer than 3 conspiracy websites, it flags this loudly and adds a per-website results table.
- **Test set lock:** the test set is read only by `fml evaluate --final`. Each read is logged with a hash. A unit test fails if model-selection code can reach the test split.

## 4. Models

### 4.1 M1: LR, words only
TF-IDF on single words (vocabulary and IDF weights fitted on training data only), L2-regularised logistic regression. The regularisation strength `C` and class weighting are chosen on the validation set. Every value tried is reported.

### 4.2 M2: LR, extended
M1's words plus the feature groups below, added one at a time (the ablation) and then all together. Numeric features are standardised with a scaler fitted on training data only.

| Group | Features (per 1,000 words unless noted) | Why this group (literature) |
|---|---|---|
| **G1 n-grams** | TF-IDF over 1- to 3-grams | Logistic regression on TF-IDF n-grams up to trigrams was the strong baseline for separating reliable from unreliable news (Rashkin et al. 2017). Bigrams consistently help linear text classifiers (Wang & Manning 2012). Phrases also capture negation ("no evidence of"), which single-word lists miss (Chen 2026). |
| **G2 epistemic markers** | Counts of hedges ("perhaps", "allegedly"), boosters / certainty ("undeniable", "clearly") and "hidden truth" / debunking vocabulary ("cover-up", "they don't want you to know"), from a published hedge/booster lexicon (Islam et al. 2020) plus a small documented list | Conspiracy discourse is marked by evidentiality and certainty markers and vocabulary claiming a hidden truth (Marini & Jezek 2024). Unreliable news used more hedges, subjective words and superlatives (Rashkin et al. 2017). The evidence on certainty is **mixed**: it was lower for conspiracy influencers but higher for their followers (Fong et al. 2021). That makes it a genuine hypothesis to test, not a given. Known weakness: word lists miss negation and scope, which is why G1 sits alongside it (Chen 2026). |
| **G3 style** | Share of ALL-CAPS words; `!` and `?` per 1,000 words; quotation marks per 1,000 words; mean sentence length; moving-average type-token ratio (repetition) | Fake-news bodies used more all-caps words, fewer quotes and more lexical repetition (Horne & Adalı 2017). Punctuation and readability features were among the strongest single feature sets for fake-news detection, but generalised poorly across domains (Pérez-Rosas et al. 2018). Hence the website-held-out test matters. |
| **G4 emotion** (dropped by team decision; kept here for the report's "what we did not do") | Mean anger, fear, disgust, sadness, joy and surprise scores from a pre-trained emotion model (`j-hartmann/emotion-english-distilroberta-base`, Hartmann 2022), averaged over 512-token chunks on CPU | Anger and negative emotion were the most consistent markers of conspiracy language on Twitter (Fong et al. 2021). Emotion features improved false-news detection on news articles (Ghanem et al. 2019). **But** 9/11 reporting is emotional in general (course feedback), so we expected weak separation. |

**What we deliberately leave out:** LIWC (paid licence), transformer fine-tuning (out of scope per course feedback), and topic-model features (they overlap with n-grams).

### 4.3 M3: LLM
- **Question (team decision):** the **source question**, "would this article come from a conspiracy-oriented or a mainstream website?", the same target the LR is scored on. Whether the article itself *endorses* a conspiracy theory is examined separately in the error analysis.
- **Model:** `claude-haiku-5-5` at temperature 0, zero-shot, one fixed instruction, strict JSON answer `{"label": "conspiracy|mainstream", "confidence": 0–1, "reason": "≤25 words"}`.
- **Same texts:** the LLM receives exactly the cleaned texts that LR receives.
- **Prompt development:** at most 3 prompt variants, tried on the **validation** set only. All variants and their scores are reported. The chosen prompt runs once on the test set.
- **Consistency:** a second pass over the test set measures how often the LLM changes its answer. This addresses the team's question about LLM answers not being stable.
- **Ground truth:** the LLM's output is *never* treated as ground truth. Both approaches are scored against the dataset labels. This answers another of the team's questions.

## 5. Evaluation

- **Baselines:** majority class ("always mainstream") and stratified random guessing.
- **Metrics:**
  - F1, precision and recall for the conspiracy class
  - Macro-F1 and balanced accuracy
  - A confusion matrix per model
  - Precision-recall curves for LR
- **Plain-language explanations:** the notebook explains F1 in everyday words, for example "of the articles we flagged, how many were right, and of the ones we should have flagged, how many did we catch".
- **Uncertainty:** 95% bootstrap confidence intervals, resampling whole websites as well as single documents. A McNemar test compares LR and the LLM on the same test documents.
- **Interpretability:** the top n-gram weights per class, standardised coefficients for G2–G3, and word-level contributions for example documents.

## 6. Error analysis

1. **Build the disagreement sets:** wrong only for LR, wrong only for the LLM, and wrong for both.
2. **Code each text's stance** toward a 9/11 conspiracy claim: *endorses*, *rejects / debunks*, *reports neutrally*, or *doesn't mention one*.
   - Claude Sonnet 5.5 makes the first pass and gives a short justification for each text.
   - **Team members then check a random sample of 40** through the website. Agreement between humans and the model is reported.
3. **Sort each disagreement** into "the model is wrong" or "the label doesn't describe this text". This directly answers the course feedback's request to look at actual disagreements.
4. **Show worked examples** with short excerpts, LR word contributions and the LLM's reason.

## 7. Deliverables (every cycle)

1. **Code** in `src/fml/`, with tests in `tests/`. Running `fml run-all` reproduces every number.
2. **Results** as `results/*.json` and figures in `results/figures/`.
3. **An interactive, notebook-style site** on GitHub Pages, for the team. Sections:
   - Overview and how to read the site
   - Research questions
   - Data and labels
   - Cleaning
   - Split
   - TF-IDF and LR explained
   - Words-only results
   - Feature engineering (G1–G3)
   - Ablation
   - LLM pipeline
   - Head-to-head comparison
   - Error analysis
   - Limitations
   - AI-use log
   - Glossary

   **Every section** contains:
   - goal and hypothesis
   - real code cells (extracted from the repo, so they never drift from the code)
   - output (table or figure)
   - a plain-language explanation
   - "what could go wrong here"
   - a "technical infrastructure" box (files, functions, how it runs)

   Each section ends with a **feedback box** with three prompts: *what I didn't understand*, *what I want changed*, *other feedback*. A 👍 reaction means "I understand and agree with this section".
4. **A report draft** (`reports/report.md`, also built as `.docx` and `.pdf`) of 3,000–5,000 words. It follows the assignment's six points: goal, resources, approach (including what we chose *not* to do), evaluation, findings, and interpretation and limitations. It includes the AI-use section.
5. **A GitHub guide page** with step-by-step instructions: creating an account, choosing a username that doesn't reveal your real name, opening the site, commenting, reacting, and seeing replies.
6. **A process log** (`docs/ai-log.md`): what the agent produced, what was checked automatically, what the team changed, and what we learned.

The **presentation** (40% of the grade) and the **individual statements** are not part of this plan yet.

## 8. Agent architecture (hybrid: LangGraph + Claude Code)

```
            ┌──────────────── resume from checkpoint ────────────────┐
            ▼                                                        │
 load_context ─► plan ─► implement ─► run_pipeline ─► verify ─► build_site ─► review ─► decide
   (code)      (Claude)  (Claude Code,   (code: pytest,   (code:        (code:      (fresh    │
                          one task at    fml run-all)     hard gates)   render)     Claude,   │
                          a time)                                                    rubric)  │
                 ▲                                                                            │
                 └────────────── issues become new tasks ◄── score < 10 and rounds left ─────┤
                                                                                              ▼
                                                                              finalize (code: log, commit,
                                                                              PR, report build, pause/stop)
```

| Node | Kind | What it does |
|---|---|---|
| `load_context` | code | Reads `AGENTS.md`, `config/*.yaml`, the current `requirements/cycle-NN.md` and the last review |
| `plan` | Claude (Sonnet 5.5, `claude -p --json-schema`) | Turns requirements and open review issues into a task list (JSON) mapped to notebook sections |
| `implement` | Claude Code (Sonnet 5.5) | Does one task per call: edits `src/`, `tests/` and `content/` (section text). Turn cap per task |
| `run_pipeline` | code | Runs `pytest` and `fml run-all` and collects metrics and figures. *No LLM: if code can answer, code answers* |
| `verify` | code | Hard gates: tests pass; TF-IDF fit only on train; test set read once; split file unchanged; no article texts committed; privacy scan; the report's word count; every section has all required parts |
| `build_site` | code | Renders `site/` from `content/*.md`, real code snippets and results (deterministic templates, so the agent edits content, not HTML) |
| `review` | Claude (a **fresh** session that hasn't seen the work) | Scores each rubric criterion 0–10 with evidence and lists must-fix issues (JSON) |
| `decide` | code | Every criterion at 10 and the gates pass → finalize. Otherwise, if rounds and time remain → issues go back to `implement`. Otherwise → finalize with a loud "not at 10/10" summary |
| `finalize` | code | Appends to the AI-use log, builds the report files, commits, pushes the `cycle-NN` branch, opens or updates a PR |

**Runtime and payment (verified in the Claude Code docs, and tested in a cloud session on 16 Oct):**
- **Where it runs:** as a **routine** in Anthropic's cloud (claude.ai/code/routines), started by hand with **Run now** (no schedule). Each run is a cloud session with both repos cloned; nothing runs on a laptop. A nested `claude -p` call from a script inside a cloud session was tested and works, which is what the LangGraph nodes need.
- **Who pays:** cloud sessions draw on the Claude plan (and the cloud-session credit); Sonnet 5.5 does the agent work. Only the Haiku article classification uses an API key, stored as the cloud-environment variable `LLM_CLASSIFIER_API_KEY` (deliberately not `ANTHROPIC_API_KEY`, which Claude Code would pick up for its own billing). The cloud's network secrets cannot be used for this key: they are never attached to `api.anthropic.com`.
- **Short resumable steps:** a single command in a cloud session may run about 10 minutes, plus up to 30 more in the background. So `python -m agent run` works for at most `run_minutes` (default 20), checkpoints after every node, commits and pushes to `claude/cycle-NN`, and prints `STATUS: CONTINUE | DONE | PAUSED | FAILED`. The routine's prompt repeats the command while the status is `CONTINUE`.
- **Stop boundaries** (in `config/agent.yaml`): `run_minutes`, `task_timeout_minutes`, `max_turns_per_task`, `max_tasks_per_plan`, `max_gate_fix_attempts`, `max_review_rounds`, `max_runs_per_cycle`.
- **Resuming:** the LangGraph checkpoint (`.agent_state/cycle-NN.sqlite`) is committed to the cycle branch. A usage limit sets `PAUSED`; the next run (started by hand) continues from the last finished node.
- **Self-grading guard:** a gate fails the run if the agent edits its own harness, the rubric or the requirements (`agent/`, `sitegen/`, `scripts/`, `.github/`, `config/rubric.yaml`, `config/agent.yaml`, `requirements/`).
- **Plan B (documented, not built):** run the graph in GitHub Actions and start cloud routines through their API trigger for the Claude steps.

## 9. Repository structure

```
FML-fake-news/
├── AGENTS.md                 # rules for any coding agent (privacy, data, test lock, style, how to resume)
├── CLAUDE.md                 # one line: @AGENTS.md
├── README.md                 # public overview + how to run
├── .env.example              # variable names only, never values
├── langgraph.json            # registers agent/agent.py:graph
├── pyproject.toml            # pinned dependencies, `fml` CLI entry point
├── agent/
│   ├── agent.py              # builds and compiles the StateGraph; CLI: python -m agent --cycle N
│   ├── state.py              # typed state: cycle, tasks, round, scores, status, budgets
│   ├── nodes.py              # the nodes above
│   ├── tools.py              # claude_code(), run(), git/PR helpers, privacy scanner
│   ├── memory.py             # SQLite checkpointer + long-term decision log (docs/decisions.md)
│   └── prompts/              # plan.md, implement.md, review.md
├── config/
│   ├── project.yaml          # seed, split, model IDs, feature groups on/off
│   ├── agent.yaml            # stop boundaries
│   └── rubric.yaml           # the "10/10" definition (section 11)
├── requirements/cycle-01.md  # initial requirements (anonymised proposal + course feedback)
├── src/fml/                  # data.py, clean.py, split.py, features/, models.py, llm.py, evaluate.py, errors.py, cli.py
├── tests/                    # tests explain WHY a rule matters (leakage, test lock, per-1,000-word normalisation)
├── content/                  # one markdown file per notebook section (written by the agent)
├── sitegen/                  # deterministic notebook-style site generator (templates + build.py)
├── site/                     # generated output (not committed; built by the Pages workflow)
├── reports/                  # report.md → report.docx / report.pdf
├── results/                  # metrics JSON + figures (no article texts)
├── data/splits/              # doc_id-only split file (committed); raw data git-ignored
├── docs/                     # PROJECT_PLAN.md, ai-log.md, decisions.md
├── scripts/collect_feedback.py # comments + 👍 sign-offs -> feedback/cycle-NN.json + next requirements draft
└── .github/workflows/
    ├── ci.yml                # tests (no data, fake LLM) + site build on pull requests
    ├── pages.yml             # deploys site/ after merge to main
    └── collect-feedback.yml  # closes a cycle → drafts the next requirements file
```

## 10. Team feedback cycle

1. **The agent finishes cycle N** and opens a PR. You merge it, and the site updates on GitHub Pages.
2. **Teammates read the site** for a few days. They comment under each section through **giscus**, which stores comments as GitHub Discussions, one discussion per section per cycle. They react 👍 when they understand and agree.
   - Requirements: the repo is public, Discussions is on, and the giscus app is installed.
   - Every commenter needs a GitHub account (see the guide page).
3. **You run "Collect feedback"** in the Actions tab. The script pulls every comment and reaction for cycle N. Usernames are replaced by "Member A–E" using a mapping kept in a **repository secret**, so it never appears in files.
4. **Claude turns the comments into numbered requirements** in `requirements/cycle-(N+1).md`. Contradicting requests are listed side by side for you to decide, never blended.
5. **The draft arrives as a PR** for you to edit and merge. Merging starts cycle N+1.
6. **The cycle ends for good** when every section has 👍 from all five members and no open comments. The collect script reports this.

## 11. Definition of done: the "10/10" rubric

The reviewer scores each criterion 0–10 and must cite evidence (file, section or number). Done means **all criteria at 10 and all hard gates passing**.

1. **Question:** clear and feasible, with stated hypotheses.
2. **Resources:** data, models, APIs and tools described, including licence status.
3. **Approach:** explained, including what we chose *not* to do and why.
4. **Evaluation:** sound: one grouped split, TF-IDF fit on train only, test used once, baselines, confidence intervals.
5. **Results:** shown with figures and tables that match `results/*.json` exactly.
6. **Interpretation and limitations:** label-vs-content issue, label noise, small number of websites, LLM prior knowledge.
7. **Feature engineering:** answers "do the extra features help?" with ablation evidence and literature reasons.
8. **Error analysis:** uses real examples and separates model errors from label problems.
9. **AI-use documentation:** what was produced, what was checked or changed, what we learned.
10. **Teaching quality:** every section has hypothesis, code, explanation, "what could go wrong" and infrastructure. Every open team question is answered.
11. **Reproducibility:** one command, pinned dependencies, fixed seeds.
12. **Report:** 3,000–5,000 words, follows the assignment structure.
13. **Privacy:** no personal data anywhere.
14. **Course-feedback compliance:** each point in section 0 is visibly met.

## 12. Privacy rules (enforced by `verify`)

- **No real names** of students, staff or annotators, and no institution, course name or team number.
- **No e-mail addresses.**
- **No GitHub handles in files**, only "Member A–E".
- **Allowed:** names of published authors in citations.
- **Already applied:** the comments in the proposal's Word file are turned into anonymous "team questions" in `requirements/cycle-01.md`.

## 13. Cost estimate

- **Classification:** Haiku 5.5 costs $0.10 per million input tokens and $0.50 per million output tokens (prompts under 100k tokens). Validation + test are about 780 filtered articles; at ~1,200 tokens each plus the instruction, one pass is about 1.2M input tokens, roughly **$0.15**. Prompt variants plus the stability pass stay **under ~$3**, well within the 20 € of API credit. The agent records actual token counts.
- **Agent work:** Sonnet 5.5 in cloud sessions, on the Pro plan and the cloud-session credit. A run may pause at a usage limit and resume on the next run.
- **GitHub Actions:** free for public repos (site publishing and feedback collection only).

## 14. Risks

| Risk | Mitigation |
|---|---|
| Losing access to the data | The full LOCO download and the subset are in the private data repo with checksums and a rebuild script; a local copy exists on the owner's computer |
| Too few conspiracy websites per split | Report counts, per-website results and website-level bootstrap. Never reshuffle the split after seeing test results |
| The LLM has seen 9/11 conspiracy content in training | Stated as a limitation. It is the point of the comparison, not a flaw |
| Word-list features misread negation | Kept alongside n-grams. Limitation stated (Chen 2026) |
| The agent "games" the reviewer | The reviewer is a fresh session. Hard gates are code, not LLM opinions. Teammates are the final reviewers |
| Subscription usage runs out mid-run | `PAUSED`, resume from the checkpoint on the next run |

## 15. Open questions

None at the moment; new ones go to `docs/decisions.md` under "Needs a team decision".

## References

- Bozarth, L., & Budak, C. (2020). Toward a better performance evaluation framework for fake news classification. *ICWSM*. https://ojs.aaai.org/index.php/ICWSM/article/view/7279
- Chen, B. (2026). When certainty is an artifact: keyword lexicon blindness and the (mis)measurement of rhetorical stance. arXiv. https://arxiv.org/pdf/2606.26062
- Fong, A., Roozenbeek, J., Goldwert, D., Rathje, S., & van der Linden, S. (2021). The language of conspiracy: A psychological analysis of speech used by conspiracy theorists and their followers on Twitter. *Group Processes & Intergroup Relations, 24*(4). https://journals.sagepub.com/doi/10.1177/1368430220987596
- Ghanem, B., Rosso, P., & Rangel, F. (2019). An emotional analysis of false information in social media and news articles. arXiv:1908.09951. https://arxiv.org/abs/1908.09951
- Hartmann, J. (2022). Emotion English DistilRoBERTa-base. https://huggingface.co/j-hartmann/emotion-english-distilroberta-base
- Horne, B., & Adalı, S. (2017). This just in: Fake news packs a lot in title, uses simpler, repetitive content in text body, more similar to satire than real news. *ICWSM workshops*. https://arxiv.org/abs/1703.09398
- Islam, J., Xiao, L., & Mercer, R. (2020). A lexicon-based approach for detecting hedges in informal text. *LREC*. https://aclanthology.org/2020.lrec-1.380
- Marini, C., & Jezek, E. (2024). What to annotate: Retrieving lexical markers of conspiracy discourse from an Italian-English corpus of Telegram data. *ISA-20 Workshop*. https://aclanthology.org/2024.isa-1.6
- Miani, A., Hills, T., & Bangerter, A. (2022). LOCO: The 88-million-word language of conspiracy corpus. *Behavior Research Methods, 54*, 1794–1817. https://doi.org/10.3758/s13428-021-01698-z
- Mompelat, L., et al. (2022). How "Loco" is the LOCO corpus? Annotating the language of conspiracy theories. *LAW-XVI @ LREC*. https://aclanthology.org/2022.law-1.14
- Pérez-Rosas, V., Kleinberg, B., Lefevre, A., & Mihalcea, R. (2018). Automatic detection of fake news. *COLING*. https://arxiv.org/abs/1708.07104
- Rashkin, H., Choi, E., Jang, J. Y., Volkova, S., & Choi, Y. (2017). Truth of varying shades: Analyzing language in fake news and political fact-checking. *EMNLP*. https://aclanthology.org/D17-1317
- Wang, S., & Manning, C. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. *ACL*. https://aclanthology.org/P12-2018
