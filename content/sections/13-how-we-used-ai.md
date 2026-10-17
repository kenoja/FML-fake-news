---
title: How we used AI
part: Use of AI tools
hypothesis: An AI agent can build most of this pipeline quickly, but only automatic gates, tests and human reading keep its mistakes and shortcuts out of the results.
---
## Goal and hypothesis

The assignment asks us to say which AI tools we used, what they produced, and how we checked and changed the output. We expected the agent to be fast and mostly right, and to make a few mistakes of a kind that a person would not notice without being told where to look. This section reports what happened. The raw log is `docs/ai-log.md`; choices and conflicts are in `docs/decisions.md`.

## Explanation

### What the AI produced

Two kinds of AI work are in this project.

1. **Setup (log entry "Setup, 16 Oct 2026").** Claude (Opus 5.5) in a chat session with the repo owner produced the project plan including the literature research, the agent harness, the site generator, the feedback collection script, the 9/11 data subset with checksums, and the repository rules in `AGENTS.md`.
2. **Cycle 1 (this page).** An agent (Claude Code) wrote the code in `src/fml/`, the tests, the result tables and these notebook sections, one task at a time. The classifier under study is Claude Haiku 5.5; a second model (Claude Sonnet 5.5) coded the stance of disagreement articles for the error analysis.

### What caught problems

The setup log says what was checked: "the data subset against the counts in the LOCO paper (3,146 = 939 + 2,207, matching exactly)", "the full archive rebuilt from its parts and compared by checksum", and "the agent graph's pause-and-resume behaviour with a test". It also records a finding that only appeared when the data were looked at: "about 17% of LOCO's 9/11 documents never mention 9/11 at all, in both classes. That only showed up once we looked at the data instead of trusting the seed label." That is why the off-topic filter exists (section 6).

Hard gates (automatic checks run after every task, tested in `tests/test_gates.py`) protect the rules that matter most:

- scoring one model on the test set with two different configurations fails;
- changing the frozen split file fails;
- private terms and e-mail addresses in public text fail.

<!-- cell: agent/tools.py#gate_test_access -->

Tests and the agent's own reading also caught real problems. These entries are quoted from `docs/decisions.md`:

- "Two exact duplicates exist (the project plan said none)": the plan was wrong, the data were checked.
- Near-duplicates: "63 near-duplicate pairs (cosine >= 0.9) were found, 36 of them across two splits: detected and logged, not removed." A possible leak between train and test, reported instead of hidden.
- "`claude-haiku-5-5` answers HTTP 400 (`temperature` is deprecated for this model)": the rule "temperature 0" could not be followed, so the agent wrote it down as a question for the team and measured the effect with a second pass.
- "Two interrupted-attempt processes ran the stage at the same time, which duplicated cache lines (1,059 dropped...)" and the test-access log got one entry twice. The agent repaired the cache reader and the log and said so.
- "Many replies ... were cut off at `max_output_tokens: 150`": the agent did not raise the limit after seeing test results, because that would be tuning on test.
- Topic filter: the removal was uneven across splits ("the test split loses 43 of 141 conspiracy documents"). The agent reported it and did not change the frozen split or the pattern.

The agent also reports against its own wishes: section 11 says that the stance codes come from Claude and are unchecked, so the H6 verdict is provisional.

### What the team still needs to verify

- **Stance codes.** Fill in `human_stance` in `results/tables/stance_check_sample.csv` (40 rows). Until then no agreement figure exists.
- **Numbers in prose.** Every number is copied from `results/`, but a person should spot-check several against `metrics.json`.
- **The "temperature" decision**, the **max-output-tokens decision** and the **licence of the texts** are open under "Needs a team decision" in `docs/decisions.md`.
- **Excerpts and privacy.** Read all sections once for article text or personal data.
- **Whether the explanations are understandable** for someone without an ML background (that is what the feedback threads are for).
- **Whether the epistemic word list** is sensible. It is a short hand-picked list, not a published one.

### Two pieces of code that make the AI parts auditable

An **API response cache** is a file that stores each answer the first time it is paid for, so that rerunning the pipeline reads the file instead of calling the model again. The cache is only trustworthy if its key changes whenever anything that could change the answer changes. The key below combines the prompt text, the model name, the settings and the pass number into one fingerprint (a **hash**: a short code that changes completely if any input changes). A document is looked up by its `doc_id` plus this key, never by its text.

<!-- cell: src/fml/llm.py#cache_key -->

The second piece is **stance coding**: we asked a second model (Sonnet 5.5) how each disagreement article treats conspiracy claims, without telling it the website. This function runs that step, reads from the cache first, and fails loudly when there is neither a cache entry nor an API key. Its answers are the unchecked part of the error analysis, so the code is short and the cache is public (ids, hashes and answers, no article text).

<!-- cell: src/fml/errors.py#code_stances -->

Last, the rule that turns a stance code into a verdict ("model wrong" or "label does not describe the text") is a plain table in code, so a teammate can check it without trusting the agent:

<!-- cell: src/fml/errors.py#classify_error -->

### Lessons

- Rules in files (`AGENTS.md`, `config/project.yaml`) plus automatic gates work better than telling an agent "be careful". The gates failed loudly in tests, which is what we wanted.
- An agent will report a conflict between a rule and reality (temperature 0) if it is told to write conflicts down. It must not resolve them silently.
- Interrupted runs leave traces (duplicate cache lines). Resumable steps need idempotent writes.
- AI output sounds equally confident when right and wrong. The checks that mattered were comparisons with outside facts (counts from the paper, checksums), not rereading the agent's text.
- The LLM used as the classifier and the LLM used as an assistant are different roles. We kept the classifier's prompts, cache and cost visible so its answers can be audited.

## What could go wrong

- Believing a test suite proves correctness. The tests encode the rules we thought of.
- The same model family writing the code and being evaluated by it may share blind spots. Human checks are part of the plan, not an extra.
- The log is only useful if humans add what they checked. `docs/ai-log.md` has the setup entry, the cycle-1 entries and a human-verification table (Members A to E). Every cell is `pending`, because no human check is recorded in the repository yet.

## Technical infrastructure

- `docs/ai-log.md` (what the AI produced and what people checked), `docs/decisions.md` (choices, conflicts, open questions).
- `agent/` and `tests/test_gates.py`: hard gates; `tests/test_sections.py`: checks that every section exists and every code and output reference resolves.
- Caches that make reruns reproducible without a key: `results/llm_cache.jsonl`, `results/stance_cache.jsonl` (document id, prompt hash and answer only, never text).
