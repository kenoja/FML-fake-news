---
title: What this project is and how to read this page
part: Overview
hypothesis: A small model trained on our own data can come surprisingly close to an LLM on full articles, and the interesting part is where each one goes wrong.
---
## Goal and hypothesis

We take about 2,600 full news and blog articles about the 9/11 attacks from the LOCO corpus. Each one is labelled by the **website** that published it: a <span class="con">conspiracy</span>-oriented site or a <span class="main">mainstream</span> one. Two approaches read exactly the same articles:

1. **Logistic regression** that we train ourselves. It turns words and short phrases into numbers (TF-IDF) and learns which ones point to which label. We first use words only, then add a few hand-made features and check whether they help.
2. **An LLM** (Claude Haiku, the cheapest Anthropic model) that has never seen our training data. It just reads each article and answers.

We do not try to make our model "win". We want to see where each approach works, where it fails, and why. One trap matters throughout: the label describes the website, not the article. A mainstream article *about* conspiracy theories, or a conspiracy-site article about something else, can look "wrong" for good reasons.

## Explanation

Every section below follows the same pattern, so you always know where to look:

- **Goal and hypothesis**: what we expected before running anything.
- **Code cells** (`In [n]`): the real code that ran, pulled from the repository when this page was built. Click the file name to see it on GitHub.
- **Output cells** (`Out [n]`): the table, number or chart that code produced.
- **Explanation**: what the result means, in plain words.
- **What could go wrong**: the mistakes this step could hide.
- **Technical infrastructure**: which files and commands are involved.

The project runs in cycles. An agent builds or improves the analysis overnight, the page is rebuilt, and then everyone has a few days to read it and leave feedback under each section. Your feedback becomes the requirements for the next cycle.

## What could go wrong

- Treating the website label as the "truth" about an article. It isn't; we discuss this in the error analysis.
- Reading accuracy alone. Most articles are mainstream, so always guessing "mainstream" already scores about 70% accuracy. We report measures that cannot be fooled that way.
- Trusting the page because an AI wrote it. Every number here comes from files in `results/` that the code produced, and the code is shown next to it. If something looks off, say so in the feedback.

## Technical infrastructure

- Analysis code: `src/fml/` (run everything with `fml run-all`).
- This page: generated from `content/sections/` by `python -m sitegen build` and published with GitHub Pages.
- The data (full article texts) lives in a separate private repository; this public one only holds code, results and short excerpts.
- How the overnight agent works: `AGENTS.md` and `agent/`.
