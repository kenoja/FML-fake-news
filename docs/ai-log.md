# How we used AI

The assignment asks us to document which AI tools we used, what for, and how we checked or adapted
their output. This log is the raw material for that part of the report. The agent appends an entry
after every cycle; team members add what they checked or changed by hand.

## Setup (16 Oct 2026)

- **Tool:** Claude (Opus 5.5) in a claude.ai session, working with the repo owner.
- **What it produced:** the project plan (`docs/PROJECT_PLAN.md`) including the literature research for
  the feature groups; the agent harness (`agent/`), the site generator (`sitegen/`), the feedback
  collection script and workflows; the 9/11 data subset and its checksums in the private data repo;
  the first requirements file and this repo's rules (`AGENTS.md`).
- **What was checked:** every product fact used in the design (pricing, authentication, cloud limits)
  against the official documentation; the data subset against the counts in the LOCO paper (3,146 = 939
  + 2,207, matching exactly); the full archive rebuilt from its parts and compared by checksum; the
  agent graph's pause-and-resume behaviour with a test; a nested Claude Code call inside a cloud session.
- **What the team decided (not the AI):** full articles, one cheap LLM asked the source question,
  feature groups (no emotion features), removing off-topic articles after the first run, public code with
  private data, running in Anthropic's cloud, pull requests per cycle.
- **What we learned:** about 17% of LOCO's 9/11 documents never mention 9/11 at all, in both classes.
  That only showed up once we looked at the data instead of trusting the seed label.
