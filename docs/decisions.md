# Decisions

Choices, conflicts between sources and open questions. Newest at the bottom of each list.
Agents: add an entry whenever you choose between alternatives or resolve a conflict (see AGENTS.md).

## Decided

- **2026-10-16 · Course feedback overrules the proposal.** Full articles instead of headlines; logistic
  regression vs one cheap LLM; no fine-tuning; a few extra feature groups compared with words only.
- **2026-10-16 · LLM question.** Ask the source question (conspiracy-oriented vs mainstream website),
  the same target the LR is scored on. Content stance is examined in the error analysis instead.
- **2026-10-16 · Feature groups.** n-grams (1–3), epistemic markers, style. Emotion scores dropped by
  the team (cost and the course's warning that emotional language is not conspiracy language).
- **2026-10-16 · Off-topic documents.** First words-only run on all 3,146 documents; then remove documents
  that never mention 9/11 and explain the effect. Pattern in `config/project.yaml`; it includes
  "pentagon", which may also match unrelated defence articles — check the removed and kept examples.
- **2026-10-16 · JFK transfer test** out of scope for now.
- **2026-10-16 · Data location.** Full texts only in the private repo `FML-data`; this repo is public.
- **2026-10-16 · Licence.** No licence found in the LOCO paper; OSF page unreachable (maintenance).
  Treat texts as not redistributable until confirmed.

## Needs a team decision

- none yet
