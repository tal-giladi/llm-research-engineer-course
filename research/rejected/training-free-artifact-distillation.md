# Artifact distillation: training-free transfer through a teacher-written procedure bank

- **Topic ID:** training-free-artifact-distillation
- **Status:** REJECT
- **Next review:** on new evidence
- **Course change:** none
- **Candidates:** C-20261102-04

## Summary

A frontier teacher writes natural-language problem-solving procedures into a bank that a smaller model reads in context at inference; no gradient update. Reported: Haiku 4.5 with an Opus-written bank beats unaided Sonnet 4.5 on one LiveBench task at ~60% of the cost.

## Evidence

- https://arxiv.org/abs/2610.37680 — two authors, four LiveBench tasks, no code, no venue.

## What would change the decision

Reopen if independently reproduced and adopted as a named technique in agent or RAG frameworks; it would then belong with retrieval/prompting material, not distillation.

## History

- 2026-11-08 — REJECT — a prompting/retrieval pattern rather than an LLM-training mechanism; single paper, no code; not a good use of curriculum space — weekly/2026-W45.md — candidates C-20261102-04
