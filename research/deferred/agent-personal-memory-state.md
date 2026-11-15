# Long-term personal-agent memory beyond flat facts (EvoMemState)

- **Topic ID:** agent-personal-memory-state
- **Status:** WAIT
- **Next review:** 2027-01-03
- **Course change:** none
- **Candidates:** C-20261108-02, C-20261109-02

## Summary

Instead of storing isolated facts retrieved by similarity (MemGPT, Mem0, A-MEM), keep a relational state (which facts are still valid and how they relate over time) and generate a request-specific reasoning state with a GRPO-trained policy on top of a frozen responder. Distinct from agent-context-compaction (trimming working context).

## Evidence

- https://arxiv.org/abs/2611.03743 and https://github.com/XMUDeepLIT/EvoMemState — four memory benchmarks (authors); code includes GRPO training; no reproduction.
- https://www.anthropic.com/research/components-of-agent-memory — Anthropic: four-level non-weight memory hierarchy plus offline 'dreaming' consolidation, shipped in Managed Agents (company claims for all numbers). Related: Letta sleep-time compute https://arxiv.org/abs/2504.13171 ; Google's 'sleep framework' (updates weights; a different mechanism).

## What would change the decision

Independent reproduction, a second group converging on relational/temporal memory state, or use in a widely used agent framework. Then a short section in the agents module.

## History

- 2026-11-08 — WAIT — single paper with code; agent memory mechanisms have not converged — weekly/2026-W45.md — candidates C-20261108-02
- 2026-11-15 — WAIT — a production-backed memory taxonomy from a frontier lab, but company evaluations only and mechanisms across groups still differ; date unchanged — weekly/2026-W46.md — candidates C-20261109-02
