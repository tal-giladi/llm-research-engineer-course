# Long-term personal-agent memory beyond flat facts (EvoMemState)

- **Topic ID:** agent-personal-memory-state
- **Status:** WAIT
- **Next review:** 2027-01-03
- **Course change:** none
- **Candidates:** C-20261108-02

## Summary

Instead of storing isolated facts retrieved by similarity (MemGPT, Mem0, A-MEM), keep a relational state (which facts are still valid and how they relate over time) and generate a request-specific reasoning state with a GRPO-trained policy on top of a frozen responder. Distinct from agent-context-compaction (trimming working context).

## Evidence

- https://arxiv.org/abs/2611.03743 and https://github.com/XMUDeepLIT/EvoMemState — four memory benchmarks (authors); code includes GRPO training; no reproduction.

## What would change the decision

Independent reproduction, a second group converging on relational/temporal memory state, or use in a widely used agent framework. Then a short section in the agents module.

## History

- 2026-11-08 — WAIT — single paper with code; agent memory mechanisms have not converged — weekly/2026-W45.md — candidates C-20261108-02
