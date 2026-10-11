# Attic-KV: query-aware 'rehearse what will be read' KV-cache compression

- **Topic ID:** query-aware-kv-cache-compression
- **Status:** REJECT
- **Next review:** on new evidence
- **Course change:** none
- **Candidates:** C-20261011-02

## Summary

KV-cache compression under tight budgets that rehearses likely future queries to choose which entries to keep.

## Evidence

- https://arxiv.org/abs/2610.12133 — single paper, no code. Same-day VFold (arXiv 2610.12338, C-class) is another unrelated angle.

## What would change the decision

Adoption by vLLM/SGLang or an independent comparison showing it beats existing eviction methods; or a course decision to add a KV-cache-compression lesson, in which case a survey-level consolidation would be the source, not a single method.

## History

- 2026-10-11 — REJECT — one more single-paper KV-eviction heuristic in a crowded field with no consolidating result; no underlying concept the course lacks beyond what 12.3 teaches about cache size — weekly/2026-W41.md — candidates C-20261011-02
