# ToolSearcher: RL-trained tool selection over large tool catalogs

- **Topic ID:** tool-selection-rl-toolsearcher
- **Status:** WAIT
- **Next review:** 2026-12-06
- **Course change:** none
- **Candidates:** C-20260929-01, C-20261023-02

## Summary

RL framework for choosing tools from a large, ambiguous catalog: category-constrained candidate narrowing, per-search-step RL events, and trajectory-aligned credit assignment.

## Evidence

- https://arxiv.org/abs/2609.30906 — NeurIPS 2026 (peer-reviewed); no code or concrete numbers in the abstract; no adoption found.
- https://arxiv.org/abs/2610.31483 — RL-trained Qwen3-8B selector reportedly beats Llama-4-Maverick on ToolBench tool selection; no code, no venue (W43).

## What would change the decision

Released code with concrete results, or adoption of RL-trained tool retrieval in an agent framework; natural home would be an extension of 18.1/18.2.

## History

- 2026-10-04 — WAIT — peer-reviewed but single paper with no code, numbers or adoption — weekly/2026-W40.md — candidates C-20260929-01
- 2026-10-25 — WAIT — a second single paper with a concrete number but no code; together the two papers still lack code plus independent results; next review date unchanged — weekly/2026-W43.md — candidates C-20261023-02
