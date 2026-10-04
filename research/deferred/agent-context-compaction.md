# Context and memory compaction for long-horizon agents

- **Topic ID:** agent-context-compaction
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20261001-03, C-20261002-03, C-20261004-02

## Summary

Several independent approaches to agents whose history outgrows the context: persistent context graphs with attention-derived importance (ReCAP), an RL-trained policy that decides when and how to compact (AutoCompact, +9.2 points on SWE-bench Verified), and explicit belief-state tracking with stagnation detection (PoS).

## Evidence

- https://arxiv.org/abs/2609.40118 — ReCAP, single paper.
- https://arxiv.org/abs/2610.02163 — AutoCompact, single paper.
- https://arxiv.org/abs/2610.01415 — PoS, single paper, 12 backbone-benchmark combinations.
- Mem++ (https://arxiv.org/abs/2610.02002) and MemCalib/DolphinBench (daily 2026-09-28) show an active cluster.

## What would change the decision

Convergence on one mechanism with independent reproduction or adoption in a major agent framework, or documented use in a frontier agent product; then an extension of 19.1 on context management for long-running agents.

## History

- 2026-10-04 — WAIT — a real and growing problem area, but three different single-paper mechanisms with no convergence or reproduction yet — weekly/2026-W40.md — candidates C-20261001-03, C-20261002-03, C-20261004-02
