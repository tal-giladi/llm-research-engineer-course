# Context and memory compaction for long-horizon agents

- **Topic ID:** agent-context-compaction
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20261001-03, C-20261002-03, C-20261004-02, C-20261010-02, C-20261012-03, C-20261017-02, C-20261020-02, C-20261028-01, C-20261107-05

## Summary

Several independent approaches to agents whose history outgrows the context: persistent context graphs with attention-derived importance (ReCAP), an RL-trained policy that decides when and how to compact (AutoCompact, +9.2 points on SWE-bench Verified), and explicit belief-state tracking with stagnation detection (PoS).

## Evidence

- https://arxiv.org/abs/2609.40118 — ReCAP, single paper.
- https://arxiv.org/abs/2610.02163 — AutoCompact, single paper.
- https://arxiv.org/abs/2610.01415 — PoS, single paper, 12 backbone-benchmark combinations.
- Mem++ (https://arxiv.org/abs/2610.02002) and MemCalib/DolphinBench (daily 2026-09-28) show an active cluster.
- https://arxiv.org/abs/2610.11287 — REMORY: learned residual memory tokens for agent context compaction; fourth single-paper approach.
- https://arxiv.org/abs/2610.07625 — Stateless Language Agents: externalize state and rebuild context each call; single paper.
- https://arxiv.org/abs/2610.20927 — Source-Judged Compaction: drop summary entries the environment does not support; code released.
- https://arxiv.org/abs/2610.23266 — JointCoreBench/JointCore: exact-oracle benchmark for joint memory sufficiency; benchmark/baseline, not a deployment method (W43).
- https://arxiv.org/abs/2610.34073 — Memory Tax: measures the net value of agent memory against a retrieval baseline; evaluation framing, not a new mechanism; code not yet live (W44).
- https://arxiv.org/abs/2611.06500 and https://github.com/THU-Agent/Coca-cola — Coca-Cola: latent (non-text) compression of tool outputs with action-preserving training; 70-89% token reduction (authors); code, no reproduction.

## What would change the decision

Convergence on one mechanism with independent reproduction or adoption in a major agent framework, or documented use in a frontier agent product; then an extension of 19.1 on context management for long-running agents.

## History

- 2026-10-04 — WAIT — a real and growing problem area, but three different single-paper mechanisms with no convergence or reproduction yet — weekly/2026-W40.md — candidates C-20261001-03, C-20261002-03, C-20261004-02
- 2026-10-11 — WAIT — a fourth single-paper approach; still no convergence or adoption; next review date unchanged — weekly/2026-W41.md — candidates C-20261010-02
- 2026-10-18 — WAIT — two more distinct mechanisms (externalize vs compact in place, provenance checking); still no convergence or adoption; next review date unchanged — weekly/2026-W42.md — candidates C-20261012-03, C-20261017-02
- 2026-10-25 — WAIT — one more single paper (memory-subset sufficiency with exact oracles); useful benchmark framing but no convergence in the cluster; next review date unchanged — weekly/2026-W43.md — candidates C-20261020-02
- 2026-11-01 — WAIT — Memory Tax adds an evaluation angle (net value of memory vs a RAG baseline), not a converging mechanism; code not live; next review date unchanged — weekly/2026-W44.md — candidates C-20261028-01
- 2026-11-08 — WAIT — a latent-compression mechanism joins the cluster; single paper with code, no convergence; next review date unchanged — weekly/2026-W45.md — candidates C-20261107-05
