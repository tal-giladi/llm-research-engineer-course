# Sharpening tax: RL post-training trades pass@k coverage for pass@1

- **Topic ID:** rl-sharpening-tax
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20261003-02

## Summary

Across 14 base/post-trained pairs on agentic benchmarks, base models with a light harness often beat post-trained models at large-k coverage; proposes the Sharpening Tax metric and posterior-tempered group sampling (PTGS) to reduce it during RL.

## Evidence

- https://arxiv.org/abs/2610.01509 — single paper (HF lists Meta affiliation, unconfirmed).
- The course already teaches the pass@k vs pass@1 gap (F.1) and mode-seeking reverse KL (17.4).

## What would change the decision

Independent reproduction of the agentic result or PTGS adoption in an RL stack; then a short extension of F.1 or 16.2 with the metric as an exercise.

## History

- 2026-10-04 — WAIT — broad single-paper evaluation of an effect the course partly covers; no reproduction — weekly/2026-W40.md — candidates C-20261003-02
