# Is 2:4 sparsity still worth it? Semi-structured sparsity vs lower precision for LLM inference

- **Topic ID:** semi-structured-sparsity-vs-quantization
- **Status:** WAIT
- **Next review:** 2026-12-20
- **Course change:** none
- **Candidates:** C-20261022-02

## Summary

At a fixed bit-width and tensor-core tier, adding 2:4 sparsity is worse than dropping precision whenever a lower precision exists; at the lowest precision 2:4 still gives 1.3-1.7x on linear layers, but needs retraining.

## Evidence

- https://arxiv.org/abs/2610.28947 — single author, hardware-measured on three GPU generations; no code.

## What would change the decision

A course-author decision to add post-training quantization (see ternary-post-training-quantization), plus independent confirmation; then a worked trade-off example in that lesson.

## History

- 2026-10-25 — WAIT — single paper for a foundation (PTQ) the course does not have — weekly/2026-W43.md — candidates C-20261022-02
