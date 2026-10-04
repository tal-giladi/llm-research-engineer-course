# PISA: hierarchical block-sparse attention with O(N log N) scoring

- **Topic ID:** block-sparse-attention-pisa
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20260928-01

## Summary

Replaces exhaustive query-block/key-block scoring in block-sparse attention with a coarse-to-fine pyramid top-K selection over O(log N) levels, with fused Triton kernels for training and inference.

## Evidence

- https://arxiv.org/abs/2609.31093 — single paper; no code confirmed, no reproduction, no serving-stack adoption.

## What would change the decision

Public code plus an independent reproduction, or adoption in a serving or training stack. Learned/top-k block-sparse attention in general (also seen in the Qwen3.8-Next report and TLX block-sparse variant) would be reviewed together with deferred/sparse-attention-decoding-ffd.md as one possible 12.5 extension.

## History

- 2026-10-04 — WAIT — single unreproduced paper in a crowded sparse-attention space — weekly/2026-W40.md — candidates C-20260928-01
