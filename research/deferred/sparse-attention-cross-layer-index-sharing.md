# Prism Attention: proposal-refined cross-layer index sharing for sparse decoding attention

- **Topic ID:** sparse-attention-cross-layer-index-sharing
- **Status:** WAIT
- **Next review:** 2026-12-20
- **Course change:** none
- **Candidates:** C-20261019-01

## Summary

Shared top-k indices across layers drift at the edges of the selected region while a stable core persists; Prism proposes candidates once and refines per layer at the edges. 3.4x end-to-end decode speedup over FlashAttention at 128K (authors).

## Evidence

- https://arxiv.org/abs/2610.22985 — NeurIPS 2026 (listing at https://neurips.cc/virtual/2026/poster/153474 per the daily run); no code.

## What would change the decision

Code plus independent reproduction, or the course adding a foundational sparse-attention lesson (DeepSeek-style indexers) that this, DoSA, PISA or FFD would extend. The cluster (four WAIT topics) is a structural-gap question for the author.

## History

- 2026-10-25 — WAIT — peer-reviewed but single paper with no code; the course has no sparse-attention foundation lesson to extend — weekly/2026-W43.md — candidates C-20261019-01
