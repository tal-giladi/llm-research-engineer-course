# Prism Attention: proposal-refined cross-layer index sharing for sparse decoding attention

- **Topic ID:** sparse-attention-cross-layer-index-sharing
- **Status:** WAIT
- **Next review:** 2026-12-20
- **Course change:** none
- **Candidates:** C-20261019-01, C-20261106-01

## Summary

Shared top-k indices across layers drift at the edges of the selected region while a stable core persists; Prism proposes candidates once and refines per layer at the edges. 3.4x end-to-end decode speedup over FlashAttention at 128K (authors).

## Evidence

- https://arxiv.org/abs/2610.22985 — NeurIPS 2026 (listing at https://neurips.cc/virtual/2026/poster/153474 per the daily run); no code.
- https://arxiv.org/abs/2611.03768 and https://github.com/ArjunKrishnak/dsa-multi-layer-indexer — MLI: reuse the previous layer's DSA indexer selection and run the indexer concurrently with attention; SGLang patch, evaluated on DeepSeek-V3.2 and GLM-5; NeurIPS 2026 LCFM workshop; 1 star.
- https://huggingface.co/zai-org/GLM-5 and https://arxiv.org/abs/2602.15763 — GLM-5 (Zhipu AI, Feb 2026, open weights) states it adopts DeepSeek Sparse Attention (lightning indexer + top-k): a second major open-weight lab using the same mechanism (company report).
- Same week, C class: CISA (arXiv 2611.03200, NVIDIA authors) retrofits chunk-indexed sparse attention into Nemotron 3 models (company numbers).

## What would change the decision

Code plus independent reproduction, or the course adding a foundational sparse-attention lesson (DeepSeek-style indexers) that this, DoSA, PISA or FFD would extend. The cluster (four WAIT topics) is a structural-gap question for the author.

Update 2026-11-08: question for the author in weekly/2026-W45.md. The 2026-10-28 decision says reopen if 'one mechanism becomes standard in major open models with independent validation'. DSA (lightning indexer + top-k) now ships in DeepSeek-V3.2 and GLM-5 (two labs, both company reports; MLI benchmarks both). If the author reopens, a DSA foundation lesson after 12.5 would let this file, DoSA, PISA and FFD be re-judged against it.

## History

- 2026-10-25 — WAIT — peer-reviewed but single paper with no code; the course has no sparse-attention foundation lesson to extend — weekly/2026-W43.md — candidates C-20261019-01
- 2026-11-08 — WAIT — MLI is a cross-layer index-reuse method like Prism, but the bigger news is that DSA is used by two major open-weight labs (DeepSeek-V3.2, GLM-5), which may meet the reopening condition of the author's 2026-10-28 sparse-attention decision; that is the author's call, so this topic is not REJECTed under that decision while the question is open; next review date unchanged — weekly/2026-W45.md — candidates C-20261106-01
