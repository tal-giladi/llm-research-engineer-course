# Training-free sparse-attention decoding (FFD, "Faster Than Flash")

- **Topic ID:** sparse-attention-decoding-ffd
- **Status:** REJECT
- **Next review:** on new evidence
- **Course change:** none
- **Candidates:** C-20260926-02

## Summary

A fused decode kernel that skips KV blocks per step using a quantized content-aware scan and an
adaptive "top-δ" sparsity rule, claiming large kernel-level and end-to-end speedups at 256K
context without retraining.

## Evidence

- https://arxiv.org/abs/2609.00097 — single academic paper; speedups are author-reported; no independent reproduction and no serving-stack integration found.

## What would change the decision

Merge into (or an equivalent technique shipped by) vLLM, SGLang or TensorRT-LLM, or an independent
reproduction of the accuracy/speed trade-off. If adopted, the natural home is an extension of
lesson 08.4 (FlashAttention) or 12.5 (hybrid/sparse attention) on "when exact attention is needed
at decode time".

Decision 2026-11-22: re-judged against lesson 12.6 (DSA), as the 2026-11-08 scope decision asks. New since W39: an ICML 2026 poster listing (https://icml.cc/virtual/2026/poster/61344); no independent reproduction and no vLLM/SGLang/TensorRT-LLM integration found. 12.6 already teaches selection-before-attention and the bytes-read argument; a training-free decode variant is a narrow kernel result on top. Reopen on a merge into a major serving stack or an independent reproduction of the accuracy/speed trade-off.

## History

- 2026-09-27 — WAIT — single unreproduced efficiency paper with no stack adoption; relevant to 08.4/12.5 if adopted — weekly/2026-W39.md — candidates C-20260926-02
- 2026-11-22 — REJECT — re-judged against the new DSA foundation (12.6): training-free decode-time block skipping, now an ICML 2026 poster, but still one paper with author-reported speedups and no serving-stack integration after eight weeks — weekly/2026-W47.md — candidates none new
