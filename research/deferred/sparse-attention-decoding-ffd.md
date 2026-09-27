# Training-free sparse-attention decoding (FFD, "Faster Than Flash")

- **Topic ID:** sparse-attention-decoding-ffd
- **Status:** WAIT
- **Next review:** 2026-11-22
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

## History

- 2026-09-27 — WAIT — single unreproduced efficiency paper with no stack adoption; relevant to 08.4/12.5 if adopted — weekly/2026-W39.md — candidates C-20260926-02
