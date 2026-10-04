# Quantizing the recurrent state of linear-attention / Gated DeltaNet layers (LeapQuant, STEPQuant)

- **Topic ID:** linear-attention-state-quantization
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20260930-01, C-20260930-02

## Summary

Two independent same-day papers quantize the recurrent state taught in 12.5: LeapQuant (per-window requantization plus high-precision compensator tokens, 8-bit, training-free, 1.47x end-to-end on Blackwell) and STEPQuant (error-persistence-aware bit allocation over key rows and value columns, ~6-bit at FP32 accuracy, up to 68.7% serving-memory reduction).

## Evidence

- https://arxiv.org/abs/2609.38166 — LeapQuant, no code yet.
- https://arxiv.org/abs/2609.38169 — STEPQuant, no code yet.

## What would change the decision

Public code for either, or recurrent-state quantization landing in vLLM/SGLang or fla kernels; then an extension of 12.5 comparing the two approaches with a from-scratch state-quantization exercise.

## History

- 2026-10-04 — WAIT — two independent groups on the same problem is a real signal, but both are days old with no code or adoption — weekly/2026-W40.md — candidates C-20260930-01, C-20260930-02
