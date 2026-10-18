# Quantizing the recurrent state of linear-attention / Gated DeltaNet layers (LeapQuant, STEPQuant)

- **Topic ID:** linear-attention-state-quantization
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20260930-01, C-20260930-02, C-20261008-02

## Summary

Two independent same-day papers quantize the recurrent state taught in 12.5: LeapQuant (per-window requantization plus high-precision compensator tokens, 8-bit, training-free, 1.47x end-to-end on Blackwell) and STEPQuant (error-persistence-aware bit allocation over key rows and value columns, ~6-bit at FP32 accuracy, up to 68.7% serving-memory reduction).

## Evidence

- https://arxiv.org/abs/2609.38166 — LeapQuant, no code yet.
- https://arxiv.org/abs/2609.38169 — STEPQuant, no code yet.
- https://arxiv.org/abs/2609.38169 — STEPQuant (C-20261008-02) reported as integrated into SGLang; integration not yet confirmed against SGLang's upstream repo/release notes.
- https://arxiv.org/abs/2610.13778 — ReplaySSM (separate topic replayssm-ssm-input-caching) keeps a high-precision checkpoint for the recurrence and a low-precision copy for reads, a serving-side answer to the same error-compounding problem.

## What would change the decision

Public code for either, or recurrent-state quantization landing in vLLM/SGLang or fla kernels; then an extension of 12.5 comparing the two approaches with a from-scratch state-quantization exercise.

## History

- 2026-10-04 — WAIT — two independent groups on the same problem is a real signal, but both are days old with no code or adoption — weekly/2026-W40.md — candidates C-20260930-01, C-20260930-02
- 2026-10-11 — WAIT — claimed SGLang integration of STEPQuant is not yet verified upstream; next review date unchanged — weekly/2026-W41.md — candidates C-20261008-02
- 2026-10-18 — WAIT — related evidence from ReplaySSM (mixed read/recurrence precision), recorded under its own topic; next review date unchanged — weekly/2026-W42.md — candidates C-20261014-01
