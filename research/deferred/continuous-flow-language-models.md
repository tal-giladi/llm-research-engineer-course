# Large Flow Language Models (Lafa): continuous flow LMs initialized from AR checkpoints, with self-speculative decoding

- **Topic ID:** continuous-flow-language-models
- **Status:** WAIT
- **Next review:** 2026-12-20
- **Course change:** none
- **Candidates:** C-20261021-02

## Summary

Blockwise continuous-flow LMs (causal across blocks, flow within) scaled to 8B on under 30B tokens from AR checkpoints; distillable to few-step flow maps; self-speculative decoding reported at 2.54x AR throughput (authors).

## Evidence

- https://arxiv.org/abs/2610.28559 — single paper; code at https://github.com/lafa-gh/large-flow-lms.

## What would change the decision

A course decision on non-autoregressive generation paradigms (discrete diffusion vs flow), plus independent reproduction or adoption.

## History

- 2026-10-25 — WAIT — a new paradigm signal, not a teachable settled mechanism; one day old — weekly/2026-W43.md — candidates C-20261021-02
