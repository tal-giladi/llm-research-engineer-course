# Rounding-location alignment: keeping RoPE/SwiGLU/RMSNorm intermediates in FP32 to remove training-inference mismatch

- **Topic ID:** rl-mismatch-rounding-location-alignment
- **Status:** WAIT
- **Next review:** 2026-12-13
- **Course change:** none
- **Candidates:** C-20261026-01

## Summary

Identifies where inside the forward pass BF16 rounding differs between trainer and inference engine and aligns only those points (RoPE, SwiGLU, residual RMSNorm intermediates kept in FP32 until their BF16 outputs), reportedly preventing RL collapse in four Qwen-family models without changing kernels or the RL objective.

## Evidence

- https://arxiv.org/abs/2610.32951 — Ke, Cai, Wang (Oct 2026): authors' runs only; no code or venue.

## What would change the decision

Code release plus an independent reproduction, or adoption of the alignment in a major RL stack (verl, TRL, TorchTitan); then a short example in 16.4's causes section.

## History

- 2026-11-01 — WAIT — a specific, checkable mechanism for 16.4's mismatch, but one paper with no code or reproduction — weekly/2026-W44.md — candidates C-20261026-01
