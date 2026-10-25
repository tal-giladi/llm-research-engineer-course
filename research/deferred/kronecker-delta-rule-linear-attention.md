# Kronecker Delta Rule (KronLSTM): rank-1 Kronecker-structured memory updates for long-context linear attention

- **Topic ID:** kronecker-delta-rule-linear-attention
- **Status:** WAIT
- **Next review:** 2026-12-20
- **Course change:** none (Gated DeltaNet is taught in 12.5)
- **Candidates:** C-20261021-01

## Summary

An xLSTM-family update that corrects memory along one direction in matrix space per step, claimed to interfere less with stored associations than the delta rule and to keep the state norm bounded without extra normalization.

## Evidence

- https://arxiv.org/abs/2610.26834 — NX-AI (Hochreiter group); stability proof and benchmarks by the authors.
- https://github.com/NX-AI/KronLSTM — Triton kernels, fla-compatible; HF checkpoints.

## What would change the decision

Independent benchmark reproduction, or adoption in flash-linear-attention mainline or a released production model; then a dated extension of 12.5.

## History

- 2026-10-25 — WAIT — credible group with code and checkpoints, but two days old with no independent validation — weekly/2026-W43.md — candidates C-20261021-01
