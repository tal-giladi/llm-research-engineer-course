# Stochastic rounding of FP32 master weights to BF16 in low-learning-rate RLVR ("weight sticking")

- **Topic ID:** bf16-stochastic-rounding-rl
- **Status:** WAIT
- **Next review:** 2027-01-10
- **Course change:** none
- **Candidates:** C-20261113-02

## Summary

At RLVR's small learning rates, round-to-nearest from FP32 masters to the BF16 working copy swallows updates smaller than half a BF16 ulp, so BF16 trains slower than FP16; stochastic rounding makes the rounded update unbiased and closes the gap (authors' claim).

## Evidence

- https://arxiv.org/abs/2611.13545 — abstract-level claims, no code.

## What would change the decision

Code or an independent confirmation. Then a worked "weight sticking" example and a CPU demo of round-to-nearest vs stochastic rounding in 08.3, cross-referenced from 16.4.

## History

- 2026-11-15 — WAIT — a clean numerics point, but a single paper with no code or numbers in the abstract — weekly/2026-W46.md — candidates C-20261113-02
