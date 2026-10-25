# A quadratic lens on Muon's polar update: when orthogonalization helps, and why few Newton-Schulz steps can beat exact

- **Topic ID:** muon-polar-update-theory
- **Status:** WAIT
- **Next review:** 2026-12-20
- **Course change:** none (Muon is taught in 03.4)
- **Candidates:** C-20261025-02

## Summary

On matrix quadratics PolarGD behaves like GD preconditioned by the iterate, so tiny singular values slow it; Muon's inexact Newton-Schulz treats them more like GD and can converge faster than exact orthogonalization. About a quarter of attention singular values in a Muon-trained transformer fall in that regime (authors).

## Evidence

- https://arxiv.org/abs/2610.30280 — Richtarik/Orvieto group; ICML 2026 workshop version; no code.

## What would change the decision

Main-venue acceptance or independent reproduction of the singular-value measurement; then a dated theory extension of 03.4 that can be checked with code/src/llmre/optim/muon.py on a toy quadratic.

## History

- 2026-10-25 — WAIT — credible theory bearing on taught code, but workshop-level evidence only — weekly/2026-W43.md — candidates C-20261025-02
