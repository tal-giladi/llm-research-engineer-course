# Convergence bounds for asynchronous GRPO under stale rollouts (GMC-GRPO)

- **Topic ID:** async-grpo-staleness-convergence
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20261002-02

## Summary

A convergence bound for GRPO-style updates with rollout staleness that separates estimator variance from bias, and group-mass capping as an alternative to ratio clipping, improving the delay term from O(eps^-4) to O(eps^-2).

## Evidence

- https://arxiv.org/abs/2610.01896 — single paper; Qwen3 experiments by the authors; no code confirmed.

## What would change the decision

Code plus adoption in an async RL stack (verl, OpenRLHF, TRL, Megatron RL) or an independent reproduction under large staleness; home would be F.1 section 2.4.

## History

- 2026-10-04 — WAIT — fits an existing section well, but single paper without code or adoption — weekly/2026-W40.md — candidates C-20261002-02
