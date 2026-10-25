# EPIC: exact-KL prefix importance-ratio clipping for off-policy LLM RL

- **Topic ID:** rl-prefix-kl-trust-region-clipping
- **Status:** WAIT
- **Next review:** 2026-12-20
- **Course change:** none (GRPO fixes are taught in F.2)
- **Candidates:** C-20261022-01

## Summary

Treats generation as a finite-horizon MDP; the cumulative prefix likelihood ratio is the exact importance weight per improvement term, and an exact reverse-KL prefix term defines a PPO-style sequence-level clip. Claimed to avoid MoE routing replay.

## Evidence

- https://arxiv.org/abs/2610.29999 — Qwen3-30B-A3B-Base off-policy RL, 39.34% mean@8, +6.5 over GSPO (authors).
- https://github.com/LiChengxi666/EPIC — verl-based code with baselines.

## What would change the decision

Independent reproduction of the margin or adoption as a selectable loss in verl/TRL/OpenRLHF; then a row in F.2's comparison table, cross-linked from 16.4.

## History

- 2026-10-25 — WAIT — baseline-inclusive code but one self-reported paper — weekly/2026-W43.md — candidates C-20261022-01
