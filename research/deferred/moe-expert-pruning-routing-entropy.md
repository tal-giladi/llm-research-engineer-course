# SPARE: per-layer MoE expert-pruning budgets from routing entropy

- **Topic ID:** moe-expert-pruning-routing-entropy
- **Status:** WAIT
- **Next review:** 2026-12-20
- **Course change:** none
- **Candidates:** C-20261025-03

## Summary

Training-free, selector-agnostic allocation of how many experts to prune per layer: layers with more diffuse routing (higher entropy of normalized top-K gates) are pruned more. Beats uniform allocation on Mixtral, Qwen3 and DeepSeek families at equal budget (authors).

## Evidence

- https://arxiv.org/abs/2610.29662 — single paper, no code.

## What would change the decision

Independent reproduction or adoption in an MoE compression toolkit, plus a course decision on deployment compression; then an extension of 12.4.

## History

- 2026-10-25 — WAIT — single paper for a deployment topic the course does not yet cover — weekly/2026-W43.md — candidates C-20261025-03
