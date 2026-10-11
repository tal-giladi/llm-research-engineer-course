# Elastic Expert Routing: smoothing the top-k exposure boundary in sparse MoE

- **Topic ID:** moe-stochastic-k-routing
- **Status:** WAIT
- **Next review:** 2026-12-06
- **Course change:** none
- **Candidates:** C-20261011-03

## Summary

Argues hard top-k routing starves experts just outside the top-k of gradient, and smooths the boundary with stochastic/elastic k.

## Evidence

- https://arxiv.org/abs/2610.11575 — single paper.

## What would change the decision

A second group reproducing the gradient-exposure framing, or a major MoE training stack (e.g. Megatron-Core) adopting stochastic-k routing; then an extension of 12.x (MoE).

## History

- 2026-10-11 — WAIT — single new paper on MoE routing, a crowded area with many unadopted proposals — weekly/2026-W41.md — candidates C-20261011-03
