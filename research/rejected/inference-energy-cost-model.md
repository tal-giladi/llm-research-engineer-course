# Closed-form energy-cost model for LLM inference (SIMPLE)

- **Topic ID:** inference-energy-cost-model
- **Status:** REJECT
- **Next review:** on new evidence
- **Course change:** none
- **Candidates:** C-20261106-04

## Summary

A four-parameter, calibration-free per-token energy model derived from architectural compute and memory-traffic expressions, separating prefill, decode KV-cache cost and next-token compute.

## Evidence

- https://arxiv.org/abs/2611.03330 — 13 dense models on H100, R^2 0.969, 11.97% MAPE (authors); under journal review; no code.

## What would change the decision

Reopen if published with code and adopted by a benchmark or measurement standard (e.g. MLPerf Power) or a serving stack's cost tooling. The underlying reasoning (FLOPs vs bytes moved, prefill vs decode) is already taught in 08.2 and 08.6.

## History

- 2026-11-08 — REJECT — cost accounting built from FLOP/byte counts the course already teaches (08.2 roofline, 08.6 serving cost model); single paper, no code — weekly/2026-W45.md — candidates C-20261106-04
