# Calibration-free channel selection for MoE decoding

- **Topic ID:** moe-decoding-channel-selection
- **Status:** WAIT
- **Next review:** 2026-12-13
- **Course change:** none
- **Candidates:** C-20261028-03

## Summary

Per-token, per-channel selection of which expert weights to stream during memory-bound MoE decoding, without calibration data, with claimed speedups relative to vLLM. Can now be judged against 08.6's memory-bound decode model.

## Evidence

- https://arxiv.org/abs/2610.37386 — single paper with authors' code (Oct 2026); no reproduction or venue.

## What would change the decision

Independent reproduction or integration into vLLM/SGLang/TensorRT-LLM; then a possible extension combining 12.4 (MoE) and 08.6 (decode cost).

## History

- 2026-11-01 — WAIT — single paper; the serving foundation now exists to place it, but no reproduction or adoption — weekly/2026-W44.md — candidates C-20261028-03
