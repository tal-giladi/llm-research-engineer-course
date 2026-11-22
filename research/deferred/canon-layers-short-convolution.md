# Canon layers: residual short causal convolutions in attention/MLP blocks

- **Topic ID:** canon-layers-short-convolution
- **Status:** WAIT
- **Next review:** 2027-01-10
- **Course change:** none
- **Candidates:** C-20261121-01

## Summary

A Canon layer is a residual per-channel causal convolution over the current and three previous tokens, inserted at up to four points per block. Physics of LMs Part 4.2 tests it in 64 from-scratch runs at 1.3B-8B on 1-2T tokens and reports that LlamaCanon reaches vanilla Llama's MMLU with ~28% fewer tokens, and that with Canon, plain gated linear attention matches Gated DeltaNet. This is the short convolution that 12.5 mentions for Gated DeltaNet and that `gated_deltanet.py` omits.

## Evidence

- https://arxiv.org/abs/2611.24963 — Zeyuan Allen-Zhu (experiments at FAIR, Meta), single author; three sizes, LR sweeps not seeds, one data mix (authors' limits).
- https://github.com/facebookresearch/PhysicsLM4 — code, configs and a HF LlamaCanon model (not run).
- Short causal convolutions are already standard inside Mamba2 / Gated DeltaNet / KDA (PUBLICLY DOCUMENTED); two open MoE models reportedly add short convolutions in related positions (authors' statement, not verified).

## What would change the decision

An independent reproduction of the token-saving claim, or a released model or major open training stack adopting Canon-style convolutions in attention blocks by name. Then a dated extension of 12.5 ("the short causal convolution"), with a hand-worked 4-tap example and a new optional short-conv function plus a causality test (without changing `GatedDeltaNet`).

## History

- 2026-11-22 — WAIT — substantial single-author study with public code, but no independent reproduction or named adoption yet; this week's ADD went to the DSA foundation — weekly/2026-W47.md — candidates C-20261121-01
