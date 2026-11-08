# Interpreting MoE experts with sparse autoencoders (exact per-latent expert contributions)

- **Topic ID:** moe-sae-expert-interpretability
- **Status:** WAIT
- **Next review:** 2027-01-03
- **Course change:** none
- **Candidates:** C-20261107-03

## Summary

An SAE trained on an MoE layer's router-weighted output lets each latent's pre-activation be split exactly across experts, measuring what each expert contributes rather than how often it is selected. On Qwen3-30B-A3B most latents have one or two dominant experts and specialization is finer than domain labels.

## Evidence

- https://arxiv.org/abs/2611.08089 — ByteDance Seed authors (code org); code https://github.com/ByteDance-Seed/SAE-interprets-MoE (Apache-2.0) and SAE weights on Hugging Face; qualitative findings, one model.

## What would change the decision

Use of the decomposition by another group or on a second model family. Then a research-connection paragraph in the MoE lesson (12.4) or the interpretability material.

## History

- 2026-11-08 — WAIT — single paper with code and weights, qualitative results on one model — weekly/2026-W45.md — candidates C-20261107-03
