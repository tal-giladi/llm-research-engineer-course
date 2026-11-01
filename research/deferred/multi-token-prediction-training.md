# Multi-token prediction (MTP) training objective and its gradient interference (Soft-MTP)

- **Topic ID:** multi-token-prediction-training
- **Status:** WAIT
- **Next review:** 2026-12-20
- **Course change:** none
- **Candidates:** C-20261031-04

## Summary

MTP adds heads that predict several future tokens (used by DeepSeek-V3 and as native speculative-decoding drafters, e.g. in Nemotron 3.5 and vLLM's MTP drafter). Soft-MTP claims a gradient-interference flaw in standard MTP training and a fix, tested up to 1.2B parameters.

## Evidence

- https://arxiv.org/abs/2610.38331 — Soft-MTP (single author, code released, up to 1.2B).
- vLLM lists MTP among its speculative-decoding drafters (docs.vllm.ai speculative_decoding page).

## What would change the decision

MTP itself may deserve a foundation (training objective + drafter for 08.6) once its use is documented across several open model families; Soft-MTP specifically needs independent reproduction at larger scale.

## History

- 2026-11-01 — WAIT — MTP is relevant to 08.6 and the DeepSeek-V3 guide but the candidate is a single small-scale paper; base MTP to be considered as its own topic — weekly/2026-W44.md — candidates C-20261031-04
