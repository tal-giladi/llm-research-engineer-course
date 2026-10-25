# FlexShard: whole-parameter and block-aligned sharding for distributed Muon and FP8 all-gathers (TorchTitan)

- **Topic ID:** flexshard-distributed-muon-sharding
- **Status:** WAIT
- **Next review:** 2026-12-20
- **Course change:** none (03.4 section 6 already covers layer-wise distributed Newton-Schulz)
- **Candidates:** C-20261025-01

## Summary

FSDP2 shards parameters row/column-wise, which conflicts with matrix-whole optimizer steps (Muon) and with blockwise quantization scales. FlexShard adds Owned (one rank owns a whole parameter) and BlockShard (tile-aligned) placements.

## Evidence

- https://pytorch.org/blog/torchtitan-update-new-sota-models-performance-at-scale-titanrl/ — up to 79% faster optimizer step for DeepSeek-V3 on 256 H100s vs FSDP-only Muon; up to 20% faster Llama-3 405B step with FP8 block-aligned all-gathers (company claim).

## What would change the decision

Independent use or measurement outside the TorchTitan team, or a course decision to extend 09.2 (ZeRO/FSDP) with optimizer-aware sharding; the delta over 03.4's layer-wise distributed optimizer would then be a 09.2 extension.

## History

- 2026-10-25 — WAIT — the TitanRL part of this candidate was added as 16.4; the FlexShard part adds little beyond 03.4's layer-wise distributed Newton-Schulz and is single-team so far — weekly/2026-W43.md — candidates C-20261025-01
