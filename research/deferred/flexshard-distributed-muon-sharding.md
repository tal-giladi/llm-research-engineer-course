# FlexShard: whole-parameter and block-aligned sharding for distributed Muon and FP8 all-gathers (TorchTitan)

- **Topic ID:** flexshard-distributed-muon-sharding
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none (03.4 section 6 already covers layer-wise distributed Newton-Schulz)
- **Candidates:** C-20261025-01, C-20261113-01

## Summary

FSDP2 shards parameters row/column-wise, which conflicts with matrix-whole optimizer steps (Muon) and with blockwise quantization scales. FlexShard adds Owned (one rank owns a whole parameter) and BlockShard (tile-aligned) placements.

## Evidence

- https://pytorch.org/blog/torchtitan-update-new-sota-models-performance-at-scale-titanrl/ — up to 79% faster optimizer step for DeepSeek-V3 on 256 H100s vs FSDP-only Muon; up to 20% faster Llama-3 405B step with FP8 block-aligned all-gathers (company claim).
- https://pytorch.org/blog/training-a-1t-parameter-moe-on-eu-public-infrastructure-with-pytorch-and-torchtitan/ — EUROPA consortium (Domyn, Fraunhofer IAIS/IIS) pretrains a 951.9B/32.8B-active MoE with Muon on 1,536 GH200 using FlexShard whole-expert ownership; upstreamed fixes https://github.com/pytorch/torchtitan/pull/5352 and https://github.com/pytorch/pytorch/pull/202343 (merged; consortium claims for throughput).

## What would change the decision

Independent use or measurement outside the TorchTitan team, or a course decision to extend 09.2 (ZeRO/FSDP) with optimizer-aware sharding; the delta over 03.4's layer-wise distributed optimizer would then be a 09.2 extension.

Update 2026-11-15: the stated trigger (independent use outside the TorchTitan team) is met. The topic now qualifies as a 09.2 extension ('optimizer-aware sharding and memory at 1T scale': Muon vs Adam state bytes, whole-parameter ownership for Newton-Schulz, activation-checkpoint offload arithmetic, per-stream allocator pools as a common mistake). Waits only for the budget.

## History

- 2026-10-25 — WAIT — the TitanRL part of this candidate was added as 16.4; the FlexShard part adds little beyond 03.4's layer-wise distributed Newton-Schulz and is single-team so far — weekly/2026-W43.md — candidates C-20261025-01
- 2026-11-15 — WAIT — trigger met (independent 1T-scale use with merged upstream fixes); qualifies as a 09.2 extension; budget only — weekly/2026-W46.md — candidates C-20261113-01
- 2026-11-22 — WAIT — still qualifies as a 09.2 extension; this week's one ADD went to the author-approved DSA foundation (queue #1); first in line next week — weekly/2026-W47.md — candidates none new
