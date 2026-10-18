# Tree attention / shared-prefix packing for GRPO rollouts (TRL AsyncGRPO)

- **Topic ID:** grpo-tree-attention-prefix-packing
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20261017-01

## Summary

Packs a group's rollouts that share a prompt prefix into one tree-masked sequence so the prefix is processed once during scoring and training.

## Evidence

- https://github.com/huggingface/trl/releases/tag/v1.16.0 and https://github.com/huggingface/trl/pull/7328 — merged and shipped in TRL; maintainer benchmarks only.

## What would change the decision

Independent benchmarks or adoption in a second RL stack (verl, OpenRLHF), or TRL making it a default; then an extension of 16.1 with a toy tree-mask implementation.

## History

- 2026-10-18 — WAIT — merged in a major library but a single source and days old — weekly/2026-W42.md — candidates C-20261017-01
