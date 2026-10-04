# Delta-Matching for native FP8 attention training

- **Topic ID:** native-fp8-training-delta-matching
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20260930-03

## Summary

Native block-scaled FP8 attention training degrades at scale because forward/backward numerical mismatch produces a stale softmax-backward delta that breaks the zero-row-sum invariant; Delta-Matching restores it without architecture changes.

## Evidence

- https://arxiv.org/abs/2609.37852 — single paper, tested to 5.29B parameters, code promised but not released.

## What would change the decision

Released code and results at larger scale, or adoption in Megatron-LM/TorchTitan/Transformer Engine; home would be an extension of 08.3 (mixed precision).

## History

- 2026-10-04 — WAIT — credible mechanism but single paper, mid-scale only, no code — weekly/2026-W40.md — candidates C-20260930-03
