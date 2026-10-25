# Bitwise-deterministic large-scale pretraining (Megatron-Core): fingerprint-and-trace debugging

- **Topic ID:** bitwise-deterministic-training
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20261009-05, C-20261025-01

## Summary

Making pretraining runs byte-for-byte reproducible at thousands of GPUs: ordered per-rank tensor fingerprints to find the first diverging step/rank/op, progressive tracing down to kernels, and deterministic kernel fixes, with the measured 'determinism tax'.

## Evidence

- https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/ — 2026-10-06; 800 deterministic steps at 2,432 GPUs on a trillion-parameter hybrid, overhead 17% -> 1.5% for one recipe (company claim); tracing workflow is Megatron-LM PR #7262, not confirmed merged.
- https://pytorch.org/blog/torchtitan-update-new-sota-models-performance-at-scale-titanrl/ — TorchTitan reports bitwise-matching eager vs graph training, deterministic pipeline parallelism, and batch-invariant trainer/generator log-probs (company claim). A second framework documenting determinism work, partly answering this topic's bar (W43).

## What would change the decision

PR #7262 merged, or a second framework/lab documenting a comparable determinism workflow. A debugging-methodology extension of 07.3 (reproducibility) would then fit, with a toy fingerprinting exercise.

## History

- 2026-10-11 — WAIT — useful engineering methodology but single-vendor and its central tool is an unmerged PR — weekly/2026-W41.md — candidates C-20261009-05
- 2026-10-25 — WAIT — TorchTitan now documents comparable determinism work (a second framework), and batch invariance for RL is taught in the new 16.4; the pretraining fingerprint-and-trace workflow itself is still single-vendor; next review date unchanged — weekly/2026-W43.md — candidates C-20261025-01
