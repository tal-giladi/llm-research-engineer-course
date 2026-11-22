# Bitwise-deterministic large-scale pretraining (Megatron-Core): fingerprint-and-trace debugging

- **Topic ID:** bitwise-deterministic-training
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20261009-05, C-20261025-01, C-20261118-01

## Summary

Making pretraining runs byte-for-byte reproducible at thousands of GPUs: ordered per-rank tensor fingerprints to find the first diverging step/rank/op, progressive tracing down to kernels, and deterministic kernel fixes, with the measured 'determinism tax'.

## Evidence

- https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/ — 2026-10-06; 800 deterministic steps at 2,432 GPUs on a trillion-parameter hybrid, overhead 17% -> 1.5% for one recipe (company claim); tracing workflow is Megatron-LM PR #7262, not confirmed merged.
- https://pytorch.org/blog/torchtitan-update-new-sota-models-performance-at-scale-titanrl/ — TorchTitan reports bitwise-matching eager vs graph training, deterministic pipeline parallelism, and batch-invariant trainer/generator log-probs (company claim). A second framework documenting determinism work, partly answering this topic's bar (W43).

## Evidence (added 2026-11-22)

- https://arxiv.org/abs/2611.21584 — CertiSum (Rhyner, Gao, de Lasa, Durvasula, Vijaykumar; U. Toronto / Vector Institute): FP32 accumulation of MXFP4 products is order-independent when every partial sum fits the 24-bit significand; outputs are certified by a bound and the rest recomputed exactly. 200-step Qwen3-8B-Base runs bit-identical across MI350X and GB200 and across DP/TP/CP layouts; +11.8-60.5% step latency over non-exact MXFP4 (authors). No code found. A hand-checkable teaching angle for 07.3: summation order matters only when a partial sum rounds.

## What would change the decision

PR #7262 merged, or a second framework/lab documenting a comparable determinism workflow. A debugging-methodology extension of 07.3 (reproducibility) would then fit, with a toy fingerprinting exercise.

## History

- 2026-10-11 — WAIT — useful engineering methodology but single-vendor and its central tool is an unmerged PR — weekly/2026-W41.md — candidates C-20261009-05
- 2026-10-25 — WAIT — TorchTitan now documents comparable determinism work (a second framework), and batch invariance for RL is taught in the new 16.4; the pretraining fingerprint-and-trace workflow itself is still single-vendor; next review date unchanged — weekly/2026-W43.md — candidates C-20261025-01
- 2026-11-22 — WAIT (evidence added, review date unchanged) — CertiSum (C-20261118-01, U. Toronto/Vector) is a third, independent group documenting bitwise-identical training, across AMD/NVIDIA and DP/TP/CP layouts in MXFP4; no code or framework integration yet; judge at the scheduled 2026-11-29 review — weekly/2026-W47.md — candidates C-20261118-01
