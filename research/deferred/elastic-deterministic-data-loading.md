# Zephon: elastic determinism for online, stateful data-loading pipelines

- **Topic ID:** elastic-deterministic-data-loading
- **Status:** WAIT
- **Next review:** 2026-12-20
- **Course change:** none
- **Candidates:** C-20261019-02

## Summary

Accumulator/worker split, fixed logical lanes redistributed over ranks, and flush-and-replay checkpoints make an on-the-fly tokenize/pack/mix pipeline produce the same global batches regardless of GPU count. The authors show data-mixture ablations can silently measure GPU count instead of the intended change.

## Evidence

- https://pytorch.org/blog/bringing-elastically-deterministic-data-loading-to-torchtitan-with-zephon/ — PyTorch blog with TorchTitan integration (company claim).
- https://arxiv.org/abs/2610.03087 — paper; DCLM Core spread 0.82 -> 0.05 points across 8-64 GPUs (company claim).
- https://github.com/datologyai/zephon — Apache 2.0 library; integration lives in a DatologyAI fork, not upstream TorchTitan.

## What would change the decision

Adoption beyond DatologyAI (upstream TorchTitan, Megatron, another lab) or an independent reproduction of the GPU-count confound; it would then be an extension of 07.3 (reproducibility) or 11.x (data pipelines).

## History

- 2026-10-25 — WAIT — teachable and well measured, but single-company evidence and integration only in the authors' fork; one ADD per week — weekly/2026-W43.md — candidates C-20261019-02
