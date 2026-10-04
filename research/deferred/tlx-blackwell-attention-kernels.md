# Blackwell-era attention kernel engineering (TLX Jagged Flash Attention, FlashAttention-4 techniques)

- **Topic ID:** tlx-blackwell-attention-kernels
- **Status:** WAIT
- **Next review:** 2026-12-06
- **Course change:** none
- **Candidates:** C-20261003-01

## Summary

Meta rewrote its production jagged (variable-length) FlashAttention kernel for B200 in TLX, a Triton extension exposing warp specialization, explicit SMEM/TMEM allocation, barriers, async TMA/MMA and Cluster Launch Control; reports +13% forward and +50% backward vs FlashAttention-4 on its ads shapes, ~87% forward / +17% backward on a dense LLM-style shape, in ~3.2K vs ~10K lines.

## Evidence

- https://pytorch.org/blog/optimizing-jagged-flash-attention-with-tlx-the-road-toward-sota-fa4-on-blackwell/ — official PyTorch blog, 2026-10-01; production use in Meta's GEM ads model (not an LLM); benchmarks are company-run (company claim) but both kernels are open source.
- https://github.com/facebookresearch/ads_model_kernel_library/tree/main/tlx_jfa — code.
- https://arxiv.org/abs/2603.05451 — FlashAttention-4; search results report FA4 techniques in cuDNN and FA4/FlashInfer backends in vLLM/SGLang on Blackwell (not verified in depth this week).

## What would change the decision

Module 8 is CPU-only and teaches the FlashAttention algorithm, not GPU-generation-specific scheduling. A hardware-agnostic extension of 08.4 on asynchrony and warp specialization (producer/consumer pipelines, ping-pong softmax/MMA) grounded in FlashAttention-3/-4 papers would qualify once it can be taught with a CPU-runnable model of the pipeline; TLX being upstreamed into Triton would strengthen the case for using it as the worked example.

## History

- 2026-10-04 — WAIT — real production engineering with public code, but specific to one GPU generation, an experimental Triton extension and a non-LLM workload; not teachable from scratch in a CPU-only course yet — weekly/2026-W40.md — candidates C-20261003-01
