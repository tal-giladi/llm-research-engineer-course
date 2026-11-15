# Post-training quantization foundation: int8/int4 weights, scales and zero-points, GPTQ, AWQ, MXFP4

- **Topic ID:** post-training-quantization-foundation
- **Status:** ADD
- **Next review:** n/a
- **Course change:** lessons/module-08/lesson-07.md (new lesson 08.7); code/src/llmre/model/quantize.py; code/tests/test_quantize.py; code/experiments/ptq_gpt.py; Kimi K3.1 Pro QAT example inside 08.7 section 6
- **Candidates:** C-20261108-05 (QAT contrast example only; the foundation itself was approved by the author on 2026-10-28, not by a candidate)

## Summary

Weight-only post-training quantization of a finished model: symmetric/asymmetric integer grids with scales and zero-points at per-tensor, per-channel and per-group granularity; the layer-output objective with H = X^T X; GPTQ's column-by-column error compensation through the Cholesky factor of H^-1; AWQ's activation-aware channel scaling; the MXFP4 block format; PTQ vs QAT; bytes per weight and when W4A16 is faster. Built from settled, widely implemented material, as SCOPE-DECISIONS 2026-10-28 asks.

## Evidence

- https://arxiv.org/abs/2210.17323 — GPTQ, ICLR 2023 (verified): 3-4 bit one-shot quantization with second-order information; 175B in about four GPU-hours (authors).
- https://arxiv.org/abs/2208.11580 — Optimal Brain Compression, NeurIPS 2022 (verified): the OBQ update GPTQ builds on.
- https://arxiv.org/abs/2306.00978 — AWQ, MLSys 2024 Best Paper (verified): activation-aware channel scaling, no backprop.
- https://arxiv.org/abs/2208.07339 — LLM.int8(), NeurIPS 2022 (verified); https://arxiv.org/abs/2211.10438 — SmoothQuant, ICML 2023 (verified): activation outliers.
- https://arxiv.org/abs/2310.10537 — Microscaling Data Formats (verified; Table 1: MXFP4 = E2M1 elements, block 32, E8M0 scale).
- https://docs.vllm.ai/en/latest/features/quantization/ — vLLM supports GPTQ, AWQ, bitsandbytes, GGUF, FP8, LLM Compressor INT4 W4A16 / INT8 W8A8 (verified 2026-11-15).
- https://huggingface.co/moonshotai/Kimi-K3.1-Pro — "MXFP4 weights / MXFP8 activations (quantization-aware training)" (verified 2026-11-15; company claim).

## What would change the decision

Extend (not replace) if low-bit activation formats (FP8/FP4 W-A quantization) become the default in open serving stacks with independent quality measurements, or when the deferred topics that depend on this foundation (kv-cache-low-bit-quantization, ternary-post-training-quantization, semi-structured-sparsity-vs-quantization) qualify on their own evidence.

## History

- 2026-11-15 — ADD — author-approved foundation (SCOPE-DECISIONS 2026-10-28) built from settled material with broad production adoption; queued since W43 and a prerequisite for the MTP foundation and several deferred quantization topics — weekly/2026-W46.md — candidates C-20261108-05
