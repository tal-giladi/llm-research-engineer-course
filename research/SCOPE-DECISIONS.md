# Author scope decisions

Decisions by Tal about what the course covers, for questions the weekly review must not settle on
its own (new foundation areas, structural gaps). The weekly review reads this file in Step 0 and
follows it. Written by the human only; the weekly review may propose changes in its report but
never edits this file.

## 2026-10-28 — answer to the W43 structural-gap question

| Area | Decision | What it means for the weekly review |
|---|---|---|
| **Inference / serving** | **Yes** — add a foundation | A foundation lesson (prefill vs decode, continuous batching, KV-cache cost, lossless draft-and-verify speculative decoding, why rollout generation dominates RL cost) may be an ADD. It is built from settled, widely implemented material (vLLM/SGLang-level evidence), not from the pending single papers. Place it after 06.3 or as a new serving lesson per the outline; it counts as that week's ADD. Once it exists, the deferred serving topics (speculative-decoding cluster, ReplaySSM, Dynamo, token-level routing) are re-judged against it on their own evidence. |
| **Post-training quantization** | **Yes** — add a short foundation | A short lesson (int8/int4 weight quantization, scales and zero-points, GPTQ/AWQ-style error compensation, contrast with training mixed precision in 08.x) may be an ADD in a later week, not the same week as the serving lesson unless both are clearly exceptional. Then RoleQ and 2:4 sparsity vs quantization are re-judged against it. |
| **Sparse attention** | **No, for now** | Do not create a sparse-attention foundation. Sparse-attention candidates (Prism, DoSA, PISA, FFD and similar) are REJECTed at their next review, citing this decision; reopen only if Tal changes this row or one mechanism becomes standard in major open models with independent validation. |
