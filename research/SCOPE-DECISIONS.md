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

## 2026-11-03 — answer to the W44 alternative-LM-paradigm question

| Area | Decision | What it means for the weekly review |
|---|---|---|
| **Non-autoregressive LMs** (discrete diffusion LMs, continuous flow / flow-matching LMs) | **Later**: no foundation for now | Do not create a lesson or module on non-autoregressive generation, and do not ADD candidates that depend on one. Keep such candidates (e.g. `diffusion-lm-rl-credit-assignment`, `continuous-flow-language-models`, diffusion speculative decoding) as WAIT, adding new evidence to their deferred files instead of opening near-duplicates; do not REJECT them for scope alone. Re-ask Tal when either trigger fires: (a) a frontier lab (OpenAI, Anthropic, Google DeepMind, Meta) or a major open model family (Qwen, DeepSeek, Llama, Mistral, Gemma) ships a production diffusion or flow LM as a main offering and others validate it independently; or (b) by 2027-02-01 at the latest, whichever comes first. Context when decided: only Inception Labs (Mercury) ships diffusion LMs commercially, Gemini Diffusion is an experimental preview, and flow LMs are research only. |

## 2026-11-03 — answer to the W44 multi-token-prediction question

| Area | Decision | What it means for the weekly review |
|---|---|---|
| **Multi-token prediction (MTP)** | **Yes**: add a short foundation, after the post-training quantization lesson | A short lesson may be an ADD in a week after the PTQ foundation has landed, not the same week unless both are clearly exceptional. Build it from settled, widely implemented material (Gloeckle et al. 2024 parallel heads; DeepSeek-V3 sequential MTP modules; native MTP drafters in Qwen3-Next, Nemotron 3 Super/3.5, Gemma 4 and vLLM), not from pending single papers. Cover the MTP training loss on the course GPT with a worked example, parallel heads vs sequential modules, MTP heads as the drafter for 08.6's speculative decoder (lossless verification), and trade-offs (batch-size cutoff, acceptance falling with depth, training interference). Keep it CPU-runnable and place it after 08.6 with links to 08.6 and the DeepSeek-V3 paper guide. Once it exists, re-judge `multi-token-prediction-training` (Soft-MTP) and MTP-related speculative-decoding candidates against it on their own evidence. |
