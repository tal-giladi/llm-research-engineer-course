# DeepSeek Sparse Attention foundation (lightning indexer + top-k selection)

- **Topic ID:** sparse-attention-dsa-foundation
- **Status:** ADD
- **Next review:** n/a
- **Course change:** lessons/module-12/lesson-06.md (new lesson 12.6); code/src/llmre/attention/sparse.py; code/tests/test_sparse_attention.py; code/experiments/dsa_indexer_warmup.py; _sidebar.md; curriculum/course-outline.md (paper-placement line). Earlier: Kimi K3.1 Pro's linear + sparse layout cited as a dated example in 12.5 (2026-11-15).
- **Candidates:** C-20261106-01, C-20261108-05

## Summary

A short foundation lesson after 12.5, approved by the author on 2026-11-08: a small lightning indexer (few ReLU heads, one shared key per token, FP8 in production) scores every past token; each query keeps the top-k (2048) and runs ordinary softmax attention over only those. The indexer is trained by KL to the head-summed main attention, first in a dense warm-up with everything else frozen, then in sparse training with a detached indexer input. Prefill attention goes from O(T^2) to a cheap O(T^2) indexer plus O(Tk); KV storage does not shrink.

## Evidence

- SCOPE-DECISIONS.md, 2026-11-08 row (author decision).
- https://arxiv.org/abs/2512.02556 — DeepSeek-V3.2 paper (full text read 2026-11-22 at https://arxiv.org/html/2512.02556v1): eqs. 1-4, dense warm-up (lr 1e-3, 1,000 steps x 16 x 128K, 2.1B tokens), sparse training (lr 7.3e-6, k=2048, 15,000 steps x 480 x 128K, 943.7B tokens), detached indexer input, MQA mode of MLA, masked-MHA mode for short prefill.
- https://huggingface.co/deepseek-ai/DeepSeek-V3.2 — config.json verified 2026-11-22: index_n_heads 64, index_head_dim 128, index_topk 2048, 128 heads, kv_lora_rank 512, qk_rope_head_dim 64.
- https://github.com/deepseek-ai/DeepSeek-V3.2-Exp — inference demo; links to TileLang kernels (https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32, verified), DeepGEMM indexer-logit kernels (https://github.com/deepseek-ai/DeepGEMM/pull/200, merged 2025-09-29) and FlashMLA sparse kernels (https://github.com/deepseek-ai/FlashMLA/pull/98, merged 2025-09-29); "on par with V3.1-Terminus" (company claim).
- https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html — vLLM recipe (verified); SGLang images per the DeepSeek README.
- https://arxiv.org/abs/2602.15763 — GLM-5 (GLM-5 Team, Feb 2026): "adopts DSA to significantly reduce training and inference costs" (company claim). https://huggingface.co/zai-org/GLM-5 config verified 2026-11-22: GlmMoeDsaForCausalLM, index_n_heads 32, index_head_dim 128, index_topk 2048.
- https://huggingface.co/moonshotai/Kimi-K3.1-Pro — 69 KDA + 24 Gated DSA, index top-k 2048, 32 index heads of dim 128 (card verified 2026-11-15).
- https://arxiv.org/abs/2611.12956 — ADIndex (daily 2026-11-13, C class): third-party work on the DSA indexer (context only, not cited in the lesson).

## What would change the decision

n/a (in the course). Downstream: DoSA, PISA/RecSA, FFD, Prism/MLI, cross-layer index sharing, ADIndex and CISA are re-judged against 12.6 at their own reviews.

## History

- 2026-11-15 — WAIT — author-approved foundation, now with a third adopter (Kimi K3.1 Pro); budget only, this week's ADD went to the older PTQ foundation; first in line with Keep Sampling Mask next week — weekly/2026-W46.md — candidates C-20261106-01, C-20261108-05
- 2026-11-22 — ADD — author-approved foundation built from the DeepSeek-V3.2 paper and three shipped model configs; new lesson 12.6 with a from-scratch DSAttention that reduces exactly to 12.3's GQA — weekly/2026-W47.md — candidates C-20261106-01, C-20261108-05
