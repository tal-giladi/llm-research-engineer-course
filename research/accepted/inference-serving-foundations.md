# Inference serving foundations: KV cache, prefill vs decode, continuous batching, paged KV memory, lossless speculative decoding

- **Topic ID:** inference-serving-foundations
- **Status:** ADD
- **Next review:** n/a
- **Course change:** lessons/module-08/lesson-06.md (new lesson 08.6); code/src/llmre/model/serving.py; code/tests/test_serving.py; code/experiments/serving_costs.py; _sidebar.md; curriculum/course-outline.md (paper-placement line)
- **Candidates:** none directly (author scope decision 2026-10-28, research/SCOPE-DECISIONS.md); related C-20261019-03, C-20261031-03

## Summary

The course generated text only with the 06.3 loop, which re-runs the prefix every step, and had promised a KV cache "in a later module". 08.6 builds the KV cache for the course GPT, derives why prefill is compute-bound and decode memory-bound from 08.2's roofline (batching amortizes weights, not the cache), and covers continuous (iteration-level) batching, paged KV memory with block tables, lossless draft-and-verify speculative decoding with its proof and speedup formula, serving metrics (TTFT, TPOT, goodput), and why rollout generation dominates RL post-training cost.

## Evidence

- https://www.usenix.org/conference/osdi22/presentation/yu — Orca (OSDI 2022): iteration-level scheduling.
- https://arxiv.org/abs/2309.06180 — vLLM / PagedAttention (SOSP 2023): block-based KV memory, 2-4x throughput vs FasterTransformer and Orca (authors' measurements).
- https://arxiv.org/abs/2211.17192 — Leviathan, Kalman, Matias (ICML 2023): speculative decoding, lossless, 2-3x on T5-XXL (authors).
- https://arxiv.org/abs/2302.01318 — Chen et al. (DeepMind 2023): speculative sampling, 2-2.5x on Chinchilla 70B (company claim).
- https://docs.vllm.ai/en/latest/features/speculative_decoding/ — vLLM ships draft-model, n-gram, EAGLE, MTP and other drafters; states the standard rejection sampler is lossless up to hardware numerics; relaxed verification marked not distribution-preserving.
- https://arxiv.org/abs/2312.07104 — SGLang RadixAttention (prefix caching).
- https://arxiv.org/abs/2403.02310 — Sarathi-Serve (chunked prefill); https://arxiv.org/abs/2401.09670 — DistServe (OSDI 2024, prefill/decode disaggregation, goodput).
- https://arxiv.org/abs/2509.21009 — RollPacker: rollout stage about 70% of each RL training step across three tasks (section 2.2); https://arxiv.org/abs/2505.24298 — AReaL: synchronous generation waits for the longest output.

## What would change the decision

n/a (added). Future extensions of 08.6 are judged on their own evidence: the deferred speculative-decoding cluster, token-level routing, Dynamo-style agentic serving, ReplaySSM; a post-training-quantization foundation is a separate author-approved ADD for a later week.

## History

- 2026-11-01 — ADD — settled, universally implemented material (vLLM/SGLang/TensorRT-LLM) that the author approved as a foundation on 2026-10-28; fills the course's largest structural gap and unlocks several deferred serving topics — weekly/2026-W44.md — candidates none directly (author scope decision 2026-10-28, research/SCOPE-DECISIONS.md); related C-20261019-03, C-20261031-03
