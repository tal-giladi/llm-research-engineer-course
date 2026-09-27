# Hybrid linear attention: Gated DeltaNet + full attention (3:1)

- **Topic ID:** hybrid-linear-attention-gated-deltanet
- **Status:** ADD
- **Next review:** n/a
- **Course change:** lessons/module-12/lesson-05.md (new lesson 12.5); code/src/llmre/attention/gated_deltanet.py; code/tests/test_gated_deltanet.py; _sidebar.md; curriculum/course-outline.md (paper-placement line)
- **Candidates:** C-20260926-01

## Summary

Linear attention drops the softmax so attention becomes an RNN with a fixed d_v×d_k matrix state
per head; Gated DeltaNet updates that state with a decay gate (alpha) and the delta rule
(beta-scaled correction toward v_t along k_t). Production models interleave three such layers with
one full softmax-attention layer, cutting long-context decode memory ~4× while keeping exact
long-range retrieval in the full layers.

## Evidence

- https://arxiv.org/abs/2412.06464 — Gated Delta Networks (Yang, Kautz, Hatamizadeh), ICLR 2025; NVIDIA-affiliated authors.
- https://arxiv.org/abs/2406.06484 — DeltaNet chunkwise parallel training algorithm (Yang et al. 2024).
- https://arxiv.org/abs/2006.16236 — linear attention as an RNN (Katharopoulos et al. 2020).
- https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct — shipped open-weight model, 48 layers, 12 × (3 × Gated DeltaNet → 1 × gated attention); supported by vLLM ≥0.10.2 and SGLang ≥0.5.2.
- https://arxiv.org/abs/2510.26692 and https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct — Moonshot's Kimi Linear, independent second lab, 3:1 KDA (Gated DeltaNet extension) : MLA; supported by vLLM/SGLang/HF Transformers. Quality and 6× decode throughput claims are (company claim).
- https://arxiv.org/abs/2608.30320 — Qwen3.8-Next architecture report (2026), continues the hybrid line and adds sparse attention (company claim on efficiency).
- https://github.com/fla-org/flash-linear-attention — open-source Triton kernels (GDN since Dec 2024, KDA, GDN variants), used by vLLM for Qwen3-Next.
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md — independent educational from-scratch implementation.

## What would change the decision

n/a (added). A later review may extend 12.5 with learned sparse attention (e.g. Qwen Sparse
Attention, DeepSeek-style indexers) if those are adopted beyond one lab.

## History

- 2026-09-27 — ADD — mechanism settled (ICLR 2025 paper, open kernels), shipped by two independent labs (Qwen, Moonshot) in 3:1 hybrids and supported by vLLM/SGLang; fills a real gap after 12.3's KV-cache lesson — weekly/2026-W39.md — candidates C-20260926-01
