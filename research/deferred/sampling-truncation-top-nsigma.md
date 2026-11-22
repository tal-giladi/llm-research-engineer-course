# Truncation samplers at high temperature: top-nσ, min-p and temperature-last ordering

- **Topic ID:** sampling-truncation-top-nsigma
- **Status:** WAIT
- **Next review:** 2027-01-10
- **Course change:** none
- **Candidates:** C-20261119-01

## Summary

A large evaluation (29 open-weight models, 1B-122B, incl. SWE-bench Verified agents and 1M-token retrieval) argues that the truncation rule decides whether long generation survives raised temperature: top-nσ (keep logits within n standard deviations of the max) keeps a small candidate set and stays coherent at T=2, while top-p/top-k admit the tail and collapse. It also identifies the order of temperature vs truncation in serving engines (HF/vLLM/SGLang apply temperature first, llama.cpp last) as the mechanism.

## Evidence

- https://arxiv.org/abs/2611.22148 — Roush et al., NeurIPS 2026 Evaluations and Datasets track (per arXiv comment); e.g. 0/100 issues resolved with top-p at T=2 vs 46-62 with top-nσ (authors); one primary LLM judge.
- https://arxiv.org/abs/2411.07641 — top-nσ (Tang et al.); https://arxiv.org/abs/2407.01082 — min-p.
- llama.cpp ships top-nσ in its default sampler chain (daily 2026-11-19); not in vLLM, SGLang or Transformers sampling params (all ship min-p).
- https://github.com/Hellisotherpeople/Long-Context-Generation-Is-a-Sampling-Problem-8- — code (not run).

## What would change the decision

A second group reproducing the high-temperature result, or vLLM/SGLang/Transformers adding top-nσ or a temperature-last option. Then a short dated extension of 06.3 (candidate-set size and truncation order worked on the 06.3 five-logit vector, with a test that top-nσ's set is temperature-invariant under temperature-last).

## History

- 2026-11-22 — WAIT — single (large, well-run) evaluation paper; adoption only in llama.cpp; mechanism is small and hand-checkable, so a short 06.3 extension once independently confirmed — weekly/2026-W47.md — candidates C-20261119-01
