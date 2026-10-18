# Muon: orthogonalized momentum (Newton-Schulz) for hidden weight matrices, with MuonClip and a budget-matched benchmark

- **Topic ID:** muon-optimizer
- **Status:** ADD
- **Next review:** n/a
- **Course change:** lessons/module-03/lesson-04.md (new lesson 03.4); code/src/llmre/optim/muon.py; code/tests/test_muon.py; code/experiments/muon_vs_adamw.py; _sidebar.md; curriculum/course-outline.md (paper-placement line)
- **Candidates:** C-20261018-01

## Summary

Muon replaces the momentum matrix of each 2-D hidden weight by (approximately) its orthogonal polar factor U V^T, computed with five quintic Newton-Schulz iterations in bf16; embeddings, LM head and 1-D parameters stay on AdamW. It is steepest descent under the spectral norm. Scaled up by Moonshot (Moonlight, Kimi K2 with the QK-Clip stabilizer) and shipped in Megatron Core and torch.optim.

## Evidence

- https://github.com/KellerJordan/Muon — reference implementation; README documents the Muon/AdamW parameter split and adoption (verified 2026-10-18).
- https://arxiv.org/abs/2502.16982 — Moonlight: weight decay + update-RMS matching needed to scale; ~2x compute efficiency vs AdamW in scaling-law runs; 16B/3B MoE on 5.7T tokens (company claim).
- https://arxiv.org/abs/2507.20534 — Kimi K2: 1T/32B MoE, 15.5T tokens with MuonClip, 'zero loss spike'; QK-Clip per head, tau = 100 (company claim; algorithm read from the HTML version).
- https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/ — Megatron Core native Muon, layer-wise distributed optimizer, three TP Newton-Schulz modes; GB300 throughput Kimi K2 1,051 -> 1,080 TFLOP/s/GPU, Qwen3-30B-A3B 713 -> 721 (company claim).
- torch.optim.Muon present in the PyTorch 2.14.1 CPU wheel installed for the tests (source read locally; same coefficients and update rule).
- https://arxiv.org/abs/2610.21555 — independent paired-budget benchmark (NeurIPS 2026 OPT workshop): Muon wins all 20 equal-data pairs; at equal time AdamW wins on small language models (Muon call 56-70% of step time).
- https://arxiv.org/abs/2409.20325 — Old Optimizer, New Norm: steepest-descent-under-a-norm framing.
- https://github.com/KellerJordan/modded-nanogpt — NanoGPT speedrun uses Muon and later variants.
- https://arxiv.org/abs/2610.20256 — Long-Horizon RL Needs Muon (daily 2026-10-15, class C): further research use, single preprint.

## What would change the decision

Revisit (as an extension, not a replacement) if a successor variant (e.g. one used in current speedrun records) gets independent large-scale validation and framework adoption, or if independent large-scale replications contradict the data-efficiency claim.

## History

- 2026-10-18 — ADD — a settled mechanism with an independent benchmark, adoption in frontier open-weight training (company reports), and native support in Megatron Core and PyTorch; fills a gap in Module 3 (AdamW-only), added as new lesson 03.4 without changing 03.1-03.3 — weekly/2026-W42.md — candidates C-20261018-01
