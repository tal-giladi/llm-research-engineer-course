# Fused linear cross-entropy: computing the LM-head loss, log-probs and entropy without materializing logits

- **Topic ID:** fused-linear-cross-entropy
- **Status:** ADD
- **Next review:** n/a
- **Course change:** lessons/module-08/lesson-05.md (new lesson 08.5); code/src/llmre/training/fused_ce.py; code/tests/test_fused_ce.py; lessons/module-17/lesson-04.md (appended 'Advanced extension: TRL's fused LM head (added 2026-10-11)'); _sidebar.md; curriculum/course-outline.md (paper-placement line)
- **Candidates:** C-20261009-02

## Summary

The cross-entropy of a vocabulary projection needs only the log-normalizer and the target logit per position, and its logits gradient is softmax minus onehot. So the loss and its gradients can be computed either chunk-by-chunk over positions with gradients in the forward pass (Liger Kernel) or with an online logsumexp over vocabulary tiles plus recomputation in backward (Cut Cross-Entropy), never building the (N, V) logits tensor. The same online recurrence, with one extra running sum, gives per-token log-probs and entropy for RL and distillation trainers.

## Evidence

- https://github.com/huggingface/trl/releases/tag/v1.15.0 — TRL v1.15.0 (2026-10-08): fused Triton LM head on by default in SFT/DPO/KTO/GRPO/RLOO/Distillation; gemma-3-1b, 262k vocab, 1 GPU: max trainable sequence length 4.0-6.9x higher, peak memory at 8k tokens -52..-82% (maintainer measurements; multi-GPU and generation not measured). Verified 2026-10-11.
- https://arxiv.org/abs/2411.09009 — Cut Cross-Entropy, Wijmans et al. (Apple), ICLR 2025 oral: online logsumexp in SRAM, Gemma 2 2B loss memory ~24 GB -> ~1 MB (authors' measurements). Verified 2026-10-11.
- https://arxiv.org/abs/2410.10989 and https://github.com/linkedin/Liger-Kernel — Liger Kernel (LinkedIn): LigerFusedLinearCrossEntropyLoss, chunk-by-chunk with gradient in forward; README lists integrations with HF Trainer, TRL SFTTrainer, Axolotl, LLaMA-Factory, SWIFT, oumi. Verified 2026-10-11.

## What would change the decision

n/a (added). A later review could extend 08.5 with a differentiable fused log-prob path for GRPO if a widely used trainer documents its internals.

## History

- 2026-10-11 — ADD — settled mechanism (FlashAttention's online softmax applied to the vocabulary axis), peer-reviewed (CCE, ICLR 2025), integrated across the open training ecosystem (Liger) and now the default in six TRL trainers; fills a gap the course names in 01.4/06.1/07.4/17.4 but never taught — weekly/2026-W41.md — candidates C-20261009-02
