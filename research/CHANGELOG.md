# Curriculum changelog

How the course evolves. Changes happen at most once a week, only when a development clears the
bar in [the weekly review protocol](research/PROTOCOL-weekly.md). New material extends the
existing lessons; it never replaces them. "No curriculum changes this week" is a normal entry.

## 2026-W42 (2026-10-18)

- Added **[03.4 · Muon: orthogonalized momentum for weight matrices](lessons/module-03/lesson-04.md)**: the SVD and spectral norm, why the steepest step for a weight matrix under the spectral norm is its orthogonal factor $UV^\top$, Newton-Schulz iterations that compute it with matrix multiplications only (worked by hand), the Muon/AdamW parameter split, MuonClip's QK-Clip, the compute and memory cost, and what an independent budget-matched benchmark says about equal-data vs equal-time wins. With `llmre/optim/muon.py`, `tests/test_muon.py` and `experiments/muon_vs_adamw.py`. Extends 03.1–03.2; AdamW stays the course default and nothing is replaced. [Review](research/weekly/2026-W42.md).

## 2026-W41 (2026-10-11)

- Added **[08.5 · Fused linear cross-entropy: the loss without the logits](lessons/module-08/lesson-05.md)**: computing the LM-head cross-entropy, its gradients, and per-token log-probs and entropy without ever building the $(B, T, V)$ logits tensor. Two schemes: row-chunked with gradients in the forward pass (Liger Kernel) and vocabulary-tiled with an online logsumexp (Cut Cross-Entropy). With `llmre/training/fused_ce.py` and `tests/test_fused_ce.py`. Extends 08.4's online softmax; the plain `lm_head` + `F.cross_entropy` loss of 06.1 stays the reference. [17.4](lessons/module-17/lesson-04.md) gains a dated extension noting that TRL v1.15.0 now uses a fused LM head by default. [Review](research/weekly/2026-W41.md).

## 2026-W40 (2026-10-04)

- Added **[17.4 · Logit distillation: forward vs reverse KL & on-policy distillation](lessons/module-17/lesson-04.md)**: matching the teacher's full next-token distribution (forward KL covers modes, reverse KL seeks one, generalized JSD in between) and on-policy distillation, where the student generates and the teacher grades every token. With `llmre/reasoning/distill.py` and `tests/test_distill.py`. Extends 17.3's sequence-level distillation; nothing replaced. [Review](research/weekly/2026-W40.md).

## 2026-W39 (2026-09-27)

- Added **[12.5 · Linear attention, Gated DeltaNet & hybrid stacks](lessons/module-12/lesson-05.md)**: linear attention as an RNN, the (gated) delta rule, and the 3:1 hybrid layouts of Qwen3-Next and Kimi Linear, with `llmre/attention/gated_deltanet.py` and `tests/test_gated_deltanet.py`. Extends 12.3 (GQA); nothing replaced. [Review](research/weekly/2026-W39.md).

## 2026-09-26 — before the weekly system

- Added **Frontier updates** (bonus): [F.1 · MLPerf post-training / RLVR as a benchmark](lessons/frontier/update-01.md) and [F.2 · GRPO fixes: Dr. GRPO, DAPO, GVPO](lessons/frontier/update-02.md), with `llmre/evaluation/pass_at_k.py` and `llmre/rl/gvpo.py`. Added directly at the author's request, before this review process existed; recorded retroactively in the registry.
