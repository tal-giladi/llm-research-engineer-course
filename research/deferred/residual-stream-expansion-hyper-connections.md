# Residual-stream expansion and learned residual mixing (mHC, Attention Residuals, gated multi-branch residuals, HiRes, SimpleHC)

- **Topic ID:** residual-stream-expansion-hyper-connections
- **Status:** WAIT
- **Next review:** 2026-12-13
- **Course change:** none
- **Candidates:** C-20261110-01

## Summary

Replacing the plain residual x + f(x) of 05.4 by a widened residual stream mixed by learned matrices (Hyper-Connections; mHC projects the mixing onto doubly stochastic matrices with Sinkhorn-Knopp), by a learned softmax over earlier layer outputs (Kimi Attention Residuals), or by gated multi-branch residuals (Qwen3.8-Next). Independent groups (HiRes, SimpleHC) compare against mHC and question which ingredient matters.

## Evidence

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro — "We incorporate mHC to strengthen conventional residual connections" (company claim on effect).
- https://arxiv.org/abs/2512.24880 — mHC (DeepSeek-AI).
- https://arxiv.org/abs/2603.15031 — Attention Residuals (Kimi Team), pretrained at 48B/1.4T (company paper).
- https://arxiv.org/abs/2608.30320 — Qwen3.8-Next report, 4-branch gated residual (company report).
- https://arxiv.org/abs/2610.41711 — HiRes (independent; authors' numbers, no code).
- https://arxiv.org/abs/2611.05984 — SimpleHC (independent; read/write gating or unconstrained mixing alone match mHC; code promised, not released).

## What would change the decision

ADD a short architecture extension after 05.4 / in Module 12 when (a) the author confirms this area belongs in the course (question in weekly/2026-W46.md), and (b) the mechanism converges: a released reference training implementation (mHC or SimpleHC code) or an independent reproduction of the gain, so the lesson can say which ingredient (widening, mixing, Sinkhorn constraint, gating) actually matters.

## History

- 2026-11-15 — WAIT — adopted in DeepSeek-V4 and explored by Moonshot and Qwen, with two independent comparisons, but three different production variants and an ablation (SimpleHC) that disputes what is essential; no training code; scope question to the author — weekly/2026-W46.md — candidates C-20261110-01
