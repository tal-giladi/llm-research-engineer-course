# Parallel decoding commit rules for masked diffusion LMs (PACT)

- **Topic ID:** diffusion-lm-parallel-decoding
- **Status:** WAIT
- **Next review:** 2027-02-01
- **Course change:** none
- **Candidates:** C-20261105-06

## Summary

In masked diffusion LMs, decoding commits several tokens per step; PACT samples first, then certifies which sampled tokens are safe to commit (temperature-independent certificate) and spaces committed positions apart. Training-free.

## Evidence

- https://arxiv.org/abs/2611.04152 and https://github.com/LiZaiyuan0619/PACT — LLaDA-8B, LLaDA-MoE, LLaDA2.1-mini, Dream-7B on 5 benchmarks (authors); MIT code with baselines.

## What would change the decision

Governed by the author's 2026-11-03 non-autoregressive-LM decision ('Later'): no lesson until the author re-decides, at trigger (a) a frontier lab or major open family ships a production diffusion LM, or (b) 2027-02-01. Kept as its own file because neither existing non-AR file (RL credit assignment, flow LMs) is about decoding.

## History

- 2026-11-08 — WAIT — non-autoregressive LM topic; kept as WAIT per the author's 2026-11-03 scope decision (not rejected for scope alone) — weekly/2026-W45.md — candidates C-20261105-06
