# Multi-teacher on-policy distillation (MOPD): merging domain-specialist teachers into one student with token-level distillation advantages

- **Topic ID:** multi-teacher-on-policy-distillation
- **Status:** WAIT
- **Next review:** 2026-11-08
- **Course change:** none
- **Candidates:** C-20261031-01

## Summary

Several domain-specialist teachers (often RL-trained) supply dense token-level signals (teacher-student log-prob gap) to one student on the student's own rollouts; each prompt is routed to one teacher. Framed as an on-policy RL objective whose advantage is the distillation signal, it consolidates specialists into one model. Extends 17.4's single-teacher on-policy distillation.

## Evidence

- https://arxiv.org/abs/2601.02780 — MiMo-V2-Flash Technical Report (Xiaomi, Jan 2026): introduces the MOPD paradigm (company claim).
- https://arxiv.org/abs/2610.38095 — Nemotron 3.5 Super/Lightning (NVIDIA, Oct 2026): MOPD in production post-training (company claim); vLLM day-0 support for the architecture (not for MOPD itself).
- https://arxiv.org/abs/2607.20062 — Solar Open 2 (Jul 2026): twelve domain specialists consolidated by MOPD (company claim).
- https://arxiv.org/abs/2609.00791 — Instella-MoE (Sep 2026): RL with multi-teacher on-policy distillation in post-training.
- NVIDIA NeMo RL documents an MOPD algorithm recipe (docs.nvidia.com, seen via search results only; host not reachable from this session, so not verified directly).
- Search results also list Motif 3 (arXiv 2608.09119) and OneReason (arXiv 2606.06260) with MOPD sections, not individually verified.

## What would change the decision

Adoption across at least four independent model reports plus a framework recipe found during this review; it meets 17.4's own stated extension condition. Waiting for the weekly ADD budget, not for evidence. Next week: ADD as an advanced extension of 17.4 if it still ranks with the Keep Sampling Mask candidate (two only if both clearly exceptional); verify the NeMo RL recipe and one report's exact advantage formula first.

## History

- 2026-11-01 — WAIT — broad multi-lab adoption found in this review; natural 17.4 extension; deferred only for the one-ADD budget; next review next week — weekly/2026-W44.md — candidates C-20261031-01
