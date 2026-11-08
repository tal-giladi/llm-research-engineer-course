# Multi-teacher on-policy distillation (MOPD): merging domain-specialist teachers into one student with token-level distillation advantages

- **Topic ID:** multi-teacher-on-policy-distillation
- **Status:** ADD
- **Next review:** n/a
- **Course change:** lessons/module-17/lesson-04.md (appended 'Advanced extension: multi-teacher on-policy distillation (added 2026-11-08)'); code/src/llmre/reasoning/mopd.py; code/tests/test_mopd.py; code/experiments/mopd_two_domains.py
- **Candidates:** C-20261031-01, C-20261107-01

## Summary

Several domain-specialist teachers (often RL-trained from the student's own SFT checkpoint) supply dense token-level signals (teacher-student log-prob gap) to one student on the student's own rollouts; each prompt is routed to its domain's teacher. Framed as an on-policy RL objective whose advantage is the distillation signal, it consolidates specialists into one model. Extends 17.4's single-teacher on-policy distillation. The main practical failure is domain imbalance: a token-averaged loss gives short-answer domains a tiny share of the gradient.

## Evidence

- https://arxiv.org/abs/2601.02780 — MiMo-V2-Flash Technical Report (Xiaomi, Jan 2026): introduces the MOPD paradigm; section 4.4 gives the advantage sg[log pi_domain(y_t) - log pi_theta(y_t)] + alpha * A_ORM, a 1/|y| per-sequence surrogate and a truncated IS weight sg[pi_theta/mu_theta] zeroed outside [eps_low, eps_high] (verified 2026-11-08; company report).
- https://arxiv.org/abs/2606.30406 — Ma et al. (Jun 2026), MOPD paper: on Qwen3-30B-A3B (math/IF/SWE) beats Mix-RL, Cascade RL, off-policy finetune and parameter merging; states deployment in MiMo-V2-Flash (authors' measurements; verified 2026-11-08).
- https://arxiv.org/abs/2608.19098 — Open-MOPD (Gao et al., Aug 2026, independent academic group, released recipe and trajectories): plain M-OPD recovers 35.6% of oracle-ensemble headroom on SmolLM3-3B; token-budget misallocation, not gradient conflict, blamed; fixes raise it to 83.4% (authors' measurements; verified 2026-11-08).
- https://huggingface.co/blog/sergiopaniego/mopd-trl and https://huggingface.co/docs/trl/async_distillation_trainer — TRL `trl.experimental.async_distillation.AsyncDistillationTrainer` (v1.17.1 docs): per-prompt `teacher_id` routed to vLLM teacher servers, generalized JSD (beta=1 reverse KL), per-teacher `teacher_jsd/<id>` and `teacher_token_frac/<id>` metrics; 60-step example measured IF prompts at 1-3% of trained tokens despite 20% of prompts (verified 2026-11-08). Experimental namespace.
- Same blog lists MOPD-style stages in Mistral Large 4, LFM2.5, MiMo-V2.6, Kimi K3, Nemotron 3 Ultra, DeepSeek-V4, GLM-5 (company claims, not checked card by card).
- https://arxiv.org/abs/2610.38095 (Nemotron 3.5), https://arxiv.org/abs/2607.20062 (Solar Open 2), https://arxiv.org/abs/2609.00791 (Instella-MoE) — further model reports with MOPD (company claims; verified in W44).
- NVIDIA NeMo RL MOPD recipe (docs.nvidia.com/nemo/rl/latest/about/algorithms/mopd.html) seen in search results with a stop-gradient teacher-minus-student log-prob advantage; host not reachable from the review session, not linked from the lesson.

## What would change the decision

n/a (added). A later review could extend the section with Open-MOPD's gap-aware allocation once its details are confirmed or reproduced, or with the synchronous TRL DistillationTrainer support (TRL PR #7215) when it lands.

## History

- 2026-11-01 — WAIT — broad multi-lab adoption found in this review; natural 17.4 extension; deferred only for the one-ADD budget; next review next week — weekly/2026-W44.md — candidates C-20261031-01
- 2026-11-08 — ADD — exact formulation published (MiMo-V2-Flash §4.4), a controlled comparison paper, an independent academic reproduction (Open-MOPD), two framework implementations (TRL experimental trainer, NeMo RL recipe) and many company reports; meets 17.4's own stated extension condition; added as an advanced extension of 17.4 — weekly/2026-W45.md — candidates C-20261031-01, C-20261107-01
