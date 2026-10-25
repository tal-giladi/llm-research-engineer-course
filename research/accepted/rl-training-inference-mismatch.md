# Training-inference mismatch in LLM RL: batch invariance and truncated importance sampling

- **Topic ID:** rl-training-inference-mismatch
- **Status:** ADD
- **Next review:** n/a
- **Course change:** lessons/module-16/lesson-04.md (new lesson 16.4); code/src/llmre/rl/mismatch.py; code/tests/test_rl_mismatch.py; code/experiments/rl_mismatch.py; _sidebar.md; curriculum/course-outline.md (paper-placement line)
- **Candidates:** C-20261025-01

## Summary

RL stacks sample rollouts with an inference engine and train with a separate trainer. With identical weights the two assign different probabilities (reduction order, batch-dependent kernels, kernels/precision, sampling transforms, MoE routing), so nominally on-policy GRPO/PPO is off-policy. Two production answers: remove the mismatch (batch-invariant / bitwise-aligned kernels: vLLM batch-invariance mode, TorchTitan TitanRL) or correct it (decoupled three-policy objective with truncated importance weights or rejection: verl rollout correction, TIS).

## Evidence

- https://pytorch.org/blog/torchtitan-update-new-sota-models-performance-at-scale-titanrl/ — TitanRL: one model definition from pretraining to vLLM; batch-invariant mode gives bitwise-identical trainer/generator log-probs in supported on-policy configurations (company claim); TorchTitan used as backend by verl and Miles (read 2026-10-25).
- https://docs.vllm.ai/en/stable/features/batch_invariance/ — vLLM beta batch-invariance mode (VLLM_BATCH_INVARIANT=1), deterministic kernels, pinned TP all-reduce order, RL listed as a use case, performance cost acknowledged (read 2026-10-25).
- https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr_math.md — verl rollout correction: three policies, decoupled PPO objective, token/sequence TIS with typical caps, rejection with K2/K3 statistics (read 2026-10-25).
- https://arxiv.org/abs/2110.00641 — Hilton, Cobbe, Schulman, NeurIPS 2022: decoupled proximal vs behavior policy.
- https://arxiv.org/abs/2512.01374 — Qwen team: token-level surrogate valid only with small training-inference discrepancy and staleness; IS-corrected policy gradient most stable on-policy; Routing Replay for MoE (company claim).
- https://arxiv.org/abs/2605.14220 — zero-mismatch reference (VeXact): mismatch alone can collapse training (v2 2026-09-29).
- https://arxiv.org/abs/2602.06107 — Jackpot, ICLR 2026: rejection sampling for extreme actor-policy mismatch.
- https://arxiv.org/abs/2605.13907 — AIS: adaptive importance sampling for FP8 rollouts.
- https://arxiv.org/abs/2512.23075 — Trust Region Masking for long-horizon LLM RL.
- Named, not opened (hosts unresolvable from the build sandbox): Thinking Machines 'Defeating Nondeterminism in LLM Inference' (H. He, 2025-09-10); Yao, Liu et al. TIS / FlashRL blog posts and NeurIPS 2025 'On the Rollout-Training Mismatch in Modern RL Systems' (existence corroborated by search results and by citations in 2602.06107 and 2605.13907).

## What would change the decision

Revisit (as an extension) if a standard emerges for long agentic rollouts (cap/granularity), or if bitwise-aligned training/serving becomes the default in major stacks so that correction is rarely needed.

## History

- 2026-10-25 — ADD — multi-source, adopted mechanism (vLLM, verl, TorchTitan, Qwen; NeurIPS/ICLR papers) that every RL lesson implicitly relies on and the course did not teach; new lesson 16.4, 16.1-16.3 unchanged — weekly/2026-W43.md — candidates C-20261025-01
