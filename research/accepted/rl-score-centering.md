# Score centering for off-policy LLM RL

- **Topic ID:** rl-score-centering
- **Status:** ADD
- **Next review:** n/a
- **Course change:** lessons/module-16/lesson-04.md (advanced extension); code/src/llmre/rl/score_centering.py; code/tests/test_rl_score_centering.py; code/experiments/rl_score_centering.py
- **Candidates:** none (requested by Tal in chat, 2026-11-09)

## Summary

Under training-inference mismatch the policy-gradient update splits per prefix into drift, E_q[R] times the sampler's expected score, and signal, Cov_q(R, score). Subtracting the sampler's expected score from each sampled score removes drift. A top-k head with a rescaled trainer tail makes it practical, and it composes with importance weights.

## Evidence

- https://arxiv.org/abs/2609.20807 — Marek and Ryabinin, September 2026: estimator, top-k approximation, composition with IS, experiments on Qwen3 0.6B/1.7B/30B (author results; headline runs use deliberately severe mismatch; 30B mostly single seed).
- https://github.com/huggingface/trl/releases/tag/v1.17.0 and https://github.com/huggingface/trl/pull/7760 — TRL default with masked IS, top-8 head (maintainer claim; single runs; cost estimates revised in PR comments to 2-59%).
- https://rlhfbook.com/c/06-policy-gradients — secondary presentation deriving from the paper.
- https://cameronrwolfe.substack.com/p/score-centering — secondary survey, 2026-11-09.
- https://chrisliu298.ai/posts/off-policy-drift-in-llm-rl/ — read; predates the paper and uses "drift" in a different, broader sense, so not used as evidence.
- Not read: TRL diff; github.com/martin-marek/score-centering; paper figures, Table 1 in full and appendix B.3/Table 2 (seed counts come from a summary of these).

## What would change the decision

Revise the extension if independent replication contradicts the results, or if long-horizon evidence settles the choice of k and the composition with importance weights.

## History

- 2026-11-09 — ADD — requested by Tal outside the weekly cycle; adopted as TRL's default and fits 16.4 as an extension; 16.4's existing text and mismatch.py unchanged — no weekly report (direct request)
