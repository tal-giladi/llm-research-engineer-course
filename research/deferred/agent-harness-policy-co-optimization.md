# Harness and policy co-optimization for coding agents (Self-HPO, HarnessPro, HERD)

- **Topic ID:** agent-harness-policy-co-optimization
- **Status:** WAIT
- **Next review:** 2026-12-20
- **Course change:** none
- **Candidates:** C-20261021-03, C-20261104-03, C-20261105-05, C-20261108-01

## Summary

The agent harness (tools, scaffold, prompts) is optimized jointly with the RL-trained policy: the policy proposes harness edits from its failures, then trains under the chosen harness; hard tasks seed the next round.

## Evidence

- https://arxiv.org/abs/2610.22791 — Self-HPO: Qwen3.5-35B-A3B resolve rate 57.8% -> 64.8% (authors); no code.
- arXiv 2610.26770 (HarnessPro) and https://arxiv.org/abs/2610.31401 (HERD) — related single papers (daily 2026-10-21/23).
- https://arxiv.org/abs/2611.02079 — ParetoSaddler: Bayesian-bootstrap Pareto promotion during harness search, cost preference deferred to the final choice; code promised, not released.
- https://developer.nvidia.com/blog/use-nvidia-nemo-to-turn-agent-failures-into-tests-and-harness-fixes/ — NeMo Compass + Eval Author: trace analysis to executable evals to harness fixes, no RL; open repos; one named external user (Dassault Systèmes) with NVIDIA-reported numbers; marketing-authored post.
- https://arxiv.org/abs/2611.04277 and https://github.com/DeepExperience/Harness-S — Harness-S: sibling-harness pool, rescue rollouts in RL, adaptive router; code released but its README says no completed end-to-end run yet.

## What would change the decision

Code release plus a second group reproducing harness co-optimization gains, or adoption in an agent-RL stack; then an extension of 19.1.

## History

- 2026-10-25 — WAIT — three concurrent single papers with different mechanisms; direction not converged, no code — weekly/2026-W43.md — candidates C-20261021-03
- 2026-11-08 — WAIT — three more mechanisms (Pareto-test search, non-RL trace-to-eval loop, sibling-harness rescue); still no convergence or independent reproduction; next review date unchanged — weekly/2026-W45.md — candidates C-20261104-03, C-20261105-05, C-20261108-01
