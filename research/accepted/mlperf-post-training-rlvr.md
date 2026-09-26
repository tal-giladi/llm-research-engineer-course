# MLPerf Training post-training benchmark (agentic RLVR)

- **Topic ID:** mlperf-post-training-rlvr
- **Status:** ADD
- **Next review:** n/a
- **Course change:** lessons/frontier/update-01.md; code/src/llmre/evaluation/pass_at_k.py; code/tests/test_frontier_updates.py
- **Candidates:** none (added before the weekly process existed)

## Summary

MLCommons' first MLPerf Training workload for LLM post-training (from the v6.1 round, Oct 2026):
a Qwen3.5-397B-A17B coding agent in the OpenHands harness, trained with GRPO on R2E-Gym
software-repair tasks with binary hidden-test rewards; validated by pass@4 ≥ 0.69; async RL with
max staleness 1; fixed 65k context / 30 turns / 16 generations.

## Evidence

- MLCommons announcement, September 2026 (industry-standard benchmark body).
- R2E-Gym — https://arxiv.org/abs/2504.07164 ; OpenHands — https://arxiv.org/abs/2407.16741 ;
  GRPO — https://arxiv.org/abs/2402.03300 ; pass@k — https://arxiv.org/abs/2107.03374

## What would change the decision

n/a — in the course. Revisit F.1 if the benchmark definition changes materially in a later round.

## History

- 2026-09-26 — ADD — added at the author's request before the weekly process existed; recorded retroactively. Builds on Module 16 (GRPO/RLVR) without replacing it. — n/a
