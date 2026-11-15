# Comparing RL reward schemes at matched compute (dense Monte-Carlo process rewards vs outcome GRPO)

- **Topic ID:** rl-reward-matched-compute-comparison
- **Status:** WAIT
- **Next review:** 2027-01-10
- **Course change:** none
- **Candidates:** C-20261112-01

## Summary

Reward schemes are usually compared at equal training steps. At equal generated tokens and GPU-hours, Monte-Carlo step scoring spends about 91% of tokens on scoring and the ranking against plain GRPO reverses (authors' claim); also, prompts with predicted success near 1/2 give the most GRPO signal.

## Evidence

- https://arxiv.org/abs/2611.13560 — 1,668 runs over six models (authors' numbers); code repository is a placeholder.

## What would change the decision

Code release or an independent matched-compute replication. Then a common-mistakes / research-connection note in 17.2 and a short note on p≈1/2 prompt selection beside GRPO's group-normalized advantage.

## History

- 2026-11-15 — WAIT — clean methodology point, but a single paper with no released code — weekly/2026-W46.md — candidates C-20261112-01
