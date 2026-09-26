# GRPO stability fixes: Dr. GRPO and DAPO

- **Topic ID:** grpo-stability-fixes
- **Status:** ADD
- **Next review:** n/a
- **Course change:** lessons/frontier/update-02.md (comparison table and discussion; no separate code)
- **Candidates:** none (added before the weekly process existed)

## Summary

Known weaknesses of GRPO (unbounded importance ratios, length/std normalization bias,
zero-signal unanimous groups) and two published fixes: Dr. GRPO (drop length and std
normalization) and DAPO (clip-higher, dynamic sampling, token-level loss, overlong shaping).

## Evidence

- Dr. GRPO — https://arxiv.org/abs/2503.20783
- DAPO — https://arxiv.org/abs/2503.14476 (open-source system; techniques used in open RL stacks)

## What would change the decision

A future weekly review may promote DAPO-style tricks to a full implemented lesson in Module 16
if they become the default in major RL stacks (verl, OpenRLHF, TRL).

## History

- 2026-09-26 — ADD — added at the author's request before the weekly process existed; recorded retroactively as comparison material extending lesson 16.1. — n/a
