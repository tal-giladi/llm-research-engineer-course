# Thinkingbox: grading agents by terminal backend state and repeat-reliability (pass^N)

- **Topic ID:** agent-eval-terminal-state-reliability
- **Status:** WAIT
- **Next review:** 2026-12-13
- **Course change:** none
- **Candidates:** C-20261012-01

## Summary

Agent benchmark that grades by inspecting the environment's final backend state instead of the agent's self-report, and reports repeat-reliability pass^N alongside pass@k.

## Evidence

- https://arxiv.org/abs/2608.19741 — single benchmark paper with code (Microsoft co-authors); the announcing blog post URL was not verified.

## What would change the decision

Use by an independent evaluation suite or lab report; then a short section in 13.x/19.x next to pass@k (llmre/evaluation/pass_at_k.py).

## History

- 2026-10-18 — WAIT — sound evaluation idea but a single benchmark without third-party use — weekly/2026-W42.md — candidates C-20261012-01
