# Per-harness LoRA adapters trained with GRPO behind a tool-call safety proxy

- **Topic ID:** harness-lora-adapters
- **Status:** REJECT
- **Next review:** on new evidence
- **Course change:** none
- **Candidates:** C-20261107-04

## Summary

One LoRA adapter per (model, agent harness) pair, trained with TRL's AsyncGRPOTrainer inside the unmodified harness; a proxy rule-checks tool calls and gives blocked calls reward 0.

## Evidence

- https://huggingface.co/blog/harness-hub/harness-adapters — Hugging Face team post; author-labeled smoke tests; the guard ablation is +0.0 +/- 4.1 points; skewed task mix.

## What would change the decision

Reopen if full-run results show a clear, reproduced benefit and the pattern is adopted beyond its authors. Even then it composes pieces the course already teaches (LoRA, GRPO, reward shaping), so it would be an example, not a lesson.

## History

- 2026-11-08 — REJECT — an engineering composition of already-taught parts (LoRA, GRPO, reward masking) with smoke-test results and a null ablation — weekly/2026-W45.md — candidates C-20261107-04
