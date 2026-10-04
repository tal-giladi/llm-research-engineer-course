# Curriculum changelog

How the course evolves. Changes happen at most once a week, only when a development clears the
bar in [the weekly review protocol](research/PROTOCOL-weekly.md). New material extends the
existing lessons; it never replaces them. "No curriculum changes this week" is a normal entry.

## 2026-W40 (2026-10-04)

- Added **[17.4 · Logit distillation: forward vs reverse KL & on-policy distillation](lessons/module-17/lesson-04.md)**: matching the teacher's full next-token distribution (forward KL covers modes, reverse KL seeks one, generalized JSD in between) and on-policy distillation, where the student generates and the teacher grades every token. With `llmre/reasoning/distill.py` and `tests/test_distill.py`. Extends 17.3's sequence-level distillation; nothing replaced. [Review](research/weekly/2026-W40.md).

## 2026-W39 (2026-09-27)

- Added **[12.5 · Linear attention, Gated DeltaNet & hybrid stacks](lessons/module-12/lesson-05.md)**: linear attention as an RNN, the (gated) delta rule, and the 3:1 hybrid layouts of Qwen3-Next and Kimi Linear, with `llmre/attention/gated_deltanet.py` and `tests/test_gated_deltanet.py`. Extends 12.3 (GQA); nothing replaced. [Review](research/weekly/2026-W39.md).

## 2026-09-26 — before the weekly system

- Added **Frontier updates** (bonus): [F.1 · MLPerf post-training / RLVR as a benchmark](lessons/frontier/update-01.md) and [F.2 · GRPO fixes: Dr. GRPO, DAPO, GVPO](lessons/frontier/update-02.md), with `llmre/evaluation/pass_at_k.py` and `llmre/rl/gvpo.py`. Added directly at the author's request, before this review process existed; recorded retroactively in the registry.
