# Logit distillation (forward vs reverse KL) and on-policy distillation

- **Topic ID:** on-policy-distillation
- **Status:** ADD
- **Next review:** n/a
- **Course change:** lessons/module-17/lesson-04.md (new lesson 17.4); code/src/llmre/reasoning/distill.py; code/tests/test_distill.py; _sidebar.md; curriculum/course-outline.md (paper-placement line)
- **Candidates:** C-20261004-01 (plus the same-week cluster C-20260928-02, C-20261001-02, tracked as variants in deferred/on-policy-distillation-variants.md)

## Summary

Instead of SFT on sampled teacher text (sequence-level KD, taught in 17.3), the student matches the teacher's full next-token distribution through forward KL (mode covering; equals SFT cross-entropy for a one-hot teacher), reverse KL (mode seeking) or the generalized JSD between them. On-policy distillation (GKD) computes that divergence on prefixes the student itself generates, so the student learns how to recover from its own mistakes (exposure bias), with a dense per-token signal instead of RL's one scalar per sequence.

## Evidence

- https://arxiv.org/abs/2306.13649 — GKD, Agarwal et al., ICLR 2024 (Google DeepMind authors): on-policy distillation with generalized JSD.
- https://arxiv.org/abs/2306.08543 — MiniLLM, Gu et al., ICLR 2024: reverse-KL on-policy distillation, 120M–13B students.
- https://arxiv.org/abs/2505.09388 — Qwen3 Technical Report (2025): small models trained by off-policy then on-policy strong-to-weak distillation (KL on logits); on-policy distillation of Qwen3-8B reported at ~1,800 GPU-hours vs 17,920 for RL with better AIME/coding scores (company claim).
- https://huggingface.co/docs/trl/main/en/distillation_trainer — TRL stable DistillationTrainer: on-policy generation (optionally vLLM), chunked generalized JSD, beta default 1.0 (reverse KL); experimental GKDTrainer kept in TRL v1.14.0 (https://github.com/huggingface/trl/releases).
- https://arxiv.org/abs/2609.35259 — Piskorz, Berthon, van der Schaar (2026): controlled study; KL direction and learning rate matter more than rollout policy; single paper.
- Same-week independent research cluster: arXiv 2609.30652, 2609.32722, 2609.39884, 2609.39687, 2610.02179 (daily files 2026-09-28 to 2026-10-02).
- Thinking Machines Lab blog "On-Policy Distillation" (2025) — corroborated via search results only; the page could not be fetched through the proxy, so it is not linked from the lesson.
- https://arxiv.org/abs/1503.02531 (Hinton et al. 2015) and https://arxiv.org/abs/1606.07947 (Kim & Rush 2016) — foundational KD and sequence-level KD.

## What would change the decision

n/a (added). A later review may extend 17.4 with a co-evolving teacher, multi-teacher distillation or distillation scaling rules if those are independently reproduced (see deferred/on-policy-distillation-variants.md).

## History

- 2026-10-04 — ADD — mechanism is settled (two ICLR 2024 papers), documented in production (Qwen3 small models), shipped as a stable TRL trainer, and an active independent research cluster; fills a real gap (the course only taught sequence-level KD in 17.3) — weekly/2026-W40.md — candidates C-20261004-01
