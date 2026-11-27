# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261126-01 · TRL v1.19.0: multi-teacher on-policy distillation lands in the stable DistillationTrainer

- **Class:** B
- **Date discovered:** 2026-11-26
- **Date published:** 2026-11-25
- **Source:** https://github.com/huggingface/trl/releases/tag/v1.19.0 (verified via web fetch)
- **Organization/researchers:** Hugging Face TRL maintainers
- **Category:** RL/post-training
- **What changed:** The stable (non-experimental) `DistillationTrainer` accepts a `dict` of teacher models; a per-row `teacher_id` column selects which teacher scores each completion. It also gains `environment_factory` (as in `GRPOTrainer`) so agents can be distilled against stateful environments. Before this, multi-teacher OPD in TRL existed only as `trl.experimental.async_distillation.AsyncDistillationTrainer`.
- **Technical summary:** Docs (https://huggingface.co/docs/trl/main/en/distillation_trainer, verified, section "Multiple teachers"): on-policy student generation (optionally vLLM), chunked generalized JSD against the selected teacher's full next-token distribution; every teacher must share the student's vocabulary; each teacher is prepared like a single teacher, so DeepSpeed ZeRO-3 shards/offloads each of them; FSDP is not supported with a teacher mapping. Cites MOPD as the method. Same release: packed SFT with `packing_strategy="bfd"` keeps packed examples as separate rows with restarting `position_ids` (about 2x faster steps with SDPA on H100 for Qwen3-1.7B, maintainer claim); gradient checkpointing renamed to activation checkpointing following transformers 5.23.
- **Why it might matter:** Lesson 17.4's multi-teacher extension currently points students at an experimental async trainer; a stable, synchronous trainer with a documented `teacher_id` interface is the simpler thing to reference, and its constraints (shared vocabulary, ZeRO-3 per teacher, no FSDP) are concrete memory-cost teaching points.
- **Evidence of adoption:** Shipped in a stable release of a major post-training library; the method itself was already accepted on multi-group evidence (see relationship). Speed number is a maintainer claim.
- **Major organizations using it:** Hugging Face (library); method used by Xiaomi MiMo and NVIDIA NeMo RL recipes per the accepted topic file.
- **Open-source implementation:** https://github.com/huggingface/trl
- **Paper:** none new (method: MOPD, already cited in the accepted file)
- **Code:** https://github.com/huggingface/trl/releases/tag/v1.19.0
- **Relationship to existing course material:** `research/accepted/multi-teacher-on-policy-distillation.md` (its "next review" note says to extend with synchronous TRL DistillationTrainer support "when it lands" — it has now landed); `research/accepted/on-policy-distillation.md`; `research/deferred/grpo-tree-attention-prefix-packing.md` (the same release extends AsyncGRPO tree packing to Nemotron-H/LFM, no change in status).
- **Potential course lesson:** none new; at most a short "production framework" note in the 17.4 multi-teacher extension pointing at the stable trainer and its ZeRO-3/vocabulary constraints.
- **Confidence:** high (that it shipped); low that it warrants more than a one-line note
- **Recommendation:** weekly review: consider a small reference update to the 17.4 extension; no new lesson.
