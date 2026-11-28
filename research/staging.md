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

### C-20261127-01 · LoopTitan: matched non-revisit controls say parameter revisit itself buys a looped MoE almost nothing

- **Class:** B
- **Date discovered:** 2026-11-27
- **Date published:** 2026-11-25
- **Source:** https://arxiv.org/abs/2611.34821 (verified)
- **Organization/researchers:** Zichen Tian, Yudong Lu, Xuansheng Wu, Na Zou, Xia Hu, Guanchu Wang, Qianru Sun (DATA Lab group per the repo org `datamllab`; affiliations not on the abs page)
- **Category:** architecture
- **What changed:** A controlled re-examination of looped MoE results. At matched parameters, FLOPs and KV cache, the authors decompose "the state-of-the-art looped recipe" (not named on the abs page) and report that a non-revisit control keeps all of the gain: most comes from network shape (more layers, fewer experts each), the rest from the residual scale, which they prove acts as a lower learning rate on revisited layers. That effect helps under AdamW's uneven per-layer step sizes and disappears under Muon. Beyond matched budgets, one extra visit beats doubling the experts, but a deeper standard MoE at equal parameters and compute gets 90.2% of that gain under Muon (authors).
- **Technical summary:** 45-page "Phase 1 technical report"; 17 pretrained models released (LoopTitan-Base and the controls); predictions pre-registered and the full experiment record kept public as GitHub issues (authors). Proposes LoopTitan-Base, a plain MoE that matches the strongest looped recipe under Muon without revisit or residual scale. Same arXiv batch, pointing the other way: Mixture-of-Loops Transformer (https://arxiv.org/abs/2611.34923, verified; hustvl, code at https://github.com/hustvl/MoLT, verified via web fetch, minimal training code + Triton kernel, no checkpoints yet) reports 1.7x CE token efficiency over OLMo3 at 700M in 400B-token pretraining, using cross-loop KV reuse rather than a plain revisit (authors); FlyingStart (https://arxiv.org/abs/2611.34492, verified) is a closed-form warm start that cuts 14-19% of decoding loops on Huginn-0125 (authors).
- **Why it might matter:** The deferred looped-LM topic has been waiting for "a consolidating result that settles which depth prior / injection / residual scheme to teach". This is the first matched-control study seen that says the gain may not come from looping at all. If it holds up, any lesson should teach that confound (shape, optimizer, per-layer learning rate) rather than a looped recipe. It also ties to the course's Muon lesson (03.4): the claimed AdamW-vs-Muon interaction is a concrete per-layer step-size argument.
- **Evidence of adoption:** none; a single group's controlled study with released models and a public record. It contradicts, but does not reproduce, earlier looped-MoE claims (e.g. 2611.34276 in `daily/2026-11-26.md`).
- **Major organizations using it:** none
- **Open-source implementation:** https://github.com/datamllab/LoopTitan (listed on the arXiv abs page; the GitHub page itself was not fetched); models at https://huggingface.co/NaturalScale (listed on the abs page)
- **Paper:** https://arxiv.org/abs/2611.34821
- **Code:** https://github.com/datamllab/LoopTitan (as listed on the abs page)
- **Relationship to existing course material:** `research/deferred/looped-recurrent-depth-lms.md` (scheduled review 2026-11-29; this is counter-evidence for that review, not an ADD signal); `research/accepted/muon-optimizer.md` / `lessons/module-03/lesson-04.md` (AdamW vs Muon per-layer step sizes); `research/deferred/muon-adamw-gap-robustness.md`.
- **Potential course lesson:** none now. If the looped topic is ever added, it would need a "matched control" caveat drawn from this study.
- **Confidence:** medium (the controls are careful and the record is public, but it is days old, from one group, and does not name the recipe it decomposes)
- **Recommendation:** weekly review: attach to the 2026-11-29 review of `deferred/looped-recurrent-depth-lms.md` as counter-evidence; keep WAIT; monitor for a response from the looped-MoE authors or an independent rerun.

### C-20261127-02 · AliceAI-Foundation-80B-A3B-Base (Yandex) technical report: a second lab adopts Attention Residuals and reports its own ablation

- **Class:** B
- **Date discovered:** 2026-11-27
- **Date published:** 2026-11-25 (report); weights published on Hugging Face 2026-09-12
- **Source:** https://arxiv.org/abs/2611.34529 (verified); full text https://arxiv.org/html/2611.34529v1 (verified via web fetch)
- **Organization/researchers:** Yandex (36 authors; "trained from scratch at Yandex" per the abstract)
- **Category:** architecture
- **What changed:** Technical report for an Apache-2.0 open-weight 80B-total / ~3B-active MoE: 48 attention layers in a 3:1 Kimi Delta Attention : full attention pattern, Block Attention Residuals (blocks of 8 layers), Muon, 17.5T pretraining tokens, two context-extension stages and a reasoning/agentic stage (company claim).
- **Technical summary:** From the report (company claims): (1) Block AttnRes on 5-15B models cut loss by about 0.017 and added about 3 points on MMLU-Pro CoT (ru/en) and 4.9 on GSM8K CoT versus the same models without it; combining it with Gated RMSNorm stabilized activations but slowed training about 5%; no separate plain-residual table at this scale. (2) A loss spike at the scaling-law learning rate, traced to growing MoE down-projection outputs, was not fixed by router z-loss, DeepSeek-style routing or clipped SwiGLU; a z-loss on MoE *output activations* (log-sum-exp penalizes the largest positive values) fixed it. (3) Rank agreement between held-out training-document loss and mean benchmark score was only 42%, versus 87-94% for curated validation sets; the predicted learning rate depended on the validation set (5.1e-4 vs 8.2e-4 ± 2.1e-4). (4) Distributed Muon under FSDP: MoE tensors split along the expert axis, matrices assigned to owner GPUs by longest-processing-time scheduling, communication pipelined with Newton-Schulz; about half the optimizer-step time in the final run.
- **Why it might matter:** The deferred residual-stream topic's condition (b) asks for "an independent reproduction of the gain". This is a second organisation (after Moonshot) training with Attention Residuals and publishing its own with/without numbers, in a released model. The held-out-loss-vs-benchmark result is a direct teaching point for scaling-law and evaluation lessons (module 10/13); the MoE output z-loss and Muon sharding are concrete stability and distributed-optimizer examples.
- **Evidence of adoption:** adoption in a released open-weight model by a second major lab (company claim for all numbers); weights at https://huggingface.co/yandex/AliceAI-Foundation-80B-A3B-Base (verified, HTTP 200; tags list transformers custom code and vLLM). Third-party FP8 and MLX quantizations exist on the Hub. No independent evaluation seen.
- **Major organizations using it:** Yandex (this model); Moonshot (Attention Residuals, Kimi Delta Attention, Muon, per existing topic files).
- **Open-source implementation:** model code ships with the weights (`custom_code`); no training code
- **Paper:** https://arxiv.org/abs/2611.34529
- **Code:** https://huggingface.co/yandex/AliceAI-Foundation-80B-A3B-Base
- **Relationship to existing course material:** `research/deferred/residual-stream-expansion-hyper-connections.md` (new evidence for condition (b); condition (a), Tal confirming the area is in scope, is still open per weekly/2026-W46.md); `research/accepted/hybrid-linear-attention-gated-deltanet.md` (another 3:1 KDA hybrid, no change); `research/accepted/muon-optimizer.md` and `research/deferred/flexshard-distributed-muon-sharding.md` (distributed Muon scheduling).
- **Potential course lesson:** none new. At most: a real-world note in 05.4 / Module 12 if the residual topic is accepted; a "held-out loss is a weak proxy for benchmarks" example in the scaling-law lessons.
- **Confidence:** medium (primary report, but every number is the company's own and the AttnRes ablation has no full table)
- **Recommendation:** weekly review: add as evidence to `deferred/residual-stream-expansion-hyper-connections.md` (next review 2026-12-13); consider the held-out-loss rank-agreement result as a small note for the evaluation/scaling lessons; no ADD on this alone.
