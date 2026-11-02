# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261102-01 · MDraft: Multi-Draft On-Policy Distillation for Diffusion Drafters

- **Class:** B
- **Date discovered:** 2026-11-02
- **Date published:** 2026-10-27 (v1, 18:02:08 UTC)
- **Source:** https://arxiv.org/abs/2610.37778 (verified resolves; v1, Tue 27 Oct 2026, cs.CL)
- **Organization/researchers:** Petr Molodyk, Yuchen Zhu, Jaemoo Choi, Molei Tao, Yongxin Chen. Molei Tao and Yongxin Chen are established Georgia Tech applied-math/ML faculty per general knowledge of the named researchers; affiliation is not stated on the abstract page itself.
- **Category:** inference/serving (speculative decoding, diffusion drafters) + RL/post-training (on-policy distillation)
- **What changed:** One-step diffusion drafters propose several speculative-decoding tokens in parallel, but training them against a verifier's feedback is hard because each drafted token in a block is scored without seeing the other tokens in the same block — the authors show distilling against these per-token verifier targets separately suppresses tokens that are plausible under some prefixes, hurting draft acceptance length. MDraft (Multi-Draft On-Policy Distillation) samples several draft blocks from the same context and aggregates the verifier's full-vocabulary predictions across their prefixes before distilling, changing only training (architecture and inference procedure are unchanged).
- **Technical summary:** Authors' own results on math, coding, and conversational tasks: MDraft yields longer accepted draft lengths than prior distillation methods for diffusion drafters, beyond the pretrained drafter's acceptance rate. No independent reproduction found; no stated venue; no code repository linked on the abstract page (checked the "Code, Data and Media" section directly — only generic third-party-tool toggles, no paper-specific link).
- **Why it might matter:** Intersects two course clusters at once: the newly-added inference-serving foundation (`lessons/module-08/lesson-06.md`, 08.6, which teaches lossless draft-and-verify speculative decoding) and the accepted on-policy-distillation lesson (17.4). The course's deferred `speculative-decoding.md` cluster already tracks "RL-trained drafters" as one open sub-problem but has no diffusion-drafter entry yet — this is the first.
- **Evidence of adoption:** None — single paper, six days old at discovery, no code, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; academic authorship, no stated affiliation on the page.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2610.37778
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "diffusion drafter"` (no hit) and by reading `research/deferred/speculative-decoding.md` directly, whose "what would change the decision" explicitly asks for independent reproduction or vLLM/SGLang/TensorRT-LLM adoption before any refinement in this cluster becomes an 08.6 extension — new evidence (a new sub-genre: diffusion drafters) for that already-active WAIT topic, not a duplicate of its five tracked papers (relaxed verification, four October single papers, the MoE-speedup NeurIPS paper).
- **Potential course lesson:** If reproduced or adopted by a serving stack, a citation inside a future 08.6 extension on drafter training, alongside the cluster's other RL/distillation-trained-drafter work.
- **Confidence:** low-medium — a clean mechanistic argument (block-level target aggregation) with concrete acceptance-length results, but single paper, no code, no stated affiliation or venue, no reproduction.
- **Recommendation:** monitor alongside `speculative-decoding.md` at its 2026-11-15 scheduled review; add as a candidate to that file.

### C-20261102-02 · FAD-KV: Adaptive Dictionary KV Caching for Efficient Reasoning Decoding

- **Class:** B
- **Date discovered:** 2026-11-02
- **Date published:** 2026-10-27 (v1, 20:05:31 UTC)
- **Source:** https://arxiv.org/abs/2610.37921 (verified resolves; v1, Tue 27 Oct 2026, cs.CL)
- **Organization/researchers:** Yuxuan Zhu, David H. Yang, Parisa Shiravani, Mohammad Mohammadi Amiri, Tejaswini Pedapati, Subhajit Chaudhury, Keerthiram Murugesan, Pin-Yu Chen. Pin-Yu Chen is a well-known IBM Research scientist per general knowledge of the name; affiliation is not stated on the abstract page itself.
- **Category:** inference/serving (KV-cache compression for long-CoT reasoning decoding)
- **What changed:** Studies dictionary-based sparse KV compression (each token's KV state as a sparse combination of learned atoms) and identifies two decoding-time problems: iterative sparse coding adds overhead, and a static dictionary falls out of sync with evolving KV distributions during generation (naive adaptive updates make the dictionary grow unboundedly). FAD-KV uses a one-shot Top-n screening step plus lightweight coefficient refinement (replacing iterative sparse coding) and a fixed-size dictionary that reuses inactive atoms (bounding memory) instead of growing it.
- **Technical summary:** Authors' own results on reasoning and long-context benchmarks: a "strong quality-efficiency trade-off" with bounded memory growth. No independent reproduction found; no stated venue; no code repository linked (checked directly — only generic third-party-tool toggles).
- **Why it might matter:** The course's deferred `reasoning-kv-compression-pause-tokens.md` (next review 2026-12-13) tracks a single training-based mechanism (AbridgeKV: summarize finished reasoning blocks into learned pause-token KV entries) for the same problem (bounding KV growth during long-CoT decoding). FAD-KV is a mechanistically distinct, training-free-at-inference (dictionary/sparse-coding) approach to the same problem, and is also directly adjacent to the newly-added `lessons/module-08/lesson-06.md` (08.6) inference-serving foundation, which teaches paged KV memory but not compression.
- **Evidence of adoption:** None — single paper, six days old at discovery, no code, no stated venue, no third-party reproduction. The rejected `query-aware-kv-cache-compression.md` and `chunked-kv-compression-phase-sensitivity.md` topics are a reminder that this sub-field has a history of single-paper KV-compression claims that did not hold up — noted as a reason for caution, not for immediate rejection (this is a different mechanism from both).
- **Major organizations using it:** None identified; authors plausibly IBM-affiliated per general knowledge of co-author Pin-Yu Chen, not confirmed on the page.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2610.37921
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "KV cache compression"` (hits the two rejected topics above, plus unrelated tokenizer/MLA mentions) and by reading `reasoning-kv-compression-pause-tokens.md` directly — this is a different mechanism for the same problem the WAIT topic already tracks, not a restatement of AbridgeKV.
- **Potential course lesson:** If reproduced, could join `reasoning-kv-compression-pause-tokens.md`'s eventual resolution as a second, mechanistically distinct data point (dictionary/sparse-coding vs. learned-summary-token), or extend 08.6 directly if a KV-compression sub-lesson is ever built.
- **Confidence:** low — single paper, no code, no stated affiliation or venue, no reproduction, and the sub-field has two prior rejects for related (but distinct) KV-compression claims.
- **Recommendation:** monitor; add as a second candidate alongside AbridgeKV in `reasoning-kv-compression-pause-tokens.md`, noting the mechanism difference explicitly so the two are not conflated at the next review.

### C-20261102-03 · ATI-OPD: Adaptive Teacher Intervention for Stable Long-Sequence On-Policy Distillation

- **Class:** B
- **Date discovered:** 2026-11-02
- **Date published:** 2026-10-28 (v1, 02:00:26 UTC)
- **Source:** https://arxiv.org/abs/2610.38160 (verified resolves; v1, Wed 28 Oct 2026, cs.CL)
- **Organization/researchers:** Jiannan Guan, Youxin Jiang, Yutai Hou, Qiguang Chen, Yirong Zeng, Haitao Wang, Dengyun Peng, Zheng Yan, Jinhao Liu, Tao Ye, Dandan Tu, Wanxiang Che. Wanxiang Che and Qiguang Chen are established Harbin Institute of Technology NLP researchers per general knowledge of the names; affiliation is not stated on the abstract page itself.
- **Category:** RL/post-training (on-policy distillation; teacher-intervention mechanism)
- **What changed:** For long responses in on-policy distillation, the student's early reasoning can drift from the teacher's, weakening the teacher's token-level feedback on later tokens and destabilizing training. Existing fixes (length control, loss reweighting) limit learning from problematic responses without changing their prefixes. ATI-OPD instead replaces the earliest high-disagreement segment of the student's own generation with a short teacher-written segment, lets the student continue from there, and then removes persistently divergent suffixes from the distillation loss (rather than the whole response) to keep useful later continuations.
- **Technical summary:** Authors' own training diagnostics report reduced teacher-student disagreement, fewer gradient spikes, and fewer extremely long responses; across five math-reasoning benchmarks and two teacher-student pairs, ATI-OPD improves average student-only accuracy over standard OPD. No code repository linked on the page; no stated venue; no independent reproduction found.
- **Why it might matter:** A twelfth single-paper variant in the course's actively-tracked `on-policy-distillation-variants.md` cluster (11 candidates already tracked), but mechanistically distinct from all of them — none of the tracked variants do segment-level teacher rewriting combined with persistent-divergence loss masking specifically for long sequences (closest tracked items are AOPD's advantage-weighted teacher-following and PR-OPD's hidden-state alignment, both different mechanisms).
- **Evidence of adoption:** None — single paper, five days old at discovery, no code, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; academic authorship (HIT per general knowledge), not confirmed on the page.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2610.38160
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "on-policy distillation"` and by reading `research/deferred/on-policy-distillation-variants.md` directly — confirms this is the same cluster/topic file as WISD, AOPD, PR-OPD, etc., but a mechanistically new variant (segment-level teacher-rewrite intervention for long-sequence stability), consistent with how the cluster file has treated each new distinct mechanism (e.g. WISD on 2026-11-01).
- **Potential course lesson:** If independently reproduced or if teacher-intervention-style fixes converge across several papers, a short addition to 17.4 on long-sequence stability tricks.
- **Confidence:** low — single paper, no code, no stated venue or independently confirmed affiliation, no reproduction; this specific "teacher rewrites the divergent segment" idea also appears (under different names/papers, e.g. MAESTRO/AC-OPD-style work found via general web search but not yet in the course's tracked list) to be an increasingly crowded sub-genre, which argues for monitoring rather than urgency.
- **Recommendation:** add as the twelfth candidate to `on-policy-distillation-variants.md`; no change to its 2026-11-29 scheduled review date.

### C-20261102-04 · Artifact Distillation: Training-Free Transfer from Frontier to Small Language Models

- **Class:** B
- **Date discovered:** 2026-11-02
- **Date published:** 2026-10-12 (v1, 20:27:04 UTC) — outside the usual ~48h discovery window but not previously recorded in any daily file; surfaced via the arXiv cs.CL Oct-30 listing rescan, flagged here as newly found, not newly published.
- **Source:** https://arxiv.org/abs/2610.37680 (verified resolves; v1, Mon 12 Oct 2026, cs.CL/cs.AI/cs.LG)
- **Organization/researchers:** Mustafa Safa Ozdayi, Somin Wadhwa. Affiliation not stated on the abstract page.
- **Category:** data/distillation (training-free, inference-time capability transfer)
- **What changed:** Proposes transferring capability from a frontier model to a smaller one without any gradient update. A teacher model solves training problems and writes its problem-solving procedures into a natural-language "artifact bank"; the student model consults this bank at inference time (in-context), rather than being fine-tuned on teacher outputs as in standard distillation.
- **Technical summary:** Authors' own results across four LiveBench tasks: Claude Haiku 4.5 using an artifact bank written by Claude Opus 4.5 outscored unaided Claude Sonnet 4.5 on spatial reasoning (0.707 vs. 0.647) at roughly 60% of the inference cost; on table-joining, Haiku-with-bank matched Sonnet 4.5 at roughly 89% of the cost; the same bank raised Qwen3.5-27B's pooled score from 0.730 to 0.800, suggesting cross-model-family transfer. These are API-measured benchmark results from named, checkable models (not a self-reported training claim), but still a single paper with no independent reproduction. No code repository linked on the page (checked directly); no stated venue.
- **Why it might matter:** A genuinely new mechanism not covered by any existing course material or registry entry: "distillation" with zero parameter updates, via natural-language procedure transfer consulted at inference. The course's on-policy-distillation material (17.4) and its variants cluster are all gradient-based; this is a distinct category (closer to in-context learning / prompting research than to RL post-training), with concrete, verifiable cross-model-family numbers.
- **Evidence of adoption:** None — single paper, two unaffiliated authors, no code, no stated venue, no third-party reproduction. The use of real production models (Claude Opus/Sonnet/Haiku 4.5, Qwen3.5-27B) as test subjects is methodologically stronger than a purely synthetic benchmark, but is not itself adoption evidence.
- **Major organizations using it:** None identified as adopters of the technique; the paper merely benchmarks against commercially available models.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2610.37680
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "training-free distillation"` and `lookup "in-context distillation procedure bank"` (no hits either way) — new topic, not a duplicate of any accepted/deferred/rejected file or lesson.
- **Potential course lesson:** If reproduced or adopted, a short new topic (not an extension of 17.4, since the mechanism is training-free) contrasting gradient-based on-policy distillation with inference-time, artifact-bank-based transfer — possibly module 17 or a RAG/memory-adjacent module given the "bank consulted at inference" structure.
- **Confidence:** low — a single paper with no code, no stated affiliation, no venue, and no reproduction, but a clean, falsifiable idea with concrete numbers on real, named production models across two different model families.
- **Recommendation:** monitor; worth a from-scratch sanity check (writing a small artifact bank and testing transfer on a toy task) if the weekly review wants an early read before deciding whether to open a new deferred topic file.
