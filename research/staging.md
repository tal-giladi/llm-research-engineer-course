# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261103-01 · FLy: lossy relaxed-verification speculative decoding, now a documented vLLM core feature

- **Class:** A
- **Date discovered:** 2026-11-03
- **Date published:** 2025-11-28 (v1); v3 2026-04-29; shipped in vLLM v0.33.0, released 2026-11-02
- **Source:** https://arxiv.org/abs/2511.22972 (verified resolves — title "Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match"; comments field states ICLR 2026); https://docs.vllm.ai/en/latest/features/speculative_decoding/fly/ (verified resolves, vLLM's own docs); https://github.com/vllm-project/vllm/releases/tag/v0.33.0 (verified resolves, lists "FLy" as a new opt-in lossy verification policy)
- **Organization/researchers:** Jinze Li, Yixing Xu, Guanchen Li, Shuo Yang, Jinfeng Xu, Xuanwu Yin, Dong Li, Edith C.H. Ngai, Emad Barsoum (affiliations not stated on the abstract page; Emad Barsoum is a known AMD GenAI researcher per general knowledge of the name, not independently confirmed on the page).
- **Category:** inference/serving (speculative decoding — relaxed/lossy verification)
- **What changed:** Standard speculative decoding verifies drafts with an exact-match rejection-sampling rule that preserves the target distribution exactly (this lossless rule is what the course's 08.6 already teaches). FLy ("Training-Free Loosely Speculative Decoding") relaxes this: at a high-entropy target position it can defer rejecting a draft token, checking a short window of later draft/verify agreement before finalizing the decision, using an entropy gate (top-k target probabilities) to decide where relaxation is safe. It needs no extra training. vLLM has now shipped this as an opt-in, fully documented `rejection_sample_method="fly"` policy (with `fly_window_size`, `fly_entropy_threshold`, `fly_entropy_top_k` knobs), explicitly flagged in vLLM's own docs as "intentionally lossy" — the first built-in vLLM verification policy that abandons the lossless guarantee by design (distinct from the existing lossless/exact-match and from DSpark's confidence-scheduled *lossless* adaptive verification).
- **Technical summary:** Paper's own numbers (Llama-3.1-70B/405B-Instruct): >99% of target-model accuracy retained, average speedups of 2.81x (70B) and 5.07x (405B); reports beating training-based EAGLE-3 by 1.62x on out-of-domain data. vLLM's own docs independently report measured quality retention across 29 configurations (minimum 97.9%) and separate NVIDIA/AMD single-pass evaluations (minimum accuracy retention 96.30% NVIDIA / 96.43% AMD, both corresponding to exactly one fewer correct AIME25 answer), plus measured speedups of ~1.07–1.94x in one evaluation setup and ~1.04–1.31x in another — i.e., vLLM re-measured the technique on its own stack rather than only citing the paper's numbers, which is a step beyond a company restating authors' claims.
- **Why it might matter:** This is precisely the trigger the deferred `speculative-decoding.md` cluster already names as the condition for re-judging its tracked "relaxed verification" sub-problem: *"independent reproduction, or adoption in vLLM/SGLang/TensorRT-LLM"* (added 2026-11-01, after 08.6). FLy is a different paper from the cluster's existing relaxed-verification candidate (C-20261019-03, arXiv 2610.23818, under review for NAACL 2027, no code, no adoption) — FLy is ICLR 2026 (peer-reviewed), and is now a shipped, documented, independently-remeasured vLLM feature. This is materially stronger evidence than anything currently in that cluster.
- **Evidence of adoption:** Strong for an A: merged into vLLM core (not a fork/plugin) and shipped in the current stable release (v0.33.0, 2 Nov 2026); vLLM's own docs independently re-measured quality/speedup rather than only citing the paper. Not yet found: SGLang or TensorRT-LLM adoption (checked SGLang v0.5.23 release notes directly — no mention); no third-party academic reproduction beyond vLLM's own re-measurement.
- **Major organizations using it:** vLLM (shipped as a core, documented feature). Possible AMD co-authorship (Emad Barsoum, per general knowledge of the name) is plausible but not confirmed on the paper page.
- **Open-source implementation:** https://github.com/vllm-project/vllm (core, v0.33.0+; docs at the URL above). No separate reference implementation from the paper authors was found.
- **Paper:** https://arxiv.org/abs/2511.22972
- **Code:** https://docs.vllm.ai/en/latest/features/speculative_decoding/fly/ (vLLM integration; no standalone author repo found)
- **Relationship to existing course material:** Checked via `lookup "FLy"` (no hit) and `lookup "relaxed verification speculative decoding"` (hits `research/deferred/speculative-decoding.md` and the accepted `inference-serving-foundations.md`/lesson 08.6). This is new evidence for the already-active WAIT cluster, not a duplicate of its existing relaxed-verification candidate (different paper, different authors, different venue, materially stronger adoption story) — exactly the kind of evidence the cluster's "what would change the decision" note asked for.
- **Potential course lesson:** A concrete extension of 08.6: after teaching the lossless draft-and-verify rule, FLy is a clean, real, shipped counter-example of a serving stack deliberately trading the lossless guarantee for speed, with vLLM's own measured accuracy-retention numbers as the worked "how much does lossy cost you" example — directly in the course's `.callout.guarantee`-style pattern (what it guarantees / does not guarantee / measured cost) if reused in the ai-security-course too.
- **Confidence:** high for "this is real and shipped" (primary sources: arXiv abstract page, vLLM's own current docs, vLLM's GitHub release notes all independently verified and mutually consistent); medium for "this generalizes" (single academic paper behind the shipped feature, no SGLang/TensorRT-LLM adoption yet, no third-party academic reproduction).
- **Recommendation:** weekly review should treat this as the headline update to `speculative-decoding.md` — likely an ADD as an 08.6 extension (lossy verification as a named, real trade-off), or at minimum a strong WAIT update that changes the cluster's priority.

### C-20261103-02 · XPress: causal refiner for block-diffusion speculative-decoding drafters, merged into vLLM core by the paper's own author

- **Class:** A
- **Date discovered:** 2026-11-03
- **Date published:** 2026-08-03 (v1); v2 2026-10-26; PR merged into vLLM 2026-10-22; shipped in vLLM v0.33.0, released 2026-11-02
- **Source:** https://arxiv.org/abs/2608.02438 (verified resolves — title "XPress: Parallel Refinement for Diffusion Drafters in Speculative Decoding"); https://github.com/vllm-project/vllm/pull/54448 (verified resolves, merged); https://github.com/vllm-project/vllm/releases/tag/v0.33.0 (verified resolves, lists "XPress" under Speculative decoding, citing PR #54448)
- **Organization/researchers:** Zheng Wang, Davis Wertheimer, Yu Chin Fabian Lim, Mudhakar Srivatsa, Raghu K. Ganti, Minjia Zhang, Naigang Wang (affiliations not stated on the abstract page; several of IBM Research's speculative-decoding team names recur across this group per general knowledge, not independently confirmed on the page). PR #54448 author "ZKBig" is the same Zheng Wang (first author), i.e. the paper's own author contributed the vLLM integration; merged by vLLM maintainer Benjamin Chislett.
- **Category:** inference/serving (speculative decoding — block-diffusion drafter training/refinement)
- **What changed:** Block-diffusion drafters (e.g. DFlash — already a shipped vLLM/vLLM-TPU speculator, >6x lossless speedup per its own paper, 3.13x measured by a Google TPU blog post) draft a whole block of tokens in one forward pass by sampling each position independently, so tokens in the block are not conditioned on each other — which causes early rejections by the target verifier. XPress adds a lightweight causal refiner, resolved via Jacobi iteration, that restores cross-token dependency within the drafted block without a token-by-token decoding loop. This is a refinement layered on top of existing diffusion drafters, not a new drafter itself.
- **Technical summary:** Paper's own numbers (Qwen3-8B, 7 math/code/chat benchmarks, vs. plain dFlash): ~30% average / up to 56% acceptance-length increase, ~1.3x average / up to 1.7x end-to-end throughput increase. The merged PR's own description (read directly) includes e2e GSM8K and throughput/acceptance-length benchmarks against both DFlash and DSpark baselines run on vLLM's own stack. No third-party academic reproduction found; no stated conference acceptance for the paper itself.
- **Why it might matter:** This is the second mechanism after yesterday's MDraft (C-20261102-01, staged as B) to target diffusion drafters for speculative decoding — yesterday's record noted the cluster "has no diffusion-drafter entry yet." XPress is mechanistically different from MDraft (causal post-hoc refinement of a drafter's output vs. MDraft's training-time multi-draft-block distillation target aggregation) but targets the same underlying cross-token-independence problem in block-diffusion drafting, and — unlike MDraft — it is not merely a paper: its own author merged it into vLLM core, and it is now a documented, shipped, V1 speculative-decoding method.
- **Evidence of adoption:** Strong: merged by a vLLM maintainer (not a fork), author-contributed, and shipped in the current stable release; PR includes vLLM-stack benchmarks against the two most relevant existing baselines (DFlash, DSpark) rather than only citing the paper's own numbers.
- **Major organizations using it:** vLLM (core, shipped). Possible IBM Research co-authorship (several named researchers associated with IBM's efficiency/speculative-decoding work per general knowledge) plausible but not confirmed on the page.
- **Open-source implementation:** https://github.com/vllm-project/vllm/pull/54448 (merged, commit ce9467b), shipped in v0.33.0.
- **Paper:** https://arxiv.org/abs/2608.02438
- **Code:** https://github.com/vllm-project/vllm (core, v0.33.0+)
- **Relationship to existing course material:** Checked via `lookup "XPress diffusion drafter"`, `lookup "DFlash"` (no hits) and `lookup "block diffusion drafter"` (hits only yesterday's staged MDraft record, confirming this is new, not a restatement) — new evidence for `research/deferred/speculative-decoding.md`, extending the diffusion-drafter sub-genre MDraft opened yesterday, but with materially stronger (merged/shipped) evidence than MDraft has.
- **Potential course lesson:** Together with C-20261103-01, supports extending 08.6 (or a future dedicated speculative-decoding lesson) with a short diffusion-drafter subsection — DFlash as the base mechanism (already real, TPU-validated), XPress as the "why naive parallel block drafting under-performs and how a causal refiner fixes it" worked example.
- **Confidence:** high for "this is real and shipped by the paper's own author into vLLM core"; medium for "this generalizes beyond the author's own benchmarks" (no independent third-party reproduction; comparisons are the PR author's own, albeit run directly on vLLM's stack).
- **Recommendation:** weekly review should read this alongside C-20261103-01 and yesterday's MDraft (B) as a cluster of real, current speculative-decoding developments; likely strengthens the case for an 08.6 extension covering lossy verification and diffusion drafters together.

### C-20261103-03 · ProVe: Proposer-Verifier Decoding (self-speculative, role-reversed block prediction)

- **Class:** B
- **Date discovered:** 2026-11-03
- **Date published:** 2026-10-20 (v1)
- **Source:** https://arxiv.org/abs/2610.27815 (verified resolves; cs.CL, cross-listed cs.LG)
- **Organization/researchers:** Ziqi Jin, Weiwen Xu, Jiaxi Li, Yujie Zhong, Qi She, Aixin Sun (affiliations not stated on the abstract page).
- **Category:** inference/serving (speculative decoding — proposer/verifier role reversal, self-speculative decoding)
- **What changed:** Standard speculative decoding uses a small drafter and a large verifier in separate forward passes. ProVe fine-tunes a standard autoregressive model into a "proposer" that predicts a whole block of future tokens in one forward pass (not via diffusion, unlike DFlash/XPress above), then verifies the block with a second forward pass. The verifier can be a smaller model, or the proposer itself — the latter ("ProVe-lossless") exactly reproduces the proposer's own greedy decoding output, making it self-speculative rather than a separate-draft-model scheme.
- **Technical summary:** Authors' own results across six benchmarks: with a 32B proposer, ProVe-lossless reaches 2.5x average throughput over plain autoregressive decoding without training a separate draft head; with a 3B verifier, 2.7x throughput at 2.6 accuracy points below AR average (a tunable lossy/speed trade-off). No code repository found on the page; no stated venue.
- **Why it might matter:** A third, mechanistically distinct speculative-decoding development found today, but self-speculative (same model plays both roles after fine-tuning) rather than diffusion-based or verification-relaxation-based — a different point in the same design space as the two A items above and as the cluster's existing five single papers.
- **Evidence of adoption:** None — single paper, two weeks old, no code, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; no stated affiliation.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2610.27815
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "proposer verifier decoding"` (no hit) and `lookup "self-speculative decoding"` (hits only the unrelated `continuous-flow-language-models.md` deferred topic, about flow-LM self-speculative decoding, a different setting) — genuinely new evidence for `speculative-decoding.md`, not a duplicate.
- **Potential course lesson:** If reproduced, a citation inside a future 08.6 extension on self-speculative (single-model) approaches, contrasted with the separate-draft-model and diffusion-drafter approaches.
- **Confidence:** low — single paper, no code, no stated affiliation or venue, no reproduction, though the "lossless when the model verifies itself" framing is a clean, checkable claim.
- **Recommendation:** add as a candidate to `speculative-decoding.md` alongside today's two A items and yesterday's MDraft; monitor at the cluster's 2026-11-15 scheduled review.

### C-20261103-04 · TESD: Teacher-Evolution Self-Distillation (reward-driven teacher updates in on-policy self-distillation)

- **Class:** B
- **Date discovered:** 2026-11-03
- **Date published:** 2026-10-28 (v1)
- **Source:** https://arxiv.org/abs/2610.38637 (verified resolves; cs.LG, cross-listed cs.AI)
- **Organization/researchers:** Shufan Shen, Zhongni Hou, Chao Bi, Yufei Zhang, Guojun Yin, Shuhui Wang (affiliations not stated on the abstract page).
- **Category:** RL/post-training (on-policy self-distillation; teacher-update mechanism)
- **What changed:** In on-policy self-distillation (OPSD), a frozen self-teacher (the same model, given privileged context) can give unreliable guidance on some samples, stalling the student's gains. TESD adds: (1) teacher-oriented optimization — the self-teacher's own rollouts are rewarded for task correctness and context-independence, so the teacher improves over training rather than staying frozen; (2) student-oriented optimization — per-sample guidance reliability is estimated from those same rewards, and unreliable samples are down-weighted in the student's distillation loss.
- **Technical summary:** Authors' abstract reports TESD "consistently outperforms representative token-level and sequence-level methods" across multiple datasets/scales; no specific numbers given in the abstract itself (full text not fetched beyond the abstract page). No code repository found on the page; no stated venue.
- **Why it might matter:** A 13th mechanistically-distinct single-paper variant for the course's already heavily-tracked `on-policy-distillation-variants.md` cluster (12 prior candidates, including yesterday's WISD/ATI-OPD). Distinct from the tracked variants: it specifically evolves the *teacher* via its own reward signal (task correctness + context-independence) rather than weakening a copy of the model (WISD), rewriting divergent segments (ATI-OPD), or aligning hidden states (PR-OPD). Given how saturated this cluster already is with single, unreplicated papers, this is recorded for completeness but flagged as lower-priority than the two A items today.
- **Evidence of adoption:** None — single paper, six days old, no code, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; no stated affiliation.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2610.38637
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "teacher evolution self-distillation"` (no hit) and by reading `research/deferred/on-policy-distillation-variants.md` directly (same cluster/topic file as WISD, AOPD, PR-OPD, ATI-OPD, etc.; this is a new, distinct mechanism, not a restatement).
- **Potential course lesson:** If independently reproduced, folds into the cluster's eventual resolution (17.4 extension or a short survey of "what makes a good self-teacher" tricks) alongside WISD/ATI-OPD.
- **Confidence:** low — single paper, no code, no stated venue or affiliation, no reproduction, abstract-only numbers (not even reported in the abstract itself).
- **Recommendation:** add as the thirteenth candidate to `on-policy-distillation-variants.md`; no change to its 2026-11-29 scheduled review. Given the cluster's size, the weekly review may want to consider whether it has enough evidence to resolve (ADD/WAIT/REJECT) rather than keep accumulating single-paper variants indefinitely — noted here as a process observation, not a decision this run can make.

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
