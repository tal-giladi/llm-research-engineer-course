# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261108-05 · Kimi K3.1 Pro (Moonshot AI): open-weight 2.8T/104B MoE with hybrid KDA + Gated DSA attention, MXFP4/MXFP8 QAT, joint multi-domain RL

- **Class:** B
- **Date discovered:** 2026-11-08
- **Date published:** ~early November 2026 (HF repo created ~2026-11-02; no release date on the card)
- **Source:** https://huggingface.co/moonshotai/Kimi-K3.1-Pro (model card and LICENSE, fetched directly). Tech blog https://www.kimi.ai/blog/kimi-k3-1-pro **not fetched** (host blocked by this session's egress policy). Serving recipes linked from the card (not fetched): https://recipes.vllm.ai/moonshotai/Kimi-K3.1-Pro , https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3.1-Pro
- **Organization/researchers:** Moonshot AI
- **Category:** architecture / attention / numerics / rl
- **What changed:** Post-trained from Kimi K3. Architecture per card: 93 layers (1 dense), 896 experts, 16 routed + 2 shared, expert hidden 3072, "latent MoE dimension" 3584; attention = 69 KDA layers + 24 Gated DSA layers (Gated DSA replaces K3's full-attention Gated MLA layers; 32 index heads, index head dim 128, index top-k 2048); 1M context; 160K vocab; MoonViT-V2 401M vision encoder; MXFP4 weights / MXFP8 activations via quantization-aware training; "large-scale joint reinforcement learning across hundreds of downstream domains"; reasoning_effort low/high/max. Custom MIT-style licence with conditions for >US$20M-revenue MaaS operators and >100M-MAU products.
- **Technical summary:** Card benchmarks (company claims; own runs often via Claude Code / Kimi Code harnesses; competitor numbers are max reported values) vs Claude Fable 5.1, GPT-6 Astra, Claude Opus 5.5, DeepSeek-V4.1-Flash, GLM-5.3: leads on SWE Marathon (51.7) and HLE w/ tools (70.0); near Opus 5.5 on Terminal-Bench 4.0 (65.7 vs 66.4) and GDPval-AA (1846 vs 1866); clearly behind the frontier models on ProgramBench (31.5 vs 50–65), HLE no tools (51.1), HLE-Diamond (39.4), Terminal-Bench-Science (38.1). Several internal benchmarks. Not disclosed on the card: pretraining tokens/data, optimizer (Muon/MuonClip?), stability tricks, RL algorithm, MTP heads, vision training; "KDA" not expanded on the card (almost certainly Kimi Delta Attention — inference).
- **Why it might matter:** (1) Third major open model family using DeepSeek Sparse Attention (after DeepSeek-V3.2 and GLM-5) — directly supports Tal's 2026-11-08 decision to reopen sparse attention and gives the planned DSA foundation a gated variant to cite. (2) A production hybrid of linear attention (KDA, ~3/4 of layers) and sparse attention (DSA, ~1/4) — extends the accepted Gated DeltaNet / hybrid-linear-attention topic with a sparse rather than full-attention partner. (3) Second lab in one week (with Inkling 2 Small, C-20261108-04) describing joint multi-domain RL rather than specialist RL + distillation — context for 17.4's MOPD extension. (4) MXFP4/MXFP8 QAT in a released frontier-class open model — contrast case for the approved PTQ foundation.
- **Evidence of adoption:** The model itself is released open-weight with vLLM, SGLang and TokenSpeed recipes (adoption of the architecture by serving engines). DSA has two prior independent adopters; KDA/Gated DeltaNet family is already accepted in the course. Benchmarks are company claims; no third-party evaluation read.
- **Major organizations using it:** Moonshot AI; serving support in vLLM and SGLang.
- **Open-source implementation:** open weights (Safetensors) on Hugging Face; inference via vLLM/SGLang.
- **Paper:** none linked (tech blog only).
- **Code:** no Moonshot training code; serving-engine recipes only.
- **Relationship to existing course material:** `accepted/hybrid-linear-attention-gated-deltanet.md` (KDA/hybrid), the sparse-attention deferred files and SCOPE-DECISIONS 2026-11-08 (DSA foundation after 12.5), 17.4 MOPD extension (joint RL contrast), approved PTQ foundation (MXFP4 QAT), MoE lessons (896 experts, 16+2 routing, latent MoE dimension).
- **Potential course lesson:** No new lesson on its own. **Tal's instruction (2026-11-08): even if the review finds nothing new in it at all, do not REJECT it for lack of novelty — still add it to the course as a real-world example.** Concretely: cite its Gated DSA layer config (top-k 2048, 32 index heads) in the DSA foundation when it is written; add it as a dated production example of a linear+sparse hybrid wherever the course covers Gated DeltaNet / hybrid attention; cite it next to Inkling 2 Small as a joint-RL data point in 17.4's extension; use MXFP4/MXFP8 QAT as the QAT-vs-PTQ contrast example in the PTQ lesson; optionally its MoE shape (896 experts, 16 routed + 2 shared) as a scale example in the MoE lessons. Label all as company claims with the model-card link and date.
- **Confidence:** high for the architecture facts (read from the card directly); low for benchmark standing (vendor numbers, harness and fallback caveats); undisclosed training details.
- **Recommendation:** Weekly review: treat as real-world-example material per Tal's instruction — add the dated example lines above to the relevant lessons (small FIX/extension-sized edits; they need not consume the week's single ADD if the protocol allows), and fold it as evidence into the sparse-attention and hybrid-linear-attention topic files. Re-check the tech blog for training details when kimi.ai is reachable. Process note: surfaced by Tal, not the automated daily sweep.

### C-20261109-01 · RecSA (Recursive Sparse Attention): tree-summarized KV cache with O(log L) per-query selection, 1M-2M-token extrapolation from a 32K-trained Qwen3-4B

- **Class:** B
- **Date discovered:** 2026-11-09
- **Date published:** 2026-11-04 (v1)
- **Source:** https://arxiv.org/abs/2611.07232 (verified resolves; HTTP 200)
- **Organization/researchers:** Ruitian Wang, Xinyu Yang, Minshen Zhang, Hao Zhang (affiliations not stated on the abstract page).
- **Category:** attention / long-context
- **What changed:** Most trainable sparse-attention schemes (including DSA's lightning indexer, which the course plans to teach after 12.5) still score every cached key per query, i.e. O(L) selection cost. RecSA builds a tree over the KV cache in which each node summarizes its subtree with one key and one mean-pooled value, and each query descends the tree, so selection is O(log L).
- **Technical summary:** Initialized from Qwen3-4B and trained on 0.62B tokens at the original 32K length. Authors report RULER wins over all baselines at every length from 131K to 2M (at 1M: 60.5 vs 40.0 for the best baseline, Dense + DCA) and, with optimized kernels, faster-than-dense end-to-end prefill and decoding from 64K to 2M tokens (at 524K on one H200: 3.2x prefill, 3.8x decode). All numbers are the authors'; no code linked on the abstract page.
- **Why it might matter:** A clean, teachable contrast to the planned DSA foundation: the same "select top-k keys per query" goal, but hierarchical summaries replace the flat indexer, so both selection cost and length extrapolation change. Also gives a long-context angle (extrapolation far beyond the training length) that the sparse-attention cluster has not had.
- **Evidence of adoption:** none; single paper, days old, no code, no reproduction.
- **Major organizations using it:** none known.
- **Open-source implementation:** none
- **Paper:** https://arxiv.org/abs/2611.07232
- **Code:** none
- **Relationship to existing course material:** `deferred/block-sparse-attention-pisa.md` (PISA, hierarchical block-sparse attention with O(N log N) scoring; the closest prior topic, so the weekly review may fold this in there rather than open a new file); SCOPE-DECISIONS 2026-11-08 (sparse attention reopened; DSA foundation after 12.5, built from DeepSeek-V3.2/GLM-5 material, not from pending single papers); Module 12 long-context lessons.
- **Potential course lesson:** none yet. Once the DSA foundation exists, possibly a "hierarchical vs flat key selection" note in its research-connection section if code or a reproduction appears.
- **Confidence:** low
- **Recommendation:** monitor; add as evidence to `block-sparse-attention-pisa.md` (WAIT) and re-judge against the DSA foundation once it is written.

### C-20261109-02 · Anthropic "The components of agent memory": transcripts → notes → wiki representations → injected context, plus offline "dreaming" consolidation (production feature)

- **Class:** B
- **Date discovered:** 2026-11-09
- **Date published:** 2026-11-06
- **Source:** https://www.anthropic.com/research/components-of-agent-memory (verified resolves; HTTP 200)
- **Organization/researchers:** Anthropic (engineering post, Lance Martin).
- **Category:** agents/tools / memory/retrieval
- **What changed:** A lab describes the agent-memory design behind a shipped product feature (Claude Managed Agents memory + "dreams"; docs at https://platform.claude.com/docs/en/managed-agents/dreams, **not fetched**: host outside this session's network allow-list). Memory is a four-level hierarchy (searchable transcripts, memory notes, structured representations such as linked wiki pages, a small always-injected context layer), explicitly excluding weight updates, in-context learning and compaction. "Dreaming" is an offline pass over logs and notes that corrects mistakes, writes notes, builds representations and promotes high-value items into injected context.
- **Technical summary:** Reported results (all **(company claim)**): CL-Bench database task mean reward baseline → with memory, e.g. Opus 5.5 0.655 → 0.838, Sonnet 5 0.306 → 0.610, Sonnet 4.6 0.330 → 0.364 (0.641 with refined prompting); a "record only failures" rule reduced the memory gain (Opus 5 −0.234); on a ~350K-file / ~420M-token corpus, wiki-only accuracy rose from 13% to 48% across 2 → 128 dreaming sessions and wiki + search reached 61% vs 53% search-only, saving ~20K tokens per question; a Harvey LAB restructuring (one page per contract) raised graded facts 41/57 → 57/57. Design guidance: tell the model what memory is for, not how to use it; keep injected context small. Related independent work it cites: Letta/UC Berkeley sleep-time compute https://arxiv.org/abs/2504.13171 (verified) and Google Research's "sleep framework" (2026-10-27, https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/ , verified; note that one *does* update weights, low-rank MoE capacity + distillation, so it is a different mechanism under a similar name).
- **Why it might matter:** First frontier-lab, production-backed account of a non-weight agent memory hierarchy with ablations, giving the agents module a concrete taxonomy (what is memory vs compaction vs ICL vs fine-tuning) and a measurable claim (offline consolidation improves retrieval accuracy and token cost). Offline "sleep/dream" consolidation now appears from three groups (Letta, Google, Anthropic), though with different mechanisms.
- **Evidence of adoption:** shipped in Anthropic's Managed Agents platform and Claude Code memory (company claim); evaluations are Anthropic's own; no independent evaluation read.
- **Major organizations using it:** Anthropic (product); Letta and Google publish related but distinct "sleep-time" methods.
- **Open-source implementation:** example quickstart linked from the post (https://github.com/anthropics/claude-quickstarts/tree/main/managed-agents/knowledge-wiki, not opened); no evaluation code.
- **Paper:** none (engineering post).
- **Code:** none for the evaluations.
- **Relationship to existing course material:** `deferred/agent-personal-memory-state.md` (its "use in a widely used agent framework" trigger is partly relevant, though the mechanism differs from EvoMemState's RL-trained relational state); `deferred/agent-context-compaction.md` (the post draws the memory/compaction boundary explicitly); `deferred/agent-continual-learning.md` (CL-Bench results); agents lessons in the course.
- **Potential course lesson:** possibly a short "agent memory taxonomy" section in the agents module (hierarchy + offline consolidation, with a toy notes/wiki store), only if the weekly review judges the cluster converged; otherwise evidence for the deferred files.
- **Confidence:** medium (the mechanism is described clearly; the numbers are company claims)
- **Recommendation:** monitor; add as evidence to `agent-personal-memory-state.md` and cross-reference from `agent-context-compaction.md`; re-judge at that topic's scheduled review.

### C-20261110-01 · Residual-stream expansion / learned residual mixing (mHC in DeepSeek-V4, Kimi Attention Residuals, Qwen3.8-Next gated multi-branch residual; new independent comparison HiRes)

- **Class:** A
- **Date discovered:** 2026-11-10
- **Date published:** 2026-10-30 (HiRes v1, the trigger); cluster sources 2025-12-31 (mHC v1) to 2026-04 (DeepSeek-V4)
- **Source:** https://arxiv.org/abs/2610.41711 (HiRes, verified HTTP 200); https://arxiv.org/abs/2512.24880 (mHC, DeepSeek-AI, verified); https://arxiv.org/abs/2603.15031 (Attention Residuals, Kimi Team, verified); https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro (model card, fetched directly: "We incorporate mHC to strengthen conventional residual connections")
- **Organization/researchers:** HiRes: Ziteng Wang, Jianqiao Lu, Xinyu Ma, Jintao Zhang, Jun Zhu, Yiyuan Ma, Jianfei Chen (affiliations not on the abstract page; Tsinghua is an inference from the author list). mHC: DeepSeek-AI (Zhenda Xie et al., 21 authors). AttnRes: Moonshot AI Kimi Team (36 authors). Qwen3.8-Next: Qwen Team, Alibaba.
- **Category:** architecture
- **What changed:** The plain residual connection `x + f(x)` (module 05.4, init scaling in 06.2) is being replaced in major open models. Three production-lab variants exist: (1) **mHC** — Hyper-Connections widen the residual stream into n parallel streams mixed by learned matrices; mHC projects the mixing matrix onto doubly stochastic matrices (Sinkhorn-Knopp) to restore the identity-mapping property and stabilise deep training; DeepSeek-V4 ships it (model card, company claim). (2) **AttnRes** — each layer takes a learned, input-dependent softmax over earlier layer outputs instead of a fixed unit-weight sum (Block AttnRes for memory), integrated into Kimi Linear 48B/3B-active and pretrained on 1.4T tokens. (3) Qwen3.8-Next's 4-branch gated residual stream (recorded 2026-09-26, tech report https://arxiv.org/abs/2608.30320). New today: **HiRes**, a simpler independent alternative — narrow PreNorm layers inside blocks, a wider PostNorm residual between blocks, learned projections at block boundaries, compared directly against VWN, mHC and AttnRes.
- **Technical summary:** HiRes (authors' numbers): fitted scaling laws suggest ~1.30x FLOPs efficiency over standard residual addition; on OLMoE-1B-7B pretraining over 500B tokens, lower training loss and higher downstream accuracy; "competitive" with VWN/mHC/AttnRes at lower activation memory and I/O; dense projections and norms only, no custom kernels; no code linked. mHC (DeepSeek): tested at 3B/9B/27B; abstract gives no numbers (secondary sources quote 27B BBH 43.8 → 48.9 HC → 51.0 mHC, not verified against the paper). AttnRes: more uniform output magnitude and gradient distribution across depth and better downstream scores at 48B/1.4T (company claim). Common thread: PreNorm residual sums grow with depth and dilute early layers; the fixes either widen the residual (mHC, HiRes) or reweight it (AttnRes, gated branches). Related C item: Multi-Stream Attention Residuals (https://arxiv.org/abs/2610.13630, daily 2026-10-13).
- **Why it might matter:** The residual connection is one of the few components every course model uses; a change to it adopted by DeepSeek-V4 and explored by Moonshot and Qwen is foundational architecture knowledge, not a niche trick. It is also very teachable on the course GPT: Sinkhorn projection of a 4x4 mixing matrix by hand, residual-norm growth with depth measured on the course model, and a CPU-sized comparison of plain vs widened vs attention-weighted residuals. HiRes gives an independent group's head-to-head of the three production variants.
- **Evidence of adoption:** DeepSeek-V4 (open weights, model card verified) uses mHC (company claim on effect). Kimi AttnRes pretrained at 48B/1.4T (company paper). Qwen3.8-Next preview uses a 4-branch gated residual (company report). Secondary sources say NVIDIA Megatron Bridge implements DeepSeek-V4's 4-stream mHC (`hc_mult = 4`) — **not verified** (docs.nvidia.com is outside this session's network allow-list). No independent reproduction of mHC's gains read; HiRes treats them as baselines, which is partial independent evidence.
- **Major organizations using it:** DeepSeek (production, V4); Moonshot AI (research at production scale); Alibaba Qwen (preview architecture).
- **Open-source implementation:** DeepSeek-V4 inference code/weights on Hugging Face; serving support in vLLM/SGLang for DeepSeek-V4.x (release notes); no reference training implementation verified; no HiRes code.
- **Paper:** https://arxiv.org/abs/2512.24880 ; https://arxiv.org/abs/2603.15031 ; https://arxiv.org/abs/2610.41711
- **Code:** none verified (training side)
- **Relationship to existing course material:** new topic (no registry file; `lookup` for hyper-connection/mHC/AttnRes finds nothing). Builds on `lessons/module-05/lesson-04.md` (residual + LayerNorm block) and `lessons/module-06/lesson-02.md` (1/sqrt(2L) residual init scaling); related to `deferred/massive-activations-weight-decay.md` (residual-stream magnitude) and `deferred/looped-recurrent-depth-lms.md` (looped residual schemes); Qwen3.8-Next already cited in `accepted/hybrid-linear-attention-gated-deltanet.md`.
- **Potential course lesson:** an advanced extension after 05.4 / in the modern-architecture module ("Beyond x + f(x): widened and learned residuals"): PreNorm magnitude growth measured on the course GPT, Hyper-Connections and the Sinkhorn doubly-stochastic projection worked by hand, Block AttnRes as softmax over depth, cost in activation memory/IO. This is a new architecture area, so the weekly review may prefer to ask Tal for a scope decision (as for sparse attention and MTP) rather than ADD directly.
- **Confidence:** medium (adoption by DeepSeek-V4 is documented on the card; the size of the gains rests on company/author numbers)
- **Recommendation:** weekly review: evaluate for ADD as a short architecture extension or raise it as a scope question for Tal; open a topic file either way. Before deciding, read the DeepSeek-V4 report's mHC section and look for an mHC or AttnRes reference implementation and an independent reproduction.

### C-20261111-01 · How much training-inference mismatch can on-policy distillation tolerate? (a "mismatch budget" for OPD; FP8 and async rollouts)

- **Class:** B
- **Date discovered:** 2026-11-11
- **Date published:** 2026-11-08 (v1)
- **Source:** https://arxiv.org/abs/2611.11450 (verified, HTTP 200)
- **Organization/researchers:** Cong Yu, Ziting Wang, Songlin Jiang, Sitong Zhang, Chengliang Chai, Youhe Jiang, Eiko Yoneki, Tuo Shi (affiliations not shown on the abstract page).
- **Category:** RL/post-training
- **What changed:** The course teaches training-inference mismatch for RL (16.4: batch invariance, truncated importance sampling) and on-policy distillation (17.4) separately. This paper measures mismatch for OPD directly: mismatch sources (rollout quantization across three student-teacher pairs; weight staleness and injected weight noise on one pair) are placed on one scale, and training quality is tracked against it.
- **Technical summary:** Authors' findings: OPD tolerates substantial mismatch with little quality loss up to an empirical "mismatch budget", beyond which it degrades; both the quality cost and the mismatch produced by quantization are model-specific, with some layers especially precision-sensitive; OPD stays stable under rollout configurations that destabilize GRPO/REINFORCE; under weight noise the damage comes from shifted generation prefixes, which per-token corrections cannot fix; a lightweight admission check screens rollout configurations against a calibrated budget without full training runs. Model-specific FP8 rollout recipes keep training quality while fitting at least 1.93x the KV-cache tokens of BF16, and FP8 + asynchronous rollouts reach the same quality 1.62x sooner than synchronous BF16 (all authors' numbers).
- **Why it might matter:** Joins two accepted lessons with one measurable claim: the dense per-token teacher signal of OPD makes it more robust to sampler/trainer divergence than sparse-reward RL. That is a concrete, testable contrast for 16.4/17.4, and the "prefix shift that per-token correction cannot fix" point sharpens what truncated IS does and does not repair.
- **Evidence of adoption:** none beyond the paper; single study, days old, no reproduction.
- **Major organizations using it:** none known.
- **Open-source implementation:** https://github.com/benyucong/train-inference-mismatch-opd (exists, Apache-2.0, built on a vendored verl with vLLM rollouts per its README; not run)
- **Paper:** https://arxiv.org/abs/2611.11450
- **Code:** https://github.com/benyucong/train-inference-mismatch-opd
- **Relationship to existing course material:** `accepted/rl-training-inference-mismatch.md` (lesson 16.4) and `accepted/on-policy-distillation.md` (lesson 17.4); related to `deferred/on-policy-distillation-variants.md` (the OPD cluster; the weekly review may fold it in there) and `deferred/rl-mismatch-rounding-location-alignment.md`.
- **Potential course lesson:** none on its own. If reproduced, a short research-connection note in 16.4 or 17.4 ("OPD's dense teacher signal tolerates more rollout mismatch than GRPO; prefix shift is not fixed by per-token IS"), possibly a toy experiment on the existing `code/experiments/rl_mismatch.py`.
- **Confidence:** low (single paper; authors' numbers)
- **Recommendation:** monitor; add as evidence to `on-policy-distillation-variants.md` (or cross-reference from `rl-training-inference-mismatch.md`); re-judge if an independent replication or adoption in verl/TRL appears.

### C-20261112-01 · Are dense (Monte-Carlo process) rewards worth their tokens in GRPO? A matched-compute comparison that reverses the ranking

- **Class:** B
- **Date discovered:** 2026-11-12
- **Date published:** 2026-11-10 (v1)
- **Source:** https://arxiv.org/abs/2611.13560 (verified, HTTP 200)
- **Organization/researchers:** Vikash Singh, Weicong Chen, Debargha Ganguly, Yanyan Zhang, Mohsen Hariri, … Vipin Chaudhary (19 authors; affiliations not shown on the abstract page).
- **Category:** RL/post-training
- **What changed:** Process/dense rewards for RL on reasoning (taught in 17.2, Process reward & DeepSeekMath) are usually compared with outcome-reward GRPO at equal *training steps*. This paper compares them at equal *generated tokens and GPU-hours*, and reports that the ranking flips: in the authors' setting the Monte-Carlo step-scoring continuations are about 91% of all generated tokens.
- **Technical summary:** Authors' claims: a matched-compute protocol (same token budget and GPU-hours per method); 1,668 runs over six models; "matching compute reverses the ranking"; on Qwen2.5-1.5B the Monte-Carlo dense reward ends 14.3 points below standard GRPO at equal tokens. Separately, selecting prompts whose predicted success rate is near 1/2 (maximum Bernoulli reward variance, i.e. largest GRPO advantage signal) improves accuracy at matched compute by 1.7 points on Qwen3-0.6B and 0.8 on Qwen2.5-1.5B. Conclusion: training interventions should be validated at matched compute.
- **Why it might matter:** A clean, teachable methodology point for 17.2 and 16.x: a "better" reward is only better per unit of compute, and MC process rewards spend most of the rollout budget on scoring. The p≈1/2 prompt-selection result connects directly to GRPO's group-normalized advantage (zero signal when all samples agree). Both are easy to illustrate with the course's toy GRPO code.
- **Evidence of adoption:** none; single study, two days old. Large run count is the main strength. Code repository exists but is a placeholder ("CODE WILL BE AVAILABLE SOON", no license).
- **Major organizations using it:** none known.
- **Open-source implementation:** https://github.com/vicky157/grpo-validation (exists; placeholder README only, no code yet)
- **Paper:** https://arxiv.org/abs/2611.13560
- **Code:** https://github.com/vicky157/grpo-validation (empty placeholder)
- **Relationship to existing course material:** `lessons/module-17/lesson-02.md` (process reward / Math-Shepherd-style MC step labels) and the GRPO lessons in module 16; related registry files `accepted/mlperf-post-training-rlvr.md`, `deferred/gvpo.md`. No topic file for "compute-matched comparison of RL rewards"; `lookup "matched compute"` finds only `accepted/muon-optimizer.md` (budget-matched benchmark, a parallel methodology point).
- **Potential course lesson:** none on its own. If code lands or the result is reproduced: a "research connection" / common-mistakes note in 17.2 ("compare reward schemes at matched generated tokens, not matched steps") and a one-paragraph note on p≈1/2 prompt filtering beside GRPO advantage normalization.
- **Confidence:** low (single paper; authors' numbers; code not released)
- **Recommendation:** monitor; re-judge when code is released or an independent matched-compute replication appears. The weekly review may fold the prompt-difficulty point into any existing difficulty-filtering/curriculum topic.
