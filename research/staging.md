# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20260928-01 · PISA: hierarchical block-sparse attention with O(N log N) complexity

- **Class:** B
- **Date discovered:** 2026-09-28
- **Date published:** 2026-09-25
- **Source:** https://arxiv.org/abs/2609.31093 "Block Sparse Attention with Log-Linear Complexity"
- **Organization/researchers:** Bohao Tang, Zhen Qin, Yuqi Pan, Zheng Li, Pengfei Liu (academic; affiliations not stated in the abstract page)
- **Category:** attention
- **What changed:** Proposes PISA, a block-sparse attention variant that replaces exhaustive query-block-pair scoring (still quadratic in standard block-sparse attention) with a pyramid top-K selection: a coarse-to-fine key hierarchy with LogSumExp scoring over bounded candidate sets at each of O(log N) levels, progressively narrowing to the finest level. Claims overall O(N log N) complexity. Ships hardware-aware Triton kernels (train + inference) that fuse hierarchical routing and LogSumExp scoring without materializing the full query-key score matrix.
- **Technical summary:** A genuinely different sparsification strategy from the decode-time KV-block-skipping approach already in `research/deferred/sparse-attention-decoding-ffd.md` (that one skips blocks at decode time via a quantized scan; this one restructures scoring itself, train and inference, into a hierarchy). Reported results: comparable accuracy to dense/baseline block-sparse attention on commonsense-reasoning benchmarks, better on retrieval tasks.
- **Why it might matter:** If reproduced, an O(N log N) sparse-attention scheme with fused Triton kernels is directly relevant to the course's attention/long-context material (module 12) and would be a natural "from-scratch mechanism" companion to the FlashAttention/Gated-DeltaNet lessons already there.
- **Evidence of adoption:** none — single paper, no independent reproduction, no adoption by any inference stack (vLLM/SGLang/TensorRT-LLM) found.
- **Major organizations using it:** none.
- **Open-source implementation:** not confirmed (page did not surface a repo link).
- **Paper:** https://arxiv.org/abs/2609.31093
- **Code:** not confirmed
- **Relationship to existing course material:** related but distinct from `research/deferred/sparse-attention-decoding-ffd.md` (WAIT); also adjacent to `research/accepted/hybrid-linear-attention-gated-deltanet.md` (module-12 lesson-05) as another long-context efficiency angle.
- **Potential course lesson:** if it holds up, an extension to module-12's attention-efficiency material (own lesson or a section next to Gated DeltaNet) once Triton code is available to implement from scratch.
- **Confidence:** low
- **Recommendation:** monitor — wait for a public implementation and independent benchmark before the weekly review considers it; classic single-paper claim at this stage.

### C-20260928-02 · Recursive self-improvement via on-policy distillation with a co-evolving teacher (DCE + SRCL)

- **Class:** B
- **Date discovered:** 2026-09-28
- **Date published:** 2026-09-25
- **Source:** https://arxiv.org/abs/2609.30652 "Recursive Self-Improvement via On-Policy Distillation for Reasoning"
- **Organization/researchers:** Shangjian Yin (lead) with nine co-authors (affiliations not stated in the abstract page)
- **Category:** RL/post-training
- **What changed:** Two techniques layered on on-policy distillation for reasoning: Dynamic Co-Evolution (DCE), where the "teacher" model is not frozen but improves alongside the student across recursive rounds, and Self-Refined Concise Learning (SRCL), which trains on shorter verified rewrites of correct solutions to counter length blow-up. Reports 65.97% Average@12 on Qwen3-8B across four math benchmarks, +35.62 points over their On-Policy Self-Distillation (OPSD) baseline, with 7.80% shorter outputs than DCE alone.
- **Technical summary:** Distinct from the course's already-accepted GRPO-stability material (`research/accepted/grpo-stability-fixes.md`): this is a distillation-based recursive self-improvement loop, not a policy-gradient RL-from-verifier-reward method, and it only compares against an on-policy-distillation baseline (OPSD), not GRPO/PPO/DAPO directly, so its relative standing against the course's existing RL post-training material is unclear.
- **Why it might matter:** "Teacher co-evolves with student" is a structurally different recursive-improvement idea from standard RLVR loops (which use a frozen verifier, not a co-evolving teacher); worth tracking as a second post-training paradigm distinct from what's taught in module-17-adjacent content.
- **Evidence of adoption:** none — single paper, results only on Qwen3-8B math benchmarks, no comparison to GRPO/DAPO baselines, no third-party reproduction.
- **Major organizations using it:** none.
- **Open-source implementation:** not confirmed.
- **Paper:** https://arxiv.org/abs/2609.30652
- **Code:** not confirmed
- **Relationship to existing course material:** adjacent to `research/accepted/grpo-stability-fixes.md` and module-17 reasoning/RL content, but a different mechanism (distillation-based, not RL-from-reward); new topic, no existing registry entry (`lookup "on-policy distillation reasoning"` returned no match).
- **Potential course lesson:** none yet — needs a GRPO/DAPO baseline comparison and independent reproduction before it's teachable as more than "an alternative worth knowing exists."
- **Confidence:** low
- **Recommendation:** monitor — interesting mechanism (co-evolving teacher) but the missing RL-baseline comparison and single-paper status keep it well short of the weekly-review bar.

### C-20260928-03 · SinkProbe: published attention-sink fixes (gated attention, Kimi Delta Attention) fail to reproduce at smaller scale

- **Class:** B
- **Date discovered:** 2026-09-28
- **Date published:** 2026-09-08 (v1); revised 2026-09-18 (v2)
- **Source:** https://arxiv.org/abs/2609.08574 "Do New Attention Mechanisms Actually Fix Attention Sinks at Million-Token Context?"
- **Organization/researchers:** not stated on the abstract page (academic)
- **Category:** interpretability | attention
- **What changed:** Builds SinkProbe, a diagnostic suite (sink mass, massive activation, position-resolved recall, recency gap) and applies it to four small models differing only in token-mixing/depth. Finds: (1) the widely cited "gated attention cuts first-token attention from 46.7% to 4.8%" result does not reproduce at the authors' scale; (2) sink mass, activation magnitude, and positional bias move independently rather than together as often assumed; (3) their evidence points to the training objective, not the architecture, as the source of the sink phenomenon. Directly examines mechanisms including gated attention, Kimi Delta Attention, and "Attention Residuals" — the same family of ideas behind the hybrid linear-attention architecture already taught in the course.
- **Technical summary:** This is exactly the kind of independent-reproduction-failure evidence the course's dedup rule calls out as noteworthy: it doesn't just repeat a claim, it empirically challenges published claims about mechanisms adjacent to `lessons/module-12/lesson-05.md` (Gated DeltaNet + full-attention hybrid, `research/accepted/hybrid-linear-attention-gated-deltanet.md`). Flagged here even though it's older than the usual ~48h window because it only surfaced in today's search and bears on already-accepted course content.
- **Why it might matter:** If it holds up, it's a methodological caution worth appending to the existing Gated DeltaNet lesson (scale-dependence of claimed architectural fixes for attention sinks; training-objective vs. architecture attribution) rather than a new lesson on its own.
- **Evidence of adoption:** none — single paper (with a v2 revision), academic, no independent replication found of *this* paper's negative result either.
- **Major organizations using it:** none.
- **Open-source implementation:** not confirmed.
- **Paper:** https://arxiv.org/abs/2609.08574
- **Code:** not confirmed
- **Relationship to existing course material:** directly relevant to `research/accepted/hybrid-linear-attention-gated-deltanet.md` and `lessons/module-12/lesson-05.md` — a scale-dependence/reproduction caveat on the mechanisms that lesson teaches, not a duplicate of the original claim.
- **Potential course lesson:** not a new lesson — a candidate caveat/addendum to the existing module-12 lesson-05, to be judged by weekly review against the lesson's current claims.
- **Confidence:** low-medium (rigorous diagnostic methodology; small-scale-only evaluation limits generality)
- **Recommendation:** weekly review should check whether `lessons/module-12/lesson-05.md` states the "46.7%→4.8%" gated-attention result as settled fact anywhere, and if so, consider a caveat citing this failure-to-reproduce at smaller scale plus the training-objective-vs-architecture distinction.

### C-20260929-01 · ToolSearcher: Optimizing Tool Selection at Scale via Reinforcement Learning

- **Class:** B
- **Date discovered:** 2026-09-29
- **Date published:** 2026-09-25
- **Source:** https://arxiv.org/abs/2609.30906 "ToolSearcher: Optimizing Tool Selection at Scale via Reinforcement Learning"
- **Organization/researchers:** Zhenlong Dai, Xujie Song, Zitong Wang, Tong Niu, Jian Liu, Weiqiang Wang, Xiu Tang, Sai Wu, Chang Yao, Jingyuan Chen (affiliations not stated on the abstract page); accepted at NeurIPS 2026
- **Category:** agents/tools
- **What changed:** RL framework for large-scale tool *selection* (as distinct from tool *use*): category-constrained discrimination narrows the candidate pool before full scoring, event-level search modeling treats each search step as its own RL event, and trajectory-aligned credit allocation propagates reward across a full multi-step, composed tool-use trajectory rather than only the final answer.
- **Technical summary:** Distinct from the course's Toolformer material (module 18): this targets the search/selection step over a large, ambiguous tool catalog, not deciding whether/how to insert a call. Peer-reviewed (NeurIPS 2026) but no released code found; abstract cites "strong baselines" without naming them or giving numbers.
- **Why it might matter:** agentic systems with large tool/API catalogs need more than embedding top-k retrieval to pick the right tool; a peer-reviewed, purpose-built RL method for that step is a plausible module-18 extension if it holds up.
- **Evidence of adoption:** none beyond peer review — no confirmed open-source implementation, no third-party reproduction, no adoption in an agent framework.
- **Major organizations using it:** none confirmed.
- **Open-source implementation:** not confirmed.
- **Paper:** https://arxiv.org/abs/2609.30906
- **Code:** not confirmed
- **Relationship to existing course material:** adjacent to module-18 (Toolformer, `papers/index.md`) and its tool-use lessons; new topic, no registry entry (`lookup "ToolSearcher tool selection"` → no match).
- **Potential course lesson:** possible module-18 extension on RL-trained tool selection for large tool catalogs, contingent on a public implementation and concrete numbers.
- **Confidence:** low-medium (peer-reviewed, but no adoption evidence and abstract omits concrete results)
- **Recommendation:** monitor — good peer-review signal; wait for released code/results and any real agent-framework adoption before weekly review considers it.

### C-20260929-02 · H2S (Highlight-Then-Summarize): trained evidence compression for long-context QA

- **Class:** B
- **Date discovered:** 2026-09-29
- **Date published:** 2026-09-25
- **Source:** https://arxiv.org/abs/2609.31382 "Highlight-Then-Summarize: Learning to Compress Evidence for Long-Context Understanding"
- **Organization/researchers:** Zhaoyuan Xia, Qinghongbing Xie, Yung Xiang Hue, Jianguang Jiang, Gaofeng Lu, Zhenyu Jiao, Xing Yuan, Dai Dai, Tong Mo, Long Zeng — Peking University, Baidu Inc., Tsinghua University
- **Category:** long-context | reasoning
- **What changed:** Two-stage pipeline for long-document QA: (1) extract source-grounded, question-relevant evidence spans, (2) compress them into a compact, question-conditioned summary, then answer from that — rather than answering over the raw context or a generic summary. Trained on a new H2S-Dataset (6,647 examples, avg. 43.9K tokens) with an RL stage (H2S-RL). Reports their 14B model beating larger open-source models on 7 long-context benchmarks with shorter outputs.
- **Technical summary:** A trained, verifiable evidence-selection step, distinct from both "extend the context window" (architectural) and generic retrieve-then-summarize pipelines. Code and dataset are public, which makes this easier to check than most B items.
- **Why it might matter:** the course's long-context material (module 12) currently covers architectural context-extension, not this kind of trained evidence-compression step for the "context doesn't fit / isn't all relevant" problem.
- **Evidence of adoption:** none — single paper, no independent reproduction, no adoption by a major lab or serving framework.
- **Major organizations using it:** none (Baidu is a co-affiliation on the paper, not a deployment).
- **Open-source implementation:** https://github.com/X-Luffy/Highlight-Then-Summarize
- **Paper:** https://arxiv.org/abs/2609.31382
- **Code:** https://github.com/X-Luffy/Highlight-Then-Summarize
- **Relationship to existing course material:** new — module-12 covers long-context architecture, not trained evidence-compression pipelines; no registry entry.
- **Potential course lesson:** possible module-12/13 extension or exercise on trained evidence compression for long-context QA, since code+dataset are actually runnable.
- **Confidence:** low-medium (reproducible artifact is a plus; still single paper, no third-party validation)
- **Recommendation:** monitor — released code+dataset make this worth a quick reproduction check if module-12/13 extensions come up at weekly review.

### C-20260929-03 · Self-play pretraining with zero natural data (generator/learner over a universal Turing machine)

- **Class:** B
- **Date discovered:** 2026-09-29
- **Date published:** 2026-09-24
- **Source:** https://arxiv.org/abs/2609.30063 "Self-Play Pretraining with Zero Data"
- **Organization/researchers:** Aditya Cowsik, Kfir Dolev, Michael Y. Li, G. Bruno De Luca, Nourya Cohen, Noah D. Goodman, Yoav Levine (Noah Goodman: Stanford)
- **Category:** training | other (pretraining paradigm)
- **What changed:** Pretrains an autoregressive "learner" with no natural data at all. A "generator" proposes programs for a universal Turing machine that emit byte sequences; the generator is RL-trained to stay at the frontier of the learner's current ability (an automatic curriculum), and the learner predicts those bytes. Claims zero-shot performance on natural-language datasets improves predictably with more compute, that the learner shows in-context learning, and that it transfers to natural data despite never training on any.
- **Technical summary:** Structurally different from ordinary synthetic-data pretraining (which still generates natural-language-like text via an LLM): here neither model ever sees natural data; the loop resembles AlphaZero-style curriculum self-play applied to next-byte prediction. If the transfer claims hold, they bear on why pretraining works at all (is exposure to natural-language structure necessary, or just to sufficiently complex/compressible sequences?).
- **Why it might matter:** touches the course's pretraining-fundamentals framing directly, from a team including a well-known researcher (Noah Goodman); a genuinely different angle rather than an incremental data-pipeline tweak.
- **Evidence of adoption:** none — single paper, extraordinary claim, no code released, no independent reproduction. Needs real scrutiny of the transfer-to-natural-data evaluation before taking at face value.
- **Major organizations using it:** none.
- **Open-source implementation:** not confirmed.
- **Paper:** https://arxiv.org/abs/2609.30063
- **Code:** not confirmed
- **Relationship to existing course material:** new — touches "why pretraining works" framing in early modules; no registry entry.
- **Potential course lesson:** none yet — far too early/unreplicated; at most a research-connection pointer in an existing pretraining lesson if it survives scrutiny.
- **Confidence:** low (striking claim, single paper, no code, no reproduction — treat skeptically)
- **Recommendation:** monitor closely, do not act on; classic single-extraordinary-claim paper. Revisit if code is released or the claim is independently reproduced/scrutinized.

### C-20260930-01 · LeapQuant: training-free 8-bit quantization of linear-attention recurrent state

- **Class:** B
- **Date discovered:** 2026-09-30
- **Date published:** 2026-09-29
- **Source:** https://arxiv.org/abs/2609.38166 "LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization"
- **Organization/researchers:** Yi Pan, Haocheng Xi, Kan Zhu, Xingyang Li, Yibo Wu, Mayank Mishra, Hongtao Zhang, William X. Zheng, Baris Kasikci, Song Han, Kurt Keutzer, Rishabh Iyer, Ion Stoica (MIT/UC Berkeley-affiliated systems researchers; affiliations not stated on the abstract page)
- **Category:** quantization
- **What changed:** A training-free method for quantizing the recurrent state carried by linear-attention layers (the same kind of state the course's Gated DeltaNet lesson introduces) to 8 bits at inference time. Two mechanisms: "per-window" quantization that re-quantizes the state only once per window of tokens instead of every step, and high-precision "compensator tokens" that absorb outliers so the rest of the state can stay low-bit. Evaluated on real hybrid linear-attention models (Qwen, Kimi-Linear, GLM families).
- **Technical summary:** Reports kernel-level speedups of 2.05-3.70x and 1.47x end-to-end inference speedup on NVIDIA B200, RTX PRO 6000, and RTX 5090, with accuracy close to unquantized state. Training-free (no fine-tuning needed), which matters for adoption since it can be applied to already-released checkpoints. No public code found yet (paper is one day old).
- **Why it might matter:** The course just added a hybrid linear-attention lesson (`lessons/module-12/lesson-05.md`, Gated DeltaNet + full attention, accepted 2026-09-27). Serving-time quantization of exactly the recurrent state that lesson teaches is a natural, concrete extension — and the author list (Song Han, Ion Stoica, Kurt Keutzer, Baris Kasikci) has a strong track record of techniques that become standard practice in serving stacks (e.g., SmoothQuant, AWQ, vLLM/Ray ecosystem).
- **Evidence of adoption:** none yet — paper submitted 2026-09-29, no released code found, no third-party reproduction, no serving-framework (vLLM/SGLang) integration confirmed.
- **Major organizations using it:** none confirmed yet.
- **Open-source implementation:** not confirmed (not yet found; may not be released).
- **Paper:** https://arxiv.org/abs/2609.38166
- **Code:** not confirmed
- **Relationship to existing course material:** directly extends `lessons/module-12/lesson-05.md` and `research/accepted/hybrid-linear-attention-gated-deltanet.md` (recurrent-state quantization for the same architecture family); new topic, no registry entry.
- **Potential course lesson:** possible module-12 addendum/exercise on quantizing the Gated DeltaNet recurrent state for serving, once code is available to implement/verify from scratch.
- **Confidence:** low-medium (strong author pedigree and concrete measured speedups on real GPUs; no code or independent validation yet)
- **Recommendation:** monitor closely — watch for released code; also see C-20260930-02 (STEPQuant), an independent same-day paper attacking the identical problem, which strengthens the signal that this is a real, currently-unsolved gap in hybrid linear-attention serving.

### C-20260930-02 · STEPQuant: error-aware precision allocation for delta-rule recurrent state

- **Class:** B
- **Date discovered:** 2026-09-30
- **Date published:** 2026-09-29
- **Source:** https://arxiv.org/abs/2609.38169 "STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization"
- **Organization/researchers:** Bingchen Yao, Haobo Xu, Haokun Lin, Yichen Wu, Ziyu Guo, Renrui Zhang, Zhichao Lu, Zhenan Sun, Ying Wei (affiliations not stated on the abstract page)
- **Category:** quantization
- **What changed:** Independent, same-day paper on the same underlying problem as C-20260930-01 (quantizing delta-rule/linear-attention recurrent state) but a different method: allocates bit-precision based on two axes — temporally, how long an error in a given state entry persists across future decoding steps, and spatially, which key rows/value columns matter most — then jointly optimizes key-row and value-column quantization scales, rather than LeapQuant's per-window-leap + compensator-token approach.
- **Technical summary:** Evaluated on Qwen3.8-27B and Kimi-Linear-48B-A3B-Instruct. Reports matching FP32-state accuracy at a 6-bit budget and beating uniform INT8 at 4 bits, with up to 68.7% serving-memory reduction. No public code found yet.
- **Why it might matter:** Same rationale as LeapQuant — directly extends the course's newly-accepted Gated DeltaNet/hybrid-linear-attention material with a memory-focused (rather than speed-focused) quantization angle, evaluated on some of the same real production model families.
- **Evidence of adoption:** none yet — single paper, no code, no reproduction, no serving-stack integration.
- **Major organizations using it:** none confirmed.
- **Open-source implementation:** not confirmed.
- **Paper:** https://arxiv.org/abs/2609.38169
- **Code:** not confirmed
- **Relationship to existing course material:** directly extends `lessons/module-12/lesson-05.md` and `research/accepted/hybrid-linear-attention-gated-deltanet.md`, same as C-20260930-01, via a different memory-bit-allocation mechanism; new topic, no registry entry.
- **Potential course lesson:** if a public implementation appears, a module-12 addendum comparing LeapQuant's (speed-oriented) and STEPQuant's (memory-oriented) approaches to quantizing the same recurrent state would be a strong from-scratch exercise.
- **Confidence:** low-medium (concrete numbers on real production model families; no code or independent validation yet)
- **Recommendation:** monitor alongside C-20260930-01 as a pair — two independent groups converging on quantizing the exact state the course just started teaching is a meaningful signal even though neither has adoption evidence yet; revisit together once either publishes code.

### C-20260930-03 · Delta-Matching: closing the accuracy gap for native FP8 8-bit LLM training

- **Class:** B
- **Date discovered:** 2026-09-30
- **Date published:** 2026-09-29
- **Source:** https://arxiv.org/abs/2609.37852 "Delta-Matching: Closing the Final Gap of Native 8-bit Training for LLMs"
- **Organization/researchers:** Haozhan Tang, Hao Kang, Han Cai, Song Han, Chenyan Xiong (Song Han's group; MIT/CMU-affiliated, affiliations not stated on the abstract page)
- **Category:** training
- **What changed:** Identifies that native FP8 attention training degrades at larger model scale because forward/backward numerical inconsistencies produce a "stale delta" that breaks the softmax gradient's zero-row-sum invariant. Delta-Matching restores that invariant, enabling native block-scaled FP8 training without architecture changes, smaller batches, or auxiliary forward passes — unlike prior FP8-training workarounds.
- **Technical summary:** Tested at 1.67B and 5.29B parameters, reportedly matching BF16/FP32 mixed-precision training quality. Authors state they will release implementation, trained models, and training-data recipes (not yet available as of paper submission). Distinct from the course's existing mixed-precision material: this targets native low-precision training (compute + memory savings during pretraining itself), not post-training quantization for serving.
- **Why it might matter:** If it holds and ships code, native FP8 training that "just works" without architectural workarounds would be directly relevant to the course's training-efficiency material — from a lab (Song Han's) with a strong history of techniques that become standard (also a co-author on C-20260930-01/LeapQuant, suggesting a research program specifically on production-grade low-precision LLM systems).
- **Evidence of adoption:** none yet — single paper one day old, code promised but not released, no independent reproduction, no adoption in a training framework (Megatron-LM, TorchTitan, etc.).
- **Major organizations using it:** none confirmed.
- **Open-source implementation:** promised, not yet available.
- **Paper:** https://arxiv.org/abs/2609.37852
- **Code:** not confirmed (authors state intent to release)
- **Relationship to existing course material:** new — no existing lesson on native FP8 pretraining; no registry entry.
- **Potential course lesson:** possible addition to the training-efficiency/mixed-precision material once code and independent scale-up results are available; only tested to 5.29B params so far, well below frontier scale.
- **Confidence:** low (promising mechanism and credible authors, but no code yet and only mid-scale results)
- **Recommendation:** monitor — revisit once code/checkpoints are released; check whether results hold at larger scale before considering for the course.

### C-20261001-01 · Periodic Weak Spots: phase sensitivity from chunked KV-cache compression

- **Class:** B
- **Date discovered:** 2026-10-01
- **Date published:** 2026-09-28
- **Source:** https://arxiv.org/abs/2609.36322 "Periodic Weak Spots: Phase Sensitivity from Chunked KV-Cache Compression"
- **Organization/researchers:** Xingyu Zhu, Pu (Luke) Yi, Ziheng Cheng, Ang Lv, Jing Liu, Lexing Ying, Yiyuan Ma, Xin Dong (affiliations not stated on the abstract page)
- **Category:** long-context | evaluation | interpretability
- **What changed:** Shows that chunked KV-cache compression (used to cut memory during long-context inference) introduces *periodic* retrieval disparities tied to token position within a compression chunk, not just overall degradation — "the same information can be easy to retrieve at one phase and difficult at another," with gaps up to 40 percentage points in open-weight models tested.
- **Technical summary:** The authors pretrain transformers with several chunked-compression designs and run mechanistic/gradient-flow analysis, finding that different attention heads specialize asymmetrically to different phases within a compression chunk, and that this phase specialization emerges naturally during training rather than being an artifact of one architecture. They argue average benchmark scores on compressed models hide these positional weak spots and that evaluation must probe multiple phases.
- **Why it might matter:** The course's long-context/efficiency material (module 12: Gated DeltaNet, FlashAttention, KV-cache topics) teaches compression techniques; this is a concrete "what can go wrong" / evaluation-methodology finding — a good debugging-exercise or common-mistake candidate for any lesson that teaches or benchmarks KV-cache compression, independent of whether the specific technique itself gets added.
- **Evidence of adoption:** none — single paper, own pretraining experiments only, no third-party reproduction, no adoption by an inference stack.
- **Major organizations using it:** none.
- **Open-source implementation:** not confirmed on the abstract page.
- **Paper:** https://arxiv.org/abs/2609.36322
- **Code:** not confirmed
- **Relationship to existing course material:** adjacent to `lessons/module-12/lesson-05.md` (Gated DeltaNet / long-context efficiency) and `papers/index.md`'s MLA/KV-cache-compression entry, but a distinct evaluation-methodology finding, not a technique already taught; `lookup "chunked KV cache"` found no exact match.
- **Potential course lesson:** not a standalone lesson; a candidate "common mistake" / debugging-exercise addition to module-12's KV-cache material if the weekly review wants an evaluation-rigor callout, once independently checked.
- **Confidence:** low
- **Recommendation:** monitor — genuinely new mechanistic claim but single-paper; wait to see whether others confirm phase-sensitivity in production compression schemes (e.g. vLLM/SGLang's own KV-cache compression) before treating it as teachable.

### C-20261001-02 · Scaling properties of same-family on-policy distillation

- **Class:** B
- **Date discovered:** 2026-10-01
- **Date published:** 2026-09-26
- **Source:** https://arxiv.org/abs/2609.32722 "Scaling Properties of Same-Family On-Policy Distillation"
- **Organization/researchers:** Yuntai Bao, Qinfeng Li, Guoqing Jiang, Liwei Chen, Zhiheng Qin, Xuanping Li, Wenqi Zhang, Xuhong Zhang (affiliations not stated on the abstract page)
- **Category:** RL/post-training
- **What changed:** Studies how RL-driven reasoning transfers across model sizes via on-policy distillation, across weak-to-strong, same-base, and strong-to-weak teacher/student pairs. Finds an early "useful-transfer regime" (held-out accuracy grows linearly with divergence from student init), that smaller students can beat larger teachers, and a power-law relating peak student performance to model sizes and teacher quality — notably, peak score improves with teacher scale only up to roughly the student's own scale, so a stronger teacher alone doesn't keep buying supervision value.
- **Technical summary:** A scaling-law study of the on-policy-distillation paradigm itself (not a new training technique), complementary to — but methodologically distinct from — the DCE/SRCL co-evolving-teacher recursive self-improvement paper already staged as `C-20260928-02`. Together with two other on-policy/self-distillation papers surfacing this same week ("OPSRD: On-Policy Self-Role Distillation", arXiv 2609.39884; "Better Supervision Is Nearby: Neighborhood On-Policy Self-Distillation", arXiv 2609.39687 — recorded as C items in `daily/2026-10-01.md`, not independently verified in depth), this is now a visible cluster of independent groups working on on-policy distillation as an alternative/complement to RLVR.
- **Why it might matter:** If the "teacher scale saturates near student scale" finding holds up, it is directly actionable guidance for anyone running distillation-based post-training (course's module-17-adjacent RL/post-training material) — a concrete scaling rule, not just a benchmark win.
- **Evidence of adoption:** none — single paper, no comparison against GRPO/DAPO baselines, no third-party reproduction. (company claim not applicable — appears academic.)
- **Major organizations using it:** none confirmed.
- **Open-source implementation:** not confirmed on the abstract page.
- **Paper:** https://arxiv.org/abs/2609.32722
- **Code:** not confirmed
- **Relationship to existing course material:** part of the same emerging topic as `C-20260928-02` (staged, on-policy distillation / recursive self-improvement) and `research/deferred/gvpo.md` (WAIT, on-policy distillation mentioned as an extension); `lookup "on-policy distillation"` confirms both.
- **Potential course lesson:** none yet; if the scaling-law finding is reproduced, a candidate addition to module-17's RL/post-training material as a "how much does distillation buy you" scaling note.
- **Confidence:** low
- **Recommendation:** monitor alongside C-20260928-02 — the cluster of independent on-policy-distillation papers this week is worth flagging to the weekly review as a trend to watch, even though no single paper here clears the bar alone.

### C-20261001-03 · ReCAP: persistent context graphs for LLM-agent memory compaction

- **Class:** B
- **Date discovered:** 2026-10-01
- **Date published:** 2026-09-30
- **Source:** https://arxiv.org/abs/2609.40118 "Persistent Context Graphs for Efficient Memory Compaction in LLM Agents"
- **Organization/researchers:** Jingbo Yang, Kwei-Herng Lai, Xiaowen Wang, Zhaoxuan Tan, Pei Zhou, Mengting Wan, Yaar Harari, Evgeniy Gabrilovich, Shiyu Chang (affiliations not stated on the abstract page)
- **Category:** agents/tools | memory/retrieval
- **What changed:** Proposes ReCAP, a memory-compaction method for long-running LLM agents that replaces model-based summarization with a lightweight, persistent context graph storing attention-derived importance scores and dependency links; at query time it ranks/restores context from the graph combining stored importance with relevance signals, with no additional model calls for the selection step itself.
- **Technical summary:** Reports ~95% latency reduction for compaction+restoration versus summarization-based memory methods, roughly half the historical context retained on some benchmarks, and up to +41.2 percentage points accuracy on code-related agent tasks versus a full-history baseline.
- **Why it might matter:** The course's agent material (module 18 tool use, module 19 SWE-agent) doesn't yet teach a memory-compaction mechanism beyond basic context handling; a no-extra-inference-call compaction technique with a large claimed accuracy delta on code tasks would be directly relevant if it holds up, and it's architecturally simple enough to implement from scratch per the course's "mechanism before framework" rule.
- **Evidence of adoption:** none — single paper, authors' own benchmarks only, no third-party reproduction or production adoption found.
- **Major organizations using it:** none confirmed (author affiliations not stated on the abstract page, though some author names suggest a possible industry-research origin — not verified).
- **Open-source implementation:** not confirmed on the abstract page.
- **Paper:** https://arxiv.org/abs/2609.40118
- **Code:** not confirmed
- **Relationship to existing course material:** adjacent to `lessons/module-18/lesson-02.md` and `lessons/module-19/lesson-01.md` (agent tool-use/SWE-agent); `lookup "memory compaction"` and `"context graph agent"` found no existing registry/course entry — new topic.
- **Potential course lesson:** if reproduced, a candidate extension to module-18/19's agent material on long-running-agent memory management.
- **Confidence:** low
- **Recommendation:** monitor — strong claimed deltas but single-paper, unverified affiliations/adoption; revisit if an open implementation or independent benchmark appears.

### C-20261002-01 · Proof that RLVR (not SFT) permits unbounded CoT language drift, with a performance/monitorability tradeoff

- **Class:** B
- **Date discovered:** 2026-10-02
- **Date published:** 2026-10-01
- **Source:** https://arxiv.org/abs/2610.02015 "On Language Drift during RLVR Post-Training"
- **Organization/researchers:** Michael Sullivan, Alexander Koller (Koller: established computational-linguistics researcher, Saarland University; affiliation not stated on the abstract page itself)
- **Category:** RL/post-training | interpretability
- **What changed:** Proves theoretically that RLVR optimization pressure permits *unbounded* language drift in chain-of-thought (unusual/nonsensical language use), while supervised fine-tuning does not. Shows empirically that drift specifically arises during RLVR on genuinely novel reasoning tasks (where the target behavior can't be elicited from the base model). Further proves that language drift cannot be constrained without constraining expected reward — i.e. CoT monitorability cannot be improved without hurting RLVR performance at the frontier.
- **Technical summary:** A theory-plus-experiment paper, not a new training technique. The theoretical result gives a mechanistic explanation (grounded in RLVR's optimization pressure vs. SFT's) for an already-observed phenomenon (frontier labs have reported CoTs drifting into odd language), and derives a hard performance/monitorability tradeoff rather than proposing a mitigation.
- **Why it might matter:** Directly relevant to the course's RL/post-training material (module 17, `lessons/frontier/update-02.md`'s RLVR content) and to CoT-faithfulness/monitorability framing used in safety-adjacent discussion — a concrete theoretical result (not speculation) about a known, previously only empirically-observed phenomenon.
- **Evidence of adoption:** none — single paper, one day old, no independent replication of the theoretical proofs or empirical claims yet.
- **Major organizations using it:** none; frames itself around a phenomenon industry labs have informally reported, but that reporting is not evidence for *this* paper's specific theoretical claims.
- **Open-source implementation:** not applicable (theory + small-scale experiments; no code link found on the abstract page).
- **Paper:** https://arxiv.org/abs/2610.02015
- **Code:** not confirmed
- **Relationship to existing course material:** extends the course's existing RLVR content (module 17, `lessons/frontier/update-02.md`) with a theoretical account of a failure mode not currently discussed there; `lookup "language drift RLVR"` found no existing match — new topic.
- **Potential course lesson:** possible addendum to module-17's RLVR material (or `lessons/frontier/`) on CoT language drift as a concrete, provable performance/monitorability tradeoff, once the proofs get scrutiny.
- **Confidence:** low-medium (rigorous theory from a credible academic; single paper, unreplicated)
- **Recommendation:** monitor — a genuinely novel theoretical contribution on a real, previously anecdotal phenomenon; worth the weekly review's attention as a possible RLVR-caveat addendum, but not yet independently checked.

### C-20261002-02 · GMC-GRPO: tighter convergence bound for asynchronous GRPO under stale rollouts

- **Class:** B
- **Date discovered:** 2026-10-02
- **Date published:** 2026-10-01
- **Source:** https://arxiv.org/abs/2610.01896 "Asynchronous LLM Post-Training: Group-Mass Capping and Convergence Analysis"
- **Organization/researchers:** Qijia He, Ruinan Jin, Jun Luo, Shaofeng Zou, Yingbin Liang (affiliations not stated on the abstract page)
- **Category:** RL/post-training | distributed
- **What changed:** Derives a convergence bound for GRPO-style algorithms under asynchronous rollout staleness that explicitly separates the gradient estimator's second moment from its bias, then proposes Group-Mass Capping GRPO (GMC-GRPO), which minimizes a ratio-based bias bound within a class of weighted estimators sharing a common second-moment guarantee. Compared with the prior TIC-GRPO baseline, it improves the threshold-dependence of the fourth-order delay term from $O(\epsilon^{-4})$ to $O(\epsilon^{-2})$, and shows the delay-dependent term shrinks as $G^{-2/5}$ with group-size tuning.
- **Technical summary:** Experiments on Qwen3 models and reasoning benchmarks show improved robustness to large rollout delays versus stable baselines. This is a formal-theory extension of exactly the staleness/async-RL mechanics the course already teaches qualitatively (`lessons/frontier/update-01.md` §2.4, "Asynchronous RL and policy staleness," and its GRPO-ratio-clipping-under-staleness math) — `lookup "asynchronous GRPO"` confirms the course section. The paper's contribution is a sharper theoretical bound plus a new capping method, not a restatement.
- **Why it might matter:** The course currently presents staleness clipping qualitatively/empirically (MLPerf harness staleness=1 bound); this paper would let a from-scratch lesson state *why* a given staleness bound degrades convergence with an actual rate, and introduce a concrete alternative-to-clipping mechanism (group-mass capping).
- **Evidence of adoption:** none — single paper, evaluated only on the authors' own Qwen3 runs, no third-party reproduction or adoption in a training framework (verl/OpenRLHF/Megatron-LM RL stacks).
- **Major organizations using it:** none confirmed.
- **Open-source implementation:** not confirmed on the abstract page.
- **Paper:** https://arxiv.org/abs/2610.01896
- **Code:** not confirmed
- **Relationship to existing course material:** genuinely new theoretical/methodological evidence on a topic the course already teaches qualitatively (`lessons/frontier/update-01.md` §2.4); per the dedup rule this is new evidence, not a plain duplicate, since it adds a formal convergence rate and a new capping mechanism absent from the existing lesson.
- **Potential course lesson:** possible addendum to `lessons/frontier/update-01.md` §2.4 with the convergence-rate intuition and GMC-GRPO as a concrete alternative to simple ratio clipping, once code/independent validation appears.
- **Confidence:** low-medium (solid theory + real-model experiments; single paper, no replication)
- **Recommendation:** monitor — strong fit for an existing lesson section; revisit if code is released or another group reproduces the robustness claim under large staleness.

### C-20261002-03 · AutoCompact: RL-trained context-compaction policy for long-horizon coding agents (part of a growing agent-memory-compaction cluster)

- **Class:** B
- **Date discovered:** 2026-10-02
- **Date published:** 2026-10-01
- **Source:** https://arxiv.org/abs/2610.02163 "AutoCompact: Learning When to Compact Context in Long-Horizon Coding Agents"
- **Organization/researchers:** Xuan Zhang, Longtao Zheng, Cunxiao Du, Bo An, Xin Dong (affiliations not stated on the abstract page)
- **Category:** agents/tools | memory/retrieval
- **What changed:** Trains a coding agent to decide, as part of its own policy, when to compact context, what working state to preserve, and how to continue afterward — rather than compacting on a fixed trigger (e.g. context-window-full) or leaving the decision to a separate heuristic. Training data comes from running the base agent, having a judge review/correct its compaction decisions and summaries, then SFT on the corrected trajectories followed by joint RL on coding + compaction with task-success reward.
- **Technical summary:** On SWE-bench Verified and SWE-PolyBench Verified, improves pass rates by an absolute 9.2 and 5.0 percentage points over the base model respectively, holding across inference budgets, including a 16K-context setting where overflow triggers fallback compaction and a 256K setting that never overflows. Architecturally simple enough (judge-corrected trajectories → SFT → RL) to fit the course's "implement the mechanism before the framework" rule.
- **Why it might matter:** This is now the *third* independent paper in ~a week on agent context/memory management surfacing in this research process: `C-20261001-03` (ReCAP, persistent context graphs, staged 2026-10-01) and "Mem++" (non-destructive read-time memory, arXiv 2610.02002, logged as C in `daily/2026-10-02.md`, same day as this paper) all attack the same underlying problem — long-running agents accumulating context they can't just keep or just discard — with three structurally different mechanisms (trained compaction policy vs. context graph vs. read-time document store). That convergence of independent approaches is itself a signal worth flagging to the weekly review, beyond any single paper's numbers.
- **Evidence of adoption:** none — single paper, authors' own SWE-bench evaluation only, no third-party reproduction or adoption in an agent framework.
- **Major organizations using it:** none confirmed.
- **Open-source implementation:** not confirmed on the abstract page.
- **Paper:** https://arxiv.org/abs/2610.02163
- **Code:** not confirmed
- **Relationship to existing course material:** adjacent to `lessons/module-18/lesson-02.md` and `lessons/module-19/lesson-01.md` (agent tool-use/SWE-agent); directly complements the already-staged `C-20261001-03` (ReCAP) as part of the same emerging agent-memory-compaction cluster; `lookup "SWE-bench coding agent context"` and `"context compaction agent"` found no exact prior match for this specific mechanism.
- **Potential course lesson:** if the cluster holds up, a candidate module-18/19 extension comparing the different compaction strategies (trained policy vs. context graph vs. read-time store) as a from-scratch exercise on long-running-agent context management.
- **Confidence:** low-medium (concrete SWE-bench numbers; single paper, no third-party validation; part of a 3-paper independent cluster)
- **Recommendation:** monitor alongside `C-20261001-03` as a cluster — flag the convergence of three independent groups on agent-memory-compaction to the weekly review even though no single paper alone clears the bar.

### C-20261003-01 · TLX Jagged Flash Attention on Blackwell: matches/beats FlashAttention-4 in production at Meta

- **Class:** A
- **Date discovered:** 2026-10-03
- **Date published:** 2026-10-01
- **Source:** https://pytorch.org/blog/optimizing-jagged-flash-attention-with-tlx-the-road-toward-sota-fa4-on-blackwell/ "Optimizing Jagged Flash Attention with TLX: The Road Toward SOTA FA4 on Blackwell" (official PyTorch engineering blog)
- **Organization/researchers:** Meta (PyTorch/GenAI infra team); builds on the TLX (Triton Low-level Extensions) line of PyTorch blog posts and on FlashAttention-4 (arXiv 2603.05451)
- **Category:** attention | efficiency | architecture
- **What changed:** Rewrites Meta's production Jagged Flash Attention kernel (the attention kernel behind Meta's Generative Ads Model, GEM) on NVIDIA Blackwell (B200) using TLX — a Triton extension exposing explicit warp specialization, SMEM/TMEM allocation, async TMA/MMA, and Cluster Launch Control — instead of hand-written CuteDSL/CUDA. Documents concrete kernel-engineering techniques (host-side jagged-tile load balancing with CLC, multi-stage double-buffered dQ reduce-add staging, early TMEM release, branch-free loop peeling via compile-time mask constants, 2-CTA collaborative MMA) and ports the same kernel to MXFP8 and block-sparse attention variants with minimal code changes.
- **Technical summary:** Benchmarked in bf16 on B200 against FlashAttention-4 (May 2026 version), the state-of-the-art open-source CuteDSL kernel: on the production jagged (broadcast-Q) shapes, the TLX kernel is ~13% faster on the forward pass (on average, trailing FA4 only at the longest/densest sequences) and ~50% faster on the backward pass; on dense LLM-style shapes (B=768, H=4, head_dim=128) it is ~87% of FA4 forward and ~+17% faster backward. The TLX implementation is ~3.2K lines vs. ~10K lines of hand-written CuteDSL for FA4. Code is public: https://github.com/facebookresearch/ads_model_kernel_library/tree/main/tlx_jfa.
- **Why it might matter:** This is a from-scratch, line-by-line account of how a real SOTA attention kernel is built and tuned on current hardware (Blackwell) — warp specialization, TMEM/SMEM management, barrier pipelining, load balancing for ragged sequence lengths, and low-precision (MXFP8) and block-sparse variants built on the same skeleton. It is squarely in the course's "implement the mechanism, explain what the framework does under the hood" territory for the attention/efficiency module, with working code and quantified numbers against the current open-source SOTA.
- **Evidence of adoption:** Real production use (Meta's GEM ads-recommendation model), not a company claim about an LLM product — the numeric comparison against FA4 is an apples-to-apples kernel benchmark, independently checkable since both are open source.
- **Major organizations using it:** Meta (production, GEM/Kunlun ads models).
- **Open-source implementation:** https://github.com/facebookresearch/ads_model_kernel_library/tree/main/tlx_jfa
- **Paper:** none (engineering blog post, not a paper); related paper is FlashAttention-4 itself, https://arxiv.org/abs/2603.05451
- **Code:** https://github.com/facebookresearch/ads_model_kernel_library/tree/main/tlx_jfa
- **Relationship to existing course material:** `lookup "FlashAttention-4"`, `"TLX Triton low-level extensions"`, and `"Blackwell kernel warp specialization"` all found no existing match. The course's efficiency/kernels module (`lessons/module-08/`) and attention lessons (`lessons/module-05/`) already cover FlashAttention; this is new material on FA4-era Blackwell kernel engineering (TLX, warp specialization, CLC) not yet taught.
- **Potential course lesson:** candidate addendum to `lessons/module-08/` (efficient training/inference kernels) covering Blackwell-generation attention kernel engineering (warp specialization, TMEM, Cluster Launch Control) using this post's worked techniques as the concrete numerical/code example, referencing FA4 as the baseline it's benchmarked against.
- **Confidence:** high (official engineering source, public code, production deployment, apples-to-apples benchmark against a named open-source SOTA)
- **Recommendation:** review for ADD as a module-08 extension on modern (Blackwell/FA4-era) attention kernel engineering — strong fit for the course's "implement before framework" rule and well-documented with code.

### C-20261003-02 · Sharpening Tax: RL post-training trades solution coverage for single-shot accuracy, even in agentic tasks

- **Class:** B
- **Date discovered:** 2026-10-03
- **Date published:** 2026-10-01
- **Source:** https://arxiv.org/abs/2610.01509 "Sharpening Tax in Post-Training"
- **Organization/researchers:** Changdae Oh, Qi Zeng, Qi Qi, Andrey Zhmoginov, Deren Lei, Yun He, Hoang Phan, Hangoo Kang, Azalia Mirhoseini, Sharon (Yixuan) Li — affiliations not listed on the abstract page; HF Daily Papers tags the paper as Meta-affiliated; Mirhoseini and Li are established academic/industry researchers (RL/MoE and OOD-robustness respectively).
- **Category:** RL/post-training | evaluation
- **What changed:** Tests the "RL post-training merely sharpens existing base-model behavior" hypothesis on agentic (multi-turn, tool-use) tasks, not just math/coding. Finds that pre-trained base models with a light inference harness often beat their post-trained counterparts on solution coverage (pass@K) under a large sampling budget, despite much lower pass@1 — because post-training pushes tasks toward "always solved or never solved," raising sampling efficiency/consistency at the cost of coverage. Introduces **Sharpening Tax**, a diagnostic metric for this test-time-scalability loss, and **posterior-tempered group sampling (PTGS)**, a Bayesian per-prompt-difficulty temperature sampler used during RL training to reduce the tax.
- **Technical summary:** Evaluated across 14 base/post-trained model pairs from four model families and three agentic benchmarks (42 cases total); the tax is "prevalent in most settings," estimable from a few rollouts, and correlates with other metrics. PTGS, applied during RL training in two agentic environments, pays a smaller tax than fixed-temperature sampling while also improving pass@1.
- **Why it might matter:** Directly bears on the course's RL/post-training and evaluation material — a concrete, broadly-tested empirical result (not a single-task anecdote) plus an actionable diagnostic metric and a training-time mitigation, relevant anywhere pass@K / test-time scaling is taught alongside RLHF/RLVR.
- **Evidence of adoption:** none — single paper, author-run evaluation only, no third-party reproduction.
- **Major organizations using it:** none confirmed; HF tags the paper as Meta-affiliated but this is not independently confirmed from the abstract page.
- **Open-source implementation:** not confirmed on the abstract page.
- **Paper:** https://arxiv.org/abs/2610.01509
- **Code:** not confirmed
- **Relationship to existing course material:** `lookup "sharpening tax post-training"` and `"pass@k RL post-training diversity"` found no exact match (the latter surfaced only the unrelated MLPerf post-training-benchmark lesson in `lessons/frontier/update-01.md`); new topic, complements existing RLVR/RLHF material.
- **Potential course lesson:** possible addendum to the RL/post-training module on pass@1-vs-pass@K tradeoffs after RLVR, using Sharpening Tax as the diagnostic and PTGS as a from-scratch-implementable mitigation.
- **Confidence:** medium (broad evaluation scope for a single paper; no independent replication yet)
- **Recommendation:** monitor — strong empirical breadth for a single-paper result; revisit at weekly review and watch for independent reproduction or adoption of PTGS in an RL training stack (verl/OpenRLHF).

### C-20261003-03 · Evaluation objectives for mechanistic circuit discovery can reward worse circuits (objective-level recovery gap)

- **Class:** B
- **Date discovered:** 2026-10-03
- **Date published:** 2026-10-01
- **Source:** https://arxiv.org/abs/2610.02098 "Are We Recovering Mechanisms? Objective-Level Recovery Gaps in Mechanistic Interpretability"
- **Organization/researchers:** Chuqin Geng, Li Zhang, Haolin Ye, Mark Zhang, Luke Zhang, Xujie Si — affiliations not listed on the abstract page.
- **Category:** interpretability | evaluation
- **What changed:** Shows that intervention-defined faithfulness — the standard objective used to score automated circuit-discovery methods (e.g. EAP, EAP-IG, ACDC, Edge-SP) — can prefer an equally-sized circuit that reproduces the model's actual behavior *less* well than an alternative, i.e. the evaluation objective itself has a recovery gap, independent of how good the search/attribution method is.
- **Technical summary:** Empirical study with controlled experiments across four human-reference circuit tasks plus InterpBench, comparing multiple circuit-discovery algorithms (EAP, EAP-IG, ACDC, Edge-SP) and showing KL-divergence-based faithfulness scores can misrank circuits relative to ground truth.
- **Why it might matter:** A methodological caution directly relevant to any interpretability curriculum content that teaches or uses automated circuit discovery (module touching mechanistic interpretability) — it argues that better search alone cannot fix circuit discovery if the scoring objective is mis-specified, which changes what "progress" in circuit discovery should be measured against.
- **Evidence of adoption:** none — single paper; no third-party replication found.
- **Major organizations using it:** none confirmed.
- **Open-source implementation:** not confirmed on the abstract page.
- **Paper:** https://arxiv.org/abs/2610.02098
- **Code:** not confirmed
- **Relationship to existing course material:** `lookup "circuit discovery faithfulness interpretability"` and `"mechanistic interpretability circuits"` found no existing match; course interpretability content (`lessons/module-20/lesson-01.md` and circuit mentions in `lessons/module-05/lesson-01.md`) does not currently discuss evaluation-objective pitfalls in circuit discovery.
- **Potential course lesson:** possible addendum to the interpretability module's circuit-discovery coverage, flagging intervention-faithfulness recovery gaps as a caveat when teaching/using EAP/ACDC-style methods.
- **Confidence:** medium (controlled multi-method empirical study; single paper, unreplicated)
- **Recommendation:** monitor — a methodological result worth the weekly review's attention if the course teaches or plans to teach automated circuit discovery; revisit if independently reproduced.

### C-20261004-01 · On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics

- **Class:** B
- **Date discovered:** 2026-10-04
- **Date published:** 2026-09-28
- **Source:** https://arxiv.org/abs/2609.35259 "On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics" (verified: resolves, abstract matches)
- **Organization/researchers:** Julianna Piskorz, Antonin Berthon, Mihaela van der Schaar — affiliation not stated on the abstract page (van der Schaar is a Cambridge professor; not confirmed as this paper's institutional affiliation from the primary source itself, so treated as unconfirmed).
- **Category:** training | RL/post-training | efficiency
- **What changed:** A controlled strong-to-weak distillation study that independently varies rollout policy (on- vs off-policy), token-level KL direction (forward vs reverse), and learning rate across the Llama3 and Qwen2.5 model families on scientific/medical/arithmetic reasoning tasks, to isolate which factor actually drives distillation outcomes — previous comparisons of SFT vs. RL-style distillation conflate these factors.
- **Technical summary:** Finds rollout policy is *not* the central factor people assume: forward KL is robust to rollout policy (stable regardless of on/off-policy rollouts), while reverse KL is far more rollout-sensitive and favors student-generated (on-policy) rollouts. Learning rate, not rollout policy, governs forgetting and update sparsity. On-policy data does help generalization on harder Countdown-arithmetic variants under both KL directions, but that edge does not reliably survive subsequent RLVR. Robust to removing gradient clipping, to sampled-KL estimators, and to longer reasoning chains.
- **Why it might matter:** Directly actionable for anyone designing a distillation/post-training pipeline (e.g. distilling reasoning traces into a smaller model, as taught in `lessons/module-17/lesson-03.md`'s R1-distillation discussion): it reframes the design question from "on-policy vs off-policy" to "which KL direction, and what learning rate," with a mechanistic (KL-gradient) explanation.
- **Evidence of adoption:** none — single paper, author-run experiments only, no third-party reproduction.
- **Major organizations using it:** none confirmed.
- **Open-source implementation:** not confirmed on the abstract page.
- **Paper:** https://arxiv.org/abs/2609.35259
- **Code:** not confirmed
- **Relationship to existing course material:** `lookup "distillation on-policy off-policy KL"` found no match. `lessons/module-17/lesson-03.md` (DeepSeek-R1 case study) mentions distilling R1's traces into dense models but does not discuss on/off-policy or KL-direction tradeoffs; this is new, complementary material.
- **Potential course lesson:** possible addendum to `lessons/module-17/lesson-03.md` (R1 distillation stage) or a new short distillation-methodology note, contrasting forward/reverse KL and rollout policy using this paper's controlled ablation as the worked example.
- **Confidence:** medium (broad, controlled, multi-model-family ablation; single paper, unreplicated)
- **Recommendation:** monitor — methodologically solid single-paper result; revisit at weekly review, watch for independent reproduction or use in an open distillation/RLVR stack (TRL, verl, OpenRLHF).

### C-20261004-02 · Beyond Memory: Harnessing Long-Horizon Agents with Explicit Belief States (PoS)

- **Class:** B
- **Date discovered:** 2026-10-04
- **Date published:** 2026-10-01
- **Source:** https://arxiv.org/abs/2610.01415 "Beyond Memory: Harnessing Long-Horizon Agents with Explicit Belief States" (verified: resolves, abstract matches)
- **Organization/researchers:** Yu Luo, Jiamin Jiang, Yimin Zuo, Xidao Wen, Rongchen Gao, Yongqian Sun, Shenglin Zhang, Guiyang Liu, Cheng Zhang, Fang Situ, Qi Zhou, Dan Pei — affiliation not stated on the abstract page (a third-party aggregator tagged it "Alibaba"-affiliated; not confirmed from the primary source, so treated as unconfirmed).
- **Category:** agents/tools | memory/retrieval
- **What changed:** Introduces PoS (Progression of States), an inference-time-only (no training) framework that has an LLM agent build and continually update an explicit "belief state" — a combination of current-world-state estimate plus unresolved task requirements — instead of just appending to a raw interaction-history/context buffer. It names and diagnoses a specific failure mode, "Belief Trapping" (agent keeps acting without making real progress: stagnation, cycles, or drift), and composes targeted recovery constraints when it is detected.
- **Technical summary:** Evaluated across 4 benchmarks (task execution + evidence-seeking diagnosis) and 3 backbone LLMs (12 backbone-benchmark combinations); PoS reportedly achieves the best overall performance in all 12 combinations, with relative gains over the strongest same-backbone baseline up to +22.68% (ALFWorld) and +37.89% (RCA-100 joint accuracy). Ablations attribute the gains to the consistency-validation and recovery-constraint steps specifically; scaling experiments show resilience as context grows.
- **Why it might matter:** The course currently covers agent loops (`lessons/module-19/lesson-01.md`, ReAct → mini SWE-agent) and tool use (module 18) but has no dedicated long-horizon-memory treatment; this gives a concrete, implementable (prompt/inference-time, not training) technique and a named failure mode (Belief Trapping) that's a good complement to the ReAct-style loop already taught.
- **Evidence of adoption:** none — single paper, author-run evaluation only.
- **Major organizations using it:** none confirmed (see affiliation caveat above).
- **Open-source implementation:** not confirmed on the abstract page.
- **Paper:** https://arxiv.org/abs/2610.01415
- **Code:** not confirmed
- **Relationship to existing course material:** `lookup "belief state long horizon agent memory"` found no match. New topic; complements `lessons/module-19/lesson-01.md`'s agent loop.
- **Potential course lesson:** possible addendum to module 19 (agents) on long-horizon context/memory management, using Belief Trapping as the diagnosed failure mode and PoS's belief-state update loop as the worked mechanism — inference-time only, so cheap to implement from scratch for an exercise.
- **Confidence:** medium (multi-benchmark, multi-backbone single-paper result; no independent replication; author affiliation unconfirmed)
- **Recommendation:** monitor — strong empirical breadth for a single paper and directly extends existing agent-loop material; revisit at weekly review.

### C-20261004-03 · GraphForge: Training Working Agents with Graph-Anchored Workspace Synthesis

- **Class:** B
- **Date discovered:** 2026-10-04
- **Date published:** 2026-09-30
- **Source:** https://arxiv.org/abs/2609.38923 "GraphForge: Training Working Agents with Graph-Anchored Workspace Synthesis" (verified: resolves, abstract matches)
- **Organization/researchers:** Qisheng Su, Hanchen Wang, Guanru Zhu, Huicheng Jiang, Qiuyinzhe Zhang, Kou Shi, Zhen Fang, Ziao Zhang, Qingnan Ren, Zehui Chen, Tao Gui, Feng Zhao — affiliation not stated on the abstract page.
- **Category:** agents/tools | data
- **What changed:** A training-data-synthesis method for fine-tuning "working agents" (agents that read real files, coordinate tools, and produce deliverables) that grounds both the generated task statement and its grading rubric in an "evidence graph" built over an assembled workspace of real files, rather than synthesizing files/tasks from scratch (which tends to produce unrealistic, ungroundable tasks). Rollout-and-revision filtering precedes data collection.
- **Technical summary:** Fine-tuning on ~2,169 GraphForge-generated trajectories, plus rejection sampling, reportedly improves performance across three benchmarks, including raising a GDPVal-style score to 144 (scale/baseline not independently confirmed from the abstract alone). Data and models released on Hugging Face (`huggingface.co/collections/groundhogLLM/graphforge` — not independently verified to load).
- **Why it might matter:** Directly relevant to the course's data-curation and agent-training material: a concrete recipe for generating verifiable, grounded agent-training tasks (task + rubric both derived from the same evidence graph) rather than hand-written or ungrounded synthetic tasks, which is a recurring practical problem in training tool-using agents.
- **Evidence of adoption:** none — single paper, author-run evaluation only.
- **Major organizations using it:** none confirmed.
- **Open-source implementation:** data/models claimed at `huggingface.co/collections/groundhogLLM/graphforge` (link not independently verified to load in this run).
- **Paper:** https://arxiv.org/abs/2609.38923
- **Code:** not confirmed
- **Relationship to existing course material:** `lookup "agent training data synthesis workspace graph"` found no match. New topic; complements module 19 (agents) and the course's general data-curation coverage.
- **Potential course lesson:** possible addendum to module 19 on synthesizing verifiable agent-training data (task+rubric grounded in a file/evidence graph), as a concrete alternative to hand-written agent benchmarks.
- **Confidence:** low-medium (single paper; key benchmark number not independently interpretable from the abstract; HF data link unverified)
- **Recommendation:** monitor — plausible and relevant technique, but verify the HF data/model release actually exists and check the GDPVal-score claim before any weekly-review action.
