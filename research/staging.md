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
