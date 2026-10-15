# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261012-01 · Thinkingbox: grading agents by terminal backend state and repeat-reliability (pass^N), not self-report

- **Class:** B
- **Date discovered:** 2026-10-12
- **Date published:** 2026-08-20 (v1); current v4 2026-10-01; publicized via a joint Microsoft/Hugging Face blog post 2026-10-03
- **Source:** https://arxiv.org/abs/2608.19741 (verified resolves; v4 "Thu, 1 Oct 2026"). A joint Microsoft/Hugging Face blog post announced this around 2026-10-03 per secondary coverage, but its exact URL could not be independently located/verified this run — the arXiv paper and its linked GitHub repos (below) are the sources actually verified.
- **Organization/researchers:** Zhuochun Li, Youngmin Ko, Ali Keramati, Tuhin Kundu, Liang-Chun Tsai, Nicola Ferri, Mirco Milletari, Jiaxiang Liu, Susana Palmaz Lopez Pelaez, Yuepeng Wang, Vadim Smolyakov, Xiang Jiang, Kjartan Olafsson, Tommy Guy — Microsoft (several authors) plus academic co-authors; affiliations not individually stated on the abstract page.
- **Category:** evaluation (agents)
- **What changed:** Thinkingbox is a sandbox (isolated MCP-compatible tool sessions) plus Thinkingbox-bench: 507 policy-governed, stateful business workflows across retail, hospitality, auto insurance, neobank IT, and consulting IT/HR. Each attempt is graded by task-specific executable tests over the *terminal backend state* (accept valid outcomes, reject wrong/missing/extra state changes) rather than by the agent's transcript or self-reported completion. The paper also reports pass^N — the same task attempted N times, checking whether the agent succeeds *every* time — as a reliability metric distinct from pass@k's coverage framing (pass@k asks "does at least one of k independent samples succeed"; pass^N asks "does it succeed on all N repeats of the identical task").
- **Technical summary:** Claude Opus 5 drops from 66.50% pass@1 to 47.53% pass^20; Kimi-K3 drops from 57.37% to 17.60%. The authors report many failed runs end "cleanly" after valid-looking state-changing actions, so response-level or tool-call-level signals are poor proxies for actual task completion. Code (sandbox) at github.com/microsoft/thinkingbox, data (benchmark) at github.com/microsoft/thinkingbox-data, both linked from the abstract page.
- **Why it might matter:** A clean, falsifiable complement to the course's existing pass@k material (`code/src/llmre/evaluation/pass_at_k.py`, from the accepted MLPerf-RLVR topic): pass^N measures repeat-reliability of one task rather than coverage across independent attempts, and the "agent claims success, backend disagrees" framing is a sharp, teachable case for grounding agent evaluation in environment side effects rather than self-report.
- **Evidence of adoption:** Integrated as an official OpenEnv environment on Hugging Face (huggingface.co/docs/openenv/environments/thinkingbox) — real infrastructure availability, though not yet adoption by another named evaluation harness or model card.
- **Major organizations using it:** Microsoft (co-author and code owner); publicized jointly with Hugging Face.
- **Open-source implementation:** https://github.com/microsoft/thinkingbox (sandbox), https://github.com/microsoft/thinkingbox-data (benchmark data)
- **Paper:** https://arxiv.org/abs/2608.19741
- **Code:** https://github.com/microsoft/thinkingbox
- **Relationship to existing course material:** New sub-topic — checked via `lookup "agent evaluation terminal state side effects benchmark"`, no match. Related to but distinct from the already-deferred `research/deferred/computer-use-agent-process-evaluation.md` (OSWorld-Pro grades intermediate process steps; Thinkingbox grades terminal backend state plus repeat-reliability) — flag both together at the next weekly review as two angles on "evaluation beyond final self-reported success."
- **Potential course lesson:** Possible extension of the evaluation module: add a pass^N reliability metric alongside the existing pass@k code, and an agent lesson contrasting self-report/transcript grading with environment-state grading.
- **Confidence:** medium-high — code and data released, concrete numbers across multiple named models, and now has HF OpenEnv infrastructure support, but still a single benchmark paper without independent reproduction or adoption by another evaluation suite.
- **Recommendation:** review at the next weekly cycle alongside OSWorld-Pro (both evaluation-methodology candidates); consider whether pass^N is a small, low-risk addition to the existing `pass_at_k.py` module even before a lesson decision (flag only — no code change made here).

### C-20261012-02 · Rethinking Relative Policy Differences Transfer of On-Policy Distillation (new evidence for deferred OPD-variants topic)

- **Class:** B
- **Date discovered:** 2026-10-12
- **Date published:** 2026-10-09
- **Source:** https://arxiv.org/abs/2610.13205 (verified resolves; v1 "Fri, 9 Oct 2026")
- **Organization/researchers:** Rong Wu, Daocheng Fu, Yu Yang, Xuemeng Yang, Siqi Li, Shu Zou, Yao Zhu, Botian Shi, Bo Zhang, Cong Zhang, Pinlong Cai, Jianbiao Mei (affiliations not stated on the abstract page)
- **Category:** RL/post-training (on-policy distillation — base technique taught in lesson 17.4)
- **What changed:** Standard on-policy distillation (OPD) assumes a much stronger teacher. This paper generalizes OPD to transfer the *relative difference* between any expert/reference policy pair (regardless of scale or training stage) by treating their log-probability ratio as an implicit reward inside KL-regularized policy optimization — standard OPD falls out as a special case. Three findings: (1) a stronger expert is not automatically a better transfer source — differences from task-specific training (RL/SFT) transfer well, differences from scale alone transfer weakly or not at all; (2) the expert supplies the capability, the reference subtracts out shared behavior to isolate the useful residual; (3) a useful difference doesn't always transfer to a given target — raw policy distance doesn't predict transferability, but choosing a better reference can rescue a failed transfer.
- **Technical summary:** No code repository linked from the abstract page; the contribution is primarily theoretical/framing plus supporting experiments (not independently reproduced).
- **Why it might matter:** Directly tests an assumption embedded in lesson 17.4's framing of OPD (bigger/stronger teacher ⇒ better source); if the scale-vs-task-specific-training distinction holds up, it is a teachable refinement of *when* OPD works, not just that it works.
- **Evidence of adoption:** None — single paper, no code, no independent reproduction.
- **Major organizations using it:** None identified (academic, no affiliations stated).
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.13205
- **Code:** none found
- **Relationship to existing course material:** New evidence for the already-deferred `research/deferred/on-policy-distillation-variants.md` — a sixth independent sub-mechanism/finding in that cluster (distinct from DCE/SRCL co-evolving teacher, the teacher-scale scaling law, NP-OPD, the "new skills not new knowledge" framing, and DIAL-OPD token selection). Checked via `lookup "on-policy distillation transfer policy difference"`, no match.
- **Potential course lesson:** Not a new lesson; a sixth data point for the deferred topic's next scheduled review (2026-11-29), specifically bearing on the "does teacher scale matter" sub-question.
- **Confidence:** low-medium — single paper, no code, no reproduction, but a conceptually clean generalization with a falsifiable scale-vs-task-specific-training distinction.
- **Recommendation:** add as a sixth data point to `on-policy-distillation-variants.md` at its 2026-11-29 review; not independently actionable before then.

### C-20261012-03 · Stateless Language Agents: externalized state + rebuilt-per-call context for long-horizon research agents (new evidence for deferred agent-context-compaction topic)

- **Class:** B
- **Date discovered:** 2026-10-12
- **Date published:** 2026-10-06
- **Source:** https://arxiv.org/abs/2610.07625 (verified resolves; v1 "Tue, 6 Oct 2026")
- **Organization/researchers:** Qizheng Zhang, Changxiu Ji, Isaac Sun, Yuetai Li, Shubhangi Upasani, Sherry Ruan, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Yoonho Lee, Yuzhen Mao, Genghan Zhang, Rulin Shao, Qiuyang Mang, Andy Dimnaku, Changran Hu, Radha Poovendran, Kunle Olukotun (18 authors; affiliations not stated on the abstract page)
- **Category:** agents/tools (long-horizon context/memory management)
- **What changed:** Argues that running agents longer doesn't automatically help: agents replay growing histories, duplicate work, or stop experimenting while still burning tokens — traced to *where research state lives and who decides what to try next*. Proposes Stateless Language Agents (SLA): "stateful search with stateless agents" — an external harness stores candidate solutions and measured outcomes and rebuilds a fresh, role-specific context for every agent call, rather than one persistent agent accumulating (and later compacting/summarizing) its own history. A stateless Advisor reviews summarized evidence and assigns experiments to parallel stateless Workers.
- **Technical summary:** Across software engineering, kernel optimization, and algorithm design, with budgets up to one billion tokens, SLA reportedly reaches the best final result on every tested task; it matches the strongest kernel baseline's final performance using over 84% fewer tokens, and the Advisor itself uses under 0.6% of total tokens. No code repository linked from the abstract page.
- **Why it might matter:** A fifth, architecturally distinct approach to the same problem already tracked in the deferred cluster below — rather than compacting or summarizing one agent's growing context (ReCAP, AutoCompact, PoS, REMORY), it externalizes all state and rebuilds context fresh per call. Strong empirical scale (up to 1B-token budgets) makes this a sharper data point than most single papers in that cluster.
- **Evidence of adoption:** None — single paper, no code, no independent reproduction.
- **Major organizations using it:** None identified; authors span several institutions but no org is named as adopting the architecture.
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.07625
- **Code:** none found
- **Relationship to existing course material:** New evidence for the already-deferred `research/deferred/agent-context-compaction.md` — a fifth independent mechanism, and the first that externalizes state entirely rather than compacting it in place. Checked via `lookup "stateless agent long horizon research context"`, no match.
- **Potential course lesson:** Not a new lesson; a fifth data point for that deferred topic's 2026-11-29 review. Worth flagging to the weekly review that the cluster may now span two genuinely different design philosophies (compact-in-place vs. externalize-and-rebuild) rather than four variants of one idea — possibly a reason to consider the *design-space distinction itself* as teachable even before one mechanism is proven standard.
- **Confidence:** medium — no code or reproduction, but unusually large-scale experiments (1B-token budgets) for a single paper in this area.
- **Recommendation:** add as a fifth data point to `agent-context-compaction.md` at its 2026-11-29 review, flagging the compact-vs-externalize distinction for discussion.

### C-20261012-04 · GenCorrect: externally-graded IMO/IOI gold-level results via generate-verify-refine test-time pipeline (Nemotron)

- **Class:** B
- **Date discovered:** 2026-10-12
- **Date published:** blog 2026-10-07 (Hugging Face); underlying papers: IMO 2026 recipe https://arxiv.org/abs/2609.10712, IOI 2026/GenCorrect https://arxiv.org/abs/2609.02849
- **Source:** https://huggingface.co/blog/nvidia/nemotron-ioi-and-imo-2026 (verified loads; publish date 2026-10-07 confirmed on page)
- **Organization/researchers:** NVIDIA (Nemotron/NeMo-Skills team)
- **Category:** reasoning / RL/post-training (test-time generate-verify-refine)
- **What changed:** Two specialized Nemotron pipelines: a 550B-total/55B-active competitive-coding model (SFT on 22K curated problems) and an IMO math pipeline (SFT on 414,890 filtered examples across 15,818 proof problems plus RL on 9,597 problems), both combined at test time with GenCorrect — an iterative generate → evaluate → refine loop, no formal prover or external tools for the math track.
- **Technical summary:** IMO 2026: 30/42, above the official gold threshold of 29, with submissions graded by official IMO graders (external, human verification — not a self-reported or company-only benchmark). IOI 2026: 535.4/600 (unofficial/unsupervised run, gold threshold 361.12, beating the reported top human score of 498.27). On IOI 2025 (where progression is given), GenCorrect alone took a 30B-A3B model from 291 (after SFT+RL) to 468, above that year's 438.3 gold threshold — the single largest jump in the reported progression. Checkpoints released (NVIDIA-Nemotron-Labs-3-Competitive-Coding-550B-A55B-NVFP4; the Nemotron Labs IMO 2026 collection), IMO training data and a 200-problem IMO-Bench released, and code released (NeMo-Skills, including the IMO inference pipeline/prompts/submitted proofs).
- **Why it might matter:** The IMO result is graded by an independent human panel (official IMO graders), not just a company's own benchmark — stronger evidence than typical "company claim" releases — and GenCorrect's test-time generate-verify-refine loop is a concrete, reproducible technique (code + checkpoints + data released) directly extendable to the course's test-time-scaling/reasoning material (module 17), distinct from pure sampling-based best-of-N since it iteratively revises rather than just reselecting among independent samples.
- **Evidence of adoption:** Externally verified for the IMO track specifically (official competition grading); no third-party adoption of GenCorrect itself as a technique yet.
- **Major organizations using it:** NVIDIA (originator).
- **Open-source implementation:** https://github.com/NVIDIA-NeMo/Skills (also the IMO-specific recipe under `recipes/nemotron-imo-tts`)
- **Paper:** https://arxiv.org/abs/2609.10712 (IMO), https://arxiv.org/abs/2609.02849 (IOI/GenCorrect)
- **Code:** https://github.com/NVIDIA-NeMo/Skills
- **Relationship to existing course material:** New sub-topic — checked via `lookup "IMO gold olympiad"` and `lookup "GenCorrect generate verify refine"`, no matches. Related to but distinct from the course's existing best-of-N/self-consistency material (17.1) and process-reward material (17.2): GenCorrect is an iterative refinement loop, not a rerank-after-independent-sampling method.
- **Potential course lesson:** Possible addition to module 17's test-time-scaling material contrasting best-of-N/self-consistency (resample + rerank) with iterative generate-verify-refine (GenCorrect), using the IOI 2025 progression numbers as a worked before/after example.
- **Confidence:** medium-high — code, checkpoints, and (for IMO) externally-graded results, but still a single organization's own technical report for the overall system (company claim for the IOI "beats top human" framing specifically, flagged as such on NVIDIA's own page as an unofficial run).
- **Recommendation:** monitor for the next weekly review as a possible module-17 extension; the IMO external grading and released code/checkpoints make this stronger than a typical single-source company claim, but still single-source for the method's generality beyond these two competitions.

### C-20261013-01 · Elastic Latent Recurrence (ELR): removing the "one-loop tax" in looped/recurrent-depth language models

- **Class:** B
- **Date discovered:** 2026-10-13
- **Date published:** 2026-10-08
- **Source:** https://arxiv.org/abs/2610.12582 (verified resolves; v1 "Thu, 8 Oct 2026")
- **Organization/researchers:** Faisal Ladhak, Shantanu Acharya, Santiago Akle Serano, Simeng Sun, Boris Ginsburg, Krishna C Puvvada — no institutional affiliation is stated on the abstract page (Boris Ginsburg and Krishna Puvvada are names associated with NVIDIA Research in prior publications, but this is not confirmed on this paper's page, so treated as unconfirmed).
- **Category:** architecture (looped/recurrent-depth language models)
- **What changed:** Looped/recurrent-depth models (reusing a shared block of layers several times per token) normally underperform a non-recurrent model of the same size when run for only one loop — the "one-loop tax" — which limits their practicality as a single checkpoint that trades inference compute for accuracy. ELR is a training recipe meant to remove that penalty via three elements: a gated state update with a learned step size, mixing loop counts within each optimizer update, and conditioning each forward pass on its loop index and compute budget. Mixed depths can be introduced late in pretraining to cut training cost; for MoE variants, reusing first-loop expert routing on later loops cuts update time 11.7% with no measurable accuracy change; at inference, keeping only the last loop's attention KV cache keeps attention memory at one-loop levels while retaining most of the gains.
- **Technical summary:** On 4B-dense and 30B-A3B-MoE hybrid models, ELR reports ~7 points of gain at four loops on a ten-task average versus one loop. The resulting model, Koi-4B, reportedly scores ~18 on the Artificial Analysis Intelligence Index at one loop (matching Qwen3.5-4B) and ~29 at four loops. No code or model-weight link is present on the abstract page.
- **Why it might matter:** Directly addresses the central open question in the course's already-deferred looped-recurrent-depth cluster — "does any recipe make one-loop performance match a dense model of the same size, so the same checkpoint is usable at variable test-time compute without a penalty." If reproduced, this would be the first recipe in that cluster to claim the one-loop-tax problem is solved rather than merely mitigated.
- **Evidence of adoption:** None yet — single paper, no released code or weights, no independent reproduction. The reported numbers are the authors' own (company/author claim).
- **Major organizations using it:** None confirmed (see affiliation note above).
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.12582
- **Code:** none found
- **Relationship to existing course material:** New evidence for the already-deferred `research/deferred/looped-recurrent-depth-lms.md` (WAIT, next review 2026-11-29) — a fourth independent sub-mechanism in that cluster, and the first to directly target the one-loop-tax framing. Checked via `lookup "looped language model recurrent depth"`, which matched that file.
- **Potential course lesson:** Not a new lesson; a data point for the deferred topic's 2026-11-29 review, specifically on whether the "same checkpoint, variable test-time compute, no one-loop penalty" claim holds up.
- **Confidence:** low-medium — single paper, no code/weights, no reproduction, but concrete named-model numbers (Koi-4B) and a plausible mechanism (gated step size + mixed loop-count training) distinct from the cluster's other three approaches.
- **Recommendation:** add as a fourth data point to `looped-recurrent-depth-lms.md` at its 2026-11-29 review; not independently actionable before then.

### C-20261013-02 · Semi-OPD: offline-student-rollout distillation often beats on-policy distillation

- **Class:** B
- **Date discovered:** 2026-10-13
- **Date published:** 2026-10-08
- **Source:** https://arxiv.org/abs/2610.11291 (verified resolves; v1 "Thu, 8 Oct 2026")
- **Organization/researchers:** Siyan Zhao, Yonggan Fu, Jindong Jiang, Shih-Yang Liu, Song Bian, Byung-Kwan Lee, Sharath Turuvekere Sreenivas, Wenliang Dai, Hanrong Ye, Aditya Grover, Pavlo Molchanov — no institutional affiliation is stated on the abstract page (Pavlo Molchanov is a name associated with NVIDIA Research in prior publications, but this is not confirmed on this paper's page, so treated as unconfirmed).
- **Category:** RL/post-training (on-policy distillation — base technique taught in lesson 17.4)
- **What changed:** Tests whether on-policy sampling is actually necessary for logit/policy distillation. Proposes Semi-OPD: distill from offline rollouts generated once by the initial student, rather than resampling on-policy from the current student throughout training. Reports Semi-OPD beats standard OPD in 14 of 17 teacher-student pairs spanning 1.5B-235B parameters, with accuracy gains up to 13.6% and an 11.4x training-time speedup. The paper's explanation: OPD's advantage depends on how aligned the initial teacher and student already are (measured by output-token overlap); for well-aligned pairs OPD helps, but for misaligned pairs the student's on-policy rollouts drift further off-distribution from the teacher as context length grows, degrading the distillation signal — Semi-OPD avoids this by training on shorter contexts while still covering full trajectories.
- **Technical summary:** No code repository is linked from the abstract page; results are the authors' own reported benchmarks across 17 teacher-student pairs, not independently reproduced.
- **Why it might matter:** This is a direct, falsifiable challenge to a premise embedded in the course's already-taught lesson 17.4 (that on-policy sampling is generally worth its extra infrastructure cost for distillation). If the teacher/student-alignment framing holds up, it's a teachable refinement of *when* OPD is worth it — paralleling the already-staged C-20261012-02 finding about when a stronger teacher transfers well.
- **Evidence of adoption:** None — single paper, no code, no independent reproduction. Numbers are the authors' own.
- **Major organizations using it:** None confirmed (see affiliation note above).
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.11291
- **Code:** none found
- **Relationship to existing course material:** Directly bears on `lessons/module-17/lesson-04.md` (17.4 · Logit distillation: forward vs reverse KL & on-policy distillation) — checked via `lookup "on-policy distillation offline student rollouts"`, which matched that lesson. Distinct from the already-staged `staging.md` C-20261012-02 (which concerns *which reference/expert pair* transfers well) — this paper instead asks whether on-policy sampling itself is necessary at all. Also related to, but a different angle from, the deferred `research/deferred/on-policy-distillation-variants.md` cluster (co-evolving teacher, teacher-scale laws); flag together at the next weekly review.
- **Potential course lesson:** Possible update to 17.4 contrasting "always resample on-policy" with "offline-rollout distillation when teacher/student are already poorly aligned," using the token-overlap alignment metric as the deciding factor.
- **Confidence:** medium — single paper, no code/reproduction, but directly tests an assumption in already-taught material with concrete numbers across a wide 1.5B-235B parameter range.
- **Recommendation:** review at the next weekly cycle specifically as a possible 17.4 amendment (not just a deferred-cluster data point), alongside C-20261012-02.

### C-20261013-03 · The causal role of weight decay in the life cycle of massive activations

- **Class:** B
- **Date discovered:** 2026-10-13
- **Date published:** 2026-09-30 (v1); v2 2026-10-05
- **Source:** https://arxiv.org/abs/2610.00423 (verified resolves; v2 "Mon, 5 Oct 2026")
- **Organization/researchers:** S. Aaron McClendon, Jorge Gallego-Feliciano, Antonios Saravanos — no institutional affiliation stated on the abstract page.
- **Category:** interpretability
- **What changed:** Studies massive activations — very-large-magnitude residual-stream values linked to attention sinks — and tracks how they emerge, grow, and stabilize over training. Main finding: weight decay causally controls the turnover of this activation scale. Removing weight decay near the activation's peak lets it keep growing; keeping decay active makes it decline even at a constant learning rate. The authors propose a balance model in which AdamW-style preconditioned growth opposes weight decay; sweeping the decay coefficient shifts the peak's timing roughly log-linearly and its peak magnitude scales roughly as (decay coefficient)^(-1/2).
- **Technical summary:** A mechanistic/causal account (ablating weight decay mid-training) rather than only a correlational observation, with a quantitative scaling law for peak timing and magnitude as a function of the weight-decay coefficient. No code repository linked from the abstract page.
- **Why it might matter:** Massive activations/attention sinks already matter for the course's quantization and long-context material (outlier activations are a known obstacle to low-bit quantization and KV-cache compression). This paper offers a specific, testable training-dynamics lever (weight decay) rather than just describing the phenomenon — a clean candidate for a from-scratch numerical/toy-training demonstration (the six-pass lesson structure's "numerical example" and "from-scratch implementation" passes would fit this well: train a tiny transformer with/without decay and watch the activation's trajectory).
- **Evidence of adoption:** None — single academic paper, no code, no independent reproduction.
- **Major organizations using it:** None identified.
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.00423
- **Code:** none found
- **Relationship to existing course material:** New sub-topic — checked via `lookup "attention sink outlier activation quantization"` and `lookup "massive activation interpretability"`, no matches in either the course or registry. Not previously covered.
- **Potential course lesson:** Possible addition to the interpretability module (or a cross-reference from the quantization module) on massive activations/attention sinks, with weight decay as the causal lever — a good candidate for a small from-scratch toy experiment given the course's six-pass lesson template.
- **Confidence:** low-medium — single paper, no code/reproduction, but a clear causal claim (ablation, not just correlation) with a concrete quantitative scaling relationship.
- **Recommendation:** monitor for the next weekly review as a possible new interpretability topic (first entry in this sub-area, so likely WAIT pending independent reproduction) rather than an extension of an existing deferred cluster.

### C-20261014-01 · ReplaySSM: cache SSM inputs, not state — removing the state read/write bottleneck in hybrid-attention decoding

- **Class:** A
- **Date discovered:** 2026-10-14
- **Date published:** 2026-10-09 (v1)
- **Source:** https://arxiv.org/abs/2610.13778 (verified resolves; v1 "Fri, 9 Oct 2026"). Also published as an NVIDIA technical blog post, https://developer.nvidia.com/blog/replayssm-rethinking-state-storage-for-faster-hybrid-model-decoding/ (verified loads, dated 2026-10-12), and as a personal blog post by Tri Dao, https://tridao.me/blog/2026/replayssm/ (verified exists via search; content cross-checked against the arXiv abstract and NVIDIA post).
- **Organization/researchers:** Ze-Wei Liou, Harris Nover, Amey Naik, Andrii Skliar, Igor Shovkun, Ido Shahaf, Tri Dao — per the GitHub README, Princeton University and NVIDIA (Tri Dao is the FlashAttention/Mamba co-author; affiliation not independently re-confirmed beyond the README's own statement).
- **Category:** serving / attention / efficiency (hybrid SSM + attention decoding)
- **What changed:** In hybrid models (SSM layers interleaved with attention, e.g. Mamba-2, Gated DeltaNet), every decode step must read and write the full fixed-size recurrent state — a memory-bandwidth-bound operation with little arithmetic. This also makes speculative-decoding rollback expensive (a state snapshot per draft token) and lets state quantization error compound through the recurrence (unlike a KV cache, which is never written back into itself). ReplaySSM represents the state as a periodically-updated checkpoint plus a buffer of recent inputs: most decode steps just append to the buffer and compute output from checkpoint+buffer without writing a new state; a buffer-fill triggers one flush that folds the buffer into the checkpoint. Speculative-decoding rollback becomes a pointer move (no snapshots). A high-precision checkpoint can be kept for the recurrence while a separate low-precision copy serves reads, which prevents quantization error from feeding back on itself.
- **Technical summary:** Implemented for Mamba-2 and Gated DeltaNet in vLLM. At batch size 256: kernel speedups of 1.46x-1.80x and end-to-end decode-step speedups of 1.20x-1.47x across four hybrid models vs. a tuned vLLM baseline (no speedup at batch size 1). Speculative decoding (T=4, batch 512, two 120B-class models): 1.85x-2.10x vLLM's speculative-decoding throughput with matched acceptance lengths. Quantization: on Qwen3.5-4B at 128K generated tokens, KL divergence from FP32 fell from 0.136 to 0.0038 when reading NVFP4 with an FP32 recurrence checkpoint; on GPQA-Diamond, a single shared NVFP4 checkpoint lost 3-31 points vs. FP32, while separate read/recurrence precisions kept 3 of 4 models within ~2 points of FP32.
- **Why it might matter:** Directly extends the course's already-accepted hybrid-linear-attention material (`lessons/module-12/lesson-05.md`, 12.5) from "how hybrid attention works" into "how to serve it efficiently" — a concrete, adopted serving-side technique for the KV/state-cache memory and quantization problems that 12.5's hybrid stacks create. The combination of real stack adoption, a named-author credibility signal (Tri Dao), and direct production use makes this unusually strong for a single week's finding.
- **Evidence of adoption:** Independently verified, not a company claim alone — the GitHub README documents merged pull requests landing the technique in TensorRT-LLM, vLLM, SGLang, and FlashInfer (the paper authors contributed the listed PRs, except one TensorRT-LLM GDN PR credited to NVIDIA independently). The NVIDIA blog post states production use in GLM-5.3-Flash and Kimi-K3 (this specific claim is sourced to NVIDIA's own post and not independently re-verified against those models' own release notes this run). A NeurIPS 2026 follow-on paper, SketchSSM (https://neurips.cc/virtual/2026/173791), explicitly builds on and extends ReplaySSM's input-buffering idea — independent academic engagement, not just the original authors.
- **Major organizations using it:** NVIDIA/Princeton (originators); merged into vLLM, SGLang, TensorRT-LLM, FlashInfer; reported production use in Zhipu's GLM-5.3-Flash and Moonshot's Kimi-K3.
- **Open-source implementation:** https://github.com/Johnny-Liou/ReplaySSM (Apache-2.0), plus the merged PRs in vLLM/SGLang/TensorRT-LLM/FlashInfer themselves.
- **Paper:** https://arxiv.org/abs/2610.13778
- **Code:** https://github.com/Johnny-Liou/ReplaySSM
- **Relationship to existing course material:** Extends `lessons/module-12/lesson-05.md` (12.5 · Linear attention, Gated DeltaNet & hybrid stacks, already ADDed per `research/accepted/hybrid-linear-attention-gated-deltanet.md`) and touches the course's quantization material (state/KV-cache quantization). Checked via `lookup "SSM state caching decoding"` and `lookup "hybrid linear attention gated deltanet"` — the latter matched the existing accepted topic and lesson, confirming this is a serving-layer extension of already-taught material rather than a wholly new topic.
- **Potential course lesson:** Extend 12.5 (or a serving/inference-memory lesson) with a from-scratch worked example of checkpoint+buffer state representation vs. naive per-step state read/write, plus the mixed-precision read/recurrence quantization trick — fits the course's existing "implement the mechanism from scratch before the production framework" rule well, since vLLM/TensorRT-LLM/SGLang/FlashInfer all now ship it.
- **Confidence:** high — paper + released code + merged PRs in four major serving stacks + independent academic follow-on work + a highly credible co-author, all pointing the same direction in one week.
- **Recommendation:** review for ADD at the next weekly cycle, most likely as an extension of lesson 12.5 (hybrid attention serving/decoding) and/or the quantization module (mixed-precision state read vs. recurrence).

### C-20261014-02 · DoSA (Dots Sparse Attention): conditional, certificate-based Top-K scoring for production sparse-attention indexers

- **Class:** B
- **Date discovered:** 2026-10-14
- **Date published:** 2026-09-26 (v1)
- **Source:** https://arxiv.org/abs/2610.13990 (verified resolves; v1 "Sat, 26 Sep 2026")
- **Organization/researchers:** Fanxu Meng, Taosong Fang, and one other author — affiliations not stated on the abstract page.
- **Category:** attention / efficiency / serving (sparse-attention indexer, e.g. DeepSeek-V3.2/GLM-5.2-style lightning indexer)
- **What changed:** Native sparse-attention indexers (as used in DeepSeek-V3.2 and GLM-5.2) score every candidate key to pick the Top-K set to attend to, but a Top-K selection only needs enough evidence to reject a losing candidate, not its exact score. DoSA introduces a certificate: a seed subset gives a lower bound on the winning threshold, and bounds on each head's positive/negative contribution determine which remaining candidates still need full scoring — a lossless method (exact same Top-K set as full scoring) that makes the scoring step conditional instead of exhaustive. Exact-score filtering also cuts downstream writeback/selection work.
- **Technical summary:** Reports the GPU implementation speeds up the full Indexer by up to 2.74x over SGLang's and 1.69x over LiteTopK's implementations, with end-to-end time-to-first-token speedups of 1.94x on GLM-5.2 and 1.50x on DeepSeek-V3.2 against their respective baselines — i.e. benchmarked against production serving-framework baselines (SGLang) and a named prior top-k method (LiteTopK), not just a naive reference implementation.
- **Why it might matter:** A concrete, production-relevant efficiency technique for the sparse-attention indexer architecture already shipping in two named frontier open-weight models, with real baseline comparisons rather than only self-comparison. Relates to (but is mechanistically distinct from) the course's existing deferred sparse-attention cluster (`block-sparse-attention-pisa.md`, `sparse-attention-decoding-ffd.md`), which target block-sparse/training-free retrofits rather than the specific lightning-indexer Top-K-certificate problem.
- **Evidence of adoption:** None yet beyond the benchmark comparison itself — single paper, no code or model-weight link found on the abstract page, no independent reproduction. Reported speedups are the authors' own.
- **Major organizations using it:** None confirmed (no stated affiliation); benchmarked against, but not adopted by, SGLang.
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.13990
- **Code:** none found
- **Relationship to existing course material:** New sub-topic within the existing sparse-attention cluster — checked via `lookup "lightning indexer DeepSeek sparse attention"` and `lookup "DSA sparse attention top-k scoring"`, no matches; checked against `research/deferred/block-sparse-attention-pisa.md` and `research/deferred/sparse-attention-decoding-ffd.md`, both of which are related but address a different mechanism (block-level scoring / training-free decode retrofits vs. this paper's certificate-based native-indexer scoring).
- **Potential course lesson:** Possible addition to module 12's attention material (or the sparse-attention-decoding cluster if/when it is ever ADDed) specifically on native sparse-attention indexer architectures (DeepSeek-V3.2-style) and how Top-K selection can be made conditional without approximation.
- **Confidence:** medium — concrete numbers against real serving-framework baselines (SGLang) and a named prior method, but no code, no reproduction, and unstated author affiliation.
- **Recommendation:** monitor; flag alongside the existing block-sparse-attention-pisa and sparse-attention-decoding-ffd deferred topics at the next weekly review as a third, architecturally distinct entry in the "efficient sparse-attention scoring" space.

### C-20261014-03 · PR-OPD: hidden-state alignment, not token-probability shift, explains why privileged self-distillation works in agentic RL

- **Class:** B
- **Date discovered:** 2026-10-14
- **Date published:** 2026-09-29 (v1)
- **Source:** https://arxiv.org/abs/2609.36642 (verified resolves; v1 "Tue, 29 Sep 2026")
- **Organization/researchers:** Muyang Li, Jie Yang, Zhengyu Fang, Junchao Zhu, Zhengkun Xiao, Ruining Deng, Zhe Jiang, Shigang Chen — affiliations not stated on the abstract page.
- **Category:** RL/post-training (on-policy/privileged self-distillation — base technique taught in lesson 17.4)
- **What changed:** Privileged self-distillation gives an agent's skill-conditioned policy as a teacher to its own skill-free version via token-probability matching. The paper reports two findings that complicate the usual "the teacher's token distribution carries the signal" story: (1) "Invisible Advantage" — a skill in context raises WebShop success from 42.2% to 56.2% while changing the sampled-token probabilities of fewer than a quarter of tokens; (2) "Much to Align" — the same skill changes the hidden states of over 80% of response tokens, and linear probes can trace these hidden-state changes back to the specific skill. I.e. the useful signal lives mostly in hidden representations, not in the output token distribution the standard method distills from.
- **Technical summary:** Proposes PR-OPD: after a GRPO warm start, the policy writes a hindsight skill per trajectory, rereads its own response with that skill as a stop-gradient teacher, and aligns projected hidden states with the teacher's at every layer alongside the RL reward objective — no external skill library, separate teacher model, or extra inference cost. On ALFWorld and WebShop with two backbones, reports the best overall results in every setting: up to +4.7 points ALFWorld success and +14.0 points WebShop accuracy over GRPO. Code is linked from the abstract page.
- **Why it might matter:** A mechanistic explanation (not just another variant) for *why* self-distillation signals transfer in agentic RL, directly relevant to the course's on-policy-distillation material (17.4) and its growing deferred-variants cluster — the hidden-state-vs-token-probability distinction is a sharp, teachable finding (and a plausible numerical/from-scratch demonstration: probe hidden states vs. token probabilities on a toy skill-conditioning task).
- **Evidence of adoption:** None — single paper; code released but no independent reproduction or adoption by another training stack.
- **Major organizations using it:** None confirmed (no stated affiliation).
- **Open-source implementation:** Linked from the abstract page (GitHub; exact URL not independently re-verified beyond the abstract-page link in this run).
- **Paper:** https://arxiv.org/abs/2609.36642
- **Code:** linked from abstract page (see above)
- **Relationship to existing course material:** Bears on `lessons/module-17/lesson-04.md` (17.4) and the already-deferred `research/deferred/on-policy-distillation-variants.md` cluster. Checked via `lookup "privileged self-distillation skill hidden state"`, no match; this is a mechanistic/explanatory angle distinct from the cluster's existing variants (co-evolving teacher, teacher-scale laws, negative rollouts, token-selection, offline rollouts, relative-policy-difference transfer).
- **Potential course lesson:** A short addition to 17.4 contrasting token-probability-level distillation signal with hidden-state-level signal, using the "Invisible Advantage"/"Much to Align" framing and a small linear-probe exercise.
- **Confidence:** medium — code released and a concrete, falsifiable mechanistic claim, but single paper, no reproduction, no stated affiliation.
- **Recommendation:** flag at the next weekly review alongside the existing on-policy-distillation-variants cluster, specifically as a mechanistic (not just empirical) data point; also note today's second OPD variant, BranchOPD (C class, see below), so the review sees the cluster's continued growth this week.

### C-20261014-04 · Uncertainty-guided progressive data contraction: 450x less SFT data for reasoning post-training via Monte Carlo reasoning trees

- **Class:** B
- **Date discovered:** 2026-10-14
- **Date published:** 2026-10-09 (v1)
- **Source:** https://arxiv.org/abs/2610.12933 (verified resolves; v1 "Fri, 9 Oct 2026")
- **Organization/researchers:** Suorong Yang, Jiaxin Wang, Jiaming Su, Chentao Yue, Furao Shen, Peng Ye, Soujanya Poria — affiliations not stated on the abstract page.
- **Category:** data / reasoning / efficiency (SFT data curation for reasoning post-training)
- **What changed:** Rather than treating a teacher model's chain-of-thought as a single fixed path to imitate, the authors build Monte Carlo reasoning trees around intermediate reasoning states to expose process-level uncertainty, then use that to progressively contract the training corpus in two stages: an offline teacher-informed pool-shrinking stage, followed by a learner-aware stage that routes examples online during SFT based on the student's own evolving state.
- **Technical summary:** Reports using as little as 0.22% of the optimization budget of full-data SFT (>450x less training exposure) while maintaining or improving reasoning performance, and claims the approach generalizes across student model families, black-box teachers, and weak-to-strong transfer settings.
- **Why it might matter:** If the 450x figure holds up under scrutiny, this is a sharp, teachable result for the course's data-curation/efficient-training material — a concrete, falsifiable claim about how much of standard reasoning-SFT data is actually redundant, and a worked mechanism (uncertainty from sampled reasoning trees, not just loss or length heuristics) suitable for the six-pass lesson template's numerical/from-scratch passes.
- **Evidence of adoption:** None — single paper, no code link found on the abstract page, no independent reproduction. The 450x figure is the authors' own.
- **Major organizations using it:** None confirmed (no stated affiliation).
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.12933
- **Code:** none found
- **Relationship to existing course material:** New sub-topic — checked via `lookup "uncertainty guided data pruning reasoning SFT"` and `lookup "Monte Carlo reasoning tree uncertainty teacher"`, no matches in course or registry.
- **Potential course lesson:** Possible addition to the data-curation/synthetic-data material (or a reasoning-module cross-reference) on uncertainty-guided data selection, if reproduced — the specific 450x claim is extraordinary enough to warrant independent replication before any course commitment.
- **Confidence:** low-medium — a single, very strong, unreplicated claim; no code to check it against.
- **Recommendation:** monitor; this is exactly the kind of outsized single-paper claim the weekly review should treat with extra skepticism (WAIT pending reproduction or code release), despite being a clean, teachable mechanism if true.

### C-20261014-05 · HyperThink: replacing visible chain-of-thought with a single-forward-pass hypernetwork parameter update (COLM 2026)

- **Class:** B
- **Date discovered:** 2026-10-14
- **Date published:** 2026-10-02 (v1)
- **Source:** https://arxiv.org/abs/2610.03039 (verified resolves; v1 "Fri, 2 Oct 2026"; accepted at COLM 2026 per the Comments field)
- **Organization/researchers:** Donggyun Kim, Jack Lu, Chanwoo Kim, Mengye Ren, Seunghoon Hong — affiliations not stated on the abstract page.
- **Category:** reasoning / efficiency (test-time-compute alternative to visible chain-of-thought)
- **What changed:** Standard long-reasoning LLMs pay a latency cost dominated by sequential decoding of a long visible trace. HyperThink instead uses a lightweight hypernetwork that reads the question and, in a single forward pass, predicts updates to a small subset of the base model's parameters; a vector-quantized decoder restricts these updates to a finite set of reusable patterns. Trained on the base model's own outputs, the adapted model then answers with a concise step-by-step solution and no intermediate visible trace — i.e. the "reasoning" is encoded as a one-shot parameter edit rather than as tokens the model must decode.
- **Technical summary:** Reports better accuracy-latency trade-offs than baselines on math and general reasoning tasks, with the largest gains in near-non-thinking (low-token-budget) settings. Peer-reviewed and accepted at COLM 2026, which is a modest independent-validation signal beyond a self-published preprint, though still a single paper/single venue.
- **Why it might matter:** A genuinely distinct mechanism from the course's existing test-time-compute material (best-of-N/self-consistency, process rewards, GenCorrect-style iterative generate-verify-refine already staged as C-20261012-04) — instead of spending more decode tokens, it spends one hypernetwork forward pass to edit parameters. If it holds up, it's a sharp contrast case for a lesson on "where does reasoning compute go: tokens vs. weights."
- **Evidence of adoption:** None — single paper, no code link found on the abstract page, no reproduction beyond peer review. Results are the authors' own.
- **Major organizations using it:** None confirmed (no stated affiliation).
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.03039
- **Code:** none found
- **Relationship to existing course material:** New sub-topic — checked via `lookup "hypernetwork parameter update reasoning"` and `lookup "hypernetwork latent reasoning"`, no matches. Distinct from the course's existing test-time-scaling material (best-of-N/self-consistency, process rewards) and from the already-staged GenCorrect generate-verify-refine candidate (C-20261012-04, in `staging.md`) — both avoid visible CoT token cost but via opposite mechanisms (parameter edit vs. iterative token-level refinement).
- **Potential course lesson:** Possible addition to module 17's test-time-compute material contrasting token-space reasoning (CoT, best-of-N, iterative refinement) with parameter-space "reasoning" (HyperThink's single-shot hypernetwork edit), if independently reproduced.
- **Confidence:** low-medium — peer-reviewed (COLM 2026) but single paper, no code, no reproduction beyond that review.
- **Recommendation:** monitor for the next weekly review as a possible module-17 contrast case; peer-review acceptance nudges confidence slightly above a typical unreviewed preprint but not to A.


### C-20261015-01 · LatentLoop: asymmetric compute allocation across passes in looped MoE models (new evidence for deferred looped-recurrent-depth-lms topic)

- **Class:** B
- **Date discovered:** 2026-10-15
- **Date published:** 2026-10-14 (v1)
- **Source:** https://arxiv.org/abs/2610.19975 (verified resolves; v1 "Wed, 14 Oct 2026")
- **Organization/researchers:** Zefeng He, Yunkai Chai, Xiaoyang Chen, Tong Zhu, Zongliang Wu, Yun Luo, Yafu Li, Yu Cheng — affiliations not stated on the abstract page.
- **Category:** architecture (looped/recurrent-depth language models, MoE variant)
- **What changed:** In looped MoE models (a shared block of layers reused across passes with sparse MoE computation), the authors find the first pass through the shared block has a smaller residual-update scale than later passes: shrinking the first pass's effective compute preserves or improves performance, while shrinking the second/later pass hurts it. LatentLoop exploits this asymmetry — it compresses the sequence seen by the first pass, keeps full resolution on the second, reallocates the saved compute to a larger hidden size, learns how strongly the first pass updates the full sequence, shares MoE weights across passes, and uses separate attention parameters per pass.
- **Technical summary:** Across 15 configurations from 1.3B to 7B parameters (largest training budget above 10^21 FLOPs), LatentLoop lowers validation NLL versus both a vanilla (non-looped) MoE baseline and a symmetric looped-MoE baseline; at a fitted validation NLL of 1.988 the authors project 19.0% compute savings vs. vanilla MoE and 7.9% vs. the symmetric-looped baseline, plus improved average 0-shot/5-shot scores across nine benchmarks. No code repository found linked from the abstract page.
- **Why it might matter:** A concrete, scaling-law-backed sub-mechanism (where in a loop the compute should go) for the course's already-tracked looped/recurrent-depth cluster (`research/deferred/looped-recurrent-depth-lms.md`), and the first entry in that cluster specifically about combining looping with MoE sparsity rather than a dense shared block.
- **Evidence of adoption:** None — single paper, no code, no independent reproduction.
- **Major organizations using it:** None identified (no stated affiliation).
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.19975
- **Code:** none found
- **Relationship to existing course material:** New evidence for `research/deferred/looped-recurrent-depth-lms.md` (next review 2026-11-29) — checked via `lookup "looped recurrent depth MoE"`, which matched the deferred topic and the already-staged `C-20261013-01` (Elastic Latent Recurrence) entry. A fourth/fifth independent angle on the cluster, and the first that is MoE-specific rather than dense.
- **Potential course lesson:** Not a new lesson; a data point for the deferred topic's 2026-11-29 review, specifically on the interaction between looping and MoE sparsity (relevant if Module 12's MoE lesson and a future looped-models lesson are ever combined).
- **Confidence:** medium — rigorous scaling-law-style evaluation (15 configs, >10^21 FLOPs budget) but single paper, no code, no reproduction.
- **Recommendation:** add as a new data point to `looped-recurrent-depth-lms.md` at its 2026-11-29 review; not independently actionable before then.

### C-20261015-02 · The COUNT: n-gram count embeddings in host (CPU) memory as a cheap, training-free-adaptable pretraining addition

- **Class:** B
- **Date discovered:** 2026-10-15
- **Date published:** 2026-10-14 (v1)
- **Source:** https://arxiv.org/abs/2610.20456 (verified resolves; v1 "Wed, 14 Oct 2026")
- **Organization/researchers:** Ian Magnusson, Noah A. Smith, Pang Wei Koh — affiliations not stated on the abstract page, but these authors are established academics (UW/Ai2-affiliated in prior published work; not independently re-confirmed on this paper's own page).
- **Category:** data / efficiency (pretraining architecture using host memory)
- **What changed:** COUNT builds n-gram "embeddings" with no learned parameters: it precomputes n-gram counts and stores the count table in ordinary host (CPU) system memory — cheap and otherwise underused relative to GPU memory — then at each step uses corpus frequencies to combine the language model's own output embeddings for likely next tokens into an n-gram feature, added into the model. Because the count table lives off-GPU and can be swapped, a model can be adapted to a new domain at inference time just by swapping the table, with no additional training.
- **Technical summary:** At a fixed compute budget, adding COUNT to pretraining of 1B-parameter models improves validation loss by about as much as giving the baseline model 57% more compute. Under GPU memory constraints, COUNT outperforms splitting the same memory budget with learned n-gram embeddings (the obvious alternative that competes for the same scarce GPU memory). No code repository found linked from the abstract page.
- **Why it might matter:** A clean, mechanistically simple (non-parametric, frequency-based) idea that directly targets the course's efficient-training/memory-hierarchy material — it reframes host (CPU) memory, usually idle during GPU-bound pretraining, as a cheap place to store a useful signal, and the training-free domain-adaptation-via-table-swap property is a sharp, teachable contrast with learned-embedding approaches.
- **Evidence of adoption:** None — single paper, no code, no independent reproduction. The 57%-compute-equivalent figure is the authors' own.
- **Major organizations using it:** None confirmed (no stated affiliation on this paper).
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.20456
- **Code:** none found
- **Relationship to existing course material:** New sub-topic — checked via `lookup "n-gram embedding host memory pretraining"`, no match in course or registry. Distinct from the course's existing data-curation and distributed-training material (this is a model-side architecture/memory-hierarchy trick, not a data-curation or multi-GPU technique).
- **Potential course lesson:** Possible small addition to the training/efficiency material: a from-scratch worked example of a non-parametric n-gram feature stored off-GPU, contrasted with a learned embedding table, plus the training-free domain-swap property as an exercise.
- **Confidence:** medium — reputable-looking author line, clean falsifiable mechanism and concrete compute-equivalence numbers, but single paper, no code, no reproduction.
- **Recommendation:** monitor for the next weekly review as a possible small addition to the efficient-training material; the "idle host memory as a cheap feature store" framing is worth flagging even before any reproduction, given how directly it maps to a real systems constraint (GPU memory scarcity) the course already discusses.

### C-20261015-03 · Ares: spectral-driven adaptive-rank optimizer exploiting gradient-subspace collapse specific to RLVR

- **Class:** B
- **Date discovered:** 2026-10-15
- **Date published:** 2026-10-12 (v1)
- **Source:** https://arxiv.org/abs/2610.16533 (verified resolves; v1 "Mon, 12 Oct 2026")
- **Organization/researchers:** Zhipeng Chen, Wayne Xin Zhao, Ji-Rong Wen — RUCAIBox (Renmin University of China); Wayne Xin Zhao and Ji-Rong Wen are established authors associated with that lab in prior published work (not independently re-confirmed on this paper's own page beyond the linked GitHub org `RUCAIBox`).
- **Category:** RL/post-training (optimizer design specific to RLVR)
- **What changed:** The authors ask whether RLVR needs an optimizer distinct from the ones (e.g. Adam/AdamW) built for supervised training. They find that during RLVR, gradient energy increasingly concentrates into a small subspace as training progresses — a pattern that is absent during supervised fine-tuning and largely disappears when rewards are randomized (i.e. it is specific to learning from verifiable rewards, not an artifact of any policy-gradient update). The number of directions needed to capture a fixed share of gradient energy also shrinks over training. Ares estimates this effective update rank directly from the gradient's spectral energy, tracks how it changes step to step, and restricts optimizer updates to that low-rank subspace.
- **Technical summary:** Reports accuracy improvements over widely used optimizers across multiple model sizes, RLVR algorithms, and tasks (exact baseline deltas not independently re-tabulated in this run). Code released at the linked GitHub repository.
- **Why it might matter:** Directly extends the course's RLVR material (`lessons/module-16/lesson-02.md`) with a mechanistic finding about *why* RLVR training dynamics differ from supervised training at the optimizer level, plus a working, code-released optimizer built on that finding — a good candidate for a from-scratch numerical demonstration of gradient-subspace collapse under verifiable vs. random rewards.
- **Evidence of adoption:** None beyond the paper's own released code — single paper, no third-party adoption or reproduction yet.
- **Major organizations using it:** RUCAIBox (originator); no other named adopters.
- **Open-source implementation:** https://github.com/RUCAIBox/Ares
- **Paper:** https://arxiv.org/abs/2610.16533
- **Code:** https://github.com/RUCAIBox/Ares
- **Relationship to existing course material:** New sub-topic bearing on `lessons/module-16/lesson-01.md` (16.1 · From PPO to GRPO) and `lessons/module-16/lesson-02.md` (16.2 · RLVR & verifiable rewards) — checked via `lookup "low-rank gradient subspace RLVR optimizer"` and `lookup "Muon optimizer"`, no matches; this is a new optimizer-design angle on RLVR, not yet covered.
- **Potential course lesson:** Possible addition to Module 16 (or the optimizer material in Module 08) contrasting standard AdamW-style updates with a verifiable-reward-specific low-rank update, using the random-vs-verifiable-reward gradient-energy comparison as the key numerical illustration.
- **Confidence:** medium-high — code released, a well-known lab, and a concrete, falsifiable mechanistic claim (gradient-subspace collapse specific to verifiable rewards), but still single-paper, no independent reproduction or third-party adoption.
- **Recommendation:** monitor for the next weekly review as a possible Module 16 extension; the released code makes an independent sanity-check (reproduce the gradient-energy-concentration plot on a toy RLVR task) feasible before any lesson decision.

### C-20261015-04 · "More Methods, Better Agents?" — a 132-run, ~12,124-GPU-hour reality check finding no self-distillation method consistently beats GRPO in agentic RL

- **Class:** B
- **Date discovered:** 2026-10-15
- **Date published:** 2026-10-14 (v1)
- **Source:** https://arxiv.org/abs/2610.19789 (verified resolves; v1 "Wed, 14 Oct 2026")
- **Organization/researchers:** Linhao Zhong, Bozhen Fang, Xuehao Wang, Duochao Shi, Zekai Luo, Guangming Yao, Yi Yuan, Hao Chen, Jingdong Chen, Chunhua Shen — affiliations not stated on the abstract page; Chunhua Shen is an established computer-vision/ML researcher in prior published work (not independently re-confirmed on this paper's own page).
- **Category:** RL/post-training (evaluation methodology for self-distillation in agentic RL)
- **What changed:** The paper argues that self-distillation methods for agentic RL have proliferated faster than rigorous comparison: prior papers often reuse published baseline numbers under inconsistent conditions, rarely compare methods head-to-head, and pay little attention to training stability or compute cost. The authors benchmark ten representative self-distillation methods against plain GRPO on ALFWorld, ScienceWorld, Search-QA, and WebShop, across three model sizes, with 132 training runs totaling roughly 12,124 H800 GPU-hours and repeated evaluation over eight rollout seeds.
- **Technical summary:** Three headline findings: (1) gains over GRPO are limited and no method consistently outperforms it across settings; (2) strong final-performance numbers can mask unstable training along the way; (3) additional training cost does not reliably translate into better final performance. Project page (no GitHub code repository found linked): https://aim-uofa.github.io/Self-Distillation-AgentRL/.
- **Why it might matter:** This is exactly the kind of rigorous, compute-heavy negative/cautionary result the weekly review should weigh against the fast-growing on-policy-distillation/self-distillation-variants cluster the daily runs have been feeding all week (`research/deferred/on-policy-distillation-variants.md`, plus this week's new OPD-variant papers logged as C items below) — it directly questions whether that proliferation of single-paper variants represents real progress or mostly noise/inconsistent baselines, which is a load-bearing input for how much weight the next weekly review should give any individual variant.
- **Evidence of adoption:** Not an adoption signal itself — it's a benchmark/evaluation study. Evidentiary strength comes from its own methodological rigor (large, controlled, multi-seed comparison) rather than external adoption.
- **Major organizations using it:** None confirmed (no stated affiliation).
- **Open-source implementation:** Project page only (https://aim-uofa.github.io/Self-Distillation-AgentRL/); no GitHub repository found.
- **Paper:** https://arxiv.org/abs/2610.19789
- **Code:** none found (project page only)
- **Relationship to existing course material:** Directly bears on `research/deferred/on-policy-distillation-variants.md` (next review 2026-11-29) and the accepted `research/accepted/on-policy-distillation.md` / `lessons/module-17/lesson-04.md` (17.4) — checked via `lookup "self-distillation agentic RL benchmark reality check"`, no match (new methodological angle, not a new variant). Flag together with this week's and prior weeks' growing OPD-variant pile (PR-OPD, relative-policy-difference OPD, BranchOPD, and today's CriticOPD, C item below) at the next weekly review.
- **Potential course lesson:** Not a new lesson on its own; strong supporting evidence for treating the on-policy-distillation-variants cluster conservatively at its next review, and a possible "how to benchmark post-training methods rigorously" cautionary example/exercise in the evaluation module.
- **Confidence:** medium-high — unusually large, controlled, multi-seed compute investment for a single paper, but still single-source (no independent replication of its own findings) and no code to verify its own setup.
- **Recommendation:** flag prominently at the next weekly review alongside the on-policy-distillation-variants cluster — this paper is evidence for treating that whole cluster's claims with more skepticism, not a reason to add anything to the course itself.
