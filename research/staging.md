# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261026-01 · Where Rounding Happens Matters: Rounding-Location Alignment Stabilizes LLM Reinforcement Learning

- **Class:** B
- **Date discovered:** 2026-10-26
- **Date published:** 2026-10-23 (v1)
- **Source:** https://arxiv.org/abs/2610.32951 (verified resolves; v1, Fri 23 Oct 2026, cs.LG; announced in Monday's batch, having missed Friday's cutoff)
- **Organization/researchers:** Yijing Ke, Tianle Cai, Ruosong Wang (affiliations not stated on the abstract page; Tianle Cai is a named author on prior speculative-decoding/inference-systems work, general knowledge, not independently re-confirmed here).
- **Category:** RL/post-training (root cause of training-inference mismatch in LLM RL)
- **What changed:** The course's new lesson 16.4 (`lessons/module-16/lesson-04.md`, accepted 2026-10-25) teaches that an RL trainer and its inference engine can assign different probabilities to the same weights/tokens even in BF16, making a nominally on-policy update off-policy, and surveys two production fixes (remove the mismatch via batch-invariant kernels, or correct it via truncated importance sampling). This paper supplies a specific, previously-unlisted root cause and fix: rather than a global precision/kernel change, it aligns only *where inside the forward pass* BF16 rounding happens — keeping RoPE, SwiGLU, and residual-RMSNorm intermediates in FP32 until they produce BF16 outputs, matching the inference engine's rounding points without touching the inference engine, the matmul/attention kernels, or the RL objective itself.
- **Technical summary:** Across four Qwen-family models, aligning all three regions (RoPE + SwiGLU + residual RMSNorm) prevents the training collapse seen with the default BF16 baseline (authors' own runs); aligning only the RoPE path alone was also sufficient in the studied runs even though some numerical discrepancy remained elsewhere. No independent reproduction found; no code repository found on the abstract page as of this check.
- **Why it might matter:** It is a concrete, mechanism-level (not just systems-level) answer to "where exactly does the bitwise mismatch 16.4 describes actually come from inside the model's own forward pass," phrased as three specific, checkable intermediate-precision changes rather than a new kernel library or a correction term in the RL loss — a natural worked addendum to 16.4's existing "what causes the mismatch" section.
- **Evidence of adoption:** None found — single paper, three days old at discovery, no stated venue, no code repository located.
- **Major organizations using it:** None identified; no stated lab affiliation.
- **Open-source implementation:** none found on the abstract page.
- **Paper:** https://arxiv.org/abs/2610.32951
- **Code:** none found
- **Relationship to existing course material:** Directly extends `lessons/module-16/lesson-04.md` and `research/accepted/rl-training-inference-mismatch.md` (16.4, ADD 2026-10-25) and `code/src/llmre/rl/mismatch.py`. Checked via `lookup "rounding LLM reinforcement learning"` (no registry/course hit beyond an unrelated precision line already in 16.4) — new, not a duplicate of the accepted topic's existing evidence.
- **Potential course lesson:** If reproduced or adopted, a short worked addition to 16.4's mismatch-causes section: a concrete example of *where* BF16 rounding inside RoPE/SwiGLU/RMSNorm creates the mismatch, and the FP32-intermediate fix, as a third concrete mechanism alongside batch-invariant kernels and TIS correction.
- **Confidence:** medium — a specific, falsifiable mechanism with authors' own measured results across four real model sizes, directly on-topic for already-accepted material, but one paper, no code, no independent reproduction.
- **Recommendation:** monitor; revisit at the next weekly cycle given how directly it extends the just-accepted 16.4 — worth an early look even without independent reproduction, since it is cheap to spot-check against `code/src/llmre/rl/mismatch.py`'s existing toy harness.

### C-20261026-02 · When and How Does Muon Precondition? The Role of Structured Gradient Noise

- **Class:** B
- **Date discovered:** 2026-10-26
- **Date published:** 2026-10-22 (v1)
- **Source:** https://arxiv.org/abs/2610.31728 (verified resolves; v1, Thu 22 Oct 2026 15:20:40 UTC, cs.LG; comments: "62 pages, 11 figures")
- **Organization/researchers:** Xiaoran Chen, Binghui Li, Liwei Wang, Pinyan Lu (affiliations not stated on the abstract page).
- **Category:** training/optimization (theoretical analysis of Muon, already taught in lesson 03.4)
- **What changed:** The course already teaches Muon (`lessons/module-03/lesson-04.md`) and the registry already has one deferred theory paper on it (`research/deferred/muon-polar-update-theory.md`, a "quadratic lens" on when orthogonalization helps via singular-value structure). This is a different theoretical account: it asks when Muon's orthogonalized update helps at all, as a function of *gradient-noise geometry* rather than curvature. It first shows, theoretically and empirically, that under isotropic noise Muon gives no benefit beyond a learning-rate rescaling — i.e., orthogonalization alone buys nothing without structure in the noise. It then analyzes structured noise from mini-batch sampling in a toy associative-memory model with uneven task frequencies, deriving a balancing threshold (extended to many tasks) under which Muon turns the usual linear dependence on task frequency into a square-root dependence, helping rare tasks train alongside frequent ones; it also identifies a rotation-induced negative-transfer failure mode of plain SignSGD that Muon's orthogonalization avoids, and gives an account of how momentum reduces sampling noise while still tracking a moving gradient.
- **Technical summary:** Controlled experiments on the toy associative-memory model support the derived threshold and the frequency-balancing claim; the paper also reports a preliminary real LLM-pretraining study suggesting practical relevance for a fast/slow-momentum variant of Muon. No code link found on the abstract page; no stated venue (62-page preprint).
- **Why it might matter:** This is a distinct, falsifiable mechanism (noise-geometry dependent task-frequency balancing) from the already-deferred quadratic-lens paper (curvature/singular-value dependent) — together they would give 03.4 two independent, non-overlapping "why orthogonalize" stories if either is eventually corroborated, rather than one being new evidence for the other's specific claim.
- **Evidence of adoption:** None — single paper, four days old at discovery, no stated venue, no code release found, no third-party reproduction.
- **Major organizations using it:** None identified; academic authorship, no stated lab affiliation.
- **Open-source implementation:** none found on the abstract page.
- **Paper:** https://arxiv.org/abs/2610.31728
- **Code:** none found
- **Relationship to existing course material:** Bears on `lessons/module-03/lesson-04.md` and `code/src/llmre/optim/muon.py`, and is thematically adjacent to (but mechanistically independent of) `research/deferred/muon-polar-update-theory.md` — checked via `lookup "Newton-Schulz orthogonalization Muon"`, which surfaces only that existing deferred file and the taught lesson; this paper's specific claim (noise-geometry-driven task-frequency balancing) is not covered there, so it is new, not a duplicate requiring no new record.
- **Potential course lesson:** If independently reproduced, a second short theory note for 03.4 alongside the quadratic-lens one: Muon's benefit under isotropic noise is nil (rules out a naive "orthogonalizing always helps" framing), and its benefit under realistic structured mini-batch noise is a frequency-balancing effect, demonstrated on a toy task mirroring 03.4's existing worked examples.
- **Confidence:** low-medium — a real theoretical analysis (not just an ablation) with a toy-model derivation and one preliminary real-pretraining check, but single paper, no code, no venue, no reproduction, and a somewhat narrow toy setting (two-task/few-task associative memory) for the main theorem.
- **Recommendation:** monitor; revisit if a main-venue version appears, code is released, or the fast/slow-momentum pretraining claim is independently checked — otherwise keep alongside (not merged into) the quadratic-lens deferred topic, since the two papers argue from different mechanisms.

### C-20261027-01 · Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution

- **Class:** B
- **Date discovered:** 2026-10-27
- **Date published:** 2026-10-23 (v1)
- **Source:** https://arxiv.org/abs/2610.32285 (verified resolves; v1, Fri 23 Oct 2026 07:43:26 UTC, cs.AI; discovered via Hugging Face's Oct 27 daily-papers curation — not yet in arXiv's announced daily listing as of this run)
- **Organization/researchers:** Zhiyong Cao, Hengjie Zhu, Yongxin Wang, and 20 further co-authors (23 total); affiliations not stated on the abstract page.
- **Category:** agents/tool use (evaluation/benchmarking of agent harnesses, distinct from evaluating the underlying model)
- **What changed:** Introduces a benchmark (named "TraceTrial-Bench" in the abstract) that scores harness reliability — separately from task/model performance — across seven harness components: context, execution, governance, lifecycle, observability, tools, and verification. It has a trajectory-tracing module to capture execution traces, a two-layer evaluation that separates "did the task succeed" from "did the harness behave correctly," a multi-source evidence-acquisition module feeding an LLM-based evaluator, and a failure-attribution module that assigns blame along mechanism chains per run.
- **Technical summary:** 4 harnesses × 9 models × 95 expert-level tasks = 3,420 runs (authors' own study). Macro-averaged harness scores ranged 59.8%–72.0%. In 31.1% of runs the agent completed the task despite the harness failing at least one of the seven criteria — an empirical demonstration that task success and harness reliability diverge. No code repository found on the abstract page; no stated venue.
- **Why it might matter:** Operationalizes "the harness needs its own evaluation, not just the model/task outcome" as a scored, multi-component, auditable benchmark — a different angle from the two related tracked topics (below), useful if the course's agents module ever teaches agent/harness evaluation as its own topic.
- **Evidence of adoption:** none found — single paper, 4 days old at discovery, no code repo, no third-party use.
- **Major organizations using it:** none identified; no stated lab affiliation.
- **Open-source implementation:** none found on the abstract page.
- **Paper:** https://arxiv.org/abs/2610.32285
- **Code:** none found
- **Relationship to existing course material:** adjacent to two already-tracked WAIT topics — `research/deferred/agent-eval-terminal-state-reliability.md` (grades by backend terminal state + pass^N repeat-reliability) and `research/deferred/agent-harness-policy-co-optimization.md` (co-trains harness and policy together) — checked via `lookup "agent harness benchmark reliability failure attribution"` (no hit). This paper's angle (component-level harness scoring + mechanism-chain failure attribution, decoupled from the policy) is distinct from both, so it is new, not a duplicate.
- **Potential course lesson:** if reproduced/adopted, a short addition to the agents/tool-use module's evaluation section distinguishing model evaluation from harness evaluation, using the 31.1% task-success-despite-harness-failure figure as a concrete worked number.
- **Confidence:** low-medium — a large study (3,420 runs, 23 authors) but one paper, no code, no reproduction, no venue.
- **Recommendation:** monitor; revisit alongside the two related WAIT topics at their scheduled reviews, or sooner if code or the benchmark's task set is released.

### C-20261027-02 · Continual Learning Exam: Benchmarking Long-Horizon RSI and Experiential Learning in LLM Agents

- **Class:** B
- **Date discovered:** 2026-10-27
- **Date published:** 2026-10-25 (v1)
- **Source:** https://arxiv.org/abs/2610.33956 (verified resolves; v1, Sun 25 Oct 2026 00:00:25 UTC; primary cs.LG, cross-listed cs.AI/cs.CL/cs.CV; discovered via Hugging Face's Oct 27 daily-papers curation — not yet in arXiv's announced daily listing as of this run)
- **Organization/researchers:** Haoyu Zhao, Sitian Tu, Yuxuan Huang, and 16 further co-authors (19 total), including Huaxiu Yao and Jun Wang; affiliations not stated on the abstract page.
- **Category:** agents/memory (benchmark for continual/experiential learning in LLM agents, not a training method itself)
- **What changed:** Introduces the Continual Learning Exam (CLE), built with 50+ domain experts: 120 human-authored tasks, each a 5-level ladder (600 levels total) spanning 89 professional roles across 12 occupational domains, where passing a level is required to unlock the next and later levels require reusing experience accumulated within the same task. Grading is by an LLM judge against human-written rubrics; the paper runs a controlled comparison of single-submission attempts vs. attempts given one diagnostic-feedback-driven revision.
- **Technical summary:** across 17 harness-model configurations (authors' own results), allowing one feedback-driven revision gives a median +5.3 percentage-point completion gain over single-shot; even the best system clears only 47.7% of levels overall and completes all 5 levels of a task in only 23.3% of tasks. A project page exists (https://haoyu-zhao.github.io/CLE/index.html); no code or dataset release found on the abstract page.
- **Why it might matter:** a large, expert-authored way to measure whether an agent actually improves *within* a task as experience accumulates (vs. single-task snapshot evaluation) — distinct from the course's existing context/memory tracking, which is about compressing/managing context rather than measuring experiential improvement; could give any future agent-memory lesson a concrete evaluation protocol.
- **Evidence of adoption:** none — single paper, two days old at discovery, no code/dataset released, no third-party use.
- **Major organizations using it:** none identified; academic authorship, no stated lab affiliation.
- **Open-source implementation:** none found (project page only, no code/dataset link).
- **Paper:** https://arxiv.org/abs/2610.33956
- **Code:** none found
- **Relationship to existing course material:** thematically adjacent to `research/deferred/agent-context-compaction.md` (WAIT — context/memory compaction for long-horizon agents) but measures a different thing (graded experiential improvement vs. context-management mechanics); checked via `lookup "continual learning long horizon LLM agents benchmark"` (no hit) — new, not a duplicate.
- **Potential course lesson:** if the dataset/code is released and the benchmark sees independent use, a short evaluation-section addition near the agent-memory material using CLE's single-shot-vs-revision protocol as a diagnostic for whether a memory/compaction mechanism captures reusable experience.
- **Confidence:** low — careful, large-scale benchmark construction (50+ experts, 600 levels) but one paper, no code/dataset, no reproduction, no venue.
- **Recommendation:** monitor; revisit if the dataset or code is released, or if it is used to evaluate a memory/continual-learning method the course already teaches.

### C-20261027-03 · Extreme Compression of Per-Layer Embeddings via Cross-Layer Latent Vector Quantization (OPTIC)

- **Class:** B
- **Date discovered:** 2026-10-27
- **Date published:** 2026-10-22 (v1)
- **Source:** https://arxiv.org/abs/2610.31309 (verified resolves; v1, Thu 22 Oct 2026 14:07:01 UTC, cs.LG; discovered via Hugging Face's Oct 27 daily-papers curation — not yet in arXiv's announced daily listing as of this run)
- **Organization/researchers:** Jiacheng Liu, Yaxin Luo, Xiaohan Zhao, Peng Sun, Xinyi Shang, Jiacheng Cui, Yonina C. Eldar, Zhiqiang Shen; affiliations not stated on the abstract page.
- **Category:** efficiency/quantization (compressing a specific embedding-table architecture component, not general weight quantization)
- **What changed:** Targets Per-Layer Embeddings (PLE) in Gemma 4 E4B, which the paper states make up 35.2% of that checkpoint's parameters. Rather than giving each token's per-layer vector its own quantization code (as conventional vector quantizers do), OPTIC stores one quantized latent code tuple per token and decodes it at every layer with a layer-specific affine decoder, fitting the codes/decoders to the PLE's output path (what the embeddings produce downstream) rather than to minimizing reconstruction error on the raw stored vectors — exploiting redundancy in how the same token's embedding is reused across layers.
- **Technical summary:** (authors' own numbers) ~100× PLE compression retains 97.8% of aggregate performance across 7 benchmarks, vs. 66.3% for the strongest conventional baseline (product quantization); 594× and 1,002× compression retain 90.5% and 84.1% respectively; applied to the official W4A16-quantized E4B release, ~100× compression retains 96.9%; at ~100× compression, BF16 E4B uses 29.7% less GPU memory than with uncompressed PLE at matched KV-cache size, with at most a 0.40% latency slowdown. Code released at github.com/VILA-Lab/optic. No stated venue.
- **Why it might matter:** a concrete, code-backed, function-aware (not reconstruction-aware) quantization technique for a specific but increasingly common architecture pattern — large per-token, per-layer embedding tables, as used in Gemma 3n/4-style elastic/matryoshka architectures — and a clean worked example of "quantize for what the thing is used for, not for minimizing its own reconstruction error."
- **Evidence of adoption:** none found beyond the authors' own code release — single paper, 5 days old at discovery, no third-party reproduction, no stated venue.
- **Major organizations using it:** none identified (authors appear academic, VILA-Lab); the technique targets Google's Gemma 4 architecture, but there is no evidence Google has adopted OPTIC itself.
- **Open-source implementation:** https://github.com/VILA-Lab/optic
- **Paper:** https://arxiv.org/abs/2610.31309
- **Code:** https://github.com/VILA-Lab/optic
- **Relationship to existing course material:** checked via `lookup "per-layer embedding compression quantization Gemma"` (no hit) — new. The course's existing quantization material (`research/deferred/semi-structured-sparsity-vs-quantization.md`, `research/deferred/low-bit-optimizer-state-quantization.md`) covers weight and optimizer-state quantization, not per-layer embedding-table compression, and the course does not currently teach the Gemma-style per-layer-embedding architecture this technique presupposes — so this is a new sub-topic, not evidence for an existing one.
- **Potential course lesson:** if the per-layer-embedding architecture pattern is ever taught (e.g. as a mobile/multimodal-efficient architecture variant) and this technique sees reuse, a short case study on function-aware vs. reconstruction-aware vector quantization for embedding tables.
- **Confidence:** medium — concrete numbers across several compression ratios against a real deployed-model target (official W4A16 Gemma 4 E4B), with code released, but one paper, no venue, no reproduction, and it presupposes an architecture the course doesn't yet teach.
- **Recommendation:** monitor; low priority unless the course later adds Gemma-style per-layer-embedding architectures, or the technique is picked up by a quantization library (e.g. bitsandbytes, AWQ/GPTQ-style tooling).
