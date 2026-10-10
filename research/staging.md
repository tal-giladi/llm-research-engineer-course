# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261005-01 · Looped (recurrent-depth) language models: practical training recipes + a second independent line with code

- **Class:** A
- **Date discovered:** 2026-10-05
- **Date published:** 2026-09-30 (source 1); 2026-10-05 (source 2, added to this record 2026-10-07)
- **Source:** https://arxiv.org/abs/2610.00673 (source 1, verified resolves; v1 "Wed, 30 Sep 2026 20:09:58 UTC"); https://arxiv.org/abs/2610.06833 (source 2, "Towards Looped Models Done Right, Part II: Rethinking at Fixed Points", verified resolves; v1 "Mon, 5 Oct 2026"); https://arxiv.org/abs/2610.11570 (source 3, "Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals" / InfiLoop, added 2026-10-09, verified resolves; v1 "Thu, 8 Oct 2026 09:26:21 UTC")
- **Organization/researchers:** Source 1: Andrei Marchenko, Viacheslav Bezrukov, Oleg Kashurin, Inessa Fedorova, Dmitry Bocharov, Yuliana Shakhvalieva, Maria Tikhonova, Valerii Ternovskii (independent/academic group). Source 2: Benhao Huang, Chufan Shi, Junlin Chen, Shicheng Wen, Zhengzhong Liu, Eric Xing, Xuezhe Ma (academic, MBZUAI/USC lineage) — a different, independent group; explicitly "Part II" of a prior paper in the same line, and compares directly against Huginn (Geiping et al.'s known recurrent-depth model). Source 3: Pengxiang Li, Dilxat Muhtar, Di He, Guinan Su, Lu Yin, Shiwei Liu (affiliations not stated; a third independent academic group).
- **Category:** architecture
- **What changed:** Practical recipe for training "looped" LMs — architectures that increase effective depth by repeatedly applying a shared block of layers (recurrent-depth, in the lineage of Universal Transformer / ALBERT-style weight sharing and 2025's recurrent-depth latent-reasoning work) — claiming large reductions in the token budget needed to make the approach competitive, plus a lightweight recipe for converting an already-pretrained dense checkpoint into a looped variant. Added 2026-10-07: a second, independent group's paper shows that because looped-model recurrent states approach a fixed point, this enables truncated backprop in training, terminal KV-cache sharing at decode time, faster prefill via a distilled student, and cheaper RL updates (gradients from saved rollout states instead of backprop through the replayed trajectory) — then improves the depth prior and input-injection scheme that shape how tightly states converge to that fixed point. Added 2026-10-09: a third, independent group (InfiLoop) identifies a different failure mode — looped models degrade in accuracy as iteration count grows because noisy state updates overwrite correct intermediate results and errors compound through the recurrence — and fixes it with a loop-native residual connection (content-weighted, learned-temporal-decay running summary of past states via an exact streaming recurrence with constant memory), rather than addressing training cost (source 1) or inference/RL efficiency (source 2).
- **Technical summary:** Source 1 combines pretraining with targeted mid-training, warmup schedules and regularization changes to cut the data needed from 7.7T to 310B tokens for a competitive LoopLM; reports a 1.4B-parameter LoopLM beating equivalently-sized dense models (+14 GSM8K, +10 MATH, +22 DROP, per the abstract) and matching a 3.9B dense model's quality at 36% of its parameters, plus a dense-to-looped conversion recipe using one learned input-mixing scalar and a smoothed exit loss. Source 2, evaluated 100M-1.6B parameters: learned depth prior + orthogonal injection lower perplexity at every scale vs. Huginn's prior/injection scheme; at 1.6B, the learned prior with a 3x smaller KV cache matches the downstream-task average of fixed-depth training with the full cache; reports 1.79x faster prefill (distilled student) and 2x faster RL gradient updates; code and checkpoints released at https://github.com/ifm-ai/xllm-loop. Source 3 (InfiLoop): a 7M-parameter model with the loop-native residual reaches 97.9% exact accuracy on Sudoku-Extreme and 13.6% pass@2 on ARC-AGI-2, outperforming existing recursive architectures, and keeps improving with test-time looping beyond 20,000 effective steps on Sudoku-Extreme — an order of magnitude beyond the iteration counts sources 1-2 discuss; code (Sudoku-Extreme portion) released and verified live at https://github.com/pixeli99/InfiLoop.
- **Why it might matter:** If the dense-to-looped conversion and the reduced token budget hold up, this would be a cheap way to trade extra compute-per-token for fewer parameters/tokens — directly relevant to the course's architecture and efficient-training material (module on attention/architecture variants, alongside the existing Gated DeltaNet lesson). The second source adds concrete, mechanism-level inference/RL efficiency payoffs (KV-cache reduction, prefill speedup, cheaper RL) and a public code release, and engages directly with Huginn, an already-known recurrent-depth model. The third source (InfiLoop) adds a third independent angle — recurrence stability at extreme test-time iteration counts — with its own public, verified code, meaningfully strengthening the evidence base: three independent groups, three complementary sub-problems (training cost, inference/RL efficiency, recurrence stability), within roughly one week.
- **Evidence of adoption:** None yet beyond the papers' own code (sources 2 and 3); no independent reproduction, no major-lab or major-stack adoption found.
- **Major organizations using it:** None identified; academic groups only.
- **Open-source implementation:** Source 1: none found. Source 2: https://github.com/ifm-ai/xllm-loop. Source 3: https://github.com/pixeli99/InfiLoop (verified live; Sudoku-Extreme code present, other benchmarks' code pending per the README).
- **Paper:** https://arxiv.org/abs/2610.00673 (source 1); https://arxiv.org/abs/2610.06833 (source 2); https://arxiv.org/abs/2610.11570 (source 3)
- **Code:** https://github.com/ifm-ai/xllm-loop (source 2); https://github.com/pixeli99/InfiLoop (source 3); none for source 1
- **Relationship to existing course material:** New — the registry/lessons have no existing "looped / recurrent-depth / weight-shared-layer" topic (checked via `research/tools/research.py lookup`); closest existing material is module-12/lesson-05 on Gated DeltaNet/hybrid linear attention, which is a different axis (attention mechanism, not depth-sharing).
- **Potential course lesson:** Candidate module-12 addition (or frontier-update note) on recurrent-depth/looped transformers: architecture + training (depth priors, input injection) + the inference/RL efficiency mechanism (fixed-point KV sharing) + recurrence-stability fixes (loop-native residuals) at extreme test-time iteration counts.
- **Confidence:** medium-high (raised from medium on 2026-10-09) — three independent groups within roughly one week, two with public code, concrete mechanism-level numbers, and direct engagement with/comparison against already-known models and techniques; still no third-party reproduction or production adoption.
- **Recommendation:** review for ADD (or at minimum a frontier-update note) at the next weekly review — three independent sources across complementary sub-problems, concrete numbers, and two public code releases clear the bar well beyond the single 10-05 paper.

### C-20261006-01 · ASCENT: online test-time training of long-horizon agents via self-distillation

- **Class:** B
- **Date discovered:** 2026-10-06
- **Date published:** 2026-10-04
- **Source:** https://arxiv.org/abs/2610.05303 (verified resolves; v1 timestamp "Sun, 4 Oct 2026")
- **Organization/researchers:** Haodong Lu, Dong Gong (affiliation not stated on the abstract page; independent/academic)
- **Category:** agents/tools (adjacent to RL/post-training — self-distillation mechanism)
- **What changed:** Proposes "Online Agentic Test-Time Training" (OaTTT): instead of storing reflections/memories as retrievable text (the usual in-context-adaptation approach), it updates the agent's weights (persistent LoRA) during deployment, directly from its own verified trajectories, with no external reference solution or stronger teacher.
- **Technical summary:** For each deployed task, a frozen initial copy of the LLM treats the agent's own successfully-verified trajectory as privileged hindsight and produces next-token target distributions along it; these are self-distilled into LoRA fast weights, with invalid-action turns stripped out before distillation to stabilize the signal (directly imitating/reinforcing the raw single-attempt trajectory is reported to destabilize the policy). Evaluated on ALFWorld, WebShop, and AppWorld across model scales; reports accumulating task-success and interaction-efficiency gains over online in-context adaptation baselines, with transfer to held-out scenes.
- **Why it might matter:** Offers a concrete, weight-update-based alternative to the context/memory-growth approaches for long-horizon agents that the course is already tracking (see deferred `agent-context-compaction.md`), and is a different lens on the self-distillation mechanism the course teaches in lesson 17.4 (there, distillation uses an external reference solution at training time; here it's the agent's own verified trajectory at test time, with no external teacher).
- **Evidence of adoption:** None — single paper (2 authors), no independent reproduction, no major-lab or stack adoption found.
- **Major organizations using it:** None identified.
- **Open-source implementation:** None linked from the abstract page as of this check.
- **Paper:** https://arxiv.org/abs/2610.05303
- **Code:** none found
- **Relationship to existing course material:** New — registry/lessons have no "test-time weight training for agents" topic. Related but distinct from `research/accepted/on-policy-distillation.md` (lesson 17.4, training-time distillation with an external reference) and `research/deferred/agent-context-compaction.md` (context/memory management rather than weight updates).
- **Potential course lesson:** Possible extension to the agents/tool-use module or the RL/post-training module (test-time training as a category), only if a second group reproduces the stability claims or a major agent framework adopts persistent-weight test-time training.
- **Confidence:** low
- **Recommendation:** monitor — plausible mechanism with only two authors' single-paper validation on three benchmark suites; revisit on independent reproduction or adoption by an agent framework.

### C-20261006-02 · OSWorld-Pro: process-based evaluation for computer-use agents

- **Class:** B
- **Date discovered:** 2026-10-06
- **Date published:** 2026-09-21
- **Source:** https://arxiv.org/abs/2609.24890 (verified resolves; v1 timestamp "Mon, 21 Sep 2026 16:55:24 UTC")
- **Organization/researchers:** Zhilin Wang, Shaokun Zhang, Yifan Zhang, Hao Zhang, Jin Xu, Binfeng Xu, Jian Hu, Yunheng Zou, Karan Sapra, Andrew Tao, Jan Kautz, Yi Dong — NVIDIA
- **Category:** evaluation (agents/tools)
- **What changed:** Moves computer-use-agent (CUA) evaluation from end-state-only functional verification (as in the original OSWorld) to procedural, subgoal-level evaluation, exposing *where and why* agents fail rather than only whether the final task succeeded.
- **Technical summary:** Introduces 300+ tasks decomposed into 2,800+ subgoals, grounded in 67,000+ human annotations; uses human-aligned LLM-judges to score sequential subgoal fulfillment. Reports that OSWorld-Pro is harder and more diagnostic than OSWorld even for the best models (e.g., Claude Opus 5 at 75.7% on OSWorld-Pro vs. 83.4% on OSWorld), and identifies concrete process-level failure modes (subgoal-irrelevant actions, click-precision errors).
- **Why it might matter:** Gives a concrete, reusable methodology — subgoal-level, human-annotation-grounded, LLM-judge-scored process evaluation — for the evaluation module, as a documented alternative to outcome-only agent benchmarks; from a major lab (NVIDIA) with substantial human-annotation investment, not just a leaderboard entry.
- **Evidence of adoption:** None yet beyond the paper itself; no third-party reuse of the benchmark confirmed.
- **Major organizations using it:** NVIDIA (authoring org only, so far).
- **Open-source implementation:** Not confirmed/linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2609.24890
- **Code:** unknown — not found on the abstract page
- **Relationship to existing course material:** New — no existing process-level CUA-evaluation topic; closest existing material is the SWE-agent reference in `papers/index.md` (a different task type, code-agent eval, not GUI/process-level).
- **Potential course lesson:** Possible addition to the evaluation module's agent-evaluation material (process-level vs. outcome-only agent scoring), contingent on independent adoption.
- **Confidence:** medium (NVIDIA-authored, large human-annotation effort, concrete methodology) but single paper with no external adoption yet.
- **Recommendation:** monitor — revisit if the benchmark is adopted by other labs' agent evaluations or cited in a major system/model card.

### C-20261007-02 · Self-generated feedback destabilizes test-time training (complicates the staged ASCENT/TTT line)

- **Class:** B
- **Date discovered:** 2026-10-07
- **Date published:** 2026-10-04
- **Source:** https://arxiv.org/abs/2610.05076 (verified resolves; v1 "Sun, 4 Oct 2026")
- **Organization/researchers:** Cheng Luo, Bing Li, Bernard Ghanem (Ghanem is a KAUST professor; affiliation not restated on the abstract page itself but is well known)
- **Category:** RL/post-training (test-time training), directly adjacent to the agents/tools category of the staged ASCENT candidate.
- **What changed:** Shows a general instability in test-time training (TTT) — updating model weights at inference time on the model's own generated text: retaining those self-generated updates measurably worsens prediction on independent, human-written text, across 125M/760M/3B-parameter models over 128K-token streams. Proposes "Fixed Generation" (using a frozen copy of the model to generate the training examples used for the weight update, rather than the actively-updating model) as a fix, removing over 98% of the measured degradation while preserving the benefit of adapting to real text.
- **Technical summary:** Measures an "endpoint gap" in nats between TTT-updated and non-updated predictions on held-out human text; reports this gap is large and growing under naive self-generated-feedback TTT, and reduced to near zero (0.07 and -0.02 nats at 125M/760M) with Fixed Generation, while real-text adaptation benefits are retained.
- **Why it might matter:** Directly relevant to the agent test-time-training line the course is already tracking via the staged ASCENT candidate (`C-20261006-01`, online TTT for long-horizon agents via self-distillation on the agent's own verified trajectories). ASCENT's design choice to strip invalid-action turns before distillation (because "directly imitating/reinforcing the raw single-attempt trajectory is reported to destabilize the policy") is the same family of self-generated-feedback instability this paper studies and names explicitly, with a general mechanism and a general fix (Fixed Generation) rather than ASCENT's task-specific workaround (filtering invalid turns).
- **Evidence of adoption:** None — single paper, no independent reproduction, no adoption found.
- **Major organizations using it:** None identified (academic/KAUST).
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.05076
- **Code:** none found
- **Relationship to existing course material:** New general finding; relates to the still-staged ASCENT candidate (`C-20261006-01`) and to `research/accepted/on-policy-distillation.md` (lesson 17.4). Not yet in any accepted/deferred/rejected file (checked via `lookup "test-time training"`).
- **Potential course lesson:** If a future lesson covers test-time training for agents (building on ASCENT or similar), this instability + the Fixed Generation fix is exactly the kind of failure-mode-and-fix pairing the course's templates ask for; on its own (without ASCENT or a similar motivating case) it's too narrow for a standalone lesson.
- **Confidence:** low — single paper, no independent reproduction, though the mechanism-level explanation plausibly generalizes the narrower instability ASCENT already works around.
- **Recommendation:** monitor alongside `C-20261006-01`; if the weekly review looks at ASCENT or any other agent-TTT candidate, flag this paper as the general mechanism behind the instability those papers work around.

### C-20261008-01 · NP-OPD: negative-policy rollouts for on-policy distillation (NAVER AI Lab)

- **Class:** B
- **Date discovered:** 2026-10-08
- **Date published:** 2026-10-06
- **Source:** https://arxiv.org/abs/2610.07874 (verified resolves; v1 "Tue, 6 Oct 2026 07:25:03 UTC")
- **Organization/researchers:** Jaehui Hwang, Dongyoon Han, Sangdoo Yun, Byeongho Heo — NAVER AI Lab (industrial AI lab, not pure academic)
- **Category:** RL/post-training (on-policy distillation, taught in lesson 17.4)
- **What changed:** On-policy distillation (OPD) trains a student on its own rollouts using token-level teacher feedback; the authors argue that when teacher and student distributions have little overlap, teacher guidance alone gives a weak signal. Negative-Policy OPD (NP-OPD) adds rollouts from a weaker "negative policy" during generation (without changing the distillation reward), giving the student tokens the negative policy prefers over the teacher as an explicit negative signal while keeping teacher supervision.
- **Technical summary:** Reports gains across model scales, generation modes, reasoning domains, and existing OPD variants; shows the method suppresses negative-policy-preferred tokens and moves the student away from the negative policy. Code promised at github.com/naver-ai/np-opd (not yet confirmed live).
- **Why it might matter:** Lesson 17.4 already teaches on-policy distillation, and the registry already carries a deferred topic (`on-policy-distillation-variants.md`, WAIT, next review 2026-11-29) tracking single-paper OPD variants (co-evolving teacher DCE/SRCL; same-family scaling). NP-OPD is a third independent single-paper OPD variant, this time from a named industrial lab rather than only academic groups, adding to the body of evidence the weekly review should weigh when that deferred topic comes up for its next review.
- **Evidence of adoption:** None yet — single paper, code not yet confirmed live, no third-party use found.
- **Major organizations using it:** NAVER AI Lab (authoring org only).
- **Open-source implementation:** Promised at github.com/naver-ai/np-opd; not yet confirmed live.
- **Paper:** https://arxiv.org/abs/2610.07874
- **Code:** github.com/naver-ai/np-opd (promised, unconfirmed)
- **Relationship to existing course material:** Extends `lessons/module-17/lesson-04.md` (on-policy distillation) and relates directly to the deferred registry topic `research/deferred/on-policy-distillation-variants.md` (a third OPD-variant data point, from an industrial lab this time).
- **Potential course lesson:** Not a standalone lesson; a candidate short addition/example to 17.4 if the weekly review decides the OPD-variants deferred topic has enough evidence, or simply more evidence to fold into that topic's file.
- **Confidence:** low-medium — single paper, no reproduction, but from a named industrial lab and directly extends already-taught material.
- **Recommendation:** fold into the existing deferred topic `on-policy-distillation-variants.md` at its next review (2026-11-29) as a third variant; not yet enough on its own for ADD.

### C-20261008-02 · STEPQuant: spatial-temporal quantization of delta-rule recurrent states, integrated into SGLang

- **Class:** B
- **Date discovered:** 2026-10-08
- **Date published:** 2026-09-29
- **Source:** https://arxiv.org/abs/2609.38169 (verified resolves; v1 "Tue, 29 Sep 2026 17:59:40 UTC")
- **Organization/researchers:** Bingchen Yao, Haobo Xu, Haokun Lin, Yichen Wu, Ziyu Guo, Renrui Zhang, Zhichao Lu, Zhenan Sun, Ying Wei (affiliations not listed on the abstract page; academic group)
- **Category:** efficiency/quantization (linear attention / Gated-DeltaNet-style recurrent state serving)
- **What changed:** Linear-attention models replace growing KV caches with a fixed-size recurrent state, but under concurrent serving that persistent state itself becomes a memory bottleneck; naively quantizing it degrades accuracy because errors compound across state updates. STEPQuant is a post-training quantization framework for delta-rule recurrent states that allocates precision by *when* an error occurs (how long it persists in the state) and *where* (which key rows/value columns matter most), jointly fitting per-row/column scales from state distributions and each key row's measured impact on output error.
- **Technical summary:** Evaluated on Qwen3.8-27B and Kimi-Linear-48B-A3B-Instruct (real, named production-scale models) on long- and short-generation benchmarks; 6-bit STEPQuant closely matches FP32-state accuracy and outperforms uniform INT8 at 4-bit. Integrated into SGLang with custom GPU kernels: 6-bit STEPQuant gives over 5x recurrent-state compression and cuts total serving memory by up to 68.7%. Code released (per abstract).
- **Why it might matter:** Directly extends the course's existing linear-attention/Gated-DeltaNet material (`lessons/module-12/lesson-05.md`) into the serving/efficiency domain this course also teaches (quantization, serving infrastructure), with a concrete mechanism (time- and space-aware error allocation) and a real systems integration (SGLang, an inference stack the course already references) rather than only an isolated benchmark number.
- **Evidence of adoption:** None beyond the authors' own SGLang integration; no confirmation this landed in mainline SGLang or was adopted by another group.
- **Major organizations using it:** None identified (academic authors); SGLang integration is the authors' own patch, not confirmed merged upstream.
- **Open-source implementation:** Reported as released (per abstract); exact repo URL not shown on the abstract page.
- **Paper:** https://arxiv.org/abs/2609.38169
- **Code:** reported released, URL not confirmed from abstract page
- **Relationship to existing course material:** Extends `lessons/module-12/lesson-05.md` (linear attention, Gated DeltaNet & hybrid stacks) into recurrent-state quantization for serving — new sub-topic, not currently in registry (checked via `lookup "delta rule quantization"`).
- **Potential course lesson:** Possible addition/example to module-12/lesson-05 or a serving/quantization lesson, covering recurrent-state (as opposed to KV-cache) quantization — contingent on independent confirmation the SGLang integration is real/merged and on a second source.
- **Confidence:** medium — concrete real-model evaluation and a named inference stack, but single paper, no independent reproduction, and the "integrated into SGLang" claim is not yet verified against SGLang's own release notes (checked SGLang v0.5.19-21 release notes this run; STEPQuant not mentioned).
- **Recommendation:** monitor; verify against SGLang's upstream repo/release notes before any course addition, and look for a second source or confirmed merge.

### C-20261008-03 · ReSAIL: a third independent line on instability in iterative/online agent self-training

- **Class:** B
- **Date discovered:** 2026-10-08
- **Date published:** 2026-09-30
- **Source:** https://arxiv.org/abs/2609.39306 (verified resolves; v1 "Wed, 30 Sep 2026 08:49:03 UTC")
- **Organization/researchers:** Shengjie Jin, Hengbo Xu, Zelong Sun, YuJie Guo, Zhiwu Lu (affiliations not listed on the page; academic group, Zhiwu Lu is a known Renmin University researcher)
- **Category:** agents/tools, adjacent to RL/post-training (self-distillation stability)
- **What changed:** Iterative self-distillation lets agents learn across successive deployments using privileged information (PI), but existing methods collapse in deployment performance over cycles. ReSAIL is a plug-in augmentation that selects the interaction steps where PI most changes the teacher's predictions, balances distillation losses across trajectories, and regularizes the student's PI-conditioned outputs toward a frozen teacher.
- **Technical summary:** On ALFWorld and TextCraft, ReSAIL sustains gains over three cycles across model scales, reporting an average absolute gain of 22.5% in final-cycle success rate over self-distillation baselines; sensitivity-guided data selection also improves action prediction for multimodal GUI agents on AITZ.
- **Why it might matter:** This is now the *third* independent paper within roughly a week touching the same underlying phenomenon the course is already tracking in staging — self-referential/iterative training signals (an agent's own trajectories, or a model's own generated text) destabilizing training without an explicit correction — alongside the already-staged ASCENT (`C-20261006-01`, online TTT via self-distillation on verified trajectories) and the Fixed-Generation TTT-instability paper (`C-20261007-02`). Each proposes a different specific fix (ASCENT: filter invalid turns; Fixed Generation: freeze the generator; ReSAIL: PI-sensitivity-weighted step selection + loss balancing + teacher regularization), so this is convergent-but-not-identical evidence rather than reproduction of one paper.
- **Evidence of adoption:** None — single paper, no independent reproduction, no adoption found.
- **Major organizations using it:** None identified (academic).
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2609.39306
- **Code:** none found
- **Relationship to existing course material:** Related to the still-staged `C-20261006-01` (ASCENT) and `C-20261007-02` (Fixed Generation); same general phenomenon (iterative/self-referential training instability), distinct specific mechanism/fix. Not in any accepted/deferred/rejected file.
- **Potential course lesson:** Not on its own; if the weekly review is already looking at ASCENT or the TTT-instability thread, this is a third data point worth weighing together with the other two — possibly enough convergent evidence, across three independent groups in one week, to justify at least a frontier-update note on "iterative/online self-distillation instability and mitigations" even though no single paper here clears the bar alone.
- **Confidence:** low-medium — plausible mechanism, real (if small-scale) benchmarks, but still single-paper per specific fix.
- **Recommendation:** weekly review should look at `C-20261006-01`, `C-20261007-02`, and this record together as one thread, since three independent groups converging on the same destabilization pattern in a week is itself notable evidence even though each individual fix is single-paper.

### C-20261008-04 · Sliding-Window Linear Attention: a "Seesaw Effect" between linear-attention and sliding-window hybrids on long context

- **Class:** B
- **Date discovered:** 2026-10-08
- **Date published:** 2026-10-07
- **Source:** https://arxiv.org/abs/2610.10114 (verified resolves; v1 "Wed, 7 Oct 2026 14:02:35 UTC")
- **Organization/researchers:** Xiaoran Liu, Ziwei He, Xipeng Qiu — Fudan University (Xipeng Qiu is an established NLP/ML-systems researcher); paper is labeled "Part 1.1" of a series ("Mechanics of Long-Context Hybrid Models")
- **Category:** architecture / long-context / attention (hybrid full+linear/sliding-window attention)
- **What changed:** Systematically studies hybrids that pair full attention with either sliding-window attention (SWA) or gated linear attention (GLA/Gated DeltaNet), reporting a "Seesaw Effect": linear-attention hybrids benefit more from long-context continual pretraining, while SWA hybrids extrapolate better to unseen lengths, attributed to differing positional inductive biases. Also documents SWA failure modes and a "Matthew Effect" in linear-attention hybrids' position extrapolation, and proposes Sliding-Window Linear Attention (SWLA) as a fix.
- **Technical summary:** Reports SWLA achieves 16x training-free length extrapolation while keeping 100% accuracy on NIAH-SK1 at 64k context. 60 pages, 36 figures, 25 tables — a systematic empirical study, not a single benchmark number.
- **Why it might matter:** Directly extends the course's existing hybrid-attention material (`lessons/module-12/lesson-05.md`, linear attention/Gated DeltaNet/hybrid stacks) with a mechanism-level explanation of *why* hybrid designs trade off long-context training benefit against length-extrapolation robustness, plus a concrete proposed fix — exactly the kind of "what PyTorch/the architecture does under the hood, and why" material the course's lesson template asks for.
- **Evidence of adoption:** None yet — single paper, "under review," no third-party validation.
- **Major organizations using it:** None identified (academic, Fudan).
- **Open-source implementation:** Not confirmed from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.10114
- **Code:** not confirmed
- **Relationship to existing course material:** Extends `lessons/module-12/lesson-05.md` (linear attention, Gated DeltaNet & hybrid stacks) with new analysis of position-extrapolation trade-offs; new sub-topic in the registry (checked via `lookup "hybrid position long context"`, which only matched the existing lesson, not a registry topic).
- **Potential course lesson:** Possible addition to module-12/lesson-05 explaining the Seesaw/Matthew Effects and SWLA, contingent on a second source or the paper clearing peer review ("under review" per the abstract page).
- **Confidence:** medium — large, systematic empirical study from an established researcher, but single paper, still under review, no code confirmed.
- **Recommendation:** monitor; revisit once the paper is no longer "under review" or a second group reproduces the Seesaw/Matthew Effect framing.

### C-20261009-01 · NVIDIA Dynamo: session-aware agentic inference serving, integrated into vLLM and SGLang

- **Class:** A
- **Date discovered:** 2026-10-09
- **Date published:** 2026-10-08
- **Source:** https://pytorch.org/blog/session-aware-agentic-inference-with-nvidia-dynamo/ (verified loads; official PyTorch Foundation blog, dated 2026-10-08)
- **Organization/researchers:** NVIDIA Dynamo team, cross-posted on the official PyTorch blog (also references joint work with the Mooncake team and SGLang's HiCache)
- **Category:** serving / distributed (agentic-inference serving infrastructure)
- **What changed:** Agentic workloads (large initial prefill, many repeated model calls, parallel subagents, long idle gaps between turns while tools run, KV cache held resident throughout) break request-level serving assumptions. Dynamo introduces a unified `session_id`/`parent_session_id` primitive (auto-recognized from Claude Code, Codex and OpenCode headers, or set by any custom harness via one header) that turns request-aware serving infrastructure into session/program-aware infrastructure: session-linked tracing and replay (content-free, via sequence hashes, compatible with Mooncake-style trace formats), session-aware admission control (a Rust scheduler porting the ThunderAgent (Kang et al., 2026) design, pausing/resuming agent "programs" at tool-call boundaries via a KV-utilization control loop with pause/resume/soft-demote thresholds), a shared-pool KV indexer that extends routing visibility into external stores (Mooncake) via events from SGLang's HiCache, and a proposed narrow "KvHint" interface (Share/Prefetch/Demote/Pin/Retain) by which the router expresses cache-movement intent to vLLM/SGLang engines that execute it.
- **Technical summary:** Reports 12-16% higher throughput than KV-aware routing alone on SWE-bench (two TP4 MiniMax-M2 replicas, 8xH100); on agentic RL rollouts (Uni-Agent SWE-Bench, Qwen3-Coder-30B-A3B-Instruct, 8xH20-3e, vs. VERL's default Global LB) it is roughly even at low concurrency but 11.0-14.6% higher model-token throughput at medium concurrency and keeps scaling at high concurrency where the baseline's throughput drops sharply, while holding prefix-cache hit rate above 94.5%. The "Share" KvHint already has a working implementation in SGLang's HiCache; Prefetch/Demote are described as in progress. Code: Dynamo (github.com/ai-dynamo/dynamo), the ThunderAgent-derived scheduler plugin, and harness-adapter plugins (github.com/ai-dynamo/agent-plugins) are open source; the KV-hint designs are tracked as open issues against both SGLang (sgl-project/sglang#27574) and vLLM (vllm-project/vllm#51428).
- **Why it might matter:** This is production serving infrastructure for the agentic/tool-use workloads this course's agents module (18/19) and distributed/serving material already care about, with a concrete, teachable mechanism (session identity propagation, a utilization control loop with named thresholds, a router/engine split of policy-vs-execution for KV movement) and real integration into two serving stacks the course already references (vLLM, SGLang) rather than an isolated benchmark number.
- **Evidence of adoption:** Beyond NVIDIA's own numbers: the session-ID convention is natively recognized by three widely used coding-agent harnesses (Claude Code, Codex, OpenCode) without configuration, and the "Share" KvHint is already implemented against SGLang's native HiCache offload mechanism; the KV-hint interface is an open, named design proposal against both vLLM's and SGLang's own repos (not yet merged in either).
- **Major organizations using it:** NVIDIA (Dynamo); harness-level support already live for Anthropic's Claude Code, OpenAI's Codex, and OpenCode (header recognition only, not a claim those orgs use Dynamo internally).
- **Open-source implementation:** https://github.com/ai-dynamo/dynamo ; https://github.com/ai-dynamo/agent-plugins ; https://github.com/ai-dynamo/aisimulate
- **Paper:** none (engineering blog post; cites academic ThunderAgent (Kang et al., 2026) for the scheduler design, paper link not given on the page)
- **Code:** https://github.com/ai-dynamo/dynamo
- **Relationship to existing course material:** New sub-topic — no existing lesson covers production serving-stack internals (session-aware routing, KV-cache tiering/hints) at this level; closest existing material is the KV-cache concepts in `lessons/module-12/lesson-02.md`/`lesson-03.md` (MQA/GQA, cache size) and tool-use/agents in module 18/19, neither of which covers serving infrastructure itself. Checked via `lookup "session aware inference"`, `"KV cache routing agentic"`, `"Dynamo serving"` — no matches.
- **Potential course lesson:** Candidate addition to the distributed/serving material (module 9) or the agents module (18/19): a worked example of session-aware routing/admission control for agentic serving, contingent on the weekly review deciding where serving-infrastructure depth belongs in the module map (currently thin — no dedicated serving lesson).
- **Confidence:** medium-high — official NVIDIA/PyTorch engineering blog (not a marketing page) with a detailed, falsifiable mechanism description and named benchmarks/configs, real open-source code, and real (if partial) adoption by harness ecosystems; still a single primary source and NVIDIA's own benchmarks, no independent third-party measurement.
- **Recommendation:** review for ADD or at minimum a frontier-update note — concrete mechanism, real code, real stack integration (vLLM + SGLang), and it fills a gap (no existing serving-infrastructure lesson) rather than only extending one.

### C-20261009-02 · TRL v1.15.0: fused LM head (Triton) makes "never materialize full logits" the default across SFT/DPO/KTO/GRPO/RLOO/Distillation

- **Class:** A
- **Date discovered:** 2026-10-09
- **Date published:** 2026-10-08
- **Source:** https://github.com/huggingface/trl/releases/tag/v1.15.0 (verified loads; published 2026-10-08 by TRL maintainer qgallouedec)
- **Organization/researchers:** Hugging Face (TRL maintainers)
- **Category:** efficiency (training-time memory), directly extending RL/post-training material
- **What changed:** TRL's SFT, DPO, KTO, GRPO, RLOO and Distillation trainers now compute log-probs/entropy through a fused Triton kernel that never builds the full `(B, T, V)` logits tensor — on by default, "nothing to turn on." The release also adds selective activation checkpointing for SFT (saves the attention output instead of recomputing it, cutting the long-context checkpointing slowdown) and assistant-only loss support for vision datasets in SFT.
- **Technical summary:** Reports up to 6.9x longer trainable sequences on the same GPU from the fused LM head alone. This is the same memory problem — a fp32 `(B,T,V)` logits tensor at Qwen-size vocabulary and realistic batch/seq-length is tens of GB — that this course's own lesson on on-policy distillation already names explicitly and already describes TRL's existing chunking behavior for (see Relationship, below); this release moves that from "chunked loss computation" to a fused kernel that avoids materializing the full logits tensor at all, across six trainers instead of one.
- **Why it might matter:** This is adoption evidence, in a library this course already names and cites by name in lesson material, for exactly the mechanism that lesson already teaches ("the same trick as fused or chunked cross-entropy in pretraining") — a concrete, dated, verifiable update to already-taught production behavior, not a new unrelated technique.
- **Evidence of adoption:** Default behavior ("on by default") across six trainers in TRL, a widely used post-training library; this is the standard-practice bar the protocol asks for, not merely a company claim — the release notes describe a concrete kernel change with a specific reported memory/sequence-length benefit, not a marketing claim.
- **Major organizations using it:** Hugging Face (library maintainer); TRL is used widely across the open post-training ecosystem (exact downstream adopters of this specific release not individually confirmed).
- **Open-source implementation:** https://github.com/huggingface/trl (v1.15.0)
- **Paper:** none (library release notes)
- **Code:** https://github.com/huggingface/trl/releases/tag/v1.15.0
- **Relationship to existing course material:** Directly updates `lessons/module-17/lesson-04.md` (on-policy distillation), which already states: "Chunking the vocabulary projection and the divergence over positions keeps peak memory at one chunk. That is the same trick as fused or chunked cross-entropy in pretraining" and cites "TRL's loss, defaults and chunking" as publicly documented. This release changes exactly that documented default (chunked -> fused, never-materialized) and extends it to five more trainers (SFT/DPO/KTO/GRPO/RLOO) beyond distillation. Checked via `lookup "fused linear cross entropy logits"` / `"chunked cross entropy"` / `"liger kernel"`.
- **Potential course lesson:** Not a new lesson; an update to the existing "why chunking is needed" passage in module-17/lesson-04 (and possibly a cross-reference from module-08's fused-kernel material) to reflect the newer fused-kernel-by-default behavior, with the memory-savings number (6.9x longer sequences) as the new concrete figure.
- **Confidence:** high — primary-source release notes from the library maintainer, a specific library the course already cites, a concrete reported number, and it corrects/updates already-taught material rather than introducing a new unvalidated claim.
- **Recommendation:** review for ADD as a direct update to `lessons/module-17/lesson-04.md`'s chunking passage — this is the kind of small, well-evidenced "keep the lesson accurate" change the modernization pipeline exists for.

### C-20261009-04 · TokenRouter: a serving system for token-level LLM routing

- **Class:** B
- **Date discovered:** 2026-10-09
- **Date published:** 2026-10-08
- **Source:** https://arxiv.org/abs/2610.12242 (verified resolves; v1 "Thu, 8 Oct 2026 16:21:50 UTC"; accepted to NeurIPS 2026 per the abstract page)
- **Organization/researchers:** Tianyu Fu, Tengxuan Liu, Ruoxi Wang, Yixin Dong, Yi Ge, Yichen You, Yu Wang (affiliations not stated on the abstract page; academic group)
- **Category:** serving (adjacent to distributed/efficiency)
- **What changed:** Token-level LLM routing (routing each generated token, not each session/query, across multiple models for a better cost-quality trade-off) has shown research-level gains, but existing serving systems assume a single LLM, so token-level routing causes step desynchronization and batch-admission delays in practice. TokenRouter uses a "request-centric programming, model-centric execution" design: developers write routing logic from a single request's point of view, while the runtime starts one subserver per model and dispatches asynchronously, with a delayed-batching scheduler whose hyperparameters come from an explicit throughput model.
- **Technical summary:** Reports 2.01-64.15x higher decoding throughput than existing systems across multiple routing algorithms, workloads and model pairs, accepted to NeurIPS 2026. Code is publicly available (per the abstract page).
- **Why it might matter:** Directly relevant to the serving-infrastructure gap this course currently has (see also `C-20261009-01`), and a concrete, peer-reviewed systems contribution (not just a benchmark number) for a known research problem (token-level routing) that the course's efficiency/serving material could use as a worked example of request-vs-model-centric serving design.
- **Evidence of adoption:** NeurIPS 2026 acceptance (peer review) is itself a form of validation beyond a single group's say-so; no third-party production adoption found yet.
- **Major organizations using it:** None identified (academic authors).
- **Open-source implementation:** Reported public on GitHub per the abstract page; exact repo URL not shown on the abstract page itself.
- **Paper:** https://arxiv.org/abs/2610.12242
- **Code:** reported public, URL not confirmed from the abstract page
- **Relationship to existing course material:** New sub-topic — no existing lesson covers serving-system design for multi-model routing; related to the serving-infrastructure gap noted in `C-20261009-01`. Checked via `lookup "token-level routing serving"` — no match.
- **Potential course lesson:** Possible worked example within a future serving-infrastructure lesson (alongside `C-20261009-01`), contingent on the weekly review's decision about where serving depth belongs in the module map.
- **Confidence:** medium — peer-reviewed (NeurIPS 2026), concrete and wide-ranging throughput numbers, public code, but single paper/group and no production adoption yet.
- **Recommendation:** monitor alongside `C-20261009-01` as a second serving-infrastructure data point; revisit together if the weekly review decides to add serving-infrastructure material.

### C-20261009-05 · NVIDIA Megatron-Core: bitwise-deterministic pretraining at trillion-parameter scale

- **Class:** B
- **Date discovered:** 2026-10-09
- **Date published:** 2026-10-06
- **Source:** https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/ (verified loads; official NVIDIA technical blog, dated 2026-10-06)
- **Organization/researchers:** NVIDIA (Megatron-Core team)
- **Category:** distributed training (reproducibility/debuggability at scale)
- **What changed:** Describes how to make large pretraining runs bitwise-deterministic (byte-for-byte reproducible across reruns) so teams can replay failures, validate system changes, and resume interrupted trillion-parameter runs without altering the training trajectory — via ordered per-rank tensor fingerprinting (`torch.hash_tensor`) to localize the first mismatching step/rank/operation, progressive tracing from end-to-end metrics down to individual kernels, and specific kernel fixes (e.g. giving each N-tile in a grouped-GEMM epilogue a private output slot combined in fixed order).
- **Technical summary:** Demonstrates bitwise determinism over 800 steps at 2,432 GPUs on a trillion-parameter Nemotron hybrid (Mamba+attention) model, with a measured "determinism tax" reduced from roughly 17% to 1.5% overhead at 3,072 GPUs for one recipe (other configurations: ~60%->2% for a hybrid Triton proxy path; ~3.6% for a CuTeDSL weight-gradient path at 256 GPUs). The tracing/fingerprinting workflow is proposed as Megatron-LM PR #7262 (not confirmed merged); a `--deterministic-mode` flag and determinism tests are described as already-current Megatron-LM features.
- **Why it might matter:** Directly extends the course's pretraining-infrastructure material (module 9, distributed training) with a concrete, named engineering technique (fingerprint-and-trace debugging methodology) and real numbers at real trillion-parameter/thousands-of-GPU scale, from the team that maintains Megatron-Core, a framework already relevant to this course's distributed-training module.
- **Evidence of adoption:** Used internally by NVIDIA at the scale described; the specific tracing workflow is an open PR against Megatron-LM, not yet confirmed merged upstream.
- **Major organizations using it:** NVIDIA (internally, at trillion-parameter scale, per the post).
- **Open-source implementation:** Megatron-LM PR #7262 (open, not confirmed merged); `--deterministic-mode` and determinism tests described as already in Megatron-LM nightly.
- **Paper:** none (engineering blog post)
- **Code:** Megatron-LM (github.com/NVIDIA/Megatron-LM), PR #7262
- **Relationship to existing course material:** New sub-topic for module 9 (distributed training) — the existing checkpointing lesson (`lessons/module-07/lesson-03.md`) covers exact-resume via saved state but not bitwise-deterministic reproduction across reruns at scale. Checked via `lookup "deterministic training reproducibility"` — no direct topic match.
- **Potential course lesson:** Possible addition to module 9's distributed-training material: a debugging-methodology sidebar on fingerprint-and-trace determinism checking, with the reported overhead numbers as a concrete cost/benefit example.
- **Confidence:** medium — official NVIDIA engineering blog with real scale and concrete overhead numbers, but single source, and the headline tracing workflow is an unmerged PR rather than a shipped feature.
- **Recommendation:** monitor; revisit once PR #7262 merges or a second lab/framework (e.g. a DeepSpeed or a major lab's training report) describes a comparable determinism effort.

### C-20261010-01 · On-policy distillation teaches new skills but not new knowledge

- **Class:** B
- **Date discovered:** 2026-10-10
- **Date published:** 2026-10-07
- **Source:** https://arxiv.org/abs/2610.09639 (verified resolves; v1 "Wed, 7 Oct 2026 08:15:10 UTC")
- **Organization/researchers:** Yixuan Tang, Yi Yang (affiliations not stated on the abstract page; appear to be an academic group)
- **Category:** RL/post-training (on-policy distillation, taught in lesson 17.4)
- **What changed:** Asks whether on-policy distillation (OPD) gives a student model new *factual knowledge* or only new *compositional reasoning skill*. Using a controlled synthetic framework across four models from three families, plus factual-QA and competition-math experiments, the authors report that reverse-KL OPD (the form the course already teaches) transfers compositional skill to unseen reasoning structures but transfers little factual knowledge; switching to forward KL restores factual transfer. Student rollouts mainly improve *how* the model executes multi-step reasoning over what it already knows, not what it knows.
- **Technical summary:** Controlled synthetic framework isolates knowledge-transfer from skill-transfer by construction (so the finding isn't just "model got better on a benchmark"); the forward-KL vs. reverse-KL contrast is directly checkable against the forward/reverse-KL distinction lesson 17.4 already teaches as the core design choice in logit distillation. No code repository found linked from the abstract page.
- **Why it might matter:** This is a precise, falsifiable characterization of *what on-policy distillation actually does* — directly testing the mechanism lesson 17.4 already teaches (reverse-KL OPD on a model's own rollouts) rather than proposing a new variant. If it holds up, it's exactly the kind of "what this technique does and doesn't do" caveat the lesson's existing forward/reverse-KL discussion should carry, and it's a different, complementary angle from the already-deferred OPD-variants thread (`research/deferred/on-policy-distillation-variants.md`, which tracks new OPD *mechanisms*, not characterizations of the existing one).
- **Evidence of adoption:** None — single paper (2 authors), no independent reproduction, no code found.
- **Major organizations using it:** None identified (academic).
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.09639
- **Code:** none found
- **Relationship to existing course material:** Directly extends `lessons/module-17/lesson-04.md` (on-policy distillation / forward vs. reverse KL) with a characterization of what the taught mechanism does and doesn't transfer; distinct from the deferred `on-policy-distillation-variants.md` topic (new mechanisms, not characterization of the base one). Checked via `lookup "on-policy distillation knowledge skills"` — no match.
- **Potential course lesson:** Not a new lesson; a candidate caveat/addition to 17.4's forward-vs-reverse-KL discussion ("reverse KL teaches skill, not knowledge; forward KL recovers knowledge transfer") if the weekly review judges a single controlled study sufficient to state as a caveat rather than a hedge.
- **Confidence:** low-medium — single paper, no reproduction, but methodologically careful (controlled synthetic isolation plus two real-task confirmations across model families) and directly testable against the course's own existing framing.
- **Recommendation:** monitor; if the weekly review wants to tighten 17.4's KL-direction discussion, this is the citation, but hold for a second source or author code before treating the "reverse KL = skill only" claim as settled.

### C-20261010-02 · REMORY: learned residual memory tokens for agent context compaction

- **Class:** B
- **Date discovered:** 2026-10-10
- **Date published:** 2026-10-08
- **Source:** https://arxiv.org/abs/2610.11287 (verified resolves; v1 "Thu, 8 Oct 2026 05:50:50 UTC")
- **Organization/researchers:** Hanchen Xia, Baoyou Chen, Yutang Ge, Naihao Deng, Senqiao Yang, Zilong Dong, Weihao Yuan, Siyu Zhu (affiliations not stated on the abstract page; academic/industrial group)
- **Category:** agents/tools, adjacent to memory/retrieval (context compaction for long-horizon agents)
- **What changed:** Long-horizon agents compact history into a text summary to fit the context window, but a summary alone can't support every later decision. REMORY appends a bounded sequence of *learned soft memory tokens* after the summary — generated by a small neural memory network, conditioned on the summary — so a frozen LLM's continuation approximates what it would have produced with the full history. This is a fourth, independent mechanism in the already-growing cluster of context-compaction approaches the registry is tracking (text-graph importance scoring, RL-trained compaction policy, belief-state tracking, and now learned soft-token residual memory).
- **Technical summary:** On SummHay, improves source attribution with nearly unchanged insight coverage, approaching the full-context joint score using only 5.2% of input positions. On long-horizon agent benchmarks (BrowseComp, Terminal-Bench 2.1), Qwen3.8-27B and GLM-5.3-Flash both improve with residual memory and show fewer repeated tool outputs/tool errors. No code repository found linked from the abstract page.
- **Why it might matter:** The deferred registry topic `agent-context-compaction.md` (WAIT, next review 2026-11-29) already tracks three single-paper mechanisms (ReCAP, AutoCompact, PoS) without convergence; REMORY is a fourth distinct mechanism (learned token-level residual memory rather than text/graph/belief-state) with concrete numbers on a named benchmark (SummHay) and two named real models, adding to — but not yet resolving — that still-fragmented picture.
- **Evidence of adoption:** None — single paper, no independent reproduction, no code found.
- **Major organizations using it:** None identified.
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.11287
- **Code:** none found
- **Relationship to existing course material:** New evidence for the deferred topic `research/deferred/agent-context-compaction.md` (a fourth independent mechanism; not yet enough convergence per that file's own "what would change the decision" bar). Checked via `lookup "context compaction memory agent"` — matched that deferred file and the still-staged ASCENT record, no exact-topic match otherwise.
- **Potential course lesson:** None on its own; flag for the weekly review alongside the three already-tracked candidates when `agent-context-compaction.md` next comes up (2026-11-29).
- **Confidence:** low — single paper, no reproduction, no code; real benchmark numbers but a fourth incompatible mechanism rather than convergence.
- **Recommendation:** add as a fourth data point to the existing deferred topic at its next scheduled review; not independently actionable before then.

### C-20261010-03 · U-Space: training-free, no-repeated-generation token-level uncertainty via a learned subspace

- **Class:** B
- **Date discovered:** 2026-10-10
- **Date published:** 2026-10-06
- **Source:** https://arxiv.org/abs/2610.09087 (verified resolves; v1 "Tue, 6 Oct 2026 20:40:29 UTC")
- **Organization/researchers:** Tobias Braun, Nils Loose, Alexander Herzog, Virginia Ceccatelli, Marcus Rohrbach, Thomas Eisenbarth, Lorenzo Cavallaro (affiliations not stated on the abstract page; academic group)
- **Category:** interpretability (uncertainty estimation / internal-state probing)
- **What changed:** Most LLM uncertainty/confidence methods need repeated sampling (self-consistency-style) or a separately trained estimator, and collapse evidence to one scalar with no view of *where* in a reasoning trace uncertainty arose. U-Space derives "doubt" and "certainty" directions from a model's internal activations, combines them into an orthogonal basis (a low-dimensional subspace), and projects each token's hidden state onto it — producing a token-level uncertainty map that can be read directly or aggregated into a scalar, with no correctness labels, no repeated generations, and no training required.
- **Technical summary:** Reports the resulting confidence score beats established baselines (self-consistency-style and supervised estimators, per the abstract) on reasoning benchmarks including length-controlled evaluation, and transfers more reliably across settings than supervised estimators. Code released and verified live at https://github.com/s2labres/U-Space, whose README states it reproduces the paper's Tables 1-2 (U-Lens vs. eight single-pass baselines: generation length, MSP, predictive entropy, max entropy, mean NLL, self-certainty, DeepConf, and an "A_cone" ablation) across three named models (Gemma-4-31B-it, Qwen3.5-27B, Magistral-Small-2507) and four benchmarks (MMLU-Pro, Omni-MATH, SuperGPQA, TriviaQA) with a runnable smoke-test config.
- **Why it might matter:** A concrete, teachable mechanism-level technique (derive interpretable directions in activation space, project onto them for a free, token-level, no-training diagnostic) squarely in the "from-scratch implementation, numerical example, what PyTorch does under the hood" register this course's lesson template asks for, with real code reproducing a real table against several real open models — this course currently has no lesson on activation-space uncertainty/confidence estimation.
- **Evidence of adoption:** None beyond the authors' own released, verified-live code; no independent reproduction or production adoption found.
- **Major organizations using it:** None identified (academic).
- **Open-source implementation:** https://github.com/s2labres/U-Space (verified live; README confirms reproduction pipeline, model list, and a runnable smoke-test config)
- **Paper:** https://arxiv.org/abs/2610.09087
- **Code:** https://github.com/s2labres/U-Space
- **Relationship to existing course material:** New sub-topic — no existing lesson covers activation-space uncertainty/confidence estimation; closest existing material is the interpretability module's probing/circuit-discovery content (different question: what a representation encodes, not how confident the model is token-by-token). Checked via `lookup "uncertainty quantification subspace"` — no match.
- **Potential course lesson:** Possible addition to the interpretability module: a from-scratch worked example of deriving doubt/certainty directions and projecting hidden states onto them, contrasted against self-consistency and predictive-entropy baselines the course may already mention.
- **Confidence:** medium — single paper, no third-party reproduction, but a real, verified-live code release with a runnable reproduction pipeline against multiple named open models, which is stronger evidence than an abstract-only claim.
- **Recommendation:** monitor; worth a closer look at the next weekly review given the working code and the gap in the interpretability module, but hold for independent confirmation of the reported baseline comparisons before considering ADD.

### C-20261010-04 · Synthesis Through Simulation (STS): schema-free synthetic enterprise agent-training data from policy-enforcing simulated APIs (SAP)

- **Class:** B
- **Date discovered:** 2026-10-10
- **Date published:** 2026-09-24
- **Source:** https://arxiv.org/abs/2610.10549 (verified resolves; v1 "Thu, 24 Sep 2026 13:55:22 UTC")
- **Organization/researchers:** Yipeng Li, Ashutosh Hathidara, Jane Lo, Harshavardhan Abichandani, Gunraj Singh, Atin Ghosh — SAP (named industrial lab, not academic)
- **Category:** data (synthetic data generation / curation for agent training and evaluation)
- **What changed:** Training and evaluating tool-calling agents for enterprise use is hard because business/legal restrictions block access to real enterprise systems, data and schemas. Synthesis Through Simulation (STS) is schema-free: an LLM agent (the "Generalist Populator", GP) generates training/eval data by running operations against *policy-enforcing simulated APIs* rather than a database schema, so the data's structural validity is guaranteed by the same environment that defines validity, instead of being checked after the fact against a schema the agent never had access to.
- **Technical summary:** Reports 0.88 average marginal fidelity and 100% constraint satisfaction across ten simulated enterprise environments without any database-schema access for the generating agent; by contrast, schema-privileged agents (given direct schema access) reportedly failed 82% of trajectories in an airline environment with tightly coupled workflows — the paper's headline evidence that schema access can hurt rather than help when workflows are highly interdependent. The paper states the full framework, ten environments and generated datasets are open-sourced at github.com/SAP/synthesis-through-simulation, but that URL returned HTTP 404 as of this check (2026-10-10) — not confirmed live.
- **Why it might matter:** A concrete, named-company (SAP, not a pure research group) contribution to synthetic data generation for agent training in a genuinely restricted domain — directly relevant to the course's data-curation/synthetic-data material — with a counterintuitive, teachable finding (schema access can actively hurt agents on tightly-coupled workflows) rather than only a benchmark number.
- **Evidence of adoption:** None yet — single paper; the claimed open-source release is not independently confirmed (repo URL 404s as of this check).
- **Major organizations using it:** SAP (authoring org only, so far).
- **Open-source implementation:** Claimed at github.com/SAP/synthesis-through-simulation (per the paper text); **not confirmed live — returned 404 on this check.**
- **Paper:** https://arxiv.org/abs/2610.10549
- **Code:** claimed, not confirmed (see above)
- **Relationship to existing course material:** New sub-topic — no existing lesson covers schema-free, simulation-grounded synthetic data generation for enterprise/agent training. Checked via `lookup "synthetic data enterprise agent"` — no match.
- **Potential course lesson:** Possible addition/example to the data-curation module on simulation-grounded (vs. schema-validated) synthetic data generation, contingent on the code repo actually becoming live and on a second source or adoption signal.
- **Confidence:** low-medium — single paper from a named industrial lab with concrete numbers and a genuinely different mechanism, but the claimed open-source release does not currently resolve, which weakens the evidence until it's checked again.
- **Recommendation:** monitor; re-check the GitHub URL in a future run before treating "open-sourced" as confirmed, and look for a second source.
