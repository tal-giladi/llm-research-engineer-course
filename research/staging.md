# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261005-01 · Looped (recurrent-depth) language models: practical training recipes + a second independent line with code

- **Class:** A
- **Date discovered:** 2026-10-05
- **Date published:** 2026-09-30 (source 1); 2026-10-05 (source 2, added to this record 2026-10-07)
- **Source:** https://arxiv.org/abs/2610.00673 (source 1, verified resolves; v1 "Wed, 30 Sep 2026 20:09:58 UTC"); https://arxiv.org/abs/2610.06833 (source 2, "Towards Looped Models Done Right, Part II: Rethinking at Fixed Points", verified resolves; v1 "Mon, 5 Oct 2026")
- **Organization/researchers:** Source 1: Andrei Marchenko, Viacheslav Bezrukov, Oleg Kashurin, Inessa Fedorova, Dmitry Bocharov, Yuliana Shakhvalieva, Maria Tikhonova, Valerii Ternovskii (independent/academic group). Source 2: Benhao Huang, Chufan Shi, Junlin Chen, Shicheng Wen, Zhengzhong Liu, Eric Xing, Xuezhe Ma (academic, MBZUAI/USC lineage) — a different, independent group; explicitly "Part II" of a prior paper in the same line, and compares directly against Huginn (Geiping et al.'s known recurrent-depth model).
- **Category:** architecture
- **What changed:** Practical recipe for training "looped" LMs — architectures that increase effective depth by repeatedly applying a shared block of layers (recurrent-depth, in the lineage of Universal Transformer / ALBERT-style weight sharing and 2025's recurrent-depth latent-reasoning work) — claiming large reductions in the token budget needed to make the approach competitive, plus a lightweight recipe for converting an already-pretrained dense checkpoint into a looped variant. Added 2026-10-07: a second, independent group's paper shows that because looped-model recurrent states approach a fixed point, this enables truncated backprop in training, terminal KV-cache sharing at decode time, faster prefill via a distilled student, and cheaper RL updates (gradients from saved rollout states instead of backprop through the replayed trajectory) — then improves the depth prior and input-injection scheme that shape how tightly states converge to that fixed point.
- **Technical summary:** Source 1 combines pretraining with targeted mid-training, warmup schedules and regularization changes to cut the data needed from 7.7T to 310B tokens for a competitive LoopLM; reports a 1.4B-parameter LoopLM beating equivalently-sized dense models (+14 GSM8K, +10 MATH, +22 DROP, per the abstract) and matching a 3.9B dense model's quality at 36% of its parameters, plus a dense-to-looped conversion recipe using one learned input-mixing scalar and a smoothed exit loss. Source 2, evaluated 100M-1.6B parameters: learned depth prior + orthogonal injection lower perplexity at every scale vs. Huginn's prior/injection scheme; at 1.6B, the learned prior with a 3x smaller KV cache matches the downstream-task average of fixed-depth training with the full cache; reports 1.79x faster prefill (distilled student) and 2x faster RL gradient updates; code and checkpoints released at https://github.com/ifm-ai/xllm-loop.
- **Why it might matter:** If the dense-to-looped conversion and the reduced token budget hold up, this would be a cheap way to trade extra compute-per-token for fewer parameters/tokens — directly relevant to the course's architecture and efficient-training material (module on attention/architecture variants, alongside the existing Gated DeltaNet lesson). The second source adds concrete, mechanism-level inference/RL efficiency payoffs (KV-cache reduction, prefill speedup, cheaper RL) and a public code release, and engages directly with Huginn, an already-known recurrent-depth model — meaningfully strengthening the evidence base within the same week.
- **Evidence of adoption:** None yet beyond the two papers' own code (source 2 only); no independent reproduction, no major-lab or major-stack adoption found.
- **Major organizations using it:** None identified; academic groups only.
- **Open-source implementation:** Source 1: none found. Source 2: https://github.com/ifm-ai/xllm-loop.
- **Paper:** https://arxiv.org/abs/2610.00673 (source 1); https://arxiv.org/abs/2610.06833 (source 2)
- **Code:** https://github.com/ifm-ai/xllm-loop (source 2 only; none for source 1)
- **Relationship to existing course material:** New — the registry/lessons have no existing "looped / recurrent-depth / weight-shared-layer" topic (checked via `research/tools/research.py lookup`); closest existing material is module-12/lesson-05 on Gated DeltaNet/hybrid linear attention, which is a different axis (attention mechanism, not depth-sharing).
- **Potential course lesson:** Candidate module-12 addition (or frontier-update note) on recurrent-depth/looped transformers: architecture + training (depth priors, input injection) + the inference/RL efficiency mechanism (fixed-point KV sharing).
- **Confidence:** medium (raised from low on 2026-10-07) — two independent groups within one week, one with public code and concrete mechanism-level efficiency numbers, and direct engagement with an already-known model (Huginn); still no third-party reproduction or production adoption.
- **Recommendation:** review for ADD (or at minimum a frontier-update note) at the next weekly review — two independent sources, concrete numbers, and public code clear the bar the single 10-05 paper didn't on its own.

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
