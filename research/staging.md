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
