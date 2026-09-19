# Paper reading curriculum

This is the paper reading list for the LLM Research Engineer course: roughly 30 papers in dependency order, each introduced right after the module that gives you the machinery to read it. You never read a paper before you have the background — the ordering is the point. See the [Curriculum map](../curriculum/course-outline.md) for how the modules line up.

## How to read a paper in this course

Never read a 40-page paper linearly. For every paper below, answer these nine questions in this order and stop when you have the answers:

1. **What problem are they solving?**
2. **What did people do before this?**
3. **What is the key idea?**
4. **What changed technically?**
5. **What equations matter?**
6. **What experiments prove the claim?**
7. **What should you implement?**
8. **What can you skip?**
9. **How did this influence later LLMs?**

Read the abstract, the intro's last paragraph, the main figure, and the equations first. Then jump to the parts that answer the questions above. Skimming with a purpose beats reading cover to cover.

## The reading order

| # | Paper | Year | Read after module | Link |
|---|-------|------|-------------------|------|
| 1 | GPT-2 | 2019 | Module 6 | [GPT-2](https://cdn.openai.com/better-language-models/language-models.pdf) |
| 2 | GPT-3 | 2020 | Module 6 | [GPT-3](https://arxiv.org/abs/2005.14165) |
| 3 | Scaling Laws | 2020 | Module 10 | [Scaling Laws](https://arxiv.org/abs/2001.08361) |
| 4 | Chinchilla | 2022 | Module 10 | [Chinchilla](https://arxiv.org/abs/2203.15556) |
| 5 | Megatron-LM | 2019 | Module 9 | [Megatron-LM](https://arxiv.org/abs/1909.08053) |
| 6 | ZeRO | 2019 | Module 9 | [ZeRO](https://arxiv.org/abs/1910.02054) |
| 7 | FlashAttention | 2022 | Module 8 | [FlashAttention](https://arxiv.org/abs/2205.14135) |
| 8 | RoPE / RoFormer | 2021 | Module 12 | [RoFormer](https://arxiv.org/abs/2104.09864) |
| 9 | LLaMA | 2023 | Module 12 | [LLaMA](https://arxiv.org/abs/2302.13971) |
| 10 | Llama 3 | 2024 | Module 12 | [Llama 3](https://arxiv.org/abs/2407.21783) |
| 11 | OLMo 2 | 2024 | Module 12 | [OLMo 2](https://arxiv.org/abs/2501.00656) |
| 12 | Switch Transformers | 2021 | Module 12 | [Switch Transformers](https://arxiv.org/abs/2101.03961) |
| 13 | DeepSeekMoE | 2024 | Module 12 | [DeepSeekMoE](https://arxiv.org/abs/2401.06066) |
| 14 | DeepSeek-V3 | 2024 | Module 12 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) |
| 15 | FLAN | 2021 | Module 14 | [FLAN](https://arxiv.org/abs/2109.01652) |
| 16 | Self-Instruct | 2022 | Module 14 | [Self-Instruct](https://arxiv.org/abs/2212.10560) |
| 17 | InstructGPT | 2022 | Module 15 | [InstructGPT](https://arxiv.org/abs/2203.02155) |
| 18 | Anthropic HH-RLHF | 2022 | Module 15 | [HH-RLHF](https://arxiv.org/abs/2204.05862) |
| 19 | Constitutional AI | 2022 | Module 15 | [Constitutional AI](https://arxiv.org/abs/2212.08073) |
| 20 | RLAIF | 2023 | Module 15 | [RLAIF](https://arxiv.org/abs/2309.00267) |
| 21 | DPO | 2023 | Module 15 | [DPO](https://arxiv.org/abs/2305.18290) |
| 22 | Chain-of-Thought | 2022 | Module 17 | [Chain-of-Thought](https://arxiv.org/abs/2201.11903) |
| 23 | STaR | 2022 | Module 17 | [STaR](https://arxiv.org/abs/2203.14465) |
| 24 | Let's Verify Step by Step | 2023 | Module 17 | [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) |
| 25 | DeepSeekMath (GRPO) | 2024 | Module 16 | [DeepSeekMath](https://arxiv.org/abs/2402.03300) |
| 26 | DeepSeek-R1 | 2025 | Module 17 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) |
| 27 | Toolformer | 2023 | Module 18 | [Toolformer](https://arxiv.org/abs/2302.04761) |
| 28 | Gorilla | 2023 | Module 18 | [Gorilla](https://arxiv.org/abs/2305.15334) |
| 29 | ReAct | 2022 | Module 19 | [ReAct](https://arxiv.org/abs/2210.03629) |
| 30 | SWE-agent | 2024 | Module 19 | [SWE-agent](https://arxiv.org/abs/2405.15793) |

---

### 1 · GPT-2

**Full title / authors / year:** "Language Models are Unsupervised Multitask Learners", Radford et al. 2019. **Paper: [language-models.pdf](https://cdn.openai.com/better-language-models/language-models.pdf)** · GitHub [openai/gpt-2](https://github.com/openai/gpt-2).

**Prerequisite modules:** Module 6.

**Why it appears here:** Right after you build a decoder-only transformer in Module 6, this shows that a plain next-token LM, scaled up, does many tasks with no task-specific training.

**What you MUST understand:**
- The model is a decoder-only transformer trained only on next-token prediction.
- "Zero-shot" here means framing tasks as text continuation, no gradient updates.
- Scale (params + WebText data) is the lever for emergent multitask ability.
- Byte-level BPE tokenization and its role.

**What you can skip:**
- The detailed per-task benchmark tables.
- The extended discussion of memorization/overlap analysis.

**Key figures/tables to inspect:** the figure showing zero-shot performance climbing with model size across the four GPT-2 sizes.

**Equations that matter:** the LM objective, maximize $\sum_i \log p(x_i \mid x_{<i})$. That is the whole training signal.

**Implementation consequence / code this after reading:** connect it to your `code/` decoder-only transformer; run zero-shot prompting against your own trained mini-GPT.

**Connections:** → GPT-3, → Scaling Laws.

---

### 2 · GPT-3

**Full title / authors / year:** "Language Models are Few-Shot Learners", Brown et al. 2020. **Paper: [arxiv 2005.14165](https://arxiv.org/abs/2005.14165)**.

**Prerequisite modules:** Module 6.

**Why it appears here:** It extends GPT-2's story — same architecture, 100x bigger — and introduces in-context (few-shot) learning, the behavior you will spend the rest of the course shaping.

**What you MUST understand:**
- Few-shot / one-shot / zero-shot via in-context examples, no weight updates.
- The 175B model is architecturally almost identical to GPT-2.
- In-context learning improves with scale.
- The data-mixture and dedup pipeline at a high level.

**What you can skip:**
- The very long per-benchmark appendix tables.
- The broader-impacts section on first read.

**Key figures/tables to inspect:** the figure showing few-shot accuracy rising with parameter count; the in-context-examples-vs-accuracy curve.

**Equations that matter:** same LM objective as GPT-2; no new loss.

**Implementation consequence / code this after reading:** add a few-shot prompt harness in `code/` that formats k examples before the query.

**Connections:** ← GPT-2, → Scaling Laws, → FLAN, → InstructGPT.

---

### 3 · Scaling Laws

**Full title / authors / year:** "Scaling Laws for Neural Language Models", Kaplan et al. 2020. **Paper: [arxiv 2001.08361](https://arxiv.org/abs/2001.08361)**.

**Prerequisite modules:** Module 10.

**Why it appears here:** After Module 10's training and optimization material, this quantifies how loss falls predictably with model size, data, and compute — the empirical backbone of modern LLM planning.

**What you MUST understand:**
- Loss follows power laws in parameters $N$, dataset size $D$, and compute $C$.
- Bigger models are more sample-efficient.
- The laws hold over many orders of magnitude.
- How compute budget is allocated between $N$ and $D$ (their answer differs from Chinchilla).

**What you can skip:**
- The detailed fitting methodology appendices.
- The batch-size critical-value derivations on first pass.

**Key figures/tables to inspect:** the scaling-law power-law fit plotted as straight lines on log-log axes.

**Equations that matter:** the power law $L(N) = (N_c / N)^{\alpha_N}$, with analogous forms $L(D)=(D_c/D)^{\alpha_D}$ and $L(C)=(C_c/C)^{\alpha_C}$.

**Implementation consequence / code this after reading:** fit a power law to your own loss-vs-size sweeps in `code/`; predict a larger run's loss.

**Connections:** ← GPT-3, → Chinchilla.

---

### 4 · Chinchilla

**Full title / authors / year:** "Training Compute-Optimal Large Language Models", Hoffmann et al. 2022. **Paper: [arxiv 2203.15556](https://arxiv.org/abs/2203.15556)**.

**Prerequisite modules:** Module 10.

**Why it appears here:** It corrects Kaplan's compute-allocation conclusion, showing most large models were badly undertrained on data — essential before you plan any real run.

**What you MUST understand:**
- For a fixed compute budget, $N$ and $D$ should scale roughly equally.
- The rule of thumb: tokens ≈ 20× parameters.
- Prior models (e.g. Gopher, GPT-3) were oversized and underfed.
- Three independent estimation approaches converge on the same answer.

**What you can skip:**
- The full per-approach uncertainty analysis.
- Downstream eval tables beyond the headline.

**Key figures/tables to inspect:** the loss-vs-compute isoFLOP curves whose minima trace the optimal $N$/$D$ frontier.

**Equations that matter:** the parametric fit $L(N,D) = E + A/N^{\alpha} + B/D^{\beta}$, minimized under a FLOP constraint.

**Implementation consequence / code this after reading:** write a small solver in `code/` that, given a FLOP budget, returns compute-optimal $N$ and token count.

**Connections:** ← Scaling Laws, → LLaMA, → Llama 3.

---

### 5 · Megatron-LM

**Full title / authors / year:** Shoeybi et al. 2019. **Paper: [arxiv 1909.08053](https://arxiv.org/abs/1909.08053)** · GitHub [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

**Prerequisite modules:** Module 9.

**Why it appears here:** After Module 9's distributed-training foundations, this shows how to split a single transformer across GPUs when it no longer fits on one.

**What you MUST understand:**
- Tensor (intra-layer) model parallelism: split attention heads and MLP matrices across devices.
- Where all-reduce communication is inserted in the forward/backward pass.
- Why this is orthogonal to data parallelism.
- The near-linear scaling they report.

**What you can skip:**
- The exact CUDA/cuBLAS implementation details.
- The BERT-specific experiments.

**Key figures/tables to inspect:** the diagram splitting the MLP and self-attention blocks column/row-wise across GPUs; the scaling-efficiency table.

**Equations that matter:** no new loss; the key is the partitioning of $Y = \text{GeLU}(XA)$ by columns of $A$ and $Z = YB$ by rows of $B$.

**Implementation consequence / code this after reading:** connect to `code/` parallelism exercises; implement a 2-GPU tensor-parallel MLP.

**Connections:** ← (Module 9), → ZeRO, → DeepSeek-V3.

---

### 6 · ZeRO

**Full title / authors / year:** Rajbhandari et al. 2019. **Paper: [arxiv 1910.02054](https://arxiv.org/abs/1910.02054)** · GitHub [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed).

**Prerequisite modules:** Module 9.

**Why it appears here:** The complement to Megatron: instead of splitting the math, ZeRO splits the memory (optimizer state, gradients, parameters) across data-parallel workers.

**What you MUST understand:**
- The three stages: partition optimizer states, then gradients, then parameters.
- Why optimizer state (Adam moments) dominates memory in mixed precision.
- The communication/memory trade-off of each stage.
- How ZeRO enables training models far larger than one GPU's memory.

**What you can skip:**
- ZeRO-Offload / NVMe details unless relevant.
- The exact bandwidth-cost derivations on first read.

**Key figures/tables to inspect:** the figure showing per-device memory shrinking across stages 1/2/3.

**Equations that matter:** the memory accounting — for Adam in fp16, roughly $16N$ bytes of state divided across $P$ devices.

**Implementation consequence / code this after reading:** in `code/`, contrast plain DDP memory with a ZeRO-style optimizer-state shard.

**Connections:** ← Megatron-LM, → DeepSeek-V3.

---

### 7 · FlashAttention

**Full title / authors / year:** Dao et al. 2022. **Paper: [arxiv 2205.14135](https://arxiv.org/abs/2205.14135)** · GitHub [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).

**Prerequisite modules:** Module 8.

**Why it appears here:** Immediately after Module 8's attention mechanics, this shows why the naive attention you just wrote is memory-bound and how to fix it without changing the math.

**What you MUST understand:**
- Attention is bottlenecked by HBM reads/writes, not FLOPs.
- Tiling + online softmax computes exact attention without materializing the full $N\times N$ matrix.
- It is exact, not an approximation.
- The IO-complexity argument (why fewer HBM accesses win).

**What you can skip:**
- The CUDA kernel specifics unless you write kernels.
- The block-sparse extension on first read.

**Key figures/tables to inspect:** the memory-hierarchy diagram (SRAM vs HBM); the runtime/memory-vs-sequence-length plots.

**Equations that matter:** standard attention $\text{softmax}(QK^\top/\sqrt{d})V$, recomputed blockwise with a running max and sum for numerically stable online softmax.

**Implementation consequence / code this after reading:** in `code/`, swap your reference attention for a tiled/online-softmax version and benchmark.

**Connections:** ← (Module 8), → LLaMA, → DeepSeek-V3.

---

### 8 · RoPE / RoFormer

**Full title / authors / year:** "RoFormer: Enhanced Transformer with Rotary Position Embedding", Su et al. 2021. **Paper: [arxiv 2104.09864](https://arxiv.org/abs/2104.09864)**.

**Prerequisite modules:** Module 12.

**Why it appears here:** Module 12 covers modern architecture choices; RoPE is the position-encoding used by nearly every current open LLM, so it comes first.

**What you MUST understand:**
- Position is injected by rotating query/key vectors by an angle proportional to position.
- The dot product then depends only on relative position.
- Why this extrapolates better than learned absolute embeddings.
- That it acts on Q and K, not on the value or the residual stream.

**What you can skip:**
- The full complex-number derivation if the 2D rotation intuition suffices.
- The non-LM experiments.

**Key figures/tables to inspect:** the figure showing pairs of dimensions rotated by position-dependent angles.

**Equations that matter:** each 2D sub-vector is rotated by $\theta_m = m\,\omega$, so $\langle R_m q, R_n k\rangle$ depends on $m-n$ — relative position emerges from absolute rotations.

**Implementation consequence / code this after reading:** implement `apply_rope(q, k)` in `code/` and drop it into your attention block.

**Connections:** ← FlashAttention, → LLaMA, → Llama 3.

---

### 9 · LLaMA

**Full title / authors / year:** "LLaMA: Open and Efficient Foundation Language Models", Touvron et al. 2023. **Paper: [arxiv 2302.13971](https://arxiv.org/abs/2302.13971)** · GitHub [meta-llama/llama](https://github.com/meta-llama/llama).

**Prerequisite modules:** Module 12.

**Why it appears here:** It bundles the modern recipe (RoPE, RMSNorm, SwiGLU, Chinchilla-style data scaling) into one open model — the reference architecture for the rest of the course.

**What you MUST understand:**
- Pre-normalization with RMSNorm, SwiGLU activation, RoPE.
- Trained on far more tokens than Chinchilla-optimal for better inference economics.
- Fully public, reproducible data sources.
- Why "compute-optimal to train" differs from "optimal to serve".

**What you can skip:**
- The carbon-footprint accounting.
- Some downstream benchmark tables.

**Key figures/tables to inspect:** the architecture/hyperparameter table for the four sizes; the training-loss-vs-tokens curves.

**Equations that matter:** RMSNorm $\bar x = x / \sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}$; SwiGLU gating in the FFN.

**Implementation consequence / code this after reading:** assemble a LLaMA-style block in `code/`: RMSNorm + RoPE attention + SwiGLU MLP.

**Connections:** ← RoPE, ← Chinchilla, ← FlashAttention, → Llama 3, → OLMo 2.

---

### 10 · Llama 3

**Full title / authors / year:** "The Llama 3 Herd of Models", Grattafiori et al. 2024. **Paper: [arxiv 2407.21783](https://arxiv.org/abs/2407.21783)** · GitHub [meta-llama/llama3](https://github.com/meta-llama/llama3).

**Prerequisite modules:** Module 12.

**Why it appears here:** It updates the LLaMA recipe at frontier scale and documents the full pretrain-plus-post-train pipeline end to end — a modern production blueprint.

**What you MUST understand:**
- The scaled-up dense architecture and data/tokenizer changes.
- The multi-stage post-training (SFT + preference optimization) at a high level.
- Long-context extension.
- Data-quality and annealing choices.

**What you can skip:**
- The multimodal-adapter sections if focusing on text.
- Exhaustive eval appendices.

**Key figures/tables to inspect:** the overall training-pipeline diagram; the scaling/data-mix tables.

**Equations that matter:** no new core loss; same LM objective plus preference-tuning losses covered later (DPO/PPO).

**Implementation consequence / code this after reading:** in `code/`, extend your LLaMA block config to a Llama-3-style setup (GQA, larger vocab, longer context).

**Connections:** ← LLaMA, ← Chinchilla, → OLMo 2, → DeepSeek-V3.

---

### 11 · OLMo 2

**Full title / authors / year:** "2 OLMo 2 Furious", Team OLMo et al. 2024. **Paper: [arxiv 2501.00656](https://arxiv.org/abs/2501.00656)** · GitHub [allenai/OLMo](https://github.com/allenai/OLMo). See also OLMo 1: [arxiv 2402.00838](https://arxiv.org/abs/2402.00838).

**Prerequisite modules:** Module 12.

**Why it appears here:** OLMo is the fully open model — data, code, checkpoints, logs — so it is the one you can actually reproduce and study end to end after learning the architecture.

**What you MUST understand:**
- Everything (data, training code, intermediate checkpoints) is released.
- The stability fixes and normalization/initialization choices they document.
- The two-stage pretraining with a late high-quality data phase.
- How full openness enables real ablation study.

**What you can skip:**
- Infrastructure logistics specific to their cluster.
- Some benchmark leaderboard tables.

**Key figures/tables to inspect:** the training-stability curves (loss spikes and their fixes); the data-mix schedule table.

**Equations that matter:** standard LM loss; focus on the normalization/reordering details rather than new equations.

**Implementation consequence / code this after reading:** use an OLMo checkpoint in `code/` as a reproducible baseline you can fine-tune.

**Connections:** ← LLaMA, ← Llama 3, → FLAN.

---

### 12 · Switch Transformers

**Full title / authors / year:** Fedus et al. 2021. **Paper: [arxiv 2101.03961](https://arxiv.org/abs/2101.03961)**.

**Prerequisite modules:** Module 12.

**Why it appears here:** It introduces the mixture-of-experts idea in the transformer FFN — decoupling parameter count from per-token compute — which the DeepSeek papers build on.

**What you MUST understand:**
- Each token is routed to one expert (top-1), so params grow but FLOPs/token stay roughly fixed.
- The routing (gating) network and its softmax over experts.
- The load-balancing auxiliary loss and why it is needed.
- Capacity factor and token dropping.

**What you can skip:**
- The distillation-back-to-dense sections.
- TPU-specific sharding details.

**Key figures/tables to inspect:** the top-1 routing diagram; the params-vs-quality-at-fixed-FLOPs plot.

**Equations that matter:** gate $p_i(x) = \text{softmax}(W_r x)_i$, route to $\arg\max_i p_i$; add a load-balancing auxiliary loss to keep expert usage uniform.

**Implementation consequence / code this after reading:** implement a top-1 MoE FFN with an aux balance loss in `code/`.

**Connections:** ← (Module 12), → DeepSeekMoE.

---

### 13 · DeepSeekMoE

**Full title / authors / year:** Dai et al. 2024. **Paper: [arxiv 2401.06066](https://arxiv.org/abs/2401.06066)** · GitHub [deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE).

**Prerequisite modules:** Module 12.

**Why it appears here:** It refines Switch-style MoE with finer-grained and shared experts, the design that powers DeepSeek-V3, so read it right after the basic MoE idea.

**What you MUST understand:**
- Fine-grained experts: split each expert into smaller ones for more routing combinations.
- Shared experts always active to capture common knowledge.
- How this improves specialization vs plain top-1.
- The balance-loss variants they use.

**What you can skip:**
- The exact ablation grid.
- Training-infra specifics.

**Key figures/tables to inspect:** the diagram contrasting conventional experts with fine-grained + shared experts.

**Equations that matter:** top-$k$ routing over many small experts plus a set of always-on shared experts; output is the sum of shared and selected routed expert outputs.

**Implementation consequence / code this after reading:** extend your `code/` MoE to top-$k$ with fine-grained and shared experts.

**Connections:** ← Switch Transformers, → DeepSeek-V3.

---

### 14 · DeepSeek-V3

**Full title / authors / year:** "DeepSeek-V3 Technical Report", DeepSeek-AI 2024. **Paper: [arxiv 2412.19437](https://arxiv.org/abs/2412.19437)** · GitHub [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3).

**Prerequisite modules:** Module 12.

**Why it appears here:** It is the capstone architecture paper — MoE, Multi-head Latent Attention, multi-token prediction, FP8 training — integrating everything from Modules 8, 9, and 12.

**What you MUST understand:**
- Multi-head Latent Attention (MLA) and its KV-cache compression.
- DeepSeekMoE with auxiliary-loss-free load balancing.
- Multi-token prediction training objective.
- FP8 mixed-precision training at scale.

**What you can skip:**
- The full infrastructure/cluster description.
- Exhaustive eval tables.

**Key figures/tables to inspect:** the MLA diagram; the overall architecture figure; the FP8 training-framework diagram.

**Equations that matter:** MLA's low-rank down/up projection of keys/values; the bias-based balancing update that avoids a separate aux loss.

**Implementation consequence / code this after reading:** in `code/`, prototype MLA's KV compression and compare cache size to standard MHA/GQA.

**Connections:** ← DeepSeekMoE, ← Megatron-LM, ← ZeRO, ← FlashAttention, → DeepSeekMath.

---

### 15 · FLAN

**Full title / authors / year:** "Finetuned Language Models Are Zero-Shot Learners", Wei et al. 2021. **Paper: [arxiv 2109.01652](https://arxiv.org/abs/2109.01652)** · GitHub [google-research/FLAN](https://github.com/google-research/FLAN).

**Prerequisite modules:** Module 14.

**Why it appears here:** Module 14 begins post-training; FLAN shows the first big step — instruction tuning on many tasks phrased as instructions dramatically improves zero-shot generalization.

**What you MUST understand:**
- Instruction tuning: fine-tune on a mix of NLP tasks phrased as natural-language instructions.
- It improves zero-shot on held-out task types.
- The importance of task/template diversity.
- It is still supervised fine-tuning, no RL yet.

**What you can skip:**
- The per-cluster benchmark breakdowns.
- Template-engineering minutiae.

**Key figures/tables to inspect:** the figure showing held-out zero-shot gains rising with the number of instruction-tuning task clusters.

**Equations that matter:** ordinary supervised cross-entropy on (instruction, output) pairs; no new loss.

**Implementation consequence / code this after reading:** build an instruction-formatting + SFT loop in `code/` over a multi-task mix.

**Connections:** ← GPT-3, ← OLMo 2, → Self-Instruct, → InstructGPT.

---

### 16 · Self-Instruct

**Full title / authors / year:** Wang et al. 2022. **Paper: [arxiv 2212.10560](https://arxiv.org/abs/2212.10560)** · GitHub [yizhongw/self-instruct](https://github.com/yizhongw/self-instruct).

**Prerequisite modules:** Module 14.

**Why it appears here:** It solves FLAN's data-cost problem: bootstrap instruction data from the model itself, the technique behind most open instruction datasets.

**What you MUST understand:**
- The pipeline: seed tasks → model generates new instructions → generate inputs/outputs → filter → fine-tune.
- Filtering/dedup steps that keep quality up.
- Why synthetic data can rival human-written data here.
- The bootstrapping loop structure.

**What you can skip:**
- The exact prompt templates.
- Per-benchmark result tables.

**Key figures/tables to inspect:** the pipeline diagram of the generate-filter-finetune loop.

**Equations that matter:** none new; it is a data-generation procedure feeding standard SFT.

**Implementation consequence / code this after reading:** implement a small self-instruct generation + filtering script in `code/`.

**Connections:** ← FLAN, → InstructGPT, → STaR.

---

### 17 · InstructGPT

**Full title / authors / year:** "Training language models to follow instructions with human feedback", Ouyang et al. 2022. **Paper: [arxiv 2203.02155](https://arxiv.org/abs/2203.02155)**.

**Prerequisite modules:** Module 15.

**Why it appears here:** Module 15 covers RL from human feedback; this is the canonical three-stage RLHF recipe (SFT → reward model → PPO) that defined modern alignment.

**What you MUST understand:**
- The three stages: SFT, reward-model training on comparisons, PPO against the reward model.
- The KL penalty to the SFT policy that keeps outputs on-distribution.
- The alignment tax and why a smaller aligned model can beat a larger raw one.
- The reward model is trained on ranked human preferences.

**What you can skip:**
- The annotator-instruction appendices.
- Some safety/eval tables on first read.

**Key figures/tables to inspect:** the three-stage RLHF pipeline diagram; the human-preference win-rate bars vs GPT-3.

**Equations that matter:** reward-model loss via Bradley-Terry $P(y_w \succ y_l) = \sigma(r(x,y_w) - r(x,y_l))$; PPO clipped objective $L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat A_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t)]$ with $r_t(\theta)=\pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$, plus a KL-to-reference penalty.

**Implementation consequence / code this after reading:** in `code/`, train a small reward model on preference pairs and wire up a PPO loop with KL penalty.

**Connections:** ← FLAN, ← Self-Instruct, ← GPT-3, → HH-RLHF, → DPO, → DeepSeekMath.

---

### 18 · Anthropic HH-RLHF

**Full title / authors / year:** "Training a Helpful and Harmless Assistant with RLHF", Bai et al. 2022. **Paper: [arxiv 2204.05862](https://arxiv.org/abs/2204.05862)** · GitHub [anthropics/hh-rlhf](https://github.com/anthropics/hh-rlhf).

**Prerequisite modules:** Module 15.

**Why it appears here:** It applies RLHF specifically to the helpful-vs-harmless trade-off and releases the preference dataset you can actually train on.

**What you MUST understand:**
- Separate helpfulness and harmlessness preference data and the tension between them.
- Online iterated RLHF: collect fresh comparisons as the policy improves.
- Reward-model calibration and scaling trends.
- The released comparison-data format.

**What you can skip:**
- The red-teaming methodology details.
- Some scaling-plot appendices.

**Key figures/tables to inspect:** the helpfulness-vs-harmlessness trade-off/Pareto plot; RLHF-vs-base win-rate curves.

**Equations that matter:** same Bradley-Terry reward-model loss and PPO objective as InstructGPT.

**Implementation consequence / code this after reading:** load the HH-RLHF pairs in `code/` as a real preference dataset for your reward model / DPO.

**Connections:** ← InstructGPT, → Constitutional AI, → DPO.

---

### 19 · Constitutional AI

**Full title / authors / year:** "Constitutional AI: Harmlessness from AI Feedback", Bai et al. 2022. **Paper: [arxiv 2212.08073](https://arxiv.org/abs/2212.08073)**.

**Prerequisite modules:** Module 15.

**Why it appears here:** It replaces human harmlessness labels with model self-critique against a written constitution — the bridge from RLHF to AI feedback.

**What you MUST understand:**
- Two phases: supervised self-critique-and-revise, then RL from AI preferences (RLAIF).
- A short list of natural-language principles ("the constitution") drives the critiques.
- It reduces the need for human harm labels.
- How this keeps the model helpful while making it harmless.

**What you can skip:**
- The full constitution text.
- Some red-team transcripts.

**Key figures/tables to inspect:** the diagram of the critique→revise→RL pipeline; the helpful/harmless trade-off improvement plot.

**Equations that matter:** same RLHF machinery, but preference labels come from an AI judge instead of humans.

**Implementation consequence / code this after reading:** implement a critique-and-revise pass in `code/` driven by a few written principles.

**Connections:** ← HH-RLHF, → RLAIF.

---

### 20 · RLAIF

**Full title / authors / year:** "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback", Lee et al. 2023. **Paper: [arxiv 2309.00267](https://arxiv.org/abs/2309.00267)**.

**Prerequisite modules:** Module 15.

**Why it appears here:** It directly measures whether AI-generated preferences can match human ones, quantifying the idea Constitutional AI introduced.

**What you MUST understand:**
- An off-the-shelf LLM labels preferences that then train the reward model.
- RLAIF can match or approach RLHF on their tasks.
- Prompting choices for the AI labeler matter.
- Direct-RLAIF (score without a separate reward model) as a variant.

**What you can skip:**
- Per-task hyperparameter tables.
- Some ablation minutiae.

**Key figures/tables to inspect:** the RLAIF-vs-RLHF win-rate comparison bars.

**Equations that matter:** unchanged RLHF pipeline; the AI labeler replaces the human in the Bradley-Terry comparison data.

**Implementation consequence / code this after reading:** in `code/`, swap your human-label step for an LLM-judge labeler and compare reward models.

**Connections:** ← Constitutional AI, → DPO.

---

### 21 · DPO

**Full title / authors / year:** "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", Rafailov et al. 2023. **Paper: [arxiv 2305.18290](https://arxiv.org/abs/2305.18290)** · GitHub [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization).

**Prerequisite modules:** Module 15.

**Why it appears here:** It collapses the whole RLHF pipeline into a single classification loss — no separate reward model, no PPO — the default preference method today.

**What you MUST understand:**
- The RLHF optimum has a closed form linking reward to the policy/reference log-ratio.
- That lets you optimize preferences directly with a supervised-style loss.
- The role of $\beta$ and the reference policy.
- Why it is far more stable than PPO.

**What you can skip:**
- The full derivation on first read (get the result, revisit the proof later).
- Some benchmark tables.

**Key figures/tables to inspect:** the figure comparing DPO to PPO-based RLHF on reward/win-rate at matched KL.

**Equations that matter:** $L_{DPO} = -\mathbb{E}_{(x,y_w,y_l)}\big[\log \sigma(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})\big]$.

**Implementation consequence / code this after reading:** implement the DPO loss in `code/` and train on the HH-RLHF pairs; compare to your PPO run.

**Connections:** ← InstructGPT, ← HH-RLHF, ← RLAIF, → DeepSeekMath.

---

### 22 · Chain-of-Thought

**Full title / authors / year:** "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", Wei et al. 2022. **Paper: [arxiv 2201.11903](https://arxiv.org/abs/2201.11903)**.

**Prerequisite modules:** Module 17.

**Why it appears here:** Module 17 is about reasoning; this is the foundational result that prompting for intermediate steps unlocks multi-step reasoning in large models.

**What you MUST understand:**
- Prompting the model to show its steps improves arithmetic/commonsense/symbolic reasoning.
- The effect is emergent — it appears only above a size threshold.
- It is purely a prompting change, no training.
- Reasoning quality vs final-answer accuracy.

**What you can skip:**
- Per-benchmark tables.
- The exact exemplar wording.

**Key figures/tables to inspect:** the figure showing CoT accuracy diverging from standard prompting as model scale grows.

**Equations that matter:** none; conceptually the model samples a reasoning chain $z$ then the answer, $p(a\mid x)=\sum_z p(a\mid x,z)p(z\mid x)$.

**Implementation consequence / code this after reading:** add a CoT prompt template and answer-extraction step in `code/`.

**Connections:** ← (Module 17), → STaR, → Let's Verify, → DeepSeek-R1.

---

### 23 · STaR

**Full title / authors / year:** "STaR: Bootstrapping Reasoning With Reasoning", Zelikman et al. 2022. **Paper: [arxiv 2203.14465](https://arxiv.org/abs/2203.14465)** · GitHub [ezelikman/STaR](https://github.com/ezelikman/STaR).

**Prerequisite modules:** Module 17.

**Why it appears here:** It turns CoT from a prompting trick into a training loop: generate rationales, keep the ones that reach the right answer, fine-tune, repeat.

**What you MUST understand:**
- The bootstrap loop: sample rationales, filter by correctness, fine-tune on them.
- Rationalization: given the answer, back-generate a rationale for problems it missed.
- Why self-generated correct reasoning is a training signal.
- The convergence behavior of iterating this.

**What you can skip:**
- Dataset-specific details.
- Some ablations.

**Key figures/tables to inspect:** the loop diagram; the accuracy-improves-per-iteration plot.

**Equations that matter:** none new; SFT on the filtered set of (problem, correct rationale, answer) triples each round.

**Implementation consequence / code this after reading:** implement the sample-filter-finetune loop in `code/` on a math dataset.

**Connections:** ← Chain-of-Thought, ← Self-Instruct, → Let's Verify, → DeepSeek-R1.

---

### 24 · Let's Verify Step by Step

**Full title / authors / year:** Lightman et al. 2023. **Paper: [arxiv 2305.20050](https://arxiv.org/abs/2305.20050)** · GitHub [openai/prm800k](https://github.com/openai/prm800k).

**Prerequisite modules:** Module 17.

**Why it appears here:** It shows that rewarding each reasoning step (process supervision) beats rewarding only the final answer — key to reliable reasoning models.

**What you MUST understand:**
- Process reward models (PRM) score each step; outcome reward models (ORM) score only the answer.
- PRMs give better, more reliable reasoning selection.
- Used at inference to rank/verify sampled solutions.
- The PRM800K human step-label dataset.

**What you can skip:**
- Annotation-interface details.
- Some active-learning specifics.

**Key figures/tables to inspect:** the figure comparing PRM vs ORM solve-rate as the number of sampled solutions grows.

**Equations that matter:** a verifier reranks $N$ samples by aggregated step scores (e.g. product/min of per-step correctness probabilities).

**Implementation consequence / code this after reading:** in `code/`, train a step-level verifier and use best-of-N reranking.

**Connections:** ← Chain-of-Thought, ← STaR, → DeepSeek-R1.

---

### 25 · DeepSeekMath (GRPO)

**Full title / authors / year:** "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", Shao et al. 2024. **Paper: [arxiv 2402.03300](https://arxiv.org/abs/2402.03300)** · GitHub [deepseek-ai/DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math).

**Prerequisite modules:** Module 16.

**Why it appears here:** After Module 16's RL algorithms, this introduces GRPO — PPO without a critic — the method that made large-scale RL for reasoning cheap.

**What you MUST understand:**
- GRPO drops PPO's learned value network and uses a group-relative baseline.
- Sample a group of outputs per prompt; the baseline is the group's mean reward.
- This halves memory vs PPO (no critic).
- The math-specific data pipeline that feeds it.

**What you can skip:**
- The corpus-construction details.
- Some eval tables.

**Key figures/tables to inspect:** the diagram contrasting PPO (with critic) vs GRPO (group baseline); the benchmark-gain bars.

**Equations that matter:** advantage $\hat A_i = (r_i - \mathrm{mean}(r))/\mathrm{std}(r)$ over a sampled group, plugged into a PPO-style clipped objective with a KL-to-reference term but no value network.

**Implementation consequence / code this after reading:** implement GRPO in `code/`: sample a group, normalize rewards, apply the clipped update.

**Connections:** ← InstructGPT, ← DPO, ← DeepSeek-V3, → DeepSeek-R1.

---

### 26 · DeepSeek-R1

**Full title / authors / year:** "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", DeepSeek-AI 2025. **Paper: [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)** · GitHub [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1).

**Prerequisite modules:** Module 17.

**Why it appears here:** The capstone reasoning paper: large-scale RL (GRPO) with simple verifiable rewards produces long chain-of-thought and emergent reasoning — pulling together everything from Modules 16 and 17.

**What you MUST understand:**
- R1-Zero: pure RL from a base model with rule-based (verifiable) rewards, no SFT first.
- Reasoning behaviors (self-reflection, longer chains) emerge from RL.
- The cold-start SFT + multi-stage pipeline for the full R1.
- Distilling reasoning into smaller dense models.

**What you can skip:**
- Some benchmark tables.
- Deployment specifics.

**Key figures/tables to inspect:** the figure showing response length and accuracy rising over RL training steps ("aha moment").

**Equations that matter:** GRPO objective from DeepSeekMath with rewards from answer-correctness and format rules rather than a learned reward model.

**Implementation consequence / code this after reading:** in `code/`, run your GRPO loop with a rule-based verifier reward on a math/coding task.

**Connections:** ← DeepSeekMath, ← Chain-of-Thought, ← STaR, ← Let's Verify, → Toolformer.

---

### 27 · Toolformer

**Full title / authors / year:** "Toolformer: Language Models Can Teach Themselves to Use Tools", Schick et al. 2023. **Paper: [arxiv 2302.04761](https://arxiv.org/abs/2302.04761)**.

**Prerequisite modules:** Module 18.

**Why it appears here:** Module 18 covers tool use; Toolformer is the first self-supervised way a model learns when and how to call APIs by inserting call tokens into its own training data.

**What you MUST understand:**
- The model annotates text with candidate API calls, then keeps calls that reduce loss on following tokens.
- Fine-tune on this filtered, call-augmented data.
- Tools (calculator, search, QA) are invoked mid-generation and results spliced back in.
- The self-supervised utility filter is the core trick.

**What you can skip:**
- Per-tool API details.
- Some benchmark tables.

**Key figures/tables to inspect:** the diagram of sampling, executing, and filtering API calls by loss reduction.

**Equations that matter:** keep a call if it lowers weighted next-token loss versus no call — a loss-difference threshold, not a new training loss.

**Implementation consequence / code this after reading:** in `code/`, implement the call-insertion + loss-filter data pipeline for one tool.

**Connections:** ← DeepSeek-R1, → Gorilla, → ReAct.

---

### 28 · Gorilla

**Full title / authors / year:** "Gorilla: Large Language Model Connected with Massive APIs", Patil et al. 2023. **Paper: [arxiv 2305.15334](https://arxiv.org/abs/2305.15334)** · GitHub [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla).

**Prerequisite modules:** Module 18.

**Why it appears here:** It scales tool use from a few tools to thousands of real APIs, and adds retrieval so the model calls the right one with correct arguments.

**What you MUST understand:**
- Fine-tuning on API documentation to generate correct API calls.
- Retriever-aware training so it uses live docs at inference.
- Reducing hallucinated/incorrect API calls.
- The AST-based evaluation of call correctness.

**What you can skip:**
- The dataset-collection minutiae.
- Some leaderboard tables.

**Key figures/tables to inspect:** the retrieval-augmented call-generation diagram; the hallucination-reduction comparison.

**Equations that matter:** none new; SFT on (instruction + retrieved docs → API call).

**Implementation consequence / code this after reading:** in `code/`, build a retrieve-then-call pipeline over an API-doc index.

**Connections:** ← Toolformer, → ReAct.

---

### 29 · ReAct

**Full title / authors / year:** "ReAct: Synergizing Reasoning and Acting in Language Models", Yao et al. 2022. **Paper: [arxiv 2210.03629](https://arxiv.org/abs/2210.03629)** · GitHub [ysymyth/ReAct](https://github.com/ysymyth/ReAct).

**Prerequisite modules:** Module 19.

**Why it appears here:** Module 19 covers agents; ReAct is the core loop — interleave reasoning traces with actions and observations — that underlies essentially every LLM agent.

**What you MUST understand:**
- The Thought → Action → Observation loop repeated until an answer.
- Reasoning helps plan actions; actions ground reasoning in real feedback.
- It reduces hallucination versus reasoning alone.
- It is a prompting/framework pattern, not new training.

**What you can skip:**
- Environment-specific setup (ALFWorld/WebShop).
- Some result tables.

**Key figures/tables to inspect:** the figure comparing reasoning-only, acting-only, and combined ReAct trajectories.

**Equations that matter:** none; the model emits interleaved thought/action tokens and consumes observations each step.

**Implementation consequence / code this after reading:** implement a ReAct agent loop in `code/` with a tool-call parser and observation feedback.

**Connections:** ← Toolformer, ← Gorilla, → SWE-agent.

---

### 30 · SWE-agent

**Full title / authors / year:** "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering", Yang et al. 2024. **Paper: [arxiv 2405.15793](https://arxiv.org/abs/2405.15793)** · GitHub [princeton-nlp/SWE-agent](https://github.com/princeton-nlp/SWE-agent).

**Prerequisite modules:** Module 19.

**Why it appears here:** The final paper: a real coding agent that fixes GitHub issues, showing that the interface you give an agent (its Agent-Computer Interface) matters as much as the model.

**What you MUST understand:**
- The Agent-Computer Interface (ACI): custom commands for viewing, editing, and running code.
- A well-designed ACI beats a raw shell for agent performance.
- Evaluation on SWE-bench real GitHub issues.
- The feedback and guardrails (linting, scoped edits) built into the interface.

**What you can skip:**
- Per-command implementation details.
- Some ablation tables.

**Key figures/tables to inspect:** the ACI command diagram; the SWE-bench resolve-rate comparison.

**Equations that matter:** none; the contribution is interface design plus a ReAct-style loop.

**Implementation consequence / code this after reading:** in `code/`, wrap your ReAct loop with a small file-view/edit/run interface and try it on a toy repo issue.

**Connections:** ← ReAct, ← Gorilla.

---

## What to do after each paper

After you finish a paper, write a 5-line summary answering the template's Q1 (what problem), Q3 (key idea), Q5 (which equations matter), Q7 (what you implemented), and Q9 (how it influenced later LLMs). File it under `code/reports/` named for the paper's short name. These summaries become your revision notes and your map of how the field connects.
