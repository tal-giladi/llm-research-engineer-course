# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261117-01 · Numerically stable KL / JSD / entropy / K1-K2-K3 estimators between nearly equal distributions (follow-up to Softmax Collapse/Cancellation)

- **Class:** B
- **Date discovered:** 2026-11-17
- **Date published:** 2026-11-13
- **Source:** https://arxiv.org/abs/2611.18511 (verified; full text read at https://arxiv.org/html/2611.18511v1)
- **Organization/researchers:** Brandon Amos (Reflection AI, per the paper)
- **Category:** RL/post-training
- **What changed:** A second, independent author extends Barron et al.'s softmax/cross-entropy numerical analysis (deferred topic) to the divergences the course uses in distillation and RL: forward and reverse KL, entropy, generalized JSD, and the Monte Carlo KL estimators K1/K2/K3 (K3 is the GRPO/DeepSeekMath KL penalty).
- **Technical summary:** When the two distributions are nearly equal (exactly the regime of a KL penalty against a reference policy early in RL, or a converged distillation student), naive FP32 log-softmax subtraction and the cancellation inside the KL sum corrupt both value and gradient. Examples in the paper (authors): a sampled log-ratio with h = 1e-8 rounds to 0; forward KL with logit gap 2^-25 returns zero value and gradient although p != q; K3 at h = 1e-8 has zero value and zero input derivatives; entropy near uniform has an accurate value but a gradient that rounds to zero. Proposed fixes (eight): rebuild log-ratios from logit differences with expm1/log1p, compute the non-negative remainder E2(x) = e^x - 1 - x with a Taylor series near 0, a zero-sum gradient correction, centring logit differences for the reverse-KL gradient, a stable K3 remainder and derivatives, and h = -log1p(beta*expm1(-s)) for the JSD mixture log-ratio. JAX listings validated by property-based tests are in the paper. Stated limits: the reconstruction can overflow far from equality, and FP32 logit subtraction can still erase tiny differences (open problem).
- **Why it might matter:** The course implements KL (01.3), the K3 penalty in GRPO (16.x) and forward/reverse KL distillation (17.4) from scratch; this gives a concrete, testable failure mode and fix for those exact functions, and turns the deferred Barron topic from a single preprint into a two-author line of work with runnable code listings.
- **Evidence of adoption:** none found. The paper isolates arithmetic and does not measure training outcomes; it does not separately reproduce Barron et al.'s Softmax Collapse.
- **Major organizations using it:** none known.
- **Open-source implementation:** code listings inside the paper (no repository link). The paper links a Barron et al. implementation page (https://jonbarron.info/softmaxfix/), which could not be fetched from this session (host not allowed), so it is unverified.
- **Paper:** https://arxiv.org/abs/2611.18511 ; related https://arxiv.org/abs/2610.31642 (verified)
- **Code:** none (listings in paper)
- **Relationship to existing course material:** `research/deferred/softmax-cross-entropy-numerical-fixes.md` (WAIT, next review 2026-11-22; its "what would change the decision" asks for code or a check against the course's 01.3/01.4, 02.3 and 17.4 implementations); lessons/module-01/lesson-03.md, lessons/module-17/lesson-04.md, GRPO KL penalty in module 16.
- **Potential course lesson:** dated extension of 01.3 / 17.4: "KL between nearly equal distributions in FP32", with a from-scratch stable K3 and a property-based test in `code/tests/`.
- **Confidence:** medium
- **Recommendation:** attach to the 2026-11-22 re-review of `softmax-cross-entropy-numerical-fixes`; check the paper's listings against the course's own KL/K3/distill code before deciding. Still no training-outcome evidence.

### C-20261118-01 · CertiSum: certified-exact accumulation for bitwise-identical MXFP4 training across GPU vendors and parallelism layouts

- **Class:** B
- **Date discovered:** 2026-11-18
- **Date published:** 2026-11-16
- **Source:** https://arxiv.org/abs/2611.21584 (verified; full text read at https://arxiv.org/html/2611.21584v1)
- **Organization/researchers:** Steve Rhyner, Yubo Gao, Stefan de Lasa, Sankeerth Durvasula, Nandita Vijaykumar (University of Toronto / Vector Institute; one author also NVIDIA, per the paper)
- **Category:** distributed
- **What changed:** A third, independent group (after Megatron-Core and TorchTitan) documents bitwise-reproducible training, and goes further: identical training states across AMD MI350X and NVIDIA GB200 and across data/tensor/context-parallel layouts, in 4-bit MXFP4.
- **Technical summary:** FP32 accumulation of MXFP4 products is exact in any summation order whenever every partial sum fits the 24-bit significand; the paper certifies this per output with a sufficient bound (sum of absolute products below 2^24, with exponent guards keeping values normal) from statistics collected during quantization (row bound, per-32-element block bound, or a tight bound from a native GEMM on sign-cleared operands). Uncertified outputs are recomputed exactly from the same quantized operands (chunked native accumulation with exact merging, FP64 when a capacity check passes, or an integer superaccumulator); attention and nearly all weight-gradient GEMMs (46-68% of step GEMM FLOPs) go straight to exact kernels. Quantization blocks are kept unchanged across layouts and partial sums are merged without rounding. Results (authors): 200-step Qwen3-8B-Base runs bit-identical across MI350X and GB200 and across DP/TP/CP changes on MI350X; +11.8-60.5% step latency over the non-exact MXFP4 twin at larger shapes (15.4-77.3% at smaller), still up to 1.24x faster than BF16 and 1.70x faster than deterministic BF16. Stated limits: format-specific bounds (MXFP8 certifies less often), per-platform kernel-exactness assumption, needs replacing GEMM/attention/collectives/autograd; RL needs it in both trainer and inference engine.
- **Why it might matter:** Turns "why is training nondeterministic" (07.3 reproducibility, 16.4 batch invariance) into a clean, hand-checkable arithmetic argument: summation order only matters when a partial sum rounds, and you can bound when it cannot. A from-scratch exercise (certify a toy int/FP4 dot product, show order-independence, then break it past 2^24) fits 07.3 and the MXFP4 material in 08.7.
- **Evidence of adoption:** none found; no framework integration reported (TorchTitan is only a BF16 baseline). Short runs (200 steps).
- **Major organizations using it:** none known.
- **Open-source implementation:** none found (no repository link in the paper text read).
- **Paper:** https://arxiv.org/abs/2611.21584
- **Code:** none
- **Relationship to existing course material:** `research/deferred/bitwise-deterministic-training.md` (WAIT, next review 2026-11-29; bar: a second framework/lab documenting a comparable determinism workflow); `research/accepted/rl-training-inference-mismatch.md`; lessons/module-07/lesson-03.md (exact resume), lessons/module-16/lesson-04.md (batch invariance), lessons/module-08/lesson-07.md (MXFP4).
- **Potential course lesson:** dated extension of 07.3: "When is floating-point summation order-independent? Certifying exact accumulation", with a small NumPy/PyTorch exercise and a test in `code/tests/`.
- **Confidence:** medium
- **Recommendation:** attach to the 2026-11-29 re-review of `bitwise-deterministic-training` as independent evidence of the determinism direction; monitor for code release or framework integration.

### C-20261119-01 · Sampler choice under high temperature: top-nσ keeps long and agentic generation intact where top-p/top-k collapse

- **Class:** B
- **Date discovered:** 2026-11-19
- **Date published:** 2026-11-17
- **Source:** https://arxiv.org/abs/2611.22148 (verified; full text read at https://arxiv.org/html/2611.22148v1)
- **Organization/researchers:** Allen G. Roush, Minh Nhat Nguyen, Ravid Shwartz-Ziv, Judah Goldfeder, Sanjay Basu (NeurIPS 2026, Evaluations and Datasets track, per the arXiv comment)
- **Category:** inference
- **What changed:** A large evaluation (29 open-weight models, 1B-122B, ten benchmarks plus needle retrieval to 1M tokens, including SWE-bench Verified coding agents) argues that the truncation rule, not the model, decides whether long generation survives raised temperature, and identifies the order of temperature vs truncation in serving engines as the mechanism.
- **Technical summary:** Samplers whose candidate set stays small as temperature rises (top-nσ: keep logits within n standard deviations of the max logit, Tang et al. https://arxiv.org/abs/2411.07641) keep quality; samplers that admit the low-probability tail (top-p, top-k, Top-H, pure sampling) fall apart. Results (authors): at T=2, coding agents with top-p/top-k/Top-H resolve 0 of 100 GitHub issues, top-nσ 46-62 on three Qwen3.5 models; median retention of the T=1 score 100% for top-nσ vs 26% for top-p; in one Llama-3.1-8B story the top-p pool grows from 4 to 51,845 tokens from T=1 to T=2 while top-nσ stays at 3; typical pool size predicts judged quality across 133 setting x temperature cells (Spearman rho = -0.77). The paper states that HF, vLLM and SGLang apply temperature before truncation by default and llama.cpp the opposite; choosing the candidate set at T=1 and applying temperature afterwards ("temperature-last") stops the collapse for min-p, Top-H and P-less. Stated limits: one primary LLM judge (gpt-oss-120b; a second judge agrees rho = 0.92 on cell means), top-nσ still loops in 71% of forced 16K generations at T=2, Gemma-4-31B shows almost no separation (attributed to soft-capped logits, not isolated), and a script issue excluded some top-nσ settings from the sweep correlation. Experiments ran through a custom vLLM logits processor.
- **Why it might matter:** The course teaches temperature, top-k and top-p from scratch on a 5-logit vector (06.3). The ordering point (temperature before vs after truncation) and the "candidate-set size" diagnostic are small, hand-checkable extensions of exactly that lesson, and they matter for RL rollouts and agents that sample at T >= 1.
- **Evidence of adoption:** top-nσ is a built-in llama.cpp sampler (`--top-nsigma`, in the default sampler chain; verified in https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md). Not found in vLLM `sampling_params.py`, SGLang `sampling_params.py` or Transformers `logits_process.py` (checked today; all three ship min-p). Single evaluation paper; no independent reproduction yet.
- **Major organizations using it:** none known beyond llama.cpp.
- **Open-source implementation:** https://github.com/Hellisotherpeople/Long-Context-Generation-Is-a-Sampling-Problem-8- (verified exists via web fetch; not run)
- **Paper:** https://arxiv.org/abs/2611.22148 ; top-nσ: https://arxiv.org/abs/2411.07641 (verified) ; min-p: https://arxiv.org/abs/2407.01082 (verified)
- **Code:** https://github.com/Hellisotherpeople/Long-Context-Generation-Is-a-Sampling-Problem-8-
- **Relationship to existing course material:** new (lookup "top-nsigma", "min-p": no matches); lessons/module-06/lesson-03.md teaches greedy/temperature/top-k/top-p.
- **Potential course lesson:** dated extension of 06.3: "Truncation order and candidate-set size: min-p and top-nσ", worked by hand on the same 5-logit vector at T=1 and T=2, with a test that top-nσ's set is temperature-invariant when temperature is applied after truncation.
- **Confidence:** medium
- **Recommendation:** monitor; review for a short 06.3 extension if a second group reproduces the high-temperature result or vLLM/SGLang/Transformers add top-nσ or a temperature-last option.

### C-20261121-01 · Canon layers at 1.3-8B / 1-2T tokens: a residual 4-tap causal convolution saves ~28-30% of pretraining tokens and closes the gap between linear-attention variants

- **Class:** B
- **Date discovered:** 2026-11-21
- **Date published:** 2026-11-12 (arXiv v1; announced in the 20 Nov listing)
- **Source:** https://arxiv.org/abs/2611.24963 (verified; full text read at https://arxiv.org/html/2611.24963v1, first ~55%)
- **Organization/researchers:** Zeyuan Allen-Zhu ("Physics of Language Models", Part 4.2); the paper states all experiments were run at FAIR at Meta. Single author.
- **Category:** architecture
- **What changed:** The Canon layer from Physics of LMs Part 4.1 (tested at <100M parameters on synthetic tasks) is now tested in 64 from-scratch pretraining runs at 1.3B, 3B and 8B on 1-2T Nemotron-CC tokens (Llama vs LlamaCanon, plus Mamba2 / GLA5 / GDN2 with and without Canon; ~4x10^4 H200 GPU-hours per 8B run, per the paper).
- **Technical summary:** A Canon layer is a residual per-channel causal convolution over the current and three previous tokens, h'_t = h_t + conv1d([h_t, h_{t-1}, h_{t-2}, h_{t-3}]), inserted at up to four points per block (before attention, after the QKV projections, before the MLP, inside the MLP), +0.07-0.15% parameters and ~2-3% step time with a preliminary fused kernel (authors). Results (authors): at 8B/1T LlamaCanon reaches vanilla Llama's final MMLU with ~28% fewer tokens (MMLU 60.6 -> 62.4, lm-eval 66.9 -> 67.7); at academic scale (1.3B/100B) architectural differences largely vanish after tuning. With Canon, plain gated linear attention matches Gated DeltaNet (within 0.3 points at every size) and beats Mamba2, i.e. much of GDN/Mamba2's extra machinery may act as local token mixing. Standard benchmarks hide weaknesses: linear models score 72-98% on retrieval across only 500 distractor tokens (Transformers >= 99.5%), and no model reliably beats chance on a seven-sentence two-hop question. Stated limits: three sizes (no scaling law), learning-rate sweeps instead of repeated seeds, one data mix, preliminary kernels.
- **Why it might matter:** Lesson 12.5 already states (PUBLICLY DOCUMENTED) that Gated DeltaNet applies a short causal convolution before its projections, while `code/src/llmre/attention/gated_deltanet.py` deliberately omits it. This paper gives a measured reason why that small component matters and a clean, hand-checkable mechanism (a 4-tap depthwise conv) that could be taught in a few lines. Its methodological point (architecture comparisons at 1B/100B can be uninformative; small synthetic probes predicted the large-scale ordering) is also useful for module 03/12 experiment design.
- **Evidence of adoption:** Code, configs and evaluation results are public: https://github.com/facebookresearch/PhysicsLM4 (verified via web fetch; README says it moved to github.com/zhuzeyuan/PhysicsLM4; Lingua-based training code and a HF LlamaCanon model; not run). The paper notes that two large open MoE models (Inkling, Marin 535B-A23B) add short convolutions in related positions, without naming Canon as the source (authors' statement; not independently verified today). Short causal convs are already standard inside Mamba2 / Gated DeltaNet / KDA and implemented in vLLM for those layers. No independent reproduction of the token-saving claim found.
- **Major organizations using it:** Meta FAIR (research runs only); no production model confirmed to use Canon layers.
- **Open-source implementation:** https://github.com/facebookresearch/PhysicsLM4
- **Paper:** https://arxiv.org/abs/2611.24963
- **Code:** https://github.com/facebookresearch/PhysicsLM4
- **Relationship to existing course material:** new topic (lookup "Canon", "short convolution": only 12.5 and `code/src/llmre/attention/gated_deltanet.py`, which notes "no short causal convolution"). Related: `accepted/hybrid-linear-attention-gated-deltanet.md`, `deferred/gated-deltanet-gating-refinements.md` (the GLA5/GDN2 low-rank output gates are a gating refinement of the same family).
- **Potential course lesson:** dated extension of 12.5: "The short causal convolution: Canon layers", with a hand-worked 4-tap example on a 3-token sequence, an optional `short_conv` flag in `gated_deltanet.py` plus a test of causality, and a note on the academic-scale caveat.
- **Confidence:** medium
- **Recommendation:** monitor; review for a short 12.5 extension if an independent group reproduces the token savings or a released model/open stack adopts Canon-style convolutions in attention blocks by name.
