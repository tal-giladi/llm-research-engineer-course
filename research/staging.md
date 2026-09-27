# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20260926-01 · Qwen3.8-Flash-Next: GDN + sparse-attention hybrid as a Qwen4 architecture preview

- **Class:** B
- **Date discovered:** 2026-09-26
- **Date published:** 2026-08-26
- **Source:** https://github.com/QwenLM/Qwen3.8-Flash-Next (model card/README); technical report https://arxiv.org/abs/2608.30320 "On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability"
- **Organization/researchers:** Qwen Team, Alibaba Group
- **Category:** attention | architecture | MoE | long-context
- **What changed:** Released as an explicit early preview of the Qwen4 architecture: a layer-wise hybrid of Gated DeltaNet (linear-attention-style token mixing that compresses history into a recurrent state) and full/sparse attention (3 GDN layers : 1 attention layer, repeated across the stack), with the periodic full-attention layers swapped for "Qwen Sparse Attention" (a lightweight indexer scoring context at micro-block granularity) during continued pretraining for long context. Also introduces a 4-branch gated residual stream and an N-gram lookup embedding table (51B extra params, offloadable to CPU RAM) alongside a 125B/6B-active MoE backbone, trained with a refined Muon optimizer.
- **Technical summary:** This is a concrete, shipped (open-weight) example of the "hybrid linear + sparse attention" design space the course's attention module (12.1–12.3) and MoE lesson (12.4) currently only discuss in the abstract/RoPE-and-GQA framing. GDN gives O(1) per-step decode state instead of a growing KV cache for most layers; the sparse-attention layers keep some full-context capability at reduced cost. Reported as maintaining accuracy while cutting inference cost relative to a dense-attention Qwen3-class model of similar capability (company claim; independent benchmarks not yet available).
- **Why it might matter:** If GDN/hybrid-linear-attention designs become standard in the next generation of open frontier MoE models (this is explicitly framed as a Qwen4 preview), "why not just always use full attention" becomes a live design question the course's attention lessons don't yet answer with a real production example.
- **Evidence of adoption:** (company claim) — single lab, but the lab is a major open-weight model provider and the architecture is described as the basis for their next full-generation model line, not a research toy.
- **Major organizations using it:** Qwen/Alibaba only, so far.
- **Open-source implementation:** https://github.com/QwenLM/Qwen3.8-Flash-Next ; weights on https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- **Paper:** https://arxiv.org/abs/2608.30320
- **Code:** https://github.com/QwenLM/Qwen3.8-Flash-Next
- **Relationship to existing course material:** new — lessons/module-12/lesson-01 (RoPE/attention variants) through lesson-04 (MoE) cover standard MHA/GQA + MoE but not linear/hybrid attention (GDN, Mamba-style recurrent state) or learned sparse-attention indexers.
- **Potential course lesson:** module-12 extension: a "hybrid attention in production" case study alongside the existing attention lesson, or a short new lesson on linear/recurrent attention (GDN) as a design-space comparison to GQA — not urgent since this is a preview architecture, not yet Qwen4 itself.
- **Confidence:** medium
- **Recommendation:** monitor — revisit at Qwen4's full release or if another major lab (DeepSeek, Meta, Mistral) ships a similar hybrid-attention architecture, which would be adoption evidence justifying ADD.

### C-20260926-02 · FFD ("Faster Than Flash"): training-free sparse-attention decoding kernel for long-context inference

- **Class:** B
- **Date discovered:** 2026-09-26
- **Date published:** 2026-08-31
- **Source:** https://arxiv.org/abs/2609.00097 "Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding"
- **Organization/researchers:** academic (per arXiv listing; not a named industry lab)
- **Category:** efficiency | attention | long-context | inference
- **What changed:** Proposes a fused, hardware-algorithm co-designed decoding kernel ("Faster Flash Decoding") that dynamically filters which KV blocks participate in attention per decode step (a "top-δ" adaptive-sparsity rule) using low-bit quantized content-aware scanning instead of external metadata indices, claiming up to 11.6x kernel-level speedup and 2.37x end-to-end throughput at 256K context, training-free and drop-in.
- **Technical summary:** Directly extends the course's module-08 FlashAttention/online-softmax lesson: same memory-wall motivation, but exploits empirical attention sparsity at decode time rather than just fusing the exact computation. Relevant comparison point for teaching "when is exact full attention actually necessary at inference time."
- **Why it might matter:** Long-context decoding throughput is a standing pain point; if this or a similar sparse-decoding technique gets merged into vLLM/SGLang/TensorRT-LLM it would be genuine adoption evidence for a course update.
- **Evidence of adoption:** none yet — single paper, code released by the authors but no evidence of merge into a major serving stack.
- **Major organizations using it:** none known.
- **Open-source implementation:** authors' repo linked from the paper (not independently verified beyond the arXiv listing page).
- **Paper:** https://arxiv.org/abs/2609.00097
- **Code:** see paper (author repo; unverified)
- **Relationship to existing course material:** lessons/module-08/lesson-04 (FlashAttention/online softmax); lessons/module-12/lesson-03 (KV cache).
- **Potential course lesson:** none yet — too early; would be a "what production serving stacks do beyond FlashAttention" sidebar if adopted.
- **Confidence:** low
- **Recommendation:** monitor only — classic single-paper efficiency claim, needs independent reproduction or stack adoption before it's teachable as more than "here's an idea in the literature."

### C-20260927-01 · Superposition Linearity Hypothesis: LLMs can generate two coherent continuations from one forward pass

- **Class:** B
- **Date discovered:** 2026-09-27
- **Date published:** 2026-09-24
- **Source:** https://arxiv.org/abs/2609.29845 "Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs"
- **Organization/researchers:** Pavel Tikhonov, Anton Korznikov, Matvey Mikhalchuk, Nikita Dragunov, Temurbek Rahmatullaev, Polina Druzhinina, Anton Razzhigaev, Ivan Oseledets, Elena Tutubalina (academic; affiliations not stated in abstract, group includes AIRI/Skoltech-associated authors per prior publications)
- **Category:** interpretability
- **What changed:** Proposes and tests a "Superposition Linearity Hypothesis": when inputs from two distinct text streams are linearly combined (averaged) at the input, the model's output logits are approximately a superposition (linear combination) of the two streams' individual next-token distributions. They argue this is an intrinsic architectural property of transformers rather than a training artifact, show it weakens over the course of pretraining, show it can be restored by lightweight fine-tuning, and build a guided-decoding procedure that generates two coherent continuations simultaneously from a single forward pass.
- **Technical summary:** Directly relevant to how the course could teach the residual stream / linear representation hypothesis (superposition of features, à la Anthropic's toy-models-of-superposition line of interpretability work) — but here the claim is about superposition of *outputs/tasks* under literal input averaging, not the usual polysemanticity-of-features claim. If it holds up, it's a clean numerical demonstration of transformer (near-)linearity that could motivate a from-scratch interpretability exercise (feed two prompts' embeddings averaged, show the logit distribution decomposes).
- **Why it might matter:** A single crisp, testable structural claim about how transformers process information, with a from-scratch demo an instructor could reproduce cheaply (small model, few forward passes) — good "check yourself" exercise material if reproduced.
- **Evidence of adoption:** none — single paper, no independent reproduction, not company-affiliated.
- **Major organizations using it:** none.
- **Open-source implementation:** not stated in abstract; not verified beyond arXiv listing.
- **Paper:** https://arxiv.org/abs/2609.29845
- **Code:** none confirmed
- **Relationship to existing course material:** new — course has no dedicated interpretability module/lesson yet (grep of `curriculum/course-outline.md` and `lessons/` finds no "interpretability" hits); closest existing material is the attention/residual-stream mechanics in module-05/06.
- **Potential course lesson:** if reproduced independently, candidate seed for a first interpretability lesson (e.g. new module or module-06 extension) on linear structure in the residual stream, using this as a hands-on reproducible demo.
- **Confidence:** low
- **Recommendation:** monitor — wait for independent reproduction or citation uptake before considering; interesting but a single unreplicated claim, and the course has no interpretability module to hang it on yet (that's a bigger prerequisite decision for the weekly/curriculum review, not this candidate alone).

### C-20260927-02 · Encoded but Not Decoded: three-level gap between what LLMs represent and what they output on syntax

- **Class:** B
- **Date discovered:** 2026-09-27
- **Date published:** 2026-09-24
- **Source:** https://arxiv.org/abs/2609.29848 "Encoded but Not Decoded: Layer-Localized Evidence for a Three-Level Gap in LLM Syntax"
- **Organization/researchers:** Zhenyan Lu, He Wang, Xiaohui Huang (academic); accepted at AACL-IJCNLP 2026
- **Category:** interpretability | evaluation
- **What changed:** Introduces a three-level evaluation framework (probe recoverability → LM-head readout → behavioral deployment) applied to a trilingual (English/Chinese/German) control-dependency syntax benchmark across 7 models. Finds a consistent ordering probe > LM-head > behavior — i.e. probing classifiers systematically overstate what a model actually deploys in its outputs, while raw behavioral testing understates what's linearly encoded internally. Activation patching localizes the gap to specific layers, with instruction-tuned models showing ~10-layer shifts between where a probe recovers the answer and where the LM head reads it out.
- **Technical summary:** A direct, sharply stated methodological warning for anyone using probing classifiers to claim a model "knows" or "understands" something: probe accuracy is not a proxy for behavior, and the size of that gap is itself measurable and layer-localized. Complements the superposition paper above (C-20260927-01) as another data point on linear structure vs. actual model behavior.
- **Why it might matter:** If this probe-vs-behavior gap generalizes beyond syntax (the authors only test control-dependency structures), it's a caution any interpretability teaching material should carry: "probing shows what's encoded, not what's used." Useful methodological point even without a dedicated interpretability module yet.
- **Evidence of adoption:** none — single paper, first venue acceptance (AACL-IJCNLP 2026), no independent reproduction.
- **Major organizations using it:** none — academic only.
- **Open-source implementation:** not confirmed in abstract.
- **Paper:** https://arxiv.org/abs/2609.29848
- **Code:** not confirmed
- **Relationship to existing course material:** new — same gap as above (no interpretability module yet).
- **Potential course lesson:** pairs with C-20260927-01 as supporting material for a future interpretability module's methodology section ("what probing can and can't tell you"), not a standalone lesson.
- **Confidence:** low
- **Recommendation:** monitor — single paper, narrow task domain (control-dependency syntax only); revisit if the three-level-gap finding is reproduced on other tasks or cited as a methodological standard.
