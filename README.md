# LLM Research Engineer — Language Modeling from Scratch

**21 modules · ~30-paper reading curriculum · a single coherent PyTorch codebase you build up stage by stage · worked-by-hand numerical examples · from-scratch implementations before any framework**

A complete, implementation-heavy apprenticeship for an **experienced software engineer** who
wants to understand how modern frontier LLMs are actually researched, trained, evaluated,
fine-tuned, aligned, and deployed — deeply enough to read current papers, implement the
important algorithms, train real models on GPUs, debug training failures, design experiments,
and work as an **ML Research Engineer / LLM Training Engineer**.

Built on the backbone of **[Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)**
and extended substantially into modern frontier-model research (Llama 3, OLMo 2, DeepSeek-V3,
DeepSeek-R1, reasoning RL, tool use, agents).

> This is **not** a beginner PyTorch course, an API course, or a "fine-tune Llama with Hugging
> Face" tutorial. Nothing important is hidden behind a high-level library until you have first
> implemented the underlying mechanism yourself.

---

## The one continuous story

Every stage exists to solve a problem created by the previous one:

```text
Mathematics ─▶ PyTorch ─▶ Tokenization ─▶ Attention ─▶ Transformer ─▶ GPT-2
     │
     ▼
Pretraining loop ─▶ GPU systems ─▶ Distributed training ─▶ Scaling laws ─▶ Data engineering
     │
     ▼
Modern architectures (RoPE/RMSNorm/SwiGLU/GQA/MoE) ─▶ Evaluation
     │
     ▼
"It doesn't follow instructions." ─▶ SFT / LoRA
"It doesn't prefer better answers." ─▶ Reward models / RLHF / PPO / DPO
"It can't reason reliably." ─▶ Reasoning RL / GRPO / RLVR ─▶ DeepSeek-R1 case study
"It needs external information & actions." ─▶ Tool use ─▶ Agents
     │
     ▼
Research engineering (experiments, ablations, reproducibility) ─▶ Capstone: a mini frontier LLM
```

See the full [**Curriculum map & dependency graph**](curriculum/course-outline.md).

---

## How every lesson is built

For each important concept, in order:

1. **Intuition** — what is happening, in plain English.
2. **Mathematics** — the equation, with *every symbol named*.
3. **Numerical example** — the same thing with tiny numbers, worked by hand.
4. **Tensor shapes** — shape, dtype, device, and how the shape changes at every step.
5. **From-scratch implementation** — the mechanism in plain PyTorch.
6. **Under the hood** — what PyTorch / the GPU / the optimizer is actually doing, and its
   compute and memory cost.

Then, as the material demands: a **unit test**, an **exercise** (hint → stronger hint →
revealable solution, never spoiled), the **research paper** it comes from with a *what to read /
what to skip* guide, **common mistakes**, a **debugging exercise**, and a **Check yourself** quiz.

Every lesson opens with **Prerequisites / You will learn / Why this matters**, and compute
tasks carry a **hardware track** (minimum & recommended GPU, expected runtime, memory, whether
CPU-only development is possible).

---

## The codebase

The course builds **one** repository, not throwaway scripts. It lives in [`code/`](code/):

```text
code/src/llmre/
    tokenizer/  model/  attention/  optim/  training/  distributed/
    data/  evaluation/  sft/  preference/  rl/  reasoning/  tools/  agents/
code/{tests, experiments, configs, scripts, notebooks, reports, checkpoints}
```

Each stage extends the previous implementation. Every algorithm ships with a test that checks
it against a PyTorch/reference implementation.

```bash
cd code && pip install -e . && pytest
```

---

## Modules

| # | Module | Covers |
|---|--------|--------|
| 0 | [ML / PyTorch foundations](lessons/module-00/lesson-01.md) | tensors, autograd, devices, dtypes, the training-loop skeleton |
| 1 | [Probability, information theory & language modeling](lessons/module-01/lesson-01.md) | distributions, entropy, cross-entropy, KL, perplexity, the LM objective |
| 2 | [Derivatives, gradients & backpropagation](lessons/module-02/lesson-01.md) | chain rule, Jacobians, VJPs, backprop, autograd internals |
| 3 | [Optimization](lessons/module-03/lesson-01.md) | SGD→Adam→AdamW, weight decay, LR schedules, warmup, clipping |
| 4 | [Tokenization & data representation](lessons/module-04/lesson-01.md) | bytes, BPE from scratch, vocab, special tokens, packing |
| 5 | [Transformer architecture from scratch](lessons/module-05/lesson-01.md) | embeddings, attention, causal mask, MHA, MLP, residual, norm |
| 6 | [GPT-2 from scratch](lessons/module-06/lesson-01.md) | the full GPT-2 model, weight tying, init, generation, sampling |
| 7 | [Pretraining infrastructure](lessons/module-07/lesson-01.md) | batching, grad accumulation, checkpointing, logging, throughput/MFU |
| 8 | [GPU performance & memory](lessons/module-08/lesson-01.md) | memory hierarchy, arithmetic intensity, profiling, Triton, FlashAttention |
| 9 | [Distributed training](lessons/module-09/lesson-01.md) | DDP, all-reduce, ZeRO/FSDP, tensor/pipeline/expert parallelism |
| 10 | [Scaling laws & compute-optimal training](lessons/module-10/lesson-01.md) | FLOPs, Kaplan vs Chinchilla, fitting scaling curves |
| 11 | [LLM data engineering](lessons/module-11/lesson-01.md) | Common Crawl, filtering, dedup/MinHash, contamination, mixing |
| 12 | [Modern architectures](lessons/module-12/lesson-01.md) | RoPE, RMSNorm, SwiGLU, MQA/GQA, MoE, DeepSeek-style routing |
| 13 | [Evaluation & experimentation](lessons/module-13/lesson-01.md) | perplexity, few-shot, contamination, calibration, eval harness |
| 14 | [Supervised fine-tuning](lessons/module-14/lesson-01.md) | chat templates, loss masking, packing, LoRA & QLoRA from scratch |
| 15 | [Preference learning / RLHF / DPO](lessons/module-15/lesson-01.md) | Bradley-Terry, reward models, PPO, KL penalty, DPO derivation |
| 16 | [RL for reasoning](lessons/module-16/lesson-01.md) | policy gradient, advantage, GRPO, RLVR, verifiable rewards |
| 17 | [Modern reasoning models](lessons/module-17/lesson-01.md) | CoT, self-consistency, process vs outcome reward, DeepSeek-R1 |
| 18 | [Tool use](lessons/module-18/lesson-01.md) | tool schemas, function calling, structured output, retries |
| 19 | [Agents & research engineering](lessons/module-19/lesson-01.md) | ReAct, planning loops, a mini SWE-agent, experiment design |
| 20 | [Capstone: a mini frontier LLM](lessons/module-20/lesson-01.md) | the full pipeline end to end, ablations, a research report |

**Papers:** [Reading curriculum (~30 papers, verified links)](papers/index.md) ·
**Reference:** [Math & ML cheat sheet](assets/math-cheatsheet.md) · [PyTorch/systems cheat sheet](assets/pytorch-cheatsheet.md) · [Glossary](assets/glossary.md) · [Research Engineer Readiness Matrix](curriculum/readiness-matrix.md)

---

## How to read it locally

Plain [docsify](https://docsify.js.org) — no build step. From this folder:

```bash
python -m http.server 8080
```

Then open <http://localhost:8080>. The Markdown is written to be legible as source too.
