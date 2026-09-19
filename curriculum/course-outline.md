# Curriculum map & dependency graph

This course is a single continuous story: every module solves a problem created by the one
before it. Use this page to answer, before any lesson, *"what must I understand first?"*

## The spine

```text
Probability ─▶ Cross-entropy ─▶ Language modeling
     │
     ▼
Tokenization ─▶ Attention ─▶ Transformer ─▶ GPT-2
     │
     ▼
Pretraining loop ─▶ GPU systems ─▶ Distributed training ─▶ Scaling ─▶ Data
     │
     ▼
Modern architectures (RoPE/RMSNorm/SwiGLU/GQA/MoE) ─▶ Evaluation
     │
     ▼
SFT ─▶ RLHF (reward model + PPO) ─▶ DPO ─▶ Reasoning RL (GRPO/RLVR)
     │
     ▼
DeepSeek-R1 ─▶ Tool use ─▶ Agents ─▶ Research engineering ─▶ Capstone
```

Two supporting chains feed the spine:

```text
Calculus ─▶ Chain rule ─▶ Gradients ─▶ Jacobians/VJPs ─▶ Backprop ─▶ Autograd
Gradient descent ─▶ SGD ─▶ Momentum ─▶ Adam ─▶ AdamW ─▶ schedules/clipping
```

## Module dependency table

| Module | Depends on | Unlocks | Codebase area |
|---|---|---|---|
| 0 · PyTorch foundations | — | everything | (whole repo) |
| 1 · Probability & LM | 0 | the training loss; evaluation | `evaluation/` |
| 2 · Gradients & backprop | 0, 1 | optimization; understanding autograd | (understanding) |
| 3 · Optimization | 2 | pretraining | `optim/` |
| 4 · Tokenization & data | 0, 1 | model input; data engineering | `tokenizer/`, `data/` |
| 5 · Transformer | 1, 2, 4 | GPT-2 | `attention/`, `model/` |
| 6 · GPT-2 | 5 | pretraining; everything downstream | `model/` |
| 7 · Pretraining infra | 3, 6 | GPU/systems; scaling | `training/` |
| 8 · GPU performance | 6, 7 | distributed; efficient attention | `attention/`, `training/` |
| 9 · Distributed training | 7, 8 | large-scale training; MoE | `distributed/` |
| 10 · Scaling laws | 7 | compute planning | `experiments/` |
| 11 · Data engineering | 4 | pretraining data; contamination in eval | `data/` |
| 12 · Modern architectures | 5, 6, 9 | Llama/DeepSeek-style models; MoE | `model/`, `attention/` |
| 13 · Evaluation | 1, 6, 11 | every experiment; alignment eval | `evaluation/` |
| 14 · SFT | 6, 13 | instruction following; RLHF/DPO | `sft/` |
| 15 · RLHF / DPO | 3, 14 | alignment; reasoning RL | `preference/`, `rl/` |
| 16 · Reasoning RL | 15 | reasoning models | `rl/`, `reasoning/` |
| 17 · Reasoning models | 16 | DeepSeek-R1 analogue | `reasoning/` |
| 18 · Tool use | 6, 14 | agents | `tools/` |
| 19 · Agents & research eng | 18 | capstone | `agents/` |
| 20 · Capstone | all | — | (whole repo) |

## Paper placement

Papers are introduced exactly where the machinery to understand them exists. The full ordered
list with verified links, prerequisites, and "read this / skip that" guides is on the
[**Paper curriculum**](../papers/index.md) page. Rough placement:

- After **6 (GPT-2)**: GPT-2, GPT-3.
- After **7–10 (systems/scaling)**: Scaling Laws, Chinchilla, Megatron-LM, ZeRO.
- After **8 (GPU)**: FlashAttention.
- After **12 (architectures)**: RoPE/RoFormer, LLaMA, Llama 3, OLMo 2, Switch Transformers, DeepSeekMoE, DeepSeek-V3.
- After **14 (SFT)**: FLAN, Self-Instruct.
- After **15 (RLHF/DPO)**: InstructGPT, Anthropic HH-RLHF, Constitutional AI, RLAIF, DPO.
- After **16–17 (reasoning)**: Chain-of-Thought, STaR, Let's Verify Step by Step, DeepSeekMath, DeepSeek-R1.
- After **18–19 (tools/agents)**: Toolformer, Gorilla, ReAct, SWE-agent.

## What "done" means

A module is complete only when you can **explain** it, **derive** the key mathematics,
**implement** it, **test** it, **benchmark** it, **debug** it, **relate** it to a paper, and
say **why** modern LLM systems use it. Track your own level on the
[Readiness matrix](readiness-matrix.md).
