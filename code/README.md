# `llmre` — the course codebase

One coherent repository built up across the 21 modules. Each stage extends the previous one;
nothing here is a throwaway toy script.

```text
src/llmre/
    tokenizer/    # M4  — BPE tokenizer from scratch
    attention/    # M5, M8, M12 — attention, flash-style, RoPE/GQA
    model/        # M5, M6, M12 — Transformer block, GPT-2, Llama/MoE variants
    optim/        # M3  — SGD/Adam/AdamW, schedules, clipping
    training/     # M7  — training loop, grad accumulation, checkpointing, metrics
    distributed/  # M9  — DDP/ZeRO-style sharding
    data/         # M4, M11 — data loading, packing, filtering, dedup
    evaluation/   # M1, M13 — loss/perplexity, few-shot, harness
    sft/          # M14 — chat templates, loss masking, LoRA
    preference/   # M15 — reward model, DPO
    rl/           # M15, M16 — PPO, GRPO
    reasoning/    # M16, M17 — RLVR loop, verifiers
    tools/        # M18 — tool schemas, function calling
    agents/       # M19 — ReAct loop, mini SWE-agent
tests/            # one test per algorithm, checked against a reference
experiments/      # scaling-law fits, ablations
configs/          # experiment configs (YAML/JSON)
scripts/          # entry points: train, eval, generate
notebooks/        # exploration
reports/          # structured experiment reports (the research-writing deliverable)
checkpoints/      # (gitignored) model checkpoints
```

## Setup

```bash
cd code
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Most of the course runs CPU-only on tiny models. Lessons that need a GPU say so in their
**hardware track** box, with expected runtime, memory, and GPU-hours.

## Design rules

- Every algorithm is implemented from scratch **before** the production framework is shown.
- Every public function documents tensor **shape, dtype, device** in its docstring.
- Every algorithm has a test in `tests/` comparing it to a PyTorch/reference implementation.
