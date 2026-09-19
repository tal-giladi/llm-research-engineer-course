# Research Engineer Readiness Matrix

The goal of this course is not to finish the website. It is to become able to pick up a new LLM
training paper and answer: *"What are they doing, why, how would I implement it, how would I
test it, and what experiment would I run next?"*

Rate yourself **Beginner / Intermediate / Advanced** on each skill. A skill is *Advanced* only
when you can **explain → derive → implement → test → benchmark → debug → relate-to-a-paper** it
from memory. Re-rate after each pass through the relevant module.

| Skill | Where it's built | B / I / A |
|---|---|---|
| PyTorch (tensors, autograd, `nn.Module`) | M0 | ☐ ☐ ☐ |
| Transformer internals (attention, blocks) | M5, M6 | ☐ ☐ ☐ |
| Backpropagation (by hand + autograd) | M2 | ☐ ☐ ☐ |
| Optimization (SGD→AdamW, schedules) | M3 | ☐ ☐ ☐ |
| LLM pretraining (end-to-end loop) | M6, M7 | ☐ ☐ ☐ |
| Data engineering (filter, dedup, contamination) | M11 | ☐ ☐ ☐ |
| GPU optimization (memory, profiling) | M8 | ☐ ☐ ☐ |
| Triton (writing a kernel) | M8 | ☐ ☐ ☐ |
| Distributed training (DDP, all-reduce) | M9 | ☐ ☐ ☐ |
| FSDP / ZeRO | M9 | ☐ ☐ ☐ |
| Megatron / tensor-pipeline parallelism | M9 | ☐ ☐ ☐ |
| Scaling laws (FLOPs, Chinchilla) | M10 | ☐ ☐ ☐ |
| Evaluation (perplexity→task eval, harness) | M13 | ☐ ☐ ☐ |
| SFT (chat templates, loss masking) | M14 | ☐ ☐ ☐ |
| LoRA / QLoRA | M14 | ☐ ☐ ☐ |
| DPO (derivation + impl) | M15 | ☐ ☐ ☐ |
| RLHF (reward model + pipeline) | M15 | ☐ ☐ ☐ |
| PPO | M15 | ☐ ☐ ☐ |
| GRPO | M16 | ☐ ☐ ☐ |
| RLVR (verifiable rewards) | M16 | ☐ ☐ ☐ |
| Reasoning training (CoT→R1) | M16, M17 | ☐ ☐ ☐ |
| Mixture of Experts | M12 | ☐ ☐ ☐ |
| Tool use (schemas, function calling) | M18 | ☐ ☐ ☐ |
| Agents (ReAct, SWE-agent) | M19 | ☐ ☐ ☐ |
| Experiment design (ablations, significance) | M19 | ☐ ☐ ☐ |
| Paper reproduction | M10, M15, M17 | ☐ ☐ ☐ |
| Research writing (structured reports) | M19 | ☐ ☐ ☐ |

## Self-check: can you answer these unaided?

- Trace a single token from string → id → embedding → one Transformer block → logits → loss.
- Derive the AdamW update and explain why weight decay is decoupled.
- Explain why FlashAttention is faster without changing the math.
- Compute the compute-optimal token count for a 1B model (Chinchilla).
- Derive the DPO loss from the RLHF objective.
- Explain why verifiable rewards make reasoning RL tractable, and what GRPO removes from PPO.
- Given a training run that NaNs at step 1200, list the five things you check first.
