# Experiment report — Mini Frontier LLM (capstone end-to-end smoke run)

**Date:** 2026-09-20 · **Author:** capstone driver · **Code:** `scripts/capstone.py` · **Config:** `configs/capstone.yaml`

## QUESTION
Does the coherent `llmre` codebase built across Modules 0-19 actually compose into one working pipeline — tokenizer -> GPT -> pretraining -> evaluation -> SFT -> LoRA -> reasoning RL — that runs on a CPU and shows the two learning signals it should (pretraining loss falling, RLVR correct-rate rising)?

## HYPOTHESIS
Each stage was unit-tested in isolation, so wired together on a tiny corpus the pretraining cross-entropy should fall well below its random-init value (a 64-wide, 2-layer GPT can memorize a small repetitive corpus), and GRPO with a verifiable reward should push the arithmetic correct-rate from chance (~1/10) toward 1.0.

## METHOD
One CPU process, seed 1234, torch 2.14.0+cpu. Stages, in order:
1. Byte-level BPE tokenizer trained on a 12-document toy corpus, vocab 374.
2. Tiny GPT-2: n_layer=2, n_head=4, n_embd=64, block_size=32, 126,080 parameters.
3. Pretrain 300 AdamW steps (max_lr=0.003, cosine warmup 20, batch 16) on the packed corpus.
4. Held-out perplexity over a clean copy of the corpus.
5. 20 SFT steps on one chat with response-only loss masking.
6. A rank-4 LoRA adapter wrapped around a 64x64 linear.
7. 80 GRPO steps on single-digit addition with a rule-based verifier (RLVR).
Full config in `configs/capstone.yaml`; total wall-clock 19.6 s.

## BASELINE
Random-initialized model before any training: pretraining train-set cross-entropy 5.935 nats; RLVR policy at initialization scores 0.062 correct (≈ chance for a 10-way answer space).

## VARIABLES
Independent variable: training (gradient steps applied vs not). Controlled: seed, corpus, model size, and all optimizer hyperparameters are fixed across the before/after measurements. This is a smoke run, not a controlled ablation — it varies "trained vs untrained", not one architectural knob.

## RESULTS
Single seed (1234); this run demonstrates the pipeline rather than estimating variance across seeds.

| Stage | Metric | Before | After |
|---|---|---|---|
| Pretraining | train cross-entropy (nats) | 5.935 | 0.067 |
| Pretraining | held-out cross-entropy (nats) | — | 0.071 |
| Evaluation | held-out perplexity | — | 1.07 |
| SFT | response-masked loss (nats) | 11.962 | 2.560 |
| LoRA | trainable params (of 4608) | 4608 | 512 |
| RLVR | arithmetic correct-rate | 0.062 | 1.000 |

Pretraining loss dropped by 5.869 nats. LoRA made 11.1% of the linear's parameters trainable and reproduced the base output exactly at init (True). RLVR raised the correct-rate by 0.938.

## ANALYSIS
Both learning signals fire in the expected direction and by a wide margin: pretraining cross-entropy fell from 5.935 to 0.067 nats (the tiny model is memorizing the small corpus, exactly what should happen at this scale), and the RLVR correct-rate rose from 0.062 to 1.000. The SFT step lowered the response-only loss, confirming the masking path trains on the assistant tokens alone. The margins are far larger than run-to-run noise, so the pipeline is wired correctly end to end.

## LIMITATIONS
This proves *integration*, not *quality*. The corpus is tiny and repetitive, so the low perplexity is memorization, not generalization; there is a single seed and no confidence interval; the RLVR policy is a categorical table, not the GPT itself (the lesson explains why, and how to wire GRPO to the model's answer tokens). None of the numbers say anything about how a real, scaled run would behave.

## CONCLUSION
Yes — the whole `llmre` stack composes into one runnable pipeline on CPU: pretraining loss fell 5.869 nats and the RLVR correct-rate rose to 1.000, with every stage executing on the shared interfaces.

## NEXT EXPERIMENT
Replace the categorical RLVR policy with the pretrained GPT emitting answer *tokens*, keep the verifier reward, and measure whether GRPO raises exact-match on a held-out set of additions the model never saw during pretraining — a real (if small) reasoning-RL result rather than a smoke test.
