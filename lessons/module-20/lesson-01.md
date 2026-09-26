# 20 · Capstone: a mini frontier LLM

<div class="prereq">
<p><strong>Prerequisites:</strong> the whole course. Concretely, you should be comfortable with the pieces you are about to wire together: <a href="#/lessons/module-04/lesson-02">byte-level BPE (04.2)</a>, <a href="#/lessons/module-06/lesson-01">assembling GPT-2 (06.1)</a>, <a href="#/lessons/module-07/lesson-01">the pretraining loop (07.1)</a>, <a href="#/lessons/module-14/lesson-02">SFT loss masking (14.2)</a>, <a href="#/lessons/module-16/lesson-02">RLVR (16.2)</a>, and <a href="#/lessons/module-19/lesson-02">experiment design (19.2)</a>. See the <a href="#/curriculum/course-outline">curriculum map</a> for the full dependency graph.</p>
<p><strong>You will learn:</strong> how every module you built snaps together into <em>one</em> project — the "Mini Frontier LLM" — that goes from raw text to a tokenizer, a GPT, a pretrained model, an evaluation number, an instruction-tuned model, a preference-aligned model, and a reasoning policy trained with verifiable rewards. You will run a tiny end-to-end version on your CPU, watch the loss fall and the correct-rate climb, and see exactly how to scale each stage up.</p>
<p><strong>Why this matters for ML:</strong> a research engineer is not judged on any single component but on the ability to make the <em>whole pipeline</em> run, produce a number, and be reproduced. This capstone is that pipeline, in miniature, on hardware you already own.</p>
</div>

## 1. What we are building

Every earlier module solved a problem created by the one before it. Read top to bottom, they are a single machine:

```text
raw text ─▶ BPE tokenizer ─▶ id stream ─▶ tiny GPT
   │                                          │
   ▼                                          ▼
 data pipeline ◀────────────────────── pretraining loop (AdamW + cosine LR + clip)
   │                                          │
   ▼                                          ▼
 held-out perplexity ◀───────────────── evaluation harness
   │
   ▼
 SFT (response-masked) ─▶ LoRA ─▶ preference data ─▶ DPO / reward-model+PPO
   │
   ▼
 GRPO + verifiable reward (RLVR) ─▶ reasoning (CoT / R1) ─▶ tool use ─▶ agent
   │
   ▼
 ablations ─▶ a written experiment report
```

The whole codebase lives under `code/src/llmre/` as one coherent package. This lesson is a guided tour of that package as a *project*: for each stage, a short paragraph, the exact `llmre` entry point, and what to look at. Then a driver — `code/scripts/capstone.py` — runs a tiny version of all of it in under 20 seconds on a CPU, and writes a filled-in experiment report.

<div class="callout key"><p>The capstone is not new code you must write from scratch — it is the code you already wrote across Modules 0–19, assembled and run as one pipeline. The point is to <em>see it work end to end</em> and know how each stage scales.</p></div>

## 2. Stage by stage

For each stage: what it does, the `llmre` entry point, and what to read.

### (1) Train a BPE tokenizer

Turn raw text into a bounded vocabulary of subword ids, starting from the 256 raw byte values and merging the most frequent adjacent pair repeatedly. Nothing is ever out-of-vocabulary, and `decode(encode(s)) == s` for any string.

- **Entry point:** `llmre.tokenizer.bpe.BPETokenizer` — `.train(text, vocab_size)`, `.encode`, `.decode`, `.vocab_size`.
- **Look at:** how `merges` (an ordered dict of `(a, b) -> new_id`) is applied *in learned order* inside `encode`. That ordering is the whole subtlety. See [04.2 · Byte-level BPE](lessons/module-04/lesson-02.md).

### (2) Build a GPT

A decoder-only transformer: token + positional embeddings, a stack of pre-norm blocks (causal multi-head attention + MLP), a final norm, and a tied `lm_head`. The `GPTConfig` dataclass fixes every tensor shape.

- **Entry point:** `llmre.model.gpt.GPT`, `llmre.model.config.GPTConfig`. `forward(idx, targets)` returns `(logits, loss)`; `generate(...)` samples.
- **Look at:** weight tying (`transformer.wte.weight is lm_head.weight`) and the scaled residual init `0.02 / sqrt(2 * n_layer)`. See [06.1 · Assembling GPT-2](lessons/module-06/lesson-01.md).

### (3) Pretrain on a toy corpus

Next-token prediction: for each position, minimize the cross-entropy of the true next token. The loop writes the learning rate from a cosine-warmup schedule, runs forward/backward, clips the global gradient norm, and takes an AdamW step.

- **Entry point:** `llmre.training.loop.train` / `TrainConfig`, built on `llmre.optim.adamw.AdamW`, `llmre.optim.schedule.cosine_warmup_lr`, `llmre.optim.clip.clip_grad_norm_`.
- **Look at:** the six-line inner step in `train` and how gradient accumulation makes a large effective batch fit in small memory. See [07.1 · The pretraining loop](lessons/module-07/lesson-01.md).

### (4) The data pipeline

Encode every document once, concatenate into a single id stream separated by an end-of-text id, then cut random fixed-length `(x, y)` windows where `y` is `x` shifted by one. At real scale this is also where you *filter, deduplicate, and decontaminate* the corpus.

- **Entry point:** `llmre.data.loader.pack_documents`, `llmre.data.loader.get_batch`. Data-quality tooling: `llmre.data.quality`, `llmre.data.minhash`, `llmre.data.contamination`.
- **Look at:** why the target is just the input shifted left by one — that single shift is the entire supervision signal. See [04.3 · The data pipeline](lessons/module-04/lesson-03.md) and [Module 11 · Data engineering](lessons/module-11/lesson-01.md).

### (5) GPU and profiling notes

On a CPU the toy run is instant, but on a GPU you must reason about *where the time and memory go*. Attention materializes a $T \times T$ matrix per head (memory grows as $T^2$); FlashAttention computes the same result without ever storing it. Throughput is tracked with the 6N FLOPs rule and MFU.

- **Entry point:** `llmre.training.metrics` — `model_flops_per_token`, `tokens_per_step`, `mfu`. Tiled attention: `llmre.attention.flash.flash_attention_reference`. Benchmark: `code/scripts/bench_attention.py`.
- **Look at:** the roofline / memory-hierarchy story and mixed precision. See [Module 08 · GPU performance](lessons/module-08/lesson-01.md).

### (6) A scaling mini-experiment

Scaling laws say loss falls predictably with model size, data, and compute. You can see the *shape* of that curve even on CPU by sweeping `n_embd` over a few sizes and plotting final loss.

- **Entry point:** `code/experiments/scaling_sweep.py` (`train_one`, `main`).
- **Look at:** the power-law fit and the 6ND compute bookkeeping — a bigger model reaches a lower loss for the same steps, but costs more per step. See [Module 10 · Scaling laws](lessons/module-10/lesson-01.md).

### (7) Evaluation

Every experiment must produce a reproducible number. Perplexity — `exp(mean cross-entropy)` on held-out text — is the pretraining metric; likelihood scoring grades multiple-choice benchmarks with no weight updates.

- **Entry point:** `llmre.evaluation.harness.evaluate_perplexity`, `sequence_loglikelihood`, `multiple_choice_score`; `llmre.evaluation.metrics.perplexity`.
- **Look at:** the token-weighted mean cross-entropy in `evaluate_perplexity`, and length-normalized log-likelihood in `multiple_choice_score`. See [Module 13 · Evaluation](lessons/module-13/lesson-01.md).

### (8) SFT (supervised fine-tuning)

Teach the model to *answer* by fine-tuning on conversations, but compute the loss on the **assistant response only**. A chat template flattens role-tagged messages into one string with delimiters; a response mask marks which tokens to train on; `build_labels` writes `ignore_index` (`-100`) into the prompt positions so cross-entropy skips them.

- **Entry point:** `llmre.sft.chat_template.render_with_response_mask`, `llmre.sft.masking.build_labels` / `masked_cross_entropy`.
- **Look at:** why the assistant's `<|end|>` closer *is* a response token (so the model learns to stop). See [14.2 · Loss masking for SFT](lessons/module-14/lesson-02.md).

### (9) LoRA

Full fine-tuning updates every weight and keeps two AdamW moments per parameter. LoRA freezes the pretrained weight and learns a tiny low-rank update `y = Wx + (alpha/r)·B(Ax)`, with `B = 0` at init so the wrapped layer starts identical to the base.

- **Entry point:** `llmre.sft.lora.LoRALinear`, `mark_only_lora_trainable`, `LoRALinear.merge`.
- **Look at:** how the trainable-parameter count collapses to `r·(in + out)`. See [14.3 · LoRA](lessons/module-14/lesson-03.md).

### (10) Preference data + DPO

Alignment starts from *pairs*: for a prompt, a chosen response `y_w` and a rejected one `y_l`. DPO optimizes the policy against these pairs directly — no reward model, no sampling — using the closed form that links reward to the policy/reference log-ratio.

- **Entry point:** `llmre.preference.dpo.dpo_loss`, `sequence_logprob`, `dpo_reward_margin`.
- **Look at:** how the prompt-only `beta·log Z(x)` term cancels in the difference, leaving a loss with only the policy and a frozen reference. See [15.3 · DPO](lessons/module-15/lesson-03.md).

### (11) A reward model and PPO

The classic RLHF path DPO short-circuits: train a **reward model** on preference pairs with the Bradley-Terry loss, then optimize the policy against it with PPO's clipped objective and a KL penalty to the reference.

- **Entry point:** `llmre.preference.reward_model.RewardModel` / `bradley_terry_loss`; `llmre.rl.ppo.compute_advantages`, `ppo_clip_objective`, `kl_penalty`.
- **Look at:** the Bradley-Terry link $P(y_w \succ y_l) = \sigma(r(x,y_w) - r(x,y_l))$ and where the KL penalty enters the reward. See [15.1–15.2 · RLHF](lessons/module-15/lesson-01.md).

### (12) GRPO + RLVR on a verifiable task

GRPO drops PPO's critic: sample a **group** of completions per prompt and use the group's own mean reward as the baseline, with advantage $A_i = (r_i - \text{mean}(r))/\text{std}(r)$. RLVR feeds it a **verifiable** reward — a rule that checks the answer (exact-match on arithmetic here), which cannot be hacked like a learned proxy.

- **Entry point:** `llmre.rl.grpo.grpo_advantages` / `grpo_objective`; `llmre.reasoning.rlvr.run_rlvr`; `llmre.reasoning.verifiers.arithmetic_verifier`.
- **Look at:** the correct-rate rising in `run_rlvr`'s history. See [16.1 · GRPO](lessons/module-16/lesson-01.md) and [16.2 · RLVR](lessons/module-16/lesson-02.md).

### (13) A reasoning task

Reasoning is elicited by making the model *show its steps* before the answer (chain-of-thought), sampling several chains and taking a majority vote (self-consistency), and — in the R1 recipe — training that behavior in with verifiable-reward RL.

- **Entry point:** `llmre.reasoning.cot.cot_prompt`, `extract_answer`, `self_consistency`; `llmre.reasoning.r1_pipeline.mini_r1`.
- **Look at:** the deterministic majority-vote tie-break, and `mini_r1`'s SFT-then-RL two-stage loop. See [Module 17 · Reasoning models](lessons/module-17/lesson-01.md).

### (14) Tool use

A model that wants to compute or look something up emits a *structured call*; a runtime executes the real function and feeds the result back. The registry advertises JSON schemas, runs the call, and turns any exception into a structured error instead of crashing.

- **Entry point:** `llmre.tools.registry.ToolRegistry`, `llmre.tools.builtins.register_builtins`, `llmre.tools.parser.parse_tool_call`, `llmre.tools.loop.run_tool_loop`.
- **Look at:** the AST-vetted `calculator` (an allow-list, not a block-list). See [18.1 · Tool schemas](lessons/module-18/lesson-01.md).

### (15) A simple agent

An agent interleaves **Thought → Action → Observation** until it emits a final answer or exhausts a step budget (ReAct). Wrap that loop with a small file view/edit/run interface and it becomes a mini SWE-agent.

- **Entry point:** `llmre.agents.react.run_react`, `format_trace`; `llmre.agents.swe_agent.solve`.
- **Look at:** how `run_react` is tool-registry-agnostic — tools are *injected*, never imported. See [19.1 · ReAct → a mini SWE-agent](lessons/module-19/lesson-01.md).

### (16) Ablations

To claim a change *caused* an improvement, vary **one** thing and hold everything else fixed. The scaling sweep is a template: same corpus, same steps, same seed, only `n_embd` changes.

- **Entry point:** `code/experiments/scaling_sweep.py` as the pattern; the config in `code/configs/capstone.yaml` as the reproducible knobs.
- **Look at:** the VARIABLES section of the report — "one independent variable, everything else a control." See [19.2 · Experiment design](lessons/module-19/lesson-02.md).

### (17) A research report

The deliverable is not the model — it is a report a colleague can reproduce: QUESTION / HYPOTHESIS / METHOD / BASELINE / VARIABLES / RESULTS / ANALYSIS / LIMITATIONS / CONCLUSION / NEXT.

- **Entry point:** `code/reports/EXPERIMENT_TEMPLATE.md`; the driver writes a filled-in one to `code/reports/capstone_report.md`.
- **Look at:** the [paper-reading and report note](papers/index.md) for the nine-question reading method and how to file each summary.

## 3. Run it yourself

The driver `code/scripts/capstone.py` runs a **tiny** version of stages (1)–(9) and (12) end to end on CPU and writes the report. From the repo root:

```bash
cd code && pip install -e .    # once
py scripts/capstone.py         # ~20 s on a laptop CPU
```

It reads `code/configs/capstone.yaml` (model size, steps, LR, seed) so the run is reproducible, and prints one line per stage. Here is an actual run (seed 1234, torch 2.14 CPU):

```text
[1] tokenizer: trained BPE, vocab_size=373 (117 merges over 256 byte ids)
[2] model: tiny GPT with 126,080 parameters (n_layer=2, n_head=4, n_embd=64)
[3] pretrain: 300 steps, train loss 5.935 -> 0.067 (drop 5.869); held-out loss 0.071
[4] eval: held-out perplexity = 1.07 (uniform baseline over vocab would be ~374)
[5] sft: response-masked loss 11.962 -> 2.560 over 11 response tokens (21 prompt tokens ignored)
[6] lora: rank-4 adapter on a 64x64 linear -> 512 trainable of 4608 params (11.1%); identical-to-base at init: True
[7] rlvr: arithmetic correct-rate 0.062 -> 1.000 over 80 GRPO steps
[done] full pipeline ran in 19.6 s on CPU
```

Read those seven lines as the whole course firing at once. The **pretraining cross-entropy falls from 5.935 to 0.067 nats** (the tiny model memorizes the tiny corpus — exactly what should happen at this scale, so perplexity drops to 1.07 against a uniform baseline of ~374). SFT lowers the response-only loss from 11.96 to 2.56. LoRA makes only 11.1% of a linear's parameters trainable while starting *identical* to the base. And RLVR lifts the arithmetic correct-rate from chance (0.062, near the ~1/10 expected for a 10-way answer space) to 1.000.

<div class="callout warn"><p>The low perplexity here is <strong>memorization, not generalization</strong> — the corpus is a dozen repeated sentences. That is the correct outcome for a smoke test (it proves the pipeline learns), and a deliberate limitation the report states plainly. Never quote a memorization perplexity as a quality result.</p></div>

The two learning signals — loss down, correct-rate up — are what a smoke test exists to check. The test `code/tests/test_capstone.py` asserts exactly this at even smaller size (`py -m pytest -q code/tests/test_capstone.py`).

## 4. Scaling each stage up

<div class="hw">
<p><strong>Hardware track.</strong></p>
<ul>
<li><strong>The provided driver:</strong> CPU-only, ~20 s, &lt;1 GB RAM, 0 GPU-hours. Runs anywhere. This is the "see it all work" configuration.</li>
<li><strong>A real GPT-2 small (124M) pretrain:</strong> 1× A100/H100 (40–80 GB), tens of GPU-hours for a few billion tokens; ~15–30 GB GPU memory at a sensible batch with mixed precision and gradient accumulation. CPU-only is not feasible.</li>
<li><strong>SFT / LoRA:</strong> LoRA fits a 7B model on a single 24 GB consumer GPU (that is the point — the optimizer state shrinks to the adapters); full SFT of the same model needs multiple 40–80 GB GPUs. Minutes-to-hours, not days.</li>
<li><strong>DPO / reward-model + PPO:</strong> DPO ≈ SFT cost (two forward passes, policy + frozen reference). PPO/GRPO with real generation is far heavier — you sample completions every step — so budget several GPUs and hours-to-days for a small model, and expect generation, not the gradient step, to dominate.</li>
<li><strong>Reasoning RL at scale:</strong> long chains-of-thought make each rollout long; frontier reasoning runs are thousands of GPU-hours. The mechanism is identical to the toy loop; only the scale of sampling changes.</li>
</ul>
<p>Rule of thumb (Chinchilla, reasonable industry practice): a compute-optimal pretrain uses ~20 tokens per parameter, and generation-based RL is dominated by sampling cost, not the update.</p>
</div>

To grow each stage from the toy driver: raise `tokenizer.vocab_size` toward 32k–50k; grow the model (`n_layer`, `n_head`, `n_embd`, `block_size`) toward GPT-2 sizes; swap the toy corpus for a real, *filtered and decontaminated* dataset (Module 11); move to a GPU and turn on mixed precision and FlashAttention (Module 8); and replace the categorical RLVR policy with the **GPT itself emitting answer tokens** (the driver's report names this as the next experiment).

## 5. Where you are now

You have built, from scratch and then connected, every stage of a modern LLM pipeline. Rate yourself honestly against the [Readiness matrix](curriculum/readiness-matrix.md): a skill is *Advanced* only when you can **explain → derive → implement → test → benchmark → debug → relate-to-a-paper** it from memory. The capstone touches nearly every row.

When you run your own experiments, write them up with the [experiment report template](papers/index.md) note — one report, one falsifiable question, real numbers with a baseline and stated limitations, exactly like `code/reports/capstone_report.md`.

## Check yourself

<details><summary>The driver reports pretraining loss dropping to 0.067 nats and perplexity 1.07. Why is this <em>not</em> evidence of a good model?</summary>

Because the corpus is a dozen sentences repeated. The tiny model has enough capacity to *memorize* them, so it predicts the next token almost perfectly on text it has already seen. Perplexity 1.07 means "barely any uncertainty" — but only on the training distribution. A quality claim needs a *held-out* set the model never trained on, and a corpus large and varied enough that memorization is impossible. The report's LIMITATIONS section says exactly this.

</details>

<details><summary>In the SFT stage, 21 prompt tokens are "ignored" and 11 response tokens are trained on. What mechanism does the ignoring, and what would break if you trained on all 32?</summary>

`build_labels` writes the sentinel `ignore_index = -100` into every prompt position, and `masked_cross_entropy` drops positions whose label is `-100`. If you trained on all 32, the model would be rewarded for reproducing the *user's* prompt as well as the assistant's answer — wasting capacity learning to echo inputs and blurring the "when is it my turn to speak" signal the chat template exists to teach.

</details>

<details><summary>Why does the capstone use a categorical policy table for RLVR instead of the GPT you just pretrained?</summary>

Speed and clarity on CPU. GRPO's algorithm — sample a group, standardize rewards into advantages, take a clipped policy-gradient step — is identical whether the policy is a lookup table over answers or a GPT emitting answer *tokens*. The table makes the correct-rate visibly climb in under a second, isolating the RL mechanism from the cost of generation. Swapping in the GPT changes the engineering (you sample token sequences and score them), not the update rule — which is why it is the report's "next experiment."

</details>

<details><summary>LoRA reports 512 trainable of 4608 params on a 64×64 linear at rank 4. Where does 512 come from, and why is the wrapped layer "identical to base at init"?</summary>

`A` is `(r, in) = (4, 64) = 256` and `B` is `(out, r) = (64, 4) = 256`, so `256 + 256 = 512` trainable adapter params; the frozen base weight is `64·64 = 4096`, giving `4608` total. It is identical to the base at init because `B` is initialized to zeros, so the update `(alpha/r)·B(Ax)` is exactly zero and `forward(x) == base(x)` — training therefore starts from the pretrained model, not a perturbed one.

</details>

<details><summary>You want to claim "RMSNorm lowers final loss vs LayerNorm." Using the ablation stage's logic, what must be true of the run?</summary>

Exactly one thing changes — the norm — and everything else (corpus, tokens, steps, seed(s), model size, optimizer, LR schedule) is held fixed, run under identical conditions. You report mean ± std across several seeds, not a single seed, and check whether the difference exceeds seed noise (do the confidence intervals overlap?). If more than one thing changed, you cannot attribute the effect. That is the VARIABLES + RESULTS + ANALYSIS discipline of the report template.

</details>

## Next: keep going

You have finished the course, but a research engineer never stops reading. Pick a paper you have *not* read from the [paper reading curriculum](papers/index.md) — a strong first choice is [OLMo 2](https://arxiv.org/abs/2501.00656), because its data, code, and checkpoints are fully open, so you can reproduce and ablate it against everything you built here. Read it with the nine-question method (what problem → key idea → which equations → what to implement → how it influenced later LLMs), then implement the one thing it changes inside `llmre`, wire it into `code/scripts/capstone.py`, and write the result up with the [experiment report template](papers/index.md). That loop — read, implement, test, measure, write — is the job. You are ready to do it.
