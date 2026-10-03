# 07.4 · Practice set: FLOPs, throughput, MFU, memory

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-07/lesson-04">07.4 · Throughput, FLOPs, MFU, memory accounting</a>.</p>
<p><strong>How to use this page:</strong> solve every exercise before scrolling to the <a href="#/lessons/module-07/practice-04?id=solutions">Solutions</a>. Exercises marked ✍️ are for pen and paper (a calculator is fine). Exercises marked 🐍 are for Python / PyTorch with the course's <code>llmre</code> package installed. Write big numbers in scientific notation and keep 3–4 significant figures. 1 GB = $10^9$ bytes, 1 TFLOP = $10^{12}$ FLOPs.</p>
<p><strong>Why this matters for ML:</strong> these are the numbers you are expected to produce in your head in a design review: how many FLOPs a run costs, how long it takes, whether the GPU is being used well, and whether the model fits. The goal of this page is that every number in 07.4 becomes automatic.</p>
</div>

<div class="hw"><p><strong>Hardware:</strong> CPU-only. Every ✍️ exercise is arithmetic. Every 🐍 exercise runs in seconds on a laptop with <code>torch</code> installed (F3 measures a real tokens/sec on whatever machine you have).</p></div>

**Formula sheet** (the only things you need):

$$
N \approx 12\, n_{\text{layer}}\, C^2 \quad(\text{attention } 4C^2 + \text{MLP } 8C^2 \text{ per layer}), \qquad
\text{FLOPs/token} \approx 6N = \underbrace{2N}_{\text{forward}} + \underbrace{4N}_{\text{backward}}
$$

$$
\text{total FLOPs} \approx 6ND, \qquad
\text{tokens/sec} = \frac{\text{tokens per step}}{\text{seconds per step}}, \qquad
\text{MFU} = \frac{6N\cdot\text{tokens/sec}}{\text{peak FLOP/s}}
$$

$$
\text{attention scores (causal)} \approx 6\, n_{\text{layer}}\, T\, C \text{ FLOPs/token}, \qquad
\frac{\text{scores}}{6N} = \frac{T}{12C}
$$

$$
\text{fixed memory (fp32, AdamW)} = \underbrace{4P}_{\text{params}} + \underbrace{4P}_{\text{grads}} + \underbrace{8P}_{\text{Adam } m,v} = 16P \text{ bytes}, \qquad
\text{one activation tensor} = (\text{number of elements})\times(\text{bytes per element})
$$

**Constants:** GPT-2 small: $n_{\text{layer}} = 12$, $C = 768$, $V = 50257$, $T = 1024$, total params $P = 124{,}439{,}808$. A100 peak: $312$ TFLOP/s (bf16, dense).

---

## Part A — Counting parameters and the 6N rule (07.4 sections 2–3)

**A1 ✍️ — A toy model end to end.** A model has $n_{\text{layer}} = 2$ and $C = 4$.

1. How many weights are in one layer's attention block? In one layer's MLP? In one layer?
2. What is $N$?
3. FLOPs per token for the forward pass, the backward pass, and both together.

**A2 ✍️ — GPT-2 small, every number.** For $n_{\text{layer}} = 12$, $C = 768$:

1. $C^2$, attention weights per layer, MLP weights per layer, weights per layer.
2. $N$.
3. Forward FLOPs/token, backward FLOPs/token, total FLOPs/token.

**A3 ✍️ — One Linear layer.** The MLP's first Linear in GPT-2 small maps $C = 768 \to 4C = 3072$.

1. How many weights does it have?
2. How many forward FLOPs does it spend on one token? Backward?
3. Of the backward FLOPs, how many go to $\partial L/\partial W$ and how many to $\partial L/\partial x$?

**A4 ✍️ — Scale it up.** GPT-2 medium has $n_{\text{layer}} = 24$, $C = 1024$. Compute $N$ and $6N$. By what factor did $N$ grow compared with GPT-2 small, and why is it $2\times$ from the layers times $(1024/768)^2$ from the width?

**A5 ✍️ — Why 6 and not 2?** In two sentences: what are the two matmuls the backward pass does for every weight matrix, and which of the two produces the thing the optimizer needs?

**A6 ✍️ — FLOPs per optimizer step.** GPT-2 small, one optimizer step of $500{,}000$ tokens. How many FLOPs is the step? Express it in TFLOP. Repeat for a step of $524{,}288 = 2^{19}$ tokens.

**A7 ✍️ — A whole run.** GPT-2 small trained on $D = 10^{10}$ tokens.

1. Total training FLOPs.
2. Wall-clock time on one A100 running at 40% MFU.

**A8 ✍️ — Which N?** A colleague computes FLOPs/token as $6 \times 124{,}439{,}808$ (the total parameter count, embeddings included). By what percentage does that overestimate the 6N cost? Why do the embeddings not belong in $N$ for the FLOP count?

---

## Part B — The attention scores 6N leaves out (07.4 section 2)

**B1 ✍️ — Tiny, by hand.** $C = 4$, $T = 3$, one layer.

1. FLOPs for $QK^\top$ for one token, no mask. For $PV$.
2. Forward total, and forward+backward total, no mask.
3. Forward+backward total with a causal mask (average of $T/2$ keys).

**B2 ✍️ — GPT-2 small at full context.** $T = 1024$.

1. Score FLOPs per token for the whole model (causal).
2. The ratio $T/(12C)$, and check it against your answer to 1 divided by $6N$.
3. A naive implementation computes the full $T\times T$ score matrix and then masks it. What are its score FLOPs per token and its ratio to $6N$?

**B3 ✍️ — Where the ignored term takes over.** For $C = 4096$:

1. At what context length $T$ does the score cost equal the 6N cost?
2. What is the ratio at $T = 4096$? At $T = 128{,}000$?

**B4 ✍️ — Same N, different T.** Two runs of the same model: one at $T = 2048$, one at $T = 8192$. Which of these numbers change between the two runs: $N$, $6N$, the score FLOPs per token, the true total FLOPs per token? Say which way.

---

## Part C — Throughput and MFU (07.4 section 4)

**C1 ✍️ — Tokens per step.** A run uses micro-batch $b = 16$, sequence length $T = 1024$, gradient accumulation $G = 32$, one GPU. How many tokens per optimizer step? If a step takes 10.5 s, what are the tokens/sec?

**C2 ✍️ — MFU of C1.** The run in C1 trains GPT-2 small on one A100. What is the MFU?

**C3 ✍️ — The two worked MFUs.** GPT-2 small:

1. One A100 at 50,000 tokens/sec.
2. Eight A100s at an aggregate 1,000,000 tokens/sec.

**C4 ✍️ — Work backwards.** How many tokens/sec must one A100 reach on GPT-2 small to hit 40% MFU?

**C5 ✍️ — What MFU costs you.** A GPT-2-small step of 500,000 tokens on one A100. How long does it take at 40% MFU? At 20%? If an A100 rents for $\$2$/hour (an assumed price, for the arithmetic only), what does the run of A7 cost at 40% MFU, and at 20%?

**C6 ✍️ — Units.** For each quantity say whether it is a count (FLOPs), a rate (FLOP/s) or a ratio (no unit): $6N$; $6ND$; 312 TFLOP/s; tokens/sec × $6N$; MFU.

---

## Part D — Memory accounting (07.4 section 5)

**D1 ✍️ — Count numbers, then bytes.** A model with $P = 1000$ parameters trained in fp32 with AdamW. How many stored numbers are in each of the three fixed buckets? Total numbers? Total bytes?

**D2 ✍️ — GPT-2 small fixed memory.** $P = 124{,}439{,}808$, fp32. Bytes and GB for params, gradients, Adam $m$ and $v$ together, and the total.

**D3 ✍️ — Bigger models.** Fixed memory ($16P$, fp32) for $P = 350$M, $1.3$B, $7$B. Which of them fit on one 24 GB GPU? On one 80 GB GPU (still ignoring activations)?

**D4 ✍️ — Largest model that fits.** Ignoring activations, what is the largest $P$ whose fp32 AdamW fixed state fits in 24 GB? In 80 GB?

**D5 ✍️ — The logits tensor.** GPT-2 small, $B = 8$, $T = 1024$.

1. Number of elements in the logits tensor $(B, T, V)$.
2. Its size in fp32 and in bf16.
3. Compare the fp32 size with the 1.99 GB fixed state.
4. Double the batch to $B = 16$. What happens to the logits size, and to the fixed state?

**D6 ✍️ — One residual-stream tensor.** One $(B, T, C)$ activation with $B = 8$, $T = 1024$, $C = 768$: elements and fp32 bytes.

**D7 ✍️ — The gradient that is not stored.** For the MLP's first Linear ($768 \to 3072$) at $B = 8$, $T = 1024$:

1. Shape and fp32 size of $\partial L/\partial W$ (the `.grad`).
2. Shape and fp32 size of $\partial L/\partial x$ (the gradient w.r.t. its input).
3. Which one is kept until the optimizer step, and which one is freed as soon as the previous layer has used it? Which one grows with the batch?

**D8 ✍️ — Two different "4"s.** The backward pass does about 4 FLOPs per parameter per token, and the fixed memory is $4P$ numbers. Explain in two sentences why these are unrelated.

---

## Part E — Putting it together

**E1 ✍️ — A 7B run.** $N = 7\times 10^9$, trained on $D = 1.4\times 10^{11}$ tokens ($20N$, the Chinchilla ballpark of Module 10) on 64 A100s at 40% MFU.

1. Total FLOPs.
2. Effective cluster FLOP/s.
3. Wall-clock time in hours and days.
4. GPU-hours.
5. Fixed fp32 AdamW memory. Does it fit on one 80 GB GPU? If the optimizer state, gradients and parameters are sharded evenly across 8 GPUs (ZeRO/FSDP, Module 9), how much is that per GPU?

**E2 ✍️ — Read a training log.** A log line says: `step 1200 | 524288 tok | 6.30 s | GPT-2 small | 1x A100`. Compute tokens/sec and MFU. Is it in the 30–50% range that well-tuned large runs report? Name one reason a model this small would sit lower.

---

## Part F — PyTorch

**F1 🐍 — Check the estimate against the real model.**

```python
from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.training.metrics import non_embedding_params, model_flops_per_token

cfg = GPTConfig()
model = GPT(cfg)
```

Print `model.num_params()`, `non_embedding_params(cfg)`, and `model_flops_per_token(cfg)`. Then count the exact non-embedding weights yourself: sum `p.numel()` over every parameter inside `model.transformer.h` (the blocks). How far is $12\,n_{\text{layer}}\,C^2$ from that exact count, in percent? What is in the exact count that the estimate leaves out?

**F2 🐍 — Count the four buckets from tensors.** Build a small model `GPT(GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=64, vocab_size=256))`, an `AdamW` optimizer, run one forward/backward/`step()` on a random batch of shape `(4, 64)`. Then compute, from the actual tensors, the bytes in parameters, in `.grad`s, and in the optimizer state (`opt.state[p]["exp_avg"]` and `["exp_avg_sq"]`). Check that grads = params and Adam state = 2 × params. Are there any other entries in `opt.state[p]`, and are they per-parameter?

**F3 🐍 — Measure your own MFU.** With the small model of F2, time 20 training steps on your machine (skip the first 3 as warm-up; on CPU use `time.perf_counter()`, on GPU call `torch.cuda.synchronize()` before reading the clock). Compute tokens/sec and FLOPs/s with `model_flops_per_token`. Look up your CPU's or GPU's peak FLOP/s for fp32 and compute the MFU with `llmre.training.metrics.mfu`. Why is it so low?

**F4 🐍 — Activation memory scales with the batch.** On a GPU, for the F2 model, record `torch.cuda.max_memory_allocated()` after one forward+backward at batch sizes 4, 8 and 16 (call `torch.cuda.reset_peak_memory_stats()` between them). Subtract the fixed state from F2. Does the rest grow linearly with $B$? On CPU only: instead compute the size of the logits tensor at each batch size from its shape and dtype.

---

## Part G — Debugging

**G1 ✍️ — Forgot backward.** A capacity plan estimates a run's compute as $2ND$. By what factor is the time and cost underestimated?

**G2 ✍️ — The wrong peak.** Someone reports 4.1% MFU for GPT-2 small at 50,000 tokens/sec on one A100. You get 8.2%. They divided by 624 TFLOP/s. What peak is that, and why is it the wrong denominator for this run?

**G3 ✍️ — OOM at the optimizer step.** A 1.3B-parameter run was budgeted as "weights + grads = 10.4 GB, fits in 16 GB with room for activations." The forward and backward pass run, then the first `opt.step()` runs out of memory. What was left out of the budget, how big is it, and why does the failure appear exactly at the first step?

**G4 ✍️ — Fast tokens, bad MFU.** Run A (125M params) reports 200,000 tokens/sec; run B (1.3B params) reports 30,000 tokens/sec, both on one A100. The team concludes run A is "using the GPU better." Compute both MFUs (use $N \approx P$) and decide.

---

## Solutions

Stop here if you have not tried the exercises yet.

### A1

1. Attention: $4C^2 = 4\cdot 16 = 64$. MLP: $8C^2 = 128$. One layer: $12C^2 = 192$.
2. $N = 12\cdot 2\cdot 16 = 384$.
3. Forward $2N = 768$ FLOPs, backward $4N = 1536$ FLOPs, total $6N = 2304$ FLOPs per token.

### A2

1. $C^2 = 589{,}824$. Attention $4C^2 = 2{,}359{,}296$. MLP $8C^2 = 4{,}718{,}592$. Per layer $12C^2 = 7{,}077{,}888$.
2. $N = 12 \cdot 7{,}077{,}888 = 84{,}934{,}656 \approx 84.9$M.
3. Forward $2N = 169{,}869{,}312 \approx 1.70\times 10^8$. Backward $4N = 339{,}738{,}624 \approx 3.40\times 10^8$. Total $6N = 509{,}607{,}936 \approx 5.10\times 10^8$ FLOPs/token.

### A3

1. $768 \cdot 3072 = 2{,}359{,}296$ weights (the same as all four attention matrices together, $4C^2$).
2. Forward: one multiply-add per weight per token = $2 \cdot 2{,}359{,}296 = 4{,}718{,}592$ FLOPs. Backward: twice that, $9{,}437{,}184$ FLOPs.
3. Half each: $4{,}718{,}592$ FLOPs compute $\partial L/\partial W$ and $4{,}718{,}592$ compute $\partial L/\partial x$.

### A4

$N = 12\cdot 24\cdot 1024^2 = 301{,}989{,}888 \approx 302$M. $6N = 1{,}811{,}939{,}328 \approx 1.81\times 10^9$ FLOPs/token. Ratio to GPT-2 small: $301{,}989{,}888 / 84{,}934{,}656 = 3.556$. Twice the layers gives $2\times$; each layer has $12C^2$ weights, so 1.333× the width gives $1.333^2 = 1.778\times$; $2 \times 1.778 = 3.556$. Width enters squared, depth enters linearly.

### A5

For every weight matrix the backward pass computes the gradient w.r.t. the layer's input $\partial L/\partial x$ (one matmul, ≈ $2$ FLOPs per weight per token, needed to keep propagating to earlier layers) and the gradient w.r.t. the weights $\partial L/\partial W$ (another matmul, another ≈ 2). The second one is what the optimizer uses; together they make backward ≈ $4N$, and with the forward's $2N$ the total is $6N$.

### A6

$500{,}000$ tokens: $509{,}607{,}936 \cdot 5\times 10^5 = 2.548\times 10^{14}$ FLOPs $= 254.8$ TFLOP. $524{,}288$ tokens: $509{,}607{,}936 \cdot 524{,}288 = 2.672\times 10^{14}$ FLOPs $= 267.2$ TFLOP.

### A7

1. $6ND = 5.096\times 10^8 \cdot 10^{10} = 5.096\times 10^{18}$ FLOPs.
2. Effective rate $= 0.4 \cdot 3.12\times 10^{14} = 1.248\times 10^{14}$ FLOP/s. Time $= 5.096\times 10^{18} / 1.248\times 10^{14} = 40{,}834$ s $\approx 11.3$ hours.

### A8

$6 \cdot 124{,}439{,}808 = 746{,}638{,}848$, versus $6N = 509{,}607{,}936$: $746.6 / 509.6 = 1.465$, a **46.5% overestimate**. The token embedding is a table lookup: reading one row costs no multiply-adds, so those $50257 \times 768$ parameters are not "used in one multiply-add per token." (In GPT-2 the same matrix is reused as the output head, and that matmul does cost $2VC$ FLOPs per token forward — a real cost the $12C^2$ estimate also leaves out, alongside the attention scores. The 6N convention drops it on purpose; know that it is there.) An overestimated FLOP count also inflates your MFU.

### B1

1. $QK^\top$: 3 dot products of length 4 = 12 multiply-adds = **24 FLOPs**. $PV$: 3 vectors of length 4 scaled and summed = 12 multiply-adds = **24 FLOPs**.
2. Forward $4TC = 48$. Forward+backward $12TC = 144$.
3. Causal: half, $6TC = 72$ FLOPs.

### B2

1. $6\cdot 12\cdot 1024\cdot 768 = 56{,}623{,}104 \approx 5.66\times 10^7$ FLOPs/token.
2. $1024 / (12\cdot 768) = 1024/9216 = 0.111$. Check: $5.66\times 10^7 / 5.10\times 10^8 = 0.111$. The scores add ≈ 11% on top of 6N.
3. Naive (un-halved): $12\cdot 12\cdot 1024\cdot 768 = 113{,}246{,}208 \approx 1.13\times 10^8$, ratio $0.222$ — 22% on top of 6N, half of it wasted on masked-out entries.

### B3

1. $T = 12C = 49{,}152$.
2. $T = 4096$: $4096/49{,}152 = 0.083$ (8.3%). $T = 128{,}000$: $128{,}000/49{,}152 = 2.60$ — the scores cost 2.6× everything 6N counts.

### B4

$N$ and $6N$ do not change: they depend only on the weights. The score FLOPs per token grow 4× (linear in $T$). So the true total FLOPs per token grows, and the 6N estimate becomes less accurate at $T = 8192$. For GPT-2 small's width the ratio would go from $2048/9216 = 0.22$ to $8192/9216 = 0.89$.

### C1

$16 \cdot 1024 \cdot 32 = 524{,}288$ tokens per step. $524{,}288 / 10.5 = 49{,}932$ tokens/sec.

### C2

$\text{MFU} = 5.096\times 10^8 \cdot 49{,}932 / 3.12\times 10^{14} = 2.545\times 10^{13} / 3.12\times 10^{14} = 0.0816 = 8.2\%$.

### C3

1. $5.096\times 10^8 \cdot 5\times 10^4 = 2.548\times 10^{13}$ FLOP/s; $/3.12\times 10^{14} = 0.0817 = 8.2\%$.
2. $5.096\times 10^8 \cdot 10^6 = 5.096\times 10^{14}$ FLOP/s; peak $8 \cdot 3.12\times 10^{14} = 2.496\times 10^{15}$; $\text{MFU} = 0.204 = 20.4\%$. Always divide by the peak of *all* the GPUs that produced the aggregate tokens/sec.

### C4

$\text{tokens/sec} = \text{MFU}\cdot\text{peak}/6N = 0.4 \cdot 3.12\times 10^{14} / 5.096\times 10^8 = 244{,}894 \approx 245{,}000$ tokens/sec — about 5× the 50,000 of C3.

### C5

Step FLOPs $= 2.548\times 10^{14}$ (A6). At 40%: $2.548\times 10^{14} / 1.248\times 10^{14} = 2.04$ s. At 20%: $4.08$ s. The A7 run: 11.34 h at 40% → $\$22.69$; at 20% it takes 22.7 h → $\$45.37$. Half the MFU, double the bill.

### C6

$6N$: a count (FLOPs per token). $6ND$: a count (FLOPs). 312 TFLOP/s: a rate. tokens/sec × $6N$: a rate (achieved FLOP/s). MFU: a ratio of two rates, no unit.

### D1

Params 1000, grads 1000, Adam $m$ 1000 + $v$ 1000 = 2000. Total **4000 numbers** $= 4P$. At 4 bytes each: **16,000 bytes** $= 16P$.

### D2

Params $4P = 497{,}759{,}232$ B ≈ 0.498 GB. Grads $4P$ ≈ 0.498 GB. Adam $8P = 995{,}518{,}464$ B ≈ 0.996 GB. Total $16P = 1{,}991{,}036{,}928$ B ≈ **1.99 GB**.

### D3

$350$M: $5.6$ GB. $1.3$B: $20.8$ GB. $7$B: $112$ GB. On 24 GB: 350M fits easily; 1.3B technically fits its fixed state but leaves only 3.2 GB for activations; 7B does not. On 80 GB: 350M and 1.3B fit; 7B (112 GB) does not, even before activations.

### D4

$24\times 10^9 / 16 = 1.5\times 10^9$ parameters. $80\times 10^9 / 16 = 5\times 10^9$ parameters. In practice lower, because activations need room too.

### D5

1. $8 \cdot 1024 \cdot 50{,}257 = 411{,}705{,}344 \approx 4.12\times 10^8$ elements.
2. fp32: $1{,}646{,}821{,}376$ B ≈ **1.65 GB**. bf16: ≈ **0.82 GB**.
3. One activation tensor (1.65 GB) is 83% of all the fixed state (1.99 GB).
4. $B = 16$: the logits double to ≈ 3.29 GB. The fixed state stays 1.99 GB — it has nothing to do with the batch.

### D6

$8\cdot 1024\cdot 768 = 6{,}291{,}456$ elements; $\times 4 = 25{,}165{,}824$ B ≈ **25.2 MB**.

### D7

1. $\partial L/\partial W$ has the shape of $W$: $(3072, 768)$ (PyTorch stores a Linear's weight as out × in), $2{,}359{,}296$ numbers, ≈ **9.4 MB**.
2. $\partial L/\partial x$ has the shape of the input $x$: $(B, T, C) = (8, 1024, 768)$, $6{,}291{,}456$ numbers, ≈ **25.2 MB**.
3. $\partial L/\partial W$ is the `.grad`; it is kept (and accumulated across tokens and micro-batches) until the optimizer step. $\partial L/\partial x$ is handed to the previous layer as its upstream gradient and freed as soon as that layer has used it. $\partial L/\partial x$ grows with $B$; $\partial L/\partial W$ does not.

### D8

The "4 FLOPs per parameter per token" counts *operations*: for every token the backward pass does a multiply and an add twice with each weight, and those results are summed into slots. The "$4P$ numbers" counts *stored slots*: one weight, one gradient, one $m$, one $v$ per parameter — millions of token contributions all end up added into the same one gradient slot, so the two 4s are a coincidence.

### E1

1. $6ND = 6\cdot 7\times 10^9 \cdot 1.4\times 10^{11} = 5.88\times 10^{21}$ FLOPs.
2. $64 \cdot 3.12\times 10^{14} \cdot 0.4 = 7.99\times 10^{15}$ FLOP/s.
3. $5.88\times 10^{21} / 7.99\times 10^{15} = 7.36\times 10^5$ s ≈ **204.5 hours ≈ 8.5 days**.
4. $204.5 \cdot 64 \approx$ **13,090 GPU-hours**.
5. $16P = 112$ GB: does not fit on one 80 GB GPU. Sharded 8 ways: $112/8 = 14$ GB per GPU, leaving room for activations.

### E2

$524{,}288 / 6.30 = 83{,}220$ tokens/sec. $\text{MFU} = 5.096\times 10^8 \cdot 83{,}220 / 3.12\times 10^{14} = 0.136 = 13.6\%$. Below 30–50%. A model this small does little arithmetic per kernel launch, so fixed overheads (Python, kernel launches, LayerNorm/softmax/elementwise work, the attention scores and the big logits matmul the 6N count leaves out, data loading) are a larger share of each step.

### F1

```python
print(model.num_params())                 # 124439808
print(non_embedding_params(cfg))          # 84934656
print(model_flops_per_token(cfg))         # 509607936.0
exact = sum(p.numel() for p in model.transformer.h.parameters())
print(exact, (non_embedding_params(cfg) - exact) / exact * 100)
```

The exact count over the blocks is 85,054,464, so the estimate is 0.14% low. The difference is the biases (each Linear has one per output) and the two LayerNorms per block (a gain and a bias of length $C$ each): all $O(C)$ per layer, against $12C^2$ weights. (The lesson's 85,056,000 also includes the final LayerNorm `ln_f`, another $2C = 1536$.)

### F2

```python
import torch
from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT

cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=64, vocab_size=256)
model = GPT(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, 256, (4, 64))
_, loss = model(x, x)
loss.backward()
opt.step()

nbytes = lambda ts: sum(t.numel() * t.element_size() for t in ts)
params = list(model.parameters())
p_bytes = nbytes(params)
g_bytes = nbytes(p.grad for p in params)
a_bytes = nbytes(t for p in params for k, t in opt.state[p].items() if k != "step")
print(p_bytes, g_bytes, a_bytes, a_bytes / p_bytes)   # grads == params, ratio 2.0
print(opt.state[params[0]].keys())
```

`g_bytes == p_bytes` and `a_bytes == 2 * p_bytes`. The other entry is `step`, one scalar per parameter tensor (not per parameter number), so it is negligible.

### F3

```python
import time
from llmre.training.metrics import model_flops_per_token, mfu

fpt = model_flops_per_token(cfg)
tok = x.numel()
times = []
for i in range(23):
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    if i >= 3:
        times.append(time.perf_counter() - t0)
tps = tok / (sum(times) / len(times))
print(f"{tps:.0f} tok/s, {fpt * tps:.3g} FLOP/s, MFU {mfu(fpt, tps, peak_flops=1e12):.4f}")  # put your machine's peak here
```

Expect a tiny MFU (often well under 5%). With $C = 64$ every matmul is so small that the time goes to Python overhead, kernel launches and elementwise ops, not arithmetic. The FLOP count is also a slight underestimate here: 6N ignores the logits matmul, $2VC = 2\cdot 256\cdot 64 = 32{,}768$ forward FLOPs per token, against $2N = 2\cdot 98{,}304 = 196{,}608$ — about 17% extra work that MFU does not credit.

### F4

```python
import torch
model = GPT(cfg).cuda()
for B in (4, 8, 16):
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, 256, (B, 64), device="cuda")
    _, loss = model(x, x)
    loss.backward()
    print(B, torch.cuda.max_memory_allocated())
    model.zero_grad(set_to_none=True)
```

After subtracting the parameters and gradients, the remainder (activations) roughly doubles each time $B$ doubles. CPU-only: logits are $B \cdot 64 \cdot 256 \cdot 4$ bytes = 262,144, 524,288, 1,048,576 bytes — exactly linear.

### G1

$6ND / 2ND = 3$: time and cost are underestimated **3×**.

### G2

624 TFLOP/s is the A100's bf16 peak *with 2:4 structured sparsity*. A dense training run cannot use it, so the right denominator is the dense 312 TFLOP/s: $2.548\times 10^{13} / 3.12\times 10^{14} = 8.2\%$. Always state the dtype and dense vs sparse when you report MFU.

### G3

The Adam state was left out: $m$ and $v$ are $8P = 10.4$ GB in fp32, as much as weights + grads together. The full fixed state is $16P = 20.8$ GB, more than the 16 GB GPU. PyTorch's AdamW creates $m$ and $v$ lazily inside the first `step()`, so the forward and backward pass succeed and the allocation fails exactly there.

### G4

Run A: $6 \cdot 1.25\times 10^8 \cdot 2\times 10^5 = 1.5\times 10^{14}$ FLOP/s → $48.1\%$ MFU. Run B: $6\cdot 1.3\times 10^9 \cdot 3\times 10^4 = 2.34\times 10^{14}$ FLOP/s → $75.0\%$ MFU. Run B uses the GPU better despite 6.7× fewer tokens/sec. (75% would be unusually high in practice — it is a constructed example; if you saw it in a real log you would first check that the FLOP count and peak are right.)

---

Back to [07.4](lessons/module-07/lesson-04.md). Continue to [08.1 · The GPU memory hierarchy](lessons/module-08/lesson-01.md).
