# 07.4 · Throughput, FLOPs, MFU, memory accounting

<div class="prereq">
<p><strong>Prerequisites:</strong> the training loop from <a href="#/lessons/module-07/lesson-01">07.1 · The training loop</a>; the parameter count and where compute/memory go from <a href="#/lessons/module-06/lesson-02">06.2 · Init, forward pass, parameter count</a>; tokens per step from <a href="#/lessons/module-07/lesson-02">07.2</a>; AdamW's two moment buffers from <a href="#/lessons/module-03/lesson-02">03.2</a>.</p>
<p><strong>You will learn:</strong> the <strong>6N rule</strong> — a forward+backward costs ≈ $6N$ FLOPs per token, and why the 6 splits as 2 (forward) + 4 (backward); how to turn seconds-per-step into <strong>tokens/second</strong> and into <strong>MFU</strong> (achieved FLOPs/s ÷ hardware peak); and how to account for training memory in four buckets — parameters, gradients, Adam optimizer state ($2\times$ params), and activations — with a worked estimate for GPT-2 small.</p>
<p><strong>Why this matters for ML:</strong> these four numbers are how engineers reason about cost. FLOPs and tokens/sec tell you how long a run takes and what it will cost; MFU tells you whether you are wasting the hardware; the memory buckets tell you whether the model even fits, and are the reason the entire next stretch of the course exists — the memory hierarchy of <a href="#/lessons/module-08/lesson-01">Module 8</a> and the scaling laws of <a href="#/lessons/module-10/lesson-01">Module 10</a> both start here.</p>
</div>

## 1. Intuition: two questions about every run

For any training run you ask two things: *how much arithmetic is this?* (which sets the time and the dollar cost) and *does it fit in memory?* (which sets whether it runs at all). Remarkably, both have simple estimates that depend mostly on one number — $N$, the parameter count — and you can compute them on the back of an envelope before launching anything.

- **Compute:** ≈ $6N$ FLOPs per token, forward+backward. Multiply by the tokens you will train on and you have the run's total arithmetic.
- **Memory:** parameters + gradients + optimizer state + activations. The first three are each ~$N$ numbers (optimizer state is $2N$ for Adam); activations depend on batch and context.

<div class="callout key"><p>Two rules run all of training-cost reasoning: <strong>compute ≈ 6N FLOPs per token × D training tokens = 6·N·D FLOPs total</strong> (N = params, D = number of tokens; e.g. N = 10⁸, D = 10¹⁰ → 6·10⁸·10¹⁰ = 6×10¹⁸ FLOPs), and <strong>memory ≈ (params + grads + 2×params optimizer state) + activations</strong>. Everything in this lesson is unpacking those two.</p></div>

## 2. The 6N rule: 2 forward + 4 backward

The dominant cost in a transformer is matrix multiplication, and the dominant matmuls are the ones against the weights. A matmul of a length-$C$ input vector by a $C\times C$ weight matrix is $C^2$ multiply-adds; counting a multiply-add as 2 FLOPs (one multiply, one add), that is $2C^2$ FLOPs. Summed over all weight matrices, the **forward** pass costs about $2N$ FLOPs per token, where $N$ is the number of (non-embedding) parameters — each parameter is used in roughly one multiply-add per token.

The **backward** pass costs about twice the forward, ≈ $4N$ per token, because it computes *two* gradients for every matmul:

- the gradient w.r.t. the layer's **input** (to keep propagating backward), one matmul, ≈ $2N$;
- the gradient w.r.t. the layer's **weights** (what the optimizer needs), another matmul, ≈ $2N$.

Forward's $2N$ plus backward's $4N$ gives the rule:

$$
\text{FLOPs per token} \approx 6N, \qquad \text{total training FLOPs} \approx 6 N D,
$$

with $D$ the total number of training tokens. This is the "$6ND$" you will see in every scaling-law paper (Module 10).

<div class="callout key"><p>Forward ≈ $2N$ (one matmul per weight), backward ≈ $4N$ (two matmuls per weight — grad-of-input and grad-of-weight). Total ≈ $6N$ FLOPs per token. The 6 is not magic: it is $2 + 2 + 2$.</p></div>

### What $N$ means here, and the caveat

We take $N$ as the **non-embedding** parameter count, estimated by `non_embedding_params(cfg) = 12 · n_layer · n_embd²`. Per layer it adds up like this:

- **Attention: four $C\times C$ weight matrices = $4C^2$.** Three of them make the query, key and value vectors from the token's $C$-dim input: $W_Q$, $W_K$, $W_V$. The **fourth is the output projection $W_O$**: after attention has mixed the value vectors of all heads into one $C$-dim vector, $W_O$ ($C\times C$) maps that result back into the residual stream. In our code the first three are fused into one Linear, `attn.c_attn` ($C\to 3C$, i.e. $3C^2$ weights), and the fourth is `attn.c_proj` ($C\to C$, $C^2$ weights) — $3C^2 + C^2 = 4C^2$. Splitting into heads does not change this: $H$ heads of size $C/H$ together still use $C\times C$ per matrix.
- **MLP: $8C^2$.** The $C\to 4C$ Linear has $4C^2$ weights and the $4C\to C$ Linear has another $4C^2$.
- **Total: $4C^2 + 8C^2 = 12C^2$ per layer**, times $n_{\text{layer}}$. (Biases and LayerNorm weights are $O(C)$, negligible next to $C^2$.)

Tiny check with $C = 4$: $W_Q, W_K, W_V, W_O$ are each $4\times 4 = 16$ weights → 64 $= 4\cdot 4^2$; the MLP is $4\times 16 + 16\times 4 = 128 = 8\cdot 4^2$; total 192 $= 12\cdot 4^2$.

Embeddings are excluded because they are a lookup (no matmul) and contribute negligible FLOPs.

### What the 6N rule leaves out: attention scores

The 6N rule counts only matmuls **against weights** — one multiply-add per parameter per token. Attention also does two matmuls that involve **no weights at all**: both operands are activations computed from the current sequence.

1. **Scores $QK^\top$:** the query of the current token is dotted with the key of every token it attends to.
2. **Weighted sum $PV$:** the softmax probabilities $P$ are used to average the value vectors of those tokens.

Because these matmuls have no parameters, they contribute nothing to $N$, so a formula built from $N$ cannot see them. And their cost depends on something $N$ does not contain at all: the context length $T$. A model with the same $N$ costs the same 6N per token whether $T$ is 128 or 128,000 — but the score work grows linearly with $T$ per token.

**Counting it, per token, per layer** (all heads together have total width $C$):

- $QK^\top$: one query (length $C$) dotted with $T$ keys (length $C$) = $T\cdot C$ multiply-adds = $2TC$ FLOPs.
- $PV$: $T$ value vectors (length $C$) each scaled by a probability and summed = $T\cdot C$ multiply-adds = $2TC$ FLOPs.
- Forward = $4TC$. Backward is again ≈ 2× forward (a gradient for each of the two operands), so forward+backward = $12TC$.
- With a causal mask, token $t$ only attends to $t$ earlier tokens, so on average $\approx T/2$ keys, not $T$. That halves it: ≈ $6TC$ per layer, i.e. ≈ $6\cdot n_{\text{layer}}\cdot T\cdot C$ FLOPs per token for the whole model. (A naive implementation computes the full $T\times T$ score matrix and then masks it, so it actually spends the un-halved $12TC$; FlashAttention-style kernels skip the masked blocks.)

**Tiny example:** $C = 4$, $T = 3$, no mask. $QK^\top$ for one token: 3 dot products of length 4 = 12 multiply-adds = 24 FLOPs. $PV$: 3 vectors of length 4 scaled and summed = 12 multiply-adds = 24 FLOPs. Forward = 48 $= 4\cdot 3\cdot 4$; forward+backward = 144 $= 12\cdot 3\cdot 4$. None of these 144 FLOPs touches a weight, so none of them is in $6N$.

**How big is it relative to 6N?** Divide the two per-layer costs:

$$
\frac{6\,T\,C}{6\cdot 12C^2} = \frac{T}{12C}.
$$

So the score term is small when $T \ll 12C$ and equals the whole 6N cost when $T = 12C$. For GPT-2 small ($C = 768$, $T = 1024$): $1024 / 9216 \approx 0.11$, i.e. the scores add ≈ 11% on top of 6N ($6\cdot 12\cdot 1024\cdot 768 \approx 5.7\times 10^7$ vs $6N \approx 5.1\times 10^8$ FLOPs/token). For a large model with $C = 4096$ at $T = 4096$, it is $4096/49152 \approx 8\%$. But at $T = 128{,}000$ with $C = 4096$ it is $\approx 2.6\times$ the 6N cost — the "ignored" term now dominates. This is one reason long-context training needs the efficient attention of [Module 8](lessons/module-08/lesson-01.md).

<div class="callout key"><p>6N counts weight matmuls, so it scales with parameters and is blind to context length. The attention-score matmuls ($QK^\top$, $PV$) multiply activations by activations, cost ≈ $6\cdot n_{\text{layer}}\cdot T\cdot C$ FLOPs per token (causal), and are ≈ $T/(12C)$ of the 6N cost — about 10% at ordinary context, dominant at very long context.</p></div>

## 3. Numerical example: GPT-2 small

GPT-2 small has $n_{\text{layer}} = 12$, $n_{\text{embd}} = 768$. The non-embedding estimate:

$$
N = 12 \cdot 12 \cdot 768^2 = 84{,}934{,}656 \approx 84.9\text{M}.
$$

(The *total* parameter count including the tied embeddings is 124,439,808 ≈ 124.4M, from [06.2](lessons/module-06/lesson-02.md); the true non-embedding count is 85,056,000, so the $12C^2$ estimate is within 0.15%.)

FLOPs per token:

$$
6N = 6 \cdot 84{,}934{,}656 = 509{,}607{,}936 \approx 5.1\times 10^8 \text{ FLOPs/token}.
$$

`model_flops_per_token(GPTConfig())` returns exactly `509607936.0`. For a step of 500,000 tokens (07.2), that is $5.1\times 10^8 \cdot 5\times 10^5 \approx 2.5\times 10^{14}$ FLOPs — 255 TFLOP — per optimizer step.

## 4. Throughput and MFU

**Tokens/sec** is the raw speedometer: measure wall-clock seconds for a step, divide the step's tokens (07.2) by it.

$$
\text{tokens/sec} = \frac{\text{tokens per step}}{\text{seconds per step}}.
$$

But tokens/sec alone does not tell you if you are *wasting* the machine — a bigger model is slower in tokens/sec yet may use the GPU better. **MFU (Model FLOPs Utilization)** normalizes by the hardware:

$$
\text{MFU} = \frac{\text{achieved FLOPs/s}}{\text{peak FLOPs/s}} = \frac{6N \cdot (\text{tokens/sec})}{\text{peak FLOPs/s}}.
$$

It is the fraction of the accelerator's advertised arithmetic throughput your run actually delivers. `mfu(flops_per_token, tokens_per_sec, peak_flops)` computes it.

### Worked MFU

An NVIDIA A100 peaks at ≈ $312$ TFLOP/s in bf16 (dense). Suppose one A100 trains GPT-2 small at 50,000 tokens/sec:

$$
\text{MFU} = \frac{5.096\times 10^8 \cdot 5\times 10^4}{3.12\times 10^{14}} = \frac{2.548\times 10^{13}}{3.12\times 10^{14}} \approx 0.082 = 8.2\%.
$$

Eight A100s at an aggregate 1,000,000 tokens/sec (peak $8\times 312$ TFLOP/s):

$$
\text{MFU} = \frac{5.096\times 10^8 \cdot 10^6}{2.496\times 10^{15}} \approx 0.204 = 20.4\%.
$$

<div class="callout key"><p>Large-scale LLM pretraining commonly runs at roughly <strong>30–50% MFU</strong> when well tuned. Treat this as <em>reasonable industry practice</em> reported across several public training write-ups — not a law and not a guarantee. The exact figure depends on model size, sequence length, hardware, and how much of the time is spent on non-matmul work (attention scores, communication, data loading). A small model at short context, like our GPT-2-small toy above, will sit lower because the fixed overheads are a larger fraction.</p></div>

Why MFU matters: at 40% MFU a step that does 255 TFLOP finishes in $\approx 2.0$ s on one A100; at 20% it takes twice as long and costs twice as much. Doubling MFU halves the bill. It is the single most useful number for spotting a run that is bottlenecked on something other than arithmetic (data loading, communication, tiny batches) — a low MFU says "the GPU is sitting idle, waiting for data or for other GPUs — go find the stall." (Engineers often call this "starving": idle not because it is done, but because its inputs have not arrived yet.)

## 5. Memory accounting: four buckets

Training memory is four contributions. Let $P$ be the total parameter count and assume fp32 (4 bytes each) for the accounting:

1. **Parameters** — $P$ numbers, $4P$ bytes. You must hold the weights.
2. **Gradients** — one per parameter, another $4P$ bytes. `loss.backward()` fills a `.grad` the same shape as every parameter. (Do not confuse this with the "2" and "4" in the FLOP count: those count *operations* — a multiply and an add — performed with each parameter for every token. Memory counts *stored numbers*: each weight $w$ has exactly one gradient $\partial L/\partial w$, and the backward pass keeps adding each token's contribution into that same slot, so millions of FLOPs end up in one 4-byte number. Of the 4 backward FLOPs per parameter, half compute $\partial L/\partial W$ (kept: it is the `.grad`, shape of $W$) and half compute $\partial L/\partial x$, the gradient w.r.t. the layer's *input*. That one has the shape of the activation, $(B, T, C)$, not of $W$; it is handed to the previous layer as its upstream gradient and freed right after, so it is a short-lived activation-sized buffer, never a per-parameter one.)
3. **Optimizer state** — AdamW keeps *two* buffers per parameter, $m$ and $v$ (03.2), so $2P$ numbers, $8P$ bytes. This is why people say "Adam **triples** your parameter memory": params + $m$ + $v$ = $3P$ numbers, and with gradients you are at $4P$.
4. **Activations** — the intermediate tensors the forward pass saves for backward. Unlike the first three, this scales with **batch × context**, not with $P$, and for language models the logits tensor $(B, T, V)$ often dominates.

<div class="callout key"><p>The first three buckets are fixed by the model: params ($P$) + grads ($P$) + Adam state ($2P$) = <strong>$4P$ numbers</strong>, independent of batch size. Activations are the only bucket you can trade against batch/context — which is exactly what gradient accumulation (07.2), activation checkpointing (Module 8), and mixed precision attack.</p></div>

### Worked memory: GPT-2 small in fp32

$P = 124{,}439{,}808 \approx 124.4$M. At 4 bytes:

$$
\begin{aligned}
\text{params} &= 4P \approx 0.498 \text{ GB} \\
\text{gradients} &= 4P \approx 0.498 \text{ GB} \\
\text{Adam } m, v &= 8P \approx 0.996 \text{ GB} \\
\hline
\text{fixed total} &= 16P \approx 1.99 \text{ GB (before activations)}.
\end{aligned}
$$

So ~2 GB is spoken for before a single activation. Now activations, for a batch $B = 8$, context $T = 1024$: the logits tensor alone is $B\cdot T\cdot V = 8\cdot 1024\cdot 50257 \approx 4.1\times 10^8$ floats $\approx 1.65$ GB in fp32 — nearly as much as all the fixed state combined, and that is just one tensor. Add the per-layer residual/attention activations (each $(B,T,C)$ tensor is ~25 MB, and there are dozens across 12 layers) and activations become the dominant, batch-scaling cost. This is the concrete reason a 124M model does not train in a batch of 8×1024 on a small GPU without help — and the motivation for everything in [Module 8](lessons/module-08/lesson-01.md).

<div class="callout pt"><p>Two standard savings, previewed: <strong>mixed precision</strong> stores activations (and often a weight copy) in bf16/fp16 at 2 bytes, roughly halving the activation and parameter-copy memory; and <strong>activation checkpointing</strong> discards most activations during forward and recomputes them during backward, trading extra FLOPs for much lower activation memory. Both are Module 8. Neither changes the $4P$ fixed cost, which is attacked instead by optimizer-state sharding (ZeRO/FSDP, Module 9).</p></div>

## 6. Under the hood: why these estimates are trustworthy

The 6N rule is an *estimate*, deliberately dropping the attention-score FLOPs, the softmax, LayerNorm, and the embedding lookups. It is trusted because in a standard-shaped transformer at moderate context those omitted terms are around 10% of the total or less — the matmuls against weights genuinely dominate. When they stop dominating (very long context, very small models, MoE routing), you switch to a fuller FLOP count; the metrics module's docstring flags exactly this. The point of the estimate is not perfect accuracy but a number you can compute from `cfg` alone, before writing any training code, that is right to within about 10% at ordinary context — enough to budget GPU-hours and choose a model size.

The memory buckets are exact for the first three (you can count $P$ and multiply by the dtype size) and estimated for activations (which depend on the implementation's fusion and what it chooses to save). That asymmetry is why optimizer-state and parameter memory are predicted precisely in capacity planning, while activation memory is usually measured empirically with a short profiling run.

## Exercise

You want to train a model with $N = 1.3\times 10^9$ non-embedding parameters on $D = 26\times 10^9$ tokens. **(a)** Estimate the total training FLOPs. **(b)** On a cluster delivering an *effective* (MFU-adjusted) $1.0\times 10^{15}$ FLOPs/s, estimate the wall-clock time. **(c)** In fp32, how much memory do the parameters, gradients, and Adam state take together (ignore activations)?

<details><summary>Hint</summary>
(a) is $6ND$. (b) is total FLOPs ÷ effective FLOPs/s. (c) is $16P$ bytes with $P \approx N$ here (treat total ≈ non-embedding for the estimate).
</details>

<details><summary>Stronger hint</summary>
(a) $6 \cdot 1.3\text{e}9 \cdot 26\text{e}9$. (b) divide by $1\text{e}15$ and convert seconds to hours. (c) params $4P$ + grads $4P$ + Adam $8P$ = $16P$ bytes.
</details>

<details><summary>Solution</summary>

**(a)** Total FLOPs $\approx 6ND = 6 \cdot 1.3\times 10^9 \cdot 26\times 10^9 = 2.028\times 10^{20}$ FLOPs (≈ 203 exaFLOP).

**(b)** Time $= \dfrac{2.028\times 10^{20}}{1.0\times 10^{15}} = 2.028\times 10^5$ s $\approx 56$ hours $\approx 2.3$ days.

**(c)** With $P \approx 1.3\times 10^9$: fixed memory $= 16P = 16 \cdot 1.3\times 10^9 = 2.08\times 10^{10}$ bytes $\approx 20.8$ GB — just for params + grads + Adam $m,v$, before any activations. This already exceeds a 16 GB GPU and eats most of a 24 GB one, which is why billion-parameter training needs optimizer-state sharding or mixed precision (Modules 8–9). (Chinchilla-style, $D \approx 20N$ tokens is the compute-optimal ballpark — Module 10 — so $D = 26$B for $N = 1.3$B is roughly in that regime.)

</details>

## Common mistakes

- **Confusing FLOPs with FLOP/s.** FLOPs is a count (work); FLOP/s is a rate (speed). $6ND$ is a count; peak is a rate; MFU is a ratio of rates.
- **Forgetting backward is 2× forward.** Using $2N$ instead of $6N$ underestimates cost threefold.
- **Ignoring optimizer state in memory planning.** Adam's $m,v$ double your parameter memory; people who budget only for weights + grads run out of memory at the optimizer step.
- **Comparing MFU across different hardware peaks without saying which peak.** Always state the dtype and the peak FLOP/s you divided by (dense vs. sparse, fp16 vs. fp32).

## Check yourself

<details><summary>A forward pass costs ≈ $2N$ FLOPs/token. Why is the backward pass ≈ $4N$ and not also $2N$?</summary>

Backward computes two gradients per matmul: the gradient w.r.t. the layer's input (to keep back-propagating) and the gradient w.r.t. the weights (what the optimizer uses). Each is about as expensive as the forward matmul, so backward ≈ $2\times 2N = 4N$, giving the $6N$ total.

</details>

<details><summary>Your run reports 8% MFU on a single GPU. Give two plausible causes and why they lower MFU.</summary>

Possible causes: (1) micro-batch too small, so the GPU is under-utilized between kernel launches; (2) time lost to data loading, host↔device transfers, or (multi-GPU) gradient communication; (3) a small model / short context where non-matmul work (attention scores, LayerNorm, Python overhead) is a large fraction. All of these mean the arithmetic units sit idle part of each step, so achieved FLOPs/s falls below peak.

</details>

<details><summary>In fp32, how much memory do parameters + gradients + Adam state take for a 350M-parameter model, ignoring activations?</summary>

$16P = 16 \cdot 350\times 10^6 = 5.6\times 10^9$ bytes ≈ 5.6 GB. (Params $4P$ ≈ 1.4 GB, grads $4P$ ≈ 1.4 GB, Adam $m,v$ $8P$ ≈ 2.8 GB.)

</details>

<details><summary>Why does activation memory, unlike the other three buckets, depend on batch size and context length?</summary>

Parameters, gradients, and optimizer state have exactly one number per parameter regardless of the batch — they describe the *model*. Activations are the intermediate tensors of the forward pass, and there is one set per example per position, so their size scales with $B\times T$ (and, for logits, $\times V$). That is why batch/context are the knobs for trading activation memory.

</details>

<div class="hw">
<p><strong>Hardware track — measuring throughput.</strong></p>
<p><strong>Minimum / recommended:</strong> the FLOP/MFU/memory arithmetic is pure Python — any machine, instant, no GPU. <strong>To measure real tokens/sec and MFU</strong> you need the target accelerator: e.g. one A100 (≈312 TFLOP/s bf16) for GPT-2-small-scale experiments. <strong>GPU-hours:</strong> 0 for the estimates; a real GPT-2-small pretraining run is on the order of tens to hundreds of A100-hours depending on token budget. <strong>CPU-only:</strong> yes for all the accounting in this lesson; measured MFU obviously requires the GPU you are profiling.</p>
</div>

## Next

You now have the full pretraining picture: the loop (07.1), how to size and split batches (07.2), how to save and resume exactly (07.3), and how to reason about speed, cost, and memory (07.4). The memory accounting here — activations dominating, the $4P$ fixed floor — is precisely the pressure that the **memory hierarchy and efficient attention** of the next module relieve, and the $6ND$ compute rule is the foundation the **scaling laws** of Module 10 build on.

Continue to [08.1 · The GPU memory hierarchy](lessons/module-08/lesson-01.md), and see the scaling-law payoff in [10.1](lessons/module-10/lesson-01.md).
