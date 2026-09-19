# 10.1 · FLOPs & parameter counting

<div class="prereq">
<p><strong>Prerequisites:</strong> the parameter count and where compute/memory go from <a href="../module-06/lesson-02.md">06.2 · Init, forward pass, parameter count</a>; the 6N rule, MFU, and the four memory buckets from <a href="../module-07/lesson-04.md">07.4 · Throughput, FLOPs, MFU, memory accounting</a>; AdamW's two moment buffers from <a href="../module-03/lesson-02.md">03.2 · RMSProp, Adam, AdamW</a>.</p>
<p><strong>You will learn:</strong> how to turn a model config into three back-of-the-envelope numbers before you launch anything — a <strong>closed-form parameter count</strong> $N$, the <strong>total training compute</strong> $C \approx 6ND$, and the <strong>memory footprint</strong> (params + grads + optimizer state + activations) — and how to convert $C$ into GPU-hours under a clearly stated throughput assumption. These are the inputs the scaling laws of the next two lessons consume.</p>
<p><strong>Why this matters for ML:</strong> every real training run starts as a spreadsheet. Before a lab spends a million dollars of GPU time, someone estimates $N$, $D$, $C$, the wall-clock, and whether the model even fits in memory — all from the formulas here. Get these wrong and you either run out of memory at step 0 or discover three weeks in that the run will take a year. Scaling laws (10.2, 10.3) are the theory of <em>how to choose</em> $N$ and $D$; this lesson is the arithmetic they are written in.</p>
</div>

## 1. Intuition: three numbers from a config

You have a `GPTConfig` — a handful of integers fixing the shape of every weight. From those integers alone, before allocating a single tensor, you can predict:

- **$N$** — how many parameters the model has. Sets model quality and memory.
- **$C \approx 6ND$** — how much arithmetic training will cost, for $D$ tokens. Sets wall-clock and dollars.
- **memory** — params + gradients + optimizer state + activations. Sets whether it fits on the GPU at all.

Module 7 introduced the $6N$ rule and the four memory buckets in the context of *one running loop*. Here we lift them to *planning*: given only a config and a token budget, compute all three, and get each right to within a few percent by hand.

<div class="callout key"><p>The whole lesson is three formulas: <strong>$N \approx VC + 12\,n_{\text{layer}}\,C^2$</strong> (parameters), <strong>$C \approx 6ND$</strong> (training FLOPs), and <strong>memory $\approx 16N$ bytes + activations</strong> (fp32). Everything else is naming the terms and plugging in numbers.</p></div>

## 2. Closed-form parameter count

A GPT's parameters live in a few named places. Let $V$ be the vocabulary size, $C$ the embedding width (`n_embd`), $T_{\max}$ the maximum context (`block_size`), and $L$ the number of layers (`n_layer`). Walking the model of [06.1](../module-06/lesson-01.md) top to bottom:

**Embeddings.**
- Token embedding `wte`: a $V \times C$ table — one $C$-vector per vocabulary entry. That is $VC$ parameters.
- Positional embedding `wpe`: a $T_{\max} \times C$ table — one $C$-vector per position. That is $T_{\max}\,C$ parameters.

**Each of the $L$ transformer blocks** (from [05.4](../module-05/lesson-04.md)):
- Attention has four $C \times C$ projection matrices — $W_Q, W_K, W_V$ and the output projection $W_O$ — giving $4C^2$ weights.
- The MLP has two Linears, $C \to 4C$ and $4C \to C$, giving $4C^2 + 4C^2 = 8C^2$ weights.
- So each block is about $4C^2 + 8C^2 = 12C^2$ weights, plus small terms: the biases ($\sim 9C$ per block) and the two LayerNorm gain/bias pairs ($4C$ per block). Those linear-in-$C$ terms are tiny next to the $C^2$ terms and we drop them.

**Final LayerNorm `ln_f`** ($2C$) and the **output head `lm_head`**, which is *tied* to `wte` (same tensor — [06.1](../module-06/lesson-01.md)), so it adds **zero** new parameters.

Summing the dominant terms:

$$
N \;\approx\; \underbrace{VC}_{\text{token emb}} \;+\; \underbrace{T_{\max}\,C}_{\text{pos emb}} \;+\; \underbrace{12\,L\,C^2}_{\text{blocks}}.
$$

The $12LC^2$ block term is the one the scaling literature calls the **non-embedding** parameter count $N$, because it is what actually does matmul work (embeddings are lookups). This is exactly `non_embedding_params(cfg)` from `llmre.training.metrics`.

<div class="callout key"><p>Per layer: attention's four $C\times C$ projections give $4C^2$, the MLP's two Linears give $8C^2$, total $12C^2$. Over $L$ layers, $12LC^2$ — the non-embedding parameter count. Add $VC + T_{\max}C$ for the embedding tables to get the total.</p></div>

### Numerical example: GPT-2 small

$V = 50257$, $C = 768$, $T_{\max} = 1024$, $L = 12$.

$$
\begin{aligned}
VC &= 50257 \cdot 768 = 38{,}597{,}376 \\
T_{\max}\,C &= 1024 \cdot 768 = 786{,}432 \\
12\,L\,C^2 &= 12 \cdot 12 \cdot 768^2 = 84{,}934{,}656 \\
\hline
N_{\text{approx}} &= 124{,}318{,}464 \approx 124.3\text{M}.
\end{aligned}
$$

The *exact* count from actually building the model, `GPT(GPTConfig()).num_params()`, is **124,439,808** ≈ 124.4M. Our closed form is within **0.1%** — the dropped biases and LayerNorm gains ($\approx 121{,}000$ parameters) are all that is missing. The non-embedding figure $12LC^2 = 84.9$M compares to the true non-embedding count of 85,056,000 (within 0.15%). Two integers and a multiply predict a 124-million-parameter model's size to three significant figures.

<div class="callout pt"><p>The exact count comes straight from the model: <code>sum(p.numel() for p in model.parameters())</code>. Because <code>parameters()</code> de-duplicates the tied <code>wte</code>/<code>lm_head</code> tensor, it is counted once — which is why weight tying saves $VC \approx 38.6$M parameters versus an untied head. Use the closed form for planning; use <code>num_params()</code> for the truth once the model exists.</p></div>

## 3. Training compute: the 6ND rule

From [07.4](../module-07/lesson-04.md): one forward+backward pass costs about $6N$ FLOPs **per token** — $2N$ forward (one multiply-add per weight, counted as 2 FLOPs) and $4N$ backward (two matmuls per weight: gradient-w.r.t.-input and gradient-w.r.t.-weight). Multiply by the total tokens $D$ the run processes:

$$
C \;\approx\; 6 N D \quad\text{FLOPs (total training compute)}.
$$

This is the single most important equation in the module. $N$ here is the non-embedding count, and the estimate deliberately omits the attention-score FLOPs ($\sim 6\,L\,T$ per token), which are small while $T \ll 12C$. `training_flops(N, D)` in `llmre.evaluation.scaling` is literally `return 6.0 * N * D`.

<div class="callout key"><p><strong>$C = 6ND$.</strong> Three symbols: 6 (2 forward + 4 backward), $N$ (non-embedding params), $D$ (training tokens). It is a FLOP <em>count</em> — work, not speed. Divide by an achieved FLOP/s <em>rate</em> to get time.</p></div>

### Numerical example: FLOPs and GPU-hours

Suppose we train a model with $N = 85\times 10^6$ non-embedding parameters (GPT-2-small scale) on $D = 1.7\times 10^9$ tokens — a Chinchilla-ish 20 tokens/param, which lesson 10.3 will justify. The compute:

$$
C = 6 \cdot 85\times 10^6 \cdot 1.7\times 10^9 = 8.67\times 10^{17}\ \text{FLOPs}.
$$

To turn that into wall-clock we need a **throughput rate**, and here we must be explicit about what is public and what is our assumption:

- **PUBLICLY DOCUMENTED:** an NVIDIA A100's advertised peak is $\approx 312$ TFLOP/s ($3.12\times 10^{14}$ FLOP/s) in bf16 dense. That is a spec-sheet number.
- **ASSUMPTION (ours):** that our run achieves an MFU (07.4) of about **32%**, i.e. an *achieved* $\approx 1.0\times 10^{14}$ FLOP/s. This is a stated modelling assumption, not a measured or vendor figure — MFU depends on model size, context length, and how well the pipeline is tuned.

Under that assumption:

$$
t = \frac{C}{\text{achieved FLOP/s}} = \frac{8.67\times 10^{17}}{1.0\times 10^{14}} = 8{,}670\ \text{s} \approx 2.4\ \text{GPU-hours}.
$$

If the pipeline is poorly tuned and achieves only $2\times 10^{13}$ FLOP/s (6.4% MFU), the same run takes $\approx 12$ GPU-hours — five times longer for the identical arithmetic. **Always state the achieved-FLOP/s assumption**; the FLOP count is objective, the time is not.

## 4. Memory: the four buckets, in one number

Also from [07.4](../module-07/lesson-04.md), training memory is four contributions. With $P$ the total parameter count, in fp32 (4 bytes each):

1. **Parameters** — $4P$ bytes.
2. **Gradients** — one `.grad` per parameter, another $4P$ bytes.
3. **Optimizer state** — AdamW keeps two buffers $m$ and $v$ per parameter (03.2), so $2\times$ params: $8P$ bytes.
4. **Activations** — intermediate forward tensors saved for backward; scale with **batch × context**, not with $P$.

The first three are fixed by the model and sum to $4P + 4P + 8P = 16P$ **bytes** in fp32 — a floor you pay before a single activation.

$$
\text{fixed memory} \approx 16 P \text{ bytes (fp32)}.
$$

<div class="callout key"><p>Params + grads + Adam ($m,v$) = <strong>$16P$ bytes in fp32</strong> (4 + 4 + 8), independent of batch size. Activations are the only bucket that scales with batch/context — the one gradient accumulation, checkpointing, and mixed precision attack.</p></div>

### Numerical example: GPT-2 small

$P = 124{,}439{,}808 \approx 124.4$M. Fixed memory:

$$
16P = 16 \cdot 1.244\times 10^8 \approx 1.99\times 10^9 \text{ bytes} \approx 1.99\ \text{GB}.
$$

So ~2 GB is gone before activations. And activations are not small: the logits tensor alone, at batch $B = 8$ and context $T = 1024$, is $B\cdot T\cdot V = 8\cdot 1024\cdot 50257 \approx 4.12\times 10^8$ floats $\approx 1.65$ GB in fp32 — comparable to the entire fixed cost, from one tensor. This is why even a "small" 124M model needs care to fit, and the motivation for [Module 8](../module-08/lesson-01.md) (mixed precision, checkpointing) and [Module 9](../module-09/lesson-02.md) (optimizer-state sharding).

## 5. Under the hood: why the estimates are trustworthy

The $6ND$ compute rule and the $16P$ memory floor are *estimates*, but disciplined ones. The compute rule drops the attention-score term, the softmax, LayerNorm, and embedding lookups — all a few percent of the total in a standard-shaped transformer at moderate context, because the weight matmuls genuinely dominate. When they stop dominating (very long context, tiny models, MoE routing) you switch to a fuller count. The memory floor is *exact* for its first three buckets (you can count $P$ and multiply by the dtype size); only activations must be measured empirically, because they depend on which intermediate tensors the implementation chooses to save. That asymmetry is why capacity planning predicts optimizer/parameter memory to the byte but profiles activation memory with a short run.

The reason any of this matters — the reason a whole module rests on $C = 6ND$ — is that it makes the central planning question *analytic*: given a compute budget $C$, how should you split it between a bigger model (larger $N$) and more data (larger $D$)? You cannot even ask that until you can write $C$ in terms of $N$ and $D$. That is exactly what 10.2 and 10.3 answer.

## 6. The code

`llmre.evaluation.scaling.training_flops` and the Module-7 helpers in `llmre.training.metrics` are all you need:

```python
from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.training.metrics import non_embedding_params, model_flops_per_token
from llmre.evaluation.scaling import training_flops

cfg = GPTConfig()                       # GPT-2 small
model = GPT(cfg)

N_exact = model.num_params()            # 124_439_808  (total, tied head counted once)
N_nonemb = non_embedding_params(cfg)    # 84_934_656   (12 * n_layer * n_embd**2)

flops_per_tok = model_flops_per_token(cfg)      # 6 * N_nonemb = 509_607_936.0
C = training_flops(N_nonemb, D=1.7e9)           # 6 N D = 8.67e17 FLOPs
```

`training_flops` is deliberately trivial — the value is in *knowing to call it before launching a run*, with the right $N$ (non-embedding) and an honest $D$.

## Exercise

You are planning a larger model: $N = 850\times 10^6$ non-embedding parameters (10× the toy above), trained Chinchilla-style at 20 tokens/param, so $D = 1.7\times 10^{10}$ tokens (10× the data). **(a)** What is the training compute $C$, and how does it compare to the $8.67\times 10^{17}$ FLOPs of the 85M run? **(b)** At an *assumed* achieved $2.0\times 10^{14}$ FLOP/s (say, two well-tuned A100s), how many GPU-hours? **(c)** In fp32, what is the fixed memory floor ($16N$, treating $P \approx N$)?

<details><summary>Hint</summary>
(a) $C = 6ND$; both $N$ and $D$ grew 10×, so think about what happens to their product. (b) time $=C / \text{achieved FLOP/s}$, then $\div 3600$. (c) $16N$ bytes.
</details>

<details><summary>Stronger hint</summary>
(a) $6 \cdot 8.5\text{e}8 \cdot 1.7\text{e}10$; and $10\times$ params $\times$ $10\times$ data $= 100\times$ compute. (b) divide by $2\text{e}14$, then by 3600. (c) $16 \cdot 8.5\text{e}8$ bytes $\to$ GB.
</details>

<details><summary>Solution</summary>

**(a)** $C = 6 \cdot 8.5\times 10^{8} \cdot 1.7\times 10^{10} = 8.67\times 10^{19}$ FLOPs. That is exactly **100×** the 85M run's $8.67\times 10^{17}$: scaling $N$ and $D$ each by 10 multiplies compute by $10\times 10 = 100$. This quadratic-in-scale blow-up of compute is the whole tension the scaling laws resolve — you cannot make the model 10× bigger *and* feed it 10× the data without 100× the budget.

**(b)** $t = \dfrac{8.67\times 10^{19}}{2.0\times 10^{14}} = 4.335\times 10^{5}$ s $\approx 120$ GPU-hours $\approx 5$ days on two A100s (under the stated 2e14 FLOP/s assumption).

**(c)** $16N = 16 \cdot 8.5\times 10^{8} = 1.36\times 10^{10}$ bytes $\approx 13.6$ GB — params + grads + Adam $m,v$ alone, before activations. That already fills a 16 GB GPU, so this run needs mixed precision (Module 8) or optimizer-state sharding (Module 9) even ignoring activations.

</details>

## Common mistakes

- **Using total params in $6ND$ instead of non-embedding.** The scaling convention uses the non-embedding $N = 12LC^2$; the embeddings are a lookup with negligible FLOPs.
- **Confusing the FLOP count with the time.** $6ND$ is objective work; the wall-clock depends on an achieved-FLOP/s rate you must state (and which hides MFU).
- **Budgeting memory for weights only.** Adam's $m,v$ double the parameter memory; the honest floor is $16P$ bytes in fp32, not $4P$.
- **Forgetting activations scale with batch×context.** The fixed $16P$ is not the whole story — the logits tensor $(B,T,V)$ can rival it on its own.

## Check yourself

<details><summary>A model has $n_{\text{layer}}=24$, $n_{\text{embd}}=1024$. Estimate its non-embedding parameter count.</summary>

$12\,L\,C^2 = 12 \cdot 24 \cdot 1024^2 = 12 \cdot 24 \cdot 1{,}048{,}576 = 301{,}989{,}888 \approx 302$M non-embedding parameters.

</details>

<details><summary>You double both the model size $N$ and the token budget $D$. By what factor does training compute change?</summary>

$C = 6ND$, so doubling each multiplies $C$ by $2 \times 2 = 4$. Compute grows with the *product* — this is why "just make it bigger and train longer" is so expensive, and why choosing the split of a fixed budget (10.3) matters.

</details>

<details><summary>In fp32, how much fixed memory (params + grads + Adam state) does a 300M-parameter model need, ignoring activations?</summary>

$16P = 16 \cdot 3.0\times 10^8 = 4.8\times 10^9$ bytes $\approx 4.8$ GB. (Params $4P\approx1.2$ GB, grads $4P\approx1.2$ GB, Adam $m,v$ $8P\approx2.4$ GB.)

</details>

<details><summary>A run does $C = 5\times 10^{20}$ FLOPs and you want it done in 24 hours. What achieved FLOP/s must the cluster sustain?</summary>

$24$ h $= 86{,}400$ s. Required rate $= 5\times 10^{20} / 8.64\times 10^{4} \approx 5.8\times 10^{15}$ FLOP/s $= 5.8$ PFLOP/s. At ~30% MFU on A100s (312 TFLOP/s peak, ~$10^{14}$ achieved each) that is roughly $5.8\times 10^{15} / 10^{14} \approx 58$ GPUs.

</details>

<div class="hw">
<p><strong>Hardware track — this lesson.</strong></p>
<p><strong>Minimum / recommended:</strong> all the arithmetic (parameter count, $6ND$, memory floor) is pure Python — any machine, instant, no GPU. <strong>Runtime:</strong> milliseconds. <strong>GPU memory / GPU-hours:</strong> 0 — this is the lesson you do <em>before</em> spending any. <strong>CPU-only:</strong> yes, entirely. The GPU only enters when you want to <em>measure</em> a real achieved FLOP/s to replace the assumption in §3.</p>
</div>

## Next

You can now size a model, estimate its training compute, and check it fits in memory — all from a config. The obvious next question is the one every lab asks: for a fixed compute budget, does the loss get lower faster by making the model bigger or by training on more tokens? The answer is empirical — loss follows a **power law** in $N$, $D$, and $C$ — and the next lesson has you *fit one yourself* from a tiny sweep.

Continue to [10.2 · Kaplan scaling laws; fit a curve](lesson-02.md).
