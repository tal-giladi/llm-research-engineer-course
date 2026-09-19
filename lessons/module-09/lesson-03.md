# 09.3 · Tensor / pipeline / expert parallelism

<div class="prereq">
<p><strong>Prerequisites:</strong> ZeRO/FSDP sharding of the model-state buckets and the all-gather / reduce-scatter collectives from <a href="lesson-02.md">09.2 · ZeRO &amp; FSDP</a>; all-reduce and the data-parallel picture from <a href="lesson-01.md">09.1 · Data parallelism, all-reduce, DDP</a>; the Transformer block — attention + MLP, the MLP's $C\to 4C$ and $4C\to C$ Linears — from <a href="../module-06/lesson-01.md">06.1 · Assembling GPT-2</a>.</p>
<p><strong>You will learn:</strong> the parallelism you reach for when the model — or even a single layer — is too big to fit or compute on one device: <strong>tensor parallelism</strong> (split a matmul's weight across devices and all-reduce the partial results — the Megatron column-then-row split for an MLP, worked with exact numbers); <strong>pipeline parallelism</strong> (assign layer ranges to stages, feed micro-batches to fill the pipe, and the "bubble" that micro-batches shrink); and a shorter tour of <strong>sequence/context parallelism</strong> and <strong>expert parallelism</strong> for MoE. Plus which framework (Megatron-LM, DeepSpeed, FSDP) solves which problem, and what you'd implement yourself.</p>
<p><strong>Why this matters for ML:</strong> data parallelism and ZeRO get you far, but frontier models are trained with all of these at once — a "3D parallelism" of data × tensor × pipeline, with expert parallelism on top for MoE. Knowing what each axis splits, and what it costs in communication, is how you read a large-model training report and how you'd plan one.</p>
</div>

<div class="callout warn"><p><strong>Single-CPU note.</strong> No real multi-device run is possible here, so the tensor-parallel example below is <em>simulated in one process</em> — we compute each "device's" partial result with plain NumPy and sum them, exactly as an all-reduce would. That reproduces the numbers; the point of real tensor parallelism (fitting a layer that doesn't fit on one device, and computing it faster) can only be shown on real hardware, noted in the <code>&lt;div class="hw"&gt;</code> box.</p></div>

## 1. When data parallelism and ZeRO run out

Data parallelism ([09.1](lesson-01.md)) replicates the model and splits the *data*. ZeRO/FSDP ([09.2](lesson-02.md)) shard the model's *memory* but still run each layer's full computation on one device (after gathering its weights). Both assume a single device can, at least momentarily, hold and compute one layer. When that assumption breaks — a layer's weight matrix or its activations are simply too large for one device — you must split the **model itself** across devices. There are three axes to split along, and real runs use them together:

- **Tensor parallelism** — split an *individual operation* (a matmul) across devices. Each device holds part of a weight matrix and computes part of the output.
- **Pipeline parallelism** — split the *layer stack* across devices. Each device holds a contiguous range of layers and passes activations to the next.
- **Expert parallelism** — for Mixture-of-Experts, split the *experts* across devices. Each device holds some experts; tokens are routed to wherever their expert lives.

<div class="callout key"><p>Three ways to cut up a model: <strong>tensor parallelism</strong> splits <em>within</em> a layer (across the width of a matmul), <strong>pipeline parallelism</strong> splits <em>across</em> layers (depth), and <strong>expert parallelism</strong> splits across the experts of an MoE. Data parallelism / ZeRO (splitting the data / the memory) compose on top of all three.</p></div>

## 2. Tensor parallelism: split the matmul (Megatron style)

The dominant compute in a Transformer is matmuls against weight matrices ([07.4](../module-07/lesson-04.md)). Tensor parallelism splits one such matmul across `W` devices so no device holds the whole weight. The elegant part, from **Megatron-LM**, is choosing *how* to split so that the two matmuls of an MLP need only **one** communication between them.

### The column-then-row trick for an MLP

A Transformer MLP is two matmuls with a nonlinearity between them ([06.1](../module-06/lesson-01.md)):

$$
Y = \operatorname{ReLU}(X A)\, B,
$$

where $A$ is the $C \to 4C$ "up" projection and $B$ is the $4C \to C$ "down" projection. (We use ReLU here for a clean worked example; GPT-2 uses GELU — the splitting is identical.) Megatron splits the two weights *complementarily*:

- **Column-parallel on $A$.** Split $A$ by its **columns** into $[A_0 \mid A_1 \mid \dots]$, one block of columns per device. Device $d$ computes $X A_d$, producing its own slice of the hidden activations. Because ReLU is applied **element-wise**, each device can apply it to its own slice *independently* — no communication needed. This is why the split is placed before the nonlinearity.
- **Row-parallel on $B$.** Split $B$ by its **rows** into $\left[\begin{smallmatrix}B_0\\B_1\\\vdots\end{smallmatrix}\right]$, matching the hidden slices. Device $d$ computes $\operatorname{ReLU}(X A_d)\, B_d$, which is a **partial** output — it has the full output shape but contains only device $d$'s contribution.
- **All-reduce to finish.** The full output is the *sum* of the partials: $Y = \sum_d \operatorname{ReLU}(X A_d)\, B_d$. One all-reduce ([09.1](lesson-01.md)) sums them across devices, and every device ends with the complete $Y$.

The reason this pairing is bandwidth-thrifty: the intermediate $4C$-wide hidden activations never have to be communicated (each device keeps its own slice through the ReLU); only the final $C$-wide output is all-reduced, once per MLP in the forward pass (and once in backward).

<div class="callout key"><p>Megatron MLP: <strong>column-split the first matmul</strong> (so the element-wise nonlinearity stays local), <strong>row-split the second</strong> (so each device makes a partial output), then <strong>all-reduce the partials into the full output</strong>. One collective per MLP per pass — the wide hidden layer is never communicated.</p></div>

### Worked example, by hand

Take one token, $C = 2$ so the hidden width is $4$, and split across `W = 2` devices. Let

$$
X = [1,\ 2], \qquad
A = \begin{bmatrix} 1 & 0 & 2 & 1 \\ 0 & 1 & 1 & 2 \end{bmatrix}, \qquad
B = \begin{bmatrix} 1 & 0 \\ 2 & 1 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}.
$$

**Single-device reference.** $XA = [1,\ 2,\ 4,\ 5]$, then $\operatorname{ReLU}$ leaves it unchanged (all positive), then $(XA)B = [1\cdot1 + 2\cdot2 + 4\cdot0 + 5\cdot1,\ \ 1\cdot0 + 2\cdot1 + 4\cdot1 + 5\cdot1] = [10,\ 11]$.

**Split across two devices.** Columns 1–2 of $A$ and rows 1–2 of $B$ go to device 0; columns 3–4 and rows 3–4 to device 1:

$$
A_0 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},\ 
B_0 = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix};
\qquad
A_1 = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix},\ 
B_1 = \begin{bmatrix} 0 & 1 \\ 1 & 1 \end{bmatrix}.
$$

- **Device 0:** $h_0 = \operatorname{ReLU}(X A_0) = [1,\ 2]$; partial $y_0 = h_0 B_0 = [1\cdot1+2\cdot2,\ \ 1\cdot0+2\cdot1] = [5,\ 2]$.
- **Device 1:** $h_1 = \operatorname{ReLU}(X A_1) = [4,\ 5]$; partial $y_1 = h_1 B_1 = [4\cdot0+5\cdot1,\ \ 4\cdot1+5\cdot1] = [5,\ 9]$.
- **All-reduce (sum):** $y_0 + y_1 = [5{+}5,\ 2{+}9] = [10,\ 11]$ — exactly the single-device output.

Each device only ever held **half** of $A$ and **half** of $B$ and computed on a hidden slice of width 2 instead of 4 — that is the point: a matmul too big for one device, split across two, reassembled with one all-reduce. Verified:

```python
import numpy as np
X = np.array([[1., 2.]])
A = np.array([[1.,0.,2.,1.], [0.,1.,1.,2.]]); B = np.array([[1.,0.],[2.,1.],[0.,1.],[1.,1.]])
relu = lambda z: np.maximum(z, 0)
single = relu(X @ A) @ B                                   # [[10. 11.]]
y0 = relu(X @ A[:, :2]) @ B[:2, :]                         # device 0 partial [[5. 2.]]
y1 = relu(X @ A[:, 2:]) @ B[2:, :]                         # device 1 partial [[5. 9.]]
print(single, y0 + y1, np.allclose(single, y0 + y1))      # [[10. 11.]] [[10. 11.]] True
```

<div class="callout pt"><p>Attention is tensor-parallelized the same way, and it splits even more naturally: give each device a subset of the <strong>attention heads</strong> (06.1 showed heads are independent). Each device runs its heads' full QKV+attention, and the output projection is row-parallel, finished with one all-reduce — same column-then-row shape as the MLP. Because tensor parallelism all-reduces on <em>every</em> layer's forward and backward, it is communication-heavy and is normally confined to devices with the fastest link (the GPUs <em>within</em> one node, over NVLink), with pipeline and data parallelism used across nodes.</p></div>

## 3. Pipeline parallelism: split the layer stack

Tensor parallelism splits *within* a layer. **Pipeline parallelism** splits *across* layers: assign a contiguous range of layers to each device ("stage"). With 4 stages and 32 layers, stage 0 holds layers 0–7, stage 1 holds 8–15, and so on. A forward pass flows stage 0 → stage 1 → … → stage 3; backward flows in reverse. Each device stores and computes only its slice of the depth, so a model far deeper than one device can hold now fits.

### The bubble

The naive version is badly inefficient. If you push one batch through, then while stage 0 computes, stages 1–3 sit idle waiting for its output; when the batch reaches stage 3, stages 0–2 are idle. Most devices are idle most of the time — the wasted time is the **pipeline bubble**.

The fix is to split the batch into `m` **micro-batches** and feed them in a staggered stream (this reuses the micro-batch idea of [07.2](../module-07/lesson-02.md)). As soon as stage 0 finishes micro-batch 1 and passes it to stage 1, it starts micro-batch 2 — so after a short fill, *all* stages are busy on different micro-batches at once, like an assembly line. The bubble is the fill + drain time at the ends of the pipe.

For `S` stages and `m` micro-batches, the bubble is the fraction of time lost to fill/drain:

$$
\text{bubble fraction} = \frac{S - 1}{m + S - 1}.
$$

The intuition: it takes $S-1$ "slots" to fill the pipe before the last stage starts, and another $S-1$ to drain it, spread over a schedule of $m + S - 1$ slots. More micro-batches `m` amortize the fixed $S-1$ fill/drain over more useful work, shrinking the bubble.

### Worked bubble

With `S = 4` stages:

```python
S = 4
for m in (1, 4, 8, 16, 32):
    print(f"m={m:2d}  bubble = {(S-1)/(m+S-1):.3f}")
# m= 1  bubble = 0.750
# m= 4  bubble = 0.429
# m= 8  bubble = 0.273
# m=16  bubble = 0.158
# m=32  bubble = 0.086
```

One micro-batch wastes 75% of the hardware; 8 micro-batches cut it to 27%; 32 to under 9%. This is why pipeline schedules use many micro-batches — and why more sophisticated schedules (GPipe, and the interleaved "1F1B" of Megatron/DeepSpeed) exist to shrink the bubble and the activation memory further.

<div class="callout key"><p>Pipeline bubble $= \dfrac{S-1}{m + S - 1}$ for $S$ stages and $m$ micro-batches. The bubble is fixed fill/drain overhead; raising $m$ amortizes it. Rule of thumb: use $m \gg S$ (e.g. $m \geq 4S$) to keep the bubble small.</p></div>

<div class="callout warn"><p>Pipeline parallelism's tradeoffs: micro-batches must be numerous to hide the bubble, but each in-flight micro-batch keeps its activations alive for its backward pass, so activation memory grows with the number of concurrent micro-batches — the schedule (GPipe vs. 1F1B) is largely about managing that. And a stage stalls if its input isn't ready, so <strong>load-balancing the layers across stages</strong> (equal compute per stage) matters; an uneven split leaves fast stages idling on slow ones.</p></div>

## 4. Sequence / context parallelism and expert parallelism

Two more axes, in brief.

**Sequence (context) parallelism** splits along the *sequence length* `T` rather than the batch or the width. Each device handles a slice of the token positions. This targets the parts of the Transformer whose memory grows with `T` — the attention scores and the per-token activations of LayerNorm/dropout — which tensor parallelism does not shard. It is what makes very long context lengths trainable, and it requires communicating across the sequence dimension inside attention (each position's query must still see all keys). It composes with tensor parallelism (Megatron combines the two).

**Expert parallelism** is specific to **Mixture-of-Experts** models. An MoE layer replaces the single MLP with many expert MLPs and a router that sends each token to only one or a few experts (you will build this in [Module 12.4](../module-12/lesson-04.md)). Expert parallelism places different experts on different devices; the router then performs an **all-to-all** communication — every device sends each of its tokens to whichever device holds that token's chosen expert, and sends the results back. Because each token uses only a couple of experts, an MoE can have a huge total parameter count while each device stores and computes only its slice of the experts.

<div class="callout key"><p><strong>Sequence parallelism</strong> splits the token axis <code>T</code> (for long context and to shard the activations tensor parallelism leaves whole). <strong>Expert parallelism</strong> splits an MoE's experts across devices and routes tokens to them with an all-to-all — huge total parameters, small per-device compute. Both stack on top of tensor / pipeline / data parallelism.</p></div>

## 5. Communication/computation overlap, and the frameworks

Every parallelism axis adds communication, and the universal performance trick — seen already with DDP (09.1) and FSDP (09.2) — is to **overlap it with computation**: start a device's next matmul (or all-gather the next pipeline stage's input) while the current collective is still in flight, so the network cost hides under compute instead of adding to it. A run is well-tuned when the bottleneck is arithmetic, not idle waiting.

Real frontier training combines the axes — commonly called **3D parallelism**: tensor parallelism *within* a node (fast NVLink handles its per-layer all-reduces), pipeline parallelism *across* a few nodes (only activations cross the slower links), and data parallelism / ZeRO *across* the remaining groups (one gradient all-reduce per step). Expert parallelism adds a fourth axis for MoE. The mapping of axes to the hardware topology — heavy communication on fast links — is the core of large-scale training system design.

What each framework gives you:

- **Megatron-LM** (NVIDIA) — the reference implementation of **tensor** parallelism (the column-then-row split of section 2) and **sequence** parallelism, plus interleaved pipeline schedules. Reach for it when you need to split the model's math across devices.
- **DeepSpeed** (Microsoft) — home of **ZeRO** (09.2) and pipeline parallelism, with heavy optimizer/memory features (ZeRO-Offload, etc.). Reach for it for memory-efficient data parallelism and easy ZeRO stages.
- **PyTorch FSDP** — native **ZeRO-3-style** fully-sharded data parallelism (09.2), increasingly with tensor-parallel support. Reach for it to shard a model across data-parallel workers with minimal code.

<div class="callout pt"><p><strong>What you'd implement yourself vs. use a framework for.</strong> You have now built the primitives that these frameworks are made of: a ring all-reduce, an all-gather, and simulated data-parallel gradient averaging (<code>llmre/distributed/</code>). Implementing <em>correct</em> production versions — topology-aware NCCL collectives, overlapped communication, interleaved pipeline schedules, all-to-all expert routing, fault tolerance across thousands of GPUs — is a large systems effort, which is exactly why Megatron/DeepSpeed/FSDP exist. In practice you <em>configure</em> these axes and <em>reason</em> about their memory and communication cost (this module); you rarely re-implement the collectives. The value of building the toy versions is that the config options — <code>tensor_parallel_size</code>, <code>pipeline_parallel_size</code>, <code>zero_stage</code>, <code>num_micro_batches</code> — are no longer magic.</p></div>

## Research connection

<div class="callout paper"><p><strong>Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism</strong> (Shoeybi et al., 2019) introduces the column-then-row tensor-parallel split of section 2 (for both the MLP and multi-head attention) and shows it scaling to billions of parameters. Together with <strong>ZeRO</strong> (09.2) it is one of the two foundational large-model systems papers, and both are cited by essentially every frontier-model report since. Read it after this module; guide and link in <a href="../../papers/index.md">the paper curriculum</a>.</p></div>

## Exercise

You have 32 GPUs and a model that needs tensor parallelism degree 4 (a layer only fits when split across 4 GPUs) and whose 32 layers you want to spread over 4 pipeline stages. **(a)** How many GPUs do tensor × pipeline parallelism consume together, and how many independent data-parallel replicas can you run on the 32 GPUs? **(b)** With 4 pipeline stages, how many micro-batches do you need to keep the pipeline bubble below 10%?

<details><summary>Hint</summary>
(a) tensor-parallel size × pipeline-parallel size GPUs per replica; divide 32 by that. (b) solve $(S-1)/(m+S-1) < 0.10$ for $m$ with $S=4$.
</details>

<details><summary>Stronger hint</summary>
(a) $4 \times 4 = 16$ GPUs per replica. (b) $(4-1)/(m+3) < 0.1 \Rightarrow 3/(m+3) < 0.1 \Rightarrow m+3 > 30$.
</details>

<details><summary>Solution</summary>

**(a)** One replica of the model uses tensor-parallel size × pipeline-parallel size $= 4 \times 4 = 16$ GPUs. With 32 GPUs total you can run $32 / 16 = 2$ data-parallel replicas — so the full layout is 3D parallelism: tensor(4) × pipeline(4) × data(2) $= 32$. Each step, the two replicas process different data shards and all-reduce their gradients (09.1), while within each replica tensor parallelism all-reduces per layer and pipeline parallelism passes activations between stages.

**(b)** $\dfrac{S-1}{m+S-1} = \dfrac{3}{m+3} < 0.10 \Rightarrow m + 3 > 30 \Rightarrow m > 27$, so **$m \geq 28$ micro-batches**. (At $m = 28$ the bubble is $3/31 \approx 9.7\%$; at $m = 32$, $\approx 8.6\%$.)

</details>

## Common mistakes

- **Confusing tensor and pipeline parallelism.** Tensor parallelism splits *within* a layer (across a matmul's width) and all-reduces on every layer; pipeline parallelism splits *across* layers (depth) and passes activations between stages. They solve different fits and are usually combined.
- **Running tensor parallelism across slow links.** Its per-layer all-reduce is heavy; put it on the fastest interconnect (within a node) or throughput collapses.
- **Too few pipeline micro-batches.** With $m$ near $S$, the bubble wastes a large fraction of the hardware; use $m \gg S$.
- **Forgetting these compose with data parallelism/ZeRO.** Frontier runs are tensor × pipeline × data (× expert for MoE) simultaneously; each axis is chosen for what it splits and mapped to the hardware topology by its communication cost.

## Check yourself

<details><summary>In the Megatron MLP split, why is the first matmul column-parallel and the second row-parallel, rather than the other way round?</summary>

Column-splitting the first matmul gives each device its own slice of the hidden activations, and because the nonlinearity (ReLU/GELU) is <em>element-wise</em>, each device can apply it to its slice with no communication. Row-splitting the second matmul then makes each device produce a partial full-width output, and a single all-reduce sums them. This ordering means the wide $4C$ hidden activations are never communicated — only the final $C$-wide output, once.

</details>

<details><summary>A pipeline has 8 stages. What bubble fraction do you get with 8 micro-batches, and with 64?</summary>

$(S-1)/(m+S-1)$ with $S=8$: at $m=8$, $7/15 \approx 0.467$ (47% wasted); at $m=64$, $7/71 \approx 0.099$ (≈10%). More micro-batches amortize the fixed 7-slot fill/drain.

</details>

<details><summary>In the worked tensor-parallel example, device 0 produced partial output $[5, 2]$ and device 1 produced $[5, 9]$. What operation yields the correct final output, and what is it?</summary>

An all-reduce with sum: $[5,2] + [5,9] = [10, 11]$, which matches the single-device result $\operatorname{ReLU}(XA)B$. Each device's row-parallel second matmul yields a partial (full-shape but incomplete) output; summing the partials reconstructs the whole.

</details>

<details><summary>Why is expert parallelism able to give a model a very large total parameter count without a proportional increase in per-token compute?</summary>

In an MoE, the router sends each token to only one or a few of the many experts, so a token's forward pass touches only those experts' parameters — not all of them. Placing different experts on different devices (with an all-to-all to route tokens) means total parameters scale with the number of experts while per-token compute (and per-device work) scales only with the few experts actually used. (Built in <a href="../module-12/lesson-04.md">Module 12.4</a>.)

</details>

<div class="hw">
<p><strong>Hardware track — model parallelism.</strong></p>
<p><strong>This lesson's code / arithmetic:</strong> pure single-CPU simulation — any laptop, instant, 0 GPU-hours, CPU-only entirely. <strong>Real tensor parallelism</strong> needs multiple GPUs on the <em>fastest</em> link (NVLink within a node), because it all-reduces on every layer's forward and backward; degree 2–8 within a node is typical. <strong>Pipeline parallelism</strong> tolerates slower links (only activations cross stage boundaries) and so spans nodes, but needs $m \gg S$ micro-batches to keep the bubble small. <strong>Frontier scale:</strong> 3D parallelism (tensor × pipeline × data, + expert for MoE) over hundreds to thousands of GPUs for weeks; the axis-to-topology mapping (heavy communication on fast links) is the heart of the systems design. <strong>GPU-hours:</strong> model parallelism is what makes a too-big model <em>trainable at all</em>; it does not reduce total compute ($6ND$ from 07.4 still holds) — it distributes it.</p>
</div>

## Next

You have now seen the full parallelism toolkit: data parallelism and ZeRO/FSDP (split the data and shard the memory, 09.1–09.2) and tensor / pipeline / expert parallelism (split the model itself, this lesson). Together these are how any model larger than one GPU is trained. The next module steps back from *how to train fast* to *how big to make the model and how much data to use in the first place* — the scaling laws that turn a compute budget into an optimal model size and token count.

Continue to [Module 10 · Scaling laws](../module-10/lesson-01.md).
