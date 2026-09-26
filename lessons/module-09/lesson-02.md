# 09.2 · ZeRO & FSDP

<div class="prereq">
<p><strong>Prerequisites:</strong> data parallelism, all-reduce, all-gather and reduce-scatter from <a href="#/lessons/module-09/lesson-01">09.1 · Data parallelism, all-reduce, DDP</a>; the four memory buckets — parameters, gradients, Adam optimizer state ($2\times$ params), activations — and the $16P$-bytes fixed cost from <a href="#/lessons/module-07/lesson-04">07.4 · Throughput, FLOPs, MFU, memory accounting</a>.</p>
<p><strong>You will learn:</strong> exactly what a training step holds in memory and why plain data parallelism <strong>replicates all of it</strong> on every worker; how <strong>ZeRO</strong> removes that redundancy by <em>sharding</em> the optimizer state (stage 1), then the gradients (stage 2), then the parameters (stage 3) across the <code>W</code> data-parallel workers — with the per-worker memory formula for each stage, worked in bytes; and how PyTorch's <strong>FSDP</strong> realizes stage-3 by all-gathering each layer's parameters just in time for its forward/backward and freeing them immediately after, trading extra communication for a large memory saving.</p>
<p><strong>Why this matters for ML:</strong> the $16P$-byte fixed cost is the wall that stops a billion-parameter model from training on a single GPU no matter how small you make the batch. ZeRO/FSDP is how the field trains models far larger than one device's memory while still using ordinary data parallelism — it is the default for large open-model training today.</p>
</div>

<div class="callout warn"><p><strong>Single-CPU note.</strong> As in 09.1, we cannot run a real sharded multi-GPU job here. This lesson is mostly <em>memory arithmetic</em> — which is exact and CPU-verifiable with <code>py</code> — plus a description of the FSDP all-gather/free dance built on the <code>all_gather</code> / <code>reduce_scatter</code> collectives from 09.1. Real-hardware behaviour is in the <code>&lt;div class="hw"&gt;</code> box.</p></div>

## 1. Intuition: data parallelism wastes memory `W` times over

Recall from [07.4](lessons/module-07/lesson-04.md) what one training step must hold, per parameter, in fp32:

- the **parameter** itself — 4 bytes,
- its **gradient** — 4 bytes,
- Adam's **optimizer state**, the two moment buffers $m$ and $v$ — 8 bytes,

for a fixed $16$ bytes per parameter, or $16P$ bytes for a $P$-parameter model, before any activations. For $P = 124.4$M (GPT-2 small) that is $\approx 1.99$ GB; for $P = 7.5$B it is $120$ GB — already past any single GPU.

Now here is the waste. In plain data parallelism ([09.1](lessons/module-09/lesson-01.md)), **every one of the `W` workers holds a complete copy of all $16P$ bytes.** The workers differ only in which *data* they process; their parameters, gradients (after the all-reduce), and optimizer state are identical. So across the cluster you are storing the same $16P$ bytes `W` times. If you have 64 GPUs, 63 of those copies are pure redundancy.

**ZeRO** (Zero Redundancy Optimizer) is the observation that you don't need to replicate what you can *partition*. Split the $16P$ bytes into `W` disjoint pieces, give each worker one piece, and reconstruct the full tensor only for the brief moment a worker actually needs it. The catch is that reconstruction costs communication — so ZeRO is a spectrum, trading progressively more communication for progressively less memory.

<div class="callout key"><p>Plain data parallelism replicates the full $16P$ bytes on all <code>W</code> workers. ZeRO <strong>shards</strong> those bytes across the workers so each holds only $\approx 1/W$ of them, all-gathering the pieces on demand. Same math, same result — less redundant storage, more communication.</p></div>

## 2. The memory a step holds, in bytes

Let $P$ be the parameter count and use the fp32 accounting of [07.4](lessons/module-07/lesson-04.md). The three model-sized buckets are:

$$
\underbrace{4P}_{\text{params}} \;+\; \underbrace{4P}_{\text{grads}} \;+\; \underbrace{8P}_{\text{Adam } m,\,v} \;=\; 16P \text{ bytes}.
$$

(Activations are the fourth bucket; they scale with batch × context, not with $P$, and are attacked separately by activation checkpointing in [Module 8](lessons/module-08/lesson-01.md). ZeRO is about the three model-sized buckets.)

Worked, for $P = 7.5\times 10^9$:

$$
16P = 16 \cdot 7.5\times 10^9 = 1.2\times 10^{11}\ \text{bytes} = 120\ \text{GB per worker}.
$$

That is the plain-DP cost on *every* worker, whether `W` is 1 or 1000. An 80 GB A100 cannot hold it. ZeRO's job is to make that per-worker number fall as `W` grows.

<div class="callout pt"><p>Real large-model training uses <strong>mixed precision</strong> (Module 8), which changes the constants: fp16 params (2 bytes) + fp16 grads (2 bytes) + fp32 optimizer state comprising a master weight copy, $m$, and $v$ (12 bytes) also totals $16P$ — the number the ZeRO paper uses. We keep the pure-fp32 breakdown from 07.4 for continuity; the <em>sharding structure</em> (each stage divides one more bucket by <code>W</code>) is identical either way.</p></div>

## 3. The three ZeRO stages

ZeRO shards the three buckets one at a time. Each stage keeps the previous stage's sharding and adds one more. Below, `W` is the number of data-parallel workers and $P$ the parameter count; the formulas give **per-worker** bytes.

### Stage 1 — shard the optimizer state ($P_{os}$)

The optimizer state ($m$ and $v$, $8P$ bytes) is only touched *inside* `optimizer.step()`. Between steps it just sits there. So ZeRO-1 gives each worker only $1/W$ of it: worker `w` owns the $m,v$ for parameters $w\cdot P/W \ldots (w{+}1)\cdot P/W$. During the step, worker `w` updates *only its slice* of the parameters (the ones it holds the optimizer state for), then an all-gather shares the updated parameters so everyone again has the full weights for the next forward pass.

Per worker: full params + full grads + sharded optimizer state.

$$
4P + 4P + \frac{8P}{W} = 8P + \frac{8P}{W}\ \text{bytes}.
$$

### Stage 2 — also shard the gradients ($P_{os+g}$)

Once worker `w` only *updates* its slice of parameters, it only *needs* the gradients for that slice. Every other gradient can be discarded after it has been reduced. So instead of an all-reduce (which leaves the full averaged gradient on everyone), ZeRO-2 uses a **reduce-scatter** ([09.1](lessons/module-09/lesson-01.md)): it averages the gradients and leaves each worker with only *its* $1/W$ slice. The other $(W-1)/W$ of the gradient never has to be stored.

Per worker: full params + sharded grads + sharded optimizer state.

$$
4P + \frac{4P}{W} + \frac{8P}{W} = 4P + \frac{12P}{W}\ \text{bytes}.
$$

### Stage 3 — also shard the parameters ($P_{os+g+p}$)

The last replicated bucket is the parameters themselves ($4P$). ZeRO-3 shards those too: worker `w` permanently stores only its $1/W$ slice of the weights. But a forward pass needs a layer's *full* weights to compute it — so just before a layer runs, ZeRO-3 **all-gathers** that layer's parameters from all workers, computes, then **frees** the gathered copy immediately, keeping only its own slice. The same all-gather-then-free happens again in backward.

Per worker: sharded params + sharded grads + sharded optimizer state.

$$
\frac{4P}{W} + \frac{4P}{W} + \frac{8P}{W} = \frac{16P}{W}\ \text{bytes}.
$$

<div class="callout key"><p>Per-worker memory (fp32, model-sized buckets only):<br>
<strong>Plain DP:</strong> $16P$<br>
<strong>ZeRO-1</strong> (optimizer state): $8P + \dfrac{8P}{W}$<br>
<strong>ZeRO-2</strong> (+ gradients): $4P + \dfrac{12P}{W}$<br>
<strong>ZeRO-3</strong> (+ parameters): $\dfrac{16P}{W}$<br>
As <code>W → ∞</code>, ZeRO-3 drives per-worker model memory to zero; stages 1 and 2 have a floor ($8P$ and $4P$) because they keep at least one full-model bucket resident.</p></div>

## 4. Worked example: 7.5B parameters on 64 workers

Take $P = 7.5\times 10^9$ and $W = 64$, fp32. Plug into the four formulas:

```python
P, W = 7.5e9, 64
base = 16 * P
z1   = 8 * P + 8 * P / W
z2   = 4 * P + 12 * P / W
z3   = 16 * P / W
for name, b in [("plain DP", base), ("ZeRO-1", z1), ("ZeRO-2", z2), ("ZeRO-3", z3)]:
    print(f"{name:9s} {b/1e9:7.2f} GB   ({b/P:.4f} bytes/param)")
```

which prints:

```text
plain DP   120.00 GB   (16.0000 bytes/param)
ZeRO-1      60.94 GB   (8.1250 bytes/param)
ZeRO-2      31.41 GB   (4.1875 bytes/param)
ZeRO-3       1.88 GB   (0.2500 bytes/param)
```

The numbers tell the whole story. Plain DP needs 120 GB per worker — impossible on an 80 GB GPU. ZeRO-1 nearly halves it (the $8P$ optimizer state is the biggest single bucket) to 60.9 GB — now it fits. ZeRO-2 roughly halves again to 31.4 GB, leaving lots of room for activations and a bigger batch. ZeRO-3 collapses it to 1.88 GB per worker ($16P/64$), because *nothing* model-sized is fully resident — at the cost of all-gathering every layer's weights on every forward and backward.

Notice the diminishing floors: ZeRO-1 and ZeRO-2 can never go below $8P$ and $4P$ respectively (each keeps a full-model bucket), so on huge `W` they plateau. Only ZeRO-3 keeps falling as $16P/W$. That is the trade you are buying: ZeRO-3's memory is best, its communication cost is highest.

## 5. FSDP: PyTorch's stage-3, and the all-gather/free dance

**FSDP** (Fully Sharded Data Parallel) is PyTorch's native implementation of ZeRO-3. "Fully sharded" means all three buckets are sharded, so each worker permanently stores only $1/W$ of the model. The mechanism is a just-in-time reconstruction, unit by unit (FSDP groups layers into "wrapping units"):

1. **All-gather** the unit's parameters. Before a unit runs its forward, FSDP calls an all-gather ([09.1](lessons/module-09/lesson-01.md)) so every worker temporarily holds that unit's *full* weights.
2. **Compute** the unit's forward. The full weights exist only for this moment.
3. **Free** the gathered weights. Immediately after the unit's forward, the non-owned $(W-1)/W$ of the weights is discarded, dropping back to the $1/W$ shard.
4. **Repeat in backward.** The same all-gather → compute → free happens again for each unit during the backward pass, and gradients are combined with a **reduce-scatter** so each worker keeps only its gradient shard (stage 2's trick).

At any instant, the *full* weights of at most one (or a few, if prefetching) units are materialized — not the whole model. That is why per-worker memory is $\approx 16P/W$ plus the transient cost of the largest unit's full parameters.

The `all_gather` you built in [09.1](lessons/module-09/lesson-01.md) is exactly the primitive step 1 uses. Conceptually, reconstructing a sharded parameter is:

```python
from llmre.distributed import all_gather
# each worker holds a shard of a layer's weight; all_gather rebuilds the full tensor
full_weight = all_gather([shard_from_worker_w for w in range(W)])   # (full_rows, ...)
# ... run the layer's forward/backward with full_weight ...
# then drop full_weight, keeping only this worker's shard
```

On real hardware every worker runs this simultaneously and ends up with an identical `full_weight`; here we simulate it as a single concatenation. `code/tests/test_distributed.py` checks that `all_gather` concatenates shards to the right shape and content.

<div class="callout pt"><p>FSDP's cost is the extra communication: it all-gathers every unit's parameters <strong>twice per step</strong> (once in forward, once in backward), where plain DDP communicates gradients only <strong>once</strong>. To hide it, FSDP <em>prefetches</em> — it starts all-gathering unit <em>k+1</em>'s parameters while unit <em>k</em> is still computing, overlapping communication with compute just as DDP overlaps the gradient all-reduce with backward (09.1). Well-tuned FSDP therefore keeps throughput close to DDP while using a fraction of the memory.</p></div>

<div class="callout warn"><p>ZeRO-3 / FSDP are not free lunches. The all-gather-per-layer traffic means they are most efficient when the interconnect is fast (NVLink within a node, high-bandwidth fabric across nodes); on a slow interconnect the communication stops hiding under compute and throughput drops. A common practical recipe is to use FSDP (or ZeRO-3) <em>within</em> a fast-linked group and plain data parallelism across groups — a hybrid you'll see again in <a href="#/lessons/module-09/lesson-03">09.3</a>.</p></div>

## 6. Which stage should you reach for?

A practical ladder, in order of increasing memory savings and communication cost:

- **Plain DDP** — the model + optimizer state fit comfortably on one GPU and you only want speed. No sharding. ([09.1](lessons/module-09/lesson-01.md).)
- **ZeRO-1** — you're just over the memory limit; sharding the $8P$ optimizer state (the biggest bucket) is often enough, and it adds almost no communication (only the parameter all-gather after the step).
- **ZeRO-2** — you need more room; reduce-scatter for gradients is a cheap extra step.
- **ZeRO-3 / FSDP** — the model itself doesn't fit even with grads and optimizer state sharded, or you want to maximize batch size / model size per GPU. Highest communication, lowest memory.

<div class="callout key"><p>Rule of thumb: climb the ZeRO ladder only as far as the memory wall forces you, because each rung adds communication. Stage 1 buys the most memory per unit of extra communication (it shards the largest, least-frequently-touched bucket); stage 3 buys the most memory overall but pays with per-layer all-gathers.</p></div>

## Research connection

<div class="callout paper"><p><strong>ZeRO: Memory Optimizations Toward Training Trillion Parameter Models</strong> (Rajbhandari et al., 2019) introduces the three stages and the "zero redundancy" framing you just learned — the per-stage memory formulas in section 3 are its central result. It is paired with <strong>Megatron-LM</strong> (next lesson) as the two foundational systems papers of large-model training. Read it after this module; the reading guide and links are in <a href="#/papers/index">the paper curriculum</a>. Its ideas ship today as Microsoft <strong>DeepSpeed</strong> (ZeRO) and PyTorch <strong>FSDP</strong>.</p></div>

## Exercise

You are training a $P = 3\times 10^9$ parameter model on `W = 8` GPUs, each with 40 GB. Using the fp32 formulas (ignore activations): **(a)** does plain data parallelism fit? **(b)** does ZeRO-1? **(c)** does ZeRO-3? Give the per-worker GB for each.

<details><summary>Hint</summary>
Plain DP is $16P$; ZeRO-1 is $8P + 8P/W$; ZeRO-3 is $16P/W$. Convert bytes to GB by dividing by $10^9$.
</details>

<details><summary>Stronger hint</summary>
$P = 3\text{e}9$, $W = 8$. Plain DP $= 16 \cdot 3\text{e}9$. ZeRO-1 $= 8 \cdot 3\text{e}9 + 8 \cdot 3\text{e}9 / 8$. ZeRO-3 $= 16 \cdot 3\text{e}9 / 8$.
</details>

<details><summary>Solution</summary>

**(a)** Plain DP $= 16P = 16 \cdot 3\times 10^9 = 48\times 10^9$ bytes $= 48$ GB per worker. **Does not fit** in 40 GB.

**(b)** ZeRO-1 $= 8P + 8P/W = 24\times 10^9 + 24\times 10^9/8 = 24 + 3 = 27$ GB per worker. **Fits** in 40 GB, with ~13 GB left for activations.

**(c)** ZeRO-3 $= 16P/W = 48\times 10^9/8 = 6$ GB per worker. **Fits** easily, leaving ~34 GB for activations and a much larger batch — at the cost of per-layer all-gathers.

So the cheapest option that fits is ZeRO-1; you'd only climb to ZeRO-3 if you also needed the extra activation headroom or a larger batch.

</details>

## Common mistakes

- **Thinking data parallelism saves memory.** It only saves *time*. Every worker holds the full $16P$ bytes; DP does nothing for the memory wall. That is the entire reason ZeRO exists.
- **Forgetting activations.** ZeRO shards the three *model-sized* buckets. Activation memory (batch × context) is untouched and is often the real limit — pair ZeRO with activation checkpointing / mixed precision (Module 8).
- **Assuming ZeRO-3 is always best.** Its per-layer all-gathers are the heaviest communication of any stage; on a slow interconnect it can be slower overall than a stage that fits with less traffic. Climb only as far as memory forces you.
- **Confusing all-reduce with reduce-scatter.** Plain DDP all-reduces (full averaged gradient on everyone); ZeRO-2/3 reduce-scatters (each worker keeps only its gradient shard). The second is what makes gradient sharding possible.

## Check yourself

<details><summary>Why does plain data parallelism, even on 100 GPUs, not help a model that is too big to fit on one GPU?</summary>

Because data parallelism <em>replicates</em> the full model, gradients, and optimizer state on every worker — all $16P$ bytes, on each GPU. Adding workers adds throughput, not per-worker capacity. To fit a bigger model you must <em>shard</em> those bytes across workers, which is what ZeRO/FSDP do.

</details>

<details><summary>Order the three ZeRO stages by per-worker memory, and say which bucket each additionally shards.</summary>

ZeRO-1 ($8P + 8P/W$) shards the optimizer state; ZeRO-2 ($4P + 12P/W$) additionally shards the gradients; ZeRO-3 ($16P/W$) additionally shards the parameters. Memory: ZeRO-3 &lt; ZeRO-2 &lt; ZeRO-1 &lt; plain DP ($16P$).

</details>

<details><summary>In FSDP, at what moment does a worker hold a layer's <em>full</em> parameters, and what happens to them right after?</summary>

Just before that layer's forward (and again before its backward), FSDP <em>all-gathers</em> the shards so every worker briefly holds the full weights; it computes, then immediately <em>frees</em> the gathered copy, dropping back to its $1/W$ shard. Only one (or a few prefetched) units are ever fully materialized at once.

</details>

<details><summary>For $P = 7.5$B and $W = 64$, why does ZeRO-1 give 60.9 GB but ZeRO-3 give 1.88 GB — where did the extra savings come from?</summary>

ZeRO-1 still keeps the <em>full</em> params ($4P$) and <em>full</em> grads ($4P$) on every worker — $8P = 60$ GB — and only shards the $8P$ optimizer state down to $8P/64 \approx 0.9$ GB. ZeRO-3 additionally shards params and grads, so all three buckets become $1/64$ of their size: $16P/64 = 1.88$ GB. The extra savings are the params and grads no longer being replicated.

</details>

<div class="hw">
<p><strong>Hardware track — sharded training.</strong></p>
<p><strong>This lesson's code / arithmetic:</strong> pure single-CPU — any laptop, instant, 0 GPU-hours, CPU-only entirely. <strong>Real ZeRO/FSDP</strong> needs <code>W</code> GPUs with a fast interconnect (NVLink/InfiniBand) so the extra all-gather traffic hides under compute. Memory per worker follows the section-3 formulas; e.g. a 7.5B model needs ~120 GB/GPU on plain DP (won't fit an 80 GB A100) but ~1.9 GB/GPU of model state on ZeRO-3 over 64 GPUs, leaving the rest for activations. <strong>Throughput:</strong> ZeRO-1 ≈ DDP; ZeRO-3/FSDP is a few percent to ~20% slower per step when communication overlaps well, and much slower on a weak interconnect. <strong>Frameworks:</strong> DeepSpeed (ZeRO stages 1–3) and PyTorch FSDP; both are configuration on top of the DDP-style loop from 09.1.</p>
</div>

## Next

ZeRO/FSDP let a model *fit* by sharding its memory across data-parallel workers, but they still run every layer's full computation on each worker (after gathering its weights). When even a single layer's compute or activations are too big — or when you want to split the *math* itself across devices — you need model parallelism: splitting individual matmuls (tensor parallelism), assigning layer ranges to different devices (pipeline parallelism), and routing experts (expert parallelism).

Continue to [09.3 · Tensor / pipeline / expert parallelism](lessons/module-09/lesson-03.md).
