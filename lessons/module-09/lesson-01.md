# 09.1 · Data parallelism, all-reduce, DDP

<div class="prereq">
<p><strong>Prerequisites:</strong> the canonical training loop <code>get_batch → model(x,y) → zero_grad → backward → clip → step</code> from <a href="../module-07/lesson-01.md">07.1 · The training loop</a>; micro-batch / global batch and gradient accumulation from <a href="../module-07/lesson-02.md">07.2 · Micro-batch, global batch, gradient accumulation</a>; that a language-model loss is a <em>mean</em> cross-entropy over tokens (<a href="../module-07/lesson-01.md">07.1</a>).</p>
<p><strong>You will learn:</strong> how <strong>data parallelism</strong> puts a full copy of the model on each of <code>W</code> workers, splits every global batch into <code>W</code> shards, and averages the per-shard gradients with an <strong>all-reduce</strong> so every worker applies the identical update; the exact reason averaging per-shard gradients equals the full-batch gradient (a two-line proof, verified numerically); how <strong>ring all-reduce</strong> moves the data in <code>2(W-1)</code> steps carrying only <code>1/W</code> of the tensor per link; what NCCL is; and how PyTorch <strong>DDP</strong> overlaps the all-reduce with the backward pass.</p>
<p><strong>Why this matters for ML:</strong> data parallelism is the first and most common way runs get faster — it is how you turn one GPU into eight, or eight into thousands, with almost no change to the training loop you already wrote. Every larger parallelism strategy in this module (ZeRO, FSDP, tensor/pipeline parallelism) is built on top of the collective you meet here.</p>
</div>

<div class="callout warn"><p><strong>Single-CPU note.</strong> This machine has one CPU and no CUDA/NCCL, so we cannot run a real multi-GPU job. Every code sample here <strong>simulates</strong> data parallelism in one process — a Python <code>for</code> loop over the "workers", with all their tensors living in one list. That reproduces the exact <em>numbers</em> a real run would produce; what it cannot show is the wall-clock speedup. Real-hardware facts live in the <code>&lt;div class="hw"&gt;</code> boxes.</p></div>

## 1. Intuition: clone the model, split the batch, agree on the update

You already have a training loop that, once per step, computes the mean loss over a batch, calls `backward()` to fill every parameter's `.grad`, and lets the optimizer nudge the weights. One GPU can only push so many tokens through that loop per second. The simplest way to go faster is embarrassingly literal: **run the same loop on `W` GPUs at once, each on a different slice of the data.**

Concretely, data parallelism does four things every step:

1. **Replicate.** Every worker holds a complete copy of the model — identical weights.
2. **Shard the batch.** The global batch of `B` examples is cut into `W` shards of `B/W` examples; worker `w` gets shard `w`.
3. **Compute local gradients.** Each worker runs forward + backward on *its* shard, producing a gradient from `B/W` examples.
4. **All-reduce.** The workers average their gradients so every worker ends up with the *same* averaged gradient — as if all `B` examples had been in one batch. Then each worker runs the identical optimizer step, and because they started identical and applied the identical update, they stay identical.

The one new operation is step 4, the **all-reduce**: "combine one tensor from every worker (here, by averaging) and give the combined result back to all of them." Everything else is the loop you already own, run `W` times.

<div class="callout key"><p>Data parallelism = <strong>same model on every worker, different data on every worker, one all-reduce per step to average the gradients.</strong> The workers stay bit-for-bit identical because they start identical and every step applies the same averaged update.</p></div>

## 2. The correctness fact: averaging shard gradients == the full-batch gradient

This is the mathematical heart of the whole scheme, and it is worth proving, because it is *why* data parallelism is exact rather than an approximation.

The training loss over a batch is a **mean** over the examples (07.1). Write the per-example loss as $\ell_i(\theta)$ for example $i$ and parameters $\theta$. The full-batch loss over all $B$ examples is

$$
\mathcal{L}_{\text{full}}(\theta) = \frac{1}{B}\sum_{i=1}^{B} \ell_i(\theta).
$$

Now split the $B$ examples into $W$ equal shards, each of size $B/W$. Let $\mathcal{S}_w$ be the set of indices on worker $w$. Worker $w$ computes the **mean loss over its own shard** and differentiates it, giving the local gradient

$$
g_w = \nabla_\theta\!\left[\frac{W}{B}\sum_{i \in \mathcal{S}_w} \ell_i(\theta)\right] = \frac{W}{B}\sum_{i \in \mathcal{S}_w} \nabla_\theta \ell_i(\theta).
$$

(The shard has $B/W$ examples, so its mean divides by $B/W$, i.e. multiplies by $W/B$.) Average the local gradients across the $W$ workers:

$$
\frac{1}{W}\sum_{w=1}^{W} g_w
= \frac{1}{W}\sum_{w=1}^{W}\frac{W}{B}\sum_{i \in \mathcal{S}_w}\nabla_\theta \ell_i
= \frac{1}{B}\sum_{w=1}^{W}\sum_{i \in \mathcal{S}_w}\nabla_\theta \ell_i
= \frac{1}{B}\sum_{i=1}^{B}\nabla_\theta \ell_i
= \nabla_\theta \mathcal{L}_{\text{full}}.
$$

The shards partition the batch, so the double sum is just the sum over all $B$ examples. **The average of the per-shard mean-loss gradients is exactly the full-batch mean-loss gradient.** No approximation.

<div class="callout key"><p>For a <strong>mean</strong> loss and <strong>equal</strong> shards: <code>mean over workers of (per-shard gradient)</code> $=$ <code>full-batch gradient</code>. This is why the all-reduce uses an <em>average</em>, not a sum, and why unequal shards would need a size-weighted average instead.</p></div>

### Verify it numerically

Take a linear-regression toy so we can compute the gradient in closed form. The loss is the mean squared error $\frac1n\sum_i (x_i\!\cdot\!\theta - y_i)^2$, whose gradient is $\frac{2}{n}X^\top(X\theta - y)$. We split a batch of `B = 32` into `W = 4` shards of 8, compute each shard's mean-loss gradient, average them, and compare to the single full-batch gradient:

```python
import numpy as np
rng = np.random.default_rng(0)
d, W = 3, 4
B = 8 * W
X = rng.standard_normal((B, d)); y = rng.standard_normal((B,))
theta = rng.standard_normal((d,))

def mean_grad(p, batch):                 # gradient of the MEAN squared error
    Xs, ys = batch; n = Xs.shape[0]
    return (2.0 / n) * (Xs.T @ (Xs @ p - ys))

full = mean_grad(theta, (X, y))
shards = [(X[i*8:(i+1)*8], y[i*8:(i+1)*8]) for i in range(W)]
avg = np.mean([mean_grad(theta, s) for s in shards], axis=0)

print(full)                    # [-1.6957546  -1.42161642 -1.99938365]
print(avg)                     # [-1.6957546  -1.42161642 -1.99938365]
print(np.max(np.abs(full-avg)))# 2.22e-16  (floating-point noise, i.e. exactly equal)
```

The averaged shard gradient matches the full-batch gradient to machine precision ($2.2\times 10^{-16}$). This exact identity is checked in `code/tests/test_distributed.py::test_data_parallel_grad_equals_full_batch` for several worker counts.

<div class="callout warn"><p>The proof needs a <strong>mean</strong> loss and <strong>equal</strong> shards. If your loss is a <em>sum</em>, or a per-token mean where shards have different token counts (e.g. variable-length sequences with padding), a plain gradient average is subtly wrong — you must weight each worker's gradient by its share of the tokens. Getting this wrong makes the effective learning rate depend on <code>W</code>, a classic silent scaling bug.</p></div>

## 3. All-reduce: the collective, and why "ring"

An **all-reduce** takes one equally-shaped tensor from each of `W` workers, combines them element-wise with an associative op (here, sum — then we divide by `W` for the mean), and leaves the *same* combined tensor on every worker. It is one member of a family of **collective communication** operations (all-gather, reduce-scatter, broadcast) that coordinate many devices at once.

The naive way to all-reduce is: every worker sends its tensor to worker 0, worker 0 sums them and sends the result back to everyone. That works, but worker 0's network link carries $W-1$ incoming tensors and then $W-1$ outgoing — it is a bottleneck that gets worse as `W` grows. For a gradient tensor that can be hundreds of megabytes, on a run with hundreds of workers, this is fatal.

### Ring all-reduce

**Ring all-reduce** arranges the workers in a logical ring (worker `w` sends only to worker `w+1`, wrapping around) and never routes everything through one node. Each worker splits its tensor into `W` chunks and the algorithm runs in two phases:

- **Reduce-scatter** (`W-1` steps): on each step, every worker sends one chunk to its right neighbour and adds the chunk it receives into its own buffer. After `W-1` steps, each worker holds the fully-summed value of *one distinct chunk* — the sum is "scattered" across the workers.
- **All-gather** (`W-1` steps): the finished chunks are passed around the ring until every worker has all `W` of them.

That is $2(W-1)$ steps total, and on each step every link carries exactly **one chunk = $1/W$ of the tensor**. So each worker sends and receives about $2(W-1)/W \approx 2$ tensors' worth of data over the whole collective, *independent of how large `W` is*. No single link is a bottleneck. This is why ring all-reduce is called **bandwidth-optimal**.

<div class="callout key"><p>Ring all-reduce: <strong>reduce-scatter then all-gather</strong>, $2(W-1)$ steps, each link carrying $1/W$ of the tensor per step. Total data per worker $\approx 2\times$ the tensor size regardless of $W$ — the naive "everyone → worker 0 → everyone" instead piles $2(W-1)$ tensors onto one link.</p></div>

### Worked example: three workers, by hand

Let `W = 3`, and give each worker a length-3 vector, so there are `W = 3` chunks of one element each (chunk `c` = index `c`):

$$
w_0 = [1,\ 2,\ 3], \qquad w_1 = [4,\ 5,\ 6], \qquad w_2 = [7,\ 8,\ 9].
$$

The true sum is $[12,\ 15,\ 18]$. Watch the ring produce it.

**Reduce-scatter, step 0.** Worker `i` sends chunk `i` to worker `i+1`, which adds it into its own chunk `i`:

- $w_0$ sends chunk 0 ($=1$) to $w_1$: $w_1$'s chunk 0 becomes $4+1=5$.
- $w_1$ sends chunk 1 ($=5$) to $w_2$: $w_2$'s chunk 1 becomes $8+5=13$.
- $w_2$ sends chunk 2 ($=9$) to $w_0$: $w_0$'s chunk 2 becomes $3+9=12$.

State: $w_0=[1,2,\mathbf{12}]$, $w_1=[\mathbf{5},5,6]$, $w_2=[7,\mathbf{13},9]$.

**Reduce-scatter, step 1.** Worker `i` now sends the chunk it just updated ($(i-1)\bmod 3$) onward:

- $w_0$ sends chunk 2 ($=12$) to $w_1$: $w_1$'s chunk 2 becomes $6+12=\mathbf{18}$.
- $w_1$ sends chunk 0 ($=5$) to $w_2$: $w_2$'s chunk 0 becomes $7+5=\mathbf{12}$.
- $w_2$ sends chunk 1 ($=13$) to $w_0$: $w_0$'s chunk 1 becomes $2+13=\mathbf{15}$.

State: $w_0=[1,\mathbf{15},12]$, $w_1=[5,5,\mathbf{18}]$, $w_2=[\mathbf{12},13,9]$. Each worker now owns exactly one *finished* chunk: $w_0$ has the final chunk 1 ($=15$), $w_1$ the final chunk 2 ($=18$), $w_2$ the final chunk 0 ($=12$). Those are precisely the three entries of the true sum $[12,15,18]$ — scattered one per worker.

**All-gather** (2 steps) then circulates those three finished chunks around the ring until all three workers hold $[12,\ 15,\ 18]$. Divide by `W = 3` for the mean: $[4,\ 5,\ 6]$.

Our simulation reproduces this exactly:

```python
import numpy as np
from llmre.distributed import ring_all_reduce
w0, w1, w2 = np.array([1.,2.,3.]), np.array([4.,5.,6.]), np.array([7.,8.,9.])
print(ring_all_reduce([w0, w1, w2], op="sum"))   # [12. 15. 18.]
print(ring_all_reduce([w0, w1, w2], op="mean"))  # [ 4.  5.  6.]
```

## 4. The simulated implementation

Here is the core of `llmre/distributed/collectives.py` — the reduce-scatter / all-gather walk you just traced by hand. It runs on one CPU, looping over the workers; on real hardware NCCL performs these same sends across the interconnect.

```python
def ring_all_reduce(shards, op="mean"):
    W = len(shards)
    buf = [s.reshape(-1).copy() for s in shards]     # each worker's flat buffer
    bounds = _chunk_bounds(len(buf[0]), W)           # W near-equal chunks

    # Phase 1: reduce-scatter (W-1 steps)
    for t in range(W - 1):
        sent = [buf[i][slice(*bounds[(i - t) % W])].copy() for i in range(W)]
        for i in range(W):
            src, c = (i - 1) % W, (i - 1 - t) % W    # chunk src just sent
            lo, hi = bounds[c]
            buf[i][lo:hi] = buf[i][lo:hi] + sent[src]

    # Phase 2: all-gather (W-1 steps) — overwrite, don't add
    for t in range(W - 1):
        sent = [buf[i][slice(*bounds[(i + 1 - t) % W])].copy() for i in range(W)]
        for i in range(W):
            src, c = (i - 1) % W, (i - t) % W
            lo, hi = bounds[c]
            buf[i][lo:hi] = sent[src]

    result = buf[0]
    return (result / W if op == "mean" else result).reshape(shards[0].shape)
```

Two implementation details worth naming. We snapshot every chunk *before* the round of adds (`sent = [...]`) so a worker never reads a neighbour's already-updated value mid-step — that mirrors real hardware, where all workers send simultaneously from their step-start state. And the reduce-scatter phase **adds** (`+`) while the all-gather phase **overwrites** (`=`): the first is combining partial sums, the second is just copying finished results around.

<div class="callout warn"><p>The reduce-scatter and all-gather index arithmetic is fiddly and easy to get subtly wrong (off-by-one on the chunk each worker sends). That is exactly why the module ships a test that compares the ring's output against <code>np.sum</code> / <code>np.mean</code> for worker counts 1–8 and for tensor lengths that don't divide evenly. If you ever re-derive the schedule, let that test catch you.</p></div>

## 5. Tying it back to the training loop: `data_parallel_grads`

With the collective in hand, one data-parallel step is: split the batch, get each worker's mean-loss gradient, all-reduce-average them. That is `llmre/distributed/data_parallel.py`:

```python
def data_parallel_grads(grad_fn, params, batch, world_size):
    shards = split_batch(batch, world_size)               # W equal shards
    grads = [grad_fn(params, shard) for shard in shards]  # per-worker local grad
    return ring_all_reduce(grads, op="mean")              # averaged == full-batch
```

In a real run there is no `for` loop here — the `W` `grad_fn` calls happen *simultaneously* on `W` GPUs, and `ring_all_reduce` is a NCCL call across the interconnect. The returned gradient is what the optimizer consumes, exactly as in the single-GPU loop of [07.1](../module-07/lesson-01.md); every worker gets the same averaged gradient and takes the same step.

Notice the relationship to **gradient accumulation** ([07.2](../module-07/lesson-02.md)): accumulation sums micro-batch gradients *sequentially on one device* to fake a big batch; data parallelism computes shard gradients *simultaneously on many devices* and averages. Real runs combine both — `global batch = micro_batch × grad_accum_steps × world_size` — which is the identity you will use in every scaling estimate.

## 6. NCCL: the collective library

On NVIDIA GPUs, the collectives themselves are implemented by **NCCL** (the NVIDIA Collective Communications Library, pronounced "nickel"). You do not write the ring by hand in production; you call `torch.distributed.all_reduce(tensor)` and NCCL:

- picks a topology-aware algorithm (ring, tree, or a hybrid) based on how the GPUs are wired — NVLink within a node, InfiniBand across nodes;
- runs the transfers on the GPU's copy engines so they overlap with compute;
- handles the different collectives (`all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`) that the higher-level strategies (DDP here, FSDP in [09.2](lesson-02.md), tensor parallel in [09.3](lesson-03.md)) are built from.

Our `collectives.py` is the *pedagogical* version of what NCCL does: same result, same algorithm structure, none of the hardware.

## 7. PyTorch DDP: overlapping the all-reduce with backward

The obvious way to use the all-reduce is: finish the *entire* backward pass, then all-reduce the whole gradient. That works but wastes time — the network sits idle during backward, then compute sits idle during the all-reduce. PyTorch's **`DistributedDataParallel` (DDP)** removes that waste by **overlapping communication with computation**.

The key observation: backward computes gradients **layer by layer, from the output back to the input**. The last layer's gradient is ready long before the first layer's. DDP exploits this:

- It groups parameters into **buckets** (a few dozen MB each).
- As soon as *all* gradients in a bucket are ready during backward, DDP fires off that bucket's all-reduce **in the background** (a NCCL call on a separate CUDA stream) while backward keeps computing gradients for earlier layers.
- By the time backward reaches the first layer, most buckets' all-reduces are already done or in flight. The step ends when the last bucket's all-reduce finishes.

So the communication is hidden *underneath* the backward compute instead of happening after it. Bucketing (rather than one all-reduce per parameter) matters because each collective has a fixed launch overhead; batching many small gradients into one bucket amortizes it, and a bucket is the granularity at which overlap is triggered.

<div class="callout pt"><p>Using DDP is a three-line change to your existing loop: wrap the model in <code>DistributedDataParallel(model)</code>, use a <code>DistributedSampler</code> so each rank sees a different shard of the data, and launch <code>W</code> processes with <code>torchrun</code>. The forward/backward/step body is <em>unchanged</em> — DDP hooks into <code>backward()</code> to run the bucketed all-reduces automatically. This is the single most common way to scale training, and it is why the loop you wrote in Module 7 already "just works" on many GPUs.</p></div>

<div class="callout warn"><p>Because DDP overlaps the all-reduce with backward, anything that makes gradients ready in an <em>unpredictable</em> order breaks the overlap — most commonly a model where some parameters don't get a gradient every step (conditional branches, unused heads). DDP will hang waiting for an all-reduce that never comes, which is why it offers a <code>find_unused_parameters</code> flag. If a real DDP job mysteriously stalls at the first backward, an unused parameter is the first thing to check.</p></div>

## Exercise

You scale a working single-GPU run to 8 GPUs with DDP. You keep the per-GPU (micro) batch size the same and change nothing else. Training is faster, but the model now converges to a noticeably worse loss than the single-GPU run at the same number of *steps*. What changed, and what is the standard fix?

<details><summary>Hint</summary>
With 8 data-parallel workers, how many examples now contribute to each optimizer step, compared to before? What does that do to the <em>effective</em> batch size, and hence to how the learning rate should be set?
</details>

<details><summary>Stronger hint</summary>
Global batch $=$ micro-batch $\times$ grad-accum $\times$ <code>world_size</code>. You multiplied the effective batch by 8 without touching the learning rate or the schedule length (measured in steps).
</details>

<details><summary>Solution</summary>

Going from 1 to 8 workers multiplied the **effective (global) batch size by 8** — each step now averages gradients over 8× as many examples. A larger batch gives a lower-variance gradient, which generally wants a **larger learning rate** (a common rule of thumb is to scale the LR roughly linearly, or by $\sqrt{8}$, with the batch size, plus a longer warmup), and it means each *step* consumes 8× as many tokens, so "same number of steps" is actually 8× more data — you may instead want the same *token* budget, i.e. fewer steps. The fix is to re-tune the learning rate and warmup for the new global batch and to compare runs at equal token budgets, not equal step counts. The subtlety that trips people up: DDP is *numerically exact* (section 2), so the worse result is **not** a bug in the averaging — it is a hyperparameter mismatch caused by silently changing the effective batch size.

</details>

## Common mistakes

- **Averaging a sum-reduced loss.** The averaging identity (section 2) assumes each worker computes a *mean* loss over its shard. If your loss sums instead of averages, or shards have unequal token counts, a plain gradient average is wrong — weight by tokens.
- **Forgetting the effective batch grew by `W`.** Adding workers multiplies the global batch; the learning rate, warmup, and step budget must be re-tuned (see the exercise).
- **Different data per worker, but not disjoint.** Each worker must see a *different* shard. If every worker draws the same batch (e.g. same RNG seed for the sampler and no rank offset), you have `W` copies of the same gradient — no variance reduction, wasted hardware.
- **Non-identical replicas.** If the workers ever drift apart (a non-deterministic op, a per-worker random init not broadcast at startup), the "everyone applies the same update" invariant breaks and training silently diverges. DDP broadcasts the initial weights from rank 0 for exactly this reason.

## Check yourself

<details><summary>Why does the all-reduce average the gradients rather than sum them?</summary>

Because the training loss is a <em>mean</em> over examples. The full-batch gradient is $\frac1B\sum_i \nabla\ell_i$; the average of the per-shard mean gradients reproduces exactly that (section 2). Summing would give $W\times$ the correct gradient, effectively multiplying the learning rate by $W$.

</details>

<details><summary>Ring all-reduce runs in $2(W-1)$ steps with each link carrying $1/W$ of the tensor per step. Roughly how much total data does each worker send over one all-reduce, and how does it depend on $W$?</summary>

About $2(W-1)/W \approx 2$ tensors' worth — one tensor's worth during reduce-scatter and one during all-gather. It is essentially <em>independent of $W$</em>, which is why the ring scales to large worker counts. The naive "send everything to worker 0" instead loads $2(W-1)$ tensors onto worker 0's single link.

</details>

<details><summary>In the three-worker worked example, after reduce-scatter finishes, which single chunk value does each worker hold, and why isn't the collective done yet?</summary>

Worker 0 holds the finished chunk 1 ($=15$), worker 1 the finished chunk 2 ($=18$), worker 2 the finished chunk 0 ($=12$). Each worker has only <em>one</em> of the three summed entries — the sum is scattered. The all-gather phase is still needed to circulate all three finished chunks so every worker holds the complete $[12,15,18]$.

</details>

<details><summary>How does DDP hide the all-reduce cost, and what property of the backward pass makes it possible?</summary>

Backward produces gradients layer by layer from output to input, so later layers' gradients are ready first. DDP groups parameters into buckets and launches a bucket's all-reduce (on a background CUDA stream) as soon as that bucket's gradients are ready, while backward keeps computing earlier layers. The communication overlaps the remaining backward compute instead of following it.

</details>

<div class="hw">
<p><strong>Hardware track — data-parallel training.</strong></p>
<p><strong>This lesson's code:</strong> pure single-CPU simulation — any laptop, instant, 0 GPU-hours, CPU-only entirely. <strong>A real data-parallel run</strong> needs <code>W</code> GPUs (e.g. 8×A100 in one node connected by NVLink, or many nodes over InfiniBand) and NCCL. Rule of thumb: DDP scales near-linearly in throughput while the all-reduce stays hidden under backward — typically until the gradient tensor is large relative to interconnect bandwidth or <code>W</code> spans many slow-linked nodes. Memory does <strong>not</strong> improve: every worker still holds a full copy of params + grads + optimizer state (the $16P$ bytes of <a href="../module-07/lesson-04.md">07.4</a>), which is the limitation <a href="lesson-02.md">09.2</a> attacks. <strong>GPU-hours:</strong> data parallelism cuts wall-clock roughly by <code>W</code> but the <em>total</em> GPU-hours are unchanged (you use <code>W</code> GPUs for $1/W$ the time).</p>
</div>

## Next

Data parallelism makes a run faster but not smaller — every worker still stores the full model, its gradients, and Adam's optimizer state, so a model that doesn't fit on one GPU still doesn't fit on eight. The next lesson attacks exactly that: **ZeRO and FSDP** shard the parameters, gradients, and optimizer state *across* the data-parallel workers, so per-worker memory falls with `W`.

Continue to [09.2 · ZeRO & FSDP](lesson-02.md).
