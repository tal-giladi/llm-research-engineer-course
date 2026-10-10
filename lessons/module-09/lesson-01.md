# 09.1 · Data parallelism, all-reduce, DDP

<div class="prereq">
<p><strong>Prerequisites:</strong> the canonical training loop <code>get_batch → model(x,y) → zero_grad → backward → clip → step</code> from <a href="#/lessons/module-07/lesson-01">07.1 · The training loop</a>; micro-batch / global batch and gradient accumulation from <a href="#/lessons/module-07/lesson-02">07.2 · Micro-batch, global batch, gradient accumulation</a>; that a language-model loss is a <em>mean</em> cross-entropy over tokens (<a href="#/lessons/module-07/lesson-01">07.1</a>).</p>
<p><strong>You will learn:</strong> how <strong>data parallelism</strong> puts a full copy of the model on each of <code>W</code> workers, splits every global batch into <code>W</code> shards, and averages the per-shard gradients with an <strong>all-reduce</strong> so every worker applies the identical update; the exact reason averaging per-shard gradients equals the full-batch gradient (a two-line proof, verified numerically); how <strong>ring all-reduce</strong> moves the data in <code>2(W-1)</code> steps carrying only <code>1/W</code> of the tensor per link; what NCCL is; and how PyTorch <strong>DDP</strong> overlaps the all-reduce with the backward pass.</p>
<p><strong>Why this matters for ML:</strong> data parallelism is the first and most common way runs get faster — it is how you turn one GPU into eight, or eight into thousands, with almost no change to the training loop you already wrote. Every larger parallelism strategy in this module (ZeRO, FSDP, tensor/pipeline parallelism) is built on top of the collective you meet here.</p>
</div>

<div class="callout warn"><p><strong>Single-CPU note.</strong> The NumPy examples <strong>simulate</strong> data parallelism in one process — a Python loop plays all the workers. They demonstrate the arithmetic, not a multi-GPU speedup. Section 5 connects that simulation to the training loop; section 7 includes a separate runnable CUDA/NCCL example for a multi-GPU machine. Real-hardware requirements live in the <code>&lt;div class="hw"&gt;</code> boxes.</p></div>

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

The true sum is $[12,\ 15,\ 18]$. Watch the ring produce it. Step through the animation with **Next** (each round is a *send* frame, where all three chunks are in flight at once, then a *received* frame showing the addition or copy), then check every number against the hand calculation below.

<iframe src="assets/interactive/ring-all-reduce.html" title="Ring all-reduce animation, three workers" style="width:100%;height:520px;border:0;border-radius:14px" onload="try{var f=this,w=f.contentWindow,m=f.contentDocument.querySelector('main'),s=function(){f.style.height=(m.offsetHeight+4)+'px'};s();w.addEventListener('resize',s);f.contentDocument.addEventListener('click',s,true)}catch(e){}"></iframe>

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

So far, we have combined vectors. Now connect those vectors to something you already know: **the `.grad` tensors that `backward()` produces**. The ring does not combine training examples, predictions, or updated weights. In this lesson, it combines gradients *before* the optimizer changes the weights.

### Follow one optimizer step, with actual numbers

Suppose our model has just three parameters, initially `params = [10, 20, 30]`. Every worker starts with that same vector. Split the global batch into three equal shards, and suppose backward on those shards produces the vectors from section 3:

| Moment | Worker 0 | Worker 1 | Worker 2 |
| --- | --- | --- | --- |
| Starting parameters | `[10, 20, 30]` | `[10, 20, 30]` | `[10, 20, 30]` |
| Data used for forward/backward | Shard 0 | Shard 1 | Shard 2 |
| Local gradient after backward | `[1, 2, 3]` | `[4, 5, 6]` | `[7, 8, 9]` |
| Gradient after all-reduce **mean** | `[4, 5, 6]` | `[4, 5, 6]` | `[4, 5, 6]` |
| Parameters after SGD, `lr = 0.1` | `[9.6, 19.5, 29.4]` | `[9.6, 19.5, 29.4]` | `[9.6, 19.5, 29.4]` |

The last row comes from the familiar SGD update:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta\,g_{\text{average}}
= [10,20,30] - 0.1[4,5,6]
= [9.6,19.5,29.4].
$$

The gradients differ initially because the workers saw different data. The weights still agree afterward because **all workers use the same averaged gradient**. Nobody needs to send the updated weights around after this step. With Adam, the same reasoning also requires matching optimizer state and settings on every worker.

<div class="callout key"><p><code>backward()</code> computes gradients; it does <strong>not</strong> update parameters. All-reduce makes the gradients agree; it also does <strong>not</strong> update parameters. Only <code>optimizer.step()</code> changes the parameters. Keep those three actions separate in your head.</p></div>

### Where the new operation goes

Here is the familiar training step, written with the gradient reset before forward. Resetting immediately before backward would also work for this simple loop:

```python
# Single GPU — conceptual training-loop fragment
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model(x), y)
loss.backward()                                # fill parameter.grad
# Optional: clip the gradients here.
optimizer.step()                               # consume parameter.grad
```

Manual data parallelism inserts synchronization **after local backward and before clipping or the optimizer step**:

```python
# Each worker runs this fragment on its OWN model and local batch.
# Assumptions: identical initial models, equal-size shards, mean loss,
# an initialized process group, and every parameter used in the loss.
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model(local_x), local_y)
loss.backward()                                # local gradients only

for parameter in model.parameters():
    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    parameter.grad.div_(world_size)             # replace sum with mean

# Optional: clip the averaged gradients here.
optimizer.step()                               # same update on every worker
```

`all_reduce` modifies the supplied tensor **in place**. For example, worker 0's `.grad` changes from `[1, 2, 3]` to the sum `[12, 15, 18]`; dividing by 3 changes it to `[4, 5, 6]`. The optimizer then reads that new `.grad` value. You do not need a different optimizer for data parallelism.

The `for` loop above walks over **parameter tensors**, not workers. A linear layer has a weight tensor and a bias tensor, so it causes two collective calls. Each worker makes those calls in the same order: weights with weights, then biases with biases. The participants must agree on tensor shapes, types, and collective order.

<div class="callout warn"><p>Do not call <code>optimizer.step()</code> on the local gradients and average afterward. The workers would already have made different updates. Also clip <em>after</em> averaging: clipping each local gradient first generally produces a different result because clipping is nonlinear.</p></div>

### What the CPU simulation represents

The helper in `llmre/distributed/data_parallel.py` compresses the gradient-producing part into three lines:

```python
def data_parallel_grads(grad_fn, params, batch, world_size):
    shards = split_batch(batch, world_size)               # W equal shards
    grads = [grad_fn(params, shard) for shard in shards]  # one local gradient per shard
    return ring_all_reduce(grads, op="mean")              # one averaged gradient
```

Read its inputs literally:

- `params`: the current parameter values, shared as the starting point for all simulated workers.
- `batch`: the full batch of inputs and targets.
- `grad_fn(params, shard)`: compute the gradient of the **mean loss on this shard** at these parameters. The `mean_grad` function from section 2 is one example.
- `world_size`: the number of workers being simulated.

This function returns a gradient; **it does not update `params`**. For the NumPy regression example, a complete SGD step is:

```python
# Continuing the NumPy setup from section 2:
from llmre.distributed.data_parallel import data_parallel_grads

learning_rate = 0.1
grad = data_parallel_grads(mean_grad, theta, (X, y), W)
theta = theta - learning_rate * grad
```

One Python process computes each shard's gradient in turn. The ring helper returns just one copy of the final result because all simulated workers' final buffers agree. On real GPUs, each process keeps its own copy of the averaged gradient and updates its own model. **The simulation reproduces the arithmetic, not the parallel execution.**

### What “one process per GPU” actually means

Imagine launching this command on a machine with four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

It starts **four Python processes running the same file**. Each process has its own model, optimizer, Python variables, and assigned GPU. A worker is one of these processes.

| Name | Meaning | Example on one four-GPU machine |
| --- | --- | --- |
| `world_size` | Total participating processes | `4` in every process |
| `rank` | This process's ID in the whole group | `0`, `1`, `2`, or `3` |
| `local_rank` | This process's GPU index on its machine | Also `0`, `1`, `2`, or `3` here |

On multiple machines, global ranks remain unique, while local ranks start at zero on each machine. For a global batch of 32 examples and four equal shards, each process runs forward/backward on 8 examples. Those computations happen concurrently.

There is no Python process looping over the other GPUs' `backward()` calls. There are four processes, each doing **one local backward** and participating in the matching collectives. Section 7 gives a runnable example using DDP to automate those collectives.

### Combining this with gradient accumulation

Accumulation and data parallelism answer two different questions: **how many micro-batches does each worker process before an update, and how many workers contribute?** If each worker processes `A` micro-batches of `b` examples, then:

$$
B_{\text{global}} = b \times A \times W.
$$

For `b = 2`, `A = 4`, and `W = 3`, each worker contributes 8 examples and one optimizer update covers 24 examples in total. For equal-size micro-batches with mean losses, divide each micro-batch loss by `A` before backward. That averages over the accumulation window; the worker average supplies the separate factor `1/W`.

<div class="callout key"><p>Accumulation combines work done <strong>over time on each worker</strong>. Data parallelism combines work done <strong>across workers</strong>. In both cases, <code>optimizer.step()</code> runs only after the intended combined gradient is ready.</p></div>

## 6. NCCL: the collective library

You now know what must happen: each GPU supplies a gradient tensor, and each GPU must receive the same combined tensor. **NCCL** (NVIDIA Collective Communications Library, pronounced “nickel”) supplies the GPU communication operations used to accomplish that.

There are three levels to distinguish:

| Level | Responsibility |
| --- | --- |
| Your training loop | Choose data, compute the loss, request backward, and run the optimizer |
| PyTorch distributed / DDP | Coordinate processes and request gradient collectives |
| NCCL backend | Execute collectives across NVIDIA GPUs using the available connections |

NCCL knows about tensors and collective operations. It does not know what a training example, loss, or learning rate means. If PyTorch asks it to sum three gradient vectors, it combines their corresponding entries. The interpretation as “training on a larger batch” comes from our loss definition and the averaging argument in section 2.

The ring from section 3 is **one implementation of all-reduce**, not a requirement of the operation. NCCL can select different algorithms and transports. NVLink, PCIe, and network connections affect the available bandwidth; the mathematical result required by the collective remains the same, up to floating-point reduction order.

<div class="hw"><p><strong>Hardware:</strong> the runnable example below uses one NVIDIA GPU per process and the NCCL backend. NVLink is not required. Communication is GPU work scheduled on CUDA streams and may overlap computation; it is not guaranteed to be a free transfer handled only by copy engines. The CPU simulation cannot measure that overlap or predict its speedup.</p></div>

For implementation details, see the [NCCL collective operations guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).

## 7. PyTorch DDP: overlapping the all-reduce with backward

The manual loop in section 5 waits for all local gradients, then communicates each parameter tensor. **`DistributedDataParallel` (DDP)** is a wrapper around your model that arranges gradient synchronization during backward.

### What changes in the training step?

After wrapping the model, the ordinary loop is enough:

```python
# Fragment: process-group setup and data loading are shown below.
model = DistributedDataParallel(model, device_ids=[local_rank])

optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model(local_x), local_y)
loss.backward()                 # local backward + DDP's gradient synchronization
optimizer.step()                # consumes averaged gradients
```

**Do not add the manual all-reduce loop from section 5 here.** Default DDP already averages the gradients. It does not shard your input batch or call the optimizer for you. See the [DDP API reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) for the wrapper's contract.

### Why communication can begin before backward finishes

Think of a model with three layers: input layer A, middle layer B, and output layer C. Forward visits A, B, C. Backward typically produces C's gradients first, then B's, then A's.

Once C's gradients exist, communicating them does not require waiting for A's gradients. DDP groups parameter gradients into **buckets** and arranges reduction when a bucket is ready, while backward continues. A bucket can contain several parameter tensors; it is not necessarily one layer.

Here is a deliberately simplified timing example with one bucket per layer. Suppose each layer's backward takes 4 ms and each bucket's communication takes 3 ms:

| Time | Gradient computation | Communication |
| --- | --- | --- |
| 0–4 ms | Compute C's gradients | Nothing ready yet |
| 4–8 ms | Compute B's gradients | Reduce C's bucket during 4–7 ms |
| 8–12 ms | Compute A's gradients | Reduce B's bucket during 8–11 ms |
| 12–15 ms | Backward computation finished | Finish A's bucket |

Doing the same three communications only after backward would take `12 + 9 = 21 ms`. This illustrative overlap takes 15 ms. **The final 3 ms is still exposed communication time.** These numbers explain the schedule; they are not a benchmark or a promised speedup.

Play the same schedule below, or step through it with **Next event**. The top shows both GPUs' A, B, C buckets moving from *computing* to *reducing* to *averaged*; the bottom timeline puts the DDP overlap and the communicate-afterward schedule on one time axis so you can see the 6 ms that get hidden and the 3 ms that stay exposed.

<iframe src="assets/interactive/ddp-overlap.html" title="DDP overlap of all-reduce with backward, two GPUs" style="width:100%;height:1400px;border:0;border-radius:14px" onload="try{var f=this,w=f.contentWindow,d=f.contentDocument,m=d.querySelector('main'),s=function(){f.style.height=(m.offsetHeight+4)+'px'};s();w.addEventListener('resize',s);d.addEventListener('click',function(){setTimeout(s,0)},true);d.addEventListener('toggle',s,true);d.addEventListener('input',s,true)}catch(e){}"></iframe>

Buckets also avoid paying collective-launch overhead separately for every tiny parameter tensor. Larger buckets can reduce overhead but become ready later; smaller buckets can start earlier but require more calls. Real performance depends on both the computation and the interconnect. The [PyTorch DDP design note](https://docs.pytorch.org/docs/main/notes/ddp.html) describes this scheduling.

<div class="callout key"><p>DDP does not make all-reduce disappear. It starts communication for ready gradients while other gradients are still being computed. Any communication left at the end must finish before the optimizer can use those gradients.</p></div>

### A complete, small DDP training script

Save this as `train.py` and run it with the four-GPU command from section 5. This is a runnable teaching example using synthetic regression data, full precision, and no accumulation. It includes process cleanup and failure logging; a real training application would additionally supply its dataset, checkpoints, and recovery policy.

```python
import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

log = logging.getLogger(__name__)


def train(local_rank):
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Every rank refers to the same logical dataset. The sampler selects rows.
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(256, 16, generator=generator)
    y = x.sum(dim=1, keepdim=True)
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler, drop_last=True)
    if len(loader) == 0:
        raise ValueError("Not enough examples for one full batch per rank.")

    # DDP synchronizes the starting model across ranks by default.
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 1).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss(reduction="mean")
    model.train()

    for epoch in range(3):
        sampler.set_epoch(epoch)                 # a new coordinated shuffle
        loss_sum = torch.zeros((), device=device)

        for local_x, local_y in loader:
            local_x = local_x.to(device)
            local_y = local_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(local_x), local_y)
            loss.backward()                     # DDP averages gradients here
            optimizer.step()                    # every rank takes the same step
            loss_sum += loss.detach()

        # Separate collective for a reporting metric; not gradient synchronization.
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        if rank == 0:
            mean_loss = (loss_sum / (len(loader) * world_size)).item()
            log.info("epoch=%d mean_loss=%.6f", epoch, mean_loss)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("Launch with torchrun, not python train.py.")
        local_rank = int(os.environ["LOCAL_RANK"])
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise RuntimeError("Each local process needs an available CUDA GPU.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
        train(local_rank)
    except Exception:
        log.exception("Training failed on rank %s", os.environ.get("RANK", "unknown"))
        raise                                   # let the launcher report the failure
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Trace the responsibilities: `torchrun` launches the processes; `init_process_group` connects them; `DistributedSampler` chooses each rank's data indices; DDP synchronizes the model and gradients; the optimizer updates the local replica. The final `all_reduce(loss_sum)` is only for logging a global loss. **DDP synchronizes gradients, not the Python `loss` variable.**

Here each local batch contains 8 examples, so four ranks contribute 32 examples per optimizer step. The two `drop_last=True` settings keep shard and batch sizes regular: the sampler can discard examples to divide the dataset among ranks, and the loader discards incomplete local batches. Without sampler dropping, it can pad with repeated indices. `set_epoch(epoch)` changes the shared shuffle each epoch; see the [sampler implementation and documentation](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/distributed.py).

### When accumulation or conditional branches enter the picture

For accumulation over `A` equal micro-batches, use `model.no_sync()` around **both forward and backward** for the first `A-1` micro-batches. On the last one, use normal synchronization. Divide each mean loss by `A`, and call the optimizer once after the window. This avoids synchronizing every intermediate backward; the [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) covers this pattern.

<div class="callout warn"><p>All ranks must follow a compatible collective sequence. A rank that skips backward, crashes, or runs out of batches early can leave peers waiting. Unused parameters are a separate issue: models with branches may need <code>find_unused_parameters=True</code>, depending on their graph. This flag is not a general repair for mismatched loops. Investigate the first error across all ranks before retrying the job.</p></div>

## Exercise

A single GPU trains with micro-batch size 8 and accumulates 4 micro-batches before each optimizer step. You move to 4 GPUs with DDP and keep both settings unchanged.

1. How many examples contribute to each optimizer step before and after the change?
2. How could you preserve the original global batch while keeping micro-batch size 8?
3. If you keep the larger global batch, is comparing the runs after 1,000 steps a comparison at equal data exposure?

<details><summary>Hint</summary>
Count examples per optimizer update using <code>micro_batch × grad_accum_steps × world_size</code>. DDP's gradient averaging controls the gradient's normalization; it does not cancel the extra examples processed by additional workers.
</details>

<details><summary>Stronger hint</summary>
The original global batch is <code>8 × 4 × 1</code>. The new one is <code>8 × 4 × 4</code>. To keep the original total, one of the other factors must shrink when the worker count grows.
</details>

<details><summary>Solution</summary>

The global batch increases from **32 to 128 examples** per update. Set accumulation to **1** on the four-GPU run to retain a global batch of `8 × 1 × 4 = 32`.

If you keep 128, then 1,000 updates process 128,000 examples instead of 32,000. For equal example exposure, compare 250 updates of the larger-batch run with 1,000 updates of the original. For language models, count valid training tokens as well.

A changed global batch can change optimization behavior, so re-evaluate the learning rate, warmup, and schedule. There is no universally correct “multiply the learning rate by the GPU count” rule, and a larger batch does not automatically make convergence worse. First decide whether your goal is to preserve the original training setup or to tune a new, larger-batch setup.

</details>

## Common mistakes

- **Confusing gradients with weights.** All-reduce combines gradients; the optimizer then updates each local copy of the weights.
- **Synchronizing twice.** The manual all-reduce loop is an explanation of DDP's job, not extra code to add after DDP backward. An extra division by `W` incorrectly shrinks the gradient.
- **Changing the global batch accidentally.** Adding workers while preserving the local micro-batch and accumulation count increases examples per update. Recalculate all three factors together.
- **Assuming DDP splits the data.** The wrapper does not choose examples. Configure the input pipeline so workers contribute the intended different samples.
- **Averaging equally when token counts differ.** Equal sequence counts do not guarantee equal valid-token counts. With local mean-loss gradients `g_w` and valid-token counts `n_w`, the global token-mean gradient is `sum(n_w * g_w) / sum(n_w)`. Default equal worker averaging only matches it when those counts agree. Accumulation windows need the same care.
- **Using a sum loss without accounting for normalization.** Decide whether your intended objective is a global sum or a global mean, then scale consistently. Averaging local sums is not the same as either objective without the appropriate factor.
- **Clipping local gradients before synchronization.** Average first, then clip the gradient the optimizer will actually use. With gradient scaling, unscale before clipping as well.
- **Letting replica state differ.** Matching gradients are insufficient if ranks use different optimizer settings, restored optimizer states, or numbers of updates. Distinct local losses or dropout masks, however, are normal and do not by themselves break synchronization.

## Check yourself

<details><summary>In the three-worker example, why does every worker update to the same parameters even though its local gradient was different?</summary>

Before the update, all-reduce replaces the local gradients with the common mean `[4, 5, 6]`. Starting from `[10, 20, 30]`, every worker's SGD update with learning rate 0.1 therefore gives `[9.6, 19.5, 29.4]`.

</details>

<details><summary>Why average gradients instead of simply summing them?</summary>

For equal-size shards and local mean losses, the average reproduces the global mean-loss gradient. Using the sum instead gives a gradient `W` times larger. The identity is mathematical; different floating-point reduction orders can still cause small numerical differences from a single-device computation.

</details>

<details><summary>After reduce-scatter in section 3, which finished chunk belongs to each worker, and what remains to be done?</summary>

Worker 0 owns chunk 1 with value 15; worker 1 owns chunk 2 with value 18; worker 2 owns chunk 0 with value 12. All-gather distributes those finished chunks so all workers have `[12, 15, 18]`. Dividing by 3 gives the mean `[4, 5, 6]`.

</details>

<details><summary>How much does each worker send during ring all-reduce?</summary>

For a tensor of size `S` bytes and equal chunks, it sends `2(W-1)S/W` bytes: `W-1` chunks in each of two phases. It also receives that amount. Each direction approaches `2S` as `W` grows; sent plus received approaches `4S`. This bandwidth accounting does not mean latency is constant: the ring still has `2(W-1)` rounds.

</details>

<details><summary>With DDP, what is ready for the optimizer after a normal synchronized backward?</summary>

The averaged gradients are ready for subsequent optimizer operations. Forward and backward used each worker's local data, but DDP arranged the communication during backward. The optimizer still runs on every rank. The scalar loss remains local unless you explicitly aggregate it for reporting.

</details>

<details><summary>What does overlap save, and what can it not guarantee?</summary>

It lets communication for ready gradient buckets run while backward computes other gradients. It can reduce the time spent waiting after gradient computation finishes. It cannot guarantee that all communication is hidden, that communication costs no GPU resources, or that adding workers gives a proportional speedup.

</details>

<div class="hw">
<p><strong>Hardware track — data-parallel training.</strong></p>
<p><strong>CPU path:</strong> sections 2–5 include arithmetic simulations that run on one CPU. <strong>GPU path:</strong> section 7 includes a real <code>torchrun</code> example for NVIDIA GPUs with NCCL; its runtime and memory use depend on the environment.</p>
<p><strong>Memory:</strong> ordinary DDP keeps a full model, gradient tensors, and optimizer state on every worker, plus communication-related storage. Adding workers does not divide model-state memory by <code>W</code>. The <code>16P</code> estimate in <a href="#/lessons/module-07/lesson-04">07.4</a> applies to that lesson's particular mixed-precision Adam accounting, not to every optimizer or precision setup.</p>
<p><strong>Speed and cost:</strong> for a fixed amount of work, ideal <code>W</code>-fold speedup would preserve total GPU-hours: <code>W</code> GPUs for <code>1/W</code> of the time. Real communication, input loading, and small local batches reduce scaling efficiency, so total GPU-hours can rise. Measure throughput and time to the desired training quality rather than assuming linear speedup.</p>
</div>

## Next

DDP distributes the examples while replicating the model state. That makes it useful when the model fits on one GPU and you want several GPUs to process training data together. It does not solve a model-state memory limit by itself.

The next lesson changes what each worker stores: **ZeRO and FSDP** partition optimizer state, gradients, and/or parameters, depending on the strategy and stage. That is how data-parallel workers can share the memory burden as well as the computation.

Continue to [09.2 · ZeRO & FSDP](lessons/module-09/lesson-02.md).
