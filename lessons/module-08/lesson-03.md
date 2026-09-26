# 08.3 · Profiling & mixed precision

<div class="prereq">
<p><strong>Prerequisites:</strong> the memory hierarchy and HBM-as-bottleneck from <a href="#/lessons/module-08/lesson-01">08.1</a>; arithmetic intensity, the roofline, and "memory-bound vs compute-bound" from <a href="#/lessons/module-08/lesson-02">08.2</a>; the four memory buckets (params, grads, optimizer state, activations) from <a href="#/lessons/module-07/lesson-04">07.4</a>; floating-point basics from <a href="#/lessons/module-01/lesson-01">Module 1</a>.</p>
<p><strong>You will learn:</strong> how to use the <strong>PyTorch profiler</strong> (<code>torch.profiler</code>) to find the hot kernels in a training step and to spot CPU↔GPU gaps (the GPU sitting idle), and how to read the resulting trace; then <strong>mixed precision</strong> — the concrete bit layouts of fp32, tf32, fp16, and bf16, the <em>dynamic range vs precision</em> trade-off, why <strong>bf16</strong> is preferred for training and needs no loss scaling while <strong>fp16</strong> needs a <code>GradScaler</code>, and how <code>torch.autocast</code> is used. You will see, computed on CPU, a case where fp16 overflows to <code>inf</code> but bf16 does not.</p>
<p><strong>Why this matters for ML:</strong> profiling is how you turn the roofline theory of 08.2 into an actual list of what to fix in <em>your</em> run — which kernels dominate, whether the GPU is starving. Mixed precision is the single highest-leverage change you can make: switching the bulk of training to 16-bit halves the bytes every memory-bound kernel moves (directly attacking the roofline), roughly halves activation memory (the bucket from 07.4), and lets the Tensor Cores run at their fast 16-bit rate. Nearly every modern LLM is trained in bf16 mixed precision; understanding exactly why is core knowledge.</p>
</div>

## 1. Intuition: measure before you optimize

The roofline (08.2) tells you what *kind* of fix a kernel needs, but not *which* kernels in your model actually matter. A transformer forward+backward launches thousands of kernels; optimizing the wrong one is wasted effort. The rule is the same as in any performance work: **measure first.** A profiler records, for one real step, how long every kernel took, how they overlap between CPU and GPU, and where time is lost. Then you attack the top of the list — and only the top.

## 2. The PyTorch profiler

`torch.profiler` wraps a slice of your code, records every operator and every CUDA kernel with timestamps, and lets you print a sorted table or export a trace to view on a timeline.

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = model.cuda()
x = x.cuda(); y = y.cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,        # keep tensor shapes so you can see WHICH matmul
    profile_memory=True,       # track allocations (feeds the memory buckets of 07.4)
    with_stack=False,          # set True to attribute time to Python source lines
) as prof:
    for _ in range(5):                     # a few steps so the schedule warms up
        with record_function("train_step"):   # a named span you'll see in the trace
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()           # make timings real (see below)

# 1) sorted table in the terminal
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
# 2) a trace you open in a viewer
prof.export_chrome_trace("trace.json")
```

Two things to internalize about *why* the calls are shaped this way:

- **`ProfilerActivity.CPU` and `.CUDA` are both needed** because work happens on two devices. The CPU (your Python + PyTorch dispatcher) *launches* kernels; the GPU *runs* them. The profiler records both timelines so you can see how they line up.
- **`torch.cuda.synchronize()` matters** because CUDA is asynchronous. When Python calls `loss.backward()`, it does not wait for the GPU — it queues the kernels and returns immediately. Without a synchronize, a naive wall-clock timer would measure only the *launch* time, not the *execution*. The profiler handles device timing correctly via CUDA events, but any manual timing around GPU code must synchronize, or the numbers are fiction. (This is the same async fact behind the `torch.cuda.synchronize()` calls in the benchmark script `code/scripts/bench_attention.py`.)

## 3. Reading the trace: hot kernels and CPU↔GPU gaps

Two questions the trace answers.

**Which kernels are hot?** The sorted table lists operators by total CUDA time. In a healthy transformer step the top entries are the big matmuls (`ampere_..._gemm`, the QKV/MLP/output projections) — that is compute-bound work doing real FLOPs, which is *good*. If instead the top entries are elementwise kernels (`elementwise_kernel`, LayerNorm, softmax, copies) eating a large fraction of time, you have found memory-bound tail (08.2) that fusion or `torch.compile` should collapse. `record_shapes=True` lets you see the shapes so you know *which* matmul or which activation tensor.

**Is the GPU starving?** Open `trace.json` in a timeline viewer (`chrome://tracing`, or Perfetto, or TensorBoard's profiler plugin). You see two rows: a CPU track (Python/launch) and a GPU track (kernel execution). The pattern to hunt for:

- **Healthy:** the GPU track is a solid wall of back-to-back kernels — the GPU is always busy.
- **Bad — a gap:** the GPU track has white space where nothing runs, while the CPU track is busy. This is a **CPU↔GPU gap**: the GPU finished its queued work and is *waiting* for the CPU to launch the next kernel (or for a `.item()` / `.cpu()` call that forces a synchronize, or for the data loader). The GPU is idle — MFU (07.4) is being thrown away not on slow kernels but on *no* kernels.

Common causes of gaps and their fixes: a Python-heavy training loop that cannot launch kernels fast enough (fuse ops / `torch.compile` to launch fewer, bigger kernels); a `loss.item()` or `print(loss)` every step forcing a synchronize (log less often, or move it off the hot path); a data loader that blocks the step (more `num_workers`, prefetching, pinned memory). The trace tells you which one by *where* the gap sits relative to the CPU activity.

<div class="callout key"><p>The profiler answers two questions: (1) <em>which kernels dominate</em> — you want the big matmuls on top (compute-bound = good) and to be suspicious of elementwise/LayerNorm/softmax/copy kernels near the top (memory-bound tail to fuse); (2) <em>is the GPU idle</em> — gaps in the GPU timeline while the CPU is busy mean the GPU is starving on launch overhead, host syncs, or data loading, not on slow math.</p></div>

## 4. Mixed precision: the idea

Everything so far says: move fewer bytes. The most direct way to move fewer bytes for *every* tensor at once is to store them in a **smaller dtype**. fp32 is 4 bytes per number; 16-bit formats are 2 bytes. Halving the dtype:

- halves the HBM traffic of every memory-bound kernel (slides them up the roofline, 08.2),
- halves the activation-memory bucket (07.4) — the one that scales with batch×context and often decides whether the model fits,
- lets the Tensor Cores run their fast 16-bit matmul path (much higher peak FLOP/s than fp32).

But you cannot naively make *everything* 16-bit — some operations lose too much accuracy at low precision. **Mixed precision** is the disciplined version: run the matmul-heavy, error-tolerant operations in 16-bit for speed and memory, but keep the numerically sensitive parts (the master copy of the weights, the optimizer's accumulation, large reductions) in fp32. `torch.autocast` automates the choice per operation.

To understand *which* 16-bit format and *why* the sensitive parts stay fp32, you have to look at the bits.

## 5. The four formats: dynamic range vs precision

A floating-point number splits its bits into a **sign** (1 bit), an **exponent** (sets the *dynamic range* — how large or small a magnitude it can represent), and a **mantissa/fraction** (sets the *precision* — how many significant digits within that magnitude). More exponent bits → wider range; more mantissa bits → finer precision. The formats differ in how they split a fixed budget of bits.

| Format | Total bits | Exponent bits | Mantissa bits | Max representable | Smallest normal | Relative precision (eps) |
|---|---|---|---|---|---|---|
| **fp32** | 32 | 8 | 23 | $3.4\times10^{38}$ | $1.18\times10^{-38}$ | $\approx 1.2\times10^{-7}$ |
| **tf32** | 19* | 8 | 10 | $3.4\times10^{38}$ | $1.18\times10^{-38}$ | $\approx 9.8\times10^{-4}$ |
| **fp16** | 16 | 5 | 10 | $6.55\times10^{4}$ | $6.1\times10^{-5}$ | $\approx 9.8\times10^{-4}$ |
| **bf16** | 16 | 8 | 7 | $3.39\times10^{38}$ | $1.18\times10^{-38}$ | $\approx 7.8\times10^{-3}$ |

(*tf32 is not a storage format — it is a Tensor-Core *compute* mode that keeps fp32's 8 exponent bits but truncates the mantissa to 10 bits for the multiply. Values are still stored as fp32; only the matmul internally rounds. It gives a big matmul speedup for almost free and is on by default for many ops via `torch.backends.cuda.matmul.allow_tf32`.)

The numbers above are exact — verify them yourself:

```python
import torch
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(dt, "max =", fi.max, "smallest_normal =", fi.tiny, "eps =", fi.eps)
# float32  max = 3.40e+38  smallest_normal = 1.18e-38  eps = 1.19e-07
# float16  max = 65504.0   smallest_normal = 6.10e-05  eps = 9.77e-04
# bfloat16 max = 3.39e+38  smallest_normal = 1.18e-38  eps = 7.81e-03
```

The crucial line of the table is the contrast between fp16 and bf16, which have the *same* 16 bits but split them oppositely:

- **fp16** spends 5 bits on exponent, 10 on mantissa. It has **good precision** (eps ≈ $10^{-3}$) but a **narrow range**: its largest value is only **65504**, and its smallest normal is $6.1\times10^{-5}$. Numbers outside $[\,6\times10^{-5},\ 6.5\times10^{4}\,]$ (in magnitude) overflow to `inf` or underflow to `0`.
- **bf16** spends 8 bits on exponent (same as fp32!), only 7 on mantissa. It has the **same enormous dynamic range as fp32** ($\pm 3.4\times10^{38}$, down to $10^{-38}$) but **coarser precision** (eps ≈ $8\times10^{-3}$, ~3 significant decimal digits).

<div class="callout key"><p>fp16 and bf16 are both 16 bits but make opposite trades. <strong>fp16</strong>: more mantissa (finer precision), fewer exponent (tiny range, overflows past 65504). <strong>bf16</strong>: more exponent (full fp32 range, essentially never overflows) at the cost of mantissa (coarser precision). For training, range matters more than the last few bits of precision — which is why bf16 won.</p></div>

## 6. Why bf16 is preferred for training (and needs no loss scaling)

During training, activations and especially **gradients** can span a huge range of magnitudes and occasionally spike very large or shrink very small. bf16's fp32-equal exponent means those values stay representable — a gradient of $10^{-10}$ or an activation of $10^{5}$ is fine. The cost is only ~3 significant digits of precision, and that turns out to be tolerable because the *master weights and the optimizer accumulation are kept in fp32* (section 7): the coarse bf16 is used for the throughput-critical matmuls and the bytes-in-flight, while the slow, error-accumulating parts stay precise.

fp16 has the opposite problem in exactly the place it hurts. Gradients are often small — well below fp16's smallest normal $6.1\times10^{-5}$ — so they **underflow to zero** in fp16 and the update is lost. The standard fix is **loss scaling** via `torch.cuda.amp.GradScaler`: multiply the loss by a large factor $s$ before `backward()`, which scales every gradient up by $s$ into fp16's representable range; then divide the gradients by $s$ (in fp32) before the optimizer step. The scaler even adjusts $s$ dynamically — if it detects an `inf`/`nan` (the gradients overflowed the top of fp16's range), it skips the step and lowers $s$. It works, but it is extra machinery and a source of bugs.

**bf16 needs none of this.** Its range already covers where gradients live, so there is nothing to underflow past and nothing to scale. You simply autocast to bf16 and train. This — not speed, since fp16 and bf16 run at the same Tensor-Core rate — is the decisive reason modern LLM pretraining uses bf16: it removes the loss-scaling failure mode entirely. (bf16 requires Ampere/A100 or newer hardware; on older GPUs fp16+GradScaler is the fallback.)

<div class="callout key"><p><strong>bf16 for training needs no loss scaling</strong> because its dynamic range equals fp32's, so small gradients do not underflow. <strong>fp16 needs a <code>GradScaler</code></strong>: multiply the loss by $s$ so gradients land in fp16's narrow representable band, then unscale in fp32 before <code>optimizer.step()</code>, skipping steps that overflowed. Same speed; bf16 just deletes a whole class of numerical bugs.</p></div>

## 7. `torch.autocast`: how mixed precision is actually used

You do not manually cast tensors. You wrap the forward pass (and loss) in an **autocast** context; PyTorch then runs each operation in the dtype appropriate for it — matmuls and convolutions in the 16-bit type (fast, memory-bound relief), while numerically sensitive ops like large reductions, softmax normalization, and losses stay in fp32 internally. The **weights remain stored in fp32** (the "master copy"); autocast casts them to 16-bit *on the fly* for the fast ops, so the optimizer still updates a precise fp32 copy.

The bf16 path (preferred, no scaler):

```python
for x, y in loader:
    x, y = x.cuda(), y.cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(x, y)      # matmuls run in bf16; loss computed safely
    loss.backward()                      # grads in fp32 (bf16 range is fine)
    optimizer.step()                     # updates the fp32 master weights
    optimizer.zero_grad(set_to_none=True)
```

The fp16 path (needs the scaler):

```python
scaler = torch.cuda.amp.GradScaler()
for x, y in loader:
    x, y = x.cuda(), y.cuda()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits, loss = model(x, y)
    scaler.scale(loss).backward()        # scale loss up so small grads survive fp16
    scaler.step(optimizer)               # unscales, skips step if inf/nan seen
    scaler.update()                      # adjust the scale factor for next step
    optimizer.zero_grad(set_to_none=True)
```

The only differences are the `dtype` and the three `scaler.*` calls — which is the entire practical cost of fp16's narrow range.

<div class="callout pt"><p>"Autocast picks the dtype for you" is not the explanation — here is what it is doing: it keeps a per-operator policy list. Ops that are matmul-shaped and error-tolerant (Linear, matmul, conv, attention) it runs in the 16-bit type, casting their fp32 inputs down on entry — this is where the speed and the halved activation bytes come from. Ops that are numerically fragile (softmax, layernorm's reduction, log/exp, the loss) it keeps in fp32 so precision is not lost where it matters. The weights themselves stay fp32 in memory; the 16-bit versions are transient. That division of labor is exactly the "mixed" in mixed precision.</p></div>

## 8. Numerical example: fp16 overflows where bf16 does not

Here is the range difference made concrete — and it runs on CPU, no GPU needed, because it is purely about the number formats. Squaring a moderately large activation, $300$:

$$
300^2 = 90000.
$$

$90000 > 65504$, the largest fp16 value. So in fp16 it overflows to `inf`; in bf16 (range up to $3.4\times10^{38}$) it is represented fine, though rounded to the nearest bf16 value:

```python
import torch
a = torch.tensor(300.0)
print((a.half()   * a.half()))    # tensor(inf,     dtype=torch.float16)
print((a.bfloat16() * a.bfloat16()))  # tensor(90112., dtype=torch.bfloat16)
print(torch.tensor(70000.0).half())     # tensor(inf,   dtype=torch.float16)  -- even storing 70000 overflows
print(torch.tensor(70000.0).bfloat16())  # tensor(70144., dtype=torch.bfloat16)
```

Read both outputs. fp16 cannot even *store* 70000 — past its max, it is `inf`. bf16 stores 70000 as 70144 (note the imprecision: only ~3 significant digits, bf16's coarse mantissa) but it does not overflow. In training, intermediate values like this — a large activation, an attention score before softmax, a squared term in a variance — appear routinely; fp16 would produce `inf`/`nan` and require loss scaling to manage, while bf16 sails through. That single behavioral difference, verified above, is why the field standardized on bf16.

## Exercise

You profile a training step and the sorted table shows: 55% of CUDA time in `ampere_sgemm` (matmuls), 30% split across `elementwise_kernel`, `layer_norm`, and `softmax`, and the trace shows small gaps in the GPU timeline right after each step where the CPU is calling `loss.item()` to log. **(a)** Which 30% is the memory-bound tail, and what should you try? **(b)** What is causing the GPU gaps and how do you close them? **(c)** You switch the run from fp32 to bf16 autocast. Which of the three cost centers benefits most, and why does bf16 (not fp16) let you do this without adding a `GradScaler`?

<details><summary>Hint</summary>
(a) elementwise/layernorm/softmax are the low-AI kernels from 08.2. (b) `.item()` forces a host sync — think about async CUDA from section 2. (c) 16-bit halves bytes for memory-bound kernels and doubles matmul rate; bf16's range vs fp16's.
</details>

<details><summary>Solution</summary>

**(a)** The 30% in `elementwise_kernel` + `layer_norm` + `softmax` is the memory-bound tail (low arithmetic intensity, 08.2). Fuse it — `torch.compile(model)` will collapse the elementwise chains and often the LayerNorm into fewer kernels, cutting HBM round trips. The 55% in matmuls is compute-bound work you leave alone (except that bf16 speeds it — part c).

**(b)** `loss.item()` copies a scalar from GPU to CPU, which forces a `torch.cuda.synchronize()` — the CPU blocks until the GPU drains its queue, and the GPU then idles waiting for the next launch. Log less often (e.g. every N steps), or accumulate the loss on-device and `.item()` occasionally, so the host sync is off the per-step hot path. The gap closes because the GPU is no longer forced to stop and wait.

**(c)** The memory-bound 30% benefits most in *relative* terms — halving the dtype halves its HBM traffic, sliding it up the roofline — and the matmuls also speed up via the fast 16-bit Tensor-Core path; activation memory roughly halves too. bf16 lets you skip the `GradScaler` because its exponent range equals fp32's, so small gradients do not underflow; fp16's narrow range ($6\times10^{-5}$ to $65504$) would drop small gradients to zero, requiring loss scaling to shift them into range.

</details>

## Common mistakes

- **Timing GPU code without synchronizing.** CUDA is async; a bare `time.perf_counter()` around `loss.backward()` measures launch time, not run time. Use the profiler or wrap manual timers with `torch.cuda.synchronize()`.
- **Calling `.item()`/`.cpu()`/`print(loss)` every step.** Each forces a host sync and stalls the GPU (the gap in section 3). Keep logging off the hot path.
- **Using fp16 without a GradScaler.** Small gradients silently underflow to zero and the model quietly fails to learn — no crash, just a bad loss curve. Either scale, or use bf16.
- **Expecting bf16 to match fp32 loss curves to many digits.** bf16 has only ~3 significant digits; small run-to-run differences are normal. Correctness comes from the fp32 master weights and fp32 accumulation, not from bf16 precision.
- **Assuming mixed precision fixes a compute-bound bottleneck by halving bytes.** For the matmuls the win is the faster 16-bit Tensor-Core path, not bandwidth; for the elementwise tail the win is the halved bytes. Different mechanisms — the profiler + roofline tell you which applies where.

## Check yourself

<details><summary>fp16 and bf16 are both 16 bits. State the bit split and the practical consequence of each.</summary>

fp16 = 1 sign + 5 exponent + 10 mantissa: fine precision but a narrow range (max 65504, smallest normal $6.1\times10^{-5}$), so it overflows/underflows easily and needs loss scaling in training. bf16 = 1 sign + 8 exponent + 7 mantissa: the same range as fp32 (max $3.4\times10^{38}$) but coarser precision (~3 significant digits), so it rarely overflows and needs no loss scaling. Range beats precision for training, so bf16 is preferred.

</details>

<details><summary>Why does fp16 training need a GradScaler but bf16 does not?</summary>

fp16's smallest normal is ~$6\times10^{-5}$; many gradients are smaller and would underflow to zero in fp16, losing the update. A GradScaler multiplies the loss by a large factor so gradients land in fp16's representable band, then unscales in fp32 before the step (and skips steps that overflowed). bf16 has fp32's full range, so small gradients stay representable and there is nothing to scale.

</details>

<details><summary>In a profiler trace, what does a gap in the GPU timeline while the CPU track is busy indicate, and name two likely causes.</summary>

The GPU is idle — it drained its queued kernels and is waiting on the CPU. Likely causes: (1) a host synchronization such as `loss.item()`/`.cpu()`/`print(loss)` each step; (2) the CPU cannot launch kernels fast enough (Python overhead, un-fused tiny kernels) or the data loader is blocking the step. Fixes: log less often / accumulate on device; `torch.compile` to launch fewer bigger kernels; more data-loader workers with prefetch.

</details>

<details><summary>You autocast to bf16 but the model's weights are still fp32 in memory. Is that a bug?</summary>

No — that is exactly how mixed precision works. The fp32 weights are the *master copy* the optimizer updates precisely; autocast casts them to bf16 on the fly for the matmul-shaped ops (for speed and to halve bytes) and computes fragile ops (softmax, reductions, loss) in fp32. Keeping the master weights and optimizer accumulation in fp32 is what preserves training quality while the 16-bit path provides the throughput.

</details>

## Next

You can now find the bottleneck in a real run (profiler) and apply the highest-leverage fix (bf16 mixed precision) with a clear picture of the bits behind it. But one operation resists all of this at long context: attention still materializes an $O(T^2)$ score matrix in HBM, a memory-bound giant that no dtype change fully tames. The final lesson rebuilds attention itself — **FlashAttention** — using the tiling and online-softmax ideas that keep the score matrix in SRAM and off HBM entirely, and you will implement the algorithm from scratch and benchmark it.

Continue to [08.4 · FlashAttention & a Triton kernel; benchmark](lessons/module-08/lesson-04.md).
