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

The snippet below shows the *shape* of the call: `model`, `optimizer`, `x`, `y` stand for whatever your training loop already has. A complete, self-contained version you can paste into Colab is in [section 2.1](#/lessons/module-08/lesson-03?id=_21-run-it-yourself-on-a-colab-t4).

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

What the printed table looks like (illustrative, from the small 4-layer model of section 2.1 on a T4 in fp32, 5 steps; the real table has more columns — CPU times, memory — trimmed here to the ones you read first):

```text
-------------------------------------------------  ------------  ------------  ----------
Name                                                  Self CUDA    CUDA total  # of Calls
-------------------------------------------------  ------------  ------------  ----------
train_step                                              0.000us     590.412ms           5
aten::mm                                              296.540ms     296.540ms         170
volta_sgemm_128x64_nt                                 157.030ms     157.030ms          85
volta_sgemm_128x64_nn                                 139.510ms     139.510ms          85
aten::addmm                                            96.020ms      96.020ms          85
volta_sgemm_128x128_tn                                 96.020ms      96.020ms          85
aten::_efficient_attention_backward                    30.480ms      30.480ms          20
fmha_cutlassB_f32_aligned_64x64_k64_sm75               30.480ms      30.480ms          20
aten::gelu                                             15.470ms      15.470ms          20
aten::native_dropout                                   13.010ms      13.010ms          60
fmha_cutlassF_f32_aligned_64x64_rf_sm75                12.060ms      12.060ms          20
aten::native_layer_norm                                11.020ms      11.020ms          45
-------------------------------------------------  ------------  ------------  ----------
Self CPU time total: 612.874ms
Self CUDA time total: 590.412ms
```

How to read it: `aten::*` rows are PyTorch operators, the rows below them with cryptic names are the actual GPU kernels they launched (so the same time appears twice — once on the op, once on its kernel). `aten::addmm` is the forward of every `nn.Linear` (17 per step × 5 steps = 85 calls); `aten::mm` is the backward (two matmuls per Linear: gradient w.r.t. input and w.r.t. weight → 170). The three `sgemm` kernels together are ~66% of GPU time — the healthy picture: compute-bound matmuls on top. GELU, dropout and LayerNorm are the memory-bound tail (08.2), small here but exactly what fusion would remove.

Two things to internalize about *why* the calls are shaped this way:

- **`ProfilerActivity.CPU` and `.CUDA` are both needed** because work happens on two devices. The CPU (your Python + PyTorch dispatcher) *launches* kernels; the GPU *runs* them. The profiler records both timelines so you can see how they line up.
- **`torch.cuda.synchronize()` matters** because CUDA is asynchronous. When Python calls `loss.backward()`, it does not wait for the GPU — it queues the kernels and returns immediately. Without a synchronize, a naive wall-clock timer would measure only the *launch* time, not the *execution*. The profiler handles device timing correctly via CUDA events, but any manual timing around GPU code must synchronize, or the numbers are fiction. (This is the same async fact behind the `torch.cuda.synchronize()` calls in the benchmark script `code/scripts/bench_attention.py`.)

## 2.1 Run it yourself on a Colab T4

<div class="hw"><p><strong>Hardware:</strong> free Colab T4 (16 GB). Runtime → Change runtime type → T4 GPU. Runs in about 1 minute, uses under 2 GB of GPU memory. Does not run on CPU (the point is to see GPU kernels).</p></div>

Paste this into one Colab cell. It builds a small 4-layer transformer (~21M parameters) on random tokens, so it needs no data and no course code. It runs three experiments:

1. **Healthy step** — batch 16 × 256 tokens, fp32. Expect the GPU busy almost the whole time and matmuls on top.
2. **Starving GPU** — the same model with batch 1 × 16 tokens. Every kernel is tiny, so the GPU finishes each one before the CPU can launch the next: you will see the GPU busy only a fraction of the wall time. This is the CPU↔GPU gap from section 3.
3. **Mixed precision** — experiment 1 again under fp16 `autocast` + `GradScaler` (sections 6–7). The T4 has fp16 Tensor Cores but no bf16 hardware, so fp16 is the right choice on this card.

```python
import time, torch, torch.nn as nn, torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"
dev = "cuda"
print(torch.cuda.get_device_name(0), "| torch", torch.__version__)
VOCAB = 8000

class TinyLM(nn.Module):
    def __init__(self, d=512, layers=4, heads=8):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, x, y):
        logits = self.head(self.blocks(self.emb(x)))
        return logits, F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))

def make(batch, seq):
    torch.manual_seed(0)
    model = TinyLM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, VOCAB, (batch, seq), device=dev)   # random token ids
    y = torch.randint(0, VOCAB, (batch, seq), device=dev)   # random targets
    return model, opt, x, y

def step(model, opt, x, y, scaler=None):
    with torch.autocast("cuda", dtype=torch.float16, enabled=scaler is not None):
        _, loss = model(x, y)
    if scaler is None:
        loss.backward(); opt.step()
    else:
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    opt.zero_grad(set_to_none=True)

def gpu_us(e):  # self GPU time of one profiler row, in microseconds (name changed in torch 2.4)
    return e.self_device_time_total if hasattr(e, "self_device_time_total") else e.self_cuda_time_total

def kind(name):
    n = name.lower()
    if "gemm" in n or "xmma" in n: return "matmul"
    if "fmha" in n or "flash" in n or "attention" in n: return "attention"
    return "other"

def run(title, batch, seq, fp16=False, steps=5, top=8):
    model, opt, x, y = make(batch, seq)
    scaler = torch.amp.GradScaler("cuda") if fp16 else None
    for _ in range(3): step(model, opt, x, y, scaler)        # warm-up: cuBLAS picks kernels here
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for _ in range(steps):
            with record_function("train_step"):
                step(model, opt, x, y, scaler)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1e3

    # keep only real GPU kernels (op / autograd / annotation rows would double-count their kernels' time)
    rows = [e for e in prof.key_averages()
            if gpu_us(e) > 0
            and not e.key.startswith(("aten::", "train_step", "autograd::", "Optimizer.", "torch::"))
            and not e.key.endswith(("Backward0", "Backward1"))]
    total = sum(gpu_us(e) for e in rows)
    by_kind = {k: sum(gpu_us(e) for e in rows if kind(e.key) == k) for k in ("matmul", "attention", "other")}

    print(f"\n=== {title} ===")
    print(f"wall {wall_ms/steps:.1f} ms/step | GPU busy {total/1e3/steps:.1f} ms/step "
          f"= {100*total/1e3/wall_ms:.0f}% of wall time")
    print("GPU time split: " + " | ".join(f"{k} {100*v/total:.0f}%" for k, v in by_kind.items()))
    print(f"{'ms/step':>8} {'share':>6} {'calls/step':>10}  kernel")
    for e in sorted(rows, key=gpu_us, reverse=True)[:top]:
        print(f"{gpu_us(e)/1e3/steps:8.2f} {100*gpu_us(e)/total:5.1f}% {e.count//steps:10d}  {e.key[:70]}")
    return prof

def timed(batch, seq, fp16=False, steps=20):  # clean timing, no profiler overhead
    model, opt, x, y = make(batch, seq)
    scaler = torch.amp.GradScaler("cuda") if fp16 else None
    for _ in range(3): step(model, opt, x, y, scaler)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(steps): step(model, opt, x, y, scaler)
    torch.cuda.synchronize()                                  # without this you time only the launches
    return (time.perf_counter() - t0) * 1e3 / steps

prof = run("1) healthy: batch 16 x 256 tokens, fp32", 16, 256)
prof.export_chrome_trace("trace.json")                       # open at https://ui.perfetto.dev
run("2) starving: batch 1 x 16 tokens, fp32", 1, 16, top=4)
run("3) mixed precision: batch 16 x 256, fp16 autocast", 16, 256, fp16=True, top=4)

t32, t16 = timed(16, 256), timed(16, 256, fp16=True)
print(f"\nclean timing: fp32 {t32:.1f} ms/step | fp16 autocast {t16:.1f} ms/step | speedup {t32/t16:.1f}x")
# to see the full table of section 2:
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
# to view the timeline: from google.colab import files; files.download("trace.json")
```

Illustrative output (a T4; your numbers will differ by ±20%, and kernel names change with the CUDA/cuBLAS version — newer versions may show `sm75_xmma_gemm_...` or `cutlass_...` instead of `volta_sgemm_...`):

```text
Tesla T4 | torch 2.8.0+cu126

=== 1) healthy: batch 16 x 256 tokens, fp32 ===
wall 122.6 ms/step | GPU busy 118.1 ms/step = 96% of wall time
GPU time split: matmul 66% | attention 7% | other 27%
 ms/step  share calls/step  kernel
   31.41  26.6%         17  volta_sgemm_128x64_nt
   27.90  23.6%         17  volta_sgemm_128x64_nn
   19.20  16.3%         17  volta_sgemm_128x128_tn
    6.10   5.2%          4  fmha_cutlassB_f32_aligned_64x64_k64_sm75
    3.09   2.6%          4  void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUD
    2.60   2.2%         12  void at::native::(anonymous namespace)::fused_dropout_kernel_vec<floa
    2.41   2.0%          4  fmha_cutlassF_f32_aligned_64x64_rf_sm75
    2.20   1.9%          9  void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<

=== 2) starving: batch 1 x 16 tokens, fp32 ===
wall 8.3 ms/step | GPU busy 1.7 ms/step = 20% of wall time
GPU time split: matmul 41% | attention 6% | other 53%
 ms/step  share calls/step  kernel
    0.29  17.1%         17  volta_sgemm_32x32_sliced1x4_tn
    0.21  12.4%         17  volta_sgemm_32x32_sliced1x4_nt
    0.18  10.6%          6  void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at:
    0.09   5.3%         34  void at::native::vectorized_elementwise_kernel<4, at::native::FillFun

=== 3) mixed precision: batch 16 x 256, fp16 autocast ===
wall 46.9 ms/step | GPU busy 44.2 ms/step = 94% of wall time
GPU time split: matmul 47% | attention 9% | other 44%
 ms/step  share calls/step  kernel
    7.62  17.2%         17  turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn
    6.94  15.7%         17  turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn
    5.31  12.0%         17  turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_nt
    3.12   7.1%          4  fmha_cutlassB_f16_aligned_64x64_k64_sm75

clean timing: fp32 121.4 ms/step | fp16 autocast 44.8 ms/step | speedup 2.7x
```

What to notice:

- **Experiment 1** is the healthy picture: GPU busy 96% of the time, and the three `sgemm` (fp32 matmul) kernels are two thirds of it. `calls/step = 17` is the 17 `nn.Linear` layers (4 per block × 4 blocks + the output head), each with one forward and two backward matmuls — that's why there are three different gemm kernels.
- **Experiment 2** is the starving GPU: 8.3 ms of wall time per step but only 1.7 ms of actual GPU work, so the GPU sits idle 80% of the time waiting for Python to launch the next kernel. Making the kernels faster would change nothing; the fix is fewer, bigger kernels (bigger batch, fusion, `torch.compile`, CUDA graphs). Open `trace.json` in Perfetto for this run and you will literally see the gaps on the GPU row. (The profiler itself adds some CPU overhead, which makes this case look a bit worse than reality — but the gap is real without it too.)
- **Experiment 3** is mixed precision: the gemm kernels switch from `sgemm` (fp32 CUDA cores) to `fp16_s1688gemm` (fp16 Tensor Cores; `1688` is the 16×8×8 Tensor-Core instruction shape) and drop from ~78 ms to ~20 ms per step. The non-matmul share rises from 27% to 44% — not because those kernels got slower but because the matmuls got so much faster. That is the roofline lesson of 08.2 in one number: after you speed up the compute-bound part, the memory-bound tail becomes the next thing to fix.

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

### Two different mechanisms: fewer bytes vs a higher ceiling

The first and third bullets sound like one benefit ("16-bit is faster") but they act on *different terms* of the roofline from 08.2, and each one only helps one kind of kernel. Recall a kernel's runtime is set by whichever limit it hits first:

$$
t \approx \max\!\left(\frac{\text{FLOPs}}{P_{\max}},\ \frac{\text{bytes}}{B}\right)
$$

where $\text{FLOPs}$ is the arithmetic the kernel does, $P_{\max}$ the peak FLOP/s of the arithmetic unit it runs on, $\text{bytes}$ the HBM traffic, and $B$ the HBM bandwidth. Going to 16-bit can change two things in this formula:

- **Halving bytes** shrinks the *second* term. It only helps if that term is the larger one — a memory-bound kernel.
- **Switching to the 16-bit Tensor-Core path** raises $P_{\max}$ and shrinks the *first* term. It only helps if that term is the larger one — a compute-bound kernel.

Work both cases by hand on an A100 ($B = 2.039\times10^{12}$ bytes/s; $P_{\max}$ = 19.5 TFLOP/s for plain fp32, 312 TFLOP/s for bf16 Tensor Cores).

**Case 1: a compute-bound matmul**, $n = 4096$ (the AI ≈ 683 dot from 08.2).

- FLOPs $= 2n^3 = 2 \cdot 4096^3 \approx 1.37\times10^{11}$.
- fp32 bytes $= 3n^2 \cdot 4 \approx 2.01\times10^{8}$ (read A, read B, write C).
- fp32: compute term $= 1.37\times10^{11} / 19.5\times10^{12} \approx 7.05$ ms; memory term $= 2.01\times10^{8} / 2.039\times10^{12} \approx 0.099$ ms. Runtime $\approx \max(7.05, 0.099) = 7.05$ ms. Compute is 70× larger than memory.
- **Halve the bytes only** (store in 16-bit but still do fp32 arithmetic): memory term $0.099 \to 0.049$ ms. Runtime $= \max(7.05, 0.049) = 7.05$ ms. **Zero speedup.** You shrank a term that was not the bottleneck.
- **Use the bf16 Tensor-Core path**: compute term $= 1.37\times10^{11} / 312\times10^{12} \approx 0.44$ ms. Runtime $= \max(0.44, 0.049) \approx 0.44$ ms — **16× faster**, and all of it came from the higher ceiling, none from the halved bytes.

**Case 2: a memory-bound elementwise add**, $n = 10^8$ elements (AI ≈ 0.083).

- FLOPs $= 10^8$; fp32 bytes $= 12 \cdot 10^8 = 1.2\times10^{9}$ (read two inputs, write one, 4 bytes each).
- fp32: compute term $= 10^8 / 19.5\times10^{12} \approx 0.005$ ms; memory term $= 1.2\times10^9 / 2.039\times10^{12} \approx 0.59$ ms. Runtime $\approx 0.59$ ms. Memory is >100× larger than compute.
- **Halve the bytes** (bf16, 6 bytes per element): memory term $\to 0.29$ ms. Runtime $\approx 0.29$ ms — **2× faster**, entirely from the bytes.
- **A faster arithmetic unit** does nothing: the compute term was already ~0.005 ms. (Elementwise ops don't even run on Tensor Cores — there is no matmul for them to accelerate.)

On the roofline picture from 08.2: halving bytes **doubles AI**, moving the dot *right*. For the elementwise add (0.083 → 0.167) that walks it up the slanted bandwidth line, so it gets 2× higher. For the $n = 4096$ matmul (683 → 1365) it slides right along the *flat* roof — no higher at all. The Tensor-Core path does something else entirely: it **raises the flat roof itself** (19.5 → 312 TFLOP/s), which lifts every compute-bound dot but leaves memory-bound dots, still under the slanted line, exactly where they were.

One subtlety: the ridge itself moves with the dtype, because $P_{\max}$ changes. For plain fp32 the A100 ridge is $19.5\times10^{12} / 2.039\times10^{12} \approx 9.6$ FLOPs/byte; for bf16 Tensor Cores it is ≈ 153. So a small matmul can be compute-bound in fp32 and memory-bound in bf16, and then it needs *both* wins. Take $n = 256$: FLOPs $= 2\cdot256^3 \approx 3.4\times10^{7}$.

- fp32: AI $= n/6 \approx 43$ (> 9.6, compute-bound). Compute $\approx 1.72\ \mu$s, memory $= 786{,}432 / 2.039\times10^{12} \approx 0.39\ \mu$s. Runtime ≈ 1.72 µs.
- bf16: AI $= n/3 \approx 85$ (< 153, now memory-bound). Compute $\approx 0.11\ \mu$s, memory $\approx 0.19\ \mu$s. Runtime ≈ 0.19 µs.

The Tensor Cores collapsed the compute term so far that the bytes became the limit — and the halved bytes are what keep that new limit at 0.19 µs instead of 0.39 µs. In practice this is why many small, skinny matmuls in a model (small heads, small batch at inference) are bandwidth-limited once they run in bf16.

<div class="callout key"><p>"16-bit is faster" is two separate wins. <strong>Halved bytes</strong> speed up memory-bound kernels (elementwise, LayerNorm, softmax) and do nothing for compute-bound matmuls. <strong>The 16-bit Tensor-Core path</strong> speeds up compute-bound matmuls and does nothing for memory-bound kernels. Before predicting what mixed precision will buy a kernel, place it on the roofline: left of the ridge, count bytes; right of the ridge, count FLOP/s.</p></div>

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
- **Assuming mixed precision fixes a compute-bound bottleneck by halving bytes.** For the matmuls the win is the faster 16-bit Tensor-Core path, not bandwidth; for the elementwise tail the win is the halved bytes. Different mechanisms — the profiler + roofline tell you which applies where. Worked through by hand in section 4: halving the bytes of an $n=4096$ matmul gives 0% speedup; the bf16 Tensor Cores give 16×.

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
