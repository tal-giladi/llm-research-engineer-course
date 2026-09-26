# 08.1 · GPU architecture & memory hierarchy

<div class="prereq">
<p><strong>Prerequisites:</strong> the memory buckets and throughput reasoning from <a href="#/lessons/module-07/lesson-04">07.4 · Throughput, FLOPs, MFU, memory accounting</a>; the idea that the logits and activation tensors dominate memory (also 07.4); tensor shape/dtype/device intuition from <a href="#/lessons/module-01/lesson-01">Module 1</a>.</p>
<p><strong>You will learn:</strong> what a GPU actually is — streaming multiprocessors (SMs), threads grouped into <strong>warps</strong>, and the memory hierarchy that runs from tiny-but-instant registers, through on-chip <strong>SRAM / shared memory</strong>, an <strong>L2</strong> cache, out to large-but-slow <strong>HBM</strong>, and finally host RAM across the PCIe/NVLink bus. You will get rough, representative bandwidth and size numbers for an A100/H100-class GPU, and the one fact that governs everything in this module: <strong>arithmetic is cheap, moving data is expensive</strong>, so most LLM kernels are <em>memory-bound</em>.</p>
<p><strong>Why this matters for ML:</strong> in 07.4 you learned to count FLOPs and to fear the activation memory bucket. But FLOPs alone do not predict runtime — a kernel that does few FLOPs can still be slow if it drags a lot of data across the chip. Understanding where data lives and how fast it moves is what lets you read a profiler trace (08.3), reason about the roofline (08.2), and understand <em>why</em> FlashAttention (08.4) is faster without doing less math. This is the mental model underneath every performance decision in the rest of the course.</p>
</div>

## 1. Intuition: a GPU is a very wide, very shallow factory

A CPU is a small team of extremely capable workers. It has a handful of cores (say 8–64), each with deep out-of-order execution, big caches, and clever branch prediction, tuned to finish one complicated sequence of instructions as fast as possible. That is the right design for code full of branches and pointer-chasing — an operating system, a web server, a compiler.

A GPU is the opposite bet. It has *thousands* of much simpler arithmetic lanes, and it wins by doing the *same* operation on a huge number of data elements at once. Adding two vectors of a million floats, multiplying two large matrices, applying GELU to every element of an activation tensor — these are jobs where there is no branching and no data dependency between elements, so you can throw thousands of lanes at them in parallel. Deep learning is almost entirely made of exactly this kind of work, which is why GPUs run it 10–100× faster than CPUs.

The catch — and the theme of this whole module — is that feeding those thousands of lanes with data is *hard*. The arithmetic units can consume numbers far faster than memory can deliver them. So the performance story of a GPU is rarely "do I have enough compute?" It is almost always "can I get the data to the compute units fast enough?"

<div class="callout key"><p>A GPU trades single-thread cleverness for raw parallel width: thousands of simple lanes doing the same operation at once. That makes arithmetic abundant and cheap — and turns <em>data movement</em> into the scarce resource. Most of this module is about not moving data you do not have to.</p></div>

## 2. The execution hierarchy: threads, warps, blocks, SMs

To use thousands of lanes you need a way to organize them. NVIDIA's model (CUDA) has a small hierarchy of its own. You do not need to write CUDA to train models, but you need this vocabulary to read a profiler and to understand FlashAttention.

- **Thread.** The smallest unit of work: one lane executing the kernel's code on (typically) one data element. A kernel launch might create millions of threads.
- **Warp.** Threads are executed in lockstep groups of **32**, called a warp. All 32 threads in a warp run the *same* instruction at the same time, each on its own data — this is called SIMT (single instruction, multiple threads). The warp is the real unit of scheduling: the hardware issues one instruction per warp, not per thread.
- **Thread block (CTA).** A group of warps (up to 1024 threads) that are guaranteed to run on the *same* SM and can cooperate — they share a fast on-chip scratchpad (**shared memory**, section 4) and can synchronize with a barrier. This cooperation is the key that FlashAttention exploits: a block loads a tile of data into shared memory once and all its threads reuse it.
- **Streaming Multiprocessor (SM).** The physical processor. An A100 has 108 SMs; an H100 has 132. Each SM has its own register file, its own shared memory, warp schedulers, and a set of arithmetic units including the **Tensor Cores** — dedicated matrix-multiply-accumulate units that do the bulk of a transformer's FLOPs. The GPU runs many thread blocks across all its SMs at once.

<div class="callout key"><p>The chain is: <strong>thread</strong> → <strong>warp</strong> (32 threads in lockstep) → <strong>thread block</strong> (warps that share on-chip memory on one SM) → <strong>SM</strong> (the physical core; ~108–132 of them) → <strong>GPU</strong>. Tensor Cores inside each SM do the matmul FLOPs. A "kernel" is one function launched across this whole grid.</p></div>

### Two consequences worth remembering

**Warps hide latency.** When one warp stalls waiting for data from memory, the SM's scheduler instantly switches to another ready warp and keeps the arithmetic units busy. This is why GPUs want *many* warps in flight (high "occupancy"): not to compute faster per warp, but to always have some warp that is not waiting. If you give a GPU too little parallel work — a tiny batch, a short sequence — there are not enough warps to hide memory latency, the arithmetic units sit idle, and your MFU (07.4) drops. That is one concrete cause of the "8% MFU" mystery from the last lesson.

**Branch divergence is expensive.** Because a warp runs one instruction across 32 threads, an `if` where some threads go one way and some the other forces the warp to execute *both* paths, masking off the inactive threads each time. Data-parallel numeric code avoids this — which is exactly why deep learning maps so well onto the hardware.

## 3. The memory hierarchy: the central object of this module

Now the part that matters most. Memory on a GPU is not one thing; it is a hierarchy of levels that trade **size against speed**. Small memories are fast and close to the arithmetic units; large memories are slow and far away. Every level down is roughly 10× bigger and several times slower.

From closest/fastest to farthest/slowest:

1. **Registers.** Per-thread private storage, physically inside the SM. Instant (0-cycle) access — an operand in a register is ready the moment the arithmetic unit wants it. But there are very few: an SM has a register file of ~256 KB *total*, split across all its live threads, so each thread gets only a few dozen registers. Values a kernel is actively computing on live here.
2. **Shared memory / SRAM.** A small scratchpad *per SM* (configurable, ~up to 164 KB per SM on A100, ~228 KB on H100), shared by all threads in a block. On-chip, so very fast (a few hundred GB/s *per SM*, and there are ~100+ SMs, so aggregate on-chip bandwidth is enormous — on the order of tens of TB/s). This is the memory FlashAttention lives in: it stages tiles of Q, K, V here so it can do lots of arithmetic on them without going back out to HBM.
3. **L2 cache.** A single cache shared by *all* SMs, ~40 MB on A100, ~50 MB on H100. It automatically caches recently-used HBM data. Faster than HBM, slower than SRAM.
4. **HBM (High-Bandwidth Memory).** The big "GPU memory" you quote when you say "an 80 GB A100." It is off-chip DRAM stacked next to the GPU die. Large (40–80 GB on A100, 80–141 GB on H100) but *slow relative to on-chip*: ~1.5–2 TB/s on A100, ~3.3 TB/s on H100. This is where your model weights, gradients, optimizer state, and activations (07.4) actually live. Every time a kernel reads a tensor "from GPU memory," it is reading from HBM. **HBM traffic is the thing FlashAttention minimizes.**
5. **Host RAM.** The CPU's ordinary DRAM, reached across the PCIe bus (~32–64 GB/s) or NVLink to other GPUs. Huge (hundreds of GB) but an order of magnitude slower than HBM. Moving a tensor from host to device (`.to("cuda")`) or between GPUs crosses this bottleneck; you want to do it rarely.

<div class="callout key"><p>Registers → SRAM/shared memory → L2 → HBM → host RAM. Each step down is bigger and slower. The two you will name constantly: <strong>SRAM</strong> (tiny, on-chip, blazing fast, where a kernel does its scratch work) and <strong>HBM</strong> (large, off-chip, the "GPU memory" your tensors live in, and the level whose bandwidth is usually the bottleneck).</p></div>

### Representative numbers (approximate — orders of magnitude, not spec sheets)

These are ballpark, publicly-documented figures for A100/H100-class hardware. Treat them as representative orders of magnitude; exact values vary by SKU (40 GB vs 80 GB A100, SXM vs PCIe, H100 vs H200) and by whether a vendor quotes peak or sustained.

| Level | Typical size | Typical bandwidth | Relative to HBM |
|---|---|---|---|
| Registers | ~256 KB / SM | effectively instant | ~thousands× |
| Shared mem / SRAM | ~164–228 KB / SM | ~10s of TB/s aggregate | ~10–20× |
| L2 cache | ~40–50 MB (whole GPU) | several TB/s | ~2–5× |
| HBM | 40–141 GB | ~1.5–3.3 TB/s | 1× (baseline) |
| Host RAM (over PCIe) | 100s of GB | ~32–64 GB/s | ~0.02–0.03× |

The two numbers to burn in: on an A100, HBM is roughly **2 TB/s** and the chip can do roughly **312 TFLOP/s** in bf16 (07.4). Hold those two side by side — they are the whole story of the next section.

## 4. The central fact: arithmetic is cheap, data movement is expensive

Put the two A100 numbers next to each other:

- **Compute:** ~$312 \times 10^{12}$ FLOP/s.
- **HBM bandwidth:** ~$2 \times 10^{12}$ bytes/s.

Divide them. The chip can perform about

$$
\frac{312 \times 10^{12} \text{ FLOP/s}}{2 \times 10^{12} \text{ byte/s}} \approx 156 \text{ FLOPs for every byte it reads from HBM.}
$$

Read that carefully. For each *single byte* the GPU pulls from HBM, it has time to do about **150 floating-point operations**. So if a kernel reads a byte and then does only one or two FLOPs with it before needing the next byte, the arithmetic units spend ~99% of their time idle, waiting for memory. The kernel is not limited by how fast the GPU can compute; it is limited by how fast HBM can deliver bytes. We call such a kernel **memory-bound** (or *bandwidth-bound*).

The opposite case — a kernel that does hundreds of FLOPs per byte, like a large matrix multiply that reuses each loaded value across a whole row of output — is **compute-bound**: it keeps the arithmetic units saturated and memory is not the bottleneck.

<div class="callout key"><p>An A100 can do ~150 FLOPs in the time it takes to read one byte from HBM. A kernel that does fewer FLOPs-per-byte than that is <strong>memory-bound</strong> — it wastes the arithmetic units waiting on data. Most of the operations in a transformer that are <em>not</em> the big matmuls (elementwise activations, LayerNorm, dropout, adding bias, softmax, the attention score matrix) are memory-bound. That is where the performance is lost, and where fusion (08.2) and FlashAttention (08.4) win it back.</p></div>

### Why LLM kernels are so often memory-bound

The big weight matmuls in a transformer — the QKV projection, the output projection, the two MLP layers — are genuinely compute-bound; they reuse each loaded number many times and keep the Tensor Cores busy. That is the $6N$ FLOPs of 07.4, and it is *good*.

But a transformer forward pass is not only those matmuls. In between and around them sit a long tail of *elementwise* and *reduction* operations:

- adding the bias vector after a Linear,
- the GELU/SwiGLU activation,
- dropout,
- LayerNorm / RMSNorm (a reduction plus a scale),
- the residual add,
- and, crucially, the attention **softmax** and the reading/writing of the $T \times T$ score matrix.

Each of these reads a big activation tensor from HBM, does a tiny amount of arithmetic per element, and writes it back to HBM. Their arithmetic intensity is near the floor. Individually they look cheap in FLOPs, but summed up they can eat a large fraction of the wall-clock time precisely because they are bandwidth-bound and each one makes a full round trip to HBM. This is the concrete, physical reason the next three lessons exist: **08.2** gives you the roofline to quantify "memory-bound vs compute-bound," fusion to cut the round trips; **08.3** gives you the profiler to *find* these kernels and mixed precision to halve the bytes; **08.4** rebuilds attention itself to stop touching HBM for the score matrix.

## 5. Tensor shapes, dtype, device — where a tensor physically lives

Everything you learned about tensor shape/dtype/device (Module 1) has a physical meaning on a GPU, and this is the lesson to make it concrete.

- **`device`** says *which memory* the tensor's bytes live in. `x.device == cpu` means the bytes are in host RAM; `x.device == cuda:0` means they are in the HBM of GPU 0. `x.to("cuda")` physically copies the bytes across the PCIe bus into HBM — a slow transfer (section 3), which is why you move data to the GPU once and keep it there.
- **`dtype`** sets *how many bytes each element occupies*, and therefore how much HBM the tensor uses and how many bytes must move to touch it. An fp32 element is 4 bytes; bf16/fp16 are 2 bytes (08.3). Halving the dtype halves both the storage and the bandwidth cost of every op that reads or writes the tensor — which is the entire performance argument for mixed precision.
- **`shape`** sets the *total* byte count: `numel() × element_size()`. A GPT-2-small logits tensor $(B, T, V) = (8, 1024, 50257)$ in fp32 is $8 \cdot 1024 \cdot 50257 \cdot 4 \approx 1.65$ GB (07.4) — 1.65 GB that must be written to HBM by the final matmul and read back by the softmax/cross-entropy. Its size, and hence its HBM cost, is why it dominated the activation bucket.

<div class="callout pt"><p>When you write <code>y = torch.nn.functional.gelu(x)</code> for an activation tensor <code>x</code> of shape <code>(B, T, 4C)</code> living in HBM, PyTorch launches a CUDA kernel that: reads every element of <code>x</code> from HBM into registers, computes GELU (a few FLOPs each), and writes every element of <code>y</code> back to HBM. Bytes moved: read <code>x</code> + write <code>y</code>. FLOPs: a handful per element. Arithmetic intensity is tiny — this is a textbook memory-bound kernel, and it is exactly the kind we will want to <em>fuse</em> with its neighbors in 08.2 so the intermediate never hits HBM.</p></div>

## 6. Under the hood: what happens when you call one PyTorch op on the GPU

Let us trace a single line to make the hierarchy concrete. Suppose `a` and `b` are fp16 tensors of shape $(4096, 4096)$ on `cuda`, and you run `c = a + b`.

1. Python calls into PyTorch's C++ dispatcher, which selects the CUDA kernel for elementwise add on fp16.
2. PyTorch **launches** that kernel: it tells the driver to create a grid of thread blocks — here about $4096 \cdot 4096 / (\text{threads per block})$ threads, one region of the tensor per thread. There is a fixed launch overhead (a few microseconds) *per* kernel; this is why launching thousands of tiny kernels is slow and why fusion helps.
3. The GPU schedules those blocks across its ~108 SMs, 32 threads per warp.
4. Each thread reads its element of `a` and its element of `b` **from HBM** into registers, adds them (one FLOP), and writes the result **to HBM**.
5. Bytes moved: $2 \times$ read + $1 \times$ write $= 3 \times 4096^2 \times 2$ bytes $\approx 100$ MB. FLOPs: $4096^2 \approx 1.7 \times 10^7$. Arithmetic intensity $\approx \tfrac{1.7\times10^7}{1\times10^8} \approx 0.17$ FLOPs/byte — hopelessly below the ~150 ridge. **Memory-bound.** The kernel's runtime is set almost entirely by HBM bandwidth: $\approx 100 \text{ MB} / 2 \text{ TB/s} \approx 50\ \mu s$, and the arithmetic is essentially free hiding behind that.

That single trace contains the whole module in miniature: a kernel launch, threads reading from HBM, doing trivial arithmetic, writing back to HBM, and being bottlenecked on bandwidth rather than compute. Keep it in mind as we now formalize "memory-bound" with the roofline model.

## Check yourself

<details><summary>Order these from fastest/smallest to slowest/largest, and give a one-line role for each: HBM, registers, shared memory (SRAM), host RAM, L2.</summary>

**Registers** (per-thread, instant, holds the values being computed on) → **shared memory / SRAM** (per-SM scratchpad, on-chip, where a block stages a reused tile — FlashAttention's home) → **L2** (one cache for the whole GPU) → **HBM** (the large off-chip "GPU memory" where your weights/activations live; ~2 TB/s on A100) → **host RAM** (CPU memory across PCIe, huge but ~30–60 GB/s). Each step down is bigger and slower.

</details>

<details><summary>An A100 does ~312 TFLOP/s (bf16) and has ~2 TB/s HBM bandwidth. Roughly how many FLOPs can it do per byte read from HBM, and what does that number mean?</summary>

$312\times10^{12} / 2\times10^{12} \approx 156$ FLOPs per byte. It means a kernel must do at least ~150 FLOPs for every byte it reads from HBM to keep the arithmetic units busy. Below that, the kernel is **memory-bound**: the compute units idle waiting for data, and runtime is set by bandwidth, not by peak FLOP/s.

</details>

<details><summary>Why does a GPU want thousands of threads (many warps) in flight, even though it can only physically run a limited number at once?</summary>

To hide memory latency. When a warp stalls waiting on a slow HBM read, the SM scheduler switches to another ready warp and keeps the arithmetic units working. With too few warps (tiny batch / short sequence) there is nothing to switch to, the units idle during every memory stall, and utilization (MFU) collapses. Extra parallelism is latency-hiding, not just raw throughput.

</details>

<details><summary>A GELU applied to a $(B, T, 4C)$ activation tensor does only a few FLOPs per element. Why is it still a meaningful cost on a GPU, and what is the fix previewed for 08.2?</summary>

It is memory-bound: it must read the whole tensor from HBM and write the whole result back, and a big activation tensor is many megabytes, so the round trip to HBM dominates its runtime regardless of how little arithmetic it does. The fix is **kernel fusion** (08.2): fuse GELU with the neighboring bias-add and dropout into one kernel so the intermediate tensors never leave the chip — one HBM read and one HBM write instead of three round trips.

</details>

## Next

You now have the machine: SMs and warps for compute, and a memory hierarchy whose bottleneck is HBM bandwidth. You also have the one governing fact — arithmetic is far cheaper than data movement, so most non-matmul kernels are memory-bound. The next lesson turns "memory-bound vs compute-bound" from a slogan into a quantitative tool: **arithmetic intensity** and the **roofline model**, and the first optimization that falls out of them — **kernel fusion**.

Continue to [08.2 · Arithmetic intensity, roofline, fusion](lessons/module-08/lesson-02.md).
