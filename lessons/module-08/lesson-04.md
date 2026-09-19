# 08.4 · FlashAttention & a Triton kernel; benchmark

<div class="prereq">
<p><strong>Prerequisites:</strong> scaled dot-product attention and the $T\times T$ score matrix from <a href="../module-05/lesson-02.md">05.2 · Q/K/V & scaled dot-product attention</a> (the exact formula $\mathrm{softmax}(QK^\top/\sqrt{d}+M)V$); arithmetic intensity, the roofline, and kernel fusion from <a href="lesson-02.md">08.2</a>; the SRAM-vs-HBM memory hierarchy from <a href="lesson-01.md">08.1</a>; softmax numerical stability (the max-subtraction trick) from the probability module.</p>
<p><strong>You will learn:</strong> why standard attention is <strong>memory-bound</strong> — it materializes the $T\times T$ score matrix in HBM, costing $O(T^2)$ memory and bandwidth — and how <strong>FlashAttention</strong> (Dao et al. 2022) computes the <em>exact same</em> output with $O(T)$ memory by tiling $Q,K,V$ into blocks that fit in SRAM and using the <strong>online-softmax</strong> recurrence (a running max $m$ and running denominator $\ell$) to accumulate the output block by block. You will work the online-softmax recurrence by hand on tiny numbers, see an illustrative <strong>Triton</strong> kernel skeleton, and run a benchmark comparing naive attention, <code>F.scaled_dot_product_attention</code>, and the FlashAttention algorithm. You implement the algorithm from scratch in <code>code/src/llmre/attention/flash.py</code>.</p>
<p><strong>Why this matters for ML:</strong> attention's $O(T^2)$ memory was the wall that capped context length. FlashAttention broke it without changing the math or the result — it is the reason models went from 1–2k context to 128k+ and it is standard in every serious training and inference stack. It is also the perfect capstone for this module: it is nothing but 08.1's memory hierarchy + 08.2's roofline and fusion applied to the one operation that needed it most.</p>
</div>

## 1. The problem: standard attention is memory-bound

Recall exact attention from [05.2](../module-05/lesson-02.md), for one head, query/key/value matrices of shape $(T, d)$:

$$
S = \frac{QK^\top}{\sqrt{d}} \in \mathbb{R}^{T\times T}, \qquad P = \mathrm{softmax}(S) \in \mathbb{R}^{T\times T}, \qquad O = P V \in \mathbb{R}^{T\times d}.
$$

The naive implementation — the one you wrote in Module 5 — does exactly this, step by step, and each intermediate lives in **HBM**:

1. compute $S = QK^\top/\sqrt d$ and **write** the $T\times T$ matrix to HBM;
2. **read** $S$ back, apply softmax, **write** the $T\times T$ matrix $P$ to HBM;
3. **read** $P$ back, multiply by $V$, write the $T\times d$ output.

Count the HBM traffic. The two big matrices $S$ and $P$ are each $T^2$ elements. For $T = 8192$ and one head in fp16, one $T\times T$ matrix is $8192^2 \cdot 2 \approx 134$ MB — and there are several reads/writes of it, per head, per layer. The **memory** is $O(T^2)$ and the **bandwidth** is $O(T^2)$, while the useful arithmetic is also $O(T^2 d)$ but with tiny reuse in the softmax step. The softmax itself is a textbook low-arithmetic-intensity, memory-bound reduction (08.2): it reads a huge matrix, does a few FLOPs per element, writes it back.

So for long sequences, attention is dominated not by its FLOPs but by shuttling the $T\times T$ score matrix in and out of HBM. On the roofline it sits far left, on the bandwidth slope. And crucially, this $O(T^2)$ **memory** is also what makes long context run out of GPU memory — the score matrix is an activation that must be kept for the backward pass.

<div class="callout key"><p>The bottleneck in attention is not the matmuls — it is <strong>materializing the $T\times T$ score/probability matrix in HBM</strong>. That is $O(T^2)$ memory and $O(T^2)$ bandwidth, and the softmax over it is a memory-bound reduction. Halving the dtype (08.3) helps a constant factor but does not change the $O(T^2)$ scaling. To break the wall you must never write the full matrix at all.</p></div>

## 2. The idea: tile into SRAM, never store the full matrix

FlashAttention's insight is that you do not need the whole matrix $P$ at once to compute $O = PV$. The output row for query $i$ is $O_i = \sum_j P_{ij} V_j$ — a weighted sum over keys. You can accumulate that sum **incrementally**, processing the keys in *blocks*: bring a block of keys/values into fast SRAM, compute the partial scores for that block, fold their contribution into a running output, discard the block, bring the next. At no point does the full $T\times T$ matrix exist in memory — only one small $B_q \times B_k$ tile at a time, in SRAM.

Concretely (this is exactly what `flash_attention_reference` does): cut $Q$ into query blocks and $K, V$ into key/value blocks of some `block_size`. For each query block, loop over the key/value blocks, and for each key block compute the tile scores $S^{(j)} = Q_{\text{blk}} K_{\text{blk}}^\top/\sqrt d$ (only $B_q\times B_k$, tiny), turn them into a partial weighted contribution, and add it to a running accumulator. The blocks are sized to fit in the SM's shared memory (08.1), so all the arithmetic on a tile happens on-chip and the only HBM traffic is reading each of $Q, K, V$ **once** and writing $O$ **once** — $O(T)$ (well, $O(Td)$) traffic, not $O(T^2)$.

The one subtlety: softmax normalizes over the *whole* row of keys, but we are only seeing one block of keys at a time. How do you softmax across blocks you have not seen yet? That is the online-softmax recurrence.

## 3. Online softmax: running max $m$ and running denominator $\ell$

Softmax of a row needs two whole-row quantities: the maximum (subtracted for numerical stability — otherwise $\exp$ of a large score overflows, cf. 08.3) and the sum of exponentials (the denominator). Online softmax computes both *incrementally* as blocks arrive, keeping just three running numbers per query row:

- $m$ — the running **max** of the scores seen so far (start at $-\infty$),
- $\ell$ — the running **denominator** $\sum \exp(\text{score} - m)$ over scores seen so far (start at $0$),
- $\mathbf{o}$ — the running **unnormalized output** $\sum \exp(\text{score} - m)\,\mathbf{v}$ (start at $\mathbf{0}$).

When a new block of scores $s_1,\dots,s_{B_k}$ (with values $\mathbf v_1,\dots,\mathbf v_{B_k}$) arrives:

1. block max $m_{\text{blk}} = \max_t s_t$; new running max $m_{\text{new}} = \max(m, m_{\text{blk}})$.
2. **correction factor** $\alpha = \exp(m - m_{\text{new}})$ — this rescales the *old* $\ell$ and $\mathbf o$, which were computed relative to the old max $m$, so they are now relative to $m_{\text{new}}$. (Since $m \le m_{\text{new}}$, $\alpha \le 1$; on the very first block $m=-\infty$ and $\alpha = 0$.)
3. exponentiate the new block against the new max: $p_t = \exp(s_t - m_{\text{new}})$.
4. update: $\ell \leftarrow \alpha\,\ell + \sum_t p_t$, and $\mathbf o \leftarrow \alpha\,\mathbf o + \sum_t p_t\,\mathbf v_t$, and $m \leftarrow m_{\text{new}}$.

After all blocks, the true attention output for that row is $\mathbf o / \ell$ — the accumulated numerator divided by the accumulated denominator, exactly the softmax-weighted average. The correction factor $\alpha$ is the whole trick: it lets you change the max midstream (whenever a later block has a bigger score) and retroactively fix the already-accumulated sum, so the final result is *identical* to having seen the whole row at once.

<div class="callout key"><p>Online softmax keeps three running numbers per query row — max $m$, denominator $\ell$, unnormalized output $\mathbf o$ — and folds in one key-block at a time. When a block raises the max, the <strong>correction factor</strong> $\alpha = \exp(m_{\text{old}} - m_{\text{new}})$ rescales the old $\ell$ and $\mathbf o$ to the new max. Divide $\mathbf o/\ell$ at the end. It is exact: same result as full-row softmax, computed with $O(1)$ state per row and never storing the full score matrix.</p></div>

## 4. Worked example (tiny numbers, by hand)

One query row attending to four keys, values scalar ($d = 1$) for clarity. After scaling, the four scores and their values are

$$
s = [\,1,\ 2,\ 3,\ 0\,], \qquad v = [\,10,\ 20,\ 30,\ 40\,].
$$

**Ground truth (full-row softmax).** Max is $3$; $\exp(s - 3) = [e^{-2}, e^{-1}, e^{0}, e^{-3}] = [0.1353, 0.3679, 1.0, 0.0498]$, sum $= 1.5530$. Weights $= [0.0871, 0.2369, 0.6439, 0.0321]$. Output $= 0.0871\cdot10 + 0.2369\cdot20 + 0.6439\cdot30 + 0.0321\cdot40 = 26.2089$.

Now the **online** computation in two blocks of two.

**Block 1:** scores $[1, 2]$, values $[10, 20]$. First block, so $m = -\infty$, $\ell = 0$, $\mathbf o = 0$.
- $m_{\text{blk}} = 2$, $m_{\text{new}} = \max(-\infty, 2) = 2$, correction $\alpha = 0$ (first block).
- $p = [\exp(1-2), \exp(2-2)] = [0.3679,\ 1.0]$.
- $\ell = 0\cdot 0 + (0.3679 + 1.0) = 1.3679$.
- $\mathbf o = 0\cdot 0 + (0.3679\cdot 10 + 1.0\cdot 20) = 23.6788$.
- running output so far $= \mathbf o/\ell = 23.6788/1.3679 = 17.3106$ (a *provisional* answer, using only the first two keys).

**Block 2:** scores $[3, 0]$, values $[30, 40]$. Now $m = 2$, $\ell = 1.3679$, $\mathbf o = 23.6788$.
- $m_{\text{blk}} = 3$, $m_{\text{new}} = \max(2, 3) = 3$ — the max just increased, so the correction kicks in.
- $\alpha = \exp(m - m_{\text{new}}) = \exp(2 - 3) = 0.3679$.
- $p = [\exp(3-3), \exp(0-3)] = [1.0,\ 0.0498]$.
- $\ell = 0.3679\cdot 1.3679 + (1.0 + 0.0498) = 0.5032 + 1.0498 = 1.5530$.
- $\mathbf o = 0.3679\cdot 23.6788 + (1.0\cdot 30 + 0.0498\cdot 40) = 8.7118 + 31.992 = 40.7024$.
- final output $= \mathbf o/\ell = 40.7024/1.5530 = 26.2089$.

That final $26.2089$ is **exactly** the full-row softmax answer. Watch what the correction factor did: after block 1 the running state was scaled to a max of $2$; block 2 found a bigger score ($3$), so $\alpha = e^{-1} = 0.3679$ shrank the old $\ell$ and $\mathbf o$ to the new reference max before adding block 2's contribution. Without $\alpha$ the two blocks would have been on incompatible scales and the sum would be wrong. Note also the denominator $\ell = 1.5530$ matches the ground-truth sum exactly. This is the entire FlashAttention numerics, on four numbers.

<div class="callout pt"><p>You can run this exact example against the from-scratch implementation. <code>flash_attention_reference</code> in <code>code/src/llmre/attention/flash.py</code> is this recurrence, vectorized over batch, heads, and query rows, with <code>block_size</code> as the tile length. The tests in <code>code/tests/test_flash.py</code> assert it matches both a plain <code>softmax(QKᵀ/√d)V</code> and <code>F.scaled_dot_product_attention</code> to 1e-4, causal and non-causal, and that the result is <em>independent of block size</em> — because tiling is an implementation detail, not a change to the math.</p></div>

## 5. The from-scratch implementation (CPU, exact)

Here is the core of `flash_attention_reference` — the algorithm above, in PyTorch, for shapes $(B, n_h, T, d)$. It is deliberately *not* a fused kernel; each block op is an ordinary PyTorch call, so it runs (slowly) on CPU and proves the numerics.

```python
def flash_attention_reference(q, k, v, causal=False, block_size=64):
    B, nh, T, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    out = torch.empty_like(q)

    for i0 in range(0, T, block_size):                      # loop over query blocks
        i1 = min(i0 + block_size, T)
        q_blk = q[:, :, i0:i1, :]
        Bq = i1 - i0
        # running stats, one per query row in this block
        m   = torch.full((B, nh, Bq, 1), float("-inf"))     # running max
        l   = torch.zeros((B, nh, Bq, 1))                   # running denominator
        acc = torch.zeros((B, nh, Bq, hd))                  # running unnormalized output

        for j0 in range(0, T, block_size):                  # loop over key/value blocks
            j1 = min(j0 + block_size, T)
            if causal and j0 > i1 - 1:                       # whole block is future -> skip
                break
            k_blk = k[:, :, j0:j1, :]; v_blk = v[:, :, j0:j1, :]
            s = (q_blk @ k_blk.transpose(-2, -1)) * scale    # (B,nh,Bq,Bk) tile — the ONLY T×T-ish object
            if causal:                                       # mask future keys inside the tile
                q_idx = torch.arange(i0, i1).view(Bq, 1)
                k_idx = torch.arange(j0, j1).view(1, j1 - j0)
                s = s.masked_fill(k_idx > q_idx, float("-inf"))
            m_blk = s.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, m_blk)
            corr  = torch.exp(m - m_new).nan_to_num_(0.0)    # alpha; exp(-inf)=0 on first block
            p     = torch.exp(s - m_new)
            l     = l * corr + p.sum(dim=-1, keepdim=True)
            acc   = acc * corr + p @ v_blk
            m     = m_new

        out[:, :, i0:i1, :] = acc / l                        # normalize once, at the end
    return out
```

Line for line this is section 3: `s` is one tile of scores (the only score object that ever exists, and it is $B_q\times B_k$, never $T\times T$); `m_new`/`corr` implement the running max and the correction factor $\alpha$; `l` and `acc` are $\ell$ and $\mathbf o$; `acc / l` is the final normalize. The causal branch masks only *within* a diagonal tile and `break`s out of whole future blocks — the tiling makes causal masking essentially free for the skipped blocks. On a real GPU, the two Python loops become the kernel's internal loops with the tiles held in SRAM; here they are Python, which is why it is a *reference*, not a fast kernel.

## 6. An illustrative Triton kernel (will not run here)

The real FlashAttention is a single fused GPU kernel. **Triton** is a Python-like language for writing GPU kernels that compile to fast code, and it is how many production attention and fusion kernels are written today (TorchInductor emits Triton). The skeleton below is *illustrative only* — it sketches the structure, not a runnable, correct kernel, and this CPU-only machine has no Triton or CUDA to run it.

```python
# ILLUSTRATIVE — not runnable on this CPU-only machine, and simplified.
import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(Q, K, V, O, scale,
                     T, BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, D: tl.constexpr):
    q_block = tl.program_id(0)                    # this kernel instance owns one query block
    # load this query block into SRAM (registers/shared memory) ONCE
    q = tl.load(Q + q_offsets(q_block, D))        # (BLOCK_Q, D)
    m = tl.full((BLOCK_Q,), -float("inf"), tl.float32)   # running max
    l = tl.zeros((BLOCK_Q,), tl.float32)                 # running denominator
    acc = tl.zeros((BLOCK_Q, D), tl.float32)             # running unnormalized output

    for k_block in range(0, T, BLOCK_K):          # stream key/value blocks through SRAM
        k = tl.load(K + k_offsets(k_block, D))    # (BLOCK_K, D)
        v = tl.load(V + v_offsets(k_block, D))    # (BLOCK_K, D)
        s = tl.dot(q, tl.trans(k)) * scale        # (BLOCK_Q, BLOCK_K) tile, in SRAM
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)                 # correction factor
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m = m_new

    tl.store(O + o_offsets(q_block, D), acc / l[:, None])   # write output ONCE
```

Notice it is the *same recurrence* as the CPU reference — `m`, `l`, `acc`, `alpha` — but now `tl.load`/`tl.store` are explicit HBM↔SRAM transfers and the whole thing is one kernel, so the tiles genuinely stay on-chip and the $T\times T$ matrix is never written to HBM. That is where the real speed and memory win come from. Writing correct, fast Triton (handling masks, backward, numerical edge cases, block-size tuning) is a specialty; the point here is that the *algorithm* you implemented on CPU is exactly the algorithm the kernel runs.

## 7. Benchmark

The script `code/scripts/bench_attention.py` compares three paths — naive full-matrix attention, `F.scaled_dot_product_attention` (PyTorch's fused kernel, which uses a FlashAttention backend on capable GPUs), and `flash_attention_reference` — checking they agree and timing them:

```bash
py code/scripts/bench_attention.py --T 512 --causal
py code/scripts/bench_attention.py --T 4096 --heads 12 --dim 64   # long context
```

On this CPU-only machine the script's real job is **correctness**: it asserts all three outputs agree to 1e-3 and prints CPU timings and the size of the $T\times T$ score tensor that naive attention allocates but flash never does. The `flash_attention_reference` path is *slower* on CPU (its block loop is Python, and CPU has no SRAM/HBM distinction to exploit) — that is expected and the script says so. The memory/speed win is a GPU HBM-traffic effect.

<div class="hw">
<p><strong>Hardware track — the benchmark's real numbers need a GPU.</strong></p>
<p><strong>Minimum:</strong> the correctness check and the online-softmax algorithm run on any CPU, instantly, no GPU — that is what the test and this machine cover. <strong>Recommended (to see the win):</strong> one A100/H100-class GPU with bf16 and a FlashAttention-capable PyTorch build. <strong>Expected GPU results</strong> (representative, from the FlashAttention paper and common reports — <em>not measured here</em>): at long context (e.g. $T=$ 4k–16k) FlashAttention / the SDPA flash backend is typically <strong>~2–4× faster</strong> than a naive PyTorch attention and, more importantly, uses <strong>linear ($O(T)$) instead of quadratic ($O(T^2)$) memory</strong> for the attention activations — the difference between out-of-memory and fitting at long context. On CPU you will see the reference path is the <em>slowest</em>; do not read a speedup from CPU timings. <strong>GPU-hours:</strong> ~0; a single benchmark is seconds on a GPU. <strong>CPU-only works:</strong> yes for correctness, no for the speed/memory story.</p>
</div>

<div class="callout paper"><p><strong>Research connection — FlashAttention (Dao et al. 2022).</strong> "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" is the paper behind this lesson. Its argument is precisely 08.1 + 08.2: attention is bottlenecked by HBM reads/writes, not FLOPs, and tiling + online softmax computes <em>exact</em> attention (not an approximation) with far fewer HBM accesses, giving $O(T)$ memory. See the reading guide in <a href="../../papers/index.md">the paper index</a> (paper #7, read after this module) — inspect its SRAM-vs-HBM memory-hierarchy diagram and its runtime/memory-vs-sequence-length plots. The follow-ups FlashAttention-2 and -3 refine the tiling and work partitioning for newer GPUs; the core algorithm is the one you just implemented.</p></div>

## Exercise

In the worked example (section 4), suppose you process the four keys in blocks in the *reverse* order: block 1 = keys $[3, 0]$ (values $[30, 40]$) first, then block 2 = keys $[1, 2]$ (values $[10, 20]$). **(a)** Will the final output still be $26.2089$? **(b)** In this order, on which block does the correction factor $\alpha$ actually do nothing ($\alpha = 1$), and why? **(c)** What does this tell you about `block_size` and block order in general?

<details><summary>Hint</summary>
(a) online softmax is exact regardless of order. (b) $\alpha = \exp(m_{\text{old}} - m_{\text{new}})$; when does the max *not* increase? (c) think about what the test `test_flash_result_independent_of_block_size` asserts.
</details>

<details><summary>Solution</summary>

**(a)** Yes — $26.2089$, unchanged. Online softmax computes the exact softmax regardless of the order blocks arrive; only the intermediate running values differ.

**(b)** On the *second* block. Processing $[3,0]$ first sets $m = 3$ (the global max). The second block $[1,2]$ has block-max $2 < 3$, so $m_{\text{new}} = \max(3, 2) = 3$ is unchanged, and $\alpha = \exp(3 - 3) = 1$ — no rescaling of the already-accumulated $\ell$ and $\mathbf o$ is needed, because they are already relative to the true max. (The first block still has $\alpha = 0$ as always, since $m$ starts at $-\infty$.)

**(c)** The result is invariant to both block size and block order — tiling is purely an implementation choice about memory traffic, never a change to the mathematics. That is exactly why the from-scratch test checks several block sizes (1, 3, 8, 16, 32, 64) all give the same answer. You choose `block_size` on a GPU to fit the tiles in SRAM and maximize throughput, with zero effect on correctness.

</details>

## Common mistakes

- **Thinking FlashAttention is an approximation.** It is *exact* — bit-for-bit the same softmax attention, up to floating-point rounding. It changes memory traffic, not math. (This is the property the tests enforce.)
- **Forgetting the correction factor.** Accumulating $\ell$ and $\mathbf o$ across blocks *without* rescaling by $\alpha = \exp(m_{\text{old}} - m_{\text{new}})$ when the max grows gives a wrong answer — the blocks end up on different exponent scales. The correction is the crux of online softmax.
- **Expecting a speedup on CPU.** The reference implementation is slower on CPU because its block loop is Python and CPU has no SRAM/HBM gap to exploit. The win is a GPU HBM-traffic effect; only benchmark speed on a GPU.
- **Confusing FLOPs saved with bytes saved.** FlashAttention does *not* reduce (and slightly increases, via recomputed exponentials) the FLOPs. Its win is fewer HBM bytes and $O(T)$ memory — the roofline lesson (08.2): it moves attention off the bandwidth slope.
- **Materializing the mask as a full $T\times T$ tensor.** That reintroduces the $O(T^2)$ memory you were trying to avoid. FlashAttention applies the causal mask per tile (and skips fully-future blocks), never building the whole mask.

## Check yourself

<details><summary>Why is standard attention memory-bound at long context, and what exactly does FlashAttention avoid storing?</summary>

Standard attention materializes the $T\times T$ score matrix $S$ and probability matrix $P$ in HBM, reading and writing $O(T^2)$ bytes; the softmax over them is a low-arithmetic-intensity reduction, so runtime and memory are dominated by shuttling those matrices through HBM. FlashAttention never stores the full $T\times T$ matrix — it computes it one $B_q\times B_k$ tile at a time in SRAM and folds each tile into a running output, so only $O(T)$-sized tensors ever touch HBM.

</details>

<details><summary>State the three running quantities in online softmax and what the correction factor does.</summary>

Running max $m$, running denominator $\ell = \sum \exp(\text{score}-m)$, and running unnormalized output $\mathbf o = \sum \exp(\text{score}-m)\mathbf v$ (final answer $\mathbf o/\ell$). When a new block raises the max to $m_{\text{new}}$, the correction factor $\alpha = \exp(m_{\text{old}} - m_{\text{new}}) \le 1$ rescales the already-accumulated $\ell$ and $\mathbf o$ so they are expressed relative to the new max — keeping all blocks on a common scale so the final sum equals the full-row softmax.

</details>

<details><summary>In the worked example, after block 1 the provisional output was 17.31 but the final answer was 26.21. Why did it change so much?</summary>

Block 1 saw only keys with scores $[1,2]$ and values $[10,20]$, so its provisional weighted average sat around 17. Block 2 contained the highest-scoring key (score $3$, value $30$), which after softmax carries most of the weight (0.6439). Incorporating it — via the correction factor rescaling the old accumulators and then adding the dominant new term — pulls the output up to 26.21. The provisional value is only correct over the keys seen so far; the correction machinery makes the final value exact over all keys.

</details>

<details><summary>Does the choice of block_size change the numerical result? What does it change?</summary>

No — the result is mathematically independent of block size (and block order); the from-scratch test verifies several block sizes give the identical answer to 1e-4. Block size changes only the memory-traffic pattern: on a GPU you pick it so the $Q,K,V$ tiles fit in SRAM and the SM stays busy, which sets speed and peak memory, not correctness.

</details>

<details><summary>FlashAttention does slightly MORE arithmetic than naive attention, yet is faster on a GPU. How is that possible?</summary>

Because attention is memory-bound (08.2), not compute-bound — its runtime at long context is set by HBM bandwidth, not by FLOP count. FlashAttention trades a little extra arithmetic (recomputing some exponentials, rescaling accumulators) for a large reduction in HBM traffic ($O(T)$ instead of $O(T^2)$). Moving far fewer bytes on the bandwidth-limited path wins despite the extra FLOPs — the roofline made this prediction exactly.

</details>

## Next

You have finished Module 8: you can reason about a GPU's SMs and memory hierarchy (08.1), diagnose any kernel with arithmetic intensity and the roofline and cut its bytes with fusion (08.2), find bottlenecks with the profiler and halve every tensor's bytes with bf16 mixed precision (08.3), and you have rebuilt attention itself into its IO-aware form, FlashAttention, implementing the online-softmax algorithm from scratch (08.4). The through-line — *arithmetic is cheap, moving data is expensive* — is the foundation for the next module, where the data movement problem scales up from one GPU's memory hierarchy to **many GPUs**: distributed training, and how to split a model and its optimizer state across devices (Module 9), building directly on the memory accounting of 07.4 and the bandwidth reasoning here.

Continue to [09.1 · Data parallelism, all-reduce, DDP](../module-09/lesson-01.md), and read [FlashAttention (paper #7)](../../papers/index.md) now that you have implemented its algorithm.
