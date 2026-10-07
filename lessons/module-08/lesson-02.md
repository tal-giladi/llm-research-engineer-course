# 08.2 · Arithmetic intensity, roofline, fusion

<div class="prereq">
<p><strong>Prerequisites:</strong> the memory hierarchy and the "arithmetic is cheap, data movement is expensive" fact from <a href="#/lessons/module-08/lesson-01">08.1 · GPU architecture & memory hierarchy</a> (especially HBM bandwidth vs peak FLOP/s); FLOP counting and the $6N$ rule from <a href="#/lessons/module-07/lesson-04">07.4</a>; the matmul FLOP cost $2n^2k$ used there.</p>
<p><strong>You will learn:</strong> <strong>arithmetic intensity</strong> (AI) — the ratio FLOPs ÷ bytes moved — as the single number that predicts whether a kernel is memory-bound or compute-bound; the <strong>roofline model</strong>, which bounds achievable performance by $\min(\text{peak FLOP/s},\ \text{bandwidth} \times \text{AI})$ and whose <strong>ridge point</strong> separates the two regimes; a worked AI for an elementwise add (very low → memory-bound) versus a large matmul (high → compute-bound); and <strong>kernel fusion</strong> — combining elementwise ops so intermediates never round-trip to HBM, which is precisely the idea FlashAttention takes to its limit.</p>
<p><strong>Why this matters for ML:</strong> the roofline is the back-of-envelope tool engineers use to decide whether an optimization can possibly help. If a kernel is memory-bound, buying more FLOP/s (a faster GPU, lower-precision matmul) does nothing — you must move fewer bytes. If it is compute-bound, the opposite. Getting this diagnosis right is the difference between an optimization that doubles throughput and one that changes nothing. It is also the exact argument for why FlashAttention (08.4) wins.</p>
</div>

## 1. Intuition: two ways a kernel can be slow

From 08.1: a kernel reads bytes from HBM and does FLOPs on them. There are two fundamentally different reasons it can be slow.

1. **It has too much arithmetic.** Even if data arrives instantly, the arithmetic units can only do so many FLOP/s. A huge matrix multiply is limited this way — it is **compute-bound**.
2. **It cannot get data fast enough.** The arithmetic is trivial, but the kernel must haul a big tensor across HBM, and bandwidth is the ceiling. An elementwise add is limited this way — it is **memory-bound**.

Which one you are in decides which optimizations can possibly help. And you can predict it from a single ratio computed before running anything.

## 2. Arithmetic intensity: FLOPs per byte

**Arithmetic intensity** (AI) of a kernel is

$$
\text{AI} = \frac{\text{total FLOPs the kernel does}}{\text{total bytes it moves to/from HBM}},
$$

measured in FLOPs per byte. It answers: for each byte I pay to move, how much useful arithmetic do I get out of it? High AI means you amortize each expensive byte-move over lots of cheap FLOPs (good — compute-bound). Low AI means you barely use each byte before needing the next (bad — memory-bound, the units starve).

The "bytes moved" is the traffic to the slow level of the hierarchy — for us, HBM. Data that stays in registers or SRAM does not count; that is exactly why fusion and FlashAttention help (they keep data on-chip so it never adds to the byte count).

<div class="callout key"><p><strong>Arithmetic intensity = FLOPs ÷ bytes moved from HBM.</strong> It is a property of the <em>algorithm</em>, not the hardware. Low AI ⇒ memory-bound (starved for data). High AI ⇒ compute-bound (starved for arithmetic units). You can compute it on paper before launching a single kernel.</p></div>

## 3. Worked example A: elementwise add (very low AI → memory-bound)

Take `c = a + b` where `a`, `b`, `c` are fp32 vectors of length $n$. Count both quantities.

**FLOPs:** one add per element $= n$ FLOPs.

**Bytes moved (HBM):** read all of `a` ($4n$ bytes), read all of `b` ($4n$), write all of `c` ($4n$). Total $= 12n$ bytes.

$$
\text{AI} = \frac{n}{12n} = \frac{1}{12} \approx 0.083 \text{ FLOPs/byte}.
$$

The $n$ cancels — the intensity does not depend on how big the vector is, only on the *shape* of the computation: one FLOP for every 12 bytes touched. That is about **1800× below** the A100 ridge of ~150 FLOPs/byte (08.1). This kernel will run at roughly $0.083/150 \approx 0.06\%$ of peak FLOP/s — it is almost purely bandwidth-limited. Its runtime is essentially "bytes moved ÷ HBM bandwidth," and no faster arithmetic unit changes that.

Every elementwise op in a transformer — bias add, GELU, dropout, residual add, the scale-and-shift of LayerNorm — has AI in this same tiny neighborhood. Individually cheap in FLOPs; collectively a real cost, because each is a full HBM round trip at ~0.1 FLOPs/byte.

## 4. Worked example B: a large matmul (high AI → compute-bound)

Now a square matmul $C = A B$ with $A, B, C$ all $n \times n$ in fp32, $n = 1024$.

**FLOPs:** a matmul of $(n \times k)$ by $(k \times n)$ is $2 n^2 k$ (each output element is a length-$k$ dot product = $k$ multiplies + $k$ adds). Here $k = n$, so

$$
\text{FLOPs} = 2 n^3 = 2 \cdot 1024^3 = 2{,}147{,}483{,}648 \approx 2.1 \times 10^9.
$$

**Bytes moved:** read $A$ ($n^2$ floats), read $B$ ($n^2$), write $C$ ($n^2$) — in the ideal case where each input is read once. $3 n^2 \cdot 4$ bytes $= 3 \cdot 1024^2 \cdot 4 = 12{,}582{,}912 \approx 1.26 \times 10^7$ bytes.

$$
\text{AI} = \frac{2 n^3}{12 n^2} = \frac{n}{6} = \frac{1024}{6} \approx 170.7 \text{ FLOPs/byte.}
$$

Notice AI grows *linearly with $n$*: the FLOPs grow as $n^3$ but the data only as $n^2$, so bigger matmuls have higher intensity. At $n = 1024$ we are already at ~171 FLOPs/byte — above the A100 ridge of ~150 — so this matmul is **compute-bound**. It keeps the Tensor Cores busy; making HBM faster would not speed it up, but a faster arithmetic unit (or lower-precision Tensor Cores) would.

This is the fundamental asymmetry that makes deep learning fast on GPUs: the dominant cost, matmul, has *rising* arithmetic intensity, so it lands on the good side of the roofline. The problem is everything *around* the matmuls.

<div class="callout key"><p>Elementwise add: AI ≈ $\tfrac{1}{12}$ FLOPs/byte, fixed — memory-bound. Square matmul: AI = $\tfrac{n}{6}$ FLOPs/byte, <em>grows with size</em> — compute-bound once $n$ is large. Matmuls reuse each loaded value across a whole row/column of output; elementwise ops use each value once. Reuse is what buys arithmetic intensity.</p></div>

## 5. The roofline model

Put both effects on one picture. The **roofline model** says the achievable performance of a kernel (in FLOP/s) is bounded by the smaller of two ceilings:

$$
\text{attainable FLOP/s} \;=\; \min\big(\underbrace{P_{\max}}_{\text{peak compute}},\ \underbrace{B \times \text{AI}}_{\text{bandwidth} \times \text{intensity}}\big),
$$

where $P_{\max}$ is the hardware's peak FLOP/s and $B$ is its HBM bandwidth (bytes/s).

Read the two terms:

- The **compute ceiling** $P_{\max}$ is a flat horizontal line: you can never beat the peak arithmetic rate, no matter how high the intensity.
- The **bandwidth ceiling** $B \times \text{AI}$ is a slanted line rising with AI: at intensity AI, the most FLOP/s you can sustain is bandwidth times intensity, because you can only feed $B$ bytes/s and each byte yields AI FLOPs.

Plotted with AI on the x-axis and attainable FLOP/s on the y-axis (both log scale), the two ceilings form a "roofline": a rising diagonal (bandwidth-limited) that hits a horizontal cap (compute-limited). Your kernel sits *under* the roofline at its own AI.

### The ridge point

The two lines cross where $B \times \text{AI} = P_{\max}$, i.e. at

$$
\text{AI}_{\text{ridge}} = \frac{P_{\max}}{B}.
$$

For the A100: $\text{AI}_{\text{ridge}} = \dfrac{312 \times 10^{12}}{2039 \times 10^{9}} \approx 153$ FLOPs/byte. (Using the more exact 2039 GB/s HBM figure; with the round ~2 TB/s it is ~156, the number from 08.1.)

The ridge point is the dividing line:

- **AI < ridge** → you are on the slanted part → **memory-bound**. You are limited by bandwidth. To go faster, *move fewer bytes* (fusion, lower precision, better data reuse) or raise AI. Buying peak FLOP/s does nothing.
- **AI > ridge** → you are on the flat part → **compute-bound**. You are limited by arithmetic. To go faster, *do fewer FLOPs* or use faster/lower-precision arithmetic units. Buying more bandwidth does nothing.

<div class="callout key"><p>Roofline: attainable FLOP/s = $\min(P_{\max},\ B \cdot \text{AI})$. The <strong>ridge point</strong> $\text{AI}_{\text{ridge}} = P_{\max}/B$ (≈ 150 FLOPs/byte on an A100) splits the world: left of it kernels are memory-bound and only fewer bytes help; right of it they are compute-bound and only fewer/faster FLOPs help. Locate a kernel's AI relative to the ridge and you instantly know which optimizations are even worth trying.</p></div>

### Placing our two examples on the roofline

- Elementwise add, AI ≈ 0.083: far left of the ridge, deep in memory-bound territory. Attainable FLOP/s $= B \cdot \text{AI} = 2\times10^{12} \cdot 0.083 \approx 1.7 \times 10^{11}$ — about 0.05% of the 312 TFLOP/s peak. The GPU's arithmetic is almost entirely idle; it is a bandwidth machine here.
- Matmul $n=1024$, AI ≈ 171: just right of the ridge, compute-bound. Attainable FLOP/s $\approx P_{\max}$ — it can actually approach peak. This is where the GPU earns its keep.

Here is the picture, with a few more kernels from this lesson placed on it. Every dot sits *on* the roof at its own AI: that is the best the kernel can possibly do. Notice how the dot for a matmul walks up the slope as $n$ grows ($n/6$: 64 → 10.7, 1024 → 171, 4096 → 683) and stops climbing once it crosses the ridge, while the elementwise ops and LayerNorm sit far down the slope no matter how big the tensor is.

![Roofline for an A100 on log-log axes: a rising diagonal (bandwidth times arithmetic intensity) meets a flat ceiling at 312 TFLOP/s at the ridge, about 153 FLOPs per byte. Elementwise add sits at 0.083, the unfused bias plus GELU plus dropout chain at 0.25, the fused version at 0.75, LayerNorm at 2 and a 64 by 64 matmul at 10.7, all on the memory-bound slope; matmuls of size 1024 and 4096 sit on the flat compute-bound ceiling](../../assets/img/roofline-a100.svg)

(The bias+GELU+dropout dots assume about one FLOP per element per op in bf16; section 6 explains why fusing them moves the dot 3× to the right.)

Try it yourself: change $n$ and the dtype and watch where the matmul crosses the ridge, then switch the GPU to an H100 and see the ridge move right (its peak FLOP/s grew faster than its bandwidth, so the same kernel can become *more* memory-bound on newer hardware).

<iframe src="assets/interactive/roofline.html?mode=matmul" title="Roofline explorer" style="width:100%;height:1150px;border:0;border-radius:14px" onload="try{var f=this,w=f.contentWindow,m=f.contentDocument.querySelector('main'),s=function(){f.style.height=(m.offsetHeight+4)+'px'};s();w.addEventListener('resize',s);f.contentDocument.addEventListener('input',s,true);f.contentDocument.addEventListener('click',s,true)}catch(e){}"></iframe>

## 6. Kernel fusion: raise AI by not writing intermediates to HBM

The roofline gives the fix for memory-bound kernels directly: **move fewer bytes.** The most important way to do that in practice is **kernel fusion**.

Consider the classic transformer MLP tail: a Linear produces `z`, then you add a bias, apply GELU, and apply dropout:

```python
z = x @ W + b          # matmul (compute-bound) — leave it
h = z + bias           # elementwise  -> reads z, writes h
g = gelu(h)            # elementwise  -> reads h, writes g
y = dropout(g)         # elementwise  -> reads g, writes y
```

Done as **four separate kernels**, the three elementwise ops each make a full HBM round trip. For an activation tensor of $M$ elements (fp16, 2 bytes), the elementwise part moves roughly:

- bias add: read $z$ + write $h$ = $4M$ bytes
- gelu: read $h$ + write $g$ = $4M$ bytes
- dropout: read $g$ + write $y$ = $4M$ bytes
- total ≈ $12M$ bytes, for only ~a few FLOPs per element.

Now **fuse** them into one kernel: read $z$ once from HBM into registers, add the bias, apply GELU, apply dropout, and write $y$ once. The intermediates `h` and `g` never leave the SM — they live in registers.

- fused: read $z$ + write $y$ = $4M$ bytes total.

![Two panels. Top, unfused: three kernels (bias add, GELU, dropout) each read their input from HBM and write their output back, so the intermediates h and g make round trips through HBM, 12M bytes in total. Bottom, fused: one kernel reads z once, keeps h and g in registers, and writes y once, 4M bytes in total](../../assets/img/fusion-hbm-traffic.svg)

Same FLOPs, **one third of the HBM traffic**, and the AI triples. On the roofline the kernel slides up its bandwidth line toward the ridge. You also save two of the three kernel launches (08.1: each launch has fixed overhead). This is why every serious framework fuses elementwise chains — `torch.compile`, TorchInductor, and hand-written fused kernels all do exactly this.

<div class="callout key"><p><strong>Fusion = do a chain of elementwise/reduction ops in one kernel so the intermediate tensors stay in registers/SRAM and never round-trip to HBM.</strong> It cuts bytes moved (raising AI toward the ridge) and cuts kernel-launch overhead. It cannot speed up an already compute-bound matmul — fusion is the lever for the memory-bound tail around the matmuls.</p></div>

<div class="callout pt"><p>In PyTorch you rarely hand-write fused kernels; you call <code>torch.compile(model)</code> and TorchInductor fuses the elementwise chains for you, generating a single kernel (often via Triton — 08.4) for a run of pointwise ops. But "PyTorch does it for you" is not an explanation: what it is <em>doing</em> is exactly the byte-count reduction above — keeping <code>h</code> and <code>g</code> in registers so the only HBM traffic is one read of <code>z</code> and one write of <code>y</code>. Knowing that is how you predict whether <code>compile</code> will help (it helps a memory-bound elementwise tail a lot; it does little for a kernel already at the compute ceiling).</p></div>

To see this move on the roofline, open the **Elementwise chain + fusion** tab in the roofline explorer above: add ops to the chain and the unfused dot stays put (every op is another full round trip) while the fused dot slides right, because its bytes stay fixed at one read and one write.

## 7. FlashAttention: fusion applied to attention, plus the tricks that make it possible

Is FlashAttention the same thing as kernel fusion, or something extra? **Both.** Its *goal* is exactly the fusion goal from section 6: run a whole chain of operations as one kernel so the intermediate results never touch HBM. What is new is that attention's intermediates are too big, and too interconnected, for the simple fusion of section 6 to work. FlashAttention adds two tricks, **tiling** and **online softmax**, that make fusion possible for attention. Fusion is the idea; FlashAttention is the hardest case of that idea plus the engineering that cracks it.

### 7.1 The chain we want to fuse

Standard attention for one head is a chain of three kernels, the same shape as section 6's bias → GELU → dropout chain:

1. $S = QK^\top/\sqrt d$ (a matmul): write $S$ to HBM;
2. $P = \mathrm{softmax}(S)$ (a row-wise reduction plus elementwise work): read $S$, write $P$;
3. $O = PV$ (a matmul): read $P$ and $V$, write $O$.

The difference from section 6 is size. $Q, K, V, O$ are $T \times d$, but the intermediates $S$ and $P$ are $T \times T$. With one head, $T = 4096$, $d = 64$, fp16:

- $Q, K, V, O$: $4096 \cdot 64 \cdot 2 \approx 0.5$ MB each, about 2 MB together.
- $S$ and $P$: $4096^2 \cdot 2 \approx 33.5$ MB **each**.
- HBM traffic: write $S$, read $S$, write $P$, read $P$ $= 4 \times 33.5 \approx 134$ MB, plus the 2 MB of inputs and outputs, $\approx 136$ MB.
- FLOPs: two matmuls of $2T^2d$ each, $4T^2d = 4 \cdot 4096^2 \cdot 64 \approx 4.3$ GFLOP.
- $\text{AI} \approx 4.3 \times 10^9 / 136 \times 10^6 \approx 32$ FLOPs/byte: left of the A100 ridge (~153), memory-bound, at most ~21% of peak.

About 98% of the bytes are $S$ and $P$, intermediates nobody needs once the output is computed. That is exactly the situation fusion fixes.

### 7.2 Why section 6's fusion does not work as-is

Fusing bias → GELU → dropout was easy for two reasons. Both fail for attention.

1. **Each output depended only on the matching input.** $\mathrm{gelu}(h_i)$ needs only $h_i$, so a thread can load a chunk, process it, write it, and never look at anything else. Softmax is different: $P_{ij} = e^{S_{ij} - m_i} / \sum_k e^{S_{ik} - m_i}$, where $m_i$ is the row maximum. To produce even one probability you need the max and the sum over the **entire row** of $T$ scores.
2. **The intermediates fit on chip.** $h$ and $g$ were the same size as the chunk being processed, so they lived in registers. A block of 128 query rows of $S$ is $128 \times 4096 \times 4$ bytes $= 2$ MB in fp32, while an A100 SM has about 164 KB of SRAM (08.1). It does not fit, and it grows with $T$.

Could you process one query row at a time instead? One row of 4096 scores is 16 KB and fits. But then every query row has to read **all of $K$ and $V$** from HBM again: $4096 \text{ rows} \times 1 \text{ MB} \approx 4$ GB of traffic, 30× *worse* than standard attention. Tiny tiles kill data reuse, and reuse is what buys arithmetic intensity (section 4).

### 7.3 The two extra tricks

- **Tiling in both dimensions.** Cut $Q$ into blocks of $B_q$ rows (say 128) and $K, V$ into blocks of $B_k$ rows. Each kernel instance owns one $Q$ block, keeps it in SRAM, and streams $K, V$ tiles past it. One score tile is $128 \times 128 \times 4$ bytes $= 64$ KB and fits. Every $K, V$ tile it loads is reused by all 128 query rows, so the work on each tile is a small matmul with matmul-like reuse.
- **Online softmax.** This removes the row dependency. For each query row the kernel keeps a running max $m$, a running sum $\ell$, and a running output accumulator. When a new tile reveals a larger max, it rescales what it has accumulated so far by $e^{m_{\text{old}} - m_{\text{new}}}$. After the last tile it divides by $\ell$. The result is exactly softmax over the full row, not an approximation. 08.4 works it through by hand.

Training adds one more roofline decision: the backward pass does not store $P$ either. It recomputes the score tiles from $Q$, $K$ and the saved $m, \ell$. That spends extra FLOPs, which are cheap, to avoid storing and re-reading $T \times T$ bytes, which are not.

### 7.4 What it buys, in numbers

Same $T = 4096$, $d = 64$, $B_q = 128$:

- read $Q$ once and write $O$ once: about 1 MB;
- stream $K$ and $V$ once per query block, $T / B_q = 32$ times: $32 \times 1.05 \approx 33.6$ MB;
- total $\approx 35$ MB instead of 136 MB, for the same 4.3 GFLOP;
- $\text{AI} \approx 124$ FLOPs/byte instead of 32. On the roofline the dot slides from ~21% of peak to ~81%, close to the ridge.

The bigger win is **memory held**. The $T \times T$ matrix never exists, so attention needs $O(T)$ memory instead of $O(T^2)$. At $T = 131{,}072$ a single fp16 $T \times T$ matrix is $131072^2 \cdot 2 \approx 34$ GB, per head, per layer. Without FlashAttention, long context does not fit on the GPU at all.

(The $K, V$ re-reads still grow like $T^2$, but scaled by $d/B_q$, which keeps them about 4× below the standard version's $S$/$P$ traffic here. What drops to $O(T)$ is the memory held, not the traffic.)

![Two panels. Top, standard attention: three kernels, with the T by T matrices S and P written to and read back from HBM, about 136 MB of traffic for T 4096 and d 64, arithmetic intensity about 32, memory-bound. Bottom, FlashAttention: one kernel per block of 128 query rows keeps a 128 by 128 score tile in SRAM, updates the running max and sum, and accumulates the output; only Q, K, V and O touch HBM, about 35 MB, arithmetic intensity about 124](../../assets/img/flashattention-vs-fusion.svg)

| | Section 6 fusion (bias → GELU → dropout) | FlashAttention |
|---|---|---|
| What is fused | 3 elementwise kernels | matmul → softmax → matmul |
| Intermediates kept off HBM | $h$, $g$ (same size as the input) | $S$, $P$ ($T \times T$) |
| Each output needs | the matching input element | a whole row of $T$ scores |
| Fits on chip? | yes, chunk by chunk | only as tiles |
| Extra tricks needed | none | tiling + online softmax (+ recompute in backward) |
| Bytes saved here | 3× | ~4×, and the $T \times T$ memory is gone |

Try it below: slide $T$ and watch the "score matrix" line. The AI of both versions barely moves with $T$ (standard stays near $d/2$, Flash near $B_q$), but the standard version's $T \times T$ matrix grows from megabytes to tens of gigabytes.

<iframe src="assets/interactive/roofline.html?mode=attn" title="Roofline explorer: attention vs FlashAttention" style="width:100%;height:1150px;border:0;border-radius:14px" onload="try{var f=this,w=f.contentWindow,m=f.contentDocument.querySelector('main'),s=function(){f.style.height=(m.offsetHeight+4)+'px'};s();w.addEventListener('resize',s);f.contentDocument.addEventListener('input',s,true);f.contentDocument.addEventListener('click',s,true)}catch(e){}"></iframe>

<div class="callout key"><p><strong>FlashAttention = kernel fusion of the whole attention chain + tiling + online softmax.</strong> Fusion is the goal: never write $S$ or $P$ to HBM. Tiling makes the intermediates fit in SRAM while keeping data reuse high. Online softmax lets softmax be computed tile by tile, exactly. Same FLOPs (slightly more), far fewer bytes, and $O(T)$ memory. Lesson 08.4 builds it.</p></div>

## Exercise

A LayerNorm over an activation tensor of $M$ fp16 elements reads the tensor, computes the mean and variance (a reduction), then writes the normalized-scaled-shifted output. Model it as: read $M$ elements once, write $M$ elements once, and do about $8$ FLOPs per element (subtract mean, square, accumulate, normalize, scale, shift). **(a)** Compute its arithmetic intensity. **(b)** On an A100 (ridge ≈ 150 FLOPs/byte), is it memory-bound or compute-bound? **(c)** Fusing the LayerNorm with the following Linear's bias/activation is often proposed — but which part dominates the LayerNorm's cost, and does fusion address it?

<details><summary>Hint</summary>
(a) bytes = read $2M$ + write $2M$ = $4M$; FLOPs = $8M$. (b) compare AI to 150. (c) think about whether LayerNorm's bottleneck is FLOPs or the two HBM passes over the tensor.
</details>

<details><summary>Stronger hint</summary>
(a) $\text{AI} = 8M / 4M = 2$ FLOPs/byte. (b) $2 \ll 150$. (c) the cost is dominated by moving the tensor (memory-bound), so what helps is reducing the number of HBM round trips over that tensor — which is what fusing it with its neighbor does.
</details>

<details><summary>Solution</summary>

**(a)** Bytes moved = $2M$ (read) + $2M$ (write) = $4M$. FLOPs = $8M$. $\text{AI} = 8M / 4M = 2$ FLOPs/byte.

**(b)** $2 \ll 150$, so it is firmly **memory-bound** — deep on the bandwidth slope of the roofline. Its runtime is set by how fast HBM can stream the tensor in and out, not by the arithmetic.

**(c)** The cost is the two HBM passes over the big activation tensor, not the ~8 FLOPs/element. So the win comes from cutting HBM round trips. Fusing LayerNorm's output directly into the next kernel (so the normalized tensor is consumed on-chip rather than written to HBM and read back) removes a full round trip and raises the effective AI. Making the arithmetic faster would do essentially nothing, because arithmetic was never the bottleneck — the roofline told us so before we ran anything.

</details>

## Common mistakes

- **Optimizing FLOPs on a memory-bound kernel.** Rewriting a LayerNorm or softmax to do fewer arithmetic operations barely moves its runtime — it is limited by bytes, not FLOPs. Always diagnose with AI first.
- **Counting all memory, not just HBM traffic.** AI is FLOPs per byte *moved to/from the slow level*. Data kept in registers/SRAM does not count; if it did, fusion would look like it changes nothing. The whole point of fusion is to move traffic off HBM and into on-chip memory that the AI denominator ignores.
- **Assuming a bigger/faster GPU always helps.** If your bottleneck kernel is memory-bound, a GPU with more peak FLOP/s but similar bandwidth gives little; you need more bandwidth or fewer bytes. State which ceiling you are under before you buy hardware.
- **Forgetting that matmul AI grows with size.** Small matmuls (tiny batch, small hidden size) can be memory-bound too — their AI ($\sim n/6$) is small when $n$ is small. Compute-bound is not automatic; it depends on the dimensions.

## Check yourself

<details><summary>Define arithmetic intensity and state what "high" vs "low" implies for which optimizations help.</summary>

AI = FLOPs ÷ bytes moved from HBM. **Low AI** (below the ridge) ⇒ memory-bound ⇒ only moving fewer bytes helps (fusion, lower precision, better reuse); faster arithmetic does nothing. **High AI** (above the ridge) ⇒ compute-bound ⇒ only fewer/faster FLOPs help; more bandwidth does nothing.

</details>

<details><summary>Why does a large matmul have high AI while an elementwise add has low AI, when both touch big tensors?</summary>

Reuse. In a matmul each loaded value participates in a whole row/column of output — many FLOPs per byte loaded — and FLOPs grow as $n^3$ while data grows as $n^2$, so AI $\sim n/6$ rises with size. In an elementwise add each loaded value is used exactly once (one FLOP), so AI is a tiny constant (~1/12) regardless of size. AI is about data reuse, not tensor size.

</details>

<details><summary>An A100 has ridge point ≈ 150 FLOPs/byte. A kernel measures AI = 4 FLOPs/byte and runs at 3% of peak FLOP/s. Is that surprising, and what should you try?</summary>

Not surprising — at AI = 4, far left of the ridge, the roofline caps you at $B\cdot\text{AI} = 2\times10^{12}\cdot 4 = 8\times10^{12}$ FLOP/s ≈ 2.6% of the 312 TFLOP/s peak, so ~3% is essentially hitting the bandwidth ceiling. The kernel is memory-bound. Try to move fewer bytes: fuse it with neighbors, use a lower-precision dtype, or restructure to reuse data on-chip. Do not bother optimizing its arithmetic.

</details>

<details><summary>How does kernel fusion raise arithmetic intensity, and why can't it speed up a compute-bound matmul?</summary>

Fusion keeps intermediate tensors in registers/SRAM instead of writing them to HBM and reading them back, so the bytes-moved denominator of AI shrinks while FLOPs stay the same — AI rises and the kernel slides up toward the ridge. A compute-bound matmul is already limited by the arithmetic-unit ceiling, not by bytes, so removing HBM traffic (which was not the bottleneck) does not make it faster. Fusion is the tool for the memory-bound tail, not the matmul.

</details>

<details><summary>A colleague says "FlashAttention is just torch.compile fusing the attention ops." What is right and what is missing?</summary>

Right: the goal is fusion, one kernel so the $T \times T$ intermediates $S$ and $P$ never go to HBM. Missing: the simple elementwise fusion a compiler does for bias → GELU → dropout cannot do this, because softmax needs a whole row of $T$ scores before it can output anything, and a block of those rows does not fit in SRAM. FlashAttention adds tiling (so tiles fit and $K, V$ are reused across a block of queries) and online softmax (running max and sum, rescaling as new tiles arrive) so the softmax can be done tile by tile, exactly.

</details>

## Next

You can now diagnose any kernel with one ratio and know which optimizations can possibly help — the roofline turns "the GPU feels slow" into "this kernel is at 4 FLOPs/byte, memory-bound, fuse it or shrink its dtype." The next lesson puts this into practice: the **PyTorch profiler** to *find* the hot, memory-bound kernels in a real training step, and **mixed precision** (bf16/fp16) — the single most effective way to halve the bytes every kernel moves.

Continue to [08.3 · Profiling & mixed precision](lessons/module-08/lesson-03.md).
