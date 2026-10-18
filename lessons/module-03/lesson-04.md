# 03.4 · Muon: orthogonalized momentum for weight matrices

<div class="callout key"><p><strong>Added 2026-10-18 by the weekly curriculum review</strong> (<a href="#/research/CHANGELOG">changelog</a>). This extends Module 3; nothing earlier is replaced. AdamW from <a href="#/lessons/module-03/lesson-02">03.2</a> is still the default optimizer of this course, still trains your GPT in Module 7, and still trains part of every model that uses Muon. This lesson adds a second optimizer that a growing number of large open-weight models use for their weight <em>matrices</em>, and it shows where its advantage is real and where it is not.</p></div>

<div class="prereq">
<p><strong>Prerequisites:</strong> momentum as an EMA of gradients and the <code>SGD</code> class from <a href="#/lessons/module-03/lesson-01">03.1</a>; AdamW, decoupled weight decay and the "first Adam step is $\eta\,\operatorname{sign}(g)$" fact from <a href="#/lessons/module-03/lesson-02">03.2</a>; warmup and clipping from <a href="#/lessons/module-03/lesson-03">03.3</a>; matrix multiplication and gradients of a linear layer from <a href="#/lessons/module-02/lesson-03">02.3</a>. The singular value decomposition is <em>taught here</em> (section 2.1); you do not need it beforehand. Section 7 (MuonClip) uses attention logits $q\cdot k/\sqrt{d}$; read it after <a href="#/lessons/module-05/lesson-02">05.2</a> if you have not met attention yet.</p>
<p><strong>You will learn:</strong> the SVD and the spectral norm, just enough to state Muon; why the update of a weight <em>matrix</em> should be measured by how much it can change the layer's output (the spectral norm), and that the steepest step under that norm is the orthogonal factor $UV^\top$ of the gradient; how <strong>Newton-Schulz</strong> iterations compute that factor with five matrix-multiply rounds and no SVD, worked by hand; Muon's full update rule and its learning-rate scaling; why embeddings, the LM head and 1-D parameters still go to AdamW; a from-scratch implementation in <code>llmre.optim.muon</code> that matches <code>torch.optim.Muon</code> exactly; the compute and memory cost; and what an independent, budget-matched benchmark says about when Muon actually wins.</p>
<p><strong>Why this matters for ML:</strong> Muon started in the open NanoGPT speedrun (2024). Moonshot AI scaled it to a 16B MoE (Moonlight) and then to the 1-trillion-parameter Kimi K2. NVIDIA added it to Megatron Core, and PyTorch now ships <code>torch.optim.Muon</code>. A research engineer will meet it in training configs and papers, will be asked whether to switch, and needs to know both the mechanism and the evidence well enough to answer "it depends, and here is on what".</p>
</div>

<div class="hw">
<p><strong>Hardware track:</strong> CPU-only is enough. Tests (<code>tests/test_muon.py</code>) run in a few seconds. The experiment <code>experiments/muon_vs_adamw.py</code> trains 14 tiny GPTs (128-wide, 2 layers, 300 steps) and takes about 1.5 minutes on a 4-thread laptop CPU, under 1 GB RAM, 0 GPU-hours. Recommended: the same; a GPU changes the timing ratios (section 6) but not the method.</p>
</div>

## 1. Intuition: a weight matrix is not a bag of numbers

Every optimizer so far treats a parameter tensor as a list of independent numbers. SGD (03.1) moves each number along its own gradient component. AdamW (03.2) also scales each number by its own running RMS. On the first step it moves every entry by exactly $\pm\eta$.

A weight matrix $W$ of a linear layer $y = Wx$ is not a list of numbers. It is a **map**. What matters for the network is how much the map's *output* changes when you change $W$. Two updates with the same entry sizes can change outputs very differently:

- An update whose entries are all $\pm 1$ in the same sign pattern, like $\begin{pmatrix}1&1\\1&1\end{pmatrix}$, pushes every input in one shared direction. It has one large "stretch" and one of zero.
- An update like $\begin{pmatrix}1&0\\0&1\end{pmatrix}$ changes every direction by the same amount.

A commonly cited motivation for Muon is that gradients and momentum of transformer weight matrices look like the first kind: close to low-rank, with a few directions dominating and the rest tiny. Do not take this on trust: the implementation exercise has you measure the singular values of a real gradient of the course GPT. Muon's idea: **keep the directions of the momentum, but make every direction equally large.** Mathematically, replace the momentum matrix $M = U S V^\top$ by $U V^\top$. Directions the gradient barely uses get a full-size step, so they learn as fast as the dominant ones.

<div class="callout key"><p>Muon = momentum-SGD whose update matrix is replaced by its nearest orthogonal matrix. Same singular directions, all singular values set to (about) 1. It needs one buffer per matrix (like SGD momentum, half of AdamW), and a few extra matrix multiplications per step.</p></div>

## 2. Mathematics

### 2.1 The SVD and the spectral norm (the prerequisite)

Any real $m \times n$ matrix $G$ can be written as

$$G = U S V^\top = \sum_{i=1}^{r} \sigma_i\, u_i v_i^\top,$$

- $r = \min(m, n)$, the number of singular values;
- $U \in \mathbb{R}^{m \times r}$ with orthonormal columns $u_i$ (output directions), so $U^\top U = I$;
- $V \in \mathbb{R}^{n \times r}$ with orthonormal columns $v_i$ (input directions), $V^\top V = I$;
- $S = \operatorname{diag}(\sigma_1, \dots, \sigma_r)$ with $\sigma_1 \ge \dots \ge \sigma_r \ge 0$, the **singular values**.

Read it as: $G$ takes the input direction $v_i$, stretches it by $\sigma_i$ and points it along $u_i$. Three norms we need:

- **Spectral norm** $\lVert G\rVert_{\mathrm{op}} = \sigma_1$: the largest stretch, $\max_{x \ne 0} \lVert Gx\rVert / \lVert x\rVert$.
- **Frobenius norm** $\lVert G\rVert_F = \sqrt{\sum_{jk} G_{jk}^2} = \sqrt{\sum_i \sigma_i^2}$. Always $\ge \sigma_1$.
- **Nuclear norm** $\lVert G\rVert_* = \sum_i \sigma_i$.

The **orthogonal (polar) factor** of $G$ is $\operatorname{orth}(G) = U V^\top$: same $u_i, v_i$, every $\sigma_i$ replaced by $1$.

Tiny example. $G = \begin{pmatrix}2&1\\1&2\end{pmatrix}$ has $v_1 = u_1 = \tfrac{1}{\sqrt2}(1,1)$ with $\sigma_1 = 3$ (since $G(1,1)^\top = 3(1,1)^\top$) and $v_2 = u_2 = \tfrac{1}{\sqrt2}(1,-1)$ with $\sigma_2 = 1$. So $\lVert G\rVert_{\mathrm{op}} = 3$, $\lVert G\rVert_F = \sqrt{4+1+1+4} = \sqrt{10} \approx 3.162 = \sqrt{3^2 + 1^2}$, $\lVert G\rVert_* = 4$, and $\operatorname{orth}(G) = u_1u_1^\top + u_2u_2^\top = I$.

### 2.2 Steepest descent depends on how you measure the step

For a small change $\Delta W$, the loss changes by about $\langle G, \Delta W\rangle = \sum_{jk} G_{jk}\,\Delta W_{jk}$, where $G = \nabla_W L$. "Steepest descent" means: among all steps of size at most $\eta$, take the one that lowers this most. The answer depends on how "size" is measured:

| Size measured by | Steepest step | Name |
|---|---|---|
| Frobenius norm $\lVert\Delta W\rVert_F \le \eta$ | $-\eta\, G / \lVert G\rVert_F$ | normalized SGD |
| largest entry $\max_{jk}\lvert\Delta W_{jk}\rvert \le \eta$ | $-\eta\,\operatorname{sign}(G)$ | sign descent (Adam's first step, 03.2) |
| spectral norm $\lVert\Delta W\rVert_{\mathrm{op}} \le \eta$ | $-\eta\, U V^\top$ | Muon (without momentum) |

The third row is the one that fits a linear layer. For any input $x$, the output change is $\lVert \Delta W\, x\rVert \le \lVert\Delta W\rVert_{\mathrm{op}}\lVert x\rVert$, so the spectral norm is exactly "the most this step can change any output, per unit of input". Bernstein and Newhouse ([Old Optimizer, New Norm, arXiv 2409.20325](https://arxiv.org/abs/2409.20325)) work out this view: Adam without its EMAs is sign descent, and different layer types call for different norms.

**Why $UV^\top$ is the answer (three lines).** Write $\langle G, \Delta W\rangle = \sum_i \sigma_i\, u_i^\top \Delta W\, v_i$. Each $\lvert u_i^\top \Delta W v_i\rvert \le \lVert\Delta W\rVert_{\mathrm{op}} \le \eta$, so $\langle G, \Delta W\rangle \ge -\eta \sum_i \sigma_i = -\eta \lVert G\rVert_*$. The choice $\Delta W = -\eta UV^\top$ gives $u_i^\top \Delta W v_i = -\eta$ for every $i$, so it reaches the bound.

**Numbers.** Take $G = \begin{pmatrix}2&1\\1&2\end{pmatrix}$ and allow spectral size $1$:

- Muon step $-UV^\top = -I$: decrease $\langle G, I\rangle = 2 + 2 = 4$ ($= \lVert G\rVert_*$, the maximum).
- Normalized-SGD direction scaled to spectral size 1, $-G/3$: decrease $\lVert G\rVert_F^2/3 = 10/3 \approx 3.33$.
- Sign step $\operatorname{sign}(G) = \begin{pmatrix}1&1\\1&1\end{pmatrix}$ has singular values $2$ and $0$; scaled to spectral size 1 it is $-\tfrac12\begin{pmatrix}1&1\\1&1\end{pmatrix}$: decrease $\tfrac12 \cdot 6 = 3$.

The sign step turned a full-rank gradient into a **rank-1** update. It changes no output along $(1,-1)$, the direction the gradient asked to change by $\sigma_2 = 1$. Muon changes it by a full unit.

### 2.3 The Muon update

For each 2-D parameter $W \in \mathbb{R}^{A \times B}$ ($A$ = fan-out, $B$ = fan-in, the `nn.Linear` weight layout) with gradient $G_t$:

$$
\begin{aligned}
M_t &= \mu M_{t-1} + (1-\mu)\, G_t && \text{momentum EMA, } M_0 = 0\\
\tilde M_t &= (1-\mu)\, G_t + \mu M_t && \text{Nesterov look-ahead (or } \tilde M_t = M_t)\\
O_t &= \mathrm{NS}_5(\tilde M_t) \approx \operatorname{orth}(\tilde M_t) && \text{section 3}\\
W_t &= W_{t-1}(1 - \eta\lambda) - \eta\, s(A,B)\, O_t && \text{decoupled decay, then the step}
\end{aligned}
$$

- $\mu$ — momentum, $0.95$ by default; $\eta$ — learning rate; $\lambda$ — weight decay (the same decoupled decay as AdamW).
- $s(A, B)$ — a shape-dependent scale. Keller Jordan's original choice is $\sqrt{\max(1, A/B)}$. Moonlight's choice ([arXiv 2502.16982](https://arxiv.org/abs/2502.16982)) is $0.2\sqrt{\max(A, B)}$, explained next.

The EMA form $(1-\mu)$ differs from 03.1's $v \leftarrow \mu v + g$ only by the constant factor $1-\mu$. Orthogonalization ignores constant factors ($\operatorname{orth}(cM) = \operatorname{orth}(M)$ for $c > 0$), so the two give the same Muon step.

**Why $0.2\sqrt{\max(A,B)}$.** An orthogonal $A \times B$ matrix has $\min(A,B)$ singular values equal to 1, so $\lVert O\rVert_F^2 = \min(A,B)$ and its RMS entry is $\sqrt{\min(A,B)/(AB)} = 1/\sqrt{\max(A,B)}$. Example: $A = 64$, $B = 256$: RMS $= 1/16 = 0.0625$. Multiplying by $0.2\sqrt{256} = 3.2$ gives RMS $0.2$, about the RMS of a typical AdamW update. With this scale Moonshot reused AdamW's learning rate and weight decay and needed no new tuning **(company claim; Moonlight report)**.

<div class="callout warn"><p><strong>Muon is only for hidden weight matrices.</strong> Biases and norm gains are 1-D, so there is nothing to orthogonalize. The token embedding is a lookup table whose input is a one-hot vector, so the natural size of its update is per row (per token), not the spectral norm of the whole table; orthogonalizing would push a rare token's row, whose momentum has decayed to almost nothing, as hard as a frequent token's. The LM head is an output classifier. In both cases the Muon README and every published Muon run keep AdamW, and the reason given is mainly empirical: it works better. All of these stay on AdamW. Every published Muon run is therefore a <em>hybrid</em>: Muon for the transformer's 2-D hidden weights, AdamW for the rest. <code>split_params_for_muon</code> and <code>MuonWithAuxAdamW</code> in the code below do this split.</p></div>

## 3. Newton-Schulz: orthogonalize without an SVD

An SVD on a GPU is slow and hard to parallelize. Matrix multiplication is the fastest thing a GPU does (08.1). Newton-Schulz builds $\operatorname{orth}(G)$ from matrix multiplications only.

### 3.1 The trick: odd polynomials act on singular values

Let $X = U S V^\top$. Then $X X^\top X = U S V^\top V S U^\top U S V^\top = U S^3 V^\top$, because $V^\top V = U^\top U = I$. In the same way $(XX^\top)^2 X = U S^5 V^\top$. So for any odd polynomial $p(x) = a x + b x^3 + c x^5$,

$$a X + b\,(XX^\top) X + c\,(XX^\top)^2 X = U\, p(S)\, V^\top.$$

The singular vectors stay the same; **each singular value goes through $p$ separately**. If we repeat a $p$ that pushes every value in $(0, 1]$ towards 1, we converge to $U V^\top$. First divide by $\lVert X\rVert_F$, which is $\ge \sigma_1$, so all singular values start in $(0, 1]$.

The classical choice is the cubic $p(x) = 1.5x - 0.5x^3$. It has $p(1) = 1$ and $p'(1) = 0$, so it converges fast near 1. Near 0, though, it only multiplies by 1.5 per step.

### 3.2 By hand: two cubic steps

Take $G = \begin{pmatrix}2&1\\1&2\end{pmatrix}$ from section 2.1, with $\sigma = (3, 1)$. Because only singular values change, track them as numbers.

- Normalize: $\lVert G\rVert_F = \sqrt{10} = 3.1623$, so $\sigma^{(0)} = (3, 1)/3.1623 = (0.9487,\ 0.3162)$.
- Step 1: $p(0.9487) = 1.4230 - 0.5\cdot 0.8538 = 0.9961$; $p(0.3162) = 0.4743 - 0.5\cdot 0.0316 = 0.4585$.
- Step 2: $p(0.9961) = 1.4942 - 0.4942 = 1.0000$; $p(0.4585) = 0.6878 - 0.0482 = 0.6396$.

Put the directions back after step 1: $X_1 = 0.9961\,u_1u_1^\top + 0.4585\,u_2u_2^\top = \begin{pmatrix}0.7273&0.2688\\0.2688&0.7273\end{pmatrix}$. The large value is done after two steps. The small one goes $0.32 \to 0.46 \to 0.64 \to 0.83$ and needs about 5 steps. If the ratio were $30{:}1$ instead of $3{:}1$ (start $0.033$), the cubic would need about 10 steps. That is the cost of a slope of only 1.5 at zero.

### 3.3 The tuned quintic: fast and deliberately inexact

Muon uses $p(x) = 3.4445x - 4.7750x^3 + 2.0315x^5$. Its slope at zero is $3.4445$, so small singular values grow more than twice as fast as with the cubic. The price: $p(1) = 3.4445 - 4.7750 + 2.0315 = 0.701$, so 1 is *not* a fixed point. Values do not settle on 1; they bounce around in a band. On the same example:

| step | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| $\sigma_1$ | 0.949 | 0.752 | 1.048 | 0.682 | 1.134 | 0.753 |
| $\sigma_2$ | 0.316 | 0.945 | 0.757 | 1.041 | 0.682 | 1.134 |

After 5 steps every singular value of a random matrix lies roughly in $[0.68, 1.13]$ (the test checks a band around 1). The PyTorch source comments that this does not hurt model quality compared with exact $UV^\top$. What matters is that *no* direction stays tiny. Exactly equal values are not needed.

Each step costs three matmuls, written as $A = XX^\top$, $B = bA + cA^2$, $X \leftarrow aX + BX$. The iteration works on the wide orientation (rows $\le$ columns, transposing if needed) so that $A$ is the small $r\times r$ Gram matrix.

## 4. Tensor shapes, dtype, device

For one hidden matrix, e.g. the GPT's `mlp.c_fc.weight` with $C = 128$:

- `p`: `(A, B) = (512, 128)` float32, on the model's device. The gradient and the momentum buffer have the same shape, dtype and device.
- Newton-Schulz runs on the transpose `(128, 512)` (wide orientation). The Gram matrix `A` is `(128, 128)`. The iteration runs in **bfloat16** in `torch.optim.Muon` and in our default, then is cast back to float32. bf16 is accurate enough here because the result only has to be *roughly* orthogonal, and it makes the matmuls about twice as fast on tensor cores.
- `lr_scale((512, 128), "original")` $= \sqrt{\max(1, 4)} = 2$; for `mlp.c_proj` `(128, 512)` it is 1.
- Every Muon tensor stays 2-D. 1-D tensors raise an error and must go to AdamW.

## 5. From scratch: `llmre.optim.muon`

The core of [`llmre.optim.muon`](../../code/src/llmre/optim/muon.py):

```python
def newton_schulz(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7, dtype=torch.bfloat16):
    a, b, c = coeffs
    X = G if dtype is None else G.to(dtype)
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T                                   # work on the wide side: (r, s), r <= s
    X = X / X.norm().clamp(min=eps)               # Frobenius >= spectral: all sigma <= 1
    for _ in range(steps):
        A = X @ X.T                               # (r, r)
        B = torch.addmm(A, A, A, beta=b, alpha=c) # b*A + c*A@A, one rounding
        X = torch.addmm(X, B, X, beta=a)          # a*X + B@X  ->  sigma -> p(sigma)
    if tall:
        X = X.T
    return X.to(G.dtype)

class Muon:
    @torch.no_grad()
    def step(self):
        for p, buf in zip(self.params, self.bufs):
            if p.grad is None:
                continue
            g = p.grad
            buf.lerp_(g, 1.0 - self.momentum)            # M <- mu*M + (1-mu)*G
            update = g.lerp(buf, self.momentum) if self.nesterov else buf
            O = self.orthogonalize(update)               # Newton-Schulz (or exact SVD)
            if self.weight_decay != 0.0:
                p.mul_(1.0 - self.lr * self.weight_decay)
            p.add_(O, alpha=-self.lr * lr_scale(p.shape, self.adjust_lr))
```

`torch.addmm(M, P, Q, beta=s, alpha=t)` computes $sM + t\,PQ$ in one call. It is the same mathematics as the readable two-line form. We use it so the bf16 rounding is identical to `torch.optim.Muon`, which lets the test require agreement to $10^{-6}$ over five steps (in practice the results are bit-identical). The module also has:

- `orthogonalize_svd(G)` — exact $UV^\top$, the reference;
- `CUBIC_COEFFS` — the classical iteration; the test checks it converges to $UV^\top$ to $10^{-10}$;
- `lr_scale`, `newton_schulz_flops`, `split_params_for_muon(model)` and `MuonWithAuxAdamW(model, muon_kwargs, adamw_kwargs)`, the hybrid. The hybrid reuses 03.2's `AdamW` class unchanged.

Tests in `code/tests/test_muon.py`: cubic convergence to the SVD answer; the quintic band and unchanged singular vectors; the hand example of section 3.2; step-for-step equality with `torch.optim.Muon` (Nesterov on/off, two lr scalings); the **reduction test**: for a $1 \times n$ matrix the only singular direction is $m/\lVert m\rVert$, so exact Muon is momentum-SGD with a normalized step; the parameter split on the course GPT (tied `wte`/`lm_head` to AdamW, once); and a hybrid training smoke test.

<div class="callout pt"><p><strong>What <code>torch.optim.Muon</code> does.</strong> The same algorithm: an EMA momentum buffer via <code>lerp_</code>, the Nesterov blend, five quintic Newton-Schulz steps in bfloat16 with Keller Jordan's coefficients, decoupled weight decay with the unscaled lr, and an <code>adjust_lr_fn</code> of <code>"original"</code>, <code>"match_rms_adamw"</code> or <code>"spectral_unclamped"</code>. It refuses non-2-D parameters, so you build a second AdamW for the rest. It has no fused or multi-tensor path yet. At scale the extra work is distributing Newton-Schulz across GPUs, which is a framework feature (section 6), not part of the optimizer math.</p></div>

## 6. Cost: compute, memory, and where the time goes

**Memory.** Muon keeps one buffer per matrix, AdamW two. For the parameters Muon handles, fp32 training needs $4P$ (weights) $+ 4P$ (grads) $+ 4P$ (momentum) $= 12P$ bytes instead of AdamW's $16P$ (03.2). That is a 25% saving on the largest part of the model.

**Compute.** One quintic step on an $m \times n$ matrix with $r = \min$, $s = \max$ costs $4r^2s + 2r^3$ FLOPs (`newton_schulz_flops`). For a square $d \times d$ matrix: $6d^3$ per step, $30d^3$ for five steps. The forward and backward pass through that matrix for a batch of $N$ tokens costs $6Nd^2$ FLOPs (2 forward, 4 backward). So

$$\frac{\text{Newton-Schulz FLOPs}}{\text{layer FLOPs}} = \frac{30 d^3}{6 N d^2} = \frac{5d}{N}.$$

- Our toy, $d = 128$, $N = 16 \times 64 = 1024$ tokens: $5 \cdot 128/1024 = 0.63$. Newton-Schulz costs more than half as much as the training itself.
- $d = 4096$ with $N = 10^6$ tokens per optimizer step: $0.02$, i.e. 2%.

Two consequences. First, Muon's overhead **shrinks with batch size** and is large for small models with small batches. Second, under plain data parallelism (09.1) every replica would repeat the same Newton-Schulz on the full matrix while the forward/backward is split, so $N$ in the formula becomes the *per-GPU* batch. NVIDIA's Megatron Core therefore gives each layer's Newton-Schulz to a single data-parallel rank (a "layer-wise distributed optimizer", in the spirit of ZeRO in 09.2). For tensor-parallel shards it offers three modes: duplicate the iteration on each shard, distribute the matmuls, or orthogonalize each block separately (no communication, but only an approximation of full orthogonalization).

**Measured.** NVIDIA reports, on GB300 NVL72 in MXFP8, Kimi K2 at 1,051 TFLOP/s/GPU with AdamW vs 1,080 with Muon (of which 51 are Newton-Schulz), and Qwen3-30B-A3B at 713 vs 721 (35 Newton-Schulz) ([NVIDIA blog, 2026-04-22](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/)) **(company claim)**. In other words, at that scale the model math runs about 2–4% slower and Newton-Schulz fills the gap.

### Run it

```bash
cd code && PYTHONPATH=src python experiments/muon_vs_adamw.py
```

It trains the course GPT ($C = 128$, 2 layers, byte vocabulary, $16 \times 64$ tokens per step, 300 steps, 2 seeds) on the text of lessons 03.1–03.3. AdamW gets lr $\in \{3\cdot10^{-4}, 10^{-3}, 3\cdot10^{-3}\}$ and Muon gets lr $\in \{0.01, 0.02, 0.05\}$ with an auxiliary AdamW at $3\cdot10^{-3}$. One run on a 4-thread CPU printed:

| | best lr | val loss after 300 steps | ms / step | share of step time in `optimizer.step()` |
|---|---|---|---|---|
| AdamW | $10^{-3}$ | 2.41 | 17.3 | 8.7% |
| Muon + AdamW | 0.02 | **1.90** | 23.5 | 33% |
| AdamW, **same wall-clock** (407 steps) | $10^{-3}$ | 2.21 | — | — |

Expected observations: at equal steps Muon is clearly ahead; the optimizer takes about a third of each step (section 6 predicts Newton-Schulz FLOPs around half of the layer FLOPs on this toy); and giving AdamW the extra time closes only part of the gap *here*. Exact numbers depend on your CPU. With the auxiliary AdamW lr set to $10^{-3}$, the same as AdamW-only, Muon still reaches about 1.92, so the gap is not caused by the embedding learning rate.

## 7. At scale: what the evidence says

**Adoption (publicly documented).**

- **NanoGPT speedrun** ([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)): Muon is one of the techniques behind its training-speed records. The current records combine it with later variants.
- **Moonlight** (Moonshot AI, [arXiv 2502.16982](https://arxiv.org/abs/2502.16982), Feb 2025): two changes were needed to scale Muon, weight decay and the per-matrix update-RMS scale of section 2.3. Their scaling-law runs report about 2× the compute efficiency of AdamW for compute-optimal training, and they trained a 16B-total / 3B-active MoE on 5.7T tokens **(company claim)**.
- **Kimi K2** ([arXiv 2507.20534](https://arxiv.org/abs/2507.20534)): a 1T-total / 32B-active MoE pretrained on 15.5T tokens with **MuonClip**, "with zero loss spike" **(company claim)**.
- **NVIDIA Megatron Core** and **PyTorch** (`torch.optim.Muon`) ship it. NVIDIA's blog also names GLM-5 as Muon-trained; we have not checked that against GLM-5's own report.

**MuonClip (read after 05.2).** Moonshot found that attention logits blow up more often with Muon than with AdamW: in a 9B-active / 53B-total run with plain Muon, the maximum logit quickly passed 1000. Their fix, **QK-Clip**, runs *after* each optimizer step. For each head $h$ it records $S^h_{\max} = \max_{i,j} q_i^h \cdot k_j^h / \sqrt{d}$ over the batch. If $S^h_{\max} > \tau$ (they use $\tau = 100$), it sets $\gamma_h = \tau / S^h_{\max}$ and multiplies that head's $W_q$ and $W_k$ by $\sqrt{\gamma_h}$ each. The logit $q\cdot k$ scales by $\sqrt{\gamma_h}\cdot\sqrt{\gamma_h} = \gamma_h$, so the next forward pass is back near $\tau$. Example: $S_{\max} = 400$ gives $\gamma = 0.25$, so both weights are multiplied by $0.5$ and logits drop to about 100. Only heads over the threshold are touched, and the step's own gradient is unchanged. (K2 uses MLA attention, so its version scales the per-head query/key parts and leaves the shared rotary key alone. The idea is the same.)

**When does Muon actually win? (independent benchmark).** Akhtar, Nounou, Mansouri and Nounou ([arXiv 2610.21555](https://arxiv.org/abs/2610.21555), NeurIPS 2026 OPT workshop) tuned both optimizers with the same 23-configuration budget and then ran 5 matched seeds on 17.7M and 33.6M-parameter causal LMs and a 2.7M-parameter ViT on one A100:

- **At equal data**, Muon had lower test loss in all 20 primary pairs (validation-loss area 3.25–22.44% lower).
- **At equal wall-clock time**, Muon still won on vision (19.68–22.31%), but **AdamW won on language** (3.38–6.96%). The full Muon call took 56.1–69.6% of the training time versus 6.1–9.1% for AdamW. That matches the $5d/N$ estimate for small models with small batches.
- With a restricted tuning budget, Muon showed higher expected regret, i.e. it was less forgiving of a poor configuration.
- The authors state that small models with reference kernels do not settle the ranking for large models or optimized implementations.

Read the two halves together: **Muon reliably gets more out of each token; whether it gets more out of each second depends on how well its overhead is amortized.** At frontier scale with a distributed implementation the overhead is a few percent (NVIDIA's numbers). On a small model with a single-GPU reference implementation it can cancel the gain. Our CPU toy happened to stay ahead at equal time; the paper's language runs did not. Both are honest results for their own setup. That is why you measure on your own workload.

<div class="callout key"><p><strong>When to still use plain AdamW.</strong> Small models with small batches, where Newton-Schulz is a large share of the step; fine-tuning a model that was pretrained with AdamW, where its existing recipe is known to work; a tight tuning budget, where AdamW's known defaults are safer; and everything that is not a hidden matrix, in every setup.</p></div>

**Documented vs inferred.** *Publicly documented:* the Muon algorithm and coefficients (the Muon repository, `torch.optim.Muon`); Moonlight's scaling changes and reported efficiency; Kimi K2's MuonClip and QK-Clip formula; NVIDIA's throughput table; the benchmark paper's numbers and caveats. *Reasonable industry practice:* the Muon/AdamW parameter split; bf16 Newton-Schulz; distributing Newton-Schulz across data-parallel ranks at scale. *Inference:* that Muon's data-efficiency advantage persists at every frontier scale for every architecture. Each lab reports its own runs, and no independent large-scale replication exists yet.

## Exercise

Show that Muon's step does not change if the gradient is multiplied by a positive constant: if every $G_t$ becomes $kG_t$ with $k > 0$, the parameters follow exactly the same path. Then name one thing AdamW (03.2) does that has the same property, and one thing in this lesson that breaks it.

<details><summary>Hint</summary>

Follow $k$ through the momentum EMA and the Nesterov blend: both are linear. Then look at the very first operation inside `newton_schulz`.

</details>

<details><summary>Stronger hint</summary>

$\operatorname{orth}(kM)$: the SVD of $kM$ is $U(kS)V^\top$. And `X / X.norm()` divides out any scale before the polynomial runs.

</details>

<details><summary>Solution</summary>

The EMA and the blend are linear, so $\tilde M_t$ becomes $k\tilde M_t$. Newton-Schulz first divides by $\lVert k\tilde M_t\rVert_F = k\lVert\tilde M_t\rVert_F$, so its input, and therefore $O_t$, is unchanged. (Exact orthogonalization agrees: $kM = U(kS)V^\top$ has the same $UV^\top$.) The step $\eta\,s(A,B)\,O_t$ and the decay do not involve $G$, so the path is identical. AdamW has the same property up to $\epsilon$: $\hat m/\sqrt{\hat v}$ is unchanged when $g \to kg$. What breaks it: the `eps` floor (only for near-zero matrices) and, more importantly, **gradient clipping** (03.3), which rescales only when the norm passes a threshold. With Muon, clipping only matters through the momentum mix of steps with different scales, not through the step size.

</details>

## Implementation exercise

Add the **cubic** variant as an option to the experiment script: train with `ns_coeffs=CUBIC_COEFFS` and `ns_steps` $\in \{5, 10\}$, and compare the validation loss and time per step with the quintic at 5 steps. Then use `newton_schulz` with `dtype=None` on a real gradient from the GPT (`model.transformer.h[0].mlp.c_fc.weight.grad` after one `backward()`) and print its singular values before and after: is the gradient really close to low-rank?

<details><summary>Hint</summary>

`torch.linalg.svdvals(g)` gives the singular values in descending order. Compare `s[0]` with `s[10]` and `s[-1]`; the ratio `s[0]**2 / (s**2).sum()` tells you how much of the gradient lives in its top direction.

</details>

<details><summary>Expected observation</summary>

The singular values fall off steeply: the first few hold most of the squared Frobenius norm. After 5 quintic steps they all sit near 1. The cubic at 5 steps leaves the small ones well below 1 (section 3.2), so it behaves more like an under-orthogonalized step. The cubic at 10 steps is close to exact but costs twice as many matmuls. Typically the loss is about the same as with the quintic, which is the reason the inexact quintic is used.

</details>

## Debugging exercise

A colleague switches a GPT training run from AdamW to `torch.optim.Muon(model.parameters(), lr=0.02)` and gets a `ValueError` about 1-D parameters. They "fix" it by reshaping every bias and LayerNorm gain to shape `(1, n)` and also pass the tied embedding/LM-head matrix to Muon. Training now runs but the loss is worse than AdamW and the embedding rows of rare tokens drift. What went wrong, in two parts?

<details><summary>Solution</summary>

(1) A `(1, n)` matrix has one singular direction, so its "orthogonalized" update is $m/\lVert m\rVert$: every bias and gain now moves by a fixed total length of $\eta\, s(A,B)$ per step, however small its gradient. That is normalized SGD, not anything Muon is designed for. Those parameters belong to AdamW. (2) The `(V, C)` embedding gets a sparse gradient: only rows of tokens in the batch are non-zero, and rare tokens' rows only carry slowly decaying momentum. Orthogonalization sets every singular value to about 1, so those weak directions, including stale rare-token rows, are pushed as hard as the strong ones. It is also the LM head, where per-entry (AdamW) scaling is the documented practice. Fix: `muon, rest = split_params_for_muon(model)`, Muon on `muon`, AdamW on `rest` (or `MuonWithAuxAdamW`).

</details>

## Common mistakes

- **Muon on everything.** 1-D tensors, embeddings and the LM head go to AdamW (section 2.3).
- **Forgetting the shape scale.** Without $s(A,B)$, wide and tall matrices get updates of very different RMS sizes, and the AdamW learning rate does not transfer. Decide on `"original"` or `"match_rms_adamw"` and tune lr for it.
- **Orthogonalizing the gradient instead of the momentum.** Momentum comes first. Orthogonalize-then-average gives a different and worse optimizer.
- **Fused QKV matrices.** Our GPT's `c_attn` is one `(3C, C)` matrix, so Muon orthogonalizes Q, K and V *together*. Many implementations split it and orthogonalize Q, K and V separately. Neither is wrong, but they are different optimizers. Say which one you used.
- **Judging by steps alone.** Report loss against tokens *and* against wall-clock time, as the benchmark paper does. Muon can win the first and lose the second.
- **Expecting exact orthogonality.** The quintic's singular values sit in roughly $[0.7, 1.1]$. Testing `O @ O.T == I` against the default iteration will fail. Use `orthogonalize_svd` or the cubic for exact checks.

## What to read / what to skip

- **Muon repository** ([GitHub](https://github.com/KellerJordan/Muon)): read the README's usage (the parameter split) and follow its link to the original writeup for the motivation and the coefficient tuning. Skip the CIFAR-10 details.
- **Old Optimizer, New Norm** ([arXiv 2409.20325](https://arxiv.org/abs/2409.20325)): read the steepest-descent-under-a-norm view and the Adam-as-sign-descent section. Skim the Shampoo and Prodigy parts.
- **Muon is Scalable for LLM Training** ([arXiv 2502.16982](https://arxiv.org/abs/2502.16982)): read the weight-decay and update-RMS sections and the scaling-law comparison. Skip the distributed implementation unless you work on Module 9 systems.
- **Kimi K2** ([arXiv 2507.20534](https://arxiv.org/abs/2507.20534)): read the MuonClip / QK-Clip section and its algorithm box. The rest is a model report.
- **When Does Muon Actually Win?** ([arXiv 2610.21555](https://arxiv.org/abs/2610.21555)): read the protocol (shared tuning budget, matched seeds, frozen test) and the equal-time results. It is a good template for any optimizer comparison you run.
- **NVIDIA Megatron Core blog** ([link](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/)): read the layer-wise distributed optimizer and the three tensor-parallel modes; the throughput table is a vendor measurement.

## Code this after reading

Implement **QK-Clip** for the course's `CausalSelfAttention`: during the forward pass record, per head, the maximum pre-softmax logit over the batch; after `optimizer.step()`, for each head over $\tau$, scale the matching rows of the Q and K parts of `c_attn.weight` (and their biases) by $\sqrt{\tau/S_{\max}}$. Test that after clipping, re-running the forward pass on the same batch gives a maximum logit of at most $\tau$ (up to floating-point error). Note that with biases the logit is not exactly bilinear in the weights; set `bias=False` for an exact test.

## Research connection

<div class="callout paper"><p><strong>Muon: An optimizer for hidden layers in neural networks</strong> — Jordan, Jin, Boza, You, Cesista, Newhouse, Bernstein, 2024 (<a href="https://github.com/KellerJordan/Muon">repository and writeup</a>). <strong>Old Optimizer, New Norm</strong> — Bernstein &amp; Newhouse, 2024, <a href="https://arxiv.org/abs/2409.20325">arXiv 2409.20325</a>. <strong>Muon is Scalable for LLM Training</strong> — Liu, Su et al. (Moonshot AI), 2025, <a href="https://arxiv.org/abs/2502.16982">arXiv 2502.16982</a>. <strong>Kimi K2: Open Agentic Intelligence</strong> — Kimi Team, 2025, <a href="https://arxiv.org/abs/2507.20534">arXiv 2507.20534</a> (MuonClip). <strong>When Does Muon Actually Win? A Paired Data-, Time-, and Tuning-Budget Benchmark</strong> — Akhtar, Nounou, Mansouri, Nounou, 2026, <a href="https://arxiv.org/abs/2610.21555">arXiv 2610.21555</a>. Open directions: the optimizer still changes quickly (newer Muon variants appear in the speedrun records, and a 2026 preprint, <a href="https://arxiv.org/abs/2610.20256">Long-Horizon RL Needs Muon</a>, uses it for agentic RL), so treat specific variants as research, not standard practice.</p></div>

## Check yourself

<details><summary>What does Muon do to the singular values and singular vectors of the momentum matrix?</summary>

It keeps the singular vectors ($U$, $V$) and sets every non-zero singular value to about 1 (exactly 1 for $UV^\top$; roughly $0.7$–$1.1$ with the five-step quintic). Directions the gradient barely used get the same step size as the dominant ones.

</details>

<details><summary>Why does $aX + b(XX^\top)X + c(XX^\top)^2X$ leave the singular vectors unchanged?</summary>

With $X = USV^\top$, $(XX^\top)X = US^3V^\top$ and $(XX^\top)^2X = US^5V^\top$ because $U^\top U = V^\top V = I$. The sum is $U(aS + bS^3 + cS^5)V^\top$: same $U$ and $V$, with the polynomial applied to each singular value.

</details>

<details><summary>Why divide by the Frobenius norm before iterating, and not by the spectral norm?</summary>

The polynomial is designed for singular values in $(0, 1]$, so the largest must be at most 1. The spectral norm would be exact, but it needs an SVD or a power iteration. The Frobenius norm is one cheap reduction and is always $\ge \sigma_1$, so dividing by it is safe. The cost is that small singular values start even smaller, which the large slope at 0 handles.

</details>

<details><summary>Under which norm is $\operatorname{sign}(G)$ the steepest step, and under which is $UV^\top$? Why is the second one natural for a linear layer?</summary>

$\operatorname{sign}(G)$ is steepest under the max-entry norm, $UV^\top$ under the spectral norm. The spectral norm bounds how much the layer's output can change for any input, $\lVert\Delta W x\rVert \le \lVert\Delta W\rVert_{\mathrm{op}}\lVert x\rVert$, which is what we want to control when we take a step.

</details>

<details><summary>A $4096 \times 4096$ matrix, 5 Newton-Schulz steps, per-GPU batch of 8192 tokens under plain data parallel. Roughly what fraction of that layer's training FLOPs does Newton-Schulz add? What does a layer-wise distributed optimizer on 64 ranks change?</summary>

$5d/N = 5 \cdot 4096/8192 = 2.5$, i.e. Newton-Schulz costs 2.5 times the layer's forward+backward. That is far too much. With a layer-wise distributed optimizer each matrix is orthogonalized by one rank instead of all 64, so the per-GPU cost drops by 64× to about $2.5/64 \approx 4\%$. Larger per-step batches reduce it further.

</details>

<details><summary>What does QK-Clip do when a head's maximum logit is 250 and $\tau = 100$?</summary>

$\gamma = 100/250 = 0.4$. That head's $W_q$ and $W_k$ are each multiplied by $\sqrt{0.4} \approx 0.632$, so its logits scale by $0.4$ and the maximum drops to about 100. Other heads are untouched. It happens after the optimizer step and does not change the gradient.

</details>

<details><summary>The benchmark paper finds Muon better at equal data but AdamW better at equal time on small language models. Which result would you expect to change at frontier scale, and why?</summary>

The equal-time result. Muon's data efficiency is a property of the update direction. Its time overhead is $\propto 5d/N$ per replica and depends on the implementation. Frontier runs have large batches and distribute Newton-Schulz across ranks, which brings it down to a few percent (NVIDIA's measurement). Then the data-efficiency advantage, if it holds at that scale, also shows up in time. Whether it holds is what each lab reports for its own runs; independent large-scale replications are still missing.

</details>

## Next

You now have three optimizers for one job: SGD with momentum (03.1), AdamW with its per-entry scaling (03.2), and Muon, which scales the whole matrix through its singular values (this lesson), together with the schedules and clipping that wrap all of them (03.3). The rest of the course trains with AdamW. Muon is an option you can now evaluate on your own runs, with both token-matched and time-matched comparisons. The next module turns to how text becomes the numbers the model consumes: **tokenization**.

Continue to [04.1 · Characters, bytes, Unicode, the vocab problem](lessons/module-04/lesson-01.md).
