# 08.5 · Fused linear cross-entropy: the loss without the logits

<div class="callout key"><p><strong>Added 2026-10-11 by the weekly curriculum review</strong> (<a href="#/research/CHANGELOG">changelog</a>). This extends Module 8; nothing earlier is replaced. The <code>lm_head</code> + <code>F.cross_entropy</code> loss you wrote in <a href="#/lessons/module-06/lesson-01">06.1</a> is still the definition of the loss, and still the right code for small models. This lesson shows how production trainers compute exactly the same number without ever holding the full logits tensor, using the online-softmax trick from <a href="#/lessons/module-08/lesson-04">08.4</a>.</p></div>

<div class="prereq">
<p><strong>Prerequisites:</strong> cross-entropy as $\operatorname{logsumexp}(z) - z_y$ and the max-subtraction trick from <a href="#/lessons/module-01/lesson-04">01.4</a>; the $(B, T, V)$ logits tensor and why it is huge from <a href="#/lessons/module-06/lesson-01">06.1</a> (section 6) and <a href="#/lessons/module-07/lesson-04">07.4</a>; kernel fusion and memory-bound operations from <a href="#/lessons/module-08/lesson-02">08.2</a>; the online-softmax recurrence (running max $m$, running sum $\ell$, correction $\alpha$) and recomputation in backward from <a href="#/lessons/module-08/lesson-04">08.4</a>; the ignore-index trick from <a href="#/lessons/module-14/lesson-02">14.2</a> is useful but not required.</p>
<p><strong>You will learn:</strong> the gradient of cross-entropy with respect to the logits, $\operatorname{softmax}(z) - \text{onehot}(y)$, and why it lets you compute <em>all</em> gradients the moment a chunk of logits exists; <strong>chunking over positions</strong> (Liger Kernel's fused linear cross-entropy); <strong>tiling over the vocabulary</strong> with an online logsumexp and recomputation in backward (Cut Cross-Entropy); per-token <strong>log-probs and entropy</strong> computed the same way (what RL and distillation trainers need); a from-scratch implementation in <code>llmre.training.fused_ce</code> checked against <code>F.cross_entropy</code> and against the GPT model's own loss; and the memory and FLOP cost of each choice.</p>
<p><strong>Why this matters for ML:</strong> with modern vocabularies (128k–262k tokens) the logits are often the single largest activation in training, larger than all the transformer's per-layer activations for a small model. Every serious open training stack now avoids materializing them: Liger Kernel's fused linear cross-entropy is integrated into Hugging Face's Trainer, TRL, Axolotl and LLaMA-Factory, and TRL v1.15.0 (October 2026) made a fused LM head the default in six trainers. If you train long contexts or score RL rollouts, you will meet this; if you read a memory profile, you need to recognise it.</p>
</div>

<div class="hw">
<p><strong>Hardware track:</strong> CPU-only is enough. The implementation is the algorithm in pure PyTorch, not a Triton kernel. The tests (<code>tests/test_fused_ce.py</code>) run in about a second; the "Run it" comparison below ($N = 2048$, $C = 256$, $V = 32{,}000$) takes a few seconds and under 1 GB of RAM. 0 GPU-hours. The real speed and memory benefit needs a GPU kernel (Liger Kernel, Cut Cross-Entropy or TRL's fused LM head); we do not use them here.</p>
</div>

## 1. Intuition: you need two numbers per row, not $V$

Recall from [01.4](lessons/module-01/lesson-04.md) that the cross-entropy loss of one position is

$$\ell = \operatorname{logsumexp}(z) - z_y,$$

the log of the softmax denominator minus the logit of the correct token. For that you need exactly **two numbers per position**: one log-normalizer and one target logit. The naive code builds all $V$ logits for every position, writes them to memory, reads them back to take a logsumexp, then keeps a same-size tensor for backward. With $V = 151{,}936$ (a Qwen-size vocabulary) and $N = B \cdot T = 32{,}768$ tokens, that tensor is $32{,}768 \times 151{,}936 \times 4 \approx 19.9$ GB in fp32. That is the same number [17.4](lessons/module-17/lesson-04.md) quoted for distillation.

The fix is the same idea as FlashAttention ([08.4](lessons/module-08/lesson-04.md)): the softmax denominator can be accumulated **one tile at a time** with a running max and a running sum. The target logit is a single dot product $h \cdot w_y$. The gradient has a closed form you can compute from a tile and the final log-normalizer. So the full logits never need to exist.

There are two ways to cut the $(N, V)$ matrix into pieces:

- **Chunk the rows (positions).** Take $n$ positions at a time and build their full $(n, V)$ logits. That is enough for an exact loss *and* exact gradients for those rows, so you compute both and throw the chunk away. Memory: one chunk. This is Liger Kernel's fused linear cross-entropy.
- **Tile the columns (vocabulary).** For all positions, walk across the vocabulary $B_v$ columns at a time and keep only the running max and sum. Memory: $O(N)$ statistics. In backward, recompute the tiles. This is Cut Cross-Entropy (CCE). A real kernel tiles both axes so each tile lives in on-chip SRAM.

## 2. Mathematics

### 2.1 The loss

Notation, all for one batch flattened to $N$ positions:

- $h_i \in \mathbb{R}^C$ — final hidden state at position $i$ (the output of `ln_f`), $i = 1..N$.
- $W \in \mathbb{R}^{V \times C}$ — the LM-head weight, row $w_j$ for token $j$ (the `nn.Linear(C, V)` layout; with weight tying it is also the token embedding).
- $z_{ij} = h_i \cdot w_j$ — logit of token $j$ at position $i$.
- $y_i$ — target token; $\mathcal{V} = \{ i : y_i \ne \text{ignore} \}$ the valid positions, $|\mathcal{V}|$ their count.
- $\operatorname{lse}_i = \log \sum_{j=1}^{V} e^{z_{ij}}$ — the log-normalizer.

$$
\mathcal{L} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \big(\operatorname{lse}_i - z_{i y_i}\big),
\qquad z_{i y_i} = h_i \cdot w_{y_i}.
$$

The division is by the **global** valid count. A chunked implementation must use the same denominator in every chunk (see the debugging exercise).

### 2.2 The gradient with respect to the logits

Differentiate one row's loss $\ell_i = \operatorname{lse}_i - z_{i y_i}$ with respect to $z_{ij}$. The derivative of $\operatorname{lse}_i$ is the softmax probability $p_{ij} = e^{z_{ij} - \operatorname{lse}_i}$, and $z_{iy_i}$ contributes $-1$ only at $j = y_i$:

$$
g_{ij} \;=\; \frac{\partial \mathcal{L}}{\partial z_{ij}} \;=\; \frac{\mathbb{1}[i \in \mathcal{V}]}{|\mathcal{V}|}\,\big(p_{ij} - \mathbb{1}[j = y_i]\big).
$$

Since $z = H W^\top$ (with $H \in \mathbb{R}^{N\times C}$ stacking the $h_i$), the chain rule through the matmul gives

$$
\frac{\partial \mathcal{L}}{\partial H} = G\,W \in \mathbb{R}^{N \times C},
\qquad
\frac{\partial \mathcal{L}}{\partial W} = G^\top H \in \mathbb{R}^{V \times C},
$$

where $G \in \mathbb{R}^{N\times V}$ collects the $g_{ij}$. Two consequences drive everything below:

1. **Row chunks are self-contained.** Row $i$ of $G$ needs only row $i$'s logits and $\operatorname{lse}_i$. Once a chunk's logits exist, its rows of $G$, its rows of $\partial\mathcal{L}/\partial H$, and its *additive contribution* $G_{\text{chunk}}^\top H_{\text{chunk}}$ to $\partial \mathcal{L}/\partial W$ can all be computed immediately.
2. **Column tiles need only $\operatorname{lse}_i$.** For a vocabulary tile $J$, $g_{ij} = (e^{z_{ij} - \operatorname{lse}_i} - \mathbb{1}[j=y_i])/|\mathcal V|$ needs that tile's logits plus the already-known scalar $\operatorname{lse}_i$. So backward can recompute the tile from $h_i$ and $w_j$ and never store it.

### 2.3 Online logsumexp over vocabulary tiles

Split the vocabulary into tiles $J_1, J_2, \dots$ of width $B_v$. Keep a running max $m$ and running sum $\ell = \sum e^{z - m}$ per row (08.4's recurrence, applied to the vocabulary axis instead of the key axis):

$$
m^{(k)} = \max\!\big(m^{(k-1)},\ \max_{j\in J_k} z_{j}\big),\quad
\ell^{(k)} = \ell^{(k-1)}\,e^{\,m^{(k-1)} - m^{(k)}} + \sum_{j \in J_k} e^{\,z_j - m^{(k)}},
$$

with $m^{(0)} = -\infty$, $\ell^{(0)} = 0$, and at the end $\operatorname{lse} = m + \log \ell$. The factor $\alpha = e^{m^{(k-1)} - m^{(k)}} \le 1$ re-expresses the old sum relative to the new max. The proof that this is exact is the one you checked in 08.4.

### 2.4 Entropy and log-probs, online

RL and distillation trainers score tokens rather than train on a loss: they need $\log p_{iy_i}$ (for importance ratios, KL penalties, distillation divergences) and often the entropy $H_i = -\sum_j p_{ij} \log p_{ij}$ (a monitoring signal and an optional bonus in GRPO-style trainers). The log-prob is $z_{iy_i} - \operatorname{lse}_i$. For the entropy, substitute $\log p_{ij} = z_{ij} - \operatorname{lse}_i$:

$$
H_i = -\sum_j p_{ij}(z_{ij} - \operatorname{lse}_i) = \operatorname{lse}_i - \sum_j p_{ij} z_{ij} = \operatorname{lse}_i - \frac{s_i}{\ell_i},
\qquad s_i = \sum_j e^{z_{ij} - m_i}\, z_{ij}.
$$

$s$ is one more running sum, rescaled by the same $\alpha$ as $\ell$. Three numbers per row ($m, \ell, s$) give both the log-prob and the entropy, without the $(N, V)$ tensor.

## 3. Numerical example by hand

One position, $C = 2$, $V = 4$, two vocabulary tiles of width $B_v = 2$.

$$
h = [1,\ 2],\quad
W = \begin{bmatrix} 1 & 0\\ 0 & 1\\ 1 & 1\\ -1 & 0 \end{bmatrix},\quad y = 1.
$$

Logits $z_j = h \cdot w_j$: $z = [1,\ 2,\ 3,\ -1]$.

**Naive.** $\operatorname{lse} = \log(e^1 + e^2 + e^3 + e^{-1}) = \log(2.71828 + 7.38906 + 20.08554 + 0.36788) = \log 30.56076 = 3.419717$. Loss $= 3.419717 - z_1 = 3.419717 - 2 = 1.419717$.

**Online, tile 1** ($z = [1, 2]$): $m = 2$, $\ell = e^{1-2} + e^{0} = 0.367879 + 1 = 1.367879$.

**Online, tile 2** ($z = [3, -1]$): the max rises to $m = 3$, so $\alpha = e^{2-3} = 0.367879$ and

$$\ell = 1.367879 \times 0.367879 + e^{0} + e^{-4} = 0.503215 + 1 + 0.018316 = 1.521530.$$

$\operatorname{lse} = 3 + \log 1.521530 = 3 + 0.419717 = 3.419717$. Same as naive. The target logit needs no tile at all: $z_y = h\cdot w_1 = 1\cdot 0 + 2 \cdot 1 = 2$.

**Gradient.** $p_j = e^{z_j - 3.419717}$: $p = [0.088947,\ 0.241783,\ 0.657233,\ 0.012038]$ (sums to 1). Subtract one at $y = 1$:

$$g = p - \text{onehot}(1) = [0.088947,\ -0.758217,\ 0.657233,\ 0.012038].$$

$\partial\ell/\partial h = g\,W = 0.088947\,[1,0] - 0.758217\,[0,1] + 0.657233\,[1,1] + 0.012038\,[-1,0] = [0.734142,\ -0.100984]$.

$\partial\ell/\partial W = g^\top h$: row $j$ is $g_j \cdot [1, 2]$, e.g. row 1 is $[-0.758217,\ -1.516435]$. In the vocabulary-tiled backward, tile 1 produces rows 0–1 of this and tile 2 rows 2–3. Each tile only needs its own logits and the scalar $3.419717$.

**Entropy.** $s$ accumulates $e^{z - m} z$ in the same two steps; at the end $s/\ell = \sum_j p_j z_j = 0.088947 + 0.483566 + 1.971699 - 0.012038 = 2.532174$, so $H = 3.419717 - 2.532174 = 0.887543$ nats. The test `test_hand_worked_example` checks every number above.

## 4. Tensor shapes, dtype, device

| tensor | shape | dtype | notes |
|---|---|---|---|
| `h` (hidden states, flattened) | $(N, C)$ | bf16 or fp32 | $N = B\cdot T$; output of `ln_f` |
| `weight` (LM head) | $(V, C)$ | same as `h` | tied to `wte` in GPT-2 |
| `targets` | $(N,)$ | int64 | `-100` (or `-1` in our GPT) = ignored |
| naive logits | $(N, V)$ | fp32 for the loss | the tensor we want to avoid |
| one row chunk of logits | $(n, V)$ | fp32 | chunked path; $n$ = `chunk_size` |
| one vocabulary tile | $(N, B_v)$ | fp32 | vocabulary-tiled path; a GPU kernel also tiles $N$ |
| `lse`, `m`, `l`, `s` | $(N,)$ | fp32 | the only per-row state kept |
| `grad_h`, `grad_w` | $(N,C)$, $(V,C)$ | as inputs | same size as the inputs, never $(N,V)$ |

Accumulate in fp32 even when `h` and `weight` are bf16: the logsumexp of 150k terms and the sum of $G^\top H$ over thousands of rows lose precision in bf16. Our code uses fp32 accumulators (fp64 if you pass fp64, which the tests do). Device: everything stays on the device of `h`; in our code that is the CPU.

## 5. From-scratch PyTorch

The full file is [`llmre.training.fused_ce`](../../code/src/llmre/training/fused_ce.py). Three functions matter.

**Chunk over positions, gradients in forward.** A custom `torch.autograd.Function` whose forward loops over row chunks, computes the loss *and* the gradients, and saves only `grad_h` and `grad_w`:

```python
for s in range(0, N, chunk_size):
    e = min(s + chunk_size, N)
    hc = h[s:e].to(acc)                       # (n, C)
    z = hc @ w32.t()                          # (n, V)  the only big tensor
    lse = torch.logsumexp(z, dim=-1)          # (n,)
    z_y = z[rows, safe_t[s:e]]                # (n,)
    loss_sum = loss_sum + ((lse - z_y) * v).sum()
    g = torch.exp(z - lse[:, None])           # softmax, (n, V)
    g[rows, safe_t[s:e]] -= 1.0               # minus onehot
    g *= (v / n_valid)[:, None]               # mask + GLOBAL mean
    grad_h[s:e] = g @ w32                     # (n, C)
    grad_w += g.t() @ hc                      # (V, C), accumulated over chunks
```

`backward` only multiplies the saved gradients by the upstream scalar `grad_out`. This works because the loss is the last operation. Computing the gradient early costs nothing extra: backward would have needed the same softmax anyway.

**Tile over the vocabulary, recompute in backward.** The forward pass keeps only `lse`:

```python
m = torch.full((N,), float("-inf")); l = torch.zeros(N)
for j in range(0, V, block_v):
    z = h32 @ weight[j:j + block_v].t()       # (N, Bv) one tile
    m_new = torch.maximum(m, z.max(dim=-1).values)
    l = l * torch.exp(m - m_new) + torch.exp(z - m_new[:, None]).sum(dim=-1)
    m = m_new
lse = m + torch.log(l)
z_y = (h32 * weight[targets]).sum(dim=-1)     # target logit: one dot product per row
```

Its backward loops over the same tiles, recomputes `z`, forms `exp(z - lse) - onehot` restricted to the tile, and accumulates `grad_h += g @ w_t` and writes `grad_w[tile] = g.t() @ h`. That is the FlashAttention bargain again: one extra matmul instead of storing a huge intermediate.

**Scoring.** `fused_logprobs_and_entropy(h, weight, targets, block_v, temperature)` runs the online loop with the extra running sum `s` of section 2.4 and returns per-token `(logprobs, entropy)` under `torch.no_grad()`.

### Run it

```python
import torch, time
from llmre.training.fused_ce import (naive_linear_cross_entropy,
    chunked_linear_cross_entropy, vocab_tiled_linear_cross_entropy, logits_bytes)

torch.manual_seed(0)
N, C, V = 2048, 256, 32000
h = torch.randn(N, C, requires_grad=True)
w = (torch.randn(V, C) * 0.05).requires_grad_(True)
t = torch.randint(0, V, (N,))
for name, fn, kw in [("naive", naive_linear_cross_entropy, {}),
                     ("chunked", chunked_linear_cross_entropy, {"chunk_size": 256}),
                     ("vocab-tiled", vocab_tiled_linear_cross_entropy, {"block_v": 4096})]:
    h.grad = w.grad = None
    t0 = time.perf_counter(); loss = fn(h, w, t, **kw); loss.backward()
    print(name, round(loss.item(), 5), f"{time.perf_counter() - t0:.2f}s", h.grad.norm().item())
print("logits MB:", logits_bytes(N, V) / 1e6, "chunk MB:", logits_bytes(256, V) / 1e6,
      "tile MB:", logits_bytes(N, 4096) / 1e6)
```

Run it from `code/` with `PYTHONPATH=src`. **Expected observations:** all three print the same loss (about 10.686, close to $\log 32000 = 10.37$ plus noise from the random weights) and the same gradient norm. The largest intermediate drops from 262 MB (full logits) to 33 MB (one 256-row chunk, or one 4096-wide tile). On a CPU the timings are of the same order (around a second each) and their order varies by machine. The point of this code is which tensors exist, not speed.

## 6. What production kernels do, and the cost

**Liger Kernel** (LinkedIn; Hsu et al. 2024, [arXiv 2410.10989](https://arxiv.org/abs/2410.10989)) ships `LigerFusedLinearCrossEntropyLoss`, a Triton implementation of the row-chunked scheme of section 5: chunk-by-chunk computation with the gradient computed in the forward pass. Its README reports up to 80% memory reduction for the fused-linear kernels, and the paper reports about 60% lower memory and 20% higher throughput for its kernel set overall against Hugging Face baselines **(publicly documented, authors' measurements)**. Its README lists integrations with the Hugging Face Trainer, TRL's `SFTTrainer`, Axolotl, LLaMA-Factory and others **(publicly documented)**.

**Cut Cross-Entropy** (Apple; Wijmans et al., ICLR 2025 oral, [arXiv 2411.09009](https://arxiv.org/abs/2411.09009)) implements the vocabulary-tiled scheme: it computes only the target logit and evaluates the logsumexp on the fly in SRAM, never writing logits to global memory. It also skips gradient entries too small to matter at the working precision. For Gemma 2 2B the paper reports loss-computation memory falling from about 24 GB to about 1 MB, and classifier-head training memory from 28 GB to 1 GB, without slowing training or hurting convergence **(publicly documented, authors' measurements)**.

**TRL v1.15.0** (Hugging Face, released 2026-10-08, [release notes](https://github.com/huggingface/trl/releases/tag/v1.15.0)) added a fused LM head to its SFT, DPO, KTO, GRPO, RLOO and Distillation trainers: "a Triton kernel projects the hidden states through the LM head in tiles and reduces to per-token log-probs and entropy directly, so the `[batch, seq, vocab]` logits tensor is never built". It is on by default. On gemma-3-1b (262k vocabulary, one 79 GiB GPU, bf16, batch size 1), the notes report maximum trainable sequence length rising 5.8× for DPO, 6.9× for KTO, 4.0× for GRPO scoring and 4.3× for RLOO scoring compared with v1.14.2, and peak memory at 8,192 tokens dropping 52–82%. SFT's existing default `chunked_nll` loss was already memory-efficient and unchanged **(publicly documented, maintainer measurements; the notes say multi-GPU and real generation were not measured)**. This is section 2.4's scoring path, in production.

### Cost

FLOPs: the LM-head matmul costs $2NCV$ in forward and $4NCV$ in backward (two matmuls, for $\partial H$ and $\partial W$), whichever scheme you use. Then:

| scheme | extra peak memory (beyond inputs and grads) | extra FLOPs | HBM traffic on a GPU |
|---|---|---|---|
| naive (06.1) | $(N, V)$ logits + $(N, V)$ saved softmax/grad, fp32 | none | writes and reads $N V$ floats several times |
| row-chunked (Liger) | one $(n, V)$ chunk | none (backward work moved into forward) | $nV$ per chunk, written and read once |
| vocabulary-tiled (CCE) | $O(N)$ statistics; tiles live in SRAM | +$2NCV$ (recompute in backward) | almost none for logits |

Worked numbers for $N = 32{,}768$, $V = 151{,}936$, fp32: naive logits are 19.9 GB. A 1,024-row chunk is $1024 \times 151{,}936 \times 4 = 622$ MB. The $(N,)$ statistics are $32{,}768 \times 4 = 131$ KB each. The softmax and logsumexp are memory-bound ([08.2](lessons/module-08/lesson-02.md)): at a few FLOPs per element they are limited by bytes moved, so avoiding the round trip of $NV$ floats through HBM is also why these kernels are usually *faster*, not only smaller.

**Side effects to know.** The row-chunked scheme computes $\partial W$ in forward, so it must know the loss is final. Anything applied to the loss later (a weight, a second loss term on the same logits) must be folded into the kernel. Logit soft-capping (Gemma 2), z-loss, label smoothing and temperature all change $g_{ij}$, so the kernel has to implement them explicitly. TRL's notes say its fused head forces Liger's `fused_linear_cross_entropy` off, because both try to own the same LM-head computation.

## 7. Side by side, and when to keep the original

| | naive `lm_head` + `F.cross_entropy` (06.1) | row-chunked | vocabulary-tiled |
|---|---|---|---|
| result | exact | exact (same up to rounding order) | exact (same up to rounding order) |
| needs a custom backward | no (autograd) | yes (grads in forward) | yes (recompute) |
| full logits available | yes | per chunk only | never |
| good when | small $V$, short $T$, debugging, generation | simple, robust default | largest $V$ and $N$, minimal memory |

Keep the original when you actually need the logits: generation (sampling, top-$k$, top-$p$), logging the full distribution, storing a teacher's top-$k$ logits for offline distillation ([17.4](lessons/module-17/lesson-04.md)), or a model small enough that the logits are cheap (our GPT-2 runs in Modules 6–7). It is also the reference every fused kernel is tested against, as `test_fused_ce.py` does here.

## Exercise: add z-loss to the vocabulary-tiled path

PaLM's auxiliary **z-loss** adds $\lambda\,\operatorname{lse}_i^2$ per valid row (with $\lambda \approx 10^{-4}$) to keep the log-normalizer near zero. Extend `_VocabTiledLinearCE` with a `z_loss` coefficient. Forward is easy. Derive the change to $g_{ij}$, implement it in backward, and test against `F.cross_entropy(h @ W.T, y) + lam * (torch.logsumexp(h @ W.T, -1)**2).mean()` (no ignored rows).

<details><summary>Hint</summary>

The forward only needs `lse`, which you already have. For backward, differentiate $\lambda\,\operatorname{lse}_i^2$ with respect to $z_{ij}$ using the chain rule and section 2.2's $\partial \operatorname{lse}_i / \partial z_{ij}$.

</details>

<details><summary>Stronger hint</summary>

$\partial(\lambda\,\operatorname{lse}_i^2)/\partial z_{ij} = 2\lambda\,\operatorname{lse}_i\,p_{ij}$. So the per-row factor in front of $p_{ij}$ changes from $1$ to $1 + 2\lambda\,\operatorname{lse}_i$, and the onehot term is unchanged. You need `lse` in backward, and it is already saved.

</details>

<details><summary>Solution</summary>

Forward: `loss = (((lse - z_y) + lam * lse**2) * valid).sum() / n_valid`. Backward, inside the tile loop, replace `g = torch.exp(z - lse[:, None])` with `g = torch.exp(z - lse[:, None]) * (1 + 2 * lam * lse)[:, None]`, then subtract the onehot and scale exactly as before. The $(N,)$ factor `1 + 2*lam*lse` is computed once outside the loop. Check: with `lam=0` you recover the original path.

</details>

## Debugging exercise

A colleague's chunked loss matches `F.cross_entropy` on unmasked data but is wrong on SFT batches where prompts are masked with `-100`:

```python
loss = 0.0
for s in range(0, N, n):
    chunk = F.cross_entropy(h[s:s+n] @ W.t(), y[s:s+n], ignore_index=-100)  # mean per chunk
    loss = loss + chunk
loss = loss / math.ceil(N / n)
```

What is wrong, and when does it happen to give the right answer?

<details><summary>Answer</summary>

Each chunk is averaged over *its own* valid count, then chunks are averaged with equal weight. That is the mean of means, which equals the global mean only when every chunk has the same number of valid tokens. With prompt masking, a chunk that is mostly prompt has few valid tokens, and each of them is over-weighted. A chunk that is all prompt returns NaN (0/0) and poisons the loss. Fix: sum per-row losses (`reduction="sum"`) and divide once by the global valid count, as `chunked_linear_cross_entropy` does (`g *= (v / n_valid)`). This is the same "normalize by what?" question as Dr. GRPO's length normalization ([F.2](lessons/frontier/update-02.md)) and gradient accumulation ([07.2](lessons/module-07/lesson-02.md)).

</details>

## Common mistakes

- **Per-chunk mean instead of global mean** (above).
- **Forgetting to accumulate $\partial W$ across chunks.** Each chunk adds $G_{\text{chunk}}^\top H_{\text{chunk}}$; overwriting instead of adding keeps only the last chunk's gradient.
- **Weight tying.** With tied embeddings the LM-head gradient must be *added* to the embedding gradient. A custom `Function` that returns `grad_w` for the `lm_head.weight` parameter gets this right through autograd. Code that writes `.grad` by hand can silently overwrite the embedding's gradient.
- **bf16 accumulation.** Summing $e^{z-m}$ over 150k vocabulary entries, or $G^\top H$ over thousands of rows, in bf16 loses digits. Accumulate in fp32.
- **Ignoring the upstream gradient.** If the loss is multiplied (loss scaling, gradient accumulation by `loss / k`), the precomputed gradients must be multiplied by `grad_out`. The test uses `2.5 * loss` to catch this.
- **Silently dropping a loss modifier.** Temperature, soft-capping, z-loss and label smoothing must be inside the kernel. A trainer that applies them to logits that no longer exist will either error out or quietly skip them.

## What to read / what to skip

- **Cut Cross-Entropy** ([arXiv 2411.09009](https://arxiv.org/abs/2411.09009)): read the method section on computing the loss as an indexed matmul plus a blockwise logsumexp, and the memory table. Skim the gradient-filtering and vocabulary-sorting details unless you write kernels.
- **Liger Kernel** ([arXiv 2410.10989](https://arxiv.org/abs/2410.10989)): read the fused linear cross-entropy section (chunking, gradient in forward); skip the per-kernel benchmarks for RMSNorm, RoPE and SwiGLU.
- **TRL v1.15.0 release notes** ([GitHub](https://github.com/huggingface/trl/releases/tag/v1.15.0)): read the fused LM head section and its measurement table, including what was not measured.

## Code this after reading

Make `fused_logprobs_and_entropy` differentiable with respect to `h` and `weight` for the log-prob output, which is what a GRPO loss needs ([16.1](lessons/module-16/lesson-01.md)). The gradient of $\log p_{iy_i}$ is $-(p_{ij} - \mathbb{1}[j = y_i])$, the negative of the cross-entropy gradient, times the upstream per-token gradient. Test it against `torch.log_softmax(h @ W.T, -1).gather(...)` with `torch.autograd.grad`.

## Research connection

<div class="callout paper"><p><strong>Cut Your Losses in Large-Vocabulary Language Models</strong> (Cut Cross-Entropy) — Wijmans, Huval, Hertzberg, Koltun, Krähenbühl, ICLR 2025, <a href="https://arxiv.org/abs/2411.09009">arXiv 2411.09009</a>. The vocabulary-tiled scheme of this lesson, implemented as an SRAM-resident kernel. <strong>Liger Kernel: Efficient Triton Kernels for LLM Training</strong> — Hsu et al. 2024, <a href="https://arxiv.org/abs/2410.10989">arXiv 2410.10989</a>. The row-chunked scheme, with gradients computed in the forward pass. Production default: <a href="https://github.com/huggingface/trl/releases/tag/v1.15.0">TRL v1.15.0 release notes</a> (fused LM head producing per-token log-probs and entropy across six trainers). Background: FlashAttention (paper #7, <a href="#/papers/index">paper index</a>) for the online-softmax recurrence reused here.</p></div>

**Documented vs inferred.** *Publicly documented:* the CCE and Liger methods and their reported savings (authors' measurements); Liger's listed framework integrations; TRL v1.15.0's fused LM head, its default status and its reported numbers (maintainer measurements). *Reasonable industry practice:* fp32 accumulation inside these kernels; never materializing logits in long-context or large-vocabulary training. *Inference:* that a given closed lab uses one specific scheme. Labs rarely say, and the schemes give the same numbers.

## Check yourself

<details><summary>Why can the row-chunked scheme compute $\partial\mathcal L/\partial W$ during the forward pass?</summary>

Because $g_{ij} = (p_{ij} - \mathbb 1[j=y_i])/|\mathcal V|$ depends only on that row's logits and the global valid count, both available as soon as the chunk's logits exist. $\partial W = G^\top H$ is a sum over rows, so each chunk's contribution can be added right away. The only assumption is that the loss is not transformed afterwards except by a scalar, which `grad_out` handles.

</details>

<details><summary>The vocabulary-tiled backward recomputes every logit tile. Why is that a good trade?</summary>

The recompute costs one extra $2NCV$ matmul, which is compute-bound and fast on tensor cores. Storing the logits would cost $NV$ floats of memory plus HBM round trips for a memory-bound softmax. Trading cheap FLOPs for scarce bytes is 08.2's roofline argument and FlashAttention's choice (08.4).

</details>

<details><summary>How do you get the target logit without computing all $V$ logits?</summary>

$z_{iy_i} = h_i \cdot w_{y_i}$: gather row $y_i$ of $W$ and take one $C$-dimensional dot product per position, i.e. $O(NC)$ work instead of $O(NCV)$.

</details>

<details><summary>Write the three running quantities needed for per-token entropy and say how each is updated when the max rises.</summary>

$m$ (running max), $\ell = \sum e^{z - m}$ and $s = \sum e^{z-m} z$. When the max rises from $m_{\text{old}}$ to $m_{\text{new}}$, both $\ell$ and $s$ are multiplied by $\alpha = e^{m_{\text{old}} - m_{\text{new}}}$ before the new tile's terms are added; $m$ becomes $m_{\text{new}}$. At the end $\operatorname{lse} = m + \log\ell$ and $H = \operatorname{lse} - s/\ell$.

</details>

<details><summary>For $N = 32{,}768$ tokens and $V = 151{,}936$, how big are the fp32 logits, and how big is a 1,024-row chunk?</summary>

$32{,}768 \times 151{,}936 \times 4 \approx 19.9$ GB; $1{,}024 \times 151{,}936 \times 4 \approx 622$ MB, a 32× reduction. Vocabulary tiling keeps $O(N)$ statistics, about 131 KB per statistic.

</details>

<details><summary>Name a situation where you should not use a fused cross-entropy.</summary>

When you need the full distribution: sampling during generation, logging or storing top-$k$ logits (e.g. offline distillation targets), or debugging a loss. Also when the model is small enough that logits are cheap, where the plain `F.cross_entropy` is simpler and is the reference implementation.

</details>

## Next

You now have both halves of the "never materialize the big intermediate" pattern: FlashAttention for the $T \times T$ scores (08.4) and fused cross-entropy for the $N \times V$ logits (this lesson). The same trade, a little recomputation for far fewer bytes, scales up from one GPU to many in the next module.

Continue to [09.1 · Data parallelism, all-reduce, DDP](lessons/module-09/lesson-01.md).
