# 05.4 · MLP, residual, LayerNorm, the block

<div class="prereq">
<p><strong>Prerequisites:</strong> the complete <code>CausalSelfAttention</code> module from <a href="#/lessons/module-05/lesson-03">05.3 · Multi-head attention</a>; the <code>(B, T, C)</code> convention and broadcasting from <a href="#/lessons/module-00/lesson-02">Module 0</a>; linear layers and the chain rule / gradient-highway intuition from <a href="#/lessons/module-02/lesson-03">Module 2</a>.</p>
<p><strong>You will learn:</strong> the three remaining pieces of a transformer block — the position-wise <strong>MLP</strong> (why the 4× hidden expansion and the GELU nonlinearity), the <strong>residual connection</strong> (why <code>x + sublayer(x)</code> is what makes deep stacks trainable), and <strong>LayerNorm</strong> (its formula, worked by hand and matched to PyTorch) — then how they assemble with attention into the <strong>pre-norm <code>Block</code></strong> that GPT stacks <code>n_layer</code> times.</p>
<p><strong>Why this matters for ML:</strong> a GPT is <em>literally</em> an embedding, a stack of identical blocks, a final norm, and an output head. Once you can build one block, you have built the whole model up to bookkeeping. And the pre-norm + residual structure you learn here is the reason 100-layer transformers train at all — it is not a detail, it is the load-bearing wall.</p>
</div>

## 1. The shape of a block

A transformer block takes an activation of shape `(B, T, C)` and returns the same shape, so
blocks stack. Inside, it does two things in sequence, each wrapped in a normalization and a
residual add:

```text
x = x + attn(ln_1(x))     # 1. communication: positions exchange information (Module 5.1–5.3)
x = x + mlp(ln_2(x))      # 2. computation: each position is transformed on its own
```

Attention is the *only* place information moves **between** positions. The MLP is where each
position is **individually** transformed. Alternating "mix across positions, then think per
position" — that is the entire computational rhythm of a transformer. We already have `attn`;
this lesson builds `mlp`, `ln`, and the residual `+`.

## 2. The position-wise MLP

### Intuition and math

After attention has gathered context into each position's vector, the MLP gives the model a
chance to do nonlinear computation on that vector — independently at every position, with the same
weights shared across positions. It is two linear layers with a nonlinearity between them:

$$
\mathrm{MLP}(x) = W_2 \,\phi(W_1 x + b_1) + b_2,
$$

where $W_1 \in \mathbb{R}^{4C \times C}$ expands the width from $C$ to $4C$, $\phi$ is the GELU
nonlinearity, and $W_2 \in \mathbb{R}^{C \times 4C}$ projects back down to $C$.

**Why 4×?** The hidden layer is where the block's per-token "capacity" lives; GPT-2 and most
descendants use a 4× expansion as the standard capacity/compute trade-off. **Why a
nonlinearity?** Without $\phi$, two stacked linear maps $W_2 W_1$ collapse into a single linear
map — the block would gain nothing from having two layers. $\phi$ is what makes the MLP able to
represent nonlinear functions of the input.

**GELU** (Gaussian Error Linear Unit) is a smooth relative of ReLU: $\phi(z) = z\,\Phi(z)$ where
$\Phi$ is the standard-normal CDF. It passes large positive values almost unchanged and squashes
negatives smoothly toward zero. A single value: $\mathrm{GELU}(1.0) \approx 0.8413$ (it keeps most
of the input; ReLU would keep all of it, and a hard step none of the smoothness).

### Tensor shapes

For input `(B, T, C)`:

```text
(B, T, C) --W1--> (B, T, 4C) --GELU--> (B, T, 4C) --W2--> (B, T, C)
```

The linear layers act on the **last axis only**; the `(B, T)` positions are all processed in
parallel with the same weights. The hidden activation `(B, T, 4C)` is 4× the size of the input —
worth remembering when we account for activation memory in [Module 7](lessons/module-07/lesson-04.md).

### From scratch (the course codebase)

`code/src/llmre/model/block.py`:

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, cfg):                       # cfg: GPTConfig
        super().__init__()
        self.c_fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)  # C -> 4C
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # 4C -> C
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):                           # x: (B, T, C)
        x = self.c_fc(x)                            # (B, T, 4C)
        x = self.gelu(x)                            # (B, T, 4C)
        x = self.c_proj(x)                          # (B, T, C)
        return self.dropout(x)
```

## 3. Residual connections

### Why add the input back

A block does not replace `x` with `sublayer(x)`; it computes `x + sublayer(x)`. That single `+`
is a **residual (skip) connection**, and it is what lets us stack dozens of blocks.

Consider the backward pass. The gradient of the loss w.r.t. the block's input is, by the chain
rule through `y = x + f(x)`:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\left(I + \frac{\partial f}{\partial x}\right)
= \frac{\partial L}{\partial y} + \frac{\partial L}{\partial y}\frac{\partial f}{\partial x}.
$$

The identity term means the upstream gradient $\partial L/\partial y$ flows to $x$ **undiminished**,
plus a correction from the sublayer. Without the skip, the gradient would be multiplied by
$\partial f/\partial x$ at every layer, and across many layers those factors compound toward zero
(vanishing gradients) or blow up (exploding). The residual gives gradients a clean "highway"
straight from the loss back to every layer, so even a very deep stack trains.

<div class="callout key"><p>A residual connection <code>x + f(x)</code> adds an identity path.
On the forward pass each sub-layer only has to learn a <em>correction</em> to <code>x</code>; on
the backward pass the identity term carries the gradient through unchanged. This is the single
most important trick for training deep networks.</p></div>

## 4. LayerNorm

### Intuition and math

As activations flow through many layers, their scale drifts — some feature vectors grow large,
others shrink — which makes optimization unstable. **LayerNorm** re-centers and re-scales each
position's feature vector to zero mean and unit variance, *independently per position and per
sequence*, then applies a learned per-channel scale $\gamma$ and shift $\beta$ so the network can
undo the normalization if it needs to:

$$
\mathrm{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta,
\qquad
\mu = \frac{1}{C}\sum_{i=1}^{C} x_i,
\qquad
\sigma^2 = \frac{1}{C}\sum_{i=1}^{C}(x_i - \mu)^2.
$$

The mean and variance are taken over the **last axis** (the $C$ channels) only — not over the
batch, which is what distinguishes LayerNorm from BatchNorm and makes it independent of batch size
(crucial for language models, where sequences vary). $\epsilon$ (default $10^{-5}$) prevents
division by zero. Note the **biased** (population) variance — divide by $C$, not $C-1$ — to match
`torch.nn.LayerNorm`.

### Numerical example (verified against PyTorch)

Take one length-4 feature vector $x = [2, 4, 4, 6]$, with $\gamma = 1$, $\beta = 0$:

- Mean: $\mu = (2+4+4+6)/4 = 4$.
- Biased variance: $\sigma^2 = \frac{(-2)^2 + 0^2 + 0^2 + 2^2}{4} = \frac{8}{4} = 2$, so
  $\sigma = \sqrt{2} \approx 1.4142$.
- Normalize: $\hat x = \frac{[2,4,4,6] - 4}{\sqrt{2 + 10^{-5}}}
  = [-2, 0, 0, 2]/1.4142 \approx [-1.4142,\ 0,\ 0,\ 1.4142]$.

```python
import torch, torch.nn as nn
x  = torch.tensor([2., 4., 4., 6.])
ln = nn.LayerNorm(4)                     # gamma=1, beta=0 at init
ln(x)   # tensor([-1.4142,  0.0000,  0.0000,  1.4142])  -> matches by hand
```

### From scratch, and matching PyTorch

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))          # gamma, (C,)
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):                                      # x: (..., C)
        mean = x.mean(dim=-1, keepdim=True)                    # (..., 1)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)     # biased, (..., 1)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        out  = xhat * self.weight
        return out if self.bias is None else out + self.bias
```

`keepdim=True` keeps the reduced axis as length 1 so it **broadcasts** back against `x`. A test in
`code/tests/test_block.py` asserts this matches `torch.nn.LayerNorm` to $10^{-6}$.

<div class="callout warn"><p>The most common LayerNorm bug is reducing over the wrong axis
(e.g. the batch) or using the unbiased variance (<code>unbiased=True</code>, dividing by
<code>C-1</code>). Both give subtly wrong values that still "look normalized" — always normalize
over the last (channel) axis with the biased variance.</p></div>

## 5. Pre-norm vs post-norm, and the block

Where does the norm go — before the sublayer (**pre-norm**) or after the residual add
(**post-norm**)? The original 2017 Transformer used post-norm; modern LLMs (GPT-2 onward) use
**pre-norm** because it keeps the residual path free of any normalization, so the gradient highway
of §3 stays perfectly clean and deep stacks train more stably without delicate warmup. The block:

```python
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)     # from 05.3
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp  = MLP(cfg)

    def forward(self, x):                          # x: (B, T, C) -> (B, T, C)
        x = x + self.attn(self.ln_1(x))            # normalize, attend, add back
        x = x + self.mlp(self.ln_2(x))             # normalize, transform, add back
        return x
```

Notice the norm is applied to a *copy* fed into the sublayer, while the raw `x` continues down the
residual path untouched — that is the whole point of pre-norm. Output shape equals input shape,
`(B, T, C)`, so `Block` is stackable.

## 6. Under the hood: parameter and compute cost of one block

For hidden width $C$, one block's parameters are dominated by four weight matrices:

- Attention QKV projection: $C \times 3C$, and output projection $C \times C$ → $4C^2$.
- MLP: $C \times 4C$ and $4C \times C$ → $8C^2$.

So a block has $\approx 12C^2$ parameters (the two LayerNorms add only $\sim 4C$, negligible). For
GPT-2 small, $C = 768$, that is about $12 \cdot 768^2 \approx 7.1$M parameters per block, times
12 blocks ≈ 85M — the bulk of the model's ~124M (the rest is the embeddings). We will do the full
parameter count in [06.2](lessons/module-06/lesson-02.md). On compute: the MLP's $8C^2$ FLOPs-per-token
usually dominates a block until the sequence gets long enough that attention's $T^2$ term
(from the score matrix of [05.2](lessons/module-05/lesson-02.md)) takes over — the crossover that
motivates FlashAttention in [Module 8](lessons/module-08/lesson-04.md).

## Exercise

You stack this `Block` 12 times but accidentally write the forward pass as
`x = self.attn(self.ln_1(x))` (dropping the `x +`). Training a 12-layer model now diverges or
learns nothing. Explain, in terms of the backward pass, why removing the residual add breaks a
deep stack — and why a 1-layer model might still seem to train.

<details><summary>Hint</summary>
Write the gradient of the loss w.r.t. the block input with and without the identity term, then
imagine multiplying that factor together 12 times.
</details>

<details><summary>Solution</summary>

With the residual, $\partial L/\partial x = \partial L/\partial y\,(I + \partial f/\partial x)$ —
the identity carries the gradient through undiminished. Without it, $\partial L/\partial x =
\partial L/\partial y \cdot \partial f/\partial x$: the gradient is multiplied by the sublayer's
Jacobian at *every* layer. Across 12 layers those 12 factors compound; if their singular values
are below 1 the gradient vanishes toward the early layers (they stop learning), and if above 1 it
explodes (NaNs). A single layer has only one such factor, so the effect is mild and the model may
appear to train — which is exactly why this bug hides until you go deep.

</details>

## Check yourself

<details><summary>What is the shape of the MLP's hidden activation for input <code>(4, 128, 768)</code>?</summary>

`(4, 128, 3072)` — the first linear expands the last axis from $C = 768$ to $4C = 3072$; batch and
time are unchanged.

</details>

<details><summary>LayerNorm on <code>x = [1, 1, 1, 5]</code> with γ=1, β=0: give μ, σ² (biased), and the normalized vector's largest entry (approx).</summary>

$\mu = 8/4 = 2$; $\sigma^2 = \frac{1+1+1+9}{4} = 3$; $\sigma = \sqrt3 \approx 1.732$. Largest entry:
$(5-2)/1.732 \approx 1.732$.

</details>

<details><summary>Why does LayerNorm normalize over the channel axis and not the batch axis?</summary>

So each position's normalization is independent of the other examples in the batch and of the
sequence length. This makes it work for variable-length sequences and for batch size 1, unlike
BatchNorm whose statistics depend on the batch.

</details>

<details><summary>In pre-norm, does the normalization sit on the residual highway or off it?</summary>

Off it. The norm is applied only to the copy fed into the sublayer (`ln_1(x)` → `attn`); the raw
`x` flows down the residual add untouched, keeping the gradient path clean.

</details>

## Next

You now have every component: token + positional embeddings ([05.1](lessons/module-05/lesson-01.md)), causal
multi-head attention ([05.2](lessons/module-05/lesson-02.md)–[05.3](lessons/module-05/lesson-03.md)), and the pre-norm `Block` with
its MLP, residuals, and LayerNorm. The next module assembles them into the full **GPT-2 model** —
the embedding tables, a stack of `n_layer` blocks, a final LayerNorm, the language-model head,
weight tying, and initialization — then runs a forward pass end to end.

Continue to [06.1 · Assembling GPT-2](lessons/module-06/lesson-01.md).
