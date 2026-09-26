# 12.2 · RMSNorm &amp; SwiGLU

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-05/lesson-04">05.4 · MLP, residual, LayerNorm, the block</a> (LayerNorm's mean/variance normalization and the $C\to 4C\to C$ GELU MLP); the $(B, T, C)$ activation shape; the GELU nonlinearity.</p>
<p><strong>You will learn:</strong> how <strong>RMSNorm</strong> drops LayerNorm's mean-subtraction and bias, normalizing by the root-mean-square alone; a worked contrast of RMSNorm vs LayerNorm on the same vector; how the <strong>SwiGLU</strong> gated feed-forward network replaces the GELU MLP, and why LLaMA shrinks its hidden width to $\tfrac{8}{3}C$ to keep the parameter count equal.</p>
<p><strong>Why this matters for ML:</strong> RMSNorm + SwiGLU are, with RoPE, the three changes that define the LLaMA-family architecture. Every serious open model since 2023 uses them. They cost fewer FLOPs than the GPT-2 originals and train at least as well.</p>
</div>

## Part A — RMSNorm

## 1. Intuition: normalization, minus the centering

LayerNorm (Module 5) does two adjustments to each length-$C$ feature vector, then a learned affine:

1. **Re-center**: subtract the mean, so the vector is centered on zero.
2. **Re-scale**: divide by the standard deviation, so it has unit spread.
3. **Affine**: multiply by a learned gain $\gamma$ and add a learned bias $\beta$.

RMSNorm (Zhang &amp; Sennrich 2019) asks: is the *centering* actually pulling its weight? Empirically, for transformers, no — the re-scaling is what stabilizes training; subtracting the mean barely matters. So RMSNorm keeps only the re-scaling and drops the mean-subtraction *and* the bias:

$$
\mathbf{y} = \frac{\mathbf{x}}{\operatorname{RMS}(\mathbf{x})}\odot\mathbf{g},
\qquad
\operatorname{RMS}(\mathbf{x}) = \sqrt{\frac{1}{C}\sum_{i=1}^{C} x_i^2 + \epsilon}.
$$

That is it: divide by the root-mean-square of the components, then apply a learned per-channel gain $\mathbf{g}$ (initialized to ones). No mean, no bias, one reduction instead of two.

<div class="callout key"><p>LayerNorm: $\;y = \dfrac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma + \beta$. &nbsp; RMSNorm: $\;y = \dfrac{x}{\sqrt{\frac{1}{C}\sum x_i^2 + \epsilon}}\,g$. &nbsp; RMSNorm removes $\mu$ (centering) and $\beta$ (bias) and normalizes by RMS instead of standard deviation.</p></div>

## 2. Mathematics: RMS vs standard deviation

The only numerical difference is what sits under the square root. Standard deviation measures spread *around the mean*: $\sigma^2 = \frac{1}{C}\sum_i (x_i - \mu)^2$. RMS measures magnitude *around zero*: $\operatorname{RMS}^2 = \frac{1}{C}\sum_i x_i^2$. They are related by $\operatorname{RMS}^2 = \sigma^2 + \mu^2$ — RMS equals the standard deviation only when the mean $\mu$ is already zero. So on a zero-mean vector the two normalizers agree; on a vector with a nonzero mean they diverge, and that divergence is exactly the centering RMSNorm chooses to skip. $\epsilon$ (typically $10^{-5}$ or $10^{-6}$) is added under the root to avoid dividing by zero for a near-zero vector.

## 3. Numerical example: RMSNorm vs LayerNorm on $\mathbf{x} = (1,2,3,4)$

Take $C = 4$, $\mathbf{x} = (1, 2, 3, 4)$, gain $\mathbf{g} = \mathbf{1}$, $\epsilon \approx 0$.

**RMSNorm.** Mean of squares: $\frac{1^2+2^2+3^2+4^2}{4} = \frac{1+4+9+16}{4} = \frac{30}{4} = 7.5$. So $\operatorname{RMS} = \sqrt{7.5} = 2.738613$. Divide each component:

$$
\mathbf{y}_{\text{RMS}} = \frac{(1,2,3,4)}{2.738613} = (0.365148,\ 0.730297,\ 1.095445,\ 1.460593).
$$

**LayerNorm.** Mean $\mu = \frac{1+2+3+4}{4} = 2.5$. Variance $\sigma^2 = \frac{(-1.5)^2+(-0.5)^2+(0.5)^2+(1.5)^2}{4} = \frac{2.25+0.25+0.25+2.25}{4} = 1.25$, so $\sigma = 1.118034$. Center then divide:

$$
\mathbf{y}_{\text{LN}} = \frac{(1,2,3,4) - 2.5}{1.118034} = (-1.341641,\ -0.447214,\ 0.447214,\ 1.341641).
$$

The results are completely different. RMSNorm's output is all positive (it never subtracted the mean, so the "3 is bigger than 1" ordering stays on the same side of zero); LayerNorm's is symmetric about zero (it centered first). Both have the right *scale*, which is what the downstream layer needs. All eight numbers are verified against PyTorch in §6.

## 4. Tensor shapes

RMSNorm is shape-preserving, exactly like LayerNorm:

$$
\underbrace{(B, T, C)}_{\text{input}} \;\longrightarrow\; \underbrace{(B, T, C)}_{\text{output}},
$$

with the reduction taken over the **last** axis only (per position, per sequence). The single learned parameter is the gain $\mathbf{g}$ of shape $(C,)$ — half the parameters of LayerNorm, which also carries $\beta$ of shape $(C,)$. Same dtype and device as the input; LLaMA computes the RMS in float32 even when the activations are half precision, for stability.

## 5. From-scratch PyTorch

From [`llmre.model.rmsnorm`](../../code/src/llmre/model/rmsnorm.py):

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # gain g, (C,)

    def _norm(self, x):
        # rsqrt(mean(x^2) + eps) computed in float32, then cast back.
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x)

    def forward(self, x):
        return self._norm(x) * self.weight            # (..., C) -> (..., C)
```

`torch.rsqrt(z)` is $1/\sqrt{z}$ in one op (a reciprocal square root), so we multiply instead of divide. `mean(dim=-1, keepdim=True)` reduces the $C$ axis to size 1 and keeps it for broadcasting. There is no `mean` subtraction and no bias tensor anywhere — that is the entire difference from the LayerNorm you built in Module 5.

## 6. Under the hood: verify, and count the cost

```python
import torch
x = torch.tensor([1., 2., 3., 4.])

rms = torch.sqrt((x**2).mean())          # 2.7386
print((x / rms).tolist())
# [0.3651, 0.7303, 1.0954, 1.4606]  -> RMSNorm

mu, sigma = x.mean(), x.std(unbiased=False)   # 2.5, 1.1180
print(((x - mu) / sigma).tolist())
# [-1.3416, -0.4472, 0.4472, 1.3416]  -> LayerNorm
```

**Cost.** RMSNorm does one reduction (sum of squares) where LayerNorm does two (mean, then variance around it), and skips the elementwise subtract and the bias add. The FLOP saving per call is tiny in isolation, but a deep model applies normalization $2 n_{\text{layer}} + 1$ times per forward pass, and the memory saving is real: no $\beta$ parameter, and half the norm-related activations to keep for the backward pass. The main reason it caught on is simply that it works as well — the centering was doing little, so removing it is free.

<div class="callout pt"><p>PyTorch shipped <code>torch.nn.RMSNorm</code> in recent versions, and there are fused CUDA kernels for it. As always in this course, we implement the mechanism first so you know exactly what the fused kernel computes: one reduction, one reciprocal-sqrt, one elementwise multiply by the gain.</p></div>

## Part B — SwiGLU

## 7. Intuition: a gated feed-forward network

The GPT-2 MLP is `Linear(C -> 4C) -> GELU -> Linear(4C -> C)`: expand to a wide hidden layer, apply a pointwise nonlinearity, project back. SwiGLU (Shazeer 2020, "GLU Variants Improve Transformer") keeps the expand-and-project shape but replaces the single nonlinear path with a **gate**. It computes *two* projections of the input and multiplies them together elementwise, letting one path modulate (gate) the other:

$$
\operatorname{SwiGLU}(\mathbf{x}) = \Big(\operatorname{Swish}(\mathbf{x}W_{\text{gate}}) \odot (\mathbf{x}W_{\text{up}})\Big)\,W_{\text{down}},
\qquad
\operatorname{Swish}(z) = z\,\sigma(z),
$$

where $\sigma$ is the logistic sigmoid and $\odot$ is elementwise multiply. There are **three** weight matrices: $W_{\text{gate}}$ and $W_{\text{up}}$ both map $C\to h$ (the hidden width), and $W_{\text{down}}$ maps $h\to C$. The "gate" is the Swish-activated $\mathbf{x}W_{\text{gate}}$; it scales the "up" signal $\mathbf{x}W_{\text{up}}$ channel by channel. A gate value near $0$ shuts a hidden channel off; a large gate lets it through amplified. This data-dependent gating is more expressive than a single fixed nonlinearity, and gated FFNs consistently win on language modeling loss at equal size.

Swish (also called SiLU) is a smooth version of ReLU: near-zero for very negative inputs, linear for large positive ones, but differentiable everywhere and slightly negative just below zero. `F.silu(z)` computes it directly.

## 8. Mathematics &amp; the $\tfrac{8}{3}$ hidden width

SwiGLU uses three matrices where the GELU MLP uses two, so at equal hidden width it would have $1.5\times$ the parameters. To keep a fair comparison — same parameter count, same FLOPs — LLaMA shrinks the hidden width from $4C$ to

$$
h = \frac{8}{3}C.
$$

Check the parameter counts (ignore biases; LLaMA drops them). The GELU MLP has $W_1\in\mathbb{R}^{C\times 4C}$ and $W_2\in\mathbb{R}^{4C\times C}$:

$$
\text{GELU MLP} = C\cdot 4C + 4C\cdot C = 8C^2.
$$

SwiGLU has three matrices, each $C\times h$ or $h\times C$, with $h = \tfrac{8}{3}C$:

$$
\text{SwiGLU} = 3\,(C\cdot h) = 3\cdot C\cdot\frac{8}{3}C = 8C^2.
$$

They match exactly: $3\cdot\tfrac{8}{3} = 8 = 2\cdot 4$. That is where the odd-looking $\tfrac{8}{3}$ comes from — it is the width that makes a three-matrix gated FFN cost the same as the two-matrix $4C$ FFN it replaces. In practice $h$ is rounded to a hardware-friendly multiple (LLaMA rounds to a multiple of 256).

## 9. Numerical example: Swish and the width identity

**Swish values** (verified in §11):

| $z$ | $-2$ | $-1$ | $0$ | $1$ | $2$ |
|-----|------|------|-----|-----|-----|
| $\operatorname{Swish}(z) = z\,\sigma(z)$ | $-0.2384$ | $-0.2689$ | $0.0000$ | $0.7311$ | $1.7616$ |

Notice Swish dips slightly negative for small negative $z$ (a ReLU would be flat zero there) and tracks $z$ for large positive $z$.

**Width identity at $C = 768$** (GPT-2 small's width). GELU MLP weights: $2\cdot 768\cdot(4\cdot 768) = 2\cdot 768\cdot 3072 = 4{,}718{,}592$. SwiGLU hidden $h = \lfloor \tfrac{8}{3}\cdot 768\rfloor = 2048$; weights: $3\cdot 768\cdot 2048 = 4{,}718{,}592$. Identical — the two FFNs have exactly the same parameter budget.

## 10. Tensor shapes &amp; from-scratch PyTorch

$$
\underbrace{(B, T, C)}_{\mathbf{x}}
\xrightarrow{W_{\text{gate}},\,W_{\text{up}}}
\underbrace{(B, T, h)}_{\text{gate, up}}
\xrightarrow{\odot}
(B, T, h)
\xrightarrow{W_{\text{down}}}
\underbrace{(B, T, C)}_{\text{output}}.
$$

From [`llmre.model.swiglu`](../../code/src/llmre/model/swiglu.py):

```python
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None, bias=False, multiple_of=1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * dim / 3)                       # (8/3) C
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)     # C -> h
        self.w_up   = nn.Linear(dim, hidden_dim, bias=bias)     # C -> h
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)     # h -> C

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

`F.silu` is exactly Swish with $\beta = 1$: $\operatorname{silu}(z) = z\,\sigma(z)$. The elementwise `*` between the two $(B, T, h)$ activations is the gate.

## 11. Under the hood: verify

```python
import torch, torch.nn.functional as F
z = torch.tensor([-2., -1., 0., 1., 2.])
print(F.silu(z).tolist())
# [-0.2384, -0.2689, 0.0, 0.7311, 1.7616]   -> Swish/SiLU

d = 768
gelu  = 2 * d * (4 * d)          # 4,718,592
h     = int(8 * d / 3)           # 2048
swiglu = 3 * d * h               # 4,718,592
print(h, gelu, swiglu, gelu == swiglu)   # 2048 4718592 4718592 True
```

**Cost.** SwiGLU runs two input projections and one output projection instead of two total matmuls, but at the reduced width $\tfrac{8}{3}C$ the total FLOPs and parameters land on the same $8C^2$ as the GELU MLP. So the win is quality at equal cost, not a cost reduction. Memory-wise the backward pass must keep both the gate and up activations (each $(B,T,h)$), a little more than the single hidden activation of the GELU MLP.

## Common mistakes

- **RMSNorm with a bias.** RMSNorm has *no* bias term — only the gain $\mathbf{g}$. Adding a bias makes it a different (and generally worse) normalizer.
- **Using $\sigma$ (std) formulas for RMSNorm.** RMS is the root mean of squares around **zero**, not around the mean. Do not subtract the mean.
- **Keeping the hidden width at $4C$ for SwiGLU.** That inflates the FFN to $12C^2$ parameters ($1.5\times$). Use $\tfrac{8}{3}C$ (rounded) to match the GELU budget.
- **Reusing one matrix for gate and up.** They are two *separate* learned projections; sharing them collapses the gate.

## Exercise

You are told RMSNorm and LayerNorm produce the **same** output (up to the affine parameters) on a particular input vector $\mathbf{x}$. What must be true of $\mathbf{x}$?

<em>Hint:</em> when does RMS equal the standard deviation?

<details><summary>Solution</summary>

$\operatorname{RMS}^2 = \sigma^2 + \mu^2$, so RMS equals the standard deviation exactly when $\mu = 0$. If $\mathbf{x}$ already has zero mean, LayerNorm's centering step does nothing and both normalizers divide by the same quantity, giving the same normalized vector. So the two agree precisely on zero-mean inputs.

</details>

## Debugging exercise

A student implements SwiGLU but writes `self.w_down(F.silu(self.w_gate(x) * self.w_up(x)))` — applying Swish *after* the multiply instead of only to the gate. Why is this not SwiGLU, and what have they built instead?

<details><summary>Answer</summary>

SwiGLU applies the nonlinearity to the gate path *before* the elementwise product: $\operatorname{Swish}(xW_{\text{gate}})\odot(xW_{\text{up}})$. The buggy version computes $\operatorname{Swish}\big((xW_{\text{gate}})\odot(xW_{\text{up}})\big)$ — it multiplies two *linear* projections first (a bilinear form) and then applies Swish once to the product. There is no gating: neither path modulates the other through a nonlinearity. It is a different, weaker function, and it breaks the parameter-matching argument's intent (the gate is gone).

</details>

## Research connection

<div class="callout paper"><p><strong>LLaMA: Open and Efficient Foundation Language Models</strong> (Touvron et al. 2023) bundles RMSNorm, SwiGLU, and RoPE into one reference architecture; see <a href="#/papers/index">the paper curriculum</a> (#9). SwiGLU traces to Shazeer's "GLU Variants Improve Transformer" (2020) and RMSNorm to Zhang &amp; Sennrich (2019). LLaMA's hyperparameter table lists the $\tfrac{8}{3}$-derived FFN widths directly.</p></div>

## Check yourself

<details><summary>Name the two things RMSNorm removes from LayerNorm.</summary>

The mean-subtraction (centering) and the learned bias $\beta$. It keeps the re-scaling — now by RMS instead of standard deviation — and the learned gain.

</details>

<details><summary>On $\mathbf{x} = (1,2,3,4)$, why is RMSNorm's output all positive while LayerNorm's is symmetric about zero?</summary>

RMSNorm never subtracts the mean, so the components keep their signs and their order relative to zero — all four inputs are positive, so all four outputs are positive. LayerNorm subtracts $\mu = 2.5$ first, sending the two below-average entries negative and the two above-average entries positive, symmetric about zero.

</details>

<details><summary>Why does LLaMA use a hidden width of $\tfrac{8}{3}C$ for SwiGLU?</summary>

SwiGLU has three weight matrices vs the GELU MLP's two. At hidden width $\tfrac{8}{3}C$ the SwiGLU parameter count is $3C\cdot\tfrac{8}{3}C = 8C^2$, exactly matching the GELU MLP's $2\cdot C\cdot 4C = 8C^2$. It keeps the parameter and FLOP budget equal for a fair comparison.

</details>

<details><summary>What is $\operatorname{Swish}(0)$, and how does Swish differ from ReLU for small negative inputs?</summary>

$\operatorname{Swish}(0) = 0\cdot\sigma(0) = 0$. For small negative $z$, ReLU is flat zero, but Swish is slightly negative (e.g. $\operatorname{Swish}(-1) = -0.2689$) and smooth — it has a nonzero gradient there, which helps optimization.

</details>

## Next

Position, normalization, and the FFN are now modern. The last structural change is to attention itself — shrinking the key/value heads to cut the inference KV cache. Continue to [12.3 · MQA / GQA](lessons/module-12/lesson-03.md).
