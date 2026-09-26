# 03.2 · RMSProp, Adam, AdamW, weight decay

<div class="prereq">
<p><strong>Prerequisites:</strong> the update rule $\theta \leftarrow \theta - \eta g$, minibatch gradients, and momentum as an EMA of gradients from <a href="#/lessons/module-03/lesson-01">03.1 · Gradient descent, SGD, momentum</a>. The gradient $\nabla L$ and <code>.grad</code> from <a href="#/lessons/module-02/lesson-01">02.1</a>. Elementwise tensor ops and in-place updates from <a href="#/lessons/module-00/lesson-01">00.1</a>.</p>
<p><strong>You will learn:</strong> <strong>RMSProp</strong> — per-parameter step scaling by an EMA of squared gradients; <strong>Adam</strong> = momentum + RMSProp + bias correction, with one full Adam step worked <em>by hand</em> on a 1-D parameter and matched to <code>torch.optim.Adam</code>; and <strong>AdamW</strong> — decoupled weight decay ($\theta \leftarrow \theta - \eta\lambda\theta$ applied separately) versus classic L2-in-the-gradient, why the decoupling matters, and a numeric contrast on one step.</p>
<p><strong>Why this matters for ML:</strong> AdamW is the default optimizer for essentially every large language model trained today — GPT, LLaMA, OLMo, and the GPT you build in Module 7 all use it. Its adaptive per-parameter scaling is what lets a single learning rate work across the wildly different gradient magnitudes of an embedding table, an attention projection, and a LayerNorm gain. Understanding Adam is understanding how modern models are actually optimized.</p>
</div>

## 1. Intuition: one learning rate cannot fit every parameter

Momentum (03.1) still uses a *single* global learning rate $\eta$ for every parameter in the model. But a real network's parameters have gradients of vastly different scales: an embedding row touched by a rare token gets tiny, sparse gradients; a LayerNorm gain gets large, dense ones. A learning rate small enough to be safe for the large-gradient parameters is far too small for the small-gradient ones, and vice versa. You cannot win with one number.

The fix, shared by RMSProp and Adam, is **adaptive per-parameter learning rates**: automatically give each parameter its *own* effective step size, scaled down where gradients have been large and scaled up where they have been small. The optimizer estimates "how big have this parameter's gradients been lately?" and divides the step by that. The result is that all parameters move at a comparable *relative* pace, and one global $\eta$ finally works for the whole model.

<div class="callout key"><p>Adaptive optimizers keep a running estimate of each parameter's recent gradient magnitude and divide that parameter's step by it. Big-gradient parameters take proportionally smaller steps; small-gradient parameters take larger ones. This per-parameter normalization is the whole idea behind RMSProp and Adam.</p></div>

## 2. RMSProp: scale by the EMA of squared gradients

### 2.1 The mathematics

RMSProp keeps, per parameter, an exponential moving average $v$ of the **squared** gradient — a running estimate of the gradient's typical size (its mean square). Then it divides the step by the square root of that estimate:

$$
v \;\leftarrow\; \beta\, v + (1-\beta)\, g^2, \qquad\qquad
\theta \;\leftarrow\; \theta - \eta\,\frac{g}{\sqrt{v} + \epsilon}.
$$

Naming symbols:

- $g$ — the current minibatch gradient for this parameter.
- $g^2$ — the **elementwise** square (each component squared), so $v$ tracks magnitude regardless of sign.
- $\beta$ — the decay rate for the average, typically $0.99$. Like momentum's $\mu$, it sets how far back the memory reaches.
- $v$ — the EMA of $g^2$: an estimate of the mean squared gradient. $\sqrt{v}$ is therefore roughly the **root-mean-square** (RMS) gradient magnitude — hence "RMSProp."
- $\epsilon$ — a tiny constant (e.g. $10^{-8}$) added to the denominator so we never divide by zero when $v$ is small.

The key term is $g / (\sqrt{v} + \epsilon)$. If this parameter's gradients have been large, $\sqrt v$ is large and the step is scaled *down*. If they have been tiny, $\sqrt v$ is tiny and the step is scaled *up*. In the steady state where the gradient is roughly constant at $g$, we get $\sqrt v \approx |g|$, so the ratio $g/\sqrt v \approx \pm 1$: the step size becomes almost independent of the gradient's magnitude and depends only on its *sign and consistency*. That is the per-parameter normalization we wanted.

RMSProp is the "adaptive scaling" half of Adam. On its own it lacks momentum's smoothing of the gradient direction. Adam adds that back.

## 3. Adam = momentum + RMSProp + bias correction

### 3.1 The two moments

Adam keeps **two** EMAs per parameter:

$$
m \;\leftarrow\; \beta_1\, m + (1-\beta_1)\, g \qquad\text{(first moment: EMA of } g\text{, like momentum)}
$$
$$
v \;\leftarrow\; \beta_2\, v + (1-\beta_2)\, g^2 \qquad\text{(second moment: EMA of } g^2\text{, like RMSProp)}
$$

- $m$ — the first moment, an EMA of the gradient itself. This is momentum's velocity (with the $(1-\beta_1)$ normalization), the smoothed *direction*.
- $v$ — the second moment, an EMA of the squared gradient. This is RMSProp's magnitude estimate, the per-parameter *scale*.
- $\beta_1 = 0.9$, $\beta_2 = 0.999$ are the standard decay rates: the direction average is short-memoried, the magnitude average is long-memoried.

Both $m$ and $v$ are initialized to **zero**. That creates a problem the next step fixes.

### 3.2 Bias correction

Because $m$ and $v$ start at zero, early in training they are biased *toward zero*: the EMA has not had time to "fill up." After the first step, $m = (1-\beta_1)g = 0.1\,g$ — only a tenth of the gradient, purely because it started from nothing. Left uncorrected, Adam would take tiny, wrong steps for the first many iterations. **Bias correction** rescales the moments to undo this startup bias:

$$
\hat m = \frac{m}{1 - \beta_1^{\,t}}, \qquad \hat v = \frac{v}{1 - \beta_2^{\,t}},
$$

where $t$ is the step number (starting at $1$). At $t = 1$, $1 - \beta_1^1 = 1 - 0.9 = 0.1$, so $\hat m = m / 0.1 = 10 m$ — exactly cancelling the $0.1$ factor above and recovering the full gradient. As $t$ grows, $\beta^t \to 0$ and the correction factors approach $1$ (no correction needed once the averages are warmed up).

### 3.3 The Adam update

Put together, one Adam step is:

$$
\theta \;\leftarrow\; \theta - \eta\,\frac{\hat m}{\sqrt{\hat v} + \epsilon}.
$$

The numerator $\hat m$ is the smoothed direction (momentum); the denominator $\sqrt{\hat v}$ is the per-parameter magnitude scale (RMSProp); both are bias-corrected. This is the single most-used optimizer step in deep learning.

<div class="callout key"><p>Adam step = (bias-corrected EMA of gradient) divided by (bias-corrected RMS of gradient). It is momentum in the numerator and RMSProp in the denominator, with a startup correction so the first steps are the right size. Three EMAs of ideas from 03.1, fused into one rule.</p></div>

### 3.4 One full Adam step, by hand

Take a single 1-D parameter $\theta = 1.0$ with gradient $g = 0.5$, learning rate $\eta = 0.1$, and the standard $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, at the first step $t = 1$. Both moments start at $0$.

**First moment.** $m = 0.9\cdot 0 + 0.1\cdot 0.5 = 0.05$.

**Second moment.** $v = 0.999\cdot 0 + 0.001\cdot 0.5^2 = 0.001\cdot 0.25 = 0.00025$.

**Bias correction** ($t = 1$). $\hat m = 0.05 / (1 - 0.9) = 0.05 / 0.1 = 0.5$; $\hat v = 0.00025 / (1 - 0.999) = 0.00025 / 0.001 = 0.25$.

**Update.** $\sqrt{\hat v} = \sqrt{0.25} = 0.5$, so
$$
\theta \leftarrow 1.0 - 0.1\cdot\frac{0.5}{0.5 + 10^{-8}} = 1.0 - 0.1\cdot 0.99999998 = 1.0 - 0.099999998 = 0.900000002.
$$

Two things worth noticing. First, $\hat m = g$ and $\sqrt{\hat v} = |g|$ exactly, because after one step the bias correction fully un-does the zero-init — so the ratio is $\pm 1$ and the step is $\eta \cdot \text{sign}(g) = 0.1$. **On the first step, Adam moves every parameter by almost exactly $\pm\eta$, regardless of the gradient's magnitude.** That is the adaptive normalization at its most extreme, and it is why Adam's early updates can be large — a fact that motivates the *warmup* in lesson 03.3. Second, the result $0.900000002$ matches `torch.optim.Adam` to floating-point precision:

```python
import torch
w = torch.tensor([1.0], requires_grad=True)
opt = torch.optim.Adam([w], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
w.grad = torch.tensor([0.5])
opt.step()
print(w.item())        # 0.9000000357627869  (matches 0.900000002 to ~1e-7)
```

## 4. Weight decay: classic L2 versus AdamW

### 4.1 What weight decay is for

**Weight decay** is regularization that gently pulls every parameter toward zero each step, discouraging the model from relying on large weights (which tend to overfit). There are two ways to implement it, and they are *not* equivalent for Adam — a subtlety that cost the field several years and is the entire reason AdamW exists.

### 4.2 Classic L2 (weight decay in the gradient)

The old approach adds an L2 penalty $\tfrac{\lambda}{2}\lVert\theta\rVert^2$ to the loss. Its gradient is $\lambda\theta$, which gets **added to the gradient** before the optimizer sees it:

$$
g' = g + \lambda\theta,
$$

and then Adam runs on $g'$. The problem: Adam *divides the step by $\sqrt{\hat v}$*. The decay term $\lambda\theta$ goes into the numerator **and** into the magnitude estimate in the denominator, so for a parameter with large gradients (large $\sqrt{\hat v}$) the decay gets divided down along with everything else. The effective amount of decay a parameter receives ends up **coupled to its gradient magnitude** — parameters that most need regularizing get the least of it. That is a bug, not a feature.

### 4.3 Decoupled weight decay (AdamW)

**AdamW** (Loshchilov & Hutter, 2019) fixes this by applying weight decay *directly to the parameter*, completely outside the Adam machinery:

$$
\theta \;\leftarrow\; \theta - \eta\lambda\theta \;=\; \theta(1 - \eta\lambda), \qquad\text{then the ordinary Adam step on the pure gradient } g.
$$

The decay is now **decoupled**: it shrinks every parameter by the same *fraction* $\eta\lambda$ per step, never touching $g$ or the moment estimates. Every parameter receives exactly the intended amount of regularization, independent of how large its gradients happen to be. $\lambda$ (typically $0.1$) is the decay coefficient; the EMAs $m, v$ are computed from $g$ alone.

<div class="callout key"><p>Classic L2 folds $\lambda\theta$ into the gradient, so Adam's per-parameter denominator distorts how much decay each parameter actually gets. AdamW subtracts $\eta\lambda\theta$ from the parameter separately, so decay is a clean, uniform shrink. This one change is why AdamW, not Adam, is the LLM default.</p></div>

### 4.4 Numeric contrast on one step

Same setup as §3.4 — $\theta = 1.0$, $g = 0.5$, $\eta = 0.1$, $t = 1$ — now with decay coefficient $\lambda = 0.1$.

**AdamW (decoupled).** First shrink the parameter: $\theta \leftarrow 1.0\cdot(1 - 0.1\cdot 0.1) = 1.0\cdot 0.99 = 0.99$. Then the ordinary Adam step on $g = 0.5$, which from §3.4 subtracts $0.1$:
$$
\theta \leftarrow 0.99 - 0.099999998 = 0.890000002.
$$

**Classic L2 (in the gradient).** Fold decay into the gradient: $g' = 0.5 + 0.1\cdot 1.0 = 0.6$. Now Adam on $g' = 0.6$: $\hat m = 0.6$, $\hat v = 0.6^2 = 0.36$, $\sqrt{\hat v} = 0.6$, so the step is $0.1\cdot 0.6/0.6 = 0.1$ and
$$
\theta \leftarrow 1.0 - 0.1 = 0.900000002.
$$

The contrast is the whole lesson in two numbers: **AdamW lands at $0.8900$, classic L2 at $0.9000$.** With classic L2 the decay was *swallowed by Adam's normalization* — adding $\lambda\theta$ to the gradient barely changed the update, because dividing by $\sqrt{\hat v}$ (which grew along with the numerator) cancelled almost all of its effect. With AdamW the $1\%$ shrink is applied cleanly on top of the Adam step. Both match PyTorch:

```python
import torch
w = torch.tensor([1.0], requires_grad=True)
opt = torch.optim.AdamW([w], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
w.grad = torch.tensor([0.5]); opt.step()
print(w.item())        # 0.89000004529953  -> AdamW, matches 0.890000002
```

## 5. The from-scratch AdamW

Here is the core of `code/src/llmre/optim/adamw.py`, in the exact order `torch.optim.AdamW` applies the operations (decay first, then the Adam step on the pure gradient):

```python
self.t += 1
bias_correction1 = 1.0 - self.beta1 ** self.t
bias_correction2 = 1.0 - self.beta2 ** self.t
for i, p in enumerate(self.params):
    if p.grad is None:
        continue
    g = p.grad
    # 1) Decoupled weight decay: shrink the parameter, do NOT touch g.
    if self.weight_decay != 0.0:
        p.mul_(1.0 - self.lr * self.weight_decay)
    # 2) Update biased first and second moments from the pure gradient.
    m, v = self.m[i], self.v[i]
    m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)          # m <- b1 m + (1-b1) g
    v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)   # v <- b2 v + (1-b2) g^2
    # 3) Bias-corrected step. denom = sqrt(v)/sqrt(bc2) + eps  ==  sqrt(v_hat) + eps
    denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(self.eps)
    step_size = self.lr / bias_correction1
    p.addcdiv_(m, denom, value=-step_size)                      # theta -= step_size * m / denom
```

Two implementation notes. The bias correction is applied to the *step*, not to the stored buffers: `step_size = lr / bias_correction1` and dividing by `denom = sqrt(v)/sqrt(bc2) + eps` together give exactly $\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$, but we never overwrite $m$ and $v$ with corrected values (they must stay biased to keep accumulating correctly). And `addcmul_`/`addcdiv_` are fused in-place multiply-add / divide-add: `v.addcmul_(g, g, value=c)` computes `v += c * g * g` in one pass with no temporary. The test `test_adamw_matches_torch_over_5_steps` confirms this matches `torch.optim.AdamW` to $10^{-5}$ over five steps of a linear-regression toy.

## 6. Tensor shapes, dtype, device, and cost

Like the SGD of 03.1, AdamW is elementwise and shape-agnostic. For each parameter tensor `p` of $P$ elements (dtype `float32` in normal training, on CPU or GPU):

- `m` and `v` each have the **same shape, dtype, and device** as `p` — two full parameter-sized state tensors.
- the step counter `t` is a single shared Python int; the bias-correction factors are scalars.
- every op (`mul_`, `add_`, `addcmul_`, `sqrt`, `addcdiv_`) is elementwise; no shape changes.

<div class="callout warn"><p><strong>Adam's memory cost is real and central to LLM training.</strong> Adam/AdamW store <em>two</em> parameter-sized buffers ($m$ and $v$) on top of the parameters and their gradients. For a model with $P$ parameters in <code>float32</code>, that is $4P$ bytes of parameters + $4P$ gradients + $8P$ optimizer state $= 16P$ bytes just to hold everything, before any activations. This $2\times$-parameters optimizer state is exactly what ZeRO and FSDP (Module 9) shard across GPUs, because for a large model it does not fit on one device. Momentum-SGD (03.1) stores only one such buffer; plain SGD stores none. The adaptivity of Adam is paid for in memory.</p></div>

Compute per step is again $O(P)$ elementwise work — negligible next to the forward/backward pass — and memory-bandwidth-bound: the step streams params, grads, $m$, and $v$ through the ALUs once each.

## Exercise

At step $t = 1$ Adam's step size is almost exactly $\pm\eta$ regardless of the gradient (§3.4). Show this holds for *any* nonzero gradient $g$ on the first step, ignoring $\epsilon$. Then explain in one sentence why this fact motivates learning-rate warmup.

<details><summary>Hint</summary>

Write out $\hat m$ and $\hat v$ at $t = 1$ in terms of $g$, using $1 - \beta_1^1 = 1 - \beta_1$ and $1 - \beta_2^1 = 1 - \beta_2$.

</details>

<details><summary>Solution</summary>

At $t = 1$: $m = (1-\beta_1)g$ so $\hat m = m/(1-\beta_1) = g$. And $v = (1-\beta_2)g^2$ so $\hat v = v/(1-\beta_2) = g^2$, giving $\sqrt{\hat v} = |g|$. Therefore the step is
$$
\eta\,\frac{\hat m}{\sqrt{\hat v}} = \eta\,\frac{g}{|g|} = \eta\,\operatorname{sign}(g),
$$
magnitude exactly $\eta$ for any nonzero $g$. Because the very first steps have this fixed, comparatively large size *and* are based on a second-moment estimate $\hat v$ built from just one or two noisy samples (so the per-parameter scaling is unreliable early), starting at full learning rate can send parameters flying in the wrong direction — which is exactly why we ramp $\eta$ up slowly with warmup (03.3).

</details>

## Common mistakes

- **Applying weight decay to the wrong parameters.** In practice you decay the weight *matrices* but not the biases, LayerNorm gains, or embeddings — decaying a 1-D gain toward zero just fights the normalization. Frameworks split parameters into two groups for this. (Our minimal class decays everything; Module 7 shows the grouped version.)
- **Correcting the stored moments instead of the step.** If you overwrite $m \leftarrow \hat m$ in the buffer, the *next* step's EMA accumulates a corrected value and the bias correction compounds wrongly. Keep $m, v$ biased; apply the correction only when computing the step (as in §5).
- **Using classic L2 and calling it AdamW.** Passing `weight_decay` to `torch.optim.Adam` (not `AdamW`) gives the coupled, gradient-folded version — the very thing §4 says to avoid.

## Check yourself

<details><summary>What does the denominator $\sqrt{\hat v} + \epsilon$ accomplish in the Adam update?</summary>

It gives each parameter its own adaptive step size. $\sqrt{\hat v}$ is an estimate of that parameter's RMS gradient magnitude, so dividing by it scales large-gradient parameters' steps down and small-gradient parameters' steps up — per-parameter normalization. $\epsilon$ just prevents division by zero when $\hat v$ is tiny.

</details>

<details><summary>Why does Adam need bias correction, and what happens without it?</summary>

$m$ and $v$ are initialized to zero, so early on they underestimate the true moments (the EMA has not warmed up). Dividing by $1 - \beta^t$ rescales them to remove this startup bias. Without it, Adam takes far-too-small, distorted steps for the first many iterations (the $v$ under-estimate is especially damaging since $\beta_2 = 0.999$ warms up slowly).

</details>

<details><summary>State the one-line difference between Adam-with-L2 and AdamW.</summary>

Adam-with-L2 adds $\lambda\theta$ to the gradient (so the decay passes through Adam's per-parameter normalization and gets distorted); AdamW subtracts $\eta\lambda\theta$ from the parameter directly (decoupled), so decay is a clean uniform shrink independent of gradient magnitude.

</details>

<details><summary>A model has $P = 10^9$ parameters trained in float32. How many bytes do the AdamW optimizer states alone occupy?</summary>

AdamW stores two parameter-sized state tensors ($m$ and $v$). Two $\times$ $10^9$ $\times$ 4 bytes $= 8\times 10^9$ bytes $= 8$ GB — on top of 4 GB of parameters and 4 GB of gradients. This is why optimizer-state sharding (Module 9) exists.

</details>

<details><summary>On the first step, by how much does Adam move a parameter whose gradient is $g = 100$, with $\eta = 3\times10^{-4}$?</summary>

By about $\eta = 3\times10^{-4}$ (in the $-\operatorname{sign}(g)$ direction), *not* by anything proportional to $100$. At $t=1$ the ratio $\hat m/\sqrt{\hat v} = g/|g| = 1$, so the step magnitude is $\eta$ regardless of how large $g$ is.

</details>

## Next

You now have AdamW, the optimizer that will train your GPT. But a fixed learning rate is not what real runs use: they *ramp $\eta$ up* over the first few thousand steps (warmup) and then *decay it down* along a cosine curve, and they *clip* the gradient's global norm to survive the occasional huge minibatch. The next lesson covers those three schedule-and-stability tricks — the ones on essentially every production LLM training run.

Continue to [03.3 · Warmup, cosine decay, gradient clipping](lessons/module-03/lesson-03.md).
