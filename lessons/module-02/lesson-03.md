# 02.3 · Backprop through linear + softmax + cross-entropy by hand

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-02/lesson-02">02.2 · Jacobians, VJPs, computational graphs</a> — the VJP view of backprop and the fan-in summation of gradients. <a href="#/lessons/module-01/lesson-03">01.3 · Entropy, cross-entropy, KL divergence</a> — cross-entropy against a one-hot target reducing to $-\log p_{\text{target}}$, and softmax as the map from scores to a distribution.</p>
<p><strong>You will learn:</strong> to derive, by hand and step by step, the gradient of cross-entropy loss through a softmax and a linear layer — the exact computation at the output of every language model. The centerpiece is the famous result <strong>$\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \text{onehot}(\text{target})$</strong>. From it we get $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{z}}\,\mathbf{x}^\top$, $\frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{z}}$, and $\frac{\partial L}{\partial \mathbf{x}} = W^\top \frac{\partial L}{\partial \mathbf{z}}$. A full 3-class, 2-dim example is computed by hand and matched to <code>loss.backward()</code> to $10^{-6}$.</p>
<p><strong>Why this matters for ML:</strong> this <em>is</em> the gradient a language model computes at its output on every single training step. The final layer of a GPT produces logits, softmax turns them into a next-token distribution, cross-entropy against the true next token is the loss, and the gradient that starts the entire backward pass through the network is $\mathbf{p} - \text{onehot}(\text{target})$ — "predicted distribution minus the truth." If you understand one derivation in this course, make it this one.</p>
</div>

## 1. Intuition: predicted minus true

Here is the punchline before the algebra, because it is worth holding in mind the whole way through. The model outputs a probability distribution $\mathbf{p}$ over the vocabulary. The truth is a one-hot vector: probability $1$ on the correct next token, $0$ everywhere else. The gradient of the loss with respect to the logits turns out to be simply

$$
\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \text{onehot}(\text{target}),
$$

the **difference between what the model predicted and what was true**. Where the model put too much probability (any wrong token with $p_i > 0$), the gradient is positive — push that logit down. Where it put too little (the true token, whose target is $1$ but $p < 1$), the gradient is negative — push that logit up. The size of each nudge is exactly how wrong the probability was. It is the most intuitive gradient imaginable, and the algebra below shows it falls out cleanly because softmax and cross-entropy are designed to fit together.

## 2. The forward pass, every symbol named

We compute a loss in four stages. Sizes: $C$ classes (the vocabulary size in a real LM), input dimension $d$.

1. **Linear layer.** $\mathbf{z} = W\mathbf{x} + \mathbf{b}$, where $\mathbf{x}$ is the input vector (shape $(d,)$), $W$ is the weight matrix (shape $(C, d)$), $\mathbf{b}$ is the bias (shape $(C,)$), and $\mathbf{z}$ is the vector of **logits** (shape $(C,)$). Componentwise, $z_i = \sum_{j} W_{ij} x_j + b_i$.
2. **Softmax.** $p_i = \dfrac{e^{z_i}}{\sum_{k} e^{z_k}}$, turning logits into a probability distribution $\mathbf{p}$ (shape $(C,)$, entries positive, summing to $1$).
3. **Pick the true class.** Let $t$ be the index of the correct class. Write $\mathbf{y} = \text{onehot}(t)$, the vector with $y_t = 1$ and $y_i = 0$ otherwise.
4. **Cross-entropy loss.** $L = -\sum_i y_i \log p_i = -\log p_t$ (from [01.3](lessons/module-01/lesson-03.md): against a one-hot target, cross-entropy collapses to the negative log-probability of the true class).

So the pipeline is $\mathbf{x} \xrightarrow{W,\mathbf{b}} \mathbf{z} \xrightarrow{\text{softmax}} \mathbf{p} \xrightarrow{-\log p_t} L$. We backpropagate right to left, exactly the VJP sweep of lesson 02.2, computing $\frac{\partial L}{\partial \mathbf{p}}$, then $\frac{\partial L}{\partial \mathbf{z}}$, then the gradients for the parameters $W, \mathbf{b}$ and the input $\mathbf{x}$.

## 3. Step one: loss with respect to the probabilities

$L = -\log p_t$ depends only on the true-class probability $p_t$. So

$$
\frac{\partial L}{\partial p_t} = -\frac{1}{p_t}, \qquad \frac{\partial L}{\partial p_i} = 0 \ \ (i \ne t).
$$

Only one entry of this gradient is nonzero. That sparsity is about to make the next step clean.

## 4. Step two: the softmax Jacobian, and the collapse to $\mathbf{p} - \mathbf{y}$

We need $\frac{\partial L}{\partial z_i}$. By the chain rule (summing over how each $p_k$ depends on $z_i$),

$$
\frac{\partial L}{\partial z_i} = \sum_k \frac{\partial L}{\partial p_k}\,\frac{\partial p_k}{\partial z_i}.
$$

Since $\frac{\partial L}{\partial p_k}$ is nonzero only at $k = t$, this reduces to one term:

$$
\frac{\partial L}{\partial z_i} = \frac{\partial L}{\partial p_t}\,\frac{\partial p_t}{\partial z_i} = -\frac{1}{p_t}\,\frac{\partial p_t}{\partial z_i}.
$$

Now we need the **softmax Jacobian** $\frac{\partial p_t}{\partial z_i}$. The standard result (derived by the quotient rule on $p_t = e^{z_t}/\sum_k e^{z_k}$) splits into two cases:

$$
\frac{\partial p_t}{\partial z_i} = \begin{cases} p_t(1 - p_t) & i = t \\ -\,p_t\, p_i & i \ne t \end{cases}
\qquad\text{compactly}\qquad \frac{\partial p_t}{\partial z_i} = p_t(\delta_{ti} - p_i),
$$

where $\delta_{ti} = 1$ if $i = t$ and $0$ otherwise. (A short derivation: for $i = t$, differentiating $e^{z_t}/S$ with $S = \sum_k e^{z_k}$ gives $\frac{e^{z_t}S - e^{z_t}e^{z_t}}{S^2} = p_t - p_t^2 = p_t(1-p_t)$. For $i \ne t$, the numerator $e^{z_t}$ has no $z_i$, so we get $\frac{-e^{z_t}e^{z_i}}{S^2} = -p_t p_i$.)

Substitute this into the chain-rule expression. The $-\frac{1}{p_t}$ cancels the leading $p_t$:

$$
\frac{\partial L}{\partial z_i} = -\frac{1}{p_t}\cdot p_t(\delta_{ti} - p_i) = -(\delta_{ti} - p_i) = p_i - \delta_{ti}.
$$

And $\delta_{ti}$ is exactly the $i$-th entry of the one-hot vector $\mathbf{y}$. Stacking over all $i$:

$$
\boxed{\ \frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y} = \mathbf{p} - \text{onehot}(t).\ }
$$

<div class="callout key"><p>The $-1/p_t$ from cross-entropy and the $p_t$ from the softmax Jacobian cancel exactly. That cancellation is why the gradient of the logits is the clean $\mathbf{p} - \mathbf{y}$, with no division and no numerical blow-up even when $p_t$ is tiny. Softmax + cross-entropy are paired on purpose: apart, each has an awkward gradient; together, the logit gradient is just "prediction minus truth."</p></div>

We denote this vector $\mathbf{g} = \frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y}$ (shape $(C,)$). It is the incoming gradient for the linear layer — the $\mathbf{v}$ we now push back through $\mathbf{z} = W\mathbf{x} + \mathbf{b}$.

## 5. Step three: through the linear layer to $W$, $\mathbf{b}$, $\mathbf{x}$

The linear layer is $z_i = \sum_j W_{ij} x_j + b_i$. We already have $\mathbf{g} = \frac{\partial L}{\partial \mathbf{z}}$; apply the chain rule to each parameter.

**Bias.** Since $z_i$ contains $+b_i$ directly, $\frac{\partial z_i}{\partial b_i} = 1$ and $\frac{\partial z_k}{\partial b_i} = 0$ for $k \ne i$. So

$$
\frac{\partial L}{\partial b_i} = \sum_k g_k \frac{\partial z_k}{\partial b_i} = g_i \quad\Longrightarrow\quad \frac{\partial L}{\partial \mathbf{b}} = \mathbf{g}.
$$

**Weights.** $z_i = \sum_j W_{ij}x_j$ means $W_{ij}$ affects only $z_i$, and $\frac{\partial z_i}{\partial W_{ij}} = x_j$. Hence

$$
\frac{\partial L}{\partial W_{ij}} = g_i\, x_j \quad\Longrightarrow\quad \frac{\partial L}{\partial W} = \mathbf{g}\,\mathbf{x}^\top,
$$

an **outer product**: column vector $\mathbf{g}$ (shape $(C,1)$) times row vector $\mathbf{x}^\top$ (shape $(1,d)$) gives a $(C, d)$ matrix — the same shape as $W$, as it must be.

**Input.** $x_j$ affects every $z_i$ (through column $j$ of $W$), so we sum over $i$:

$$
\frac{\partial L}{\partial x_j} = \sum_i g_i \frac{\partial z_i}{\partial x_j} = \sum_i g_i W_{ij} \quad\Longrightarrow\quad \frac{\partial L}{\partial \mathbf{x}} = W^\top \mathbf{g}.
$$

The $W^\top$ (shape $(d, C)$) times $\mathbf{g}$ (shape $(C,)$) gives shape $(d,)$ — matching $\mathbf{x}$. This $\frac{\partial L}{\partial \mathbf{x}}$ is what gets handed to the *previous* layer as its incoming gradient; it is how the backward sweep continues down into the rest of the network.

<div class="callout key"><p>The three linear-layer gradients — memorize their shapes, they recur in every layer of the course:</p>
<p>$$\frac{\partial L}{\partial W} = \mathbf{g}\,\mathbf{x}^\top\ (C{\times}d), \qquad \frac{\partial L}{\partial \mathbf{b}} = \mathbf{g}\ (C), \qquad \frac{\partial L}{\partial \mathbf{x}} = W^\top\mathbf{g}\ (d),$$</p>
<p>with $\mathbf{g} = \mathbf{p} - \text{onehot}(t)$. The forward multiplies by $W$; the backward-to-input multiplies by $W^\top$. That "transpose on the way back" is the linear layer's VJP.</p></div>

## 6. Full numeric example, by hand

Take $C = 3$ classes, input dimension $d = 2$. Concrete numbers:

$$
W = \begin{bmatrix} 0.1 & -0.2 \\ 0.3 & 0.4 \\ -0.5 & 0.6 \end{bmatrix}\ (3{\times}2), \quad
\mathbf{b} = \begin{bmatrix} 0.1 \\ -0.1 \\ 0.2 \end{bmatrix}, \quad
\mathbf{x} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad t = 0\ (\text{true class}).
$$

**Forward — logits** $\mathbf{z} = W\mathbf{x} + \mathbf{b}$:

$$
\begin{aligned}
z_0 &= 0.1\cdot 1 + (-0.2)\cdot 2 + 0.1 = 0.1 - 0.4 + 0.1 = -0.2,\\
z_1 &= 0.3\cdot 1 + 0.4\cdot 2 + (-0.1) = 0.3 + 0.8 - 0.1 = 1.0,\\
z_2 &= -0.5\cdot 1 + 0.6\cdot 2 + 0.2 = -0.5 + 1.2 + 0.2 = 0.9.
\end{aligned}
$$

So $\mathbf{z} = [-0.2,\ 1.0,\ 0.9]$.

**Forward — softmax.** Exponentiate: $e^{-0.2} = 0.818731$, $e^{1.0} = 2.718282$, $e^{0.9} = 2.459603$. Sum $S = 5.996616$. Divide:

$$
\mathbf{p} = \left[\frac{0.818731}{5.996616},\ \frac{2.718282}{5.996616},\ \frac{2.459603}{5.996616}\right] = [\,0.136532,\ 0.453303,\ 0.410165\,].
$$

**Forward — loss.** $L = -\log p_0 = -\log(0.136532) = 1.991195$.

**Backward — logit gradient** $\mathbf{g} = \mathbf{p} - \text{onehot}(0)$, subtracting $1$ from the true class:

$$
\mathbf{g} = [\,0.136532 - 1,\ 0.453303,\ 0.410165\,] = [\,-0.863468,\ 0.453303,\ 0.410165\,].
$$

Read the story: the model gave the true class $0$ only $0.137$ probability, so its gradient is strongly negative ($-0.863$) — push that logit *up*. Classes $1$ and $2$ got too much mass, so their gradients are positive — push those logits *down*. The three entries sum to $0$ (they always do, since $\mathbf{p}$ sums to $1$ and one-hot sums to $1$), meaning softmax gradients only ever *redistribute* probability, never create it.

**Backward — bias** $\frac{\partial L}{\partial \mathbf{b}} = \mathbf{g} = [-0.863468,\ 0.453303,\ 0.410165]$.

**Backward — weights** $\frac{\partial L}{\partial W} = \mathbf{g}\,\mathbf{x}^\top$, outer product of $\mathbf{g}$ with $\mathbf{x} = [1, 2]$:

$$
\frac{\partial L}{\partial W} = \begin{bmatrix} -0.863468 \\ 0.453303 \\ 0.410165 \end{bmatrix}\begin{bmatrix} 1 & 2 \end{bmatrix}
= \begin{bmatrix} -0.863468 & -1.726936 \\ 0.453303 & 0.906605 \\ 0.410165 & 0.820330 \end{bmatrix}.
$$

**Backward — input** $\frac{\partial L}{\partial \mathbf{x}} = W^\top \mathbf{g}$:

$$
\begin{aligned}
\left(W^\top\mathbf{g}\right)_0 &= 0.1(-0.863468) + 0.3(0.453303) + (-0.5)(0.410165) = -0.155439,\\
\left(W^\top\mathbf{g}\right)_1 &= -0.2(-0.863468) + 0.4(0.453303) + 0.6(0.410165) = 0.600114.
\end{aligned}
$$

So $\frac{\partial L}{\partial \mathbf{x}} = [-0.155439,\ 0.600114]$.

## 7. Tensor shapes and the batched reality

Everything above was for a single example. In a real training step you process a batch of $N$ tokens at once, so shapes gain a leading batch axis:

| quantity | single example | batch of $N$ | dtype / device |
|---|---|---|---|
| input $\mathbf{x}$ | $(d,)$ | $(N, d)$ | float32, cuda |
| logits $\mathbf{z}$ | $(C,)$ | $(N, C)$ | float32, cuda |
| probs $\mathbf{p}$ | $(C,)$ | $(N, C)$ | float32, cuda |
| targets $t$ | scalar int | $(N,)$ long | int64, cuda |
| $\mathbf{g} = \mathbf{p}-\mathbf{y}$ | $(C,)$ | $(N, C)$ | float32, cuda |
| $\partial L/\partial W$ | $(C, d)$ | $(C, d)$ | float32, cuda |

Note the last row: the loss is the **mean over the batch** (a scalar), so the weight gradient is *summed* (then averaged) over the $N$ examples and stays shape $(C, d)$ — the same shape as $W$, independent of batch size. In matrix form the batched weight gradient is $\frac{1}{N} G^\top X$ where $G$ is $(N, C)$ and $X$ is $(N, d)$, which is the batched version of the per-example outer product. In a GPT, $C$ is the vocabulary size ($\approx 50{,}000$) and this final projection's gradient is one of the largest single tensors in the model.

## 8. From-scratch, then verified against `loss.backward()`

First the from-scratch backward, using only the formulas we derived — no autograd:

```python
import torch

W = torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]])   # (3,2)
b = torch.tensor([0.1, -0.1, 0.2])                          # (3,)
x = torch.tensor([1.0, 2.0])                                # (2,)
t = 0

# forward
z = W @ x + b                                               # (3,) logits
p = torch.softmax(z, dim=0)                                 # (3,) probs
L = -torch.log(p[t])                                        # scalar loss

# backward, by our hand-derived rules
y = torch.zeros(3); y[t] = 1.0
g = p - y                                                   # dL/dz  = p - onehot(t)
dW = torch.outer(g, x)                                      # dL/dW  = g x^T   (3,2)
db = g                                                      # dL/db  = g       (3,)
dx = W.t() @ g                                              # dL/dx  = W^T g   (2,)
print(L.item())          # 1.9911953
print(g.tolist())        # [-0.86346..., 0.45330..., 0.41016...]
print(dx.tolist())       # [-0.15543..., 0.60011...]
```

Now the same computation with autograd, and an assertion that the two agree to $10^{-6}$:

```python
import torch, torch.nn.functional as F

W = torch.tensor([[0.1,-0.2],[0.3,0.4],[-0.5,0.6]], requires_grad=True)
b = torch.tensor([0.1,-0.1,0.2], requires_grad=True)
x = torch.tensor([1.0,2.0], requires_grad=True)
t = torch.tensor(0)

z = W @ x + b                                     # (3,)
L = F.cross_entropy(z.unsqueeze(0), t.unsqueeze(0))   # fused softmax+NLL, scalar
L.backward()                                      # autograd fills .grad on W, b, x

# our analytic values
p = torch.softmax(z, dim=0).detach()
g = p - torch.eye(3)[t]
assert torch.allclose(W.grad, torch.outer(g, x.detach()), atol=1e-6)
assert torch.allclose(b.grad, g,                          atol=1e-6)
assert torch.allclose(x.grad, W.detach().t() @ g,         atol=1e-6)
print("all match:", L.item())                    # all match: 1.9911953
```

All three assertions pass. `W.grad`, `b.grad`, `x.grad` from `loss.backward()` equal our hand-derived $\mathbf{g}\mathbf{x}^\top$, $\mathbf{g}$, and $W^\top\mathbf{g}$ — confirming the derivation to machine precision.

<div class="callout pt"><p><code>F.cross_entropy(logits, target)</code> fuses softmax, log, and the negative-log-likelihood pick into one op — it never forms <code>p</code> explicitly in the forward pass. Its backward is hard-coded to return exactly <code>p - onehot(t)</code> (scaled by <code>1/N</code> for the batch mean). PyTorch is not discovering this gradient by differentiating softmax and log separately; the framework <em>knows</em> the closed form you just derived and applies it directly, which is faster and numerically safer (no <code>log</code> of a possibly-tiny <code>p</code>).</p></div>

## 9. Under the hood: numerical stability and why the fused op exists

If you compute softmax naively as `exp(z)/sum(exp(z))`, a large logit overflows `exp`. Real implementations subtract the max logit first — $p_i = e^{z_i - \max_k z_k}/\sum_k e^{z_k - \max_k z_k}$ — which is mathematically identical (the shared factor cancels) but keeps every exponent $\le 0$. And computing $-\log p_t$ via `log_softmax` avoids ever forming a tiny $p_t$ and then taking its log (which loses precision). Fusing the whole thing into `cross_entropy` means the forward is stable and the backward is the single subtraction $\mathbf{p} - \mathbf{y}$ — no chain of softmax and log Jacobians to multiply at runtime. This is a running theme: the framework hard-codes the VJP of common composite ops both for speed and for numerical safety.

## 10. Research connection

<div class="callout paper"><p>This exact output gradient is what trains GPT-2: the final linear layer projects the last hidden state to <code>vocab_size</code> logits, softmax + cross-entropy against the true next token gives the loss, and $\mathbf{p} - \text{onehot}(t)$ is the gradient that launches the backward pass through all 12 (or 48) layers. See the <a href="#/papers/index">paper curriculum</a> — GPT-2 ("Language Models are Unsupervised Multitask Learners", Radford et al. 2019), which you assemble from scratch in Module 6.</p></div>

## Exercise

Redo the forward and the logit gradient $\mathbf{g} = \mathbf{p} - \mathbf{y}$ for the *same* $W, \mathbf{b}, \mathbf{x}$ but with true class $t = 1$ instead of $0$. Which entry of $\mathbf{g}$ becomes negative, and does $\mathbf{g}$ still sum to zero?

<details><summary>Optional hint</summary>

The forward is unchanged — $\mathbf{p} = [0.136532, 0.453303, 0.410165]$ as before. Only the one-hot vector moves: now $\mathbf{y} = [0, 1, 0]$.

</details>

<details><summary>Stronger hint</summary>

Subtract $1$ from entry $1$ instead of entry $0$: $\mathbf{g} = \mathbf{p} - [0,1,0]$.

</details>

<details><summary>Solution</summary>

$\mathbf{g} = [0.136532,\ 0.453303 - 1,\ 0.410165] = [0.136532,\ -0.546697,\ 0.410165]$. Entry $1$ (the new true class) is negative — its logit gets pushed up. The entries still sum to $0$ ($0.136532 - 0.546697 + 0.410165 = 0$), because $\mathbf{p}$ sums to $1$ and one-hot sums to $1$ regardless of which class is true. The loss is now lower, $L = -\log(0.453303) = 0.7912$ (versus $1.9912$ when $t=0$), because the model happened to give class $1$ more probability than class $0$ — it is "less wrong," so its true-class gradient is smaller in magnitude ($-0.547$ vs $-0.863$).

</details>

## Common mistakes

- **Transposing $W$ the wrong way (or not at all).** Forward uses $W$; the gradient to the input uses $W^\top$. Mixing them up is a silent shape bug when $C \ne d$ and a silent *wrong-answer* bug when $C = d$.
- **Outer product in the wrong order.** $\frac{\partial L}{\partial W} = \mathbf{g}\mathbf{x}^\top$ has shape $(C, d)$. Writing $\mathbf{x}\mathbf{g}^\top$ gives $(d, C)$ — the transpose of what you want.
- **Forgetting the batch mean.** With a batch, the loss is averaged over $N$, so every parameter gradient carries a $1/N$. Compare against `reduction='mean'` (the default) not `'sum'`.
- **Building $\mathbf{p}$ then taking `log`.** Numerically worse than `log_softmax`/`cross_entropy`. Fine for a hand check; do not ship it.

## Check yourself

<details><summary>Why do the entries of $\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \text{onehot}(t)$ always sum to zero?</summary>

Because $\sum_i p_i = 1$ (softmax output) and $\sum_i y_i = 1$ (one-hot), so $\sum_i(p_i - y_i) = 1 - 1 = 0$. Interpretation: the gradient redistributes probability mass between classes — every unit pushed off wrong classes is pushed onto the right one — it never adds or removes total mass.

</details>

<details><summary>Where exactly does the derivation avoid dividing by the possibly-tiny $p_t$?</summary>

$\frac{\partial L}{\partial p_t} = -1/p_t$ does contain the dangerous division, but when multiplied by the softmax Jacobian entry $p_t(\delta_{ti}-p_i)$, the $p_t$ cancels the $1/p_t$, leaving $p_i - \delta_{ti}$ with no division at all. This cancellation is why the fused softmax+cross-entropy backward is just a subtraction and is numerically safe.

</details>

<details><summary>In the worked example, the true-class gradient was $-0.863$ while the others were $+0.453$ and $+0.410$. What is the model being told to do?</summary>

Raise the logit of the true class $0$ (negative gradient → gradient descent increases it) and lower the logits of classes $1$ and $2$ (positive gradients → decrease them). The magnitudes say class $0$ needs the biggest correction because it was the most under-predicted ($p_0 = 0.137$, far below the target $1$).

</details>

<details><summary>For a batch of $N=32$ tokens with vocab $C=50257$ and hidden size $d=768$, what shape is $\partial L/\partial W$ for the final projection, and does it depend on $N$?</summary>

Shape $(C, d) = (50257, 768)$ — the same as $W$ — and it does **not** depend on $N$. The batch contributes by summing/averaging the $N$ per-token outer products into that same $(C, d)$ matrix (the $\frac{1}{N}G^\top X$ form). Batch size changes the compute, not the gradient's shape.

</details>

## Next

You have derived and verified the gradient that every language model computes at its output, and the linear-layer VJPs ($W$ on the way forward, $W^\top$ on the way back) that carry it into the rest of the network. The last lesson of this module opens the box completely: how PyTorch builds the computational graph as you compute, and how `.backward()` walks it — which we prove by writing a tiny autograd engine from scratch and checking it against PyTorch.

Continue to [02.4 · Autograd internals](lessons/module-02/lesson-04.md).
