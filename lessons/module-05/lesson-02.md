# 05.2 · Q/K/V & scaled dot-product attention

<div class="prereq">
<p><strong>Prerequisites:</strong> the position-aware input grid $h$ of shape $(B, T, C)$ from <a href="lesson-01.md">05.1 · Embeddings &amp; positional encoding</a>; softmax and the cross-entropy machinery from <a href="../module-02/lesson-03.md">02.3 · Backprop through linear + softmax + CE by hand</a>; the dot product and matrix multiply from Module 2.</p>
<p><strong>You will learn:</strong> what queries, keys, and values are and how they come from three learned linear projections; the scaled dot-product attention formula $\mathrm{softmax}(QK^\top/\sqrt{d_h} + M)V$ term by term; why we divide by $\sqrt{d_h}$; how the causal mask forbids looking at the future; and why the $T \times T$ score matrix is the memory bottleneck of the whole architecture. You will work a $T=3$, $d_h=2$ example fully by hand and check it against PyTorch's fused kernel.</p>
<p><strong>Why this matters for ML:</strong> attention is the one operation that makes a transformer a transformer — it is how each token pulls in information from the other tokens. Every GPT layer is attention plus a small feed-forward network, so once you can compute attention by hand and read its PyTorch form, the rest of the model is assembly.</p>
</div>

## 1. Intuition: a soft, learned dictionary lookup

A position in the sequence carries a $C$-vector (from Module 5.1) that so far only knows about *its own* token. To predict the next word, a position needs information from *other* positions: the subject a verb agrees with, the noun a pronoun refers to, the opening bracket a closing one must match. **Attention** is the mechanism for gathering that information.

The cleanest analogy for a programmer is a soft dictionary lookup. A normal dictionary looks up one exact key and returns its one value. Attention does a *fuzzy, weighted* lookup: every position asks a question, compares it against what every position advertises, and receives back a blend of everyone's contents, weighted by how well each matched. Three learned roles make this precise, one vector per role per position:

- **Query** $\mathbf{q}$ — *what I am looking for.* The current position's question.
- **Key** $\mathbf{k}$ — *what I offer.* An advertisement each position publishes.
- **Value** $\mathbf{v}$ — *what I hand over* if I am attended to.

A position compares its query to every key to decide how much to listen to each other position, then takes that weighted blend of their values. That is the whole idea; the rest of the lesson turns each step into arithmetic.

## 2. Q, K, V are three learned linear projections

Where do $\mathbf{q}$, $\mathbf{k}$, $\mathbf{v}$ come from? From the input. Given the input $x$ of shape $(B, T, C)$, we produce queries, keys, and values by three **learned linear maps** — three weight matrices $W_Q, W_K, W_V$, each of shape $(C, d_h)$ in the single-head case:

$$
Q = x\,W_Q, \qquad K = x\,W_K, \qquad V = x\,W_V.
$$

Each of $Q, K, V$ then has shape $(B, T, d_h)$: one $d_h$-vector per position. The point of learning three *separate* projections is that the same token can advertise one thing (its key), ask for a different thing (its query), and contribute a third thing (its value). Nothing forces $\mathbf{q}$, $\mathbf{k}$, $\mathbf{v}$ to be equal or even related except through training.

For the worked example below we skip the projection and simply hand you $Q, K, V$ directly, so we can focus on attention itself. In real code (section 8, and the `CausalSelfAttention` module of the next lesson) the projection is a single `nn.Linear`.

<div class="callout key"><p>Attention is defined by one formula, which the rest of the lesson unpacks term by term:</p>
<p>$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}} + M\right) V.$$</p>
<p>$QK^\top$ are the raw scores, $\sqrt{d_h}$ is the scale, $M$ is the causal mask, softmax turns each row into weights, and multiplying by $V$ takes the weighted average.</p></div>

## 3. The setup for the worked example

We use $T = 3$ positions and head dimension $d_h = 2$. The three matrices, each shape $(T, d_h) = (3, 2)$:

$$
Q = \begin{bmatrix} 2 & 0 \\ 1 & 1 \\ 0 & 2 \end{bmatrix}, \qquad
K = \begin{bmatrix} 1 & 1 \\ 2 & 0 \\ 0 & 1 \end{bmatrix}, \qquad
V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}.
$$

Read the rows as positions: $\mathbf{q}_1 = [2, 0]$, $\mathbf{q}_2 = [1, 1]$, $\mathbf{q}_3 = [0, 2]$, and likewise for $K$ and $V$.

## 4. Step one — scores are dot products

The first thing attention computes is, for every pair $(i, j)$, a **score** measuring how much position $i$'s query matches position $j$'s key. That match is a **dot product**: large and positive when the vectors point the same way, near zero when unrelated, negative when opposed. The score for query $i$ against key $j$ is

$$
s_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j = \sum_{m=1}^{d_h} q_{im}\, k_{jm}.
$$

Stacking all $T \times T$ scores into a matrix is one matrix multiplication, $S = Q K^\top$. The transpose $K^\top$ turns $K$'s rows (keys) into columns, so entry $(i, j)$ of the product is exactly $\mathbf{q}_i \cdot \mathbf{k}_j$. The result has shape $(T, T) = (3, 3)$.

Compute the nine entries. Recall $\mathbf{k}_1 = [1,1]$, $\mathbf{k}_2 = [2,0]$, $\mathbf{k}_3 = [0,1]$:

- Row 1 ($\mathbf{q}_1 = [2,0]$): $[2,0]\cdot[1,1]=2$, $\;[2,0]\cdot[2,0]=4$, $\;[2,0]\cdot[0,1]=0$.
- Row 2 ($\mathbf{q}_2 = [1,1]$): $[1,1]\cdot[1,1]=2$, $\;[1,1]\cdot[2,0]=2$, $\;[1,1]\cdot[0,1]=1$.
- Row 3 ($\mathbf{q}_3 = [0,2]$): $[0,2]\cdot[1,1]=2$, $\;[0,2]\cdot[2,0]=0$, $\;[0,2]\cdot[0,1]=2$.

$$
S = Q K^\top = \begin{bmatrix} 2 & 4 & 0 \\ 2 & 2 & 1 \\ 2 & 0 & 2 \end{bmatrix}.
$$

## 5. Step two — scale by $1/\sqrt{d_h}$

We do not feed raw scores into softmax; we first divide by $\sqrt{d_h}$. Here $d_h = 2$, so the scale is $1/\sqrt{2} \approx 0.7071$:

$$
\frac{S}{\sqrt 2} = \begin{bmatrix} 1.4142 & 2.8284 & 0 \\ 1.4142 & 1.4142 & 0.7071 \\ 1.4142 & 0 & 1.4142 \end{bmatrix}.
$$

**Why divide at all?** A dot product over $d_h$ dimensions is a sum of $d_h$ products. As $d_h$ grows (real models use $d_h = 64$), that sum tends to grow, so raw scores get large. Softmax on large-magnitude inputs **saturates**: nearly all weight lands on the single biggest score, producing a near-one-hot distribution — a hard pick instead of a soft blend — and a nearly flat gradient, so learning stalls. Concretely, if the entries of $\mathbf{q}$ and $\mathbf{k}$ are independent with unit variance, the dot product $\mathbf{q}\cdot\mathbf{k}$ has variance $d_h$; dividing by $\sqrt{d_h}$ rescales it back to unit variance, keeping softmax in its sensitive regime regardless of head size.

## 6. Step three — the causal mask: no peeking at the future

GPT is trained to predict the *next* token. At position $i$ it must use only positions $1, \dots, i$ — tokens it has already seen. If it could attend to positions after $i$, it would be reading the future it is meant to predict, and training would be a cheat that fails at generation time.

We enforce this with a **causal mask** $M$: before softmax, add $-\infty$ to every "future" score (column $j > i$), leaving allowed entries ($j \le i$) untouched:

$$
M = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix},
\qquad
\frac{S}{\sqrt 2} + M = \begin{bmatrix} 1.4142 & -\infty & -\infty \\ 1.4142 & 1.4142 & -\infty \\ 1.4142 & 0 & 1.4142 \end{bmatrix}.
$$

Why $-\infty$ rather than $0$? Because the next step is softmax and $\exp(-\infty) = 0$: a masked score contributes *exactly zero* weight. The future position is not down-weighted, it is switched off entirely, and the surviving weights still sum to $1$.

## 7. Step four — softmax turns each row into weights

Softmax converts a row of scores into non-negative weights summing to $1$ — a probability distribution over "how much to attend to each position." For a row $\mathbf{r}$,

$$
\mathrm{softmax}(\mathbf{r})_j = \frac{\exp(r_j)}{\sum_k \exp(r_k)},
$$

summing only over unmasked entries (masked ones contribute $\exp(-\infty) = 0$). We apply softmax independently to each row, because each row is one query distributing its attention.

**Row 1**, live entries $[1.4142, -\infty, -\infty]$. Only one option survives, so it takes all the weight: $[1, 0, 0]$.

**Row 2**, live entries $[1.4142, 1.4142, -\infty]$. Exponentiate: $\exp(1.4142) \approx 4.1132$ for both. Sum $= 8.2264$. Each weight $= 4.1132 / 8.2264 = 0.5$. So $[0.5, 0.5, 0]$ — a tie, because the two live scores are equal.

**Row 3**, live entries $[1.4142, 0, 1.4142]$. Exponentiate: $\exp(1.4142) \approx 4.1132$, $\exp(0) = 1$, $\exp(1.4142) \approx 4.1132$. Sum $= 9.2264$. Weights:

$$
\frac{4.1132}{9.2264} \approx 0.4458, \quad \frac{1}{9.2264} \approx 0.1084, \quad \frac{4.1132}{9.2264} \approx 0.4458.
$$

Collecting the rows gives the **attention weight matrix**

$$
A = \begin{bmatrix} 1.0000 & 0 & 0 \\ 0.5000 & 0.5000 & 0 \\ 0.4458 & 0.1084 & 0.4458 \end{bmatrix}.
$$

Two sanity checks: every row sums to $1$ ($0.4458 + 0.1084 + 0.4458 = 1$), and the matrix is lower-triangular — the upper-right entries are exactly zero, the causal mask doing its job.

## 8. Step five — output is a weighted average of values

Each query now has weights saying how much to listen to each position. Its output is the corresponding **weighted sum of value vectors**, $\sum_j A_{ij}\,\mathbf{v}_j$. Stacking all queries, this is one more matmul $O = A V$, of shape $(T, T) \times (T, d_h) = (3, 3) \times (3, 2) = (3, 2)$. With $\mathbf{v}_1 = [1,0]$, $\mathbf{v}_2 = [0,1]$, $\mathbf{v}_3 = [1,1]$:

**Row 1:** weights $[1, 0, 0]$ → $\mathbf{v}_1 = [1.0000,\ 0.0000]$.

**Row 2:** weights $[0.5, 0.5, 0]$ → $0.5[1,0] + 0.5[0,1] = [0.5000,\ 0.5000]$.

**Row 3:** weights $[0.4458, 0.1084, 0.4458]$ →

$$
0.4458[1,0] + 0.1084[0,1] + 0.4458[1,1] = [\,0.4458 + 0.4458,\ 0.1084 + 0.4458\,] = [0.8916,\ 0.5542].
$$

$$
O = \begin{bmatrix} 1.0000 & 0.0000 \\ 0.5000 & 0.5000 \\ 0.8916 & 0.5542 \end{bmatrix}.
$$

That is a full scaled dot-product attention worked end to end. Trace the pipeline once more: **scores** ($QK^\top$) → **scale** ($/\sqrt{d_h}$) → **mask** ($+M$) → **weights** (softmax per row) → **output** ($AV$).

## 9. From scratch in PyTorch — and matching the fused kernel

Here is the computation of sections 4–8, reproducing every number, and a check against PyTorch's own fused kernel `F.scaled_dot_product_attention` with `is_causal=True`.

```python
import torch
import torch.nn.functional as F

Q = torch.tensor([[2., 0.], [1., 1.], [0., 2.]])   # (T, d_h) = (3, 2)
K = torch.tensor([[1., 1.], [2., 0.], [0., 1.]])
V = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
T, d_h = Q.shape                                     # 3, 2

scores = Q @ K.transpose(-2, -1) / d_h**0.5          # (T, T); QK^T scaled
mask = torch.tril(torch.ones(T, T))                  # lower-triangular ones
scores = scores.masked_fill(mask == 0, float('-inf'))
weights = F.softmax(scores, dim=-1)                  # per-row weights
out = weights @ V                                    # (T, d_h)
print(out)
# tensor([[1.0000, 0.0000],
#         [0.5000, 0.5000],
#         [0.8916, 0.5542]])

# PyTorch's fused causal kernel computes the identical thing:
ref = F.scaled_dot_product_attention(Q[None, None], K[None, None], V[None, None],
                                     is_causal=True)
print(torch.allclose(out, ref[0, 0], atol=1e-6))     # True
```

Line for line: `Q @ K.transpose(-2, -1)` is $QK^\top$ (the `-2, -1` swap the last two axes, so it works unchanged when there are leading batch/head axes); dividing by `d_h**0.5` is the $1/\sqrt{d_h}$ scale; `masked_fill(mask == 0, -inf)` writes $-\infty$ into future positions; `F.softmax(..., dim=-1)` is the per-row softmax (`dim=-1` normalizes along each row); `weights @ V` is the weighted sum. The from-scratch result matches PyTorch's fused kernel to floating-point tolerance — same math, different bookkeeping.

<div class="callout pt"><p>The from-scratch <code>scaled_dot_product_attention(q, k, v, mask=None)</code> in <code>llmre/attention/attention.py</code> is exactly these three lines (scale, add mask, softmax, matmul), written to accept the leading <code>(B, nh)</code> batch/head axes. The test <code>test_scaled_dot_product_attention_matches_torch_causal</code> in <code>code/tests/test_attention.py</code> checks it against <code>F.scaled_dot_product_attention(..., is_causal=True)</code> to 1e-5.</p></div>

## 10. Tensor shapes: where the $T \times T$ comes from

In a real block the tensors carry leading batch and head axes, but the last two axes are what attention operates on. For a single head:

| tensor | shape | meaning |
|---|---|---|
| $x$ | $(B, T, C)$ | input activations |
| $Q, K, V$ | $(B, T, d_h)$ | three linear projections of $x$ |
| $S = QK^\top$ | $(B, T, T)$ | one score per (query, key) pair |
| $A = \mathrm{softmax}(S + M)$ | $(B, T, T)$ | attention weights |
| $O = AV$ | $(B, T, d_h)$ | attended output, same shape as $V$ |

The output has the same shape as the input to attention, which is what lets attention slot into a stack of layers. The dtype is whatever $x$ is (`float32` or, in mixed precision, `bfloat16`); everything stays on $x$'s device.

## 11. Under the hood: the $T \times T$ matrix is the bottleneck

Look at $S$ and $A$: both are $(B, T, T)$. Their size grows with $T^2$. For a context of $T = 1024$ that is about a million entries per sequence per head; at $T = 8192$ it is 67 million. This quadratic scaling in sequence length is the defining cost of attention:

- **Memory.** Materializing $A$ of shape $(B, n_h, T, T)$ dominates activation memory for long contexts. At $T = 8192$, $B = 8$, $n_h = 12$ in `float32`, the weight matrix alone is $8 \cdot 12 \cdot 8192^2 \cdot 4$ bytes $\approx$ 26 GB — larger than most GPUs, just for one layer's attention weights.
- **Compute.** Forming $S$ costs $O(B \cdot n_h \cdot T^2 \cdot d_h)$ multiply-adds, and $AV$ costs the same order. Both grow as $T^2$.

The mathematics does not require ever storing the full $T \times T$ matrix, though. Production **FlashAttention** kernels compute the identical output by streaming over the score matrix in tiles, keeping a running softmax (a running max and sum) and accumulating the weighted value sum on the fly — never materializing $A$. The result is bit-for-bit the same weighted average you computed by hand; only the memory traffic changes. We build the intuition and a Triton kernel in Module 8.

<div class="callout paper"><p>FlashAttention is paper #7 in the <a href="../../papers/index.md">paper curriculum</a> (Dao et al., 2022), read after Module 8, and covered in <a href="../module-08/lesson-04.md">08.4 · FlashAttention &amp; a Triton kernel</a>. Its one-line thesis: attention is bottlenecked by memory reads/writes of that $T\times T$ matrix, not by FLOPs, so avoiding the materialization — not doing fewer multiplies — is the win.</p></div>

**Gradients.** Autograd carries backward rules for `@`, `softmax`, and `masked_fill`, so `loss.backward()` differentiates the whole block using the vector–Jacobian products from Module 2. Softmax has a known Jacobian; the masked (`-inf`) entries receive zero gradient, so no signal flows to forbidden positions — the causal structure is respected in the backward pass too.

## Exercise

With $C = 768$ and a single head so $d_h = C = 768$, a batch of $B = 4$ sequences of length $T = 512$: what is the shape of the score matrix $S$, and roughly how many entries does it hold?

*Optional hint:* scores are one number per (query position, key position) pair, per sequence.

*Stronger hint:* the shape is $(B, T, T)$; multiply it out.

<details><summary>Solution</summary>

$S$ has shape $(B, T, T) = (4, 512, 512)$. That is $4 \cdot 512 \cdot 512 = 1{,}048{,}576$ entries — about a million scores. Note $d_h$ does *not* appear in the score-matrix shape: it was summed away by the dot product. The head dimension controls the *cost* of forming each score, not the *size* of $S$.

</details>

## Common mistakes

- **Forgetting the scale.** Dropping the $1/\sqrt{d_h}$ makes scores large, softmax saturates, and training stalls or diverges — especially at realistic $d_h = 64$. Symptom: attention weights collapse to near one-hot almost immediately.
- **Masking after softmax.** Zeroing out future *weights* after softmax leaves the surviving weights summing to less than 1, breaking the weighted-average interpretation. The mask must be added ($-\infty$) *before* softmax.
- **Wrong softmax axis.** `F.softmax(scores, dim=-2)` normalizes down columns (over queries) instead of across keys. Each *query* must distribute its own attention, so it is `dim=-1`.

## Check yourself

<details><summary>Why divide the scores by $\sqrt{d_h}$ before softmax, and what breaks without it?</summary>

A dot product over $d_h$ dimensions has variance proportional to $d_h$, so scores grow with head size. Large scores make softmax saturate — near-one-hot weights, a flat gradient, stalled learning. Dividing by $\sqrt{d_h}$ rescales the scores back to unit variance so softmax stays soft and trainable, independent of $d_h$.

</details>

<details><summary>In the worked example row 2 came out $[0.5, 0.5, 0]$. Why exactly a tie?</summary>

Position 2 may attend to positions 1 and 2 (position 3 is masked). Its two live scaled scores were equal ($1.4142$ and $1.4142$), and softmax of two equal values is $[0.5, 0.5]$. The equal scores trace back to $\mathbf{q}_2 = [1,1]$ dotting $\mathbf{k}_1 = [1,1]$ and $\mathbf{k}_2 = [2,0]$ both giving $2$.

</details>

<details><summary>Why is the mask added as $-\infty$ before softmax rather than by zeroing weights afterward?</summary>

Because $\exp(-\infty) = 0$: a $-\infty$ score contributes exactly zero to both the softmax numerator and its normalizing sum, so the surviving weights still sum to $1$ — a valid distribution. Zeroing weights after softmax would leave them summing to less than $1$, breaking the weighted average.

</details>

<details><summary>For $B = 8$ sequences, $n_h = 12$ heads, $T = 1024$, what is the shape of the attention-weight tensor, and why is it the memory bottleneck?</summary>

$(B, n_h, T, T) = (8, 12, 1024, 1024)$ — over 100 million entries. Because it grows as $T^2$, it dominates activation memory for long contexts; FlashAttention (Module 8) computes the same output without ever storing it.

</details>

<details><summary>The output $O = AV$ has shape $(B, T, d_h)$, the same as $V$. Why does that shape-preservation matter?</summary>

Because it lets attention be stacked: its output can feed the next sub-layer (and, after merging heads, the next block) with the same shape convention. A layer that changed the per-position width would break the residual-stream structure the whole transformer relies on.

</details>

## Next

You can now compute single-head scaled dot-product attention by hand and read its three-line PyTorch core. But a real transformer runs *several* attentions in parallel, each on its own slice of the channels, so different heads can specialize in different kinds of relationships. The next lesson splits the channel dimension into heads and shows every shape of multi-head attention.

Continue to [05.3 · Multi-head attention](lesson-03.md).
