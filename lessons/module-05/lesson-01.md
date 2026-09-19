# 05.1 · Embeddings & positional encoding

<div class="prereq">
<p><strong>Prerequisites:</strong> the tokenizer output — a sequence of integer token ids — from <a href="../module-04/lesson-03.md">04.3 · Special tokens, packing, data loading</a>; tensor shape, dtype and device from <a href="../module-00/lesson-01.md">00.1 · Tensors: shape, dtype, device</a>; matrix indexing and the $(B, T)$ / $(B, T, C)$ conventions.</p>
<p><strong>You will learn:</strong> how a token <em>id</em> (a bare integer) becomes a learned vector through the embedding table <code>wte</code>; why you cannot feed raw ids into a network; how a second table <code>wpe</code> injects <em>position</em> so the model knows word order; and how the two are added to form the input to the first transformer block. You will trace one concrete id through both lookups with tiny numbers.</p>
<p><strong>Why this matters for ML:</strong> embeddings are the model's entire interface to language — every downstream layer only ever sees these vectors, never the original text. Positional embeddings are what turn a permutation-blind bag of vectors into a model of ordered sequences. Get this layer right and the rest of GPT is arithmetic on top of it.</p>
</div>

## 1. Intuition: from integers to meaning

The tokenizer (Module 4) hands us a sequence of integers — token ids. A sentence like `"the cat sat"` might become `[464, 3797, 3332]`. Each id is just an index into the vocabulary; the number `3797` is not "bigger" or "more" than `464` in any meaningful sense. They are names, not quantities.

A neural network, though, computes with *quantities*: it multiplies, adds, and differentiates real numbers. So we need to turn each id into a vector of real numbers that the network can compute with — and, crucially, a vector whose values *carry learned meaning*, so that tokens used in similar ways end up with similar vectors.

That is exactly what an **embedding** is: a lookup table with one learned row per vocabulary entry. Token id $i$ selects row $i$ of the table, a vector of $C$ numbers. Those numbers start random and are trained by gradient descent along with everything else, so by the end of training the row for `"cat"` and the row for `"dog"` sit near each other, while `"cat"` and `"parliament"` sit far apart. The table *is* the model's learned dictionary of word meanings.

<div class="callout key"><p>An embedding table is a matrix <code>wte</code> of shape <code>(vocab_size, n_embd)</code>. Looking up token id $i$ means "take row $i$." That single indexing operation is how discrete language enters a continuous network.</p></div>

## 2. Why raw ids cannot feed a network

It is worth being precise about *why* we cannot just feed `[464, 3797, 3332]` into a linear layer directly. Three reasons:

1. **Magnitude is meaningless.** Token id `50000` is not "100× more" than id `500`. But a linear layer $Wx$ treats its input as a magnitude — it would multiply that `50000` by a weight and produce a huge activation, purely because of an arbitrary id assignment. The numeric value of an id encodes no useful structure.
2. **No notion of similarity.** Ids `464` and `465` might be `"the"` and `"zebra"` — adjacent numbers, unrelated meanings. A single scalar cannot express "these two tokens play similar roles." A $C$-dimensional vector can: similarity becomes geometric closeness.
3. **One scalar is too little capacity.** A word's usage is multi-faceted (part of speech, topic, sentiment, formality, ...). One number cannot hold that; a vector of $C = 768$ numbers can dedicate different directions to different facets.

The classic fix you may have seen is a **one-hot** vector: represent id $i$ as a length-$V$ vector that is $1$ at position $i$ and $0$ elsewhere. Multiplying that one-hot row-vector by a weight matrix $W$ of shape $(V, C)$ gives $\text{onehot}(i)\, W = W_i$ — exactly row $i$ of $W$. So an embedding lookup *is* a one-hot times a weight matrix, but done without ever materializing the huge $V$-long one-hot or doing the wasteful multiply-by-zeros. The `wte` table is that $W$, and indexing is the efficient form of the multiplication.

## 3. The token embedding table `wte`

Formally, the token embedding is a learned matrix

$$
W_{te} \in \mathbb{R}^{V \times C},
$$

where $V = $ `vocab_size` (e.g. $50257$ for GPT-2) and $C = $ `n_embd` (e.g. $768$). Its row $i$, written $W_{te}[i] \in \mathbb{R}^{C}$, is the embedding vector of token id $i$.

Given a batch of token ids `idx` of shape $(B, T)$ — $B$ sequences, each $T$ positions long, dtype `int64` — the lookup produces

$$
\text{tok}[b, t, :] = W_{te}[\ \text{idx}[b, t]\ ], \qquad \text{tok} \in \mathbb{R}^{B \times T \times C}.
$$

Read that carefully: we replace each scalar id at position $(b, t)$ with the whole $C$-vector living in that id's row. The shape therefore grows by one axis:

$$
(B, T) \;\xrightarrow{\text{embedding lookup}}\; (B, T, C).
$$

Every later layer of GPT operates on that $(B, T, C)$ tensor — the "residual stream." The integers are gone; from here on the model sees only vectors.

## 4. Numerical example: one id through the table

Let us make it concrete with a toy vocabulary of $V = 5$ tokens and width $C = 3$. Here is a `wte` table (in real life these numbers are learned; here we pick them so we can do the arithmetic):

$$
W_{te} = \begin{bmatrix}
0.10 & -0.20 & 0.30 \\
0.40 & 0.50 & -0.60 \\
-0.70 & 0.80 & 0.90 \\
0.00 & 0.10 & 0.20 \\
0.30 & -0.40 & 0.50
\end{bmatrix}
\quad \text{(rows are ids } 0,1,2,3,4\text{)}.
$$

Take the token sequence `idx = [[2, 0, 4]]` (one sequence, $B=1$, $T=3$). The lookup replaces each id by its row:

- id $2$ → row $2$ = $[-0.70,\ 0.80,\ 0.90]$
- id $0$ → row $0$ = $[\ 0.10,\ -0.20,\ 0.30]$
- id $4$ → row $4$ = $[\ 0.30,\ -0.40,\ 0.50]$

So

$$
\text{tok} = \begin{bmatrix}
-0.70 & 0.80 & 0.90 \\
0.10 & -0.20 & 0.30 \\
0.30 & -0.40 & 0.50
\end{bmatrix}, \qquad \text{shape } (1, 3, 3).
$$

That is the entire token-embedding step: three integers became three learned 3-vectors. We verify it in PyTorch in section 8; the printed tensor matches these numbers exactly.

## 5. The order problem: attention is permutation-blind

We now have one vector per position. But there is a subtle catastrophe waiting. The attention mechanism you will build in the next lesson treats its input as a *set*, not a *sequence*: if you shuffle the positions, attention shuffles its outputs the same way but computes the same values. Attention has no built-in idea of "position 1 comes before position 2." To it, `"dog bites man"` and `"man bites dog"` would produce the same bag of vectors.

That is fatal for language, where order is meaning. So we must *inject* position information into the vectors themselves, before attention ever sees them. The model should receive, at each position, not just "which token" but "which token, and where."

## 6. Learned absolute positional embeddings `wpe`

GPT-2's answer is beautifully simple: keep a **second** learned table, indexed by *position* instead of by token id.

$$
W_{pe} \in \mathbb{R}^{T_{\max} \times C},
$$

where $T_{\max} = $ `block_size` (the maximum context length, e.g. $1024$) and $C$ is the same embedding width. Row $t$ is a learned vector describing "what it means to be at position $t$." Then we simply **add** the positional vector to the token vector at each position:

$$
h[b, t, :] = \underbrace{W_{te}[\ \text{idx}[b,t]\ ]}_{\text{which token}} + \underbrace{W_{pe}[t]}_{\text{which position}}.
$$

Both terms are $C$-vectors, so the sum is a $C$-vector: the shape stays $(B, T, C)$. The positional table has no batch dependence — position $t$ gets the same vector $W_{pe}[t]$ in every sequence — so it broadcasts across the batch.

Why *add* rather than concatenate? Adding keeps the width at $C$ (concatenation would double it and force every downstream matrix to be wider). Because both tables are learned jointly, the model can carve out "directions" in the $C$-dimensional space for positional signal and other directions for token identity; it learns to keep them separable enough to use both. This is the "absolute" scheme: each position $0, 1, 2, \dots$ gets its own free vector, learned from scratch.

<div class="callout warn"><p>Learned absolute positions have a hard limit: <code>wpe</code> only has <code>block_size</code> rows. The model literally has no vector for position <code>block_size</code> or beyond, so a GPT-2-style model cannot process a sequence longer than its trained context — there is no row to look up. This is one motivation for the relative schemes in Module 12.</p></div>

## 7. Numerical example: adding position

Continue the toy example. Add a positional table (again, chosen for easy arithmetic):

$$
W_{pe} = \begin{bmatrix}
0.01 & 0.02 & 0.03 \\
0.04 & 0.05 & 0.06 \\
0.07 & 0.08 & 0.09
\end{bmatrix}
\quad \text{(rows are positions } 0, 1, 2\text{)}.
$$

Our sequence has $T = 3$ positions, so we use rows $0, 1, 2$ of $W_{pe}$ and add them to the token embeddings from section 4, elementwise, row by row:

- position 0: $[-0.70, 0.80, 0.90] + [0.01, 0.02, 0.03] = [-0.69,\ 0.82,\ 0.93]$
- position 1: $[\ 0.10, -0.20, 0.30] + [0.04, 0.05, 0.06] = [\ 0.14,\ -0.15,\ 0.36]$
- position 2: $[\ 0.30, -0.40, 0.50] + [0.07, 0.08, 0.09] = [\ 0.37,\ -0.32,\ 0.59]$

$$
h = \begin{bmatrix}
-0.69 & 0.82 & 0.93 \\
0.14 & -0.15 & 0.36 \\
0.37 & -0.32 & 0.59
\end{bmatrix}, \qquad \text{shape } (1, 3, 3).
$$

This $h$ is the input to the first transformer block. Notice the same token id appearing at two different positions would now produce two *different* $h$ vectors — the model can tell them apart. Order information is in the vectors.

## 8. From scratch in PyTorch

Two `nn.Embedding` layers plus an add. `nn.Embedding` is exactly the lookup table of section 3 — its `.weight` is the $(V, C)$ (or $(T_{\max}, C)$) matrix, and calling it with integer indices returns the selected rows.

```python
import torch
import torch.nn as nn

V, C, T_max = 5, 3, 3          # tiny toy sizes

wte = nn.Embedding(V, C)       # token table:    (V, C) = (5, 3)
wpe = nn.Embedding(T_max, C)   # position table: (T_max, C) = (3, 3)

# Overwrite the random init with our worked-example tables so numbers match.
with torch.no_grad():
    wte.weight.copy_(torch.tensor([
        [ 0.10, -0.20,  0.30],
        [ 0.40,  0.50, -0.60],
        [-0.70,  0.80,  0.90],
        [ 0.00,  0.10,  0.20],
        [ 0.30, -0.40,  0.50]]))
    wpe.weight.copy_(torch.tensor([
        [ 0.01, 0.02, 0.03],
        [ 0.04, 0.05, 0.06],
        [ 0.07, 0.08, 0.09]]))

idx = torch.tensor([[2, 0, 4]])          # (B, T) = (1, 3), dtype int64
B, T = idx.shape

tok = wte(idx)                           # (B, T, C) = (1, 3, 3)
pos = wpe(torch.arange(T))               # (T, C)    = (3, 3), positions 0..T-1
h = tok + pos                            # (B, T, C); pos broadcasts over batch
print(h)
# tensor([[[-0.6900,  0.8200,  0.9300],
#          [ 0.1400, -0.1500,  0.3600],
#          [ 0.3700, -0.3200,  0.5900]]])
```

Line by line: `wte(idx)` takes the integer tensor `idx` and gathers rows — the `(B, T)` id grid becomes a `(B, T, C)` vector grid. `torch.arange(T)` builds `[0, 1, 2]`, the position indices, and `wpe(...)` gathers their rows into a `(T, C)` tensor. The add `tok + pos` broadcasts the `(T, C)` positional tensor across the batch axis of the `(B, T, C)` token tensor — the same positional vectors are reused for every sequence in the batch. The printed numbers match section 7 to four decimals.

<div class="callout pt"><p><code>nn.Embedding(V, C)</code> is not doing a matrix multiply at runtime — it is an indexed gather. Under the hood it reads the selected rows directly out of the <code>(V, C)</code> weight buffer. That is why embedding a token is $O(C)$ work, not $O(V \cdot C)$: the one-hot multiply of section 2 is short-circuited into a memory read.</p></div>

## 9. Tensor shapes, dtype, device — the full accounting

| tensor | shape | dtype | notes |
|---|---|---|---|
| `idx` | $(B, T)$ | `int64` | token ids from the tokenizer; must be `long` for indexing |
| `wte.weight` | $(V, C)$ | `float32` | learned; the token dictionary |
| `wpe.weight` | $(T_{\max}, C)$ | `float32` | learned; one row per position slot |
| `tok` | $(B, T, C)$ | `float32` | gathered token vectors |
| `pos` | $(T, C)$ | `float32` | gathered position vectors, broadcast over $B$ |
| `h` | $(B, T, C)$ | `float32` | input to the first block |

Two things to keep straight. First, `idx` must be an **integer** tensor (`int64`/`long`); indexing an embedding with floats raises an error, because you cannot use a fractional row number. Second, everything must be on the **same device**: if the model is on the GPU, `torch.arange(T)` must also be moved to that device (`device=idx.device`) or the add fails with a device-mismatch error.

<div class="callout warn"><p>A classic bug: your ids include a value $\ge$ <code>vocab_size</code> (e.g. a special token you forgot to size the vocab for). The embedding lookup then indexes past the end of the table. On CPU you get a clean <code>IndexError</code>; on CUDA it often surfaces later as a corrupted-memory crash far from the real cause. Always assert <code>idx.max() &lt; vocab_size</code> when debugging.</p></div>

## 10. Under the hood: memory, parameters, and gradients

**Parameter cost.** For GPT-2 small, the two tables dominate the "non-transformer" parameters:

- `wte`: $V \times C = 50257 \times 768 \approx 38.6$M parameters.
- `wpe`: $T_{\max} \times C = 1024 \times 768 \approx 0.79$M parameters.

The token table alone is about 31% of GPT-2 small's 124M parameters — embeddings are not a footnote, they are a large fraction of the model. (In Module 6 you will see GPT-2 *ties* the `wte` matrix to the output projection, reusing those 38.6M parameters at both ends so they are not paid for twice.)

**What the gradient does.** Because the lookup gathers only the rows for the ids actually present in the batch, the backward pass writes gradients into *only those rows*. A token that never appears in a batch gets zero gradient that step — its embedding does not move. So rare tokens learn slowly (they are updated only on the rare steps they appear), a real effect that makes the embedding for a token seen five times far noisier than one seen five million times. Autograd handles this automatically via a scatter-add of the incoming gradients back into `wte.weight` at the used row indices.

**Compute.** The forward lookup is a gather: $B \cdot T$ row reads of $C$ floats each, essentially free compared to the matrix multiplies inside a block. The positional add is $B \cdot T \cdot C$ additions, also negligible. The embedding layer costs memory (the big tables) far more than it costs compute.

## 11. A brief contrast: sinusoidal and what comes later

GPT-2 uses *learned* absolute positions. The original Transformer paper instead used *fixed sinusoidal* positional encodings — for position $t$ and channel $j$, a deterministic function like $\sin(t / 10000^{\,2j/C})$ and $\cos(\cdot)$ — computed, not learned. The advantage of sinusoidal is that it is defined for *any* position, even beyond those seen in training, and its values for nearby positions are smoothly related. The advantage of learned (GPT-2's choice) is flexibility: the model can shape each position vector however training finds useful. Both are *absolute*: they answer "where am I" with a per-position vector added to the token.

Modern models mostly abandon both in favor of **relative** schemes, above all **RoPE** (Rotary Position Embedding), which encodes position by *rotating* the query and key vectors inside attention so that the attention score depends only on the *distance* between two positions, not their absolute indices. That extrapolates to longer contexts far better than a fixed-size `wpe` table. We build RoPE from scratch in Module 12.

<div class="callout paper"><p>RoPE is paper #8 in the <a href="../../papers/index.md">paper curriculum</a> (Su et al., 2021), read after Module 12. When you get there, the contrast to keep in mind is exactly this lesson's <code>wpe</code>: a learned lookup added once at the input, versus a rotation applied to Q and K at every layer.</p></div>

## Exercise

You have a model with `n_embd = 768` and a token embedding `wte` of shape `(50257, 768)`. You embed a batch `idx` of shape `(4, 128)`.

*Optional hint:* the embedding lookup adds exactly one axis.

*Stronger hint:* the new axis has length `n_embd`, appended last.

<details><summary>Solution</summary>

The output `tok = wte(idx)` has shape `(4, 128, 768)`: the `(B, T) = (4, 128)` id grid keeps its two axes and gains a trailing channel axis of length `768`. Dtype is `float32` (the table's dtype), even though `idx` was `int64`. The positional part `wpe(torch.arange(128))` has shape `(128, 768)` and broadcasts across the batch of 4 when added.

</details>

## Debugging exercise

A learner writes the input layer like this and gets a runtime error the moment $T$ exceeds a small value. What is the bug?

```python
class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_head, cfg.n_embd)   # <-- ?
    def forward(self, idx):
        B, T = idx.shape
        tok = self.wte(idx)
        pos = self.wpe(torch.arange(T, device=idx.device))
        return tok + pos
```

<details><summary>Solution</summary>

The positional table is sized `nn.Embedding(cfg.n_head, cfg.n_embd)` — it uses `n_head` (e.g. 12) as the number of positions instead of `block_size`. As soon as the sequence length $T$ exceeds `n_head`, `torch.arange(T)` produces an index past the end of `wpe`, and the lookup raises an `IndexError` (CPU) or crashes CUDA. It must be `nn.Embedding(cfg.block_size, cfg.n_embd)`: one row per position slot, up to the maximum context length.

</details>

## Check yourself

<details><summary>Why can't we feed token ids like <code>[464, 3797, 3332]</code> directly into a linear layer?</summary>

Because a linear layer treats its input as magnitudes, and token ids are arbitrary names — id `3797` is not "7× more" than `464`. Ids carry no similarity structure (adjacent ids can be unrelated words) and a single scalar has no capacity to represent a word's many facets. The embedding table replaces each id with a learned $C$-vector so that meaning becomes geometry.

</details>

<details><summary>What are the shapes of <code>wte.weight</code>, <code>wpe.weight</code>, and the output <code>h</code>, for a batch of shape <code>(B, T)</code>?</summary>

`wte.weight` is $(V, C) = ($`vocab_size`$,$ `n_embd`$)$. `wpe.weight` is $(T_{\max}, C) = ($`block_size`$,$ `n_embd`$)$. The token lookup gives $(B, T, C)$, the position lookup gives $(T, C)$, and after the broadcast add, $h$ is $(B, T, C)$.

</details>

<details><summary>Why do we add the positional embedding at all — what breaks without it?</summary>

Attention is permutation-invariant: without position information it treats the sequence as an unordered set, so `"dog bites man"` and `"man bites dog"` would be indistinguishable. Adding `wpe[t]` gives each position a distinct signature, so the same token at different positions yields different vectors and the model can use word order.

</details>

<details><summary>In the toy example, the token at position 0 was id 2 with row <code>[-0.70, 0.80, 0.90]</code>. Why is its value in <code>h</code> <code>[-0.69, 0.82, 0.93]</code> and not the row itself?</summary>

Because we add the positional embedding for position 0, $[0.01, 0.02, 0.03]$: $[-0.70, 0.80, 0.90] + [0.01, 0.02, 0.03] = [-0.69, 0.82, 0.93]$. The value in $h$ is always token embedding plus position embedding.

</details>

<details><summary>GPT-2 uses learned absolute positions. Name one concrete limitation this imposes at inference time.</summary>

The context length is hard-capped at `block_size`: `wpe` has exactly that many rows, so there is no positional vector to look up for any position at or beyond `block_size`. The model cannot process a longer sequence at all without changing the position scheme (e.g. switching to RoPE, Module 12).

</details>

## Next

You can now turn a sequence of token ids into the position-aware vector grid $h$ of shape $(B, T, C)$ that a transformer block consumes. The next lesson takes that $h$ and builds the operation at the heart of the block — scaled dot-product attention — the mechanism by which each position gathers information from the others.

Continue to [05.2 · Q/K/V & scaled dot-product attention](lesson-02.md).
