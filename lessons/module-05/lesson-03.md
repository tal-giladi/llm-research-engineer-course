# 05.3 · Multi-head attention

<div class="prereq">
<p><strong>Prerequisites:</strong> single-head scaled dot-product attention — scores, scale, causal mask, softmax, weighted value sum — from <a href="lesson-02.md">05.2 · Q/K/V &amp; scaled dot-product attention</a>; the reshape / view / transpose machinery and the $(B, T, C)$ convention from Module 0.</p>
<p><strong>You will learn:</strong> why one attention is not enough; how to split the channel width $C$ into $n_h$ heads of size $d_h = C/n_h$; the exact reshape/transpose that turns $(B, T, C)$ into $(B, n_h, T, d_h)$ so all heads run in one batched matmul; how to merge the heads back and apply the output projection $W_O$; and every intermediate shape.</p>
<p><strong>Why this matters for ML:</strong> multi-head attention is the form used in every real transformer. Understanding the split-and-merge is what lets you read any GPT attention module, reason about its memory, and later modify it (GQA, MQA, MLA in Module 12 all change exactly this head structure).</p>
</div>

## 1. Intuition: several relationships at once

A single attention lets each position gather *one* blended view of the others. But a token usually relates to different positions for different *reasons* simultaneously: one relationship is grammatical agreement (a verb to its subject), another is coreference (a pronoun to its noun), another is positional (the previous token). Forcing all of that through a single set of $Q, K, V$ vectors is limiting — the position would have to compromise on one blend.

**Multi-head attention** runs several attentions in parallel, each on its own slice of the feature channels, so each "head" can specialize in a different kind of relationship. The heads do not share weights and do not see each other's scores; they are fully independent attentions until the very end, where their outputs are combined.

<div class="callout key"><p>Multi-head attention = split $C$ into $n_h$ independent subspaces of size $d_h = C/n_h$, run the entire scaled dot-product attention of 05.2 in each, concatenate the $n_h$ outputs back to width $C$, then mix them with a learned output projection $W_O$. Same math as one head, done $n_h$ times in parallel on different channel slices.</p></div>

## 2. The split: $C = n_h \cdot d_h$

The full hidden vector at a position has $C$ channels. We choose a number of heads $n_h$ that divides $C$ evenly, and set the per-head dimension $d_h = C / n_h$. For GPT-2 small, $C = 768$ and $n_h = 12$, so $d_h = 64$. Each head will operate on its own contiguous block of $d_h = 64$ channels.

Multi-head attention first produces $Q, K, V$ each of shape $(B, T, C)$ — the full width — using a single linear projection. Then it *reshapes* the channel axis into $(n_h, d_h)$ and moves the head axis next to the batch axis:

$$
(B, T, C) \;\xrightarrow{\text{view}}\; (B, T, n_h, d_h) \;\xrightarrow{\text{transpose(1,2)}}\; (B, n_h, T, d_h).
$$

The `view` re-groups the $C$ channels into $n_h$ groups of $d_h$ *without moving any data* — it only reinterprets how the existing contiguous memory is indexed. The `transpose(1, 2)` swaps the $T$ and $n_h$ axes so the head axis sits right after the batch axis. After it, the last two axes are $(T, d_h)$ — exactly the little $Q, K, V$ matrices attention operates on — and there are $B \cdot n_h$ of them, one per (sequence, head).

## 3. Numerical example: splitting one vector into heads

Take $C = 4$ and $n_h = 2$, so $d_h = 2$. Consider a single position whose query vector is

$$
\mathbf{q} = [\,q_0,\ q_1,\ q_2,\ q_3\,] = [\,10,\ 20,\ 30,\ 40\,].
$$

The `view` into $(n_h, d_h) = (2, 2)$ splits the four channels into two contiguous halves:

$$
\text{head 0} = [\,10,\ 20\,], \qquad \text{head 1} = [\,30,\ 40\,].
$$

Head 0 gets channels $0{:}2$, head 1 gets channels $2{:}4$. That is the whole "split" — the first $d_h$ channels are head 0's query, the next $d_h$ are head 1's, and so on. Each head then runs its *own* scaled dot-product attention using only its slice; head 0 never sees channels $2$ and $3$, and head 1 never sees channels $0$ and $1$. Because the slices sit on a separate axis, all $n_h$ heads compute at once as a batched matmul — no Python loop over heads.

## 4. Attention per head, in one batched matmul

Once $q, k, v$ have shape $(B, n_h, T, d_h)$, the attention formula from 05.2 runs unchanged — PyTorch's `@` treats every axis before the last two as a batch axis and loops over them in parallel:

$$
S = q\, k^\top : (B, n_h, T, d_h) \times (B, n_h, d_h, T) \to (B, n_h, T, T).
$$

So there is one $T \times T$ score matrix *per head per sequence*. The scale $1/\sqrt{d_h}$, the causal mask, and the softmax all apply exactly as before; the mask $(T, T)$ **broadcasts** across the leading $(B, n_h)$ axes, so one small triangular matrix masks every head and every sequence at once. Then

$$
\text{head outputs} = A\, v : (B, n_h, T, T) \times (B, n_h, T, d_h) \to (B, n_h, T, d_h).
$$

Each head has produced its own $(T, d_h)$ attended output, independently.

## 5. The merge: concatenate and project

Each head produced $(B, n_h, T, d_h)$. To hand a single $C$-vector per position back to the rest of the network, we reverse the split — transpose the head axis back down and merge $n_h$ and $d_h$ into $C$:

$$
(B, n_h, T, d_h) \;\xrightarrow{\text{transpose(1,2)}}\; (B, T, n_h, d_h) \;\xrightarrow{\text{view}}\; (B, T, C).
$$

Merging the head axis back into the channel axis is **concatenation**: for each position we lay the $n_h$ head outputs (each $d_h$ long) end to end into one $C$-long vector. In the $C = 4$, $n_h = 2$ example, if head 0 output $[a_0, a_1]$ and head 1 output $[b_0, b_1]$, the merged vector is $[a_0, a_1, b_0, b_1]$ — the exact inverse of the split in section 3.

Finally we multiply by a learned **output projection** $W_O$ of shape $(C, C)$:

$$
\text{MultiHead} = \mathrm{Concat}(\text{head}_1, \dots, \text{head}_{n_h})\, W_O.
$$

$W_O$ lets the model mix information *across* heads. Without it, the head outputs would just sit in disjoint channel slices, unable to combine — head 0's finding could never influence the channels head 1 wrote. $W_O$ blends the independent per-head results into the final output, which has shape $(B, T, C)$: the same width as the input, ready for the residual add and the next sub-layer.

<div class="callout warn"><p>The <code>transpose(1, 2)</code> before the final <code>view</code> makes the tensor non-contiguous in memory (the axes no longer match the memory layout). <code>view</code> requires contiguous memory, so real code calls <code>.contiguous()</code> first: <code>y.transpose(1, 2).contiguous().view(B, T, C)</code>. Skip the <code>.contiguous()</code> and PyTorch raises a runtime error telling you the view is incompatible with the stride layout.</p></div>

## 6. From scratch in PyTorch — the `CausalSelfAttention` module

This is the module the rest of the course uses (`llmre/attention/attention.py`). It takes a `GPTConfig`, does the QKV projection, the split, per-head causal attention, the merge, and the output projection.

```python
import torch
import torch.nn as nn
from llmre.attention.attention import scaled_dot_product_attention  # the 05.2 core

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.d_h = cfg.n_embd // cfg.n_head
        # One fused Linear producing Q, K, V stacked: (B, T, C) -> (B, T, 3C).
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # Output projection W_O: (C -> C), mixes the concatenated heads.
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        # Causal mask (1, 1, block_size, block_size): 0 on/below diag, -inf above.
        mask = torch.triu(torch.full((cfg.block_size, cfg.block_size), float('-inf')),
                          diagonal=1)
        self.register_buffer('attn_mask', mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        # Project once, split the 3C axis into three (B, T, C) tensors.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Split channels into heads: (B, T, C) -> (B, T, nh, d_h) -> (B, nh, T, d_h).
        q = q.view(B, T, self.n_head, self.d_h).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_h).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_h).transpose(1, 2)
        # Per-head causal attention; mask sliced to T and broadcast over (B, nh).
        y = scaled_dot_product_attention(q, k, v, mask=self.attn_mask[:, :, :T, :T])
        # Merge heads back: (B, nh, T, d_h) -> (B, T, nh, d_h) -> (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)                                  # output projection
```

Why one fused `c_attn` `Linear` producing $3C$ instead of three separate `Linear`s of width $C$? It is the same math — the three weight matrices $W_Q, W_K, W_V$ are stacked into one $(C, 3C)$ matrix — but a single larger matmul is more efficient on a GPU than three smaller ones, and `.split(self.n_embd, dim=2)` just carves the $3C$-wide result back into the three $(B, T, C)$ pieces. The causal mask is built once in `__init__` as a registered buffer (not a parameter — it carries no gradient) so it moves with `.to(device)` and is sliced to the current $T$ each forward.

<div class="callout pt"><p>The test <code>test_causal_self_attention_forward_shape</code> in <code>code/tests/test_attention.py</code> checks that <code>forward</code> maps $(B, T, C)$ to $(B, T, C)$, and <code>test_causality_future_does_not_leak_into_past</code> perturbs the token at one position and asserts that outputs at all <em>earlier</em> positions are unchanged — the causal mask, verified rather than assumed.</p></div>

## 7. Every shape, end to end

For a concrete forward pass with $B = 8$, $T = 1024$, $C = 768$, $n_h = 12$ (so $d_h = 64$):

| step | tensor | shape |
|---|---|---|
| input | $x$ | $(8, 1024, 768)$ |
| after `c_attn` | $qkv$ | $(8, 1024, 2304)$ |
| after `split` | $q, k, v$ each | $(8, 1024, 768)$ |
| after split into heads | $q, k, v$ each | $(8, 12, 1024, 64)$ |
| scores $qk^\top$ | $S$ | $(8, 12, 1024, 1024)$ |
| weights (softmax) | $A$ | $(8, 12, 1024, 1024)$ |
| head outputs $Av$ | $y$ | $(8, 12, 1024, 64)$ |
| after merge | $y$ | $(8, 1024, 768)$ |
| after `c_proj` | output | $(8, 1024, 768)$ |

Notice the input and output shapes are identical, $(8, 1024, 768)$ — attention is a shape-preserving map on the residual stream. Notice too that the total per-position width across all heads, $n_h \cdot d_h = 12 \cdot 64 = 768 = C$: the heads *partition* the channels, they do not add width. Multi-head attention costs the same $Q,K,V$ compute as single-head attention of width $C$ would; splitting into heads is essentially free, it just re-slices the same numbers.

## 8. Under the hood: why the head axis goes next to the batch axis

The `transpose(1, 2)` that produces $(B, n_h, T, d_h)$ is not cosmetic. PyTorch's batched `@` treats *all* axes before the last two as batch axes and dispatches an independent matmul for each combination. By placing $n_h$ right after $B$, the operation `q @ k.transpose(-2, -1)` becomes $B \cdot n_h$ independent $(T, d_h) \times (d_h, T)$ matmuls, all launched together on the GPU with no Python loop. Had we left the head axis in the middle as $(B, T, n_h, d_h)$, the last two axes would be $(n_h, d_h)$ and the matmul would multiply the wrong dimensions. The layout *is* the parallelism.

**Memory.** The dominant cost remains the score/weight tensors $(B, n_h, T, T)$ — quadratic in $T$, now also linear in $n_h$. More heads means more $T \times T$ matrices (though each is over a smaller $d_h$, so forming each is cheaper). This is the same $T^2$ bottleneck as 05.2, and FlashAttention (Module 8) addresses it identically for the multi-head case.

## Exercise

A model has $C = 512$. You want $n_h = 8$ heads. What is $d_h$? Now someone proposes $n_h = 7$ heads instead — what goes wrong?

*Optional hint:* $d_h = C / n_h$, and it must be a whole number.

<details><summary>Solution</summary>

With $n_h = 8$: $d_h = 512 / 8 = 64$. With $n_h = 7$: $512 / 7 = 73.14\ldots$, not an integer, so the channels cannot be split evenly into 7 heads — the `view(B, T, n_h, d_h)` would be impossible (there is no integer $d_h$ with $7 \cdot d_h = 512$). The `assert cfg.n_embd % cfg.n_head == 0` in `CausalSelfAttention.__init__` catches exactly this. `n_head` must divide `n_embd`.

</details>

## Common mistakes

- **Forgetting `.contiguous()` before the merge `view`.** After `transpose(1, 2)` the memory is not contiguous; `view` errors. Use `.contiguous().view(...)` (or `.reshape(...)`, which copies when needed).
- **Splitting on the wrong axis.** The channel axis (dim 2 of $(B, T, C)$) is the one that splits into heads, not the time axis. Splitting time would scramble positions across heads.
- **Reusing one head's mask shape wrongly.** The mask is $(1, 1, T, T)$ so it broadcasts over $(B, n_h)$. Building it as $(T, T)$ still broadcasts, but building it as $(B, T, T)$ (missing the head axis) will misalign against the 4-D scores.

## Check yourself

<details><summary>With $C = 768$ and $n_h = 12$, what is $d_h$, and what is the shape of the per-head query tensor for a batch $(B, T) = (4, 256)$?</summary>

$d_h = 768 / 12 = 64$. After the split, $q$ has shape $(B, n_h, T, d_h) = (4, 12, 256, 64)$.

</details>

<details><summary>Why do we need the output projection $W_O$ at all?</summary>

Because after the merge the $n_h$ head outputs occupy disjoint channel slices — nothing has combined them. $W_O$ (shape $(C, C)$) mixes across the full width so information found by one head can influence the whole output vector. Without it the heads would remain siloed in their own channels.

</details>

<details><summary>Why is the head axis placed at position 1 (right after batch), giving $(B, n_h, T, d_h)$, rather than left as $(B, T, n_h, d_h)$?</summary>

So that the last two axes are $(T, d_h)$ — the matrices attention multiplies — and the leading $(B, n_h)$ axes are treated as batch axes by `@`. This runs all $B \cdot n_h$ head-attentions as one batched matmul with no Python loop. In the $(B, T, n_h, d_h)$ layout the matmul would combine the wrong axes.

</details>

<details><summary>Does using more heads increase the total Q/K/V compute compared to a single head of width $C$?</summary>

No. The heads partition the same $C$ channels ($n_h \cdot d_h = C$), and $Q, K, V$ are still one $(C \to C)$ projection each. Splitting into heads only re-slices those numbers onto a new axis; the projection and the total attention FLOPs are the same as a single width-$C$ attention. What multiple heads buy is representational: independent relationship subspaces.

</details>

## Next

You can now split attention into heads, run them in parallel, and merge them — the complete `CausalSelfAttention` module. But attention is only half of a transformer block. The next lesson adds the position-wise MLP, the residual connections that make deep stacks trainable, and LayerNorm, then assembles all of it into the pre-norm `Block` that GPT stacks $n_{\text{layer}}$ times.

Continue to [05.4 · MLP, residual, LayerNorm, the block](lesson-04.md).
