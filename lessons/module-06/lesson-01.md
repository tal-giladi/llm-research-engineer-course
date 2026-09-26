# 06.1 · Assembling GPT-2

<div class="prereq">
<p><strong>Prerequisites:</strong> the pre-norm <code>Block</code> from <a href="#/lessons/module-05/lesson-04">05.4 · MLP, residual, LayerNorm, the block</a>; token and positional embeddings from <a href="#/lessons/module-05/lesson-01">05.1 · Embeddings &amp; positional encoding</a>; the language-modeling cross-entropy objective from <a href="#/lessons/module-01/lesson-04">01.4 · The language-modeling objective &amp; perplexity</a>.</p>
<p><strong>You will learn:</strong> how the pieces you built in Module 5 snap together into the complete GPT-2 model — the two embedding tables (<code>wte</code>, <code>wpe</code>), the stack of <code>n_layer</code> blocks, the final <code>ln_f</code>, and the language-model head <code>lm_head</code>; the <strong>weight-tying</strong> trick that makes <code>wte</code> and <code>lm_head</code> the same tensor and why it makes sense; and the full <code>forward(idx, targets)</code> pass traced shape-by-shape from token ids <code>(B, T)</code> to logits <code>(B, T, V)</code> and a scalar loss.</p>
<p><strong>Why this matters for ML:</strong> this is the moment the abstract "transformer" becomes a concrete object you can instantiate, count the parameters of, and call. Every model you train, fine-tune, or serve for the rest of the course is this class or a small variation of it. There is no more machinery hiding — a GPT really is an embedding, a stack of identical blocks, a norm, and a linear head.</p>
</div>

## 1. Intuition: a GPT is four things in a row

Module 5 ended with one `Block`: normalize, attend, add back; normalize, MLP, add back. A GPT is that block stacked `n_layer` times, with a small amount of bookkeeping at the two ends. In full:

```text
token ids ──▶ embed each token          (wte)          ──┐
positions ──▶ embed each position        (wpe)          ──┴─▶ add ─▶ dropout
          ──▶ Block × n_layer  (the transformer stack)
          ──▶ final LayerNorm  (ln_f)
          ──▶ linear to vocabulary scores (lm_head)  ──▶ logits
```

The stack of blocks is where all the thinking happens. The two ends are pure translation: the **front end** turns integer token ids into vectors the blocks can work with, and the **back end** turns the blocks' output vectors back into a score for every possible next token. That is the entire architecture. This lesson builds the front end, the back end, and the `forward` that runs them; [06.2](lessons/module-06/lesson-02.md) covers initialization and the parameter count; [06.3](lessons/module-06/lesson-03.md) covers using a trained model to generate text.

## 2. The front end: two embedding tables

### Intuition and math

The model's input is a batch of token-id sequences: an integer tensor `idx` of shape $(B, T)$, where each entry is an index into the vocabulary (a number in $0 \dots V-1$). Integers carry no useful geometry — token 5 is not "half of" token 10 — so the first thing we do is look up a learned vector for each one.

Two look-ups happen, and they are **added**:

$$
x_{b,t} = \underbrace{E^{\text{tok}}[\,\text{idx}_{b,t}\,]}_{\text{what token}} \;+\; \underbrace{E^{\text{pos}}[\,t\,]}_{\text{which position}}.
$$

- $E^{\text{tok}} = $ `wte` (word-token embedding) is a table of shape $(V, C)$: row $i$ is the $C$-dimensional vector for token id $i$.
- $E^{\text{pos}} = $ `wpe` (word-position embedding) is a table of shape $(\text{block\_size}, C)$: row $t$ is the vector for *position* $t$.

Why add a position vector at all? Attention (Module 5) is **permutation-invariant** — it treats its inputs as an unordered set, so on its own it cannot tell "the dog bit the man" from "the man bit the dog". The positional embedding stamps each token with *where* it sits, breaking that symmetry. GPT-2 uses *learned* absolute position embeddings: `wpe` is just another table, trained like any weight. (Module 12 replaces this with rotary embeddings; here we build exactly what GPT-2 did.)

<div class="callout key"><p>The input to the transformer stack is <code>wte[idx] + wpe[positions]</code>: a per-position sum of "what token is here" and "where here is". Both tables are learned. Everything downstream sees only these vectors, never the integer ids.</p></div>

### Tensor shapes

```text
idx           (B, T)            int64      token ids, 0 .. V-1
wte(idx)      (B, T, C)         float      one C-vector per token
positions     (T,)             int64      arange(T): 0, 1, ..., T-1
wpe(positions)(T, C)            float      one C-vector per position
tok + pos     (B, T, C)         float      pos_emb broadcasts over the batch axis
```

The addition relies on broadcasting: `pos_emb` is $(T, C)$ and `tok_emb` is $(B, T, C)$, so PyTorch stretches `pos_emb` across the $B$ axis — every sequence in the batch gets the *same* position vectors, which is exactly right.

## 3. The back end: final norm and the LM head

After the last block, `x` is still $(B, T, C)$: one refined vector per position. To predict the next token we need, at each position, a score for every vocabulary token — a **logit vector** of length $V$. Two steps:

1. **`ln_f`** — a final LayerNorm over the $C$ channels. Because the blocks are pre-norm (Module 5.4), nothing normalizes the residual stream on its way out; `ln_f` gives the head a clean, unit-scale input.
2. **`lm_head`** — a linear map $C \to V$ with **no bias**: `logits = x @ W_lm^T`. Row $i$ of $W_{\text{lm}}$ (shape $(V, C)$) is the "detector" for token $i$; the dot product of a position's vector with that row is how strongly the model thinks token $i$ comes next.

The result is `logits` of shape $(B, T, V)$: for every position in every sequence, a raw score per vocabulary token. These are exactly the logits that [01.4](lessons/module-01/lesson-04.md) fed into softmax and cross-entropy.

## 4. Weight tying: `wte` and `lm_head` are the same tensor

Here is the one genuinely surprising line in the whole model:

```python
self.transformer.wte.weight = self.lm_head.weight
```

`wte.weight` has shape $(V, C)$ and `lm_head.weight` has shape $(V, C)$ — the same shape. **Weight tying** points them at the *same* Parameter object, so they share one block of memory and one gradient.

### Why this is natural

Think about what each does. The embedding takes a token id and produces its $C$-vector: row $i$ of the table *is* the code for token $i$. The head takes a $C$-vector and scores token $i$ by dotting the vector with... row $i$ of *its* table. Both operations are organized around "the vector that represents token $i$". Tying says: use the *same* vector for both directions. The map that reads a token in and the map that writes a token out are transposes of one learned dictionary.

### What it buys

- **Parameters.** In GPT-2 small the table is $50257 \times 768 \approx 38.6\text{M}$ parameters. Untied, we would pay for it twice; tied, we pay once — saving ~40M parameters (roughly a *third* of the model). We verify the full count in [06.2](lessons/module-06/lesson-02.md).
- **Quality.** Tying was shown to improve language-model perplexity (Press & Wolf 2016; used in GPT-2). Intuitively, the input and output representations of a token stay consistent, and the rarer tokens' output vectors get gradient signal from their (more frequent) use as inputs too.

<div class="callout warn"><p>Tying is an <em>aliasing</em> of two attributes to one tensor, not a copy. If you later reload weights or reset one of them, do the tie <em>again</em> afterward — assigning a fresh tensor to <code>lm_head.weight</code> silently breaks the link, and you will train two separate $38.6$M tables without noticing until your parameter count jumps.</p></div>

## 5. The `forward` pass, end to end

Here is the full model. It lives in `code/src/llmre/model/gpt.py` and imports the `Block` and `LayerNorm` you built in Module 5 — we do not re-implement them.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from llmre.model.block import Block, LayerNorm
from llmre.model.config import GPTConfig

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),   # (V, C)
            wpe  = nn.Embedding(cfg.block_size, cfg.n_embd),   # (block_size, C)
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # (C -> V)
        self.transformer.wte.weight = self.lm_head.weight                 # weight tying
        # ... initialization: see lesson 06.2 ...

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, dtype=torch.long, device=idx.device)  # (T,)

        tok_emb = self.transformer.wte(idx)            # (B, T, C)
        pos_emb = self.transformer.wpe(pos)            # (T, C)
        x = self.transformer.drop(tok_emb + pos_emb)   # (B, T, C)
        for block in self.transformer.h:
            x = block(x)                               # (B, T, C)
        x = self.transformer.ln_f(x)                   # (B, T, C)
        logits = self.lm_head(x)                       # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),      # (B*T, V)
                targets.view(-1),                      # (B*T,)
                ignore_index=-1,
            )
        return logits, loss
```

### Why `nn.ModuleDict`

Grouping the sub-modules under `self.transformer` (a `ModuleDict`) is not cosmetic: it gives every weight a *name* like `transformer.h.3.attn.c_proj.weight`. Those names are the keys in the saved `state_dict`, and they match the names in OpenAI's released GPT-2 checkpoints — so with this exact layout you can load the real pretrained weights later. `nn.ModuleList` for `h` does the same for the numbered blocks.

### The loss: flatten, then cross-entropy

`targets` is $(B, T)$: `targets[b, t]` is the token that should follow position $t$ in sequence $b$ (Module 4's data loader builds these by shifting the input by one). Cross-entropy wants a 2-D `(N, V)` logits tensor and a 1-D `(N,)` target tensor, so we **flatten** batch and time into one axis of $B \cdot T$ independent classification problems:

$$
\mathcal{L} = \frac{1}{BT}\sum_{b,t} -\log \operatorname{softmax}(\text{logits}_{b,t})[\,\text{targets}_{b,t}\,].
$$

This is precisely the mean next-token cross-entropy of [01.4](lessons/module-01/lesson-04.md), now computed for every position of every sequence at once. `ignore_index=-1` lets us mark padding positions with a target of $-1$ so they contribute nothing to the loss.

<div class="callout pt"><p><code>logits.view(-1, logits.size(-1))</code> collapses <code>(B, T, V)</code> to <code>(B*T, V)</code> without copying data — <code>view</code> just reinterprets the strides. <code>F.cross_entropy</code> then fuses <code>log_softmax</code> and the negative-log-likelihood gather into one numerically stable kernel; it never materializes the softmax probabilities as a separate tensor.</p></div>

### Full shape trace

For a batch $B = 2$, sequence length $T = 8$, GPT-2-small dims $C = 768$, $V = 50257$:

```text
idx            (2, 8)          int64
wte(idx)       (2, 8, 768)     float32
wpe(0..7)      (8, 768)        float32
x = sum        (2, 8, 768)     float32     (broadcast over B)
after 12 blocks(2, 8, 768)     float32     (shape never changes in the stack)
ln_f(x)        (2, 8, 768)     float32
logits         (2, 8, 50257)   float32
```

The stack preserves shape at every step — that invariance is exactly why blocks are stackable. Only the two ends change the last axis: `wte` maps $\text{id} \to C$, `lm_head` maps $C \to V$.

## 6. Under the hood: where the compute and memory go

Two observations you will lean on when you train this in Module 7.

- **The logits tensor is huge.** $(B, T, V)$ with $V = 50257$ dwarfs the $(B, T, C)$ activations inside the stack — for $B{=}2, T{=}1024$ that is $2 \cdot 1024 \cdot 50257 \approx 10^8$ floats, ~400 MB in fp32, just for one layer's output. This is why the final projection and the cross-entropy over it are a real slice of training memory, and why long-context training fuses them carefully.
- **`arange` is on-device.** `torch.arange(T, device=idx.device)` builds the positions where the model already lives (CPU or GPU); forgetting the `device=` is a classic bug that throws a "tensors on different devices" error the first time you move the model to a GPU.

## Exercise

You instantiate the model, then load a checkpoint with
`model.load_state_dict(sd)`, and *afterward* re-create the head with
`model.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)`. Training now
uses ~40M more parameters than you expected and the loss behaves oddly. What went
wrong?

<details><summary>Hint</summary>
What is the relationship between <code>lm_head.weight</code> and <code>wte.weight</code> supposed to be, and does assigning a brand-new <code>nn.Linear</code> preserve it?
</details>

<details><summary>Stronger hint</summary>
The tie is a Python assignment aliasing two attributes to one tensor. Replacing <code>lm_head</code> with a fresh module gives it its <em>own</em> freshly-allocated weight.
</details>

<details><summary>Solution</summary>

Weight tying is the single line `self.transformer.wte.weight = self.lm_head.weight`, which makes both attributes point at one Parameter. Creating a new `nn.Linear` allocates a *separate* $(V, C)$ weight and rebinds `lm_head.weight` to it, so `wte` and `lm_head` are no longer the same tensor. You now carry two independent $50257 \times 768 \approx 38.6$M tables (hence the ~40M jump), the head starts from random init instead of the loaded/embedding weights, and the two representations drift apart during training. Fix: after any operation that reassigns either weight, redo the tie — `self.transformer.wte.weight = self.lm_head.weight`.

</details>

## Check yourself

<details><summary>Why must we add positional embeddings — what property of attention makes them necessary?</summary>

Self-attention is permutation-invariant: it treats the positions as an unordered set and would give the same output for any reordering of the tokens. Adding a position-dependent vector to each token breaks that symmetry so the model can use word order.

</details>

<details><summary>The token-embedding table is <code>(V, C)</code> and the LM-head weight is <code>(V, C)</code>. Given weight tying, how many parameters do the two together contribute?</summary>

$V \cdot C$ — counted once, because they are the *same* tensor. For GPT-2 small that is $50257 \times 768 \approx 38.6$M, not $2 \times 38.6$M.

</details>

<details><summary>For <code>idx</code> of shape <code>(4, 128)</code> with <code>V = 50257</code>, what shape is <code>logits</code>, and what shape does it become just before <code>F.cross_entropy</code>?</summary>

`logits` is $(4, 128, 50257)$. Before cross-entropy it is flattened to $(4 \cdot 128, 50257) = (512, 50257)$, with targets flattened to $(512,)$.

</details>

<details><summary>Why is <code>ln_f</code> needed at all, given every block already contains two LayerNorms?</summary>

The blocks are *pre-norm*: each LayerNorm sits on the branch feeding a sub-layer, never on the residual stream itself. So the stream that exits the last block has never been normalized. `ln_f` normalizes it once before the head, giving the linear projection a stable, unit-scale input.

</details>

## Next

You can now assemble and run a GPT: embed, stack, norm, project, and (with targets) get a loss. What we skipped is *how the weights start*: GPT-2 uses a specific initialization — including a subtle per-layer scaling of the residual projections — and it is worth being able to derive the parameter count and predict the loss of an untrained model. That is the next lesson.

Continue to [06.2 · Init, forward pass, parameter count](lessons/module-06/lesson-02.md).
