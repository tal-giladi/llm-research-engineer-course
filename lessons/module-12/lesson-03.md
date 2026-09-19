# 12.3 · MQA / GQA

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="../module-05/lesson-03.md">05.3 · Multi-head attention</a> (the $n_h$ query/key/value heads, the $(B, n_h, T, d_h)$ layout, the output projection); scaled dot-product attention from <a href="../module-05/lesson-02.md">05.2</a>; the KV-cache idea from generation in <a href="../module-06/lesson-03.md">06.3 · Generation</a>.</p>
<p><strong>You will learn:</strong> why standard multi-head attention makes the inference KV cache large; how <strong>MQA</strong> (one shared key/value head) and <strong>GQA</strong> ($g$ key/value heads shared by groups of query heads) shrink it; the KV-cache memory arithmetic that motivates the change; and how to implement GQA so that it reduces exactly to MHA when $g = n_h$ and to MQA when $g = 1$.</p>
<p><strong>Why this matters for ML:</strong> GQA is the attention variant in Llama&nbsp;2-70B, Llama&nbsp;3, Mistral, and most current models. It is a pure inference-efficiency win — smaller KV cache means longer contexts and larger batches on the same GPU — at almost no quality cost. Knowing the KV-cache math is essential for serving.</p>
</div>

## 1. Intuition: the KV cache is the bottleneck at inference

During generation a decoder produces one token at a time. To generate token $t+1$, attention needs the keys and values of **every previous token** $1\dots t$. Recomputing them each step would be quadratic and wasteful, so we compute each token's K and V once and **cache** them — the **KV cache**. It is what makes autoregressive generation affordable (Module 6.3).

The problem: that cache grows with the sequence, and in standard multi-head attention it stores keys and values for **all $n_h$ heads**. For a long context and a big model it becomes the dominant consumer of GPU memory at inference — often larger than the model weights themselves for long sequences. It also has to be *read* from memory every single decode step, so its size directly caps both how long a context you can serve and how many sequences you can batch together.

Crucially, the number of **query** heads is what gives attention its expressive power (each head asks a different question), but the number of **key/value** heads is what sets the cache size. MQA and GQA exploit this asymmetry: keep many query heads, use *fewer* key/value heads.

<div class="callout key"><p>Query heads cost compute; key/value heads cost <em>cache memory</em>. Standard MHA ties them ($n_h$ of each). MQA and GQA break the tie — keep $n_h$ query heads but only $g < n_h$ KV heads — shrinking the KV cache by a factor of $n_h/g$ with little quality loss.</p></div>

## 2. The three variants

Let $n_h$ be the number of query heads and $g$ the number of key/value heads.

- **MHA** (standard): $g = n_h$. Every query head has its own K and V head. Maximum quality, maximum cache.
- **MQA** (Multi-Query Attention, Shazeer 2019): $g = 1$. *All* query heads share a single K/V head. Cache shrinks $n_h\times$; a small quality drop and some training instability were reported.
- **GQA** (Grouped-Query Attention, Ainslie et al. 2023): $1 < g < n_h$, with $g$ dividing $n_h$. The query heads are split into $g$ groups of $n_h/g$; each group shares one K/V head. It interpolates between MHA ($g = n_h$) and MQA ($g = 1$), recovering almost all of MHA's quality at a fraction of the cache. Llama&nbsp;2-70B and Llama&nbsp;3 use $g = 8$.

The mechanism is simple: compute $g$ key/value heads, then **repeat** each one across the $n_h/g$ query heads in its group so the per-head attention math is unchanged.

## 3. Mathematics: the KV-cache size

The KV cache stores, per layer, the keys and values (that is the factor $2$) for every cached position, across all KV heads:

$$
\text{KV bytes} = 2 \cdot n_{\text{layer}} \cdot g \cdot d_h \cdot T \cdot b \cdot B,
$$

where $g$ is the number of KV heads, $d_h$ the per-head dim, $T$ the context length, $b$ the bytes per number (2 for fp16/bf16), and $B$ the batch size. Only $g$ appears — not $n_h$ — so cutting KV heads cuts the cache linearly. The ratio between MHA and a $g$-head variant is exactly

$$
\frac{\text{MHA cache}}{\text{$g$-head cache}} = \frac{n_h}{g}.
$$

## 4. Numerical example: Llama-2-70B-shaped model

Use the Llama-2-70B geometry: $n_{\text{layer}} = 80$, $n_h = 64$ query heads, $d_h = 128$, context $T = 4096$, fp16 ($b = 2$), batch $B = 1$. The KV cache for the three variants (verified in §7):

| Variant | KV heads $g$ | KV cache | vs MHA |
|---------|-------------|----------|--------|
| MHA | 64 | 10.737 GB | 1× |
| GQA | 8 | 1.342 GB | 8× smaller |
| MQA | 1 | 0.168 GB | 64× smaller |

MHA's 10.7 GB of cache is a large slice of an 80 GB GPU *before you have batched anything or counted the ~140 GB of fp16 weights* — this is why 70B models are served with GQA. Dropping to $g = 8$ frees ~9.4 GB, which is what lets you raise the batch size or the context length. MQA squeezes hardest (0.17 GB) but Llama uses $g = 8$ as the quality/memory sweet spot.

## 5. Tensor shapes: project small, repeat, attend

The only shape change from MHA is that K and V are projected to $g$ heads instead of $n_h$:

$$
\begin{aligned}
Q &: (B, T, C) \xrightarrow{W_Q} (B, T, n_h d_h) \to (B, n_h, T, d_h) \\
K, V &: (B, T, C) \xrightarrow{W_K, W_V} (B, T, g\,d_h) \to (B, g, T, d_h)
\end{aligned}
$$

Before the scores, each KV head is **repeated** $n_h/g$ times along the head axis so K and V become $(B, n_h, T, d_h)$ again, matching Q:

$$
(B, g, T, d_h) \;\xrightarrow{\text{repeat\_interleave}(n_h/g)}\; (B, n_h, T, d_h).
$$

`repeat_interleave` copies KV head $j$ to query heads $j\cdot\tfrac{n_h}{g}, \dots, j\cdot\tfrac{n_h}{g} + \tfrac{n_h}{g} - 1$ — the group that shares it. After that the attention is identical to MHA: scores $QK^\top$, causal mask, softmax, weighted $V$, merge heads, output projection. Note the KV *projection weights* $W_K, W_V$ are smaller ($C\times g\,d_h$ instead of $C\times n_h d_h$), so GQA also has slightly fewer parameters.

## 6. From-scratch PyTorch

From [`llmre.attention.gqa`](../../code/src/llmre/attention/gqa.py) (RoPE-optional; pass `freqs_cis` from lesson 12.1 to rotate Q and K):

```python
import torch
import torch.nn as nn
from llmre.attention.attention import scaled_dot_product_attention

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, n_kv_head=None):
        super().__init__()
        self.n_head    = cfg.n_head
        self.n_kv_head = cfg.n_head if n_kv_head is None else n_kv_head
        assert self.n_head % self.n_kv_head == 0
        self.n_rep = self.n_head // self.n_kv_head          # query heads per KV head
        self.d_h   = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, self.n_head    * self.d_h, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.d_h, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.d_h, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x, freqs_cis=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head,    self.d_h).transpose(1, 2)  # (B,nh,T,hd)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.d_h).transpose(1, 2)  # (B,g,T,hd)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.d_h).transpose(1, 2)  # (B,g,T,hd)
        if self.n_rep > 1:                                  # share each KV head across its group
            k = k.repeat_interleave(self.n_rep, dim=1)      # (B,g,T,hd) -> (B,nh,T,hd)
            v = v.repeat_interleave(self.n_rep, dim=1)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        y = scaled_dot_product_attention(q, k, v, mask=mask)          # (B,nh,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)             # merge heads
        return self.c_proj(y)
```

When `n_kv_head == n_head`, `n_rep == 1`, the repeat is skipped, and the module computes exactly the same function as `CausalSelfAttention` (the only difference is that Q/K/V come from three separate `Linear`s instead of one fused `c_attn`). When `n_kv_head == 1`, all query heads share the one KV head — that is MQA.

## 7. Under the hood: verify the cache numbers, and the MHA-equivalence

```python
# KV-cache sizes (Section 4)
L, nh, hd, T, b = 80, 64, 128, 4096, 2       # Llama-2-70B-ish, fp16
mha = 2 * L * nh * hd * T * b
for g in (64, 8, 1):
    c = 2 * L * g * hd * T * b
    print(g, f"{c/1e9:.3f} GB", f"{mha//c}x")
# 64 10.737 GB 1x
#  8 1.342 GB 8x
#  1 0.168 GB 64x
```

The equivalence to MHA when $g = n_h$ (with shared weights) is checked in the test suite — `test_arch.py::test_gqa_equals_mha_when_kv_heads_full` copies a `CausalSelfAttention`'s fused `c_attn` weights into GQA's separate `q/k/v` projections and asserts the outputs match to `1e-10` in float64.

<div class="callout pt"><p>At <em>training</em> time GQA barely helps memory or speed — the repeat materializes the full $n_h$ KV heads anyway, and you process all positions in parallel (no cache). The win is entirely at <em>inference</em>: the cache stores only $g$ heads, and the repeat happens on the fly. This is why GQA is described as an inference optimization, even though it is chosen at architecture-design time.</p></div>

<div class="callout warn"><p><strong>Converting an MHA checkpoint to GQA.</strong> You cannot just delete KV heads. The standard recipe (Ainslie et al.) is to <em>mean-pool</em> the KV projection weights within each group to form the $g$ shared heads, then <em>uptrain</em> (fine-tune) for a small fraction of the original compute to recover quality. Naively dropping heads throws away learned information.</p></div>

## 8. Cost summary

| | Query heads | KV heads | KV cache | Quality |
|---|---|---|---|---|
| MHA | $n_h$ | $n_h$ | largest | best |
| GQA | $n_h$ | $g$ | $n_h/g$ smaller | ≈ MHA |
| MQA | $n_h$ | $1$ | $n_h\times$ smaller | slightly below |

Compute for the scores and the softmax is essentially unchanged across the three (there are still $n_h$ query heads doing $n_h$ attentions). The difference is memory: KV-cache bytes, KV-projection parameters, and memory-bandwidth per decode step.

## Common mistakes

- **$g$ not dividing $n_h$.** The query heads must split into $g$ equal groups; enforce `n_head % n_kv_head == 0`.
- **Forgetting to repeat K and V.** Without the `repeat_interleave`, the shapes won't broadcast for $QK^\top$ (or worse, will broadcast wrongly). Each group's query heads must see their shared KV head.
- **Expecting a training speedup.** GQA's benefit is the inference KV cache; training throughput is roughly unchanged.
- **Deleting heads to convert a checkpoint.** Mean-pool within groups and uptrain instead.

## Exercise

A model has $n_h = 32$ query heads and uses GQA with $g = 4$. (a) How many query heads share each KV head? (b) By what factor is its KV cache smaller than the MHA version? (c) What would $g$ have to be to make it MQA?

<details><summary>Solution</summary>

(a) $n_h/g = 32/4 = 8$ query heads per KV head. (b) The cache scales with $g$, so it is $n_h/g = 32/4 = 8\times$ smaller than MHA's. (c) MQA is $g = 1$ (a single shared KV head).

</details>

## Debugging exercise

Someone implements GQA but repeats the KV heads with `k.repeat(1, self.n_rep, 1, 1)` (tile the whole head block) instead of `k.repeat_interleave(self.n_rep, dim=1)`. With $g = 2$, $n_{\text{rep}} = 2$, and KV heads labeled $[A, B]$, which query heads end up paired with which KV head under each call, and why does the buggy one break the grouping?

<details><summary>Answer</summary>

`repeat_interleave(2, dim=1)` on $[A, B]$ gives $[A, A, B, B]$ — query heads 0,1 share $A$; heads 2,3 share $B$. Correct contiguous groups. `repeat(1, 2, 1, 1)` *tiles* the block: $[A, B, A, B]$ — query heads 0,2 share $A$ and 1,3 share $B$. The groups are now interleaved rather than contiguous. It still runs and shapes match, but it silently pairs query heads with the wrong KV head relative to how the weights were trained/expected, scrambling the intended grouping. Use `repeat_interleave`.

</details>

## Research connection

<div class="callout paper"><p><strong>GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints</strong> (Ainslie et al. 2023) introduced GQA and the mean-pool-and-uptrain conversion; MQA is from Shazeer's "Fast Transformer Decoding" (2019). <strong>Llama&nbsp;3</strong> (<a href="../../papers/index.md">paper curriculum</a> #10) uses GQA with 8 KV heads. The next step in this line is DeepSeek-V3's Multi-head Latent Attention (MLA), which compresses the KV cache differently — see #14.</p></div>

## Check yourself

<details><summary>Why does the number of KV heads, not query heads, determine the KV-cache size?</summary>

The cache stores the keys and values of past tokens, and there is one K and one V vector per *KV* head per position. Query vectors are computed fresh for the current token each step and are not cached. So the cache size is proportional to the number of KV heads $g$, independent of $n_h$.

</details>

<details><summary>What values of $g$ make GQA reduce to MHA and to MQA?</summary>

$g = n_h$ gives MHA (every query head has its own KV head; $n_{\text{rep}} = 1$, no sharing). $g = 1$ gives MQA (all query heads share the single KV head).

</details>

<details><summary>A model with $n_h = 64$ switches from MHA to GQA with $g = 8$. Its KV cache was 10.7 GB. What is it now?</summary>

The cache scales linearly with $g$, so it shrinks by $64/8 = 8\times$: $10.7 / 8 \approx 1.34$ GB — matching the table in §4.

</details>

<details><summary>Why is GQA called an inference optimization if the architecture is fixed at training time?</summary>

The savings (smaller KV cache, less memory-bandwidth per decode step) only appear during autoregressive generation, where the cache exists and tokens are produced one at a time. During training there is no cache and all positions are processed in parallel, so the KV heads are materialized to full width and throughput is roughly unchanged. You must *choose* GQA when designing the model, but the payoff is at serving time.

</details>

## Next

The last modern change scales *parameters* without scaling per-token compute: replacing the dense FFN with a mixture of experts. Continue to [12.4 · Mixture of Experts (DeepSeek-style)](lesson-04.md).
