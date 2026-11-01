# 08.6 · Inference serving: KV cache, prefill vs decode, batching, speculative decoding

<div class="callout key"><p><strong>Added 2026-11-01 by the weekly curriculum review</strong> (<a href="#/research/CHANGELOG">changelog</a>). This extends Module 8; nothing earlier is replaced. The generation loop you wrote in <a href="#/lessons/module-06/lesson-03">06.3</a> is still the definition of autoregressive sampling, and it stays the reference every faster path is tested against. 06.3 promised that the KV cache would be built "in a later module". This is that lesson. It also covers what a serving engine does with the cache once it exists.</p></div>

<div class="prereq">
<p><strong>Prerequisites:</strong> the autoregressive loop, temperature and top-$k$ from <a href="#/lessons/module-06/lesson-03">06.3 · Generation</a>; queries, keys, values and the causal mask from <a href="#/lessons/module-05/lesson-02">05.2</a> and <a href="#/lessons/module-05/lesson-03">05.3</a>; parameter and FLOP counting from <a href="#/lessons/module-07/lesson-04">07.4</a>; arithmetic intensity, the roofline and the A100 ridge point (≈153 FLOPs/byte) from <a href="#/lessons/module-08/lesson-02">08.2</a>. FlashAttention (<a href="#/lessons/module-08/lesson-04">08.4</a>) helps but is not required. You do not need GQA yet: <a href="#/lessons/module-12/lesson-03">12.3</a> comes later and shrinks the cache this lesson builds.</p>
<p><strong>You will learn:</strong> the <strong>KV cache</strong>, built from scratch for the course's own GPT and checked against the plain loop; the two phases of inference, <strong>prefill</strong> (compute-bound) and <strong>decode</strong> (memory-bound), and why that follows from 08.2's roofline; why batching amortizes the weights but not the cache; <strong>continuous (iteration-level) batching</strong> versus static batching; <strong>paged KV memory</strong> with block tables (vLLM's PagedAttention); <strong>lossless speculative decoding</strong> (draft, then verify with a rejection rule that leaves the target distribution exactly unchanged), with the proof worked by hand; the serving metrics (TTFT, TPOT, throughput, goodput); and why rollout generation is the largest cost in RL post-training (Module 16).</p>
<p><strong>Why this matters for ML:</strong> a research engineer runs inference constantly: evaluation (Module 13), sampling RL rollouts (16.2, 16.4), distillation teachers (17.4), agents (Module 19). The serving engine (vLLM, SGLang, TensorRT-LLM) is part of your training system. If you cannot predict its cost from first principles, you cannot budget an RL run, explain why a rollout batch is slow, or tell whether a "faster decoding" paper changed the outputs.</p>
</div>

<div class="hw">
<p><strong>Hardware track:</strong> CPU-only is enough. <code>tests/test_serving.py</code> (16 tests) runs in about 3 seconds. <code>experiments/serving_costs.py</code> runs in about 30 seconds on a laptop CPU (it trains two tiny models for part C), under 1 GB RAM, 0 GPU-hours. The cost numbers for 7B models in sections 3 and 6 are worked from the roofline, not measured: reproducing them needs one 80 GB GPU and a serving engine, which this lesson does not require.</p>
</div>

## 1. Intuition: the loop of 06.3 does almost all of its work twice

Look at what the 06.3 loop does at step $t$: it runs the whole model over all $t$ tokens, gets $t$ rows of logits, and throws away all but the last. Then at step $t+1$ it does it again over $t+1$ tokens.

Most of that work repeats. In a causal model, the keys and values that position $j$ produces depend only on tokens $1..j$. Adding token $t+1$ cannot change them. So we compute each position's $K$ and $V$ **once**, keep them in a **KV cache**, and at each new step run the model on the **one new token** only. Its query attends to the cached keys and values plus its own.

That splits inference into two phases with very different hardware behaviour:

- **Prefill**: process the whole prompt at once (many tokens in parallel), fill the cache, and produce the first output token. Big matmuls, so it is **compute-bound**, like training.
- **Decode**: one new token per sequence per step. Every step must stream **all the weights and the whole cache** from HBM to do a small amount of arithmetic, so it is **memory-bound**.

Everything else in this lesson follows from that asymmetry. Batching many sequences raises decode's arithmetic intensity. Continuous batching keeps the batch full. Paging keeps the cache from wasting memory, so the batch can be bigger. Speculative decoding spends spare compute to verify several tokens per weight read.

<div class="callout key"><p>Prefill is a training-like forward pass over the prompt. Decode reads every weight and every cached key/value to produce <em>one</em> token per sequence. On a modern GPU the first is limited by FLOPs and the second by bytes. Most serving techniques are ways to do more useful work per byte read during decode.</p></div>

## 2. Mathematics

### 2.1 The cached forward pass

Write the attention of one head at layer $\ell$ as in 05.2. For the token at position $t$ (1-indexed) with layer input $x_t \in \mathbb{R}^{C}$:

$$
q_t = x_t W_Q,\quad k_t = x_t W_K,\quad v_t = x_t W_V,\qquad
o_t = \sum_{j=1}^{t} \operatorname{softmax}_j\!\left(\frac{q_t \cdot k_j}{\sqrt{d_h}}\right) v_j .
$$

Here $W_Q, W_K, W_V \in \mathbb{R}^{C \times d_h}$ are the head's projections, $d_h$ is the head dimension, and the softmax runs over $j = 1..t$ (the causal mask). The output $o_t$ needs $q_t$ (from the new token) and $k_j, v_j$ for all $j \le t$. The $k_j, v_j$ for $j < t$ were computed at earlier steps and do not change, because $x_j$ at every layer depends only on tokens $1..j$. The cache stores them:

$$
\mathcal{K}^{(\ell)}_{t} = [k_1; \dots; k_t],\qquad \mathcal{V}^{(\ell)}_{t} = [v_1; \dots; v_t],\qquad
\mathcal{K}^{(\ell)}_{t} = [\mathcal{K}^{(\ell)}_{t-1};\, k_t].
$$

A **chunk** of $T_{\text{new}}$ new tokens at positions $s+1..s+T_{\text{new}}$ (where $s$ is the cache length) is the general case: prefill is $s = 0$, decode is $T_{\text{new}} = 1$, and speculative verification (section 2.5) is $T_{\text{new}} = k+1$ with $s > 0$. Query $i$ of the chunk sits at absolute position $s + i$ and may attend to key $j$ only when $j \le s + i$. The mask is the causal mask shifted right by $s$: an additive matrix $M \in \mathbb{R}^{T_{\text{new}} \times (s + T_{\text{new}})}$ with $M_{ij} = 0$ if $j \le s+i$ and $-\infty$ otherwise.

**Cost of the work.** Generating $N$ tokens from a prompt of $P$ tokens with the 06.3 loop runs the model over $\sum_{t=P}^{P+N-1} t$ positions. With the cache it runs over $P$ (prefill) plus $N - 1$ (decode) positions. The last sampled token is never fed back. Attention still reads all $t$ cached keys per step, so that part stays $O(t)$ per step. The MLP and projections, which dominate for moderate contexts, become $O(1)$ per step instead of $O(t)$.

### 2.2 Cache size

Per layer, per token, the cache holds one key and one value vector per KV head. With $n_{\text{layer}}$ layers, $g$ KV heads (here $g = n_h$; 12.3 makes $g$ smaller), head dimension $d_h$, $b$ bytes per number, sequence length $S$ and batch $B$:

$$
\text{KV bytes} = 2 \cdot n_{\text{layer}} \cdot g \cdot d_h \cdot S \cdot B \cdot b .
$$

This is the same formula as 12.3 section 3. `kv_cache_bytes` in the code reproduces 12.3's Llama-2-70B numbers in a test.

### 2.3 Arithmetic intensity of a linear layer: prefill vs decode

Every projection and MLP layer is a matmul of $t$ token rows, $(t, d_{\text{in}}) \times (d_{\text{in}}, d_{\text{out}})$. From 08.2, with $b$ bytes per element:

$$
\text{FLOPs} = 2\,t\,d_{\text{in}}\,d_{\text{out}},\qquad
\text{bytes} = b\,(d_{\text{in}} d_{\text{out}} + t\,d_{\text{in}} + t\,d_{\text{out}}),\qquad
\text{AI} = \frac{2\,t\,d_{\text{in}} d_{\text{out}}}{b\,(d_{\text{in}} d_{\text{out}} + t\,d_{\text{in}} + t\,d_{\text{out}})} .
$$

For a square layer ($d_{\text{in}} = d_{\text{out}} = d$) in bf16 ($b = 2$) this is $\text{AI} = t\,d / (d + 2t)$. When $t \ll d$ the weight read dominates the bytes and $\text{AI} \approx t$. That gives the key fact:

- **Decode**: $t = B$ (one token per sequence). AI $\approx B$ FLOPs/byte.
- **Prefill**: $t = B \cdot P$. AI is in the hundreds or thousands.

Compare with the A100 ridge $\text{AI}_{\text{ridge}} \approx 153$ from 08.2. Decode at batch 1 has AI ≈ 1, so it is about 150× below the ridge.

**The cache is not amortized.** Attention during decode reads sequence $b$'s own cache, $2\,n_{\text{layer}}\,g\,d_h\,S$ numbers, and does about 2 FLOPs per number read ($q \cdot k$, then weight times $v$). Its intensity is about 1 FLOP per element, whatever the batch size, because no two sequences share a cache. Batching amortizes the weight read across sequences. It does nothing for the cache read.

### 2.4 A roofline model of a decode step

With $N_p$ parameters, peak compute $P_{\max}$ and bandwidth $\beta$, one decode step over a batch of $B$ sequences at context $S$ takes at least

$$
t_{\text{step}} \;\ge\; \max\!\left(\frac{2 N_p B}{P_{\max}},\ \frac{b\,N_p + \text{KV bytes}(S, B)}{\beta}\right),
$$

and produces $B$ tokens. (`roofline_time` computes this. Attention FLOPs are $\approx 4\,n_{\text{layer}}\,g\,d_h\,S$ per token, small for moderate $S$, and left out here.)

### 2.5 Speculative decoding: draft, verify, and why it is lossless

Decode is memory-bound, so a forward over $k+1$ new tokens costs about the same as a forward over 1 token: the weights are read once either way. Speculative decoding (Leviathan et al. 2023; Chen et al. 2023) uses this.

1. A cheap **draft** model $q$ proposes $k$ tokens $x_1, \dots, x_k$ autoregressively, sampling $x_i \sim q_i(\cdot)$.
2. The **target** model $p$ scores all of them in **one** forward over $k+1$ positions. That gives $p_1(\cdot), \dots, p_{k+1}(\cdot)$, where $p_i$ is the target's distribution for the position of $x_i$, and $p_{k+1}$ is the position after all drafts.
3. For $i = 1..k$: draw $r \sim \mathcal{U}(0,1)$ and **accept** $x_i$ if $r < \min\!\big(1, p_i(x_i)/q_i(x_i)\big)$. At the first rejection, sample a replacement from the **residual distribution**
$$
\operatorname{res}_i(x) = \frac{\max\big(0,\ p_i(x) - q_i(x)\big)}{\sum_y \max\big(0,\ p_i(y) - q_i(y)\big)}
$$
and stop. If all $k$ are accepted, sample a **bonus** token from $p_{k+1}$.

Each round emits between 1 and $k+1$ tokens for one target forward.

**Why the output distribution is exactly $p$.** Take one position and write $p = p_i$, $q = q_i$. The emitted token is $x$ if either the draft proposed $x$ and it was accepted, or something was rejected and the residual produced $x$:

$$
\Pr[\text{emit } x] = q(x)\,\min\!\Big(1, \frac{p(x)}{q(x)}\Big) + \Big(1 - \sum_y \min\big(p(y), q(y)\big)\Big)\operatorname{res}(x).
$$

The first term is $\min(p(x), q(x))$. For the second, note $1 - \sum_y \min(p,q) = \sum_y \big(p(y) - \min(p(y),q(y))\big) = \sum_y \max(0, p(y) - q(y))$, which is exactly the residual's normalizer. So the second term is $\max(0, p(x) - q(x))$, and

$$
\Pr[\text{emit } x] = \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x).
$$

The procedure is **lossless**: whatever the draft, the tokens come from the target's distribution. A bad draft only makes it slower. The same argument applies position by position, because after an acceptance the next position is conditioned on a token that was itself distributed as $p$.

**Speed.** The **acceptance rate** at a position is $\alpha = \sum_x \min(p(x), q(x)) = 1 - \operatorname{TV}(p, q)$, where TV is total-variation distance. If each draft is accepted independently with probability $\alpha$, the expected number of tokens per round is

$$
\mathbb{E}[\text{tokens per round}] = \sum_{i=0}^{k} \alpha^i = \frac{1 - \alpha^{k+1}}{1 - \alpha}.
$$

If one draft forward costs a fraction $c$ of a target forward, a round costs $kc + 1$ target-forward units, and the expected speedup over plain decoding is $\dfrac{1 - \alpha^{k+1}}{(1-\alpha)(kc + 1)}$ (the analysis in Leviathan et al., section 3). That assumes the $k+1$-token verify costs the same as a 1-token decode step, which holds only while decode is memory-bound (section 6 returns to this).

**Greedy decoding** is the special case where $p$ and $q$ are one-hot at their argmax. Accept $x_i$ if it equals the target's argmax. Otherwise emit the target's argmax. The output is token-for-token the target's greedy output, and `test_speculative_greedy_is_lossless_vs_target_greedy` checks it.

## 3. Worked examples with tiny numbers

**Cache saves positions.** Prompt $P = 4$, generate $N = 4$ tokens. The 06.3 loop runs the model over $4 + 5 + 6 + 7 = 22$ positions. With the cache it is a 4-position prefill plus 3 one-position decode steps: $7$ positions. For $P = 1000$, $N = 1000$ the counts are $\sum_{t=1000}^{1999} t = 1{,}499{,}500$ versus $1{,}999$, about 750× fewer.

**Prefill vs decode intensity.** A square bf16 layer with $d = 4096$, $\text{AI} = t\,d/(d + 2t)$:

| tokens $t$ | case | AI (FLOPs/byte) | vs A100 ridge 153 |
|---|---|---|---|
| 1 | decode, batch 1 | $4096 / 4098 \approx 1.0$ | memory-bound, ~150× below |
| 64 | decode, batch 64 | $64 \cdot 4096 / 4224 \approx 62.1$ | still memory-bound |
| 4096 | prefill, one 4k prompt | $4096 \cdot 4096 / 12288 \approx 1365$ | compute-bound |

**A 7B model decoding on one A100.** Use a Llama-2-7B shape: $N_p \approx 7 \times 10^9$, $n_{\text{layer}} = 32$, $g = n_h = 32$ (no GQA), $d_h = 128$, bf16. A100: $P_{\max} = 312$ TFLOP/s, $\beta = 2.039$ TB/s.

- Weights: $2 \times 7 \times 10^9 = 14$ GB. KV per token: $2 \cdot 32 \cdot 32 \cdot 128 \cdot 2 = 524{,}288$ bytes = 0.5 MiB.
- **Batch 1, context 1024.** KV $= 0.54$ GB. Bytes $= 14.54$ GB, so $14.54 / 2.039 \approx 7.1$ ms. FLOPs $= 1.4 \times 10^{10}$, so $0.045$ ms. Memory-bound by 160×. That is about **140 tokens/s**, and the arithmetic units are idle more than 99% of the time.
- **Batch 64, context 1024.** KV $= 64 \cdot 1024 \cdot 0.5$ MiB $= 34.4$ GB. Bytes $= 48.4$ GB, so 23.7 ms. FLOPs $= 9 \times 10^{11}$, so 2.9 ms. Still memory-bound. The step produces 64 tokens: about **2,700 tokens/s**, 19× the batch-1 throughput. But the KV cache is now 71% of the bytes read, and $14 + 34.4 = 48.4$ GB of the 80 GB is taken. Batch 128 would need 83 GB and does not fit. **Cache memory caps the batch size.** This is the motivation for GQA (12.3) and for paging (section 6).

**Continuous batching.** Eight requests needing $[10, 1, 1, 1, 10, 1, 1, 1]$ output tokens, batch size 4.

- *Static*: requests 1–4 run together until the longest finishes (10 steps), then requests 5–8 (10 steps). That is **20 steps**. Useful token slots are $26$ out of $20 \times 4 = 80$, so the batch is 32.5% busy.
- *Continuous*: after step 1, three requests are done, and requests 5, 6, 7 take their slots at step 2. After step 2, request 8 joins. Only the two 10-token requests remain after step 3, and they finish at step 11. That is **11 steps**, 59% busy.

**Paging.** Block size 16 tokens. A sequence of 37 tokens holds $\lceil 37/16 \rceil = 3$ blocks, 48 slots, so 11 slots are wasted. A contiguous allocator that reserves the maximum length (say 2048) for every request wastes 2011 slots on the same sequence.

**Speculative acceptance, by hand.** Target $p = [0.5, 0.3, 0.2]$, draft $q = [0.2, 0.6, 0.2]$ over a 3-token vocabulary.

- Acceptance probabilities $\min(1, p/q)$: token 0: $\min(1, 2.5) = 1$. Token 1: $0.3/0.6 = 0.5$. Token 2: $1$.
- $\alpha = \sum \min(p, q) = 0.2 + 0.3 + 0.2 = 0.7$. Reject mass $= 0.3$.
- Residual: $\max(p - q, 0) = [0.3, 0, 0]$, normalized $[1, 0, 0]$.
- Emitted: $\Pr[0] = 0.2 \cdot 1 + 0.3 \cdot 1 = 0.5$. $\Pr[1] = 0.6 \cdot 0.5 = 0.3$. $\Pr[2] = 0.2 \cdot 1 = 0.2$. Exactly $p$.

With $\alpha = 0.7$ and $k = 4$: $(1 - 0.7^5)/0.3 = (1 - 0.16807)/0.3 \approx 2.77$ tokens per target forward. If the draft costs $c = 0.1$ of the target, a round costs $4 \cdot 0.1 + 1 = 1.4$, and the speedup is $2.77 / 1.4 \approx 1.98\times$.

## 4. Tensor shapes, dtype, device

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| prompt ids (prefill input) | $(B, P)$ | int64 | model's |
| decode input | $(B, 1)$ | int64 | model's |
| cache, per layer, `k[l]` / `v[l]` | $(B, n_h, S, d_h)$ | activation dtype (bf16 in production; fp32/fp64 in our tests) | GPU in production; the largest inference allocation after the weights |
| new $q, k, v$ for a chunk | $(B, n_h, T_{\text{new}}, d_h)$ | activation dtype | model's |
| chunk mask $M$ | $(T_{\text{new}}, S)$, broadcast over $(B, n_h)$ | activation dtype (0 / $-\infty$) | model's |
| positions | $(T_{\text{new}},)$ = `arange(s, s + T_new)` | int64 | model's |
| chunk logits | $(B, T_{\text{new}}, V)$; only the last row is sampled from in plain decoding | activation dtype | model's |
| speculative verify: input / target probs / draft probs | $(1, k+1)$ / $(k+1, V)$ / $(k, V)$ | int64 / float / float | model's |
| paged cache in vLLM-style engines | e.g. a pool of $n_{\text{blocks}}$ blocks of $\text{block\_size}$ token slots per layer (exact layout differs by engine and kernel) plus an int32 block table per sequence | bf16 / int32 | GPU |

The positions matter: with the learned absolute position embedding `wpe` of 06.1, a decode step at cache length $s$ must use position $s$, not 0.

## 5. From scratch: `llmre.model.serving`

The code is in `code/src/llmre/model/serving.py`, tested in `code/tests/test_serving.py`; the experiment is `code/experiments/serving_costs.py`. It reuses the GPT's own weights (`c_attn`, `c_proj`, `ln_1`, `mlp`, `wte`, `wpe`, `lm_head`) and does not change `GPT.forward` or `GPT.generate`.

### 5.1 The cache and the chunked attention

```python
class KVCache:
    def __init__(self, n_layer):
        self.n_layer = n_layer
        self.k = [None] * n_layer          # each (B, nh, S, d_h) once filled
        self.v = [None] * n_layer

    @property
    def length(self):
        return 0 if self.k[0] is None else self.k[0].size(2)

    def append(self, layer, k, v):         # k, v: (B, nh, T_new, d_h)
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]


def _attention_with_cache(attn, x, cache, layer, start):
    B, T, C = x.shape
    q, k, v = attn.c_attn(x).split(attn.n_embd, dim=2)
    q = q.view(B, T, attn.n_head, attn.d_h).transpose(1, 2)    # (B, nh, T, d_h)
    k = k.view(B, T, attn.n_head, attn.d_h).transpose(1, 2)
    v = v.view(B, T, attn.n_head, attn.d_h).transpose(1, 2)
    K, V = cache.append(layer, k, v)                           # (B, nh, start+T, d_h)
    S = K.size(2)
    q_pos = torch.arange(start, start + T)[:, None]            # absolute query positions
    k_pos = torch.arange(S)[None, :]
    mask = torch.zeros(T, S, dtype=x.dtype).masked_fill(k_pos > q_pos, float("-inf"))
    y = scaled_dot_product_attention(q, K, V, mask=mask)       # the 05.2 function
    return attn.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))
```

`forward_with_cache(model, idx_new, cache)` is `GPT.forward` with that attention: embed the new tokens at positions `start..start+T-1`, run each block as `x + attn(ln_1(x))` then `x + mlp(ln_2(x))`, apply `ln_f` and `lm_head`. `generate_cached` is one prefill followed by one-token decode steps, sampling with the same temperature/top-$k$ rule as 06.3.

`test_serving.py` checks, in float64: prefill logits equal `model(idx)` to $10^{-10}$; prefill-then-decode logits equal the full forward at every position; a 4-token chunk on top of a truncated cache equals the full forward too; and greedy `generate_cached` produces exactly the tokens of the plain loop.

One limitation is deliberate. 06.3 crops the context to `block_size` when it gets too long. With learned absolute positions, cropping shifts every token's position, so every cached key would be wrong and the cache would have to be rebuilt. `forward_with_cache` asserts that the sequence fits instead. Models with RoPE (12.1) and sliding-window attention can evict old entries, because a key's rotation depends only on its own position.

### 5.2 Schedulers and paging

`simulate_static_batching` and `simulate_continuous_batching` count decode steps for a list of output lengths (prefill is ignored, since both would pay it). `PagedKVAllocator` is the bookkeeping half of PagedAttention: a free list of physical blocks and a block table per sequence. `append_token` takes a new block only when the last one is full, and `free` returns a finished sequence's blocks. The attention kernel would then gather keys through the block table instead of from one contiguous tensor. We keep the contiguous `KVCache` for the math and use the allocator only to measure memory.

### 5.3 Speculative decoding

```python
def speculative_accept(p, q, draft, generator=None, greedy=False):
    k = draft.numel()                       # p: (k+1, V), q: (k, V), draft: (k,)
    for i in range(k):
        x = int(draft[i])
        if greedy:
            if x == int(p[i].argmax()):
                continue
            return i, int(p[i].argmax())
        r = torch.rand((), generator=generator)
        if float(r) < min(1.0, float(p[i, x]) / max(float(q[i, x]), 1e-30)):
            continue                        # accept draft i
        resid = residual_distribution(p[i], q[i])
        return i, int(torch.multinomial(resid, 1, generator=generator))
    if greedy:
        return k, int(p[k].argmax())
    return k, int(torch.multinomial(p[k], 1, generator=generator))   # bonus token
```

`speculative_generate(target, draft_model, idx, n, k)` runs rounds. The draft proposes $k$ tokens with $k$ cached decode steps. The target runs **one** `forward_with_cache` over the committed tokens it has not cached yet plus the $k$ drafts. Then the round accepts a prefix and appends one token. Both caches are then **truncated** to the committed sequence, which throws away the rejected drafts' keys and values. The invariant that keeps this simple is that each cache always holds a prefix of the committed tokens, and before any forward you feed it the committed tokens it is missing.

The tests check three things. The term-by-term formula of section 2.5 equals $p$ for random $p, q$. A 40,000-sample Monte Carlo of `speculative_accept` reproduces $p = [0.5, 0.3, 0.2]$ within 0.01. Greedy speculative output equals the target's greedy output token for token, and a draft identical to the target accepts every draft.

### 5.4 Run it

```bash
cd code && PYTHONPATH=src python experiments/serving_costs.py
```

What you should see (numbers from one laptop CPU; yours will differ, the pattern should not):

```text
A. KV cache (200 new tokens, 4-layer d=128 model, CPU)
   no cache :   1.34 s
   KV cache :   0.43 s   speedup 3.1x
   identical tokens: True

B. Static vs continuous batching (64 requests, long-tailed lengths)
   batch  4: static  2769 steps (45% busy) | continuous  1413 steps (87% busy)
   batch  8: static  2028 steps (30% busy) | continuous   813 steps (76% busy)
   batch 16: static  1192 steps (26% busy) | continuous   615 steps (50% busy)

C. Speculative decoding (target 4 layers, draft 1 layer)
   k=4: accepted 48/52 drafts (92%), 4.62 tokens per target forward, identical to target greedy: True
```

Observations to check. (A) The speedup grows with the number of generated tokens; on a CPU it is modest because a tiny model's per-call overhead dominates. (B) With long-tailed lengths, static batching leaves most slots idle waiting for the longest request, and continuous batching roughly halves the steps. (C) The corpus is repetitive, so the 1-layer draft is usually right and each target forward yields several tokens. The output is still exactly the target's. On real text, acceptance is much lower, and the gain depends on how well the draft matches the target.

## 6. What production engines do, and what it costs

<div class="callout pt"><p><strong>PUBLICLY DOCUMENTED.</strong></p>
<ul>
<li><strong>Iteration-level (continuous) batching</strong>: Orca (Yu et al., OSDI 2022) schedules one model iteration at a time, so finished requests leave and new ones join between steps. It is the default scheduler in vLLM, SGLang and TensorRT-LLM.</li>
<li><strong>Paged KV memory</strong>: vLLM's PagedAttention (Kwon et al., SOSP 2023) stores the cache in fixed-size blocks addressed through block tables, like virtual memory pages. It reports near-zero KV waste, block sharing across sequences, and 2–4× higher throughput than the systems it compared against (authors' measurements).</li>
<li><strong>Prefix caching</strong>: requests that share a prefix (a system prompt, or the $G$ samples of one GRPO prompt) can share its cached blocks. SGLang's RadixAttention (Zheng et al. 2023) keeps cached prefixes in a radix tree and reuses them automatically.</li>
<li><strong>Chunked prefill</strong>: a long prompt's prefill is split into chunks and batched together with other requests' decode steps, so decodes are not stalled behind it (Sarathi-Serve, Agrawal et al. 2024). Our `forward_with_cache` already supports chunks: it is the $s > 0$, $T_{\text{new}} > 1$ case.</li>
<li><strong>Prefill/decode disaggregation</strong>: run the two phases on different GPUs, sized for each (DistServe, Zhong et al., OSDI 2024), and move the cache between them.</li>
<li><strong>Speculative decoding</strong> is built into vLLM with several drafters: a separate draft model, n-gram lookup in the prompt, EAGLE heads, and the model's own multi-token-prediction (MTP) heads. vLLM's documentation states that the standard rejection sampler is lossless up to hardware numerics, and marks relaxed-verification variants as not distribution-preserving.</li>
</ul></div>

**Cost summary.**

| | compute | memory traffic per step | memory held |
|---|---|---|---|
| no cache (06.3) | $O(t)$ positions per step, $O(t^2)$ total | weights + activations for $t$ positions | activations only |
| KV cache, decode | 1 position per sequence | all weights + each sequence's whole cache | $2\,n_{\text{layer}}\,g\,d_h\,S\,B\,b$ |
| continuous batching | same per token | same | same; higher utilization |
| paging | same | gather through block table (small overhead) | waste $< 1$ block per sequence |
| speculative decoding | $kc + 1$ target-forward units per round, extra FLOPs for rejected drafts | weights read once per round for up to $k+1$ tokens | draft model weights + cache |

**When the tricks stop paying.** Speculative decoding converts spare compute into fewer weight reads. At large batch sizes decode moves toward the ridge point, there is less spare compute, and the extra FLOPs on rejected drafts start to cost real time. Gains are largest at small batch and low latency targets, which is REASONABLE INDUSTRY PRACTICE and also follows from section 2.4. Relaxed acceptance rules that accept more drafts change the output distribution. One October 2026 paper reports that they also make completions longer, which cancels much of the speed gain (single paper, tracked in the course's research records, not taught here).

**Serving metrics.** **TTFT** (time to first token) is mostly prefill plus queueing. **TPOT** or inter-token latency is the decode step time. **Throughput** is tokens/s across all requests. **Goodput** is the request rate served within latency targets on both TTFT and TPOT, which is DistServe's metric. A scheduler that maximizes throughput can still miss latency targets. Say which metric you mean.

### 6.1 Why rollouts dominate RL post-training cost

In GRPO (16.1) each step samples $G$ completions for each prompt, often thousands of tokens long, then trains on them. Count the cost per token for a 7B model on one A100:

- **Training** costs about $6 N_p$ FLOPs per token (07.4) and is compute-bound. At 40% MFU (125 TFLOP/s) that is $4.2 \times 10^{10} / 1.25 \times 10^{14} \approx 0.34$ ms per token.
- **Generating** that token costs $2 N_p$ FLOPs, 3× fewer. But decode at batch 64 and context 1024 took 23.7 ms per 64 tokens in section 3, which is 0.37 ms per token: slightly *more* time per token than training on it, because decode runs at about 12% of peak.

Then add the synchronous-batch effect from section 3. The trainer cannot start until the longest completion in the batch finishes, and reasoning-length distributions are long-tailed, so many slots sit idle, as in part B of the experiment. Both effects push rollout above half of the step. This is INFERENCE from the cost model. The measured version is PUBLICLY DOCUMENTED: RollPacker (Gao et al. 2025) reports that the rollout stage takes about 70% of each training step across its three tasks, and AReaL (Fu et al. 2025) decouples generation from training because synchronous generation "must wait until the longest output in the batch is completed". This is also why RL stacks use a separate serving engine for rollouts, which creates the training–inference mismatch of [16.4](lessons/module-16/lesson-04.md).

## 7. Side by side, and when to keep the original

| | 06.3 loop | KV-cached decode | + speculative decoding |
|---|---|---|---|
| output | reference | same distribution (same tokens for greedy, up to float rounding) | same distribution (lossless rule) |
| work per new token | whole prefix | one position + cache read | $kc + 1$ forwards per $1..k+1$ tokens |
| extra memory | none | the cache | the cache + the draft model and its cache |
| code complexity | 10 lines | positions, masks, cache management | rollback, two caches, acceptance rule |
| use it when | testing, debugging, tiny models, one-off samples | always for real generation | latency-sensitive, small-batch decoding with a good draft |

Keep the 06.3 loop as the test oracle. Every cached or speculative path in this lesson is checked against it, and any change you make to a serving path should be too.

## Exercise: what does GQA buy the 7B model?

Redo the batch-64, context-1024 calculation of section 3 with $g = 8$ KV heads instead of 32 (the same idea as 12.3). What are the KV bytes, the step time and the tokens/s? What is the largest batch that fits in 80 GB at context 1024, before and after?

<details><summary>Hint</summary>

Only the KV term changes, by a factor $g / n_h = 8/32$. The weights (14 GB) and the FLOPs stay the same.

</details>

<details><summary>Stronger hint</summary>

KV per token becomes $2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072$ bytes = 0.125 MiB. Max batch: $(80 - 14)$ GB divided by the KV per sequence at 1024 tokens.

</details>

<details><summary>Solution</summary>

KV $= 64 \cdot 1024 \cdot 131{,}072 \approx 8.6$ GB. Bytes $= 14 + 8.6 = 22.6$ GB, so $22.6 / 2.039 \approx 11.1$ ms, versus 2.9 ms of FLOPs: still memory-bound. Throughput $64 / 0.0111 \approx 5{,}800$ tokens/s, 2.1× the MHA figure. Max batch: per sequence MHA needs 0.537 GB, so $66 / 0.537 \approx 122$. GQA-8 needs 0.134 GB, so about 490. (In practice you also reserve memory for activations and fragmentation.) Larger batches move decode toward the ridge, which is why GQA is an inference optimization.

</details>

## Implementation exercise: prefix sharing for GRPO groups

GRPO samples $G$ completions of the **same** prompt. Write `generate_group(model, prompt, G, n, generator)` that prefills the prompt **once** (batch 1), then expands the cache to batch $G$ and decodes all $G$ completions together. Test that each completion's logits match running `generate_cached` on that completion alone. Measure how many prefill positions you saved.

<details><summary>Hint</summary>

After the prefill, every layer's `k[l]` is $(1, n_h, P, d_h)$. You need $(G, n_h, P, d_h)$, and you also need $G$ copies of the last-position logits to sample $G$ different first tokens.

</details>

<details><summary>Stronger hint</summary>

`cache.k[l] = cache.k[l].expand(G, -1, -1, -1)` creates a view without copying. The next `torch.cat` in `append` then materializes a $(G, \dots)$ tensor, which is fine. A paged engine would avoid the copy altogether by pointing $G$ block tables at the same prompt blocks.

</details>

<details><summary>Solution</summary>

```python
@torch.no_grad()
def generate_group(model, prompt, G, n, generator=None):
    cache = KVCache(model.cfg.n_layer)
    last = forward_with_cache(model, prompt, cache)[:, -1]           # (1, V)
    for l in range(cache.n_layer):
        cache.k[l] = cache.k[l].expand(G, -1, -1, -1)
        cache.v[l] = cache.v[l].expand(G, -1, -1, -1)
    logits = last.expand(G, -1)                                      # (G, V)
    out = prompt.expand(G, -1)
    for step in range(n):
        nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator)
        out = torch.cat([out, nxt], dim=1)
        if step < n - 1:
            logits = forward_with_cache(model, nxt, cache)[:, -1]
    return out
```

The prefill runs over $P$ positions instead of $G \cdot P$. To test, take each row of `out`, run `model(row[None])` and compare its logits with the ones you sampled from (record them in the loop). For $P = 1000$ and $G = 8$ you save 7,000 prefill positions per prompt. For long reasoning completions decode still dominates, which is why engines also share the prompt's *memory* (paging), not just its compute.

</details>

## Debugging exercise

A colleague's cached generation matches the plain loop for the first token and then drifts:

```python
def forward_with_cache_buggy(model, idx_new, cache):
    B, T = idx_new.shape
    pos = torch.arange(T)                       # positions for the new tokens
    x = model.transformer.wte(idx_new) + model.transformer.wpe(pos)
    ...
```

What is wrong, and why is the first token still right?

<details><summary>Answer</summary>

The positions restart at 0 on every call. During prefill the cache is empty, so `arange(T)` happens to be correct, and the first generated token, which comes from the prefill logits, is right. Every decode step then embeds the new token at position 0 instead of position $s$, so its query, key and value are computed from the wrong input, and the outputs drift. Fix: `pos = torch.arange(cache.length, cache.length + T)`. `test_prefill_then_decode_matches_full_forward` catches this; a prefill-only test would not. The same bug in a RoPE model rotates the new query and key by the wrong angle.

</details>

## Common mistakes

- **Wrong positions on decode** (above), or a causal mask that is not offset by the cache length for multi-token chunks. Query $i$ of a chunk sees keys up to $s + i$, not up to $i$.
- **Feeding the sampled token twice, or never.** Each committed token must enter the cache exactly once. After the last step, the final sampled token has not been fed, which is correct: nothing needs its keys yet.
- **Not rolling back rejected drafts.** If the target cache keeps the keys of rejected draft tokens, every later step attends to tokens that are not in the text.
- **Calling a relaxed acceptance rule "speculative decoding" without saying so.** Only the $\min(1, p/q)$ rule with residual resampling is lossless. Anything that accepts more changes the outputs and must be evaluated as a different model.
- **Measuring speedup at the wrong batch size.** Speculative decoding's gain at batch 1 says little about batch 256. Report the batch size, context length and acceptance rate.
- **Reading GPU memory as "free" for the batch.** Weights, cache, activations and allocator fragmentation all count. The KV cache grows with every generated token, so a batch that fits at step 1 may not fit at step 4000.
- **Assuming cached logits are bitwise equal to uncached ones.** Different matmul shapes can reduce in a different order, so the values agree only up to rounding. 16.4 shows why that matters for RL.

## What to read / what to skip

- **Efficient Memory Management for LLM Serving with PagedAttention** (vLLM, [arXiv 2309.06180](https://arxiv.org/abs/2309.06180)): read the sections on KV-cache memory waste and on the block-table design; skim the kernel and distributed-execution details.
- **Orca** ([OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)): read the motivation for iteration-level scheduling; skip selective batching unless you build a scheduler.
- **Fast Inference from Transformers via Speculative Decoding** ([arXiv 2211.17192](https://arxiv.org/abs/2211.17192)): read the algorithm, the correctness proof and the expected-speedup analysis (sections 2–3); skim the T5 experiments. **Accelerating LLM Decoding with Speculative Sampling** ([arXiv 2302.01318](https://arxiv.org/abs/2302.01318)) is the same idea found independently at DeepMind; read its short algorithm box.
- **vLLM speculative decoding docs** ([docs.vllm.ai](https://docs.vllm.ai/en/latest/features/speculative_decoding/)): read the list of drafter types and the losslessness section, including its caveats about numerics and batch size.
- **Sarathi-Serve** ([arXiv 2403.02310](https://arxiv.org/abs/2403.02310)) and **DistServe** ([arXiv 2401.09670](https://arxiv.org/abs/2401.09670)): read the introductions to see the prefill/decode interference problem and its two answers; skip the system details.
- **SGLang** ([arXiv 2312.07104](https://arxiv.org/abs/2312.07104)): read the RadixAttention section; skip the frontend language.
- **RL cost**: RollPacker ([arXiv 2509.21009](https://arxiv.org/abs/2509.21009)), section 2.2, for the measured rollout share; AReaL ([arXiv 2505.24298](https://arxiv.org/abs/2505.24298)), introduction, for why synchronous rollouts waste GPU time.

## Code this after reading

Add **top-$p$** to `speculative_generate`'s sampling path. The draft and the target must both sample from their *truncated, renormalized* distributions, and the acceptance rule uses those truncated $p$ and $q$. Write a Monte Carlo test like `test_speculative_accept_monte_carlo_matches_target`, but against the target's top-$p$ distribution. Then explain why an RL trainer that scores these tokens under the *untruncated* distribution gets biased importance ratios (16.4, section on sampling transforms).

## Research connection

<div class="callout paper"><p><strong>Efficient Memory Management for Large Language Model Serving with PagedAttention</strong> — Kwon et al., SOSP 2023, <a href="https://arxiv.org/abs/2309.06180">arXiv 2309.06180</a>. <strong>Orca: A Distributed Serving System for Transformer-Based Generative Models</strong> — Yu et al., OSDI 2022, <a href="https://www.usenix.org/conference/osdi22/presentation/yu">USENIX</a>. <strong>Fast Inference from Transformers via Speculative Decoding</strong> — Leviathan, Kalman, Matias, ICML 2023, <a href="https://arxiv.org/abs/2211.17192">arXiv 2211.17192</a>. <strong>Accelerating Large Language Model Decoding with Speculative Sampling</strong> — Chen et al. 2023, <a href="https://arxiv.org/abs/2302.01318">arXiv 2302.01318</a>. <strong>SGLang</strong> — Zheng et al., <a href="https://arxiv.org/abs/2312.07104">arXiv 2312.07104</a>. <strong>Sarathi-Serve</strong> — Agrawal et al., <a href="https://arxiv.org/abs/2403.02310">arXiv 2403.02310</a>. <strong>DistServe</strong> — Zhong et al., OSDI 2024, <a href="https://arxiv.org/abs/2401.09670">arXiv 2401.09670</a>. RL rollout cost: <strong>RollPacker</strong> — Gao et al., <a href="https://arxiv.org/abs/2509.21009">arXiv 2509.21009</a>; <strong>AReaL</strong> — Fu et al., <a href="https://arxiv.org/abs/2505.24298">arXiv 2505.24298</a>.</p></div>

**Documented vs inferred.** *Publicly documented:* the algorithms of all the papers above and their reported speedups (authors' measurements); vLLM's list of drafters and its losslessness statement; the RollPacker rollout share. *Reasonable industry practice:* continuous batching and paged or block-based KV memory as the default in open serving engines; using speculative decoding mainly for latency-bound, small-batch serving; a separate serving engine for RL rollouts. *Inference:* the 7B-on-A100 numbers in sections 3 and 6.1, which are roofline lower bounds, not measurements. Real engines add kernel overheads, attention FLOPs and scheduling gaps.

## Check yourself

<details><summary>Why can the keys and values of earlier tokens be cached, and what in the architecture makes that true?</summary>

In a causal decoder the representation of position $j$ at every layer depends only on tokens $1..j$, because the mask stops position $j$ from attending to later positions. Adding a new token at the end cannot change $k_j$ or $v_j$ for $j$ already processed. A bidirectional encoder (BERT) has no such property: every new token would change every position.

</details>

<details><summary>A decode step at batch 1 takes 7 ms on an A100 for a 7B model. Which resource is the bottleneck, and what would buying a GPU with twice the FLOP/s do?</summary>

HBM bandwidth: the step must read 14 GB of weights (plus the cache) and does only about $1.4 \times 10^{10}$ FLOPs, which takes 0.045 ms at peak. Doubling peak FLOP/s changes almost nothing. Doubling bandwidth would roughly halve the step time.

</details>

<details><summary>Batching 64 sequences gave 19× the throughput, not 64×. Why?</summary>

Batching shares the weight read across 64 sequences, but each sequence still reads its own KV cache. At context 1024 the 64 caches add 34 GB per step, so the bytes per step grow from 14.5 GB to 48 GB. Throughput rises by $64 \times 14.5/48.4 \approx 19$.

</details>

<details><summary>Show in one line that the speculative acceptance rule emits token $x$ with probability $p(x)$.</summary>

$\Pr[x] = \min(p(x), q(x)) + \big(1 - \sum_y \min(p(y), q(y))\big)\cdot\frac{\max(0, p(x)-q(x))}{\sum_y \max(0, p(y)-q(y))} = \min(p(x),q(x)) + \max(0,p(x)-q(x)) = p(x)$, because $1 - \sum_y \min(p,q) = \sum_y \max(0, p - q)$.

</details>

<details><summary>A draft model has acceptance rate 0.5 and costs 5% of the target. Is $k = 8$ a good choice?</summary>

Tokens per round: $(1 - 0.5^9)/0.5 \approx 2.0$. Cost: $8 \cdot 0.05 + 1 = 1.4$. Speedup $\approx 1.43$. With $k = 3$: $(1 - 0.5^4)/0.5 = 1.875$ tokens for $1.15$, speedup $\approx 1.63$. With a low acceptance rate, long drafts are mostly rejected and their cost is wasted, so $k = 8$ is worse than a short draft.

</details>

<details><summary>Why does static batching waste so much in RL rollouts, and what fixes it inside one engine?</summary>

Reasoning completions have long-tailed lengths. A static batch runs until its longest member finishes, so most slots sit idle for most of the batch's lifetime. Continuous batching refills finished slots at the next iteration. Across the whole RL step, the trainer still waits for the slowest completion, which is what asynchronous RL systems (AReaL) and tail-aware schedulers (RollPacker) address.

</details>

<details><summary>Why does `forward_with_cache` refuse to go past `block_size` when 06.3's loop simply crops the context?</summary>

Cropping shifts every token's absolute position by one. With learned position embeddings, each cached key was computed with the old position, so all of them become wrong and the cache would have to be recomputed from scratch, which removes the benefit. Relative schemes (RoPE, 12.1) plus sliding-window attention let you evict old entries without recomputing the rest.

</details>

## Next

You can now predict where inference time goes: prefill is a training-like pass, decode is a bandwidth problem, and the cache is the memory that caps the batch. Module 9 takes the same bytes-versus-FLOPs reasoning from one GPU to many. When you reach [12.3](lessons/module-12/lesson-03.md), you will see GQA attack the cache size directly, and in [16.4](lessons/module-16/lesson-04.md) you will see what happens when the rollout engine of section 6.1 and the trainer disagree about the numbers.

Continue to [09.1 · Data parallelism, all-reduce, DDP](lessons/module-09/lesson-01.md).
