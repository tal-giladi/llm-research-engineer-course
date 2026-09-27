# 12.5 · Linear attention, Gated DeltaNet &amp; hybrid stacks

<div class="callout key"><p><strong>Added 2026-09-27 by the weekly curriculum review</strong> (<a href="#/research/CHANGELOG">changelog</a>). This extends Module 12; nothing earlier is replaced. The softmax attention you built in <a href="#/lessons/module-05/lesson-02">05.2</a> and shrank with GQA in <a href="#/lessons/module-12/lesson-03">12.3</a> is still in every model below: hybrid stacks keep it for one layer in four and swap the rest for a cheaper recurrent layer.</p></div>

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-05/lesson-02">05.2 · Q/K/V &amp; scaled dot-product attention</a> (scores $QK^\top$, causal mask); <a href="#/lessons/module-12/lesson-03">12.3 · MQA / GQA</a> (the KV-cache formula $2\,n_{\text{layer}}\,g\,d_h\,T\,b$); <a href="#/lessons/module-12/lesson-02">12.2 · RMSNorm &amp; SwiGLU</a> (RMSNorm, SiLU gating); outer products and matrix–vector products from <a href="#/lessons/module-01/lesson-01">Module 1</a>.</p>
<p><strong>You will learn:</strong> why the KV cache still grows linearly with context even after GQA; how removing the softmax turns attention into an RNN with a fixed-size matrix state (<strong>linear attention</strong>); why that state gets "full" and how a <strong>decay gate</strong> and the <strong>delta rule</strong> fix it; the <strong>Gated DeltaNet</strong> update written term by term and worked by hand; a from-scratch recurrent implementation in <code>llmre.attention.gated_deltanet</code> checked against a quadratic reference; and the memory arithmetic of the 3:1 <strong>hybrid</strong> layouts in Qwen3-Next and Kimi Linear.</p>
<p><strong>Why this matters for ML:</strong> hybrid linear/full-attention stacks are shipped in open-weight frontier MoE models from several labs and supported by vLLM and SGLang. Reading a 2026 model card, you will see "Gated DeltaNet", "3:1 hybrid" or "KDA" next to "GQA" and "MoE"; you need to know what state such a layer carries, what it costs, and what it gives up.</p>
</div>

<div class="hw">
<p><strong>Hardware track:</strong> CPU-only is enough. All code runs on toy sizes ($T \le 512$, $C \le 64$) in seconds; the tests take &lt; 2 s. No GPU memory, 0 GPU-hours. Production training of these layers needs the chunked Triton kernels of <code>flash-linear-attention</code> on a GPU; we do not use them.</p>
</div>

## 1. Intuition: GQA made the cache smaller, not bounded

In [12.3](lessons/module-12/lesson-03.md) the KV cache was $2\,n_{\text{layer}}\,g\,d_h\,T\,b$ bytes. GQA shrank the factor $g$. The factor $T$ is still there: every new token appends one more key and one more value to every layer, and every decode step reads the whole cache. At 256K tokens of context, the cache — not the weights, not the FLOPs — decides how many sequences fit on a GPU and how fast each step is.

The alternative is an old idea: an **RNN**. An RNN keeps a fixed-size summary of the past and updates it once per token. Decode cost and memory are constant in $T$. The price is that the summary is lossy: a fixed-size state cannot remember everything a growing cache can.

This lesson takes three steps from attention toward a good RNN, then shows how production models combine the two:

1. **Linear attention**: delete the softmax; attention becomes an RNN whose state is a $d_v \times d_k$ matrix.
2. **Gating**: let the state forget ($\alpha_t$), so old junk does not pile up forever.
3. **The delta rule**: instead of *adding* each new key→value pair, *correct* the state so that key now returns that value ($\beta_t$). This is Gated DeltaNet.
4. **Hybrid**: keep full softmax attention in one layer out of every four for exact long-range lookup.

<div class="callout key"><p>Softmax attention stores every past key and value (memory grows with $T$) and looks them all up exactly. A linear-attention layer compresses the past into one fixed matrix per head (memory constant in $T$) and looks it up approximately. A hybrid stack uses mostly the cheap layer and a few exact ones.</p></div>

## 2. Mathematics: from softmax attention to an RNN

### 2.1 Linear attention

Causal softmax attention for query position $t$ (one head; §05.2):

$$
o_t = \sum_{s \le t} \frac{\exp(q_t \cdot k_s / \sqrt{d_k})}{\sum_{s' \le t}\exp(q_t \cdot k_{s'} / \sqrt{d_k})}\, v_s ,
$$

where $q_t, k_s \in \mathbb{R}^{d_k}$ are the query and key vectors, $v_s \in \mathbb{R}^{d_v}$ the value, $o_t \in \mathbb{R}^{d_v}$ the output. The softmax couples every $s$ through the denominator, so you cannot pre-sum anything.

Linear attention (Katharopoulos et al. 2020) replaces $\exp(q\cdot k)$ by a plain dot product (in general $\phi(q)\cdot\phi(k)$ for a feature map $\phi$; we fold $\phi$ into the projections and drop the normalizer, as modern variants do):

$$
o_t = \sum_{s \le t} (q_t \cdot k_s)\, v_s = \sum_{s \le t} v_s\,(k_s^\top q_t) = \Big(\underbrace{\sum_{s\le t} v_s k_s^\top}_{S_t}\Big)\, q_t .
$$

The bracket does not depend on $t$'s query, so define the **state** $S_t \in \mathbb{R}^{d_v \times d_k}$:

$$
S_t = S_{t-1} + v_t k_t^\top, \qquad o_t = S_t\, q_t, \qquad S_0 = 0 .
$$

Each $v_t k_t^\top$ is an **outer product** — a rank-1 $d_v\times d_k$ matrix that "stores $v_t$ under address $k_t$". Reading with $q_t$ returns a mix of stored values weighted by how well $q_t$ matches each key. The state has $d_v d_k$ numbers regardless of $T$.

### 2.2 Decay gate

Nothing is ever removed from $S_t$ above; after thousands of tokens it is a sum of thousands of rank-1 matrices in a $d_k$-dimensional space, and reads become mush. Add a per-token scalar **decay** $\alpha_t \in (0,1]$ (the Mamba2 / GLA idea):

$$
S_t = \alpha_t S_{t-1} + v_t k_t^\top .
$$

Unrolling, token $s$'s contribution reaching step $t$ is multiplied by $D_{t,s} = \prod_{r=s+1}^{t} \alpha_r$, so the equivalent quadratic form is

$$
o_t = \sum_{s \le t} D_{t,s}\,(q_t\cdot k_s)\, v_s .
$$

That is exactly "causal attention with no softmax and a decay mask" — the function `causal_linear_attention_parallel` computes it with a $(T,T)$ matrix, and the test suite checks the recurrence against it.

### 2.3 The delta rule

Decay forgets *everything* uniformly. The **delta rule** (Widrow–Hoff; used for linear attention by Schlag et al. 2021 and scaled up by Yang et al. 2024) edits one slot. Before writing, ask the memory what it currently returns for key $k_t$: $\hat v_t = S_{t-1} k_t$. Then move that answer a fraction $\beta_t \in [0,1]$ toward the new value:

$$
S_t = S_{t-1} + \beta_t\,(v_t - S_{t-1} k_t)\, k_t^\top = S_{t-1}\,(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top .
$$

Here $I$ is the $d_k\times d_k$ identity, $\beta_t$ the **write strength**, and $(v_t - S_{t-1}k_t)$ the **prediction error** — the "delta". If $\lVert k_t \rVert = 1$ and $\beta_t = 1$ then $S_t k_t = S_{t-1}k_t + (v_t - S_{t-1}k_t)(k_t^\top k_t) = v_t$: the key now returns exactly the new value. This is one step of gradient descent on the loss $\tfrac12\lVert S k_t - v_t\rVert^2$ with learning rate $\beta_t$ — the state is a tiny associative memory trained online.

### 2.4 Gated delta rule (Gated DeltaNet)

Yang, Kautz &amp; Hatamizadeh (ICLR 2025) combine both:

$$
\boxed{S_t = \alpha_t\, S_{t-1}\,(I - \beta_t k_t k_t^\top) + \beta_t\, v_t k_t^\top, \qquad o_t = S_t\, q_t}
$$

equivalently $S_t = \alpha_t S_{t-1} + \beta_t\,(v_t - \alpha_t S_{t-1}k_t)\,k_t^\top$ (forget first, then correct). $\alpha_t$ erases fast (e.g. at a topic change), $\beta_t$ edits precisely. Both are produced per token and per head by small linear layers from the input $x_t$. Setting $\alpha_t = 1$ recovers DeltaNet; setting $\beta_t=0$ means "don't write".

<div class="callout pt"><p><strong>PUBLICLY DOCUMENTED.</strong> Gated DeltaNet also L2-normalizes $q$ and $k$ (so $I - \beta k k^\top$ has eigenvalues in $[0,1]$ and the state cannot blow up), applies a short causal convolution before the projections, and gates the output with $\mathrm{RMSNorm}(o_t)\odot \mathrm{SiLU}(W_g x_t)$. Kimi Linear's KDA replaces the scalar $\alpha_t$ by one decay per key channel (a diagonal matrix).</p></div>

## 3. Numerical example by hand

One head, $d_k = 2$, $d_v = 1$ (values are scalars, so $S_t$ is a $1\times 2$ row). Three tokens:

| $t$ | $k_t$ | $v_t$ | meaning |
|---|---|---|---|
| 1 | $(1, 0)$ | 3 | store 3 under address $e_1$ |
| 2 | $(1, 0)$ | 5 | *update* address $e_1$ to 5 |
| 3 | $(0, 1)$ | 2 | store 2 under address $e_2$ |

Read every step with $q_t = (1,0)$, i.e. "what is at $e_1$?". The right answer after step 2 is 5.

**Linear attention.** $S_1 = 3\,(1,0) = (3, 0)$, $o_1 = 3$. $S_2 = (3,0) + 5\,(1,0) = (8,0)$, $o_2 = 8$. $S_3 = (8,0) + 2\,(0,1) = (8,2)$, $o_3 = 8$. The two writes to the same address **collide** and sum to 8 — neither the old nor the new value.

**Delta rule ($\alpha = \beta = 1$).** $S_1 = 0 + (3 - 0)(1,0) = (3,0)$, $o_1 = 3$.
Step 2: prediction $S_1 k_2 = 3$, error $5 - 3 = 2$, $S_2 = (3,0) + 2\,(1,0) = (5,0)$, $o_2 = 5$.
Step 3: prediction $S_2 k_3 = 0$, error $2$, $S_3 = (5,0) + 2\,(0,1) = (5,2)$, $o_3 = 5$. The address was **overwritten**, and the orthogonal write did not disturb it. Reading $q=(0,1)$ from $S_3$ gives 2.

**Gated delta rule** with $\alpha_2 = 0.5$, $\beta_2 = 0.5$ at step 2 (others 1). Forget: $0.5\,(3,0) = (1.5, 0)$. Prediction $1.5$, error $5 - 1.5 = 3.5$, correction $0.5 \times 3.5 = 1.75$. $S_2 = (3.25, 0)$, $o_2 = 3.25$: half-forgotten old value plus a half-strength write. The gates are *learned*, so the model chooses per token how much to forget and how hard to write.

These exact numbers are asserted in `test_gated_deltanet.py::test_hand_worked_example_from_lesson`.

## 4. Tensor shapes, dtype, device

Using the head layout of 05.3/12.3, with $H$ heads and $d = C/H$:

| Tensor | Shape | Notes |
|---|---|---|
| input `x` | $(B, T, C)$ | float32 / bf16, on the model's device |
| `q`, `k` | $(B, H, T, d_k)$ | SiLU, then L2-normalized over $d_k$; `q` scaled by $d_k^{-1/2}$ |
| `v` | $(B, H, T, d_v)$ | |
| `alpha` | $(B, H, T)$ | $\exp(-\mathrm{softplus}(W_a x)) \in (0,1)$ |
| `beta` | $(B, H, T)$ | $\sigma(W_b x) \in (0,1)$ |
| state `S` | $(B, H, d_v, d_k)$ | **no $T$ axis**; kept in fp32 in practice |
| output `o` | $(B, H, T, d_v)$ → $(B, T, C)$ | per-head RMSNorm, SiLU gate, `o_proj` |

Compare the cache of one layer at decode time: softmax attention stores K and V of shape $(B, g, T, d_h)$ each, growing with $T$; Gated DeltaNet stores $S$ of shape $(B, H, d_v, d_k)$, fixed. Same device as the activations; nothing is ever appended.

## 5. From-scratch PyTorch

The core recurrence, from [`llmre.attention.gated_deltanet`](../../code/src/llmre/attention/gated_deltanet.py):

```python
def gated_delta_rule_recurrent(q, k, v, alpha, beta, initial_state=None):
    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    S = q.new_zeros(B, H, d_v, d_k) if initial_state is None else initial_state
    outs = []
    for t in range(T):
        k_t = k[..., t, :]                               # (B, H, d_k)
        S = alpha[..., t, None, None] * S                # forget
        pred = (S @ k_t[..., None]).squeeze(-1)          # memory's answer for k_t, (B, H, d_v)
        err = beta[..., t, None] * (v[..., t, :] - pred) # delta, scaled by write strength
        S = S + err[..., :, None] * k_t[..., None, :]    # rank-1 correction err k_t^T
        outs.append((S @ q[..., t, :, None]).squeeze(-1))  # read: o_t = S_t q_t
    return torch.stack(outs, dim=-2), S
```

`linear_attention_recurrent` is the same loop with `S = alpha*S + v_t k_t^T` and no prediction. The layer `GatedDeltaNet(cfg)` wraps it: projections for $q,k,v$, one decay and one write strength per head from `a_proj`/`b_proj` (`nn.Linear(C, n_head)`), per-head `RMSNorm` from 12.2, a SiLU output gate, and `o_proj`. It has two entry points:

```python
layer = GatedDeltaNet(GPTConfig(n_embd=16, n_head=4, bias=False))
y = layer(x)                                   # (B, T, C) -> (B, T, C), like GroupedQueryAttention
y_t, state = layer.forward_with_state(x_t, state)  # decode: feed one token, carry a (B, H, d, d) state
```

`hybrid_layer_pattern(n_layer, 3)` returns the layer types of a 3:1 stack, and `hybrid_decode_cache_bytes` adds the KV cache of the `'full'` layers to the states of the `'linear'` ones.

**Unit tests** (`code/tests/test_gated_deltanet.py`): recurrent linear attention equals the quadratic form with and without decay (float64, `1e-10`); the hand example above; write-then-read returns $v$ exactly; $\beta=0$ never writes; running on a split sequence with the carried state equals running on the whole sequence; the layer is causal; token-by-token decode reproduces the full forward; and the memory arithmetic below.

## 6. Under the hood: what production kernels do, and the cost

**The Python loop is the definition, not the implementation.** A loop over $T$ with tiny matmuls leaves a GPU idle. Production kernels (`flash-linear-attention`, used by vLLM and SGLang for these models) use the **chunkwise** form: split the sequence into chunks of e.g. 64 tokens; *within* a chunk, compute outputs with a small quadratic, masked matmul like §2.2 (tensor-core friendly); *between* chunks, pass the state $S$ along. For the delta rule the product of the $(I - \beta k k^\top)$ factors inside a chunk is not diagonal, and Yang et al. (2024) represent it compactly (a WY / Householder-product representation) so it too becomes a few matmuls. The result is the same function as our loop, up to float rounding.

**Cost per head** ($T$ tokens, $d_k = d_v = d$):

| | Softmax attention | Gated DeltaNet |
|---|---|---|
| Training FLOPs (prefill) | $O(T^2 d)$ | $O(T d^2)$ (chunked: $O(T\,c\,d + T d^2)$ for chunk size $c$) |
| Decode FLOPs per token | $O(T d)$ — read whole cache | $O(d^2)$ — update + read the state |
| Decode memory | $2 T d$ per KV head, grows | $d^2$ per head, constant |
| Exact recall of any past token | yes | no — lossy, $d^2$ numbers |

With $d = 128$, one state is $16{,}384$ numbers; a softmax head reading a 256K cache touches $2 \times 262{,}144 \times 128 \approx 67$M numbers per step.

**A real model's decode memory.** Qwen3-Next-80B-A3B (config from its model card, PUBLICLY DOCUMENTED): 48 layers in the pattern 3 × Gated DeltaNet → 1 × gated softmax attention; attention layers have $g = 2$ KV heads of $d_h = 256$; DeltaNet layers have 32 value heads of dimension 128 (and 16 query/key heads of dimension 128). Per sequence, KV in bf16, states in fp32 (the arithmetic is ours; run it with the module's helpers):

| Context $T$ | If all 48 layers were full attention | Hybrid: 12 full layers' KV + 36 states | Ratio |
|---|---|---|---|
| 4,096 | 0.403 GB | 0.101 GB + 0.075 GB = 0.176 GB | 2.3× |
| 32,768 | 3.22 GB | 0.805 GB + 0.075 GB = 0.881 GB | 3.7× |
| 262,144 | 25.8 GB | 6.44 GB + 0.075 GB = 6.52 GB | 4.0× |

The 36 states are a constant 75.5 MB; at long context the hybrid approaches the 4× saving of keeping only one layer in four with a cache. The remaining cost is the 12 full layers — which is why newer designs (the Qwen3.8-Next preview, 2026) replace even those with sparse attention.

<div class="callout pt"><p><strong>Serving consequence (PUBLICLY DOCUMENTED for vLLM).</strong> A paged KV-cache manager assumes every layer's cache grows by one slot per token. A hybrid model has two kinds of per-sequence memory — growing pages for the full layers and one fixed state block per linear layer — so vLLM added a hybrid cache manager and the flash-linear-attention Triton kernels to support Qwen3-Next. Prefix caching is also harder: a state summarizes the whole prefix, so you can only reuse it at the exact positions where you saved a snapshot.</p></div>

## 7. Side by side: which to use

| | MHA/GQA (12.3) | Linear attn / GDN only | 3:1 hybrid |
|---|---|---|---|
| Decode memory | grows with $T$ | constant | ~¼ of full, + constant |
| Exact copy / retrieval from far context | strong | weak (fixed memory) | recovered by the full layers |
| Maturity of tooling | universal | research kernels | vLLM, SGLang, HF Transformers |
| Used by | Llama 3, most models | research models | Qwen3-Next, Kimi Linear (2025–26) |

**When you would still use plain softmax attention (GQA):** short contexts where the cache is small anyway; tasks that need exact recall of arbitrary earlier tokens (copying, long-range lookup — pure fixed-state models are known to struggle there, which is why production models keep some full layers); and any time you want the simplest, best-supported stack. The attention you implemented in Module 5 remains the reference that every hybrid is measured against.

<div class="callout warn"><p><strong>Claim labels.</strong> <em>PUBLICLY DOCUMENTED:</em> the Gated DeltaNet update (ICLR 2025 paper); the 3:1 layouts of Qwen3-Next and Kimi Linear (model cards); vLLM/SGLang support. <em>(company claim):</em> Kimi Linear "outperforms full MLA" under an identical recipe with up to 75% less KV cache and up to 6× decode throughput at 1M context — reported by Moonshot, not independently reproduced. <em>INFERENCE:</em> that hybrid linear/sparse attention will become the default for long-context open models; several labs are moving that way, but full-attention models remain common.</p></div>

## Experiment: watch linear attention fill up

Write $N$ random key→value pairs (unit keys, $d_k = 16$, scalar values with variance 1) into a state, then read keys back and measure the mean squared error against their values — over all $N$ pairs and over the last 4 written — for linear attention (no decay) and the delta rule ($\alpha=\beta=1$), averaged over 50 random draws:

```python
import torch, torch.nn.functional as F
from llmre.attention.gated_deltanet import linear_attention_recurrent, gated_delta_rule_recurrent
torch.manual_seed(0)
d, trials = 16, 50
def mse(S, k, v): return ((k @ S[0, 0].T) - v).pow(2).mean().item()
for N in (4, 16, 64, 256):
    r = torch.zeros(4)
    for _ in range(trials):
        k = F.normalize(torch.randn(1, 1, N, d), dim=-1); v = torch.randn(1, 1, N, 1)
        ones = torch.ones(1, 1, N)
        _, S_lin = linear_attention_recurrent(k, k, v)           # q = k: read back each key
        _, S_del = gated_delta_rule_recurrent(k, k, v, ones, ones)
        kk, vv = k[0, 0], v[0, 0]
        r += torch.tensor([mse(S_lin, kk, vv), mse(S_del, kk, vv),
                           mse(S_lin, kk[-4:], vv[-4:]), mse(S_del, kk[-4:], vv[-4:])])
    print(N, [round(x, 2) for x in (r / trials).tolist()])
```

Output on CPU (PyTorch 2.x; the digits may differ slightly by version):

| $N$ | linear, all | delta, all | linear, last 4 | delta, last 4 |
|---|---|---|---|---|
| 4 | 0.27 | 0.16 | 0.27 | 0.16 |
| 16 | 0.90 | 0.64 | 0.85 | 0.14 |
| 64 | 3.61 | 1.39 | 3.85 | 0.19 |
| 256 | 15.59 | 1.98 | 15.43 | 0.25 |

**Expected observations.** Linear attention's error grows roughly like $N/d$: every stored pair adds cross-talk to every read, and nothing is ever removed, so after 256 writes a read is far worse than guessing 0 (MSE 1). The delta rule's error over *all* pairs stays bounded (old pairs are gradually overwritten — this is the fixed-memory limit of a $1	imes 16$ state), while the *most recent* writes stay readable (≈0.2) no matter how many came before. That recency is what the learned gates $\alpha_t,\beta_t$ then shape, and what the full-attention layers of a hybrid compensate for.

## Common mistakes

- **Forgetting that the state has no $T$ axis.** If your decode "cache" for a linear layer grows per token you have implemented softmax attention with extra steps.
- **Unnormalized keys in the delta rule.** With $\lVert k\rVert > 1$, $I - \beta k k^\top$ has the eigenvalue $1 - \beta\lVert k\rVert^2$, which falls below $-1$ once $\beta\lVert k\rVert^2 > 2$, and the state can explode. L2-normalize $k$.
- **Applying the decay after the write.** $S_t = \alpha_t(S_{t-1} + v_t k_t^\top)$ also decays the token you just wrote; the definition decays only the past.
- **Outer product the wrong way round.** $S$ is $(d_v, d_k)$ and the write is $v_t k_t^\top$; with $d_k = d_v$ the swapped version runs silently and computes $S^\top$.
- **Assuming a hybrid saves 4× at every context length.** The constant state memory dominates at short $T$ (2.3× at 4K in the table above).

## Exercise

A hybrid model has 64 layers in a 3:1 pattern. Full layers use $g = 8$ KV heads of $d_h = 128$ in bf16. Linear layers have 16 heads with $d_k = d_v = 128$, states in fp32. For batch 1, at what context length $T$ does the full layers' KV cache equal the total linear-state memory?

<details><summary>Hint</summary>

Count the layers of each type first, then write both memories with the formulas from §6 and set them equal.

</details>

<details><summary>Stronger hint</summary>

16 full layers, 48 linear layers. KV $= 2 \cdot 16 \cdot 8 \cdot 128 \cdot T \cdot 2$ bytes. States $= 48 \cdot 16 \cdot 128 \cdot 128 \cdot 4$ bytes.

</details>

<details><summary>Solution</summary>

KV $= 65{,}536\,T$ bytes. States $= 48 \cdot 16 \cdot 16{,}384 \cdot 4 = 50{,}331{,}648$ bytes (≈ 50 MB). Equal at $T = 50{,}331{,}648 / 65{,}536 = 768$ tokens. Beyond a few thousand tokens the full layers dominate. Check: `hybrid_decode_cache_bytes(hybrid_layer_pattern(64, 3), 768, 8, 128, 16, 128, 128)` is exactly twice `kv_cache_bytes(16, 8, 128, 768)`.

</details>

**Implementation exercise (code this after reading).** Add an optional per-channel decay (Kimi Linear's KDA idea): `alpha` of shape $(B, H, T, d_k)$, applied as `S = S * alpha_t[..., None, :]` (it scales the columns of $S$, i.e. each key channel). Write it as a new function next to `gated_delta_rule_recurrent`, and test that it reduces to the scalar version when all channels of `alpha` are equal.

## Debugging exercise

A colleague's Gated DeltaNet layer passes the "recurrent equals full-sequence" test but its token-by-token decode drifts from the full forward after a few hundred tokens in bf16, and the state's norm grows without bound. Their update line is:

```python
S = alpha_t * S + beta_t * (v_t - S @ k_t)[..., None] * k_t[..., None, :]
```

and `k` comes straight from `k_proj(x)`. Find the two problems.

<details><summary>Answer</summary>

(1) The prediction uses the *undecayed* state: it should be `v_t - alpha_t * S @ k_t` (forget first, then predict); otherwise, whenever $\alpha_t < 1$, the error is measured against a value the decayed state no longer holds, so afterwards $S_t k_t \neq v_t$ even with $\beta_t = 1$. (2) `k` is not L2-normalized, so $\lVert k\rVert$ can exceed 1 and $I - \beta k k^\top$ stops being a contraction: the state grows each step, and bf16 then loses precision. Normalize `k` (and keep the state in fp32).

</details>

## Research connection

<div class="callout paper"><p><strong>Gated Delta Networks: Improving Mamba2 with Delta Rule</strong> — Yang, Kautz, Hatamizadeh, ICLR 2025, <a href="https://arxiv.org/abs/2412.06464">arXiv 2412.06464</a>. The layer in this lesson. Builds on <strong>Transformers are RNNs</strong> (Katharopoulos et al. 2020, <a href="https://arxiv.org/abs/2006.16236">arXiv 2006.16236</a>) for linear attention and <strong>Parallelizing Linear Transformers with the Delta Rule over Sequence Length</strong> (Yang et al. 2024, <a href="https://arxiv.org/abs/2406.06484">arXiv 2406.06484</a>) for the chunked training algorithm. Production: <a href="https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct">Qwen3-Next model card</a> (3 × Gated DeltaNet : 1 × gated attention); <strong>Kimi Linear</strong> (Moonshot, <a href="https://arxiv.org/abs/2510.26692">arXiv 2510.26692</a>; 3:1 KDA : MLA); the Qwen3.8-Next architecture report (<a href="https://arxiv.org/abs/2608.30320">arXiv 2608.30320</a>), which adds sparse attention to the full layers. Kernels: <a href="https://github.com/fla-org/flash-linear-attention">flash-linear-attention</a>. An independent from-scratch walkthrough: <a href="https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md">LLMs-from-scratch, Gated DeltaNet</a>.</p></div>

**What to read / what to skip.** In the Gated DeltaNet paper read §2 (background: linear attention, Mamba2, delta rule) and §3.1 (the gated delta rule and its online-learning view). Skim §3.2–3.3 (chunkwise WY algorithm) only if you want to write kernels. Skip the ablation tables on first read. In Kimi Linear, read the architecture section and the 3:1 hybrid ablation; treat the headline comparisons as company claims.

## Check yourself

<details><summary>Why is linear attention an RNN but softmax attention is not?</summary>

Without the softmax, $\sum_{s\le t}(q_t\cdot k_s)v_s = (\sum_{s\le t} v_s k_s^\top)\,q_t$: the sum over the past can be accumulated into one $d_v\times d_k$ matrix before the query arrives, updated once per token. The softmax's normalizer $\sum_{s'}\exp(q_t\cdot k_{s'})$ depends on $q_t$ and couples all past positions non-linearly, so there is no query-independent summary; you must keep every $k_s, v_s$.

</details>

<details><summary>In the hand example, why does linear attention return 8 while the delta rule returns 5?</summary>

Linear attention adds $v_t k_t^\top$ unconditionally, so two writes under the same key sum: $3 + 5 = 8$. The delta rule first reads what the key currently returns (3) and writes only the error $5 - 3 = 2$, so afterwards the key returns 5.

</details>

<details><summary>What do $\alpha_t$ and $\beta_t$ each control, and what are their "off" values?</summary>

$\alpha_t$ is global forgetting of the whole state (1 = keep everything). $\beta_t$ is how strongly the current key→value pair is written (0 = don't write). $\alpha=1$ gives DeltaNet; $\beta = 0$ gives a pure decay step.

</details>

<details><summary>Why do production models keep one full-attention layer in four instead of going fully linear?</summary>

A fixed $d_k \times d_v$ state per head cannot store an unbounded number of facts exactly; tasks that need precise retrieval from far back (copying, needle-in-a-haystack, long-range lookup) suffer. A few full-attention layers restore exact access to any past token while the linear layers do most of the mixing cheaply. Both Qwen3-Next and Kimi Linear chose 3:1.

</details>

<details><summary>Qwen3-Next's decode memory at 256K tokens is about 6.5 GB instead of 25.8 GB. Where do the remaining 6.5 GB come from?</summary>

Almost entirely the KV cache of the 12 full-attention layers ($2\cdot 12 \cdot 2 \cdot 256 \cdot 262{,}144 \cdot 2 \approx 6.44$ GB). The 36 Gated DeltaNet states add only a constant 75.5 MB.

</details>

## Next

Module 12 is complete: you have RoPE, RMSNorm, SwiGLU, GQA, MoE and now the linear/hybrid alternative to a growing KV cache. Module 13 turns to measuring these models. Continue to [13.1 · Loss, perplexity, zero-/few-shot](lessons/module-13/lesson-01.md).
