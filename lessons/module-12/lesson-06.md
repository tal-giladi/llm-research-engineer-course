# 12.6 · Sparse attention: DeepSeek Sparse Attention (lightning indexer + top-k)

<div class="callout key"><p><strong>Added 2026-11-22 by the weekly curriculum review</strong> (<a href="#/research/CHANGELOG">changelog</a>). This extends Module 12; nothing earlier is replaced. The softmax attention you built in <a href="#/lessons/module-05/lesson-02">05.2</a> and shared across heads in <a href="#/lessons/module-12/lesson-03">12.3</a> is still the computation every query does here. The only change is <em>which</em> past tokens each query is allowed to look at. With the selection switched off, the layer in this lesson is exactly 12.3's GQA layer (a test checks this).</p></div>

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-05/lesson-02">05.2 · Q/K/V &amp; scaled dot-product attention</a> (scores, causal mask, softmax); <a href="#/lessons/module-12/lesson-03">12.3 · MQA / GQA</a> (KV heads shared across query heads, the KV-cache formula); <a href="#/lessons/module-12/lesson-05">12.5 · Linear attention &amp; hybrids</a> (the other way to make long context cheap, and the "one layer in four" hybrid layouts); KL divergence from <a href="#/lessons/module-01/lesson-03">01.3</a>; the decode roofline (time ≈ bytes read / bandwidth) from <a href="#/lessons/module-08/lesson-06">08.6</a>. MLA, DeepSeek's compressed-KV attention, is named but not needed: everything here works on the GQA layer you already have.</p>
<p><strong>You will learn:</strong> why dense attention costs $O(T^2)$ in prefill and $O(T)$ per decoded token, and which part of that cost sparsity can remove; <strong>DeepSeek Sparse Attention (DSA)</strong>: a small <strong>lightning indexer</strong> that scores every past token cheaply, <strong>top-k selection</strong> of the 2048 best, and full softmax attention over only those; how the indexer is trained to imitate the attention it replaces (a <strong>dense warm-up</strong>, then <strong>sparse training</strong> with a KL loss and a detached input); a hand-worked 4-token example; a from-scratch layer <code>llmre.attention.sparse.DSAttention</code> checked against 12.3's GQA; and the real arithmetic for DeepSeek-V3.2 at 128K context: about 11× fewer prefill multiply-adds, 13× fewer per decoded token, and <em>no</em> saving in KV-cache storage.</p>
<p><strong>Why this matters for ML:</strong> DSA is the sparse attention of DeepSeek-V3.2 and GLM-5 and, as "Gated DSA", the sparse layers of Kimi K3.1 Pro's linear + sparse hybrid. Long-context RL rollouts, agents with 100K-token transcripts and 1M-context models make attention the dominant cost, and the papers you will read next (indexer sharing across layers, block-sparse variants, sparse decoding kernels) all start from this mechanism. You need to know what it computes, what it saves, what it does not save, and how it can fail.</p>
</div>

<div class="hw">
<p><strong>Hardware track:</strong> CPU-only is enough. <code>tests/test_sparse_attention.py</code> (11 tests) runs in about 2 seconds. <code>experiments/dsa_indexer_warmup.py</code> runs in about 5 seconds on a laptop CPU, under 500 MB RAM, 0 GPU-hours. The DeepSeek-V3.2 numbers in section 6 are worked from its published config, not measured. Production DSA needs Hopper/Blackwell kernels (FlashMLA sparse, DeepGEMM indexer logits); we do not use them.</p>
</div>

## 1. Intuition: most of a long context gets almost no attention

Look at one row of a trained attention matrix at position 100,000. The softmax gives a weight to every one of the 100,000 earlier tokens, but almost all of the mass sits on a few hundred or a few thousand of them: the instruction at the start, the function being edited, the lines just above. The other 98% get weights so small that dropping them barely changes the output. Dense attention still computes a score for every one of them, and at decode time still reads every one of their keys and values from memory.

The obvious fix is "only attend to the important tokens". The catch is that you only know which tokens are important *after* computing the scores, which is the expensive part. DSA's answer is to compute the scores twice:

1. A **lightning indexer**: a second, much smaller attention-like scorer (a few heads of width 128, ReLU instead of softmax, FP8 in production). It looks at every past token, but each look is about 17× cheaper than the main attention's.
2. **Top-k selection**: each query keeps the $k = 2048$ past tokens the indexer scored highest.
3. **Main attention over the selection**: ordinary softmax attention, but over 2048 keys instead of $T$.

The indexer is trained to predict where the main attention would put its weight. When it does that well, the sparse layer computes almost the same output as the dense one, for a fraction of the cost.

<div class="callout key"><p>Linear attention (12.5) makes each step cheaper by replacing <em>exact</em> lookup with a fixed-size summary. Sparse attention keeps exact softmax lookup but makes each query look at <em>fewer</em> tokens, chosen by a cheap learned scorer. The KV cache is still stored in full; what shrinks is the computation and the bytes read per step.</p></div>

## 2. Mathematics

### 2.1 What dense attention costs

One layer, one sequence of $T$ tokens, $H$ query heads, key width $d_{qk}$, value width $d_v$. Causal attention scores every pair $(t, s)$ with $s \le t$: there are $T(T+1)/2$ such pairs. Each pair costs $H\,d_{qk}$ multiply-adds for the scores $q_t \cdot k_s$ and $H\,d_v$ for the weighted sum of values, so

$$
\text{MACs}_{\text{dense prefill}} = \frac{T(T+1)}{2}\; H\,(d_{qk} + d_v), \qquad \text{MACs}_{\text{dense decode}}(T) = T\; H\,(d_{qk}+d_v).
$$

Prefill grows like $T^2$; each decode step grows like $T$ and reads all $T$ cached keys and values.

### 2.2 The lightning indexer

DeepSeek-V3.2 (eq. 1 of the paper) scores past token $s$ for query token $t$ as

$$
I_{t,s} = \sum_{j=1}^{H^I} w^I_{t,j}\;\mathrm{ReLU}\!\left(q^I_{t,j} \cdot k^I_s\right),
$$

where

- $H^I$ is the number of **indexer heads** (64 in DeepSeek-V3.2, 32 in GLM-5);
- $q^I_{t,j} \in \mathbb{R}^{d^I}$ is indexer head $j$'s query for token $t$, a linear projection of $t$'s hidden state $h_t$ ($d^I = 128$ in both models);
- $k^I_s \in \mathbb{R}^{d^I}$ is the indexer key of token $s$, **one per token, shared by all indexer heads** (so the indexer cache is one 128-vector per token);
- $w^I_{t,j} \in \mathbb{R}$ is a per-query weight for head $j$, also projected from $h_t$; it can be negative, so a head can vote *against* a token;
- $\mathrm{ReLU}$ replaces the softmax: there is no normalization over $s$, so each score is computed on its own. The paper chooses ReLU for throughput.

$I_{t,s}$ is a ranking score, not a probability. Only its order matters for selection.

### 2.3 Top-k selection and the sparse output

Let $\mathcal{S}_t = \{\,s \le t : I_{t,s} \in \text{Top-}k(I_{t,:})\,\}$, the $k$ highest-scoring past positions ($\mathcal{S}_t$ is all of $\{1..t\}$ while $t \le k$). The main attention then runs only over that set (eq. 2):

$$
o_t = \sum_{s \in \mathcal{S}_t} \frac{\exp(q_t \cdot k_s/\sqrt{d_{qk}})}{\sum_{s' \in \mathcal{S}_t}\exp(q_t\cdot k_{s'}/\sqrt{d_{qk}})}\; v_s .
$$

This is 05.2's attention with an extra mask: $-\infty$ on every $s \notin \mathcal{S}_t$. One selection is made per query token and shared by all $H$ main heads, which matches how the keys are stored: DeepSeek-V3.2 runs MLA in its **MQA mode**, one shared latent key-value vector per token (the $g = 1$ case of 12.3). The main attention now scores $\sum_t \min(t, k) \approx Tk$ pairs:

$$
\text{MACs}_{\text{DSA prefill}} = \underbrace{\tfrac{T(T+1)}{2}\,H^I d^I}_{\text{indexer, still } O(T^2)} + \underbrace{\textstyle\sum_t \min(t,k)\; H\,(d_{qk}+d_v)}_{\text{main attention, } O(Tk)} .
$$

The indexer is still quadratic. DSA wins because $H^I d^I$ is much smaller than $H(d_{qk}+d_v)$ and runs in FP8.

### 2.4 Training the indexer: imitate the attention you replace

Top-k is a hard selection with no useful gradient, so the language-modeling loss cannot teach the indexer which tokens matter. DeepSeek trains it by distillation from the main attention. For query $t$, sum the main attention's probabilities over all heads and L1-normalize along $s$:

$$
p_{t,s} = \frac{\sum_{h=1}^{H} a^{(h)}_{t,s}}{\sum_{s'}\sum_{h} a^{(h)}_{t,s'}},
$$

where $a^{(h)}_{t,s}$ is head $h$'s softmax weight from $t$ to $s$. Then minimize

$$
\mathcal{L}^I = \sum_t D_{\mathrm{KL}}\!\left(p_{t,:}\;\big\|\;\mathrm{Softmax}(I_{t,:})\right).
$$

The softmax here only turns the scores into a distribution for the loss; selection still uses the raw scores. DeepSeek did this in two stages, starting from the dense DeepSeek-V3.1-Terminus checkpoint (PUBLICLY DOCUMENTED, arXiv 2512.02556 §2.1):

1. **Dense warm-up.** Attention stays dense, every parameter except the indexer is frozen, and only $\mathcal{L}^I$ trains (learning rate $10^{-3}$, 1,000 steps of 16 × 128K-token sequences, 2.1B tokens).
2. **Sparse training.** Selection is switched on and every parameter trains. The KL is taken only over the selected set, $D_{\mathrm{KL}}(p_{t,\mathcal{S}_t} \,\|\, \mathrm{Softmax}(I_{t,\mathcal{S}_t}))$, and the **indexer's input is detached**: the indexer learns only from $\mathcal{L}^I$, the main model only from the language-modeling loss. $k = 2048$, learning rate $7.3\times 10^{-6}$, 15,000 steps of 480 × 128K-token sequences, 943.7B tokens.

The warm-up matters because a random indexer selects random tokens: if you switched sparsity on immediately, the main model would be trained on garbage contexts while the indexer was still learning.

## 3. Numerical example by hand

Four tokens. We look at the last query, $t = 4$. Indexer: $H^I = 2$ heads, $d^I = 2$.

| $s$ | $k^I_s$ | head 1: $q^I_{4,1}=(1,0)$ | head 2: $q^I_{4,2} = (0,2)$ |
|---|---|---|---|
| 1 | $(2, 0)$ | $2$ | $0$ |
| 2 | $(-1, 1)$ | $-1 \to \mathrm{ReLU} = 0$ | $2$ |
| 3 | $(0.5, 0)$ | $0.5$ | $0$ |
| 4 | $(1, 1)$ | $1$ | $2$ |

With weights $w^I_4 = (1, 0.5)$:

$$
I_{4,:} = 1\cdot(2, 0, 0.5, 1) + 0.5\cdot(0, 2, 0, 2) = (2,\ 1,\ 0.5,\ 2).
$$

With $k = 2$, the selection is $\mathcal{S}_4 = \{1, 4\}$. Head 2 alone would have kept token 2; head 1's weight outvotes it.

**Main attention.** Suppose the main attention's scaled logits for $t = 4$ are $(\ln 4, 0, 0, \ln 4)$ and the values are $v = (4, 8, 0, 2)$ (scalars, to keep it short).

- Dense: $\exp(\text{logits}) = (4, 1, 1, 4)$, so $a = (0.4, 0.1, 0.1, 0.4)$ and $o_4 = 0.4\cdot 4 + 0.1\cdot 8 + 0.1 \cdot 0 + 0.4\cdot 2 = 3.2$.
- Sparse over $\{1, 4\}$: $(4, 4)$ renormalizes to $(0.5, 0.5)$, so $o_4 = 0.5 \cdot 4 + 0.5\cdot 2 = 3.0$.

The selected tokens held 80% of the dense attention mass, and the output moved from 3.2 to 3.0. The error comes from the 20% that was dropped. With $k = 2048$ out of 128K, the question is the same: how much mass falls outside the selection.

**Indexer loss.** With one head, $p_{4,:} = a = (0.4, 0.1, 0.1, 0.4)$. $\mathrm{Softmax}(I_{4,:}) = (e^2, e^1, e^{0.5}, e^2)/19.145 = (0.386, 0.142, 0.086, 0.386)$. Then

$$
D_{\mathrm{KL}} = 2\cdot 0.4\ln\tfrac{0.4}{0.386} + 0.1\ln\tfrac{0.1}{0.142} + 0.1\ln\tfrac{0.1}{0.086} = 0.0286 - 0.0351 + 0.0150 = 0.0085 .
$$

Small, because the indexer already ranks the tokens the way the attention does. Gradient descent on it would push $I_{4,2}$ down (the indexer gives token 2 more probability than the attention) and $I_{4,3}$ up.

These numbers are asserted in `test_sparse_attention.py` (`test_hand_worked_*`).

## 4. Tensor shapes, dtype, device

$B$ batch, $T$ tokens, $C$ model width, $H$ main heads of width $d_h = C/H$, $g$ KV heads, $H^I$ indexer heads of width $d^I$.

| Tensor | Shape | Notes |
|---|---|---|
| input `x` | $(B, T, C)$ | fp32 here; bf16 in training |
| indexer queries `q_idx` | $(B, T, H^I, d^I)$ | from `idx_q`, `nn.Linear(C, H^I d^I)` |
| indexer keys `k_idx` | $(B, T, d^I)$ | one per token, shared by indexer heads; **cached** at decode (FP8 in production) |
| head weights `w` | $(B, T, H^I)$ | from `idx_w`, `nn.Linear(C, H^I)` |
| index scores `I` | $(B, T, T)$ | $-\infty$ above the diagonal; the reference builds it in full, kernels never do |
| selection | $(B, T, T)$ bool, or indices $(B, T, k)$ | row $t$ has $\min(t+1, k)$ entries; shared by all heads |
| main `q` | $(B, H, T, d_h)$ | as in 12.3 |
| main `k`, `v` | $(B, g, T, d_h)$ → repeated to $(B, H, T, d_h)$ | $g = 1$ (MQA) mirrors DSA on MLA |
| output | $(B, T, C)$ | |

All on the activations' device. The indexer input is `x.detach()`.

## 5. From-scratch PyTorch

The module is `code/src/llmre/attention/sparse.py`. The indexer score is one `einsum`, a ReLU and a weighted sum over heads:

```python
def lightning_indexer_scores(q_idx, k_idx, w):
    T = q_idx.shape[1]
    dots = torch.einsum("bthd,bsd->bths", q_idx, k_idx)  # (B, T, H_I, T): every head, every pair
    scores = (w[..., None] * F.relu(dots)).sum(dim=2)    # (B, T, T): weighted vote over heads
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q_idx.device), diagonal=1)
    return scores.masked_fill(future, float("-inf"))     # never select the future
```

Selection takes the top $k$ of each row and keeps only causal entries (in early rows, where $t + 1 < k$, `topk` has to pick some $-\infty$ future slots, which are dropped):

```python
def topk_selection_mask(index_scores, topk):
    B, T, _ = index_scores.shape
    idx = index_scores.topk(min(topk, T), dim=-1).indices      # (B, T, k)
    mask = torch.zeros(B, T, T, dtype=torch.bool, device=index_scores.device)
    mask.scatter_(-1, idx, True)
    return mask & torch.tril(torch.ones(T, T, dtype=torch.bool, device=index_scores.device))
```

The layer `DSAttention(cfg, n_kv_head=1, n_index_head, d_index, topk)` has 12.3's projections (`q_proj`, `k_proj`, `v_proj`, `c_proj`) plus the indexer's (`idx_q`, `idx_k`, `idx_w`). Its forward pass:

```python
q, k, v = self._qkv(x)                                    # 12.3's GQA layout
scores = self.index_scores(x.detach())                    # KL gradients stop here
select = topk_selection_mask(scores, T if dense else self.topk)
mask = torch.zeros(select.shape, dtype=x.dtype, device=x.device).masked_fill(~select, float("-inf"))
probs = ((q @ k.transpose(-2, -1)) / self.d_h**0.5 + mask[:, None]).softmax(dim=-1)
y = self.c_proj((probs @ v).transpose(1, 2).contiguous().view(B, T, C))
```

With `return_aux=True` it also returns the target $p$ (head-summed `probs`, detached) and `indexer_loss`: eq. 3 in dense mode, eq. 4 (KL over the selected set) in sparse mode. Gradients separate as DeepSeek describes. The indexer reads `x.detach()`, so the KL cannot reach the main model, and the selection is a boolean mask, so the LM loss cannot reach the indexer.

The reference builds the full $(T, T)$ score matrix and masks it. `sparse_attention_gather` computes the same function the way a kernel does: gather each query's $k$ selected keys into a $(k, d)$ block and attend over that block only.

**Unit tests** (`code/tests/test_sparse_attention.py`): the hand example (scores, selection, dense 3.2 vs sparse 3.0, KL 0.0085); every row selects exactly $\min(t+1, k)$ causal positions; the gather version equals the mask reference; with $k \ge T$ the layer equals `GroupedQueryAttention` with the same weights, in both modes; the layer is causal; the LM loss gives no gradient to the indexer and the indexer loss gives none to the main weights or the input; the KL is zero when $\mathrm{softmax}(I) = p$; a dense warm-up halves the indexer loss and raises recall; and the cost arithmetic of section 6.

## 6. Under the hood: what production does, and what it costs

### 6.1 DeepSeek-V3.2, worked from its config

From the published `config.json` (PUBLICLY DOCUMENTED): 61 layers, 128 query heads, MLA with a 512-dim latent plus 64 RoPE dims, `index_n_heads` 64, `index_head_dim` 128, `index_topk` 2048. In MLA's MQA mode each head scores a $512 + 64 = 576$-dim key and sums a 512-dim latent value, so per $(t, s)$ pair and per layer (our arithmetic):

| | MACs per pair | precision |
|---|---|---|
| main attention (MLA, MQA mode) | $128 \times (576 + 512) = 139{,}264$ | bf16 |
| lightning indexer | $64 \times 128 = 8{,}192$ | FP8 |

The indexer is 17× cheaper per pair before counting FP8's higher throughput. Per layer, one sequence (our arithmetic, `selected_pairs` and `per_pair_macs` in the module):

| $T$ | dense prefill | DSA prefill | ratio | dense decode step | DSA decode step | ratio |
|---|---|---|---|---|---|---|
| 4,096 | $1.17\times10^{12}$ | $9.45\times10^{11}$ | 1.2× | $5.7\times10^8$ | $3.2\times10^8$ | 1.8× |
| 32,768 | $7.48\times10^{13}$ | $1.35\times10^{13}$ | 5.6× | $4.6\times10^9$ | $5.5\times10^8$ | 8.2× |
| 131,072 | $1.20\times10^{15}$ | $1.07\times10^{14}$ | 11.1× | $1.83\times10^{10}$ | $1.36\times10^9$ | 13.4× |

Three things to read from it:

- **The saving appears only at long context.** At 4K a decode step saves 1.8×. Below about 2,200 tokens DSA costs more than dense attention (exercise below). DeepSeek's paper says that for short prefills it runs a masked dense ("MHA mode") kernel that simulates DSA, because that is faster there.
- **The indexer becomes the bottleneck.** At 128K the indexer is 65% of DSA's prefill attention MACs: it is the only part that is still quadratic. This is why most follow-up work (for example, reusing one layer's selection in the next few layers) targets the indexer.
- **Storage does not shrink.** The full KV cache is still kept, because a later query may select any token. DSA *adds* the indexer key cache: 128 FP8 bytes per token per layer next to MLA's 576 bf16 numbers (1,152 bytes), about 11% more. What shrinks is the bytes *read* per decode step. At 128K, dense reads 151 MB per layer, while DSA reads 16.8 MB of indexer keys plus $2048 \times 1152$ bytes = 2.4 MB of selected entries, 7.9× less. By 08.6's roofline, decode time follows the bytes read.

<div class="callout pt"><p><strong>PUBLICLY DOCUMENTED.</strong> DeepSeek ships three kernel implementations with the V3.2-Exp release: TileLang reference kernels, DeepGEMM kernels for the indexer logits (including a paged version for decode), and FlashMLA sparse-attention kernels that gather only the selected latent entries. vLLM had day-0 support (its recipe notes DeepGEMM for the "MQA logits" and a custom FP8 KV cache) and SGLang ships DSA images. The kernels never build the $(T, T)$ matrix our reference builds. The indexer kernel streams $k^I$ and keeps a running top-k, and the attention kernel gathers the $k$ rows, like <code>sparse_attention_gather</code>.</p></div>

### 6.2 Who uses it

| Model | DSA configuration (from model config/card) |
|---|---|
| DeepSeek-V3.2 (Dec 2025) | every layer; 64 indexer heads × 128, top-k 2048 |
| GLM-5 (Feb 2026, Zhipu / Z.ai) | every layer (`GlmMoeDsaForCausalLM`); 32 indexer heads × 128, top-k 2048 |
| Kimi K3.1 Pro (Nov 2026, Moonshot) | 24 "Gated DSA" layers among 69 KDA (linear) layers; 32 index heads × 128, top-k 2048 |

The Kimi K3.1 Pro row is the hybrid 12.5 described: about 3:1 linear-to-attention again, with the attention layers made sparse. That attacks both costs: the linear layers carry no growing cache, and the sparse layers read only 2048 entries of theirs. What exactly "Gated" adds is not specified on the card. Treat it as an output gate in the style of 12.5's gated attention (INFERENCE).

<div class="callout warn"><p><strong>Claim labels.</strong> <em>PUBLICLY DOCUMENTED:</em> the DSA equations, the two-stage training recipe and its hyperparameters (DeepSeek-V3.2 paper); the indexer sizes and top-k in the three model configs/cards; the kernels and the vLLM/SGLang support. <em>(company claim):</em> DeepSeek reports that V3.2-Exp performs on par with the dense V3.1-Terminus across its benchmarks, with much lower long-context serving cost; GLM-5's report says DSA "significantly reduce[s] training and inference costs while maintaining long-context fidelity". Neither claim has been independently reproduced at matched training. <em>Our arithmetic:</em> all MAC and byte counts in 6.1. <em>INFERENCE:</em> that indexer + top-k will remain the dominant sparse-attention design; it is the common base today, but block-sparse and threshold-based variants are active research.</p></div>

## 7. Side by side, and when to keep dense attention

| | Dense GQA/MLA (12.3) | Linear / hybrid (12.5) | DSA |
|---|---|---|---|
| What each query attends to | every past token, exactly | a fixed-size summary (+ full layers) | $k$ selected past tokens, exactly |
| Prefill attention cost | $O(T^2)$ | $O(T)$ | $O(T^2)$ cheap indexer + $O(Tk)$ |
| Decode cost per token | $O(T)$ reads | $O(1)$ (+ full layers) | $O(T)$ small reads + $O(k)$ |
| KV storage | grows with $T$ | constant (+ full layers) | grows with $T$, +~11% indexer keys |
| Exact recall of a far token | yes | lossy | yes, *if the indexer selects it* |
| Training | standard | standard | indexer distillation, two stages |
| Used by | most models | Qwen3-Next, Kimi Linear | DeepSeek-V3.2, GLM-5, Kimi K3.1 Pro |

**When you would still use dense attention:** at short context, where the indexer is overhead (below ~2K tokens for DeepSeek's sizes); when you train from scratch at small scale and cannot afford a second training stage and its kernels; and whenever you need a reference to measure a sparse model against. Dense attention is the definition: DSA is judged by how close its output stays to it. The failure mode to watch is a query whose answer lives in a token the indexer did not select. That error is silent: nothing in the output tells you that the needed token was left out.

## Experiment: warm up an indexer

`code/experiments/dsa_indexer_warmup.py` freezes a `DSAttention` layer with random main weights (queries scaled ×6 so its attention is peaked, like a trained layer), feeds it 64-token sequences from a 16-token vocabulary, and trains only the indexer with the dense-warm-up loss. On held-out sequences it prints the KL, recall@8 (the fraction of the attention's own top-8 keys that the indexer also selects), the relative error of the top-8 sparse output against the dense output, and the same error for an *oracle* that selects the attention's true top-8.

```bash
cd code && PYTHONPATH=src python experiments/dsa_indexer_warmup.py
```

Output on CPU (PyTorch 2.15; digits may differ by version):

| step | KL | recall@8 | sparse rel. error | oracle rel. error |
|---|---|---|---|---|
| 0 | 49.98 | 0.31 | 0.979 | 0.403 |
| 100 | 5.24 | 0.84 | 0.453 | 0.403 |
| 200 | 5.17 | 0.85 | 0.449 | 0.403 |
| 300 | 5.09 | 0.86 | 0.451 | 0.403 |

**Expected observations.** The untrained indexer selects close to at random (recall 0.31) and the sparse output is about as far from the dense one as the output itself is large (error ≈ 1). A hundred steps of KL training bring recall to 0.84 and the error to 0.45, close to the oracle floor of 0.40. The floor is not zero: with only 8 of up to 64 keys, this attention keeps meaningful mass outside any 8-token set. That is what DeepSeek's second stage addresses. Once selection is on and the whole model trains, the main attention learns to concentrate its mass in what it is given. The KL levels off well above zero: a 4-head, 8-dim ReLU scorer cannot reproduce the exact shape of the attention distribution, only its ranking. For selection, only the ranking matters. Try `K = 16` and a `q_proj` scale of 3: both floors move, and you can see how sparsity and attention sharpness trade off.

## Common mistakes

- **Selecting before masking the future.** If the indexer scores are not set to $-\infty$ above the diagonal before `topk`, a query can select later tokens, and training loss looks excellent because the model sees the answer.
- **Letting the LM loss train the indexer, or the KL train the model.** Without the detach, the KL pulls the shared hidden states toward whatever the indexer can predict easily. The recipe keeps the two losses on separate parameters.
- **Expecting DSA to shrink the KV cache.** It reduces compute and bytes read, not bytes stored. Memory capacity planning (08.6) still uses the full cache, plus the indexer keys.
- **Normalizing the index scores before selection.** The softmax over $I_{t,:}$ exists only inside the loss. Selection uses raw scores, and since top-k only needs the order, a softmax there is wasted work.
- **Selecting per head.** In DSA one selection per query token is shared by all heads, which is what lets the kernel gather one set of latent rows. Per-head selection is a different design with different kernels and different memory traffic.
- **Comparing at short context.** A benchmark at 2K tokens shows DSA slower than dense. The design targets 32K and beyond.

## Exercise: where does DSA start to pay?

With DeepSeek-V3.2's per-pair costs from 6.1 (main 139,264 MACs, indexer 8,192 MACs) and $k = 2048$, at what context length $T$ does one DSA decode step cost the same number of multiply-adds as a dense one? (Ignore FP8's speed advantage.)

<details><summary>Hint</summary>

A dense step scores $T$ pairs with the main attention. A DSA step scores $T$ pairs with the indexer and $k$ pairs with the main attention. Set the two equal.

</details>

<details><summary>Stronger hint</summary>

$139{,}264\,T = 8{,}192\,T + 2048 \times 139{,}264$. Note that $139{,}264 - 8{,}192$ is a power of two.

</details>

<details><summary>Solution</summary>

$T\,(139{,}264 - 8{,}192) = 131{,}072\,T = 2048 \times 139{,}264$, so $T = 2048 \times 139{,}264 / 131{,}072 = 2{,}176$ tokens. Below that, DSA does more work than dense attention (and for $T \le 2048$ it selects everything anyway, so the indexer is pure overhead). With FP8 counted as roughly 2× bf16 throughput, the break-even moves a little lower, but the conclusion holds. This is why short prefills run a dense kernel.

</details>

## Implementation exercise: a decode step with an indexer cache (code this after reading)

Write `dsa_decode_step(layer, x_t, cache)` in a new function next to `DSAttention` (do not change the class). `x_t` is one new token $(B, 1, C)$. `cache` holds the main keys and values $(B, g, T, d_h)$ and the indexer keys $(B, T, d^I)$ of all previous tokens. Append the new token's entries, score all $T+1$ positions with the indexer, select the top $k$, gather only those $k$ keys and values, and attend. Test that feeding a sequence token by token reproduces `layer(x)` row by row (like 12.5's decode test). Then count, for your toy sizes, how many cached numbers each step reads compared with the dense step.

## Debugging exercise

A colleague converts a dense model to DSA. They skip the dense warm-up and start sparse training directly with $k = 2048$. In the first few hundred steps the training loss jumps, then slowly recovers. Their long-context retrieval score ends well below the dense model's, while short-context benchmarks are unchanged. Their indexer code is

```python
scores = self.index_scores(x)                        # x from the residual stream
select = topk_selection_mask(scores, self.topk)
loss = lm_loss + indexer_kl_loss(target, scores, select)
```

Name the two problems.

<details><summary>Answer</summary>

(1) No warm-up: at the start the indexer is random, so for every query past 2048 tokens the main attention sees an essentially random subset of its context. The loss jumps, and the model is trained for hundreds of steps on wrong contexts. It partly adapts to that and partly loses the long-range lookups the dense model had. Short-context benchmarks are unchanged because for $T \le 2048$ everything is selected anyway. Train the indexer alone first, with everything else frozen and attention dense. (2) `x` is not detached, so the KL gradient flows through the indexer projections into the residual stream and trains the main model to make the indexer's job easy instead of to predict tokens. Use `self.index_scores(x.detach())`, as in the recipe, so each loss trains only its own parameters.

</details>

## What to read / what to skip

In the DeepSeek-V3.2 paper, read §2 (architecture: DSA, the two training stages, the inference-cost figure) and Appendix A (the MHA and MQA modes of MLA, if you want to know why one selection serves all heads). Skim the RL and agent-data sections. They are a different topic (Modules 16–18). In the GLM-5 report, read the architecture section for a second team's use of DSA at 744B parameters, and treat the benchmark tables as company claims. For kernels, the TileLang example (`tile-ai/tilelang`, `examples/deepseek_v32`) is the most readable implementation after ours.

## Research connection

<div class="callout paper"><p><strong>DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models</strong>, DeepSeek-AI, December 2025, <a href="https://arxiv.org/abs/2512.02556">arXiv 2512.02556</a>. Introduces DSA: eqs. 1–4 of this lesson, the dense warm-up and sparse training stages, and the cost comparison against the dense V3.1-Terminus. Code and kernels: <a href="https://github.com/deepseek-ai/DeepSeek-V3.2-Exp">DeepSeek-V3.2-Exp</a> (inference demo, links to the <a href="https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32">TileLang kernels</a>, <a href="https://github.com/deepseek-ai/DeepGEMM/pull/200">DeepGEMM indexer kernels</a> and <a href="https://github.com/deepseek-ai/FlashMLA/pull/98">FlashMLA sparse kernels</a>); <a href="https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html">vLLM recipe</a>; <a href="https://huggingface.co/deepseek-ai/DeepSeek-V3.2">model card and config</a>. Second adopter: <strong>GLM-5: from Vibe Coding to Agentic Engineering</strong>, GLM-5 Team, February 2026, <a href="https://arxiv.org/abs/2602.15763">arXiv 2602.15763</a> (<a href="https://huggingface.co/zai-org/GLM-5">model card</a>). Linear + sparse hybrid: <a href="https://huggingface.co/moonshotai/Kimi-K3.1-Pro">Kimi K3.1 Pro model card</a>. Background on the attention DSA is built on: DeepSeek-V2 (MLA, <a href="https://arxiv.org/abs/2405.04434">arXiv 2405.04434</a>).</p></div>

## Check yourself

<details><summary>DSA's indexer is still $O(T^2)$. Why is the layer cheaper at all?</summary>

Each indexer pair costs $H^I d^I$ multiply-adds (8,192 for DeepSeek-V3.2), in FP8, against $H(d_{qk}+d_v)$ (139,264) in bf16 for the main attention. The main attention, now $O(Tk)$, does the expensive work on 2048 tokens only. At 128K the total attention prefill drops about 11× even though one part is still quadratic.

</details>

<details><summary>Why can't the language-modeling loss train the indexer directly?</summary>

The indexer's output only affects the loss through top-k selection, which is piecewise constant: a small change in a score leaves the selected set unchanged, so the gradient is zero almost everywhere. DSA gives the indexer its own dense signal instead: the KL to the main attention's head-summed distribution.

</details>

<details><summary>Why sum the main attention over heads before using it as the target?</summary>

DSA makes one selection per query token, shared by all heads. The set must serve every head at once, so the target is the total attention mass each past token receives across heads. A token that one head relies on heavily still gets a large share.

</details>

<details><summary>In the hand example, which token would head 2 alone have kept, and why was it dropped?</summary>

Token 2 (head 2 scores it 2, head 1 scores it $\mathrm{ReLU}(-1) = 0$). With $w = (1, 0.5)$ its total is 1, below tokens 1 and 4 (both 2), so with $k = 2$ it is not selected.

</details>

<details><summary>Does DeepSeek-V3.2 at 128K need less GPU memory for its KV cache than V3.1-Terminus?</summary>

No. Every token's latent KV entry is still stored, because any later query may select it, and the indexer adds a 128-byte FP8 key per token per layer (about 11% more). What drops is the computation and the bytes read per decode step (about 13× and 8× per layer at 128K by our arithmetic).

</details>

<details><summary>Kimi K3.1 Pro combines KDA linear layers with DSA layers. What does each part remove?</summary>

The linear (KDA) layers have a constant-size state, so they remove the growing cache and the per-step reads for three quarters of the layers. The DSA layers keep exact lookup, but each query attends to only 2048 selected tokens, so their per-step cost grows only through the cheap indexer. Together they attack both the memory and the compute side of long context.

</details>

## Next

Module 12 now covers both ways to make long context affordable: fixed-size memories (12.5) and sparse selection (12.6). Module 13 turns to measuring these models. Continue to [13.1 · Loss, perplexity, zero-/few-shot](lessons/module-13/lesson-01.md).
