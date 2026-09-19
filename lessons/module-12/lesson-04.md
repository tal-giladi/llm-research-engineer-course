# 12.4 · Mixture of Experts (DeepSeek-style)

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="../module-05/lesson-04.md">05.4 · MLP, residual, LayerNorm, the block</a> (the dense feed-forward network a MoE replaces); <a href="../module-09/lesson-03.md">09.3 · Tensor / pipeline / expert parallelism</a> (how experts are sharded across devices); SwiGLU from <a href="lesson-02.md">12.2</a> (each expert is a small FFN); softmax from <a href="../module-01/lesson-03.md">01.3</a>.</p>
<p><strong>You will learn:</strong> how a Mixture of Experts replaces the one dense FFN with $N$ experts and a router that sends each token to its top-$k$ experts, decoupling parameter count from per-token compute; why routers collapse without <strong>load balancing</strong> and how the auxiliary loss (and DeepSeek-V3's aux-loss-free bias trick) prevent it; <strong>shared experts</strong>; a routing example worked by hand; and a small MoE layer implemented from scratch.</p>
<p><strong>Why this matters for ML:</strong> MoE is how frontier models get very large parameter counts at manageable inference cost — DeepSeek-V3 has 671B parameters but activates only ~37B per token. Mixtral, DeepSeek, Qwen-MoE, and others are all sparse. Understanding routing and load balancing is core to modern architecture.</p>
</div>

## 1. Intuition: more parameters, same compute per token

In a dense transformer, every token flows through the *same* feed-forward network, so parameters and per-token FLOPs rise together — a bigger FFN costs more for *every* token. A **Mixture of Experts** (MoE) breaks that coupling. Replace the single FFN with $N$ separate expert FFNs and a small **router** (a gating network). For each token the router picks only the **top-$k$** experts (often $k = 1$ or $2$ out of dozens), and the token is processed by just those.

The payoff: total parameters scale with $N$ (you can have hundreds of experts), but each token only ever touches $k$ of them, so the FLOPs *per token* stay close to a single dense FFN. You buy capacity (more parameters = more knowledge the model can store) without buying proportional compute. This is **sparse activation** — most of the network is idle for any given token.

The intuition for why it works: different experts specialize. One may handle code, another prose, another arithmetic. The router learns to send each token to the experts most useful for it. A dense FFN has to be a jack-of-all-trades; experts can be specialists.

<div class="callout key"><p>Dense FFN: every token, one network, params and compute coupled. MoE: $N$ expert FFNs + a router; each token uses only its top-$k$ experts. Parameters scale with $N$; FLOPs/token stay ~constant. Sparse activation trades memory for capacity.</p></div>

## 2. Mathematics: routing

Let $\mathbf{x}\in\mathbb{R}^C$ be one token's vector, $E_1,\dots,E_N$ the expert FFNs, and $W_r\in\mathbb{R}^{N\times C}$ the router weight. The router scores every expert and softmaxes:

$$
\mathbf{p} = \operatorname{softmax}(W_r\mathbf{x}) \in \mathbb{R}^N,
$$

so $p_i$ is the router's affinity for expert $i$. Select the top-$k$ experts $\mathcal{T} = \operatorname{top\text{-}k}(\mathbf{p})$, renormalize their probabilities to gate weights $g_i = p_i / \sum_{j\in\mathcal{T}} p_j$, and the output is the gate-weighted sum of just those experts:

$$
\mathbf{y} = \sum_{i\in\mathcal{T}} g_i\, E_i(\mathbf{x}).
$$

Experts not in $\mathcal{T}$ are never evaluated for this token — that is where the compute saving lives. With $k = 1$ (Switch Transformers) each token hits exactly one expert; with $k = 2$ (Mixtral, DeepSeek) two, blended by their gates.

## 3. The load-balancing problem

The router is learned, and left to itself it tends to **collapse**: a few experts get chosen slightly more often early on, so they get more gradient signal, so they improve faster, so they get chosen even more — a rich-get-richer feedback loop. The endpoint is a handful of overworked experts and a majority that are dead weight (never selected, never trained). That wastes the parameters MoE was supposed to buy, and it wrecks the load balance across devices when experts are sharded (Module 9.3): one GPU is swamped while others idle.

The fix is to *encourage the router to spread tokens evenly*. Switch Transformers adds an **auxiliary load-balance loss**. Let $f_i$ be the fraction of tokens routed to expert $i$ (a hard count) and $P_i$ the mean router probability assigned to expert $i$ (a soft average). The aux loss is

$$
\mathcal{L}_{\text{aux}} = N \sum_{i=1}^{N} f_i\, P_i.
$$

It is minimized when both $f_i$ and $P_i$ are uniform ($= 1/N$), giving $\mathcal{L}_{\text{aux}} = N\cdot N\cdot\tfrac{1}{N}\cdot\tfrac{1}{N} = 1$; it is maximized ($= N$) when everything piles onto one expert. Gradients flow through $P_i$ (the differentiable part), nudging the router to raise probability on under-used experts. The total training loss is $\mathcal{L}_{\text{LM}} + \alpha\,\mathcal{L}_{\text{aux}}$ with a small $\alpha$.

<div class="callout key"><p>$\mathcal{L}_{\text{aux}} = N\sum_i f_i P_i$: $f_i$ = fraction of tokens sent to expert $i$ (count), $P_i$ = mean routing probability to expert $i$. Minimum $1$ when perfectly balanced, maximum $N$ when fully collapsed. It penalizes putting many tokens (high $f_i$) on experts the router is already confident about (high $P_i$).</p></div>

### 3.1 DeepSeek-V3's aux-loss-free balancing (PUBLICLY DOCUMENTED)

The aux loss has a cost: its gradient competes with the language-modeling loss and can slightly hurt quality. DeepSeek-V3 (technical report, 2024) documents an **auxiliary-loss-free** alternative: add a per-expert **bias** $b_i$ to the router *scores* used for top-$k$ selection (but not to the gate weights). After each step, nudge $b_i$ down for over-loaded experts and up for under-loaded ones. Balancing then happens through the selection bias, not through a loss-gradient term, so it does not pull on the LM objective. This is a PUBLICLY DOCUMENTED technique from the DeepSeek-V3 report; the classic aux-loss form is what we implement in `code/`.

## 4. Shared experts (DeepSeekMoE)

DeepSeekMoE (Dai et al. 2024) adds **shared experts**: one or more FFNs that are *always* active for every token, in addition to the routed top-$k$. The reasoning: some knowledge is common to almost all tokens (basic syntax, frequent patterns). If every routed expert has to relearn that common knowledge, capacity is wasted on redundancy. Dedicating always-on shared experts to the common case frees the routed experts to specialize on the rest. The token output becomes

$$
\mathbf{y} = \underbrace{\sum_{s} E^{\text{shared}}_s(\mathbf{x})}_{\text{always on}} \;+\; \underbrace{\sum_{i\in\mathcal{T}} g_i\, E_i(\mathbf{x})}_{\text{routed top-}k}.
$$

DeepSeekMoE also uses **fine-grained** experts — many small experts instead of a few big ones — giving the router more combinations to choose from at the same total parameter count.

## 5. Numerical example: routing one token

Take $N = 4$ experts, $k = 2$, and a token whose router logits are $W_r\mathbf{x} = (2.0,\ 1.0,\ 0.1,\ -1.0)$. Softmax them (verified in §7):

$$
\mathbf{p} = (0.6381,\ 0.2347,\ 0.0954,\ 0.0318).
$$

Top-2 are experts $0$ and $1$ with $p_0 = 0.6381$, $p_1 = 0.2347$. Renormalize over the chosen pair (sum $= 0.8728$):

$$
g_0 = \frac{0.6381}{0.8728} = 0.7311, \qquad g_1 = \frac{0.2347}{0.8728} = 0.2689.
$$

So this token's output is $\mathbf{y} = 0.7311\,E_0(\mathbf{x}) + 0.2689\,E_1(\mathbf{x})$ — experts 2 and 3 are never evaluated. The token used exactly $k = 2$ experts, weighted $73\%$/$27\%$.

**Aux loss at the extremes.** With $N = 4$: a perfectly balanced batch ($f_i = P_i = 0.25$ for all $i$) gives $\mathcal{L}_{\text{aux}} = 4\sum_i 0.25\cdot 0.25 = 4\cdot 4\cdot 0.0625 = 1.0$. A fully collapsed batch ($f_0 = P_0 = 1$, rest $0$) gives $\mathcal{L}_{\text{aux}} = 4\cdot(1\cdot 1) = 4.0 = N$. Both verified in §7.

## 6. Tensor shapes &amp; from-scratch PyTorch

Flatten the batch to a list of tokens, route, dispatch, scatter back:

$$
(B, T, C) \to \underbrace{(N_{\text{tok}}, C)}_{N_{\text{tok}} = BT} \xrightarrow{\text{router}} \underbrace{(N_{\text{tok}}, N)}_{\text{probs}} \xrightarrow{\text{top-}k} \text{indices, gates} \to \text{sum of experts} \to (B, T, C).
$$

From [`llmre.model.moe`](../../code/src/llmre/model/moe.py) (each expert is a `SwiGLU` from lesson 12.2):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from llmre.model.swiglu import SwiGLU

class MoE(nn.Module):
    def __init__(self, dim, n_experts, top_k, hidden_dim=None, n_shared=0, bias=False):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router  = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim, bias) for _ in range(n_experts)])
        self.shared  = nn.ModuleList([SwiGLU(dim, hidden_dim, bias)
                                      for _ in range(n_shared)]) if n_shared else None

    def route(self, x_flat):
        probs = F.softmax(self.router(x_flat), dim=-1)          # (N_tok, N)
        topk_gate, topk_idx = probs.topk(self.top_k, dim=-1)    # (N_tok, k)
        topk_gate = topk_gate / topk_gate.sum(dim=-1, keepdim=True)   # renormalize
        return topk_idx, topk_gate, probs

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)                               # (N_tok, C)
        topk_idx, topk_gate, probs = self.route(x_flat)

        out = torch.zeros_like(x_flat)
        for e, expert in enumerate(self.experts):              # sparse dispatch
            sel = (topk_idx == e)                              # (N_tok, k) bool
            token_mask = sel.any(dim=-1)                       # tokens that chose expert e
            if not bool(token_mask.any()):
                continue
            gate = (topk_gate * sel).sum(dim=-1)               # gate weight for expert e
            ye = expert(x_flat[token_mask])                    # run expert on its tokens only
            out[token_mask] += gate[token_mask].unsqueeze(-1) * ye
        if self.shared is not None:                            # always-on shared experts
            for sh in self.shared:
                out = out + sh(x_flat)

        # auxiliary load-balance loss (Switch Transformers)
        assign = F.one_hot(topk_idx, self.n_experts).float().sum(dim=1)  # (N_tok, N)
        f = assign.mean(dim=0)                                 # tokens-per-expert fraction
        P = probs.mean(dim=0)                                  # mean prob per expert
        aux = self.n_experts * torch.sum(f.detach() * P)
        return out.view(B, T, C), aux
```

The per-expert loop gathers each expert's tokens with a boolean mask, runs the expert **once** on that sub-batch, and scatters the gate-weighted result back — so an expert with no tokens does no work, and every token is processed by exactly its $k$ experts. `f.detach()` freezes the hard count so gradients flow only through the soft $P$, matching the aux-loss derivation.

## 7. Under the hood: verify the routing and the aux loss

```python
import torch, torch.nn.functional as F

logits = torch.tensor([2., 1., 0.1, -1.])
probs  = torch.softmax(logits, 0)
print([round(p, 4) for p in probs.tolist()])       # [0.6381, 0.2347, 0.0954, 0.0318]
g, idx = probs.topk(2)
print(idx.tolist(), [round(x, 4) for x in (g / g.sum()).tolist()])
# [0, 1] [0.7311, 0.2689]

N = 4
f = P = torch.tensor([.25, .25, .25, .25])          # balanced
print(float(N * (f * P).sum()))                      # 1.0
f = P = torch.tensor([1., 0., 0., 0.])              # collapsed
print(float(N * (f * P).sum()))                      # 4.0
```

**Cost.** With $N$ experts and top-$k$ routing, per-token compute is $\approx k$ dense-FFN passes (plus the tiny router matmul), *independent of $N$*. But *memory* holds all $N$ experts' parameters, and at scale they are sharded across devices (**expert parallelism**, Module 9.3): each GPU holds a subset of experts, and tokens are shipped (all-to-all) to wherever their chosen experts live. This all-to-all communication and the load imbalance are the real engineering costs of MoE — which is exactly why load balancing matters so much. Sparse activation is a memory-for-compute trade: DeepSeek-V3 stores 671B parameters but computes as if it were a ~37B dense model per token.

<div class="callout warn"><p><strong>Capacity and token dropping.</strong> Real MoE training caps how many tokens each expert will accept per batch (a <em>capacity factor</em>). If the router over-sends to an expert beyond its capacity, the overflow tokens are <em>dropped</em> — they skip the expert (only the residual passes through). Good load balancing keeps dropping rare; a collapsed router drops heavily. Our teaching implementation has no capacity limit (it runs every routed token), which is fine on CPU but not how production kernels behave.</p></div>

## Common mistakes

- **No load balancing.** Without the aux loss (or the bias trick) the router collapses onto a few experts and the rest die. Always balance.
- **Gate gradient through the hard count.** The aux loss's $f_i$ is a count and must be detached; gradients flow through $P_i$. Backprop through the top-$k$ selection itself is not done.
- **Forgetting to renormalize the top-$k$ gates.** After selecting $k$ of $N$, the kept probabilities no longer sum to 1; renormalize so the output is a proper weighted average.
- **Confusing MoE parameter count with active parameters.** "671B" is total; "~37B" is what each token activates. Report both — they measure different things (memory vs compute).

## Exercise

A dense model's FFN has $P$ parameters. You replace it with an MoE of $N = 8$ experts (each the same size as the dense FFN) and top-$k = 2$ routing, ignoring the tiny router. (a) How many FFN parameters does the layer now have? (b) Roughly how many does each token activate? (c) What is the ratio of total to active?

<details><summary>Solution</summary>

(a) $8P$ — eight experts, each of size $P$. (b) $\approx 2P$ — each token runs its top-2 experts. (c) $8P / 2P = 4\times$ more total parameters than are active per token. You get 8× the FFN capacity at ~2× the per-token FFN compute.

</details>

## Debugging exercise

An engineer trains an MoE and finds that after a few hundred steps, expert 3 processes ~90% of all tokens and most other experts process almost none — yet the loss is still going down. They forgot one term in the loss. What did they forget, what is this failure called, and why does the loss still fall for a while?

<details><summary>Answer</summary>

They forgot the **auxiliary load-balance loss** (or an equivalent balancing mechanism). This is **router collapse**: the rich-get-richer feedback loop concentrates tokens on a few experts, leaving the rest untrained and effectively wasting most of the parameters. The LM loss can still fall for a while because the one or two overworked experts do learn — the model behaves like a much smaller dense model. But it never realizes the capacity MoE promised, and if experts are sharded the load is badly imbalanced across devices. Adding $\mathcal{L}_{\text{aux}}$ (or DeepSeek-V3's bias update) spreads the tokens.

</details>

## Research connection

<div class="callout paper"><p><strong>Switch Transformers</strong> (Fedus et al. 2021) introduced top-1 MoE routing with the load-balance aux loss; <strong>DeepSeekMoE</strong> (Dai et al. 2024) added fine-grained and shared experts; <strong>DeepSeek-V3</strong> (2024) added aux-loss-free bias balancing and scaled to 671B params / ~37B active. All three are in <a href="../../papers/index.md">the paper curriculum</a> (#12, #13, #14). Read them in that order — each refines the last.</p></div>

## Check yourself

<details><summary>How does MoE add parameters without adding per-token compute?</summary>

It has $N$ expert FFNs but routes each token to only its top-$k$. Total parameters scale with $N$ (all experts are stored), but each token is processed by just $k$ of them, so per-token FLOPs are $\approx k$ dense-FFN passes regardless of $N$. Most experts are idle for any given token (sparse activation).

</details>

<details><summary>What is router collapse and how does the aux loss prevent it?</summary>

Router collapse is when a few experts get most of the tokens (a rich-get-richer loop) and the rest go untrained. The aux loss $N\sum_i f_i P_i$ is large when tokens pile on already-favoured experts and minimal when load is uniform, so its gradient (through $P_i$) pushes the router to raise probability on under-used experts, spreading the load.

</details>

<details><summary>For the token with logits $(2, 1, 0.1, -1)$ and $k = 2$, which experts run and with what gate weights?</summary>

Softmax gives $(0.6381, 0.2347, 0.0954, 0.0318)$; the top 2 are experts 0 and 1. Renormalizing their probabilities over the pair (sum 0.8728) gives gates $0.7311$ and $0.2689$. Experts 2 and 3 are not evaluated.

</details>

<details><summary>What do shared experts add on top of routed experts, and why?</summary>

Shared experts are always active for every token (not routed). They capture knowledge common to nearly all tokens (basic syntax, frequent patterns), so the routed experts don't each have to relearn it and can specialize. The output sums the shared-expert outputs and the gate-weighted routed-expert outputs.

</details>

<details><summary>DeepSeek-V3 is "671B parameters, ~37B active." What do the two numbers mean?</summary>

671B is the *total* parameter count (all experts across all layers, stored in memory). ~37B is the number *activated per token* — the shared plus top-$k$ routed experts a single token actually flows through, which sets the compute cost. MoE makes these two numbers very different.

</details>

## Next

You now have every piece that separates a GPT-2 from a LLaMA/DeepSeek-style model: RoPE positions, RMSNorm, SwiGLU, GQA, and MoE. Module 13 turns to measuring these models. Continue to [13.1 · Loss, perplexity, zero-/few-shot](../module-13/lesson-01.md).
