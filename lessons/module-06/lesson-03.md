# 06.3 · Generation: greedy, temperature, top-k, top-p

<div class="prereq">
<p><strong>Prerequisites:</strong> the <code>forward(idx)</code> pass and logits <code>(B, T, V)</code> from <a href="#/lessons/module-06/lesson-01">06.1 · Assembling GPT-2</a>; the categorical distribution and sampling with <code>torch.multinomial</code> from <a href="#/lessons/module-01/lesson-01">01.1 · Random variables, expectation, variance</a>; softmax from <a href="#/lessons/module-01/lesson-04">01.4</a>.</p>
<p><strong>You will learn:</strong> the <strong>autoregressive loop</strong> — predict one token from the last position, append it, repeat, cropping the context to <code>block_size</code>; and the four decoding strategies that turn a logit vector into a next token: <strong>greedy</strong> (argmax), <strong>temperature</strong> scaling (sharpen or flatten), <strong>top-k</strong> (keep the $k$ best), and <strong>top-p / nucleus</strong> (keep the smallest set whose probability mass reaches $p$). Each is worked by hand on the same 5-logit vector and verified in Python.</p>
<p><strong>Why this matters for ML:</strong> a trained model only ever outputs a distribution; <em>decoding</em> is the separate, tunable step that converts that distribution into actual text. The same model can sound robotic or unhinged purely from the decoding knobs. Every serving stack (and the sampling params in every chat API) is exactly these four operations, so this is the code you will reach for constantly.</p>
</div>

## 1. The autoregressive loop

A language model predicts *one* thing: a distribution over the next token given a prefix. To generate a whole passage we apply it repeatedly — feed the prefix, sample one token, glue it onto the end, feed the longer prefix, and so on. That feedback (the model's own output becomes its next input) is what **autoregressive** means.

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # 1. crop the context to the last block_size tokens
        idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
        # 2. forward; we only need the logits at the LAST position
        logits, _ = self(idx_cond)          # (B, T, V)
        logits = logits[:, -1, :] / temperature   # (B, V)
        # 3. optional top-k filtering
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float("inf")
        # 4. softmax -> categorical -> sample one token
        probs = F.softmax(logits, dim=-1)   # (B, V)
        idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)
        # 5. append and continue
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

Five points, each load-bearing:

- **`@torch.no_grad()`** — generation is inference; we never call `.backward()`, so we tell PyTorch not to build the autograd graph. That halves the memory and speeds every forward up.
- **Last position only.** `forward` returns logits for *every* position, but only the last one predicts the token that comes *next*. We slice `logits[:, -1, :]` — shape $(B, T, V) \to (B, V)$ — and discard the rest.
- **Crop to `block_size`.** The model has only `block_size` position embeddings; feed it a longer context and `wpe` has no row for the extra positions (and the assert in `forward` fires). So once the running sequence passes `block_size`, we keep only its last `block_size` tokens. The model never sees more context than it was built for.
- **Sample, don't just pick.** `torch.multinomial(probs, 1)` draws one token *in proportion to* its probability (Module 1.1) — this is what makes generation varied rather than a single fixed continuation.
- **`torch.cat`** grows the sequence by one column each step; after `max_new_tokens` iterations the output is $(B,\ T_0 + \text{max\_new\_tokens})$.

<div class="callout warn"><p>Every step re-runs the model on the whole (cropped) prefix, so naive generation is $\mathcal{O}(T^2)$ over the run — it recomputes the keys and values for tokens it already processed. Production serving avoids this with a <strong>KV cache</strong> that stores past keys/values so each step is $\mathcal{O}(T)$. We build the cache in a later module; the loop here is the correct, simple reference version.</p></div>

For everything below, we focus on step 3–4 for a single sequence — how one logit vector becomes one next-token id. We use the same running example throughout: a $V = 5$ vocabulary with logits

$$
\mathbf{z} = [\,2.0,\ 1.0,\ 0.0,\ -1.0,\ 0.5\,].
$$

Its plain softmax ($T = 1$, no filtering) is our baseline distribution:

$$
q = [\,0.5630,\ 0.2071,\ 0.0762,\ 0.0280,\ 0.1256\,].
$$

(Token 0 has the largest logit and thus the largest probability; token 3 the smallest of both. The five sum to 1.)

## 2. Greedy decoding

The simplest rule: always take the highest-logit token. No sampling, no randomness.

$$
\text{next} = \arg\max_i z_i = \arg\max [\,2.0, 1.0, 0.0, -1.0, 0.5\,] = 0.
$$

Greedy is deterministic — the same prompt always yields the same continuation. That is good for reproducibility and for tasks with one right answer, but it makes free-form text bland and repetitive (it often falls into loops, repeating a phrase forever). In the loop, greedy is exactly `top_k=1`: keep only the single best logit, so softmax puts all mass on it and `multinomial` can only draw that token.

## 3. Temperature

Temperature $T$ rescales the logits *before* softmax:

$$
q_i(T) = \operatorname{softmax}\!\left(\frac{\mathbf z}{T}\right)_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}.
$$

Dividing by $T$ stretches or squashes the *gaps* between logits, which softmax then turns into sharper or flatter probabilities:

- $T < 1$ **sharpens** — magnifies the gaps, so the top token dominates (toward greedy).
- $T > 1$ **flattens** — shrinks the gaps, so the distribution approaches uniform (more random, more "creative").
- $T = 1$ leaves it unchanged.
- $T \to 0$ is the greedy limit; $T \to \infty$ is uniform.

Our vector at three temperatures (verified in Python):

| token | $z$ | $T=0.5$ | $T=1$ | $T=2$ |
|---|---|---|---|---|
| 0 | 2.0 | **0.8292** | 0.5630 | 0.3745 |
| 1 | 1.0 | 0.1122 | 0.2071 | 0.2272 |
| 2 | 0.0 | 0.0152 | 0.0762 | 0.1378 |
| 3 | -1.0 | 0.0021 | 0.0280 | 0.0836 |
| 4 | 0.5 | 0.0413 | 0.1256 | 0.1769 |

Read across a row: as $T$ rises from $0.5$ to $2$, token 0's mass falls from $0.83$ to $0.37$ while the low-probability tokens (2, 3, 4) all rise. At $T = 0.5$ the model is nearly committed to token 0; at $T = 2$ it is close to a coin-toss among the top few. Worked by hand for token 0 at $T = 0.5$: $e^{2.0/0.5} = e^{4} = 54.60$, and dividing by the sum of all five $e^{z_j/0.5}$ ($= 65.84$) gives $54.60 / 65.84 = 0.8292$.

```python
import torch, torch.nn.functional as F
z = torch.tensor([2.0, 1.0, 0.0, -1.0, 0.5])
for T in (0.5, 1.0, 2.0):
    print(T, F.softmax(z / T, dim=-1).round(decimals=4).tolist())
# 0.5 [0.8292, 0.1122, 0.0152, 0.0021, 0.0413]
# 1.0 [0.5630, 0.2071, 0.0762, 0.0280, 0.1256]
# 2.0 [0.3745, 0.2272, 0.1378, 0.0836, 0.1769]
```

<div class="callout warn"><p>In the loop, temperature is <code>logits / temperature</code>, so <code>temperature=0</code> would divide by zero. Handle the greedy case with <code>top_k=1</code> (or a special-case argmax) rather than <code>temperature=0</code>. Typical text-generation values are $0.7$–$1.0$; below $0.5$ it turns nearly greedy, above $1.5$ it starts to babble.</p></div>

## 4. Top-k sampling

Temperature reshapes the *whole* distribution but never zeroes anything out — even a nonsense token keeps a sliver of probability and will, eventually, be sampled. **Top-k** truncates: keep only the $k$ highest-logit tokens as candidates, set the rest to $-\infty$ (so they get exactly zero probability), then softmax over the survivors.

With $k = 2$ on $\mathbf z = [2.0, 1.0, 0.0, -1.0, 0.5]$: the two largest logits are $2.0$ (token 0) and $1.0$ (token 1). Everything below the 2nd-largest becomes $-\infty$:

$$
\mathbf z' = [\,2.0,\ 1.0,\ -\infty,\ -\infty,\ -\infty\,].
$$

Softmax over the two survivors: $e^{2}=7.389$, $e^{1}=2.718$, sum $10.107$, giving

$$
q' = [\,0.7311,\ 0.2689,\ 0,\ 0,\ 0\,].
$$

Now only tokens 0 and 1 can ever be drawn; the tail is truly gone. The code matches the loop's step 3:

```python
k = 2
v, _ = torch.topk(z, k)          # v = [2.0, 1.0]; v[-1] is the k-th largest = 1.0
z_filt = z.clone()
z_filt[z < v[-1]] = -float('inf')   # kill everything below the threshold
print(F.softmax(z_filt, dim=-1).round(decimals=4).tolist())
# [0.7311, 0.2689, 0.0, 0.0, 0.0]
```

The `z < v[-1]` comparison uses the $k$-th largest logit as the cutoff. Top-k's weakness: $k$ is *fixed*, but distributions vary. When the model is confident (one token at 0.95) $k = 40$ needlessly admits 39 junk tokens; when it is unsure (mass spread over hundreds), a small $k$ chops off good options. That is what top-p fixes.

## 5. Top-p (nucleus) sampling

**Top-p** keeps a *variable* number of tokens: the smallest set whose cumulative probability reaches a threshold $p$. Sort the probabilities descending, walk down accumulating mass, and stop once the running total is $\ge p$; keep exactly those tokens, drop the rest, and renormalize.

On our baseline $q$ (the $T=1$ softmax) with $p = 0.9$. Sort $q$ descending and take the cumulative sum:

| rank | token | prob | cumulative |
|---|---|---|---|
| 1 | 0 | 0.5630 | 0.5630 |
| 2 | 1 | 0.2071 | 0.7701 |
| 3 | 4 | 0.1256 | 0.8958 |
| 4 | 2 | 0.0762 | **0.9720** ← crosses 0.9 |
| 5 | 3 | 0.0280 | 1.0000 |

The cumulative mass first reaches $0.9$ at rank 4, so the **nucleus** is $\{0, 1, 4, 2\}$ — the four tokens up to and including the one that tips the total past $p$. Token 3 (the last) is dropped. Renormalize the survivors so they sum to 1 again:

$$
q_{\text{top-}p} = [\,0.5793,\ 0.2131,\ 0.1293,\ 0.0784\,] \quad (\text{for tokens } 0, 1, 4, 2).
$$

Each survivor was divided by the kept mass $0.9720$: e.g. token 0 becomes $0.5630 / 0.9720 = 0.5793$.

```python
q = F.softmax(z, dim=-1)
sp, si = torch.sort(q, descending=True)           # sorted probs and their indices
csum = torch.cumsum(sp, dim=-1)                     # [0.5630,0.7701,0.8958,0.9720,1.0]
keep = (csum - sp) < 0.9        # keep a token if the mass BEFORE it is still < p
kept = sp * keep
kept = kept / kept.sum()
print(kept.round(decimals=4).tolist())              # [0.5793,0.2131,0.1293,0.0784,0.0]
```

The trick in the mask is `(csum - sp) < p`: `csum - sp` is the cumulative mass *excluding the current token*, so we keep every token whose predecessors have not yet reached $p$ — which correctly includes the one token that pushes the total over the line. Because the cutoff adapts to the distribution's shape, top-p keeps many tokens when the model is uncertain and few when it is confident — the adaptivity top-k lacks. In practice $p \approx 0.9$–$0.95$ is common, often combined with a temperature.

<div class="callout key"><p>All four strategies are just ways to sculpt the logit vector before <code>multinomial</code>. <strong>Greedy</strong>: take the max. <strong>Temperature</strong>: divide logits by $T$ to sharpen ($T<1$) or flatten ($T>1$). <strong>Top-k</strong>: keep the $k$ best, mask the rest to $-\infty$. <strong>Top-p</strong>: keep the smallest set reaching cumulative mass $p$. Temperature reshapes; top-k and top-p truncate; they are routinely combined.</p></div>

<div class="callout paper"><p>Top-p / nucleus sampling comes from "The Curious Case of Neural Text Degeneration" (Holtzman et al. 2020), which showed that maximization-based decoding (greedy/beam) produces bland, repetitive text and that truncating to the nucleus fixes it. See the reading guides in <a href="#/papers/index">the paper index</a>.</p></div>

## 6. Under the hood: what generation costs

Each generated token is one full forward pass over the current context. Sorting for top-p is $\mathcal{O}(V \log V)$ per step — negligible next to the forward. The dominant cost is re-processing the prefix every step (the $\mathcal{O}(T^2)$ noted in §1); for the tiny models in this course that is imperceptible, which is why generation runs comfortably on a CPU.

<div class="hw">
<p><strong>Hardware — generation from a small model.</strong> Minimum: any CPU (no GPU needed). Recommended: same; a GPU helps only for large models or long batches. Expected runtime: a few hundred tokens from a &lt;10M-parameter model is well under a second on a laptop CPU. GPU memory: n/a (CPU); on GPU, inference needs far less than training since <code>@torch.no_grad()</code> stores no activations. GPU-hours: ~0. CPU-only: fully supported — every example in this lesson runs with <code>torch 2.14</code> CPU.</p>
</div>

## Exercise

You generate from a well-trained model with `top_k=1` and complain that the output repeats the same sentence over and over. A colleague suggests raising `temperature` to 1.3 to "add variety", but with `top_k=1` still set, nothing changes. Explain why temperature has no effect when `top_k=1`, and give a decoding setting that would actually add variety without admitting obvious garbage tokens.

<details><summary>Hint</summary>
What does <code>top_k=1</code> leave for softmax to work with, and does scaling a single surviving logit change which token is chosen?
</details>

<details><summary>Stronger hint</summary>
After top-k filtering keeps exactly one logit, the other $V-1$ are $-\infty$. Softmax of "[one finite value, rest $-\infty$]" is $[1, 0, \dots, 0]$ regardless of what the finite value is.
</details>

<details><summary>Solution</summary>

With `top_k=1`, filtering sets every logit except the single largest to $-\infty$. Softmax then puts probability 1 on that one token and 0 on all others — and dividing the surviving logit by any temperature does not change the fact that it is the *only* finite entry, so the distribution stays $[1, 0, \dots, 0]$. Sampling is forced to the argmax; temperature is inert. That is also why the output is repetitive: `top_k=1` is greedy, which loops. To add variety while keeping quality, widen the candidate set and let temperature act on it — e.g. `top_k=40` with `temperature≈0.8`, or `top_p=0.9`. Now several plausible tokens survive with nonzero probability, temperature reshapes their relative mass, and sampling breaks the loops, while the truncation still excludes the long tail of nonsense tokens.

</details>

## Check yourself

<details><summary>In the generation loop, why do we take <code>logits[:, -1, :]</code> and ignore the logits at all other positions?</summary>

Only the last position's logits predict the token that comes *next* — the one we are about to append. The logits at earlier positions predict tokens we already have. (During training we use all positions at once; during generation only the final one is new.)

</details>

<details><summary>The running sequence has grown to 1500 tokens but <code>block_size = 1024</code>. What context does the model actually see on the next step?</summary>

The last 1024 tokens: `idx[:, -1024:]`. Earlier tokens are cropped away because the model has only 1024 position embeddings. So the model "forgets" anything more than `block_size` tokens back.

</details>

<details><summary>On logits <code>[3.0, 2.0, 1.0]</code> with $k = 2$, what probabilities does top-k give? (Give the two nonzero values.)</summary>

Keep logits $3.0$ and $2.0$, mask $1.0$ to $-\infty$. $e^{3}=20.09$, $e^{2}=7.39$, sum $27.48$: probabilities $[0.7311,\ 0.2689,\ 0]$. (Same shape as the §4 example, since only the gap of 1.0 between the top two matters.)

</details>

<details><summary>Why can top-p keep a different number of tokens at different steps, and why is that an advantage over top-k?</summary>

Top-p keeps the smallest set whose cumulative probability reaches $p$, and how many tokens that takes depends on the distribution's shape — few when the model is confident (mass concentrated), many when it is uncertain (mass spread out). Top-k's fixed $k$ cannot adapt: it admits junk when the model is sure and truncates good options when it is unsure.

</details>

## Next

You now have the complete GPT-2: assembled ([06.1](lessons/module-06/lesson-01.md)), initialized and sized ([06.2](lessons/module-06/lesson-02.md)), and able to generate text with four decoding strategies. What it *cannot* yet do is learn — its weights are still random. Module 7 builds the training loop: the data batches, the AdamW optimizer you wrote in Module 3, the learning-rate schedule, gradient clipping, and the metrics (loss curves, MFU) that tell you a run is healthy — and watches the init loss you predicted here fall from $\ln V$ toward something that produces real language.

Continue to [Module 7 · Training a language model](lessons/module-07/lesson-01.md).
