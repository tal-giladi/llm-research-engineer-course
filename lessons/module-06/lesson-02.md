# 06.2 · Init, forward pass, parameter count

<div class="prereq">
<p><strong>Prerequisites:</strong> the assembled model from <a href="lesson-01.md">06.1 · Assembling GPT-2</a>; the per-block "$\approx 12C^2$ parameters" estimate from <a href="../module-05/lesson-04.md">05.4</a>; cross-entropy and its value on a uniform prediction from <a href="../module-01/lesson-04.md">01.4</a>.</p>
<p><strong>You will learn:</strong> GPT-2's weight <strong>initialization</strong> — why every weight starts as $\mathcal{N}(0, 0.02^2)$, and the special <strong>scaled residual init</strong> that shrinks the residual-writing projections by $1/\sqrt{2\,n_{\text{layer}}}$ so the residual stream's variance does not blow up with depth; how to <strong>count the parameters</strong> of GPT-2 small from a closed-form formula and verify it against <code>sum(p.numel())</code>, arriving at exactly $124{,}439{,}808$; and why an <em>untrained</em> model's loss is $\approx \ln(V)$, verified numerically.</p>
<p><strong>Why this matters for ML:</strong> "the loss started at 10.8 and dropped" is the first sanity check of every training run — if step 0 is not near $\ln(V)$, your model or data pipeline is broken before you have wasted a single GPU-hour. And knowing the parameter count cold lets you predict memory, FLOPs, and cost without running anything.</p>
</div>

## 1. Initialization: why the starting weights matter

A network's weights have to *start* somewhere. Start them too large and activations (and gradients) explode as they pass through layer after layer; too small and the signal decays to nothing. The goal of an init scheme is that a freshly-built, untrained network passes a signal of roughly **unit scale** all the way through — neither growing nor shrinking systematically with depth.

GPT-2's scheme is deliberately simple:

- Every `nn.Linear` weight and every `nn.Embedding` weight: $\mathcal{N}(0,\ 0.02^2)$ — a normal with standard deviation $0.02$.
- Every bias: $0$.

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

self.apply(self._init_weights)   # walks every submodule and calls _init_weights on it
```

`self.apply(fn)` recursively visits every sub-module and runs `fn` on it — a clean way to set a rule once and have it hit all 100+ weight tensors. Why $0.02$ specifically and not the more principled $1/\sqrt{C} = 1/\sqrt{768} \approx 0.036$? It is the value GPT-2 shipped; it is small enough to keep early activations calm and was found to train well. We reproduce what GPT-2 did rather than re-derive it.

## 2. Scaled residual init: keeping the stream calm as it deepens

### The problem

Recall the residual stream (Module 5.4): each block does `x = x + sublayer(ln(x))`. Every block *adds* its output into the same running vector `x`. If each of the $n_{\text{layer}}$ additions contributes an independent chunk of variance $v$, then — because variances of independent terms add — the stream's variance after $L$ blocks is roughly

$$
\operatorname{Var}(x_L) \approx \operatorname{Var}(x_0) + L \cdot v.
$$

It **grows linearly with depth**. By layer 12 (or 96, in a big model) the stream is far larger than it started, which destabilizes the LayerNorms and the final logits.

### The fix

GPT-2 shrinks the *specific* weights that write into the stream. In each block, two projections produce what gets added: the attention output projection `attn.c_proj` and the MLP output projection `mlp.c_proj`. If we scale their init standard deviation down by $1/\sqrt{N}$, the variance each contributes drops by $1/N$. Choosing $N = 2\,n_{\text{layer}}$ — the total number of residual adds in the network (2 per layer) — makes the accumulated growth cancel, so the stream stays $\mathcal{O}(1)$ regardless of depth:

$$
\text{std}(\texttt{c\_proj}) = \frac{0.02}{\sqrt{2\,n_{\text{layer}}}}.
$$

```python
for name, p in self.named_parameters():
    if name.endswith("c_proj.weight"):          # attn.c_proj and mlp.c_proj
        nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layer))
```

For GPT-2 small, $n_{\text{layer}} = 12$, so the factor is $1/\sqrt{24} \approx 0.204$ and the residual-writing projections start at std $\approx 0.02 \times 0.204 \approx 0.00408$ — five times smaller than every other weight.

<div class="callout key"><p>Scaled residual init = shrink only the projections that <em>write into</em> the residual stream by $1/\sqrt{2\,n_{\text{layer}}}$. Because independent variances add, this keeps the stream's variance $\approx$ constant as you stack layers, so a 12-layer and a 96-layer model both start out well-behaved. The "2" counts the two residual adds per block (attention and MLP).</p></div>

## 3. Counting the parameters of GPT-2 small

We will count from the shapes, then verify against PyTorch. Write $C = n_{\text{embd}}$, $V = \text{vocab\_size}$, $L = n_{\text{layer}}$, and assume `bias=True`.

### Embeddings

- `wte`: $V \times C$. Tied into `lm_head`, so counted **once**.
- `wpe`: $\text{block\_size} \times C$.

$$
\text{embeddings} = C\,(V + \text{block\_size}).
$$

### One block

From [05.4](../module-05/lesson-04.md), the weight matrices in a block are:

| tensor | shape | params |
|---|---|---|
| `attn.c_attn` (QKV) | $C \times 3C$ | $3C^2$ |
| `attn.c_proj` | $C \times C$ | $C^2$ |
| `mlp.c_fc` | $C \times 4C$ | $4C^2$ |
| `mlp.c_proj` | $4C \times C$ | $4C^2$ |

So the weight matrices sum to $3C^2 + C^2 + 4C^2 + 4C^2 = 12C^2$ — the "$\approx 12C^2$ per block" from Module 5. The smaller terms: the four bias vectors on those linears total $3C + C + 4C + C = 9C$, and the two LayerNorms add $2C$ (weights) $+ 2C$ (biases) $= 4C$. Altogether

$$
\text{per block} = 12C^2 + 9C + 4C = 12C^2 + 13C.
$$

### Final norm and total

`ln_f` adds $2C$ (weight + bias). The head is tied, so it adds nothing new. Total:

$$
\boxed{\;N = C\,(V + \text{block\_size}) \;+\; L\,(12C^2 + 13C) \;+\; 2C\;}
$$

### Plug in GPT-2 small

$C = 768$, $V = 50257$, $\text{block\_size} = 1024$, $L = 12$:

- **Embeddings:** $768 \times (50257 + 1024) = 768 \times 51281 = 39{,}383{,}808$.
  - of which `wte` (tied): $50257 \times 768 = 38{,}597{,}376$; `wpe`: $1024 \times 768 = 786{,}432$.
- **Per block:** $12 \times 768^2 + 13 \times 768 = 7{,}077{,}888 + 9{,}984 = 7{,}087{,}872$. Times 12: $85{,}054{,}464$.
- **`ln_f`:** $2 \times 768 = 1{,}536$.

$$
N = 39{,}383{,}808 + 85{,}054{,}464 + 1{,}536 = 124{,}439{,}808 \approx 124\text{M}.
$$

That is the famous "GPT-2 124M". Note it is *dominated* by the blocks (85M); the embeddings are 39M, of which the position table is a mere 0.8M.

### Verify in PyTorch

```python
from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT

cfg = GPTConfig()                 # GPT-2 small defaults
model = GPT(cfg)
print(sum(p.numel() for p in model.parameters()))   # 124439808
```

This prints exactly `124439808`, matching the formula. `model.parameters()` de-duplicates the tied `wte`/`lm_head` tensor, so it is counted once — which is why our formula counted it once too.

<div class="callout pt"><p>nanoGPT reports a slightly smaller "<em>non-embedding</em>" count of $123{,}653{,}376$ — it subtracts the position table <code>wpe</code> ($786{,}432$) on the grounds that it is not part of the "core" transformer. It does <em>not</em> subtract <code>wte</code>, because the tie means that table is also the output head and is genuinely used to produce logits. Our <code>num_params(non_embedding=True)</code> returns this figure.</p></div>

## 4. The loss of an untrained model is $\ln(V)$

### Why

At initialization the model knows nothing. Its logits at each position are near-zero, tiny random numbers, so after softmax the predicted distribution over the $V$ tokens is almost **uniform**: each token gets probability $\approx 1/V$. Cross-entropy against the true token $t$ is $-\log q(t)$ (from [01.4](../module-01/lesson-04.md)), and with $q(t) \approx 1/V$:

$$
\mathcal{L}_{\text{init}} \approx -\log\!\frac{1}{V} = \log V.
$$

For GPT-2's vocabulary, $\ln(50257) \approx 10.82$. So the very first training step should report a loss right around $10.8$. If it reports $30$, your logits are exploding (init or a bug); if it reports $2$, your "random" targets are not random (a data-pipeline leak). This single number is the cheapest, most reliable smoke test in all of LLM training.

### Verify on a small vocab

Numbers this large are hard to eyeball, so we check on a tiny model with $V = 65$ (a character-level vocabulary), where $\ln(65) \approx 4.174$:

```python
import torch, math
from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT

torch.manual_seed(0)
cfg = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=4, n_embd=64)
model = GPT(cfg).eval()
idx     = torch.randint(0, 65, (4, 16))
targets = torch.randint(0, 65, (4, 16))
_, loss = model(idx, targets)
print(round(loss.item(), 4), round(math.log(65), 4))   # 4.2278 4.1744
```

The measured init loss is $4.2278$ against a theoretical $\ln(65) = 4.1744$ — a hair high (the logits are not *exactly* zero at init), but unmistakably "about $\ln V$". A unit test in `code/tests/test_gpt.py` asserts exactly this, with a tolerance of $0.5$.

<div class="callout warn"><p>Do not expect the init loss to hit $\ln V$ to three decimals. The random init gives slightly non-uniform logits, and with a small batch there is sampling noise. "Within a few tenths of $\ln V$" is the assertion; a value that is off by a whole point or more is the signal that something is wrong.</p></div>

## 5. Under the hood: init cost, and a small-config check

Initialization itself is cheap — filling ~124M floats from a normal RNG is milliseconds — but note two subtleties. First, `self.apply(self._init_weights)` runs *before* the scaled-residual loop, so the `c_proj` weights are first set to std $0.02$ and then **overwritten** to the smaller std; order matters. Second, because `wte` and `lm_head` are tied, `_init_weights` will touch that one tensor twice (once as the embedding, once as the linear) — harmless, since both branches use the same std $0.02$, but worth knowing when you reason about determinism.

To make the formula concrete on numbers you can check by hand, here is the small config used by the tests ($V=100$, $\text{block\_size}=16$, $L=3$, $C=32$):

$$
N = 32(100 + 16) + 3(12\cdot 32^2 + 13\cdot 32) + 2\cdot 32 = 3712 + 3\cdot 12704 + 64 = 41{,}888,
$$

which is exactly what `GPT(cfg).num_params()` returns.

## Exercise

You build a **48-layer** model (keeping $C = 768$) but forget to change the scaled-residual init — you leave every `c_proj` at std $0.02$ like the other weights. Predict, qualitatively, what happens to the residual stream's scale by the last layer at initialization, and to the very first training step's loss.

<details><summary>Hint</summary>
The stream's variance grows like $\operatorname{Var}(x_0) + (\text{number of adds}) \times v$. How many adds now, and what is $v$ without the shrink?
</details>

<details><summary>Stronger hint</summary>
With the shrink, each add contributes variance $\propto (0.02/\sqrt{2L})^2 \propto 1/L$, so the $L$ adds sum to $\mathcal{O}(1)$. Without it, each add contributes a fixed $\propto 0.02^2$, so the sum grows like $L$.
</details>

<details><summary>Solution</summary>

Without the $1/\sqrt{2L}$ shrink, each of the $2L = 96$ residual adds injects a roughly constant variance, so the stream's variance grows linearly to $\approx 96\times$ its per-layer contribution by the top of the stack — far from unit scale. The final `ln_f` still normalizes the magnitude, but the *relative* contributions of early vs. late layers are badly skewed and the logits are large and erratic. Concretely, the step-0 loss will come out well above $\ln V$ (often 15–30 instead of ~10.8), and training is unstable or diverges early. The scaled-residual init exists precisely so that going from 12 to 48 to 96 layers does not require re-tuning anything — the stream stays $\mathcal{O}(1)$ at every depth.

</details>

## Check yourself

<details><summary>Which weights get the shrunk std, and what is the shrink factor for a 24-layer model?</summary>

Only the residual-writing projections — `attn.c_proj` and `mlp.c_proj` in every block. The factor is $1/\sqrt{2\,n_{\text{layer}}} = 1/\sqrt{48} \approx 0.144$, so their std is $0.02 \times 0.144 \approx 0.00289$.

</details>

<details><summary>A block has $\approx 12C^2$ parameters. For $C = 1600$ (GPT-2 XL's width), how many parameters is that per block, ignoring the linear-in-$C$ terms?</summary>

$12 \times 1600^2 = 12 \times 2{,}560{,}000 = 30{,}720{,}000 \approx 30.7$M per block.

</details>

<details><summary>You launch training and step 0 reports a loss of 2.3 on a $V = 50257$ model. Is that plausible for a correct setup? What does it suggest?</summary>

No — a correct untrained model should report $\approx \ln(50257) \approx 10.8$. A loss of $2.3$ at step 0 means the model is already "predicting" the targets, which almost always indicates a data leak: the targets are visible in the inputs, or inputs and targets are misaligned so the task is trivial.

</details>

<details><summary>Why does <code>sum(p.numel() for p in model.parameters())</code> give 124M and not ~162M, even though there is both a <code>wte</code> and an <code>lm_head</code>?</summary>

Weight tying makes them the *same* Parameter object, and `model.parameters()` de-duplicates shared tensors (it yields each unique parameter once). So the $38.6$M table is counted a single time; counting it twice would give $\approx 163$M.

</details>

## Next

The model is assembled, initialized, and its size understood. The last thing to do with a language model is the thing it was built for: produce text. The next lesson implements the autoregressive generation loop and the four decoding strategies — greedy, temperature, top-k, and top-p — each worked out on a tiny logit vector by hand.

Continue to [06.3 · Generation: greedy, temperature, top-k, top-p](lesson-03.md).
