# 14.2 · Loss masking & packing

<div class="prereq">
<p><strong>Prerequisites:</strong> chat templates and the response mask from <a href="#/lessons/module-14/lesson-01">14.1</a>; the next-token cross-entropy loss from <a href="#/lessons/module-01/lesson-04">01.4 · The language-modeling loss</a>; how <code>GPT.forward</code> already ignores label <code>-1</code> from <a href="#/lessons/module-06/lesson-01">06.1</a>.</p>
<p><strong>You will learn:</strong> why in SFT we compute cross-entropy on the <strong>response tokens only</strong> and not the prompt; how the <code>ignore_index = -100</code> trick implements that; a tiny worked example proving the masked loss equals the loss over just the response positions; and how to <strong>pack</strong> several short examples into one fixed-length block while keeping each example's mask correct.</p>
<p><strong>Why this matters for ML:</strong> masking the prompt is the difference between "learn to answer" and "learn to also parrot the question." Packing is how SFT stays GPU-efficient: instruction examples are short and wildly uneven in length, and padding each one to the block size would waste most of the compute. Every SFT trainer you will use (TRL's <code>SFTTrainer</code>, Axolotl) does exactly these two things under the hood.</p>
</div>

## 1. Intuition: only score what you want the model to generate

In pretraining, every position is a legitimate prediction target — the model should be able to continue *any* text, so the loss covers the whole sequence. SFT is different. A fine-tuning example is `prompt + response`, and at inference time the **prompt is given to us** — the user typed it, the template built it. We never ask the model to *generate* the prompt. So we should not train it to.

If we included the prompt tokens in the loss, we would be optimizing the model to predict user questions and system messages. That wastes capacity on a distribution we do not care about, and it can actively hurt: the model learns to *continue* prompts rather than *respond* to them — the very failure SFT is supposed to cure.

<div class="callout key"><p>SFT loss rule: compute cross-entropy on the <strong>assistant-response tokens only</strong>. Every prompt position (system + user text, and all role headers) is <em>ignored</em> — it contributes exactly zero to the loss and zero gradient. We implement "ignore" with a sentinel label value, <code>-100</code>.</p></div>

## 2. Mathematics: the masked loss

Let the sequence have $N$ next-token prediction positions. At position $i$ the model outputs logits $\mathbf{z}_i \in \mathbb{R}^V$ and there is a target token $y_i$. Ordinary (pretraining) cross-entropy averages over all positions:

$$
\mathcal{L}_{\text{full}} = \frac{1}{N} \sum_{i=1}^{N} \big(-\log \operatorname{softmax}(\mathbf{z}_i)_{y_i}\big).
$$

For SFT we define a response mask $m_i \in \{0, 1\}$: $m_i = 1$ if position $i$ is an assistant-response token, else $0$. The **masked loss** averages only over the response positions:

$$
\mathcal{L}_{\text{SFT}} = \frac{\sum_{i=1}^{N} m_i \big(-\log \operatorname{softmax}(\mathbf{z}_i)_{y_i}\big)}{\sum_{i=1}^{N} m_i}.
$$

The denominator is the *number of response tokens*, not $N$ — we average over what we actually scored. Every symbol: $\mathbf{z}_i$ the logit row at position $i$, $y_i$ the true token id there, $m_i$ the mask, $V$ the vocabulary size.

In code we do not carry a separate mask into the loss. Instead we fold it into the **labels**: set the label to $y_i$ where $m_i = 1$ and to a sentinel $\texttt{ignore\_index} = -100$ where $m_i = 0$. Cross-entropy then skips every position whose label is $-100$. That is precisely $\mathcal{L}_{\text{SFT}}$ — a mean over the surviving positions.

<div class="callout warn"><p><code>GPT.forward</code> (Module 6) uses <code>ignore_index = -1</code>. The SFT modules use <strong>-100</strong>, which is HuggingFace's convention and the default of <code>torch.nn.functional.cross_entropy</code>. The value is arbitrary as long as it is a number that is never a valid token id; keep it consistent between <code>build_labels</code> and <code>masked_cross_entropy</code>.</p></div>

## 3. Numerical example (worked by hand, verified in Python)

Take a tiny sequence with $N = 4$ prediction positions and a vocabulary of $V = 3$. The first two positions are the prompt, the last two are the response, so the mask is $m = [0, 0, 1, 1]$. The target ids are $y = [0, 1, 2, 0]$ and the logits are

$$
Z = \begin{bmatrix} 2.0 & 0.0 & 0.0 \\ 0.0 & 2.0 & 0.0 \\ 0.0 & 0.0 & 3.0 \\ 1.0 & 0.5 & 0.0 \end{bmatrix}.
$$

**Step 1 — build labels.** Replace prompt positions with $-100$, keep the target id on response positions:

$$
\text{labels} = [\,-100,\ -100,\ 2,\ 0\,].
$$

**Step 2 — per-position response loss.** Only positions 3 and 4 count. Position 3 has logits $[0,0,3]$ and target $2$; softmax puts most mass on class 2, so its loss $-\log\operatorname{softmax}([0,0,3])_2$ is small:

$$
-\log \frac{e^{3}}{e^{0}+e^{0}+e^{3}} = 0.0949.
$$

Position 4 has logits $[1,0.5,0]$ and target $0$:

$$
-\log \frac{e^{1}}{e^{1}+e^{0.5}+e^{0}} = 0.6803.
$$

**Step 3 — average over the two response positions:**

$$
\mathcal{L}_{\text{SFT}} = \frac{0.0949 + 0.6803}{2} = 0.3876.
$$

Compare: the *full* loss over all four positions is $0.3136$ — a **different number**, because it also averages in the (here, easy) prompt positions. The masked loss equals the cross-entropy computed on positions 3–4 alone, which is the whole point. Verified in Python:

```python
import torch
from llmre.sft.masking import build_labels, masked_cross_entropy
from llmre.evaluation.metrics import cross_entropy

logits = torch.tensor([[2.,0.,0.],[0.,2.,0.],[0.,0.,3.],[1.,.5,0.]])
ids    = torch.tensor([0, 1, 2, 0])
mask   = torch.tensor([0, 0, 1, 1])

labels = build_labels(ids, mask)                 # [-100, -100, 2, 0]
print(masked_cross_entropy(logits, labels))      # tensor(0.3876)
print(cross_entropy(logits[2:], ids[2:]))        # tensor(0.3876)  <- identical
print(cross_entropy(logits, ids))                # tensor(0.3136)  <- full, different
```

<div class="callout key"><p>The masked cross-entropy over the full sequence <em>equals</em> the ordinary cross-entropy computed on only the response rows. That equality is the correctness test for the whole mechanism (<code>tests/test_sft.py</code>).</p></div>

## 4. Tensor shapes / dtype / device

- `input_ids`: `(T,)` or `(B, T)`, `torch.long`, on the training device.
- `response_mask`: same shape, bool or int (`1` = response).
- `build_labels` returns `labels`: same shape as `input_ids`, `torch.long`, `labels[i] = input_ids[i]` where the mask is truthy, else `-100`.
- `masked_cross_entropy(logits, labels)`: `logits` is `(..., V)` float (e.g. `(T, V)` or `(B, T, V)`); `labels` is `(...)`; returns a scalar `()` tensor in nats, on the logits' device.

Positions are aligned **one-to-one**: `logits[i]` is scored against `labels[i]`. In a real training loop you first *shift* so that position $i$ predicts token $i+1$ — feed `x = ids[:-1]`, build labels from `ids[1:]` and `response_mask[1:]`, and compute `masked_cross_entropy(model(x)[0], labels)`. Do the shift once, outside the loss, and pass already-aligned labels.

## 5. From-scratch PyTorch

```python
def build_labels(input_ids, response_mask, ignore_index=-100):
    mask = response_mask.to(torch.bool)
    labels = torch.full_like(input_ids, ignore_index)   # everything ignored...
    labels[mask] = input_ids[mask]                      # ...except response tokens
    return labels

def masked_cross_entropy(logits, labels, ignore_index=-100):
    flat_logits = logits.reshape(-1, logits.shape[-1])  # (N, V)
    flat_labels = labels.reshape(-1)                    # (N,)
    keep = flat_labels != ignore_index                  # drop ignored positions
    kept_logits, kept_labels = flat_logits[keep], flat_labels[keep]
    logp = kept_logits - torch.logsumexp(kept_logits, dim=-1, keepdim=True)
    rows = torch.arange(kept_labels.shape[0], device=kept_labels.device)
    return -logp[rows, kept_labels].mean()              # mean over kept rows
```

`build_labels` starts from an all-`ignore_index` tensor and writes the true ids back only on response positions — the inverse framing of "mask out the prompt," and less error-prone. `masked_cross_entropy` uses the same numerically stable log-softmax as [`llmre.evaluation.metrics.cross_entropy`](lessons/module-13/lesson-01.md) (subtract `logsumexp`, gather the true-token log-prob), just restricted to the kept rows.

<div class="callout pt"><p><code>torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)</code> does exactly this masking natively — its default <code>ignore_index</code> is even <code>-100</code>. We implement it by hand so the mechanism is visible; the production one-liner behaves identically and fuses log-softmax, the gather, and the mask into a single kernel.</p></div>

## Sequence packing

### 1. Intuition

Instruction examples are short and uneven: one is 20 tokens, the next 300. If we pad every example out to the block size $T$ (say 1024) and train one example per row, most positions are padding — the GPU does mostly wasted work. **Packing** concatenates several short examples end-to-end into one full block of length $T$, so almost every position is real. Fewer padded positions = higher throughput for the same hardware.

### 2. The catch, and the fix

Two examples now share a block. Two things must stay correct per example:

1. **The loss mask.** Each example keeps its own prompt/response split. We concatenate the per-example response masks the same way we concatenate the tokens, so `build_labels` still masks each example's prompt independently.
2. **Attention must not cross the boundary.** Example B's tokens must not attend to example A's — otherwise the model "reads" an unrelated earlier conversation. The clean fix is a **block-diagonal attention mask** (sometimes via `document_ids` / position resets) so each example only attends within itself. A cheaper approximation used in practice is to just concatenate and rely on the causal mask plus a separator token, accepting a little cross-contamination; the honest version resets attention at each boundary.

### 3. Worked shape

Pack three examples of lengths 20, 50, 30 into a block of $T = 100$:

```text
[  ex A (20)  ][      ex B (50)      ][   ex C (30)  ]   ->  length 100, 0 padding
 mask: A's own   mask: B's own          mask: C's own      (concatenate the masks)
 attn: A only    attn: B only           attn: C only       (block-diagonal)
```

Versus one-example-per-row padded to 100: rows would be 80%, 50%, 70% padding respectively — over half the compute wasted. Packing recovers essentially all of it.

<div class="callout warn"><p>The dangerous packing bug is a <strong>plain causal mask across a packed block</strong>: example C can attend all the way back to example A. Nothing crashes, the loss looks fine, and the model quietly learns to condition responses on unrelated preceding examples. Always reset attention (block-diagonal mask / position ids) at example boundaries, or verify your trainer does.</p></div>

## Exercise

You pack two examples into one block. Example 1 is `prompt=[a,b] response=[c]`; example 2 is `prompt=[d] response=[e,f]`. Using the delimiter-free token view (ignore chat tokens for this exercise), write the packed `input_ids` and the `labels` (with `-100` for masked positions) that `build_labels` would produce from the concatenated response mask.

*Hint:* concatenate the tokens, concatenate the masks `[0,0,1]` and `[0,1,1]`, then apply `build_labels`.

*Stronger hint:* labels keep the token id exactly where the mask is 1 and are `-100` everywhere else. The tokens themselves are unchanged by masking.

<details><summary>Solution</summary>

Concatenated tokens: `input_ids = [a, b, c, d, e, f]`.
Concatenated response mask: `[0, 0, 1] + [0, 1, 1] = [0, 0, 1, 0, 1, 1]`.
`build_labels` copies the id where the mask is 1, else writes `-100`:

```text
labels = [-100, -100, c, -100, e, f]
```

So the loss is computed only on `c` (example 1's response) and `e, f` (example 2's response) — each example's prompt is masked independently, which is exactly what packing must preserve. (In real packed data the chat delimiters are also present and masked as prompt, and attention is reset between the two examples.)

</details>

## Common mistakes

- **Averaging over $N$ instead of over the response count.** Dividing the masked-loss numerator by the sequence length instead of by $\sum_i m_i$ silently shrinks the loss and distorts gradients between short- and long-response examples.
- **Masking with `0` instead of a sentinel.** Setting ignored labels to token id `0` trains the model to predict token 0 on the prompt — the opposite of ignoring. Use `-100`.
- **Off-by-one in the shift.** Forgetting to shift, so `logits[i]` is scored against token $i$ (which it already saw) rather than token $i+1$. The loss collapses toward zero and the model learns nothing useful.
- **Packing with a plain causal mask.** Cross-example attention leakage (see the warning above).

## Check yourself

<details><summary>Why is the SFT loss averaged over the number of response tokens, not the full sequence length?</summary>

Because only the response positions contribute to the numerator (the rest are ignored). Dividing by the full length $N$ would systematically shrink the loss by the fraction of prompt tokens, and would weight examples by how much prompt they have rather than how much response — distorting the gradient. Averaging over $\sum_i m_i$ gives a proper mean over the positions we actually scored.

</details>

<details><summary>In the worked example, why is the masked loss (0.3876) larger than the full loss (0.3136)?</summary>

The prompt positions happened to be easy (confident, low loss). Averaging them in pulls the full-sequence loss down. The masked loss ignores them and averages only positions 3–4, so it reflects only the response difficulty. Neither is "wrong" — they measure different things; SFT wants the response-only number.

</details>

<details><summary>What is the failure mode of packing several examples into one block with an ordinary causal mask?</summary>

Later examples in the block attend to earlier, unrelated examples (cross-example leakage). Nothing errors, but the model learns to condition its response on preceding unrelated conversations, degrading quality. Fix: block-diagonal attention (reset attention/position ids at each example boundary).

</details>

<details><summary>Given <code>input_ids = [5, 6, 7]</code> and <code>response_mask = [0, 1, 1]</code>, what does <code>build_labels</code> return?</summary>

`[-100, 6, 7]` — position 0 is masked to the sentinel, positions 1 and 2 keep their token ids as targets.

</details>

## Next

We can now train on the right tokens efficiently. But full fine-tuning still updates *every* weight and keeps optimizer state for all of them. The next lesson makes SFT cheap: **LoRA** freezes the pretrained weights and trains a tiny low-rank update instead.

Continue to [14.3 · LoRA & QLoRA from scratch](lessons/module-14/lesson-03.md).
