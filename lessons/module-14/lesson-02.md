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

You now train on the right tokens. This part is about not paying for the wrong ones. Packing is a plain idea, but it has three ways to go wrong without any error message, so we will go slowly and do every step by hand once.

### 1. Intuition: a delivery truck that leaves mostly empty

Think of each training row as a delivery truck with room for $T$ boxes. Our block size is $T = 2048$ tokens, so every truck has 2048 slots, and the GPU charges us for the whole truck whether it is full or not.

Now look at what SFT data actually looks like. One example is "What is the capital of France?" plus a one-line answer, about 30 tokens. The next is a code review with a long reply, 1,500 tokens. Most are somewhere around a few hundred.

The simple way to batch them is one example per truck, with the empty space filled with padding. A 30-token example then rides alone in a 2048-slot truck: 2,018 of the slots are padding. The GPU still runs every layer on every one of those slots. It multiplies matrices full of pad embeddings, and then we throw the results away.

**Packing** fills the truck. We put several examples one after another in the same row until it is (almost) full. Same GPU, same step time, but almost every slot now carries a real token, so we get far more learning per second.

The idea is easy. The hard part is that the examples now sit next to each other, and the model must never notice that. Section 4 is about that.

### 2. How much do we actually waste? Real numbers

Words like "most of the compute is wasted" are easy to say, so let's measure. Below is a dataset of 10,000 SFT examples with lengths drawn from a log-normal distribution, which is the long-tailed shape real instruction data has: many short examples, a few very long ones. Examples longer than 2048 are truncated to 2048.

| Statistic | Value |
|---|---|
| Examples | 10,000 |
| Real tokens in total | 3,638,136 |
| Mean length | 364 tokens |
| Median length | 246 tokens |
| 90th percentile | 780 tokens |
| Longest | 2,048 tokens (truncated) |

Now four ways to put them into rows of $T = 2048$. "Useful" is the share of row positions that hold a real token, not padding:

| Strategy | Rows | Useful | What it means |
|---|---|---|---|
| One example per row, padded to $T$ | 10,000 | 17.8% | 82% of the GPU's work is on padding |
| Pad to the longest in each batch of 16 | (variable) | 28.4% | Better, but one long example per batch spoils it |
| Pack in dataset order, new row when the next one doesn't fit ("next-fit") | 2,115 | 84.0% | Simple and already about 4.7× better |
| Pack longest-first into the first row with room ("first-fit decreasing") | 1,777 | 99.97% | Almost perfect |

Two things to take from this table.

First, padding to the longest in the batch, the trick many people think solves the problem, barely helps with long-tailed data. In a batch of 16 random examples there is often one long one, and then the 15 short ones are padded up to its length.

Second, the last row is about as good as it can get. The total is 3,638,136 tokens and $3{,}638{,}136 / 2048 = 1776.4$, so no method can use fewer than **1,777** rows. First-fit decreasing hit exactly that. Compared with one example per row, we go from 10,000 rows to 1,777. That is **5.6× fewer forward and backward passes** for the same data, which means the same epoch runs about 5.6× faster (a bit less in practice, see section 7).

You can rerun these numbers yourself: the script is a dozen lines on top of `first_fit_decreasing` below, seeded with `torch.manual_seed(0)`.

### 3. Who shares a row? A small bin-packing problem

Deciding which examples go together is the classic **bin-packing** problem: put items of different sizes into as few bins of capacity $T$ as possible. Finding the true optimum is NP-hard, but a very simple rule is close enough in practice.

**First-fit decreasing (FFD):**

1. Sort the examples from longest to shortest.
2. Take them one at a time. Put each one in the first row that still has room.
3. If no row has room, start a new row.

Why longest first? Big items are the hard ones to place. If you place them first, the small ones later fill the gaps they leave. If you do it the other way round, small items scatter across rows and a big item at the end gets a row of its own, half empty.

Let's do it by hand. Three examples with lengths **5, 4, 3** and a tiny row length **$T = 8$**:

| Step | Example (length) | Row 0 free | Row 1 free | Decision |
|---|---|---|---|---|
| start | | | | no rows yet |
| 1 | ex 0 (5) | 8 → 3 | | no row exists, open row 0 |
| 2 | ex 1 (4) | 3 | 8 → 4 | 4 > 3, doesn't fit row 0, open row 1 |
| 3 | ex 2 (3) | 3 → 0 | 4 | fits row 0 exactly |

Result: row 0 = [ex 0, ex 2] (5 + 3 = 8, full), row 1 = [ex 1] (4 real + 4 padding). Total length is 12, and $\lceil 12 / 8 \rceil = 2$, so two rows is the best possible.

The example from earlier, lengths 20, 50 and 30 with $T = 100$, packs into a single row with no padding at all: 50 first, then 30, then 20.

TRL calls this strategy `bfd` ("best-fit decreasing"). Best-fit is a small change: instead of the *first* row with room, pick the row with the *least* room that still fits. Both give nearly the same results.

<div class="callout key"><p>We never cut an SFT example in two to fill a gap. In pretraining (<a href="#/lessons/module-04/lesson-03">04.3</a>) we simply concatenate everything and slice, because any stretch of text is a valid training window. An SFT example cut in half loses either its question or its answer, so it is useless. Too-long examples are truncated or dropped before packing. (TRL's <code>bfd_split</code> strategy, below, is the one exception: it splits long examples into separate chunks on purpose, accepting that trade.)</p></div>

### 4. Three things that break when examples share a row

Let's use real tokens. Three tiny examples, each a prompt followed by a response:

| Example | Tokens | Response mask |
|---|---|---|
| A | `11 12 13 14 15` | `0 0 0 1 1` (3 prompt, 2 response) |
| B | `21 22 23 24` | `0 0 1 1` (2 prompt, 2 response) |
| C | `31 32 33` | `0 1 1` (1 prompt, 2 response) |

With $T = 8$, FFD gives the same answer as before: row 0 = A + C, row 1 = B + padding. Here is row 0 with everything we need to keep for each position:

| position $t$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `input_ids` | 11 | 12 | 13 | 14 | 15 | 31 | 32 | 33 |
| `response_mask` | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 1 |
| `doc_ids` | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| `position_ids` | 0 | 1 | 2 | 3 | 4 | **0** | **1** | **2** |

`doc_ids` says which example owns each position (0 = the first example in this row, 1 = the second, −1 = padding). `position_ids` starts again at 0 when C begins. Row 1 looks like this:

| position $t$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `input_ids` | 21 | 22 | 23 | 24 | 0 | 0 | 0 | 0 |
| `response_mask` | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| `doc_ids` | 0 | 0 | 0 | 0 | −1 | −1 | −1 | −1 |
| `position_ids` | 0 | 1 | 2 | 3 | 0 | 0 | 0 | 0 |

Now the three things we must protect.

#### Problem 1: the labels

Remember the shift: the logits at position $t$ predict the token at $t+1$. So the label at $t$ is `input_ids[t+1]`, and we keep it only if that next token is a response token **of the same example**. Let's go through row 0 one position at a time ($-100$ means "ignored"):

| $t$ | sees token | next token | next is response? | same example? | label |
|---|---|---|---|---|---|
| 0 | 11 | 12 | no | yes | −100 |
| 1 | 12 | 13 | no | yes | −100 |
| 2 | 13 | 14 | **yes** | yes | **14** |
| 3 | 14 | 15 | **yes** | yes | **15** |
| 4 | 15 | 31 | no | **no**, boundary | −100 |
| 5 | 31 | 32 | **yes** | yes | **32** |
| 6 | 32 | 33 | **yes** | yes | **33** |
| 7 | 33 | (none) | | | −100 |

So `labels = [-100, -100, 14, 15, -100, 32, 33, -100]`. The model is trained on exactly the four response tokens, two from A and two from C.

Look closely at position 4. The model has just read the last token of A, and the next token in the row is the first token of C. If we kept that label, we would be teaching the model "after the end of one conversation comes the start of a random other one." Here the response mask already blocks it, because 31 is a prompt token. But if you train on all tokens, or a template puts a response token first, only the "same example?" check saves you. That is why `packed_labels` checks `doc_ids` too and does not rely on the response mask alone.

#### Problem 2: attention

This is the big one. With an ordinary causal mask, position 6 (token 32, inside C) can attend to every earlier position, including all of A. C's answer is now computed while "reading" an unrelated conversation. In training this is wrong. At inference time C will never have A in front of it, so the model trains on a situation it will never see.

The fix is a **block-diagonal causal mask**: position $i$ may look at position $j$ only if $j$ is not in the future *and* both belong to the same example. For row 0 (✓ = allowed, · = blocked):

```text
            key j →   0  1  2  3  4 | 5  6  7
                      A  A  A  A  A | C  C  C
query i  0  A         ✓  ·  ·  ·  · | ·  ·  ·
   ↓     1  A         ✓  ✓  ·  ·  · | ·  ·  ·
         2  A         ✓  ✓  ✓  ·  · | ·  ·  ·
         3  A         ✓  ✓  ✓  ✓  · | ·  ·  ·
         4  A         ✓  ✓  ✓  ✓  ✓ | ·  ·  ·
            ----------------------------------
         5  C         ·  ·  ·  ·  · | ✓  ·  ·
         6  C         ·  ·  ·  ·  · | ✓  ✓  ·
         7  C         ·  ·  ·  ·  · | ✓  ✓  ✓
```

Two small triangles on the diagonal, one per example, and nothing in the lower-left block. A plain causal mask would also fill that lower-left 3×5 block with ✓. Those 15 extra cells are the leak.

#### Problem 3: positions

Even with a perfect attention mask, there is a quieter problem. If positions just count 0, 1, 2, … 7 across the row, then token 31, the first token of C, gets position 5. On its own it would be at position 0. The model's position signal (learned position embeddings, or RoPE in modern models) is now wrong for every example except the first in each row. That is why `position_ids` restart at 0 for each example. With them, C at positions 5–7 of the row looks exactly like C at positions 0–2 of a row of its own.

<div class="callout key"><p>When all three are right, the packed row is not an approximation. Every example's output is <strong>identical</strong> to running it alone. <code>test_packed_attention_equals_separate_attention</code> in <code>code/tests/test_sft.py</code> checks exactly this, to 5 decimal places.</p></div>

### 5. Mathematics: the packed mask and labels

Let $d_t$ be the doc id at position $t$ ($-1$ for padding) and $m_t$ the response mask. The additive attention mask $M \in \mathbb{R}^{T \times T}$ that is added to the scores before softmax is

$$
M_{ij} = \begin{cases} 0 & \text{if } j \le i \text{ and } d_i = d_j \ne -1 \\ -\infty & \text{otherwise.} \end{cases}
$$

So the attention output at position $i$ is

$$
\text{out}_i = \sum_{j} \operatorname{softmax}_j\!\Big(\frac{q_i \cdot k_j}{\sqrt{d_h}} + M_{ij}\Big)\, v_j ,
$$

and because $e^{-\infty} = 0$, every key from another example gets weight exactly 0. The sum only runs over the same example's keys, which is the same sum you would compute with the example alone.

Small numbers: in row 0, take query $i = 6$ (token 32). Then $d_6 = 1$. For $j = 2$: $j \le 6$ but $d_2 = 0 \ne 1$, so $M_{6,2} = -\infty$, blocked. For $j = 5$: $5 \le 6$ and $d_5 = 1$, so $M_{6,5} = 0$, allowed. For $j = 7$: $7 > 6$, the future, so blocked.

One special case: a padding query has no example to look at, so its whole row would be $-\infty$, and softmax of all $-\infty$ is $0/0 = $ NaN. We let each padding position attend to itself only. Its output is garbage, but its label is $-100$, so the garbage never reaches the loss.

The labels, with $x_t$ the token at position $t$:

$$
\text{label}_t = \begin{cases} x_{t+1} & \text{if } m_{t+1} = 1 \text{ and } d_{t+1} = d_t \ne -1 \\ -100 & \text{otherwise (including } t = T-1\text{).} \end{cases}
$$

Check $t = 4$ in row 0: $d_5 = 1 \ne d_4 = 0$, so the label is $-100$, as in the table above.

### 6. Tensor shapes / dtype / device

For a batch of $R$ packed rows of length $T$:

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `input_ids` | `(R, T)` | `torch.long` | `pad_id` in the tail |
| `response_mask` | `(R, T)` | `torch.long` | 0 on prompt and padding |
| `doc_ids` | `(R, T)` | `torch.long` | −1 on padding |
| `position_ids` | `(R, T)` | `torch.long` | restarts at 0 per example |
| `packed_attention_mask(doc_ids)` | `(R, 1, T, T)` | `float32` | 0 / −inf; the 1 broadcasts over heads |
| `packed_labels(...)` | `(R, T)` | `torch.long` | aligned with `logits[:, t]` |

`pack_examples` builds everything on the CPU, in the data loader. Move the batch to the GPU with `.to(device)` like any other batch.

Notice the mask is $T \times T$ per row. At $T = 2048$ in float32 that is $2048^2 \times 4$ bytes $= 16$ MiB per row, and for a batch of 8 that is 128 MiB, only to say "these two examples don't talk." Section 7 shows how production code avoids building it at all.

### 7. From-scratch PyTorch

The full module is `code/src/llmre/sft/packing.py`. The three key pieces:

```python
def first_fit_decreasing(lengths, block_size):
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])  # longest first
    rows, free = [], []                     # free[r] = slots left in row r
    for i in order:
        for r, room in enumerate(free):
            if lengths[i] <= room:          # first row with room wins
                rows[r].append(i); free[r] -= lengths[i]
                break
        else:                               # no row had room: open a new one
            rows.append([i]); free.append(block_size - lengths[i])
    return rows

def packed_attention_mask(doc_ids):                       # (..., T)
    T = doc_ids.shape[-1]
    causal = torch.ones(T, T, dtype=torch.bool).tril()    # j <= i
    same = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2) # d_i == d_j, (..., T, T)
    real = (doc_ids != -1).unsqueeze(-1)                  # query is not padding
    allowed = (causal & same & real) | (torch.eye(T, dtype=torch.bool) & ~real)
    mask = torch.zeros(allowed.shape).masked_fill(~allowed, float("-inf"))
    return mask.unsqueeze(-3)                             # (..., 1, T, T)

def packed_labels(input_ids, response_mask, doc_ids, ignore_index=-100):
    labels = torch.full_like(input_ids, ignore_index)
    keep = ((response_mask[..., 1:] != 0)                 # next token is a response token
            & (doc_ids[..., 1:] == doc_ids[..., :-1])     # ...of the same example
            & (doc_ids[..., 1:] != -1))                   # ...and not padding
    labels[..., :-1] = torch.where(keep, input_ids[..., 1:],
                                   torch.full_like(input_ids[..., 1:], ignore_index))
    return labels
```

Try it on our toy examples:

```python
from llmre.sft.packing import pack_examples, packed_labels, packing_efficiency

A = ([11, 12, 13, 14, 15], [0, 0, 0, 1, 1])
B = ([21, 22, 23, 24],     [0, 0, 1, 1])
C = ([31, 32, 33],         [0, 1, 1])

p = pack_examples([A, B, C], block_size=8)
print(p["input_ids"])      # [[11,12,13,14,15,31,32,33], [21,22,23,24,0,0,0,0]]
print(p["position_ids"])   # [[0,1,2,3,4,0,1,2],         [0,1,2,3,0,0,0,0]]
print(packed_labels(p["input_ids"], p["response_mask"], p["doc_ids"]))
# [[-100,-100,14,15,-100,32,33,-100], [-100,23,24,-100,-100,-100,-100,-100]]
print(packing_efficiency(p["doc_ids"]))   # 0.75  (12 real tokens / 16 slots)
```

And the test that proves it works (shortened here; the full version is in `code/tests/test_sft.py`). It runs one attention layer over the packed row, then over A alone and C alone, and checks the outputs match. It also checks that a plain causal mask gives *different* outputs for C, so the test would catch the leak:

```python
packed = attend(ids, position_ids, packed_attention_mask(doc_ids)[0])
for example, (lo, hi) in [(A, (0, 5)), (C, (5, 8))]:
    alone = attend(example_ids, torch.arange(n), causal_mask[:n, :n])
    assert torch.allclose(packed[lo:hi], alone, atol=1e-5)    # identical

leaky = attend(ids, torch.arange(8), causal_mask)             # the bug
assert not torch.allclose(leaky[5:8], packed[5:8], atol=1e-3) # C really changed
```

Run them with `pytest tests/test_sft.py -k pack`.

### 8. What production code does, and what it costs

Our mask is correct but wasteful in two ways.

**Memory:** we build a $T \times T$ mask for every row (16 MiB per row at $T = 2048$, from section 6).

**Compute:** a mask doesn't skip work. The GPU still computes all $T^2$ scores and then sets most of them to $-\infty$. For our 10,000-example dataset, the dense mask over 1,777 rows computes **2.8× more** attention scores than the examples actually need. The dense count is $1777 \times 2048^2$, and what's needed is $\sum_i L_i^2$ over the examples.

Production kernels fix both by never building the mask. FlashAttention's "variable length" mode (`flash_attn_varlen_func`) takes all the examples flattened into one long sequence, plus a small list of where each example starts, called `cu_seqlens` ("cumulative sequence lengths"). For row 0:

```text
lengths     = [5, 3]
cu_seqlens  = [0, 5, 8]      # A is positions 0..4, C is 5..7
```

The kernel runs causal attention on each slice separately. Blocked cells are never computed, and no $T \times T$ tensor ever exists. For row 0 the causal-dense way computes $8 \cdot 9 / 2 = 36$ scores, while varlen computes only $5 \cdot 6/2 + 3 \cdot 4/2 = 15 + 6 = 21$. Note that MLP and projection layers cost the same either way: they work on each token alone, so packing saves them only the padding.

Here is how the Hugging Face stack does it (**publicly documented**, checked 2026-11-11 against the TRL `main` docs):

- `SFTConfig(packing=True, max_length=2048)` turns packing on. `max_length` is the row length $T$.
- `packing_strategy` defaults to `"bfd"` (best-fit decreasing, extra tokens of too-long examples are dropped). `"bfd_split"` splits long examples into chunks instead of dropping their tail. `"wrapped"` concatenates everything and cuts at fixed lengths. That is the pretraining approach, and the docs warn it can mix unrelated examples.
- `padding_free=True` flattens the batch into one sequence with no padding at all. The docs say it is supported only with FlashAttention 2 or 3, and that it is turned on automatically when you use `packing_strategy="bfd"`.
- Under the hood, the Hugging Face collator `DataCollatorWithFlattening` returns `position_ids` that restart at 0 for each example. FlashAttention 2 finds the example boundaries from those restarts and builds `cu_seqlens` from them. So in the HF stack, the `position_ids` from Problem 3 also solve Problem 2.

How much does it help in practice? The Hugging Face blog post [Improving Hugging Face Training Efficiency Through Packing with Flash Attention 2](https://huggingface.co/blog/packing-with-FA2) (August 2024) measured about **2× throughput** on FLAN and **1.4×** on OrcaMath for 7–8B models on 8×A100, with about 20% and 6% less peak memory, and the same validation loss. The gain depends on how uneven your lengths are: FLAN has many short examples, so it gains more.

<div class="callout warn"><p>The silent failure is packing with <strong>an attention backend that ignores the boundaries</strong>. For example, you turn <code>packing=True</code> on but load the model with the default eager or SDPA attention instead of FlashAttention. Depending on the library version, examples can then attend to each other across the row (TRL's docs only say padding-free needs FlashAttention; they don't spell out what happens without it, so treat this as <em>reasonable industry practice</em>, not a documented guarantee). Nothing crashes, the loss looks normal, and the model quietly learns to answer with an unrelated conversation in view. Before a long run, do the check our test does: take one batch, run the model on a packed row and on one of its examples alone, and compare the logits for that example. If they differ by more than float noise, your packing leaks.</p></div>

### 9. One more subtle thing: how examples are weighted

Packing also changes, a little, how much each example counts in the loss. Our `masked_cross_entropy` takes the mean over **all response tokens in the batch**. So an example with a 400-token answer pulls the gradient 100 times harder than an example with a 4-token answer. That is also true without packing, if you average over tokens across a padded batch. But many people imagine "one example = one vote", and that is never what a token-mean gives you.

Usually this is what you want, because long answers have more to learn. But if your data has many short, important answers ("Yes.", tool-call ids, a classification label), they can get drowned out. In that case, compute each example's mean loss first, using `doc_ids` to group the tokens, then average over examples. Either way, decide it on purpose and write it in your experiment notes, because two runs that differ only in this choice can give different models.

<div class="callout key"><p>Packing in one sentence: put several examples in one row to stop paying for padding, then make sure each example's <strong>labels</strong>, <strong>attention</strong> and <strong>positions</strong> behave exactly as if it were alone, and test that by comparing outputs, not by looking at the loss.</p></div>

<div class="callout key"><p><strong>Pretraining packing vs SFT packing: two differences, and one of them is changing.</strong></p>
<p><strong>1. Splitting.</strong> Pretraining cuts a long document across two windows; SFT never splits an example, because half an example has no question or no answer. This difference stays.</p>
<p><strong>2. Attention across the boundary.</strong> Plain pretraining packing lets a document attend to the previous, unrelated documents in its window, and relies on <code>&lt;|endoftext|&gt;</code> to tell the model to ignore them. SFT blocks that with a block-diagonal causal mask (section 4 above). <em>This difference is disappearing.</em> Meta's <a href="https://arxiv.org/abs/2407.21783">Llama 3</a> (2024) uses the same document mask in pretraining. They report it had limited impact on standard pretraining but was important for continued pretraining on very long sequences (publicly documented). <a href="https://arxiv.org/abs/2402.13991">Zhao et al. (ACL 2024)</a> found that blocking attention between documents improves pretraining results. The SFT way is becoming the standard way in pretraining too.</p></div>

## Exercise

You pack two examples into one block. Example 1 is `prompt=[a,b] response=[c]`; example 2 is `prompt=[d] response=[e,f]`. Using the delimiter-free token view (ignore chat tokens for this exercise), write the packed `input_ids` and the `labels` (with `-100` for masked positions) that `build_labels` would produce from the concatenated response mask.

<details><summary>Hint</summary>

Concatenate the tokens, concatenate the masks `[0,0,1]` and `[0,1,1]`, then apply `build_labels`.

</details>

<details><summary>Stronger hint</summary>

Labels keep the token id exactly where the mask is 1 and are `-100` everywhere else. The tokens themselves are unchanged by masking.

</details>

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
- **Forgetting to restart `position_ids`.** The attention mask is right, but every example after the first in a row gets the wrong positions, so it trains on positions it will never have at inference.
- **Keeping the label across a boundary.** The last token of one example is trained to predict the first token of the next. Check `doc_ids` when building labels, not only the response mask.
- **Packing in dataset order and calling it done.** Next-fit wastes 16% of the row in our example; sorting longest-first (FFD / BFD) gets it to almost zero for the cost of a sort.

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

<details><summary>You pack examples of lengths 6, 5, 3 and 2 into rows of <code>T = 8</code> with first-fit decreasing. Which examples end up together, and how full is each row?</summary>

Sorted: 6, 5, 3, 2. The 6 opens row 0 (2 left). The 5 doesn't fit, so it opens row 1 (3 left). The 3 doesn't fit row 0 but fits row 1 exactly. The 2 fits row 0 exactly. Result: row 0 = [6, 2], row 1 = [5, 3], both 100% full. Total 16 tokens, $16/8 = 2$ rows, so this is the best possible.

</details>

<details><summary>Your packed run's loss curve looks normal. How would you prove there is no cross-example attention?</summary>

Not from the loss: leakage doesn't show up there. Take one packed row, run the model on it, and run one of its examples alone (positions starting at 0). Compare the logits at that example's positions. With correct packing they match to float noise; if they differ, attention or positions leak across the boundary.

</details>

<details><summary>Given <code>input_ids = [5, 6, 7]</code> and <code>response_mask = [0, 1, 1]</code>, what does <code>build_labels</code> return?</summary>

`[-100, 6, 7]` — position 0 is masked to the sentinel, positions 1 and 2 keep their token ids as targets.

</details>

## Next

We can now train on the right tokens efficiently. But full fine-tuning still updates *every* weight and keeps optimizer state for all of them. The next lesson makes SFT cheap: **LoRA** freezes the pretrained weights and trains a tiny low-rank update instead.

Continue to [14.3 · LoRA & QLoRA from scratch](lessons/module-14/lesson-03.md).
