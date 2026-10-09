# 01.4 · The language-modeling objective & perplexity

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-01/lesson-03">01.3 · Entropy, cross-entropy, KL divergence</a> — cross-entropy against a one-hot target reducing to $-\log q(t)$. From <a href="#/lessons/module-01/lesson-02">01.2</a>, the chain rule and maximum likelihood. From <a href="#/lessons/module-01/lesson-01">01.1</a>, the categorical distribution and softmax as the map from scores to a distribution.</p>
<p><strong>You will learn:</strong> the exact objective a language model minimizes — the mean cross-entropy between the model's softmax distribution and the true next token — worked end to end on a 4-token vocabulary (logits → softmax → true-token probability → loss). Then <strong>perplexity</strong> $= \exp(\text{cross-entropy})$, why it reads as the "effective branching factor", and how to verify your by-hand loss against <code>torch.nn.functional.cross_entropy</code> and the from-scratch <code>llmre.evaluation.metrics</code> code.</p>
<p><strong>Why this matters for ML:</strong> this is the loss. Every pretraining run in this course, and every frontier LLM, is trained by minimizing exactly this number, and progress is reported as perplexity. Once you can compute it by hand on four tokens, the training loop in Module 7 and the evaluation in Module 13 are just this same computation at scale.</p>
</div>

## 1. Intuition: next-token prediction is classification over the vocabulary

A language model's job at each position is a **classification problem** with one class per vocabulary token. Given the prefix $x_{<i}$, it must pick the next token out of $V$ possibilities — a $V$-way classification, where $V$ might be 50257. From lesson 01.2, the model produces the conditional distribution $p(x_i \mid x_{<i})$; from lesson 01.3, we score that distribution with cross-entropy against the one true token. This lesson puts the two together into a single, concrete number.

The pipeline at one position is always the same four steps, which we now do by hand:

$$
\text{logits} \;\xrightarrow{\text{softmax}}\; \text{distribution } q \;\xrightarrow{\text{pick true token } t}\; q(t) \;\xrightarrow{-\log}\; \text{loss}.
$$

## 2. From logits to a distribution: softmax

The model's raw output at a position is a vector of $V$ real numbers called **logits** — unbounded scores, one per token, that do *not* form a probability distribution (they can be negative and do not sum to 1). To turn logits $\mathbf{z} = (z_1, \dots, z_V)$ into a categorical distribution $q$, we apply the **softmax** function:

$$
q_i = \mathrm{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_{j=1}^{V} \exp(z_j)}.
$$

Every symbol: $z_i$ is the logit for token $i$; $\exp(z_i)$ is always positive (fixing non-negativity); dividing by the sum over all tokens forces the outputs to sum to 1 (fixing normalization). Softmax is exactly the function that produces a valid categorical (lesson 01.1) from arbitrary scores, and larger logits get exponentially more probability.

### 2.1 Worked example: a 4-token vocabulary

Let the vocabulary be $V = 4$ and the model's logits at some position be

$$
\mathbf{z} = (2,\ 1,\ 0,\ -1).
$$

Exponentiate each: $\exp(2) = 7.389$, $\exp(1) = 2.718$, $\exp(0) = 1$, $\exp(-1) = 0.3679$. The normalizing sum is $7.389 + 2.718 + 1 + 0.3679 = 11.475$. Divide:

$$
q = \Big(\tfrac{7.389}{11.475},\ \tfrac{2.718}{11.475},\ \tfrac{1}{11.475},\ \tfrac{0.3679}{11.475}\Big) = (0.6439,\ 0.2369,\ 0.0871,\ 0.0321).
$$

These four numbers sum to 1 (a valid categorical), and the ordering follows the logits: token 0 has the highest logit (2) and the highest probability (0.6439), token 3 the lowest of both.

## 3. The per-token loss: negative log-probability of the true token

Suppose the token that *actually* comes next is **token 0**. From lesson 01.3, the true distribution is one-hot on index 0, and cross-entropy collapses to the negative log-probability of that true token:

$$
\ell = -\log q(t) = -\log q(0) = -\ln(0.6439) = 0.4402 \text{ nats}.
$$

That single scalar is the loss for this position. Read it as a penalty: the model gave the correct token probability $0.6439$, and it is charged $-\ln(0.6439) = 0.44$ nats for not being fully confident. If it had put probability 1 on token 0, the loss would be $-\ln(1) = 0$; if it had put nearly 0, the loss would rocket toward $+\infty$. Lower loss ⇔ more probability on the right token.

**Same prediction, different reality.** The loss depends on two things: what the model predicted ($q$) and which token actually came next ($t$). Keep the model's prediction exactly the same — $q = (0.6439,\ 0.2369,\ 0.0871,\ 0.0321)$ — and imagine a different training example where the text continues with **token 3** instead of token 0. The model has not changed; only the answer key has.

The recipe is identical: look up the probability the model gave to *whichever token actually occurred*, and take $-\ln$ of it. Now that token is 3, so

$$
\ell = -\ln q(3) = -\ln(0.0321) = 3.44 \text{ nats}.
$$

Why so much bigger? The model bet 64% on token 0 and only 3% on token 3. It was confidently wrong, so it pays about 8× more (3.44 vs 0.44). Here is the same prediction scored against each of the four possible true tokens:

| true next token $t$ | $q(t)$ | loss $-\ln q(t)$ | verdict |
|---|---|---|---|
| 0 | 0.6439 | 0.44 | model's top guess — small penalty |
| 1 | 0.2369 | 1.44 | second guess — moderate |
| 2 | 0.0871 | 2.44 | unlikely — large |
| 3 | 0.0321 | 3.44 | model's *least* likely — largest |

(The losses step by exactly 1 because the logits step by exactly 1: $-\ln q(t) = \log\sum_j e^{z_j} - z_t$, so each 1-unit drop in logit adds 1 nat of loss.)

The takeaway: the loss only ever looks at one entry of $q$ — the one for the token that really happened. The other three entries do not appear in the formula at all. What pushes them down during training is softmax: raising $q(t)$ forces the others to shrink, because they must sum to 1.

## 4. The full objective: mean cross-entropy over all positions

One position gives one loss. A language model is trained over a whole corpus, so the **objective** is the average of the per-token cross-entropies across every position it predicts. For a sequence of tokens $x_1, \dots, x_N$ (or a batch of them), with $q_i$ the model's predicted distribution at position $i$ and $t_i$ the true token there:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \big(-\log q_i(t_i)\big) = -\frac{1}{N} \sum_{i=1}^{N} \log q_i(t_i).
$$

This is the mean negative log-likelihood — and by the chain rule of lesson 01.2, $\sum_i \log q_i(t_i)$ is the log-probability of the entire sequence. So minimizing $\mathcal{L}$ is maximizing the likelihood of the training text, averaged per token. We average (rather than sum) so the loss is comparable across sequences of different lengths and batch sizes; it is a per-token quantity in nats.

### 4.1 Worked example: the loss of `the cat sat`

Reuse the sentence and model $\theta_B$ from lesson 01.2. The model makes three predictions, one per position; at each one we look up the probability it gave the true token and take $-\ln$ of it (exactly the section 3 recipe, repeated):

| position $i$ | context | true token $t_i$ | $q_i(t_i)$ | $-\ln q_i(t_i)$ |
|---|---|---|---|---|
| 1 | (empty) | `the` | 0.2 | 1.6094 |
| 2 | `the` | `cat` | 0.5 | 0.6931 |
| 3 | `the cat` | `sat` | 0.6 | 0.5108 |

Sum the last column, then divide by $N = 3$:

$$
\mathcal{L} = \frac{1.6094 + 0.6931 + 0.5108}{3} = \frac{2.8133}{3} = 0.9378 \text{ nats per token}.
$$

**Check the link to likelihood.** In lesson 01.2 the likelihood of this sentence under $\theta_B$ was $0.2 \times 0.5 \times 0.6 = 0.06$. Its log is $\ln 0.06 = -2.8134$ — the negative of the sum above (up to rounding). So the sum of per-token losses *is* the negative log-likelihood of the whole sentence, and $\mathcal{L}$ is that divided by the number of tokens.

**Why the mean and not the sum.** Suppose the text were twice as long, `the cat sat the cat sat`, and the model were equally good at every position (same three losses, repeated). The sum doubles to $5.6266$, but the mean stays $5.6266 / 6 = 0.9378$. The sum would say "the model got worse" just because the text got longer; the mean correctly says "same quality per token". The same applies across a batch: 8 sequences or 512, the mean is on the same scale.

**Compare with the worse model.** Model $\theta_A$ from lesson 01.2 gave $0.2, 0.1, 0.3$: losses $1.6094 + 2.3026 + 1.2040 = 5.1160$, mean $1.7053$ nats. Higher loss, lower likelihood ($0.006$) — the same ranking MLE gave, now expressed as a number to minimize.

<div class="callout key"><p>The language-modeling loss is the mean, over all predicted positions, of $-\log q_i(t_i)$: the negative log-probability the model assigned to each true next token. Minimizing it = maximum likelihood = pushing the model's softmax distribution toward the observed tokens. This one formula is what every pretraining run optimizes.</p></div>

## 5. Tensor shapes: what this looks like in a real model

In code the logits are not a single vector but a batched tensor. Naming the axes:

- **Logits:** shape `(B, T, V)`, dtype float32 (often bf16 in training), on the GPU. `B` sequences, each of `T` positions, each position a length-`V` logit vector.
- **Targets:** shape `(B, T)`, dtype `int64` (`long`), same device. Each entry is the true next-token index in `[0, V)`.
- The loss reshapes logits to `(B*T, V)` and targets to `(B*T,)`, computes one cross-entropy per row, and averages to a **scalar** (shape `()`).

The `V` axis is always the one softmax normalizes over and the one the true index selects from. The reshape from `(B, T, V)` to `(N, V)` with `N = B*T` is just "treat every (sequence, position) as an independent classification example"; the chain rule guarantees averaging them is the right thing.

### 5.1 Worked example: `B = 2`, `T = 2`, `V = 3`

A toy vocabulary of 3 tokens (ids 0, 1, 2) and a batch of 2 sequences, each 3 tokens long:

```text
tokens = [[1, 0, 2],     # sequence 0
          [2, 1, 0]]     # sequence 1        shape (2, 3)
```

**Where the targets come from.** The model reads the first `T = 2` tokens of each sequence and, at each position, predicts the *next* one. So the inputs are the tokens without the last column, and the targets are the tokens shifted left by one:

```text
inputs  = tokens[:, :-1] = [[1, 0],
                            [2, 1]]          shape (2, 2) = (B, T)
targets = tokens[:, 1:]  = [[0, 2],
                            [1, 0]]          shape (2, 2) = (B, T), int64
```

Read row 0: after seeing token `1` the true next token is `0`; after seeing `1, 0` the true next token is `2`.

**The logits.** For each of the $B \times T = 4$ positions the model outputs $V = 3$ logits, so the logits tensor has shape `(2, 2, 3)`:

```text
logits = [[[2, 1, 0],    # seq 0, pos 0
           [0, 0, 0]],   # seq 0, pos 1
          [[1, 2, 0],    # seq 1, pos 0
           [0, 1, 2]]]   # seq 1, pos 1        shape (2, 2, 3) = (B, T, V)
```

**Where the logits come from.** The model reads the inputs and, at every input position, outputs $V = 3$ numbers: one score per vocabulary token, meaning "how much I think this token comes next". In a real model a neural network computes them (embeddings + transformer, built later in the course). Here they are hand-picked so the arithmetic is clean. For now the model is a black box: inputs in, logits out.

**Why the shape is `(2, 2, 3)`.** 2 sequences (`B`), each with 2 input positions (`T`), each position getting 3 scores (`V`). Every entry of `inputs` gets its own row of 3 logits, in the same (seq, pos) slot.

**Reading each row.** Pair each row with the input the model saw and the target it should have predicted:

- **seq 0, pos 0** — model has seen `[1]`. Logits `[2, 1, 0]`: it likes token 0 most, then 1, then 2. Truth is `0` → good guess.
- **seq 0, pos 1** — model has seen `[1, 0]`. Logits `[0, 0, 0]`: all equal, no preference. Truth is `2` → the model shrugs.
- **seq 1, pos 0** — model has seen `[2]`. Logits `[1, 2, 0]`: likes token 1 most. Truth is `1` → good guess.
- **seq 1, pos 1** — model has seen `[2, 1]`. Logits `[0, 1, 2]`: likes token 2 most. Truth is `0` → confidently wrong.

**Logits are not probabilities yet.** They can be negative, they do not sum to 1, and only their *differences* matter: `[2, 1, 0]` and `[12, 11, 10]` give the same result. Softmax (next step) turns them into probabilities.

**At each position, the model never sees that position's own target.** The whole input row `[1, 0]` goes into the model at once, but each position may only look at itself and what is to its left:

- At pos 0 the model may use only `[1]`. The `0` sitting next to it in the input is exactly the answer for pos 0, so it must not look at it.
- At pos 1 the model may use `[1, 0]`. Here the `0` is fine: it is the past (it was pos 0's target, now it is context). The answer for pos 1 is `2`, which is not in the inputs at all.

So every token except the first plays two roles: the *target* for the position before it, and *context* for the positions after it. The only token the model never sees anywhere is the last one (`2`), because nothing comes after it to use it as context. There is nothing special about any position: predicting the second token, the third token, or the thousandth is the same job — "given everything to my left, score what comes next" — and each one contributes one row to the loss.

**What enforces this: the causal mask.** Inside the model, every position computes its output by mixing information from positions of the input. The causal mask is a rule on that mixing: position $i$ may only mix in positions $0 \dots i$, never anything to its right. For the input `[1, 0]`:

```text
          reads pos 0   reads pos 1
pos 0        yes            no
pos 1        yes            yes
```

It is a lower-triangular yes/no matrix. The "no" slots get their attention score set to $-\infty$ before softmax, so after softmax their weight is exactly 0 and no information flows from the future. Without the mask, pos 0 could simply copy the `0` to its right, get near-zero loss, and learn nothing.

The payoff: one forward pass over a row of length `T` trains `T` predictions at once, each honestly using only its past. Without the mask you would have to run the model `T` separate times on growing prefixes to get the same thing. You implement the mask yourself in [05.2 · Q/K/V & scaled dot-product attention](lessons/module-05/lesson-02.md).

**Reshape.** Flatten the first two axes: logits become `(4, 3)`, targets become `(4,)`. Now each row is one independent classification example. For each row: softmax over the 3 logits, pick the entry at the target index, take $-\ln$.

| row | (seq, pos) | logits | softmax $q$ | target $t$ | $q(t)$ | $-\ln q(t)$ |
|---|---|---|---|---|---|---|
| 0 | (0, 0) | $(2, 1, 0)$ | $(0.6652, 0.2447, 0.0900)$ | 0 | 0.6652 | 0.4076 |
| 1 | (0, 1) | $(0, 0, 0)$ | $(0.3333, 0.3333, 0.3333)$ | 2 | 0.3333 | 1.0986 |
| 2 | (1, 0) | $(1, 2, 0)$ | $(0.2447, 0.6652, 0.0900)$ | 1 | 0.6652 | 0.4076 |
| 3 | (1, 1) | $(0, 1, 2)$ | $(0.0900, 0.2447, 0.6652)$ | 0 | 0.0900 | 2.4076 |

Row 1 is a model with no opinion (all logits equal → uniform), paying $\ln 3 = 1.0986$. Row 3 put its highest logit on token 2 but the truth was token 0 — the expensive mistake.

**Average to a scalar.**

$$
\mathcal{L} = \frac{0.4076 + 1.0986 + 0.4076 + 2.4076}{4} = \frac{4.3214}{4} = 1.0804 \text{ nats}.
$$

Shape trail: `(2, 2, 3)` logits → `(4, 3)` → softmax `(4, 3)` → pick target entry `(4,)` → mean `()`. The `V` axis disappears at the "pick" step (the target index selects one of its 3 entries); the `B*T` axis disappears at the mean.

PyTorch agrees:

```python
import torch, torch.nn.functional as F
logits  = torch.tensor([[[2., 1, 0], [0, 0, 0]],
                        [[1., 2, 0], [0, 1, 2]]])      # (2, 2, 3) float32
targets = torch.tensor([[0, 2], [1, 0]])               # (2, 2)    int64
loss = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))
print(loss)                                            # tensor(1.0804)
```

### 5.2 What that one line does, step by step

`F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))` receives logits of shape `(4, 3)` and targets of shape `(4,)`. **Each of the 4 rows is one prediction for one next token**: 3 scores, one per vocabulary token, and one target index saying which of the 3 actually came next. Behind the scenes it does four things, and each is something from this module.

**Step 1 — pairing.** `reshape` flattens both tensors in the same order: (seq 0, pos 0), (seq 0, pos 1), (seq 1, pos 0), (seq 1, pos 1). So logits row `r` and `targets[r]` still describe the same slot. `F.cross_entropy` always treats the second axis of the logits (size 3 = `V`) as the vocabulary, and the target as an index into it.

```text
z = [[2, 1, 0],        t = [0,      <- row 0: true token is 0
     [0, 0, 0],             2,      <- row 1: true token is 2
     [1, 2, 0],             1,      <- row 2: true token is 1
     [0, 1, 2]]             0]      <- row 3: true token is 0
    shape (4, 3)           shape (4,)
```

**Step 2 — logits → log-probabilities, per row** (section 2 + the underflow lesson in 01.3). For each row it computes

$$
\log q_j = z_j - \log\big(e^{z_0} + e^{z_1} + e^{z_2}\big),
$$

which is softmax followed by log, done in one step so a tiny $q$ never rounds to 0.

*Why that line is "softmax, then log".* Start from softmax and take the log of both sides, using two rules: $\log(a/b) = \log a - \log b$, and $\log(e^{x}) = x$.

$$
q_j = \frac{e^{z_j}}{e^{z_0} + e^{z_1} + e^{z_2}}
\;\;\Longrightarrow\;\;
\log q_j = \log\big(e^{z_j}\big) - \log\big(e^{z_0} + e^{z_1} + e^{z_2}\big)
= z_j - \log\big(e^{z_0} + e^{z_1} + e^{z_2}\big).
$$

The division inside softmax became a subtraction, and $\log$ undid the $e^{z_j}$ on top, leaving the raw logit $z_j$. Check it both ways on row 0, $z = (2, 1, 0)$, for $j = 0$:

- Softmax first, then log: $q_0 = \dfrac{e^2}{e^2 + e^1 + e^0} = \dfrac{7.389}{11.107} = 0.6652$, and $\log 0.6652 = -0.4076$.
- The one-step formula: $z_0 - \log(11.107) = 2 - 2.4076 = -0.4076$.

Same number. The one-step version never forms the fraction $q_0$ itself, so even when $q_0$ would be something like $10^{-50}$ (which rounds to 0 in float32, and $\log 0 = -\infty$), the subtraction still gives a finite answer like $-115$. Section 6.1 derives this again with the max-shift added.

The logits are *scores*, not probabilities; this step is where they become (log-)probabilities. Row 0: $\log(e^2 + e^1 + e^0) = \log(7.389 + 2.718 + 1) = 2.4076$, so $\log q = (2 - 2.4076,\ 1 - 2.4076,\ 0 - 2.4076) = (-0.4076,\ -1.4076,\ -2.4076)$. All four rows:

```text
log q = [[-0.4076, -1.4076, -2.4076],
         [-1.0986, -1.0986, -1.0986],
         [-1.4076, -0.4076, -2.4076],
         [-2.4076, -1.4076, -0.4076]]     shape (4, 3)
```

**Step 3 — pick the true token and negate** (01.3 section 6 + section 3 here). The full cross-entropy is $-\sum_x p(x)\log q(x)$ over all 3 tokens. The target is one index, which means $p$ is one-hot. For row 0 the target is 0, so $p = (1, 0, 0)$:

$$
\ell_0 = -\big(1 \cdot \log q_0 + 0 \cdot \log q_1 + 0 \cdot \log q_2\big) = -1 \cdot \log q_0 = -(-0.4076) = 0.4076.
$$

The 1 is a *multiplication* by the one-hot weight, and the two other tokens are multiplied by 0 and vanish. That is why PyTorch takes an integer index instead of a one-hot vector: it just reads the one entry that survives, `log_q[r, t[r]]`, and negates it. For the four rows that gives `[0.4076, 1.0986, 0.4076, 2.4076]` — the last column of the table above. Shape `(4, 3)` → `(4,)`: the vocabulary axis is gone.

**Step 4 — average** (section 4). The mean of the 4 numbers is $4.3214 / 4 = 1.0804$, a scalar (shape `()`). The sum before dividing, $4.3214$, is the negative log-likelihood of all four predictions together (01.2: log of a product = sum of logs); dividing by 4 makes it "per token".

So the one line is the whole chain of this module: logits → softmax (in log form) → cross-entropy against a one-hot target, which collapses to $-\log q(\text{true token})$ → mean over every position → the maximum-likelihood objective.

The same four steps written out by hand reproduce PyTorch exactly:

```python
z = logits.reshape(-1, 3)                              # (4, 3)  step 1
t = targets.reshape(-1)                                # (4,)    step 1
log_q = z - torch.logsumexp(z, dim=1, keepdim=True)    # (4, 3)  step 2: log-softmax per row
losses = -log_q[torch.arange(4), t]                    # (4,)    step 3: pick true token, negate
print(losses)                                          # tensor([0.4076, 1.0986, 0.4076, 2.4076])
print(losses.mean())                                   # tensor(1.0804)  step 4 == F.cross_entropy
```

`log_q[torch.arange(4), t]` is "row 0 column t[0], row 1 column t[1], ..." — one entry per row. Written with an explicit one-hot it is `-(F.one_hot(t, 3) * log_q).sum(dim=1)`, which gives the same four numbers, only wastefully multiplying by zeros.

## 6. From-scratch: the loss in `llmre.evaluation.metrics`

This module owns the evaluation code at `code/src/llmre/evaluation/metrics.py`. It implements cross-entropy from scratch — via a numerically stable log-softmax — rather than calling the framework, so you can see every step. The core is:

```python
def log_softmax(logits):                 # (N, V) -> (N, V) log-probabilities
    m = logits.max(dim=-1, keepdim=True).values   # per-row max, (N, 1)
    shifted = logits - m                          # subtract max: stability
    lse = shifted.exp().sum(dim=-1, keepdim=True).log()   # logsumexp, (N, 1)
    return shifted - lse                          # log q, rows exp-sum to 1

def cross_entropy(logits, targets):       # (N, V), (N,) long -> scalar (nats)
    logp = log_softmax(logits)
    rows = torch.arange(logits.shape[0], device=logits.device)
    true_logp = logp[rows, targets]       # gather -log q(t) per row, (N,)
    return -true_logp.mean()

def perplexity(logits, targets):
    return cross_entropy(logits, targets).exp()
```

### 6.1 Why these four lines compute *exactly* log-softmax

The code for `log_softmax` never writes "divide by the sum", and it subtracts a max that appears nowhere in the definition of softmax. So why is it the same thing? This section derives it from the definition, one identity at a time, so that you could reconstruct the four lines yourself. The chain is always: **definition → algebra → transformed equation → code**.

Everything happens one row at a time. Take one row of logits $z_1, \dots, z_V$ ($V$ = vocabulary size).

#### Step 1 — the definition of softmax, and of log-softmax

Softmax turns the row of scores into probabilities:

$$
q_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}.
$$

It does two things:

1. **exponentiate** every logit ($e^{z_i}$ is always positive, so every score becomes a positive number);
2. **normalize**: divide each exponential by the sum of all of them, so the results add up to 1.

Log-softmax is nothing more than the logarithm of that probability:

$$
\log q_i = \log\left(\frac{e^{z_i}}{\sum_j e^{z_j}}\right).
$$

Now apply two identities.

The log of a fraction is the log of the top minus the log of the bottom: $\log\frac{a}{b} = \log a - \log b$. With $a = e^{z_i}$ and $b = \sum_j e^{z_j}$:

$$
\log q_i = \log\big(e^{z_i}\big) - \log\Big(\sum_j e^{z_j}\Big).
$$

The log undoes the exponential: $\log(e^{x}) = x$. So $\log(e^{z_i}) = z_i$:

$$
\boxed{\ \log q_i = z_i - \log\Big(\sum_j e^{z_j}\Big)\ }
$$

This is the fundamental equation. The division in softmax has become a *subtraction* in log space. The subtracted term has a name: $\log\sum_j e^{z_j}$ is called **logsumexp** — literally "exponentiate, sum, take the log". So:

$$
\boxed{\ \log\operatorname{softmax}(z) = z - \operatorname{logsumexp}(z)\ }
$$

Read it as: "the log-probability of token $i$ is its score, minus one number shared by the whole row". That shared number is the log of the normalization denominator. Subtracting it is how you normalize in log space. This is why the code can end with `return <logits> - <logsumexp>` and never divide.

Numbers, row $z = [2, 1, 0]$: $e^2 + e^1 + e^0 = 7.389 + 2.718 + 1 = 11.107$, and $\log 11.107 = 2.4076$. So $\log q = [2, 1, 0] - 2.4076 = [-0.4076,\ -1.4076,\ -2.4076]$.

If that equation is already correct, why does the code subtract a max first? Because of floating point, not math — and Step 2 proves that doing so changes nothing.

#### Step 2 — subtracting a constant does not change softmax (a proof, not a trick)

**Claim.** For *any* constant $c$ (the same number subtracted from every logit in the row):

$$
\operatorname{softmax}(z - c) = \operatorname{softmax}(z).
$$

**Proof.** Write the softmax of the shifted row, entry $i$:

$$
\frac{e^{z_i - c}}{\sum_j e^{z_j - c}}.
$$

Use $e^{a-b} = e^{a}\,e^{-b}$ on the top and on every term of the bottom:

$$
= \frac{e^{z_i}\,e^{-c}}{\sum_j e^{z_j}\,e^{-c}}.
$$

$e^{-c}$ is the same in every term of the sum (it does not depend on $j$), so pull it out of the sum:

$$
= \frac{e^{z_i}\,e^{-c}}{e^{-c}\sum_j e^{z_j}}.
$$

Now $e^{-c}$ appears once on top and once on the bottom. It is a positive number, so it cancels:

$$
= \frac{e^{z_i}}{\sum_j e^{z_j}} = q_i. \qquad\blacksquare
$$

The probabilities are *identical*, digit for digit in exact arithmetic. Subtracting a constant does not make softmax more correct, and it does not approximate it. It changes the *representation* of the calculation — which numbers pass through `exp` — while leaving the mathematical result exactly unchanged. Intuitively: softmax only cares about the *differences* between scores. $[2, 1, 0]$ and $[12, 11, 10]$ and $[0, -1, -2]$ all have the same differences, so they give the same probabilities.

**Why choose $c = m = \max_j z_j$ specifically?** Any $c$ is valid; the max is the one that makes the computation safe. If $m$ is the largest logit in the row, then for every $i$

$$
z_i \le m \;\;\Longrightarrow\;\; z_i - m \le 0 \;\;\Longrightarrow\;\; e^{z_i - m} \le e^0 = 1.
$$

So after the shift, the largest value is exactly $0$ (its exponential is exactly $1$) and all others are negative (their exponentials are between 0 and 1). `exp` never sees a large positive number. Without the shift, a logit of $z = 90$ gives $e^{90} \approx 1.2 \times 10^{39}$, which is larger than float32 can hold ($\approx 3.4 \times 10^{38}$) and becomes `inf`; then `inf / inf = nan` and training is ruined. With the shift, that same logit becomes $90 - 90 = 0$ and $e^0 = 1$. Also, the sum $\sum_j e^{z_j - m}$ always contains at least one term equal to 1, so it is $\ge 1$ and its log is never $-\infty$.

In code: `m = logits.max(dim=-1, keepdim=True).values` is $m$, and `shifted = logits - m` is $z' = z - m$.

#### Step 3 — the log-softmax formula still holds after the shift

Step 2 was about softmax. The code works in log space, so derive the log version too. Define

$$
m = \max_j z_j, \qquad z'_i = z_i - m \quad\text{(the shifted logits)}.
$$

Start from the normalization term in the fundamental equation, $\sum_j e^{z_j}$. Every original logit is its shifted version plus $m$: $z_j = z'_j + m$. Substitute:

$$
\sum_j e^{z_j} = \sum_j e^{z'_j + m}.
$$

Use $e^{a+b} = e^{a}e^{b}$, and pull the common factor $e^{m}$ out of the sum:

$$
= \sum_j e^{z'_j}\,e^{m} = e^{m}\sum_j e^{z'_j}.
$$

Take the log of both sides, and use $\log(ab) = \log a + \log b$ and then $\log e^{m} = m$:

$$
\log\Big(\sum_j e^{z_j}\Big) = \log\Big(e^{m}\sum_j e^{z'_j}\Big) = m + \log\Big(\sum_j e^{z'_j}\Big).
$$

In words: the logsumexp of the original row equals the max plus the logsumexp of the shifted row. Now substitute this, and $z_i = z'_i + m$, into the fundamental equation $\log q_i = z_i - \log\sum_j e^{z_j}$:

$$
\log q_i = (z'_i + m) - \Big[m + \log\Big(\sum_j e^{z'_j}\Big)\Big].
$$

Remove the brackets: $+m$ and $-m$ cancel.

$$
\boxed{\ \log q_i = z'_i - \log\Big(\sum_j e^{z'_j}\Big)\ }
$$

This is the *same* log-softmax formula as in Step 1, with $z$ replaced by $z'$ everywhere. It is not a different formula or an approximation: it is the original one after algebraic rearrangement. Written in one line:

$$
\boxed{\ \log\operatorname{softmax}(z) = (z - m) - \log\Big(\sum_j e^{z_j - m}\Big)\ }
$$

and now every piece maps to one line of code:

| math | code | shape |
|---|---|---|
| $m = \max_j z_j$ | `m = logits.max(dim=-1, keepdim=True).values` | `(N, 1)` |
| $z'_i = z_i - m$ | `shifted = logits - m` | `(N, V)` |
| $e^{z'_j}$ | `shifted.exp()` | `(N, V)` |
| $\sum_j e^{z'_j}$ | `.sum(dim=-1, keepdim=True)` | `(N, 1)` |
| $\log\sum_j e^{z'_j}$ | `.log()` → `lse` | `(N, 1)` |
| $z'_i - \log\sum_j e^{z'_j}$ | `return shifted - lse` | `(N, V)` |

`lse` is the log of the normalization denominator of the shifted row, so `shifted - lse` is exactly "normalize, in log space". `keepdim=True` keeps `m` and `lse` as columns of shape `(N, 1)` so that the subtraction broadcasts: each row's own `m` and `lse` are subtracted from every entry of that row.

#### Step 4 — the full numerical example, both ways

Row $z = [2, 1, 0]$.

**Ordinary softmax, straight from the definition:**

$$
q = \frac{[e^2,\ e^1,\ e^0]}{e^2 + e^1 + e^0} = \frac{[7.389,\ 2.718,\ 1]}{11.107} = [0.6652,\ 0.2447,\ 0.0900].
$$

**What the code does.** The max is $m = 2$. Shift: $z' = [2-2,\ 1-2,\ 0-2] = [0,\ -1,\ -2]$. Exponentiate: $[e^0,\ e^{-1},\ e^{-2}] = [1,\ 0.3679,\ 0.1353]$. Normalize:

$$
\frac{[1,\ e^{-1},\ e^{-2}]}{1 + e^{-1} + e^{-2}} = \frac{[1,\ 0.3679,\ 0.1353]}{1.5032} = [0.6652,\ 0.2447,\ 0.0900].
$$

Same numbers. And algebraically it *must* be: multiply the top and the bottom of that fraction by $e^2$ (multiplying top and bottom by the same positive number leaves a fraction unchanged), using $e^2 \cdot e^{k} = e^{2+k}$:

$$
\frac{[1,\ e^{-1},\ e^{-2}] \cdot e^2}{(1 + e^{-1} + e^{-2}) \cdot e^2} = \frac{[e^2,\ e^1,\ e^0]}{e^2 + e^1 + e^0}.
$$

That is exactly the original softmax. Check with numbers: $1.5032 \times e^2 = 1.5032 \times 7.389 = 11.107$ — the original denominator.

**The log values the code returns:**

$$
\text{shifted} - \text{lse} = [0,\ -1,\ -2] - \log(1 + e^{-1} + e^{-2}) = [0,\ -1,\ -2] - \log 1.5032 = [0,\ -1,\ -2] - 0.4076 = [-0.4076,\ -1.4076,\ -2.4076].
$$

This matches Step 1's un-shifted answer, $[2, 1, 0] - 2.4076$. The two routes differ only in where the 2 lives: Step 3 says $\log 11.107 = 2 + \log 1.5032$, i.e. $2.4076 = 2 + 0.4076$. The unshifted route subtracts $2.4076$ from $[2, 1, 0]$; the shifted route first subtracts the $2$ (the max), then the remaining $0.4076$ (the `lse`).

And exponentiating the returned values gives back the exact softmax probabilities:

$$
[e^{-0.4076},\ e^{-1.4076},\ e^{-2.4076}] = [0.6652,\ 0.2447,\ 0.0900], \qquad 0.6652 + 0.2447 + 0.0900 = 1.
$$

#### Step 5 — why compute log-softmax directly instead of `log(softmax(z))`

The loss needs $\log q$, not $q$. One could compute `softmax(z)` and then take `.log()`. The problem is the middle step: it builds the probabilities themselves, and a very small probability can round to exactly `0.0` in float32 (for example $e^{-200} \approx 10^{-87}$, below float32's smallest value). Then `log(0.0) = -inf` and the loss is infinite. The direct formula $z' - \operatorname{logsumexp}(z')$ never constructs $q$ at all: it only subtracts two ordinary-sized numbers. For that same token it returns about $-200$, a perfectly finite log-probability. Same math, but no tiny intermediate. (This is the log-space rule from 01.2 and the underflow case from 01.3, in code form. The same max-subtraction reappears in attention, Module 5, and FlashAttention, Module 8.)

#### Mental model

- **Softmax:** exponentiate the logits, then normalize them (divide by their sum).
- **Log-softmax:** the same normalization, done in log space — divide becomes subtract: $\log q_i = z_i - \log\sum_j e^{z_j}$.
- **Subtract the max:** shift every logit in the row by the same constant. Softmax is exactly invariant to this shift, because $e^{-c}$ cancels between top and bottom.
- **Why the max, not another constant?** It makes the largest shifted value 0 and all others negative, so every exponential is at most 1 — no overflow — and the sum is at least 1 — no $\log 0$.
- **Why `shifted - lse`?** Because, after the shift, $\log q_i = z'_i - \log\sum_j e^{z'_j}$, and `lse` is exactly $\log\sum_j e^{z'_j}$.
- **Therefore** the four lines are softmax/log-softmax *exactly*, by algebra — not an approximation, and not a different formula.

### 6.2 Tracing both functions with numbers

Run the same 4-row example from section 5.1 through the code, one line at a time. Input: logits `(4, 3)`, targets `(4,)`.

```text
logits = [[2, 1, 0],        targets = [0, 2, 1, 0]
          [0, 0, 0],
          [1, 2, 0],
          [0, 1, 2]]
```

**`log_softmax(logits)`**

```python
m = logits.max(dim=-1, keepdim=True).values
# largest logit in each row, shape (4, 1):
# [[2],
#  [0],
#  [2],
#  [2]]

shifted = logits - m
# subtract each row's max from that row; the max entry becomes 0, the rest negative, (4, 3):
# [[ 0, -1, -2],      row 0: [2,1,0] - 2
#  [ 0,  0,  0],      row 1: [0,0,0] - 0
#  [-1,  0, -2],      row 2: [1,2,0] - 2
#  [-2, -1,  0]]      row 3: [0,1,2] - 2

shifted.exp()
# e^0 = 1, e^-1 = 0.3679, e^-2 = 0.1353; nothing is ever larger than 1, (4, 3):
# [[1.0000, 0.3679, 0.1353],
#  [1.0000, 1.0000, 1.0000],
#  [0.3679, 1.0000, 0.1353],
#  [0.1353, 0.3679, 1.0000]]

.sum(dim=-1, keepdim=True)
# add across each row, (4, 1):
# [[1.5032],          1 + 0.3679 + 0.1353
#  [3.0000],          1 + 1 + 1
#  [1.5032],
#  [1.5032]]

lse = ... .log()
# natural log of each row sum, (4, 1):
# [[0.4076],          ln 1.5032
#  [1.0986],          ln 3
#  [0.4076],
#  [0.4076]]

return shifted - lse
# subtract each row's lse from that row -> log-probabilities, (4, 3):
# [[-0.4076, -1.4076, -2.4076],     row 0: [0,-1,-2] - 0.4076
#  [-1.0986, -1.0986, -1.0986],     row 1: [0, 0, 0] - 1.0986
#  [-1.4076, -0.4076, -2.4076],
#  [-2.4076, -1.4076, -0.4076]]
```

Check row 0 against section 5.2: there we computed $\log q = z - \log(e^2 + e^1 + e^0) = [2,1,0] - 2.4076 = [-0.4076, -1.4076, -2.4076]$. Same answer. Here `lse` is only $0.4076$, not $2.4076$, because it was computed on the *shifted* logits — the missing 2 is exactly the max $m$ we subtracted first: $2 + 0.4076 = 2.4076$. The shift moves the 2 out of the exponent and back in by subtraction; the result is identical. Sanity check on the output: $e^{-0.4076} + e^{-1.4076} + e^{-2.4076} = 0.6652 + 0.2447 + 0.0900 = 1$ — each row is a valid distribution.

**`cross_entropy(logits, targets)`**

```python
logp = log_softmax(logits)
# the (4, 3) matrix just computed above

rows = torch.arange(logits.shape[0], device=logits.device)
# [0, 1, 2, 3]  -- one index per row, shape (4,)

true_logp = logp[rows, targets]
# pair rows with targets: (0,0), (1,2), (2,1), (3,0) -> pick one entry per row, (4,):
#   logp[0, 0] = -0.4076      row 0, true token 0
#   logp[1, 2] = -1.0986      row 1, true token 2
#   logp[2, 1] = -0.4076      row 2, true token 1
#   logp[3, 0] = -2.4076      row 3, true token 0
# -> [-0.4076, -1.0986, -0.4076, -2.4076]

return -true_logp.mean()
# mean = (-0.4076 - 1.0986 - 0.4076 - 2.4076) / 4 = -4.3214 / 4 = -1.0804
# negate -> 1.0804, shape ()   == F.cross_entropy on the same input
```

The indexing `logp[rows, targets]` is the "multiply by the one-hot and keep the one surviving term" step from section 5.2, done as a direct lookup. The shapes go `(4, 3)` → `(4,)` → `()`: the vocabulary axis disappears at the lookup, the position axis at the mean.

### 6.3 Verify the by-hand loss against PyTorch

Our worked loss was $0.4402$ nats. Confirm it three ways — by hand, via `llmre`, and via PyTorch's fused kernel:

```python
import torch, math
import torch.nn.functional as F
from llmre.evaluation.metrics import cross_entropy, perplexity

logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])   # (N=1, V=4)
target = torch.tensor([0])                         # true token index 0

print(cross_entropy(logits, target).item())        # 0.44018969... (our code)
print(F.cross_entropy(logits, target).item())      # 0.44018969... (PyTorch, identical)
print(-math.log(0.6439))                            # 0.4402...     (by hand)
```

All three agree. `torch.nn.functional.cross_entropy` takes the *logits* (not probabilities) and the integer target, fuses log-softmax and the gather into one kernel, and by default returns the mean over the batch in nats — exactly what our from-scratch `cross_entropy` computes. The unit test in `code/tests/test_metrics.py` asserts this equality to `1e-5` on a random `(8, 10)` batch, and that perplexity of uniform logits over `V` equals `V`.

<div class="callout pt"><p><strong>Feed logits, not probabilities, to <code>F.cross_entropy</code>.</strong> It applies log-softmax internally. If you softmax first and pass probabilities, you soft-max twice and get a wrong, too-flat loss — a very common bug. The target argument is a <code>long</code> tensor of class indices of shape <code>(N,)</code>, not a one-hot matrix.</p></div>

## 7. Perplexity: the effective branching factor

Cross-entropy in nats is the number optimizers minimize, but it is not intuitive to read. **Perplexity** makes it interpretable by exponentiating:

$$
\text{perplexity} = \exp(\mathcal{L}) = \exp\!\Big(-\frac{1}{N}\sum_{i=1}^N \log q_i(t_i)\Big).
$$

Because we work in nats, the exponential is the natural $\exp$ (if the loss were in bits you would use $2^{\mathcal{L}}$). Perplexity is interpreted as the **effective branching factor**: the model is, on average, as uncertain about the next token as if it were choosing uniformly among this many equally likely options. A perplexity of 20 means "as confused as a fair 20-sided die at each step". Lower is better; a perfect model has perplexity 1 (it always knows the next token), and the worst a sane model does is $V$ (pure uniform guessing).

### 7.1 Worked example: uniform over 4 gives perplexity 4

Take the extreme case of a model that has learned nothing: uniform logits over $V = 4$ tokens, so $q = (0.25, 0.25, 0.25, 0.25)$ regardless of the true token. The loss is

$$
\mathcal{L} = -\log(0.25) = \ln 4 = 1.386 \text{ nats},
$$

and the perplexity is

$$
\text{perplexity} = \exp(1.386) = 4.0.
$$

Exactly the vocabulary size — a uniform model over 4 tokens is "as uncertain as a fair 4-sided die", branching factor 4. This is the ceiling: any model that has learned anything scores below $V$. Contrast our earlier confident prediction, loss $0.4402$, whose perplexity is $\exp(0.4402) = 1.553$ — an effective branching factor of about 1.5, i.e. the model has narrowed 4 options down to roughly "1.5 live choices". Verify:

```python
import torch, math
from llmre.evaluation.metrics import cross_entropy, perplexity

# uniform over V=4: loss = ln 4, perplexity = 4
uni = torch.zeros(1, 4)                     # equal logits -> uniform softmax
print(cross_entropy(uni, torch.tensor([0])).item())  # 1.3862943611198906  (= ln 4)
print(perplexity(uni, torch.tensor([0])).item())     # 4.0

# the confident case from section 3
logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
print(perplexity(logits, torch.tensor([0])).item())  # 1.5530017927759188
```

<div class="callout key"><p>Perplexity $= \exp(\text{cross-entropy in nats})$. Read it as the effective number of equally likely tokens the model is choosing among: 1 = perfect, $V$ = uniform ignorance. It carries the same information as the loss but on a scale humans can reason about, which is why papers and leaderboards report it.</p></div>

## 8. Under the hood: cost of the loss at scale

At real vocabulary sizes the loss layer is not free. For logits of shape `(B, T, V)`:

- **Memory.** The logits tensor holds $B \cdot T \cdot V$ floats. With $B = 8$, $T = 1024$, $V = 50257$ in float32 that is about $8 \cdot 1024 \cdot 50257 \cdot 4 \approx 1.6\ \text{GB}$ for a single tensor — often the largest activation in the whole forward pass. This is why the vocabulary projection and the loss are a real memory concern, addressed later with techniques like fused cross-entropy kernels.
- **Compute.** Softmax and the log are $O(B \cdot T \cdot V)$ elementwise-ish work; the dominant cost is usually the preceding matrix multiply that produces the logits (the `(C → V)` output projection), which is $O(B \cdot T \cdot C \cdot V)$. Details below.
- **Gradient.** The gradient of cross-entropy with respect to the logits has an elegant closed form, $\partial \mathcal{L} / \partial z_i = q_i - p_i$ — the predicted distribution minus the one-hot target. For our worked example the gradient on the true token 0 is $0.6439 - 1 = -0.3561$ (push that logit up) and on token 3 it is $0.0321 - 0 = +0.0321$ (push it down). We derive this backward pass by hand in [Module 2](lessons/module-02/lesson-03.md); for now, note the loss and its gradient are both simple functions of the softmax output.

### 8.1 Compute, step by step

Until now the logits were given to us. In a real model they are produced by the last layer, and that layer is where most of the loss-side compute goes. Two pieces of work happen at every position:

**(1) The output projection (makes the logits).** The network ends each position with a *hidden vector* $h$ of $C$ numbers (its summary of the context so far; $C$ is the model width, 768 for GPT-2 small). A weight matrix $W$ of shape `(C, V)` turns it into $V$ logits: $z = hW$, i.e. $z_j = \sum_{k=1}^{C} h_k W_{kj}$. Each logit is a dot product of length $C$: $C$ multiplications and about $C$ additions, which we count as $2C$ FLOPs (floating-point operations). There are $V$ logits per position, so $2CV$ FLOPs per position, and for the whole batch:

$$
\text{FLOPs}_{\text{projection}} = 2 \cdot B \cdot T \cdot C \cdot V.
$$

**(2) Softmax + loss (scores the logits).** Per logit: subtract the max, one `exp`, add into the sum, subtract `lse` — a handful of operations, independent of $C$. Call it about 5 FLOPs per logit (the exact constant does not matter; what matters is that there is no factor of $C$):

$$
\text{FLOPs}_{\text{softmax+loss}} \approx 5 \cdot B \cdot T \cdot V.
$$

The ratio between them is $\frac{2BTCV}{5BTV} = \frac{2C}{5}$. The projection costs about $0.4\,C$ times more than the softmax, so the wider the model, the more the matmul dominates.

**Toy example: one position, $C = 2$, $V = 3$.** Take hidden vector $h = [1, 2]$ and

$$
W = \begin{bmatrix} 0 & -1 & 2 \\ 1 & 1 & -1 \end{bmatrix} \quad (C \times V = 2 \times 3).
$$

Each logit is $h$ dotted with one column of $W$:

$$
z_0 = 1\cdot 0 + 2\cdot 1 = 2, \qquad z_1 = 1\cdot(-1) + 2\cdot 1 = 1, \qquad z_2 = 1\cdot 2 + 2\cdot(-1) = 0.
$$

That is $z = [2, 1, 0]$, the row used throughout this lesson. Count the work: each logit took 2 multiplications + 1 addition; by the $2C$ convention that is $2 \cdot 2 \cdot 3 = 12$ FLOPs for the projection. Then softmax + loss on 3 logits: $\approx 5 \cdot 3 = 15$ FLOPs. At this toy size the two costs are about equal, because $C = 2$ is tiny ($2C/5 = 0.8$).

**Same count at GPT-2 small scale.** $B = 8$, $T = 1024$, $C = 768$, $V = 50257$, so $B \cdot T = 8192$ positions.

$$
\text{projection: } 2 \cdot 8192 \cdot 768 \cdot 50257 \approx 6.3 \times 10^{11}\ \text{FLOPs}
$$

$$
\text{softmax+loss: } 5 \cdot 8192 \cdot 50257 \approx 2.1 \times 10^{9}\ \text{FLOPs}
$$

The ratio is $2 \cdot 768 / 5 \approx 307$: the projection does about 300× more arithmetic than the softmax and loss combined. For scale, a common rule of thumb (REASONABLE INDUSTRY PRACTICE, derived in Module 7) puts a whole forward pass at about $2 \times (\text{parameters}) \times (\text{tokens})$ FLOPs, which for GPT-2 small's 124M parameters is $2 \cdot 124\text{M} \cdot 8192 \approx 2.0 \times 10^{12}$. The output projection alone is about $6.3 / 20 \approx 30\%$ of that, because $V = 50257$ is so large.

**So why worry about the softmax at all, if it is 300× cheaper?** Because its cost is not arithmetic but memory traffic. The softmax reads and writes the whole $B \cdot T \cdot V$ logits tensor (1.6 GB in the memory bullet above) several times (find the max, exponentiate and sum, subtract), doing only a few FLOPs per number it moves. A GPU can do far more arithmetic per second than it can move bytes, so this step is *memory-bound*: its runtime is set by bytes moved, not FLOPs. The matmul does hundreds of FLOPs per number it loads, so it is *compute-bound*. That distinction — compute-bound vs memory-bound — is the central idea behind the fused cross-entropy kernels mentioned above and behind FlashAttention (Module 8).

## Research connection

<div class="callout paper"><p>The entire GPT-2 result rests on this one objective. The paper's training signal is exactly the sum of log next-token probabilities you built here — see the reading guide for <a href="#/papers/index">GPT-2 ("Language Models are Unsupervised Multitask Learners", Radford et al. 2019)</a>, whose key equation is maximize $\sum_i \log p(x_i \mid x_{<i})$. Every later model in the <a href="#/papers/index">paper curriculum</a> — GPT-3, LLaMA, DeepSeek — pretrains on this same cross-entropy loss; what changes is scale, data, and architecture, not the objective.</p></div>

## Debugging exercise

A student computes their validation loss and reports a suspiciously *low* number, then a colleague says the perplexity looks impossibly good. Here is the snippet. Find the bug.

```python
import torch
import torch.nn.functional as F

logits = model(inputs)                 # (B, T, V)
probs  = F.softmax(logits, dim=-1)     # (B, T, V)
loss   = F.cross_entropy(
    probs.view(-1, probs.size(-1)),    # (B*T, V)
    targets.view(-1),                  # (B*T,)
)
```

<details><summary>Optional hint</summary>

What does `F.cross_entropy` expect as its first argument — logits or probabilities?

</details>

<details><summary>Stronger hint</summary>

`F.cross_entropy` applies log-softmax internally. The code passes `probs`, which has already been through softmax once.

</details>

<details><summary>Solution</summary>

`F.cross_entropy` expects **raw logits** and applies softmax (log-softmax) itself. The snippet softmaxes first and passes `probs`, so the values are softmaxed *twice*. The double softmax squashes the distribution toward uniform, which distorts the loss (and, depending on the target, can make it misleadingly small or large). The fix is to pass the logits directly:

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),   # raw logits, NOT probs
    targets.view(-1),
)
```

Rule of thumb: never call `softmax` before `cross_entropy`. If you genuinely need probabilities elsewhere, compute them separately and still feed *logits* to the loss.

</details>

## Check yourself

<details><summary>Why do we feed logits, not probabilities, to <code>F.cross_entropy</code> (and to <code>llmre</code>'s <code>cross_entropy</code>)?</summary>

Because both apply log-softmax internally, for numerical stability (subtract-the-max) and to avoid `log(0)`. Passing probabilities means the softmax is applied twice, flattening the distribution and producing a wrong loss. The function wants raw logits of shape `(N, V)` and integer targets of shape `(N,)`.

</details>

<details><summary>A model has cross-entropy loss 1.386 nats on a 4-token vocabulary. What is its perplexity, and what does that say about the model?</summary>

Perplexity $= \exp(1.386) = 4.0$, which equals the vocabulary size $V = 4$. That is the uniform ceiling: the model is as uncertain as a fair 4-sided die and has effectively learned nothing about which token comes next. Any useful model would score below 4.

</details>

<details><summary>For logits <code>(B, T, V)</code> and targets <code>(B, T)</code>, what reshaping happens before the loss, and what is the shape of the final loss?</summary>

Logits reshape to `(B*T, V)` and targets to `(B*T,)`, turning every (sequence, position) pair into one $V$-way classification example. The loss computes $-\log q_i(t_i)$ per row and averages, giving a scalar of shape `()` in nats.

</details>

<details><summary>The gradient of the loss with respect to the logits is $q_i - p_i$. For our example (softmax $(0.6439, 0.2369, 0.0871, 0.0321)$, true token 0), what is the gradient on the true token's logit, and which way does an update push it?</summary>

$q_0 - p_0 = 0.6439 - 1 = -0.3561$. A gradient-descent step subtracts (a multiple of) the gradient, so it *increases* the true token's logit — pushing more probability onto the correct token, exactly as maximum likelihood should. The wrong tokens have positive gradient $q_i - 0 = q_i$ and get pushed down.

</details>

<details><summary>Why average the per-token losses instead of summing them?</summary>

So the reported loss is a per-token quantity, comparable across sequences of different length and across different batch sizes. Summing would make longer sequences and bigger batches look "worse" purely because they have more terms. Averaging yields a nats-per-token number, which is also what perplexity exponentiates into an effective branching factor independent of length.

</details>

## Practice

A separate page of practice problems for lessons 01.3 and 01.4 (math by hand, softmax and log-softmax, shapes, PyTorch, debugging), with full worked solutions at the bottom: [01.3–01.4 · Practice set](lessons/module-01/practice-03-04.md).

## Next

You have now built the complete language-modeling objective — softmax over the vocabulary, cross-entropy against the true token, averaged over positions — verified it by hand and against PyTorch and the `llmre` code, and learned to read it as perplexity. This closes Module 1: you can state exactly what a language model computes and exactly how its predictions are scored. Module 2 opens the other half of training — how the gradient of this loss flows back through the network via backpropagation, starting with derivatives and the chain rule of calculus.

Continue to [02.1 · Derivatives, partials, chain rule, gradients](lessons/module-02/lesson-01.md).
