# 01.1 · Random variables, expectation, variance

<div class="prereq">
<p><strong>Prerequisites:</strong> tensors — shape, dtype, device — from <a href="#/lessons/module-00/lesson-01">00.1 · Tensors: shape, dtype, device</a>. No probability background is assumed; this lesson starts from zero.</p>
<p><strong>You will learn:</strong> what a discrete random variable is, why its probabilities must sum to 1, how to compute the expectation $E[X]$ and the variance $\mathrm{Var}(X)$ (both by hand on a fair die and in PyTorch), and then the one distribution that matters most for the rest of this course — the <strong>categorical distribution</strong> over a vocabulary, which is exactly what a language model outputs at every position — including how to sample from it with <code>torch.multinomial</code>.</p>
<p><strong>Why this matters for ML:</strong> a language model is, at heart, a machine that reads some text and emits a categorical distribution over the next token. Everything in Module 1 — expectation, likelihood, entropy, cross-entropy, perplexity — is machinery for measuring how good that distribution is. If you understand this one object, the training loss in lesson 01.4 will read as an obvious consequence rather than a formula to memorize.</p>
</div>

## 1. Intuition: a random variable is a number you do not know yet

As a programmer you already have a mental model for this, you just call it something else. A **random variable** is like the return value of a function that has some randomness inside it — `roll_die()` returns a number in `1..6`, but you cannot say *which* until you call it. The random variable is the *description* of that return value together with how often each outcome happens.

We write random variables with capital letters, $X$, and their concrete outcomes with lowercase, $x$. So "$X$ is the result of rolling a fair die" means $X$ can take the values $x \in \{1, 2, 3, 4, 5, 6\}$, and "$X = 4$" is one particular outcome.

This lesson only deals with **discrete** random variables — ones whose outcomes come from a finite (or countable) list. That is exactly the case we care about: a token is one of a finite vocabulary of, say, 50257 possibilities. Continuous random variables (heights, temperatures) need calculus and we do not need them here.

## 2. The PMF: how probability is spread across outcomes

A discrete random variable is fully described by its **probability mass function** (PMF), written $p(x)$: for each possible outcome $x$, it gives the probability that $X$ takes that value.

$$
p(x) = P(X = x).
$$

Two rules make something a valid PMF, and they are non-negotiable:

1. **Non-negativity:** $p(x) \ge 0$ for every $x$. A probability is never negative.
2. **Normalization:** the probabilities over *all* outcomes sum to exactly 1:

$$
\sum_{x} p(x) = 1.
$$

The normalization rule is the one to burn into memory. It says "something must happen" — the total probability mass is 1, and the PMF only ever *distributes* that fixed budget of 1 across the outcomes. When we get to a language model's output, this is the reason its predicted probabilities over the vocabulary must sum to 1, and the reason softmax (which enforces exactly that) is the function that produces them.

For a fair six-sided die, every face is equally likely, so

$$
p(x) = \frac{1}{6} \quad \text{for each } x \in \{1,2,3,4,5,6\}.
$$

Check normalization: $6 \times \tfrac{1}{6} = 1$. Good.

<div class="callout key"><p>A discrete random variable = a list of outcomes plus a PMF $p(x)$ that is non-negative and sums to 1. The PMF spreads a total probability budget of exactly 1 across the outcomes. This "sums to 1" constraint is the thread that runs through the entire module.</p></div>

## 3. Expectation: the long-run average outcome

The **expectation** (or **expected value**, or **mean**) $E[X]$ is the average value of $X$ you would see if you sampled it a huge number of times and averaged the results. It is a weighted average of the outcomes, where each outcome is weighted by its probability:

$$
E[X] = \sum_x x \cdot p(x).
$$

Naming every symbol: $x$ ranges over the possible outcomes; $p(x)$ is the probability of that outcome; $x \cdot p(x)$ is that outcome's contribution to the average, scaled by how often it happens; and we sum those contributions over all outcomes.

### 3.1 Worked example: the fair die

For the fair die, $p(x) = \tfrac16$ for all six faces, so

$$
E[X] = \sum_{x=1}^{6} x \cdot \tfrac16 = \tfrac16 (1 + 2 + 3 + 4 + 5 + 6) = \tfrac16 \cdot 21 = \frac{21}{6} = 3.5.
$$

Notice the expectation is $3.5$ — a value the die can never actually show. That is normal: the expectation is the *balance point* of the distribution, not necessarily an achievable outcome. If you roll many times and average, the running average settles toward $3.5$.

## 4. Variance: how spread out the outcomes are

The expectation tells you the center; the **variance** tells you how far outcomes typically sit from that center. It is the expected value of the squared distance from the mean. Writing $\mu = E[X]$ for the mean,

$$
\mathrm{Var}(X) = E\big[(X - \mu)^2\big] = \sum_x (x - \mu)^2\, p(x).
$$

We square the distance $(x - \mu)$ for two reasons: squaring makes every term non-negative (so distances above and below the mean do not cancel), and it penalizes large deviations more than small ones. The variance is therefore always $\ge 0$, and it is $0$ only when $X$ is a constant (no spread at all).

There is an equivalent, often handier formula. Expanding the square gives

$$
\mathrm{Var}(X) = E[X^2] - (E[X])^2,
$$

where $E[X^2] = \sum_x x^2 p(x)$ is the expectation of the *squared* variable. "Mean of the square minus square of the mean." We will use this form for the die because it is less arithmetic.

### 4.1 Worked example: the fair die

First $E[X^2]$:

$$
E[X^2] = \tfrac16(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) = \tfrac16(1 + 4 + 9 + 16 + 25 + 36) = \frac{91}{6} \approx 15.1667.
$$

We already have $E[X] = 3.5$, so $(E[X])^2 = 12.25$. Therefore

$$
\mathrm{Var}(X) = \frac{91}{6} - 12.25 = 15.1667 - 12.25 = 2.9167 = \frac{35}{12}.
$$

So $\mathrm{Var}(X) = \tfrac{35}{12} \approx 2.9167$. The **standard deviation** — the square root of the variance, back in the original units — is $\sqrt{2.9167} \approx 1.708$, a typical distance of a roll from the mean of $3.5$.

### 4.2 Verify it in Python

Every number above must survive contact with a computer. Here it is with `py` (torch 2.14 CPU):

```python
import torch

x = torch.arange(1, 7, dtype=torch.float64)          # outcomes 1..6, shape (6,)
p = torch.full((6,), 1/6, dtype=torch.float64)       # PMF, shape (6,)

assert torch.isclose(p.sum(), torch.tensor(1.0, dtype=torch.float64))  # sums to 1

E   = (x * p).sum()                                  # E[X] = sum x p(x)
Var = ((x - E)**2 * p).sum()                         # sum (x-E)^2 p(x)

print(E.item(), Var.item())      # 3.5   2.916666666666666
print(35/12)                     # 2.9166666666666665
```

The printed variance `2.9166...` matches $\tfrac{35}{12}$ to machine precision. We used `float64` here purely so the printout is clean; real models run in `float32` or lower, which we will confront in lesson 01.2 when tiny probabilities start to underflow.

## 5. The categorical distribution — a language model's output

Now the payoff. The single most important distribution in this course is the **categorical distribution**: a random variable with $K$ distinct outcomes ("categories"), each with its own probability. It is the natural generalization of the die — a die is just a categorical with $K = 6$ equal probabilities — except the probabilities need not be equal.

Concretely, a categorical over $K$ outcomes is given by a probability vector

$$
\mathbf{p} = (p_1, p_2, \dots, p_K), \qquad p_i \ge 0, \qquad \sum_{i=1}^{K} p_i = 1,
$$

where $p_i$ is the probability of category $i$.

<div class="callout key"><p>A language model, at each position, outputs exactly a categorical distribution over the vocabulary: a vector of $V$ non-negative numbers that sum to 1, giving the probability of each possible next token. "Predicting the next token" <em>is</em> "producing this categorical". Everything else in this module is about scoring that prediction.</p></div>

Here the categories are the tokens of the vocabulary (words or word-pieces — Module 4 builds the real tokenizer), and $V$ (the vocabulary size) plays the role of $K$. When you hear "the model puts 40% probability on the token `the`", that 40% is one entry $p_i$ of this categorical.

### 5.1 A length-5 categorical in PyTorch

Let us make a small categorical over 5 outcomes — think of a toy vocabulary of 5 tokens — and treat it as a tensor.

```python
import torch

probs = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1])   # shape (5,), dtype float32
print(probs.shape)     # torch.Size([5])
print(probs.sum())     # tensor(1.)   <- a valid categorical: sums to 1
```

What this tensor represents: `probs` is a **1-D tensor of shape `(5,)`, dtype `float32`, on the CPU**. Entry `probs[i]` is the probability of token `i`. The whole vector *is* the categorical distribution — token 2 (0-indexed) is the most likely at probability `0.4`, tokens 0 and 4 the least likely at `0.1`. In a real model this vector would have shape `(V,)` with `V = 50257`, and it would be produced by a softmax over the model's raw scores (logits); we build that in lesson 01.4.

### 5.2 Sampling from a categorical with `torch.multinomial`

Producing the distribution is one thing; drawing an actual token from it is another. To *generate* text, a model samples a concrete token from its categorical — outcome $i$ chosen with probability $p_i$. PyTorch does this with `torch.multinomial`:

```python
import torch

probs = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1])   # (5,) categorical
torch.manual_seed(0)                               # make sampling reproducible

# draw 10 independent samples, each an index in [0, 5)
samples = torch.multinomial(probs, num_samples=10, replacement=True)
print(samples.tolist())   # [4, 3, 2, 4, 2, 3, 1, 2, 2, 1]
```

`torch.multinomial(probs, num_samples=10, replacement=True)` returns a **`(10,)` `int64` tensor** of category indices, each drawn independently with probability `probs[i]`. `replacement=True` means each draw is independent and the same category can be picked repeatedly (exactly what next-token sampling wants — token 2 can be chosen many times). With the seed fixed to 0 you will get the indices above every run; index 2 (the highest-probability token) shows up most often, index 0 and 4 rarely, which is the distribution shape reflected in the samples.

<div class="callout pt"><p><code>torch.multinomial</code> requires the input to be non-negative, and it does <em>not</em> require it to sum to 1 — it normalizes internally, treating the values as unnormalized weights. In an LM we usually hand it a proper probability vector from softmax anyway. Passing negative values raises an error; passing all-zeros raises an error too, since there is nothing to sample.</p></div>

### 5.3 Tensor-shape summary

Keep the shapes straight, because this exact object reappears in every later module:

- A single categorical: shape `(V,)`, dtype float, sums to 1 along its one axis.
- A model over a sequence of `T` positions in a batch of `B` sequences: shape `(B, T, V)`, and it sums to 1 along the **last** axis (`dim=-1`) — one categorical per (sequence, position). The `V` axis is always the one that "sums to 1".
- Sampling one token per position: `torch.multinomial` operates on a `(N, V)` matrix and returns `(N,)` indices, so you typically reshape `(B, T, V)` to `(B*T, V)` first.

## Common mistakes

- **Forgetting the distribution must sum to 1.** If you build a "probability" vector that sums to `0.98` or `1.3`, every downstream quantity (expectation, entropy, loss) is wrong. When in doubt, `assert torch.isclose(p.sum(), torch.tensor(1.0))`.
- **Confusing logits with probabilities.** A model's raw output (logits) can be any real numbers, positive or negative, and does *not* sum to 1. Only after softmax do you have a categorical. Sampling from logits directly with `multinomial` is a bug (it will error on negatives, or silently misbehave).
- **Expecting `E[X]` to be an achievable outcome.** $3.5$ is not a die face. The mean is a balance point, not a value the variable takes.
- **Using the biased vs unbiased variance formula.** The $\sum (x-\mu)^2 p(x)$ here is the *population* variance of a known distribution — there is no "$n-1$ correction". That correction belongs to *estimating* variance from a data sample, a different task.

## Exercise

Build a **biased** categorical over a 3-token vocabulary with probabilities $\mathbf{p} = (0.2, 0.5, 0.3)$, then compute its expectation $E[X]$ treating the token indices $\{0, 1, 2\}$ as the outcome values, and its variance — first by hand, then verify in Python.

<details><summary>Optional hint</summary>

Use $E[X] = \sum_i i \cdot p_i$ with outcomes $i \in \{0,1,2\}$, then $\mathrm{Var}(X) = E[X^2] - (E[X])^2$ with $E[X^2] = \sum_i i^2 p_i$.

</details>

<details><summary>Stronger hint</summary>

$E[X] = 0(0.2) + 1(0.5) + 2(0.3)$. For $E[X^2]$ use $0^2, 1^2, 2^2$ against the same probabilities. Then subtract the square of the mean.

</details>

<details><summary>Solution</summary>

$E[X] = 0(0.2) + 1(0.5) + 2(0.3) = 0 + 0.5 + 0.6 = 1.1$.

$E[X^2] = 0(0.2) + 1(0.5) + 4(0.3) = 0 + 0.5 + 1.2 = 1.7$.

$\mathrm{Var}(X) = 1.7 - 1.1^2 = 1.7 - 1.21 = 0.49$, so the standard deviation is $0.7$.

Verify:

```python
import torch
i = torch.tensor([0., 1., 2.], dtype=torch.float64)
p = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
E   = (i * p).sum()
Var = (i**2 * p).sum() - E**2
print(E.item(), Var.item())   # 1.1  0.48999999999999994
```

The tiny `0.4899...` instead of `0.49` is ordinary floating-point rounding, not an error.

</details>

## Check yourself

<details><summary>Why must the entries of a categorical distribution sum to exactly 1?</summary>

Because the outcomes are exhaustive and mutually exclusive: exactly one of them happens on each draw. The total probability mass is 1 and the PMF only distributes that fixed budget. For a language model this is why its predicted next-token probabilities over the whole vocabulary must sum to 1 — and why softmax, which guarantees that, is used to produce them.

</details>

<details><summary>A model outputs a next-token distribution of shape <code>(B, T, V)</code>. Along which axis does it sum to 1, and what does one such "row" represent?</summary>

Along the last axis, `dim=-1` (the `V` axis). Each length-`V` vector — one per (batch element `b`, position `t`) — is a categorical distribution over the vocabulary: the model's predicted probabilities for the token at that position. There are `B * T` such distributions in the tensor.

</details>

<details><summary>You have logits (raw scores) from a model. Can you pass them straight to <code>torch.multinomial</code>?</summary>

No. Logits can be negative and do not sum to 1; `torch.multinomial` errors on negative weights. You must first convert logits to a valid non-negative distribution with softmax, then sample. (`multinomial` does not require the input to sum to 1 — it renormalizes — but it does require non-negativity.)

</details>

<details><summary>For the fair die, is the standard deviation larger or smaller than the mean, and roughly what is it?</summary>

Smaller. The variance is $35/12 \approx 2.9167$, so the standard deviation is $\sqrt{2.9167} \approx 1.71$, well below the mean of $3.5$. A typical roll lands about $1.7$ away from $3.5$.

</details>

## Next

You can now describe a discrete random variable, verify its PMF sums to 1, compute its expectation and variance, and — most importantly — recognize the categorical distribution as the object a language model emits at every position and sample from it. Next we ask how the probabilities of *many* tokens combine into the probability of a whole sentence, which is where the language-modeling factorization comes from.

Continue to [01.2 · Conditional probability, Bayes, likelihood](lessons/module-01/lesson-02.md).
