# 01.3 · Entropy, cross-entropy, KL divergence

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-01/lesson-02">01.2 · Conditional probability, Bayes, likelihood</a> — the categorical distribution, log-probabilities, and the log-likelihood. From <a href="#/lessons/module-01/lesson-01">01.1</a>, expectation as a probability-weighted sum.</p>
<p><strong>You will learn:</strong> <strong>entropy</strong> $H(p)$ (the average surprise, or information content, of a distribution), <strong>cross-entropy</strong> $H(p, q)$ (the cost of using distribution $q$ to encode data that truly comes from $p$), and <strong>KL divergence</strong> $\mathrm{KL}(p \parallel q)$ (how far $q$ is from $p$). You will prove and verify the identity $H(p, q) = H(p) + \mathrm{KL}(p \parallel q)$, all on a worked 3-outcome example in nats, and see why minimizing cross-entropy against the one-hot true next token is precisely the training signal for a language model.</p>
<p><strong>Why this matters for ML:</strong> the loss that trains every language model is cross-entropy between the model's predicted distribution $q$ and the true next token. This lesson defines that quantity exactly, shows it decomposes into an irreducible part (the data's own entropy) plus a part you can drive down (the KL gap between model and truth), and connects it back to the negative log-likelihood from lesson 01.2. Lesson 01.4 then applies it to the LM objective and perplexity.</p>
</div>

## 1. Intuition: surprise, measured in nats

Information theory starts from one idea: a **rare** event is more surprising, and carries more information, than a common one. If a weather model says "sunny" every day in a desert and it is sunny, you learned almost nothing. If it rains, you learned a lot. We want a number for "how surprised should I be to see outcome $x$", and it should be large when $p(x)$ is small.

The right measure is the **surprisal** (or information content) of an outcome:

$$
\text{surprisal}(x) = -\log p(x) = \log \frac{1}{p(x)}.
$$

Naming it: $p(x)$ is the probability of the outcome; $-\log p(x)$ is large when $p(x)$ is near 0 (a rare, surprising event) and is 0 when $p(x) = 1$ (a certain event — no surprise). Because $p(x) \le 1$, the log is $\le 0$ and the minus sign makes surprisal non-negative.

**A note on the log base — nats vs bits.** If you use $\log_2$, surprisal is measured in **bits** (the number of yes/no questions needed to pin down the outcome). If you use the natural log $\ln$, the unit is **nats**. They differ only by the constant factor $\ln 2 \approx 0.6931$ (since $\log_2 x = \ln x / \ln 2$). **This course uses nats throughout**, because PyTorch's `log`, `softmax`, and `cross_entropy` all use the natural log, and we want our hand computations to match PyTorch to the last decimal. Keep this in mind whenever you see an entropy value elsewhere quoted "in bits" — divide our nats by $0.6931$ to convert.

## 2. Entropy: the average surprisal of a distribution

**Entropy** $H(p)$ is the *expected* surprisal — the average amount of information (in nats) you get per draw from a distribution $p$. Using the expectation-as-weighted-sum from lesson 01.1, weight each outcome's surprisal $-\log p(x)$ by its probability $p(x)$:

$$
H(p) = E_{x \sim p}\big[-\log p(x)\big] = -\sum_x p(x) \log p(x).
$$

Every symbol: the sum is over all outcomes $x$; $p(x)$ is the probability of $x$ and also the weight; $-\log p(x)$ is $x$'s surprisal; the overall minus sign makes $H(p) \ge 0$ (each term $p(x)\log p(x)$ is $\le 0$). By convention $0 \log 0 = 0$ — an impossible outcome contributes no surprise.

Entropy is largest when the distribution is **uniform** (maximum uncertainty — every outcome equally surprising) and smallest, exactly 0, when one outcome has probability 1 (a sure thing — no uncertainty at all). For a categorical over $K$ outcomes, the maximum possible entropy is $\log K$ nats, achieved by the uniform distribution. This $\log K$ ceiling is the fact behind perplexity in lesson 01.4.

### 2.1 Worked example: a 3-outcome distribution

Let the true distribution over three tokens be

$$
p = (0.5,\ 0.25,\ 0.25).
$$

Its entropy in nats (using $\ln 0.5 = -0.6931$, $\ln 0.25 = -1.3863$):

$$
H(p) = -\big[0.5\ln 0.5 + 0.25\ln 0.25 + 0.25\ln 0.25\big] = -\big[0.5(-0.6931) + 0.25(-1.3863) + 0.25(-1.3863)\big].
$$

$$
H(p) = -[-0.34657 - 0.34657 - 0.34657] = 1.03972 \text{ nats}.
$$

For comparison, a uniform distribution over 3 outcomes would have entropy $\ln 3 = 1.0986$ nats — slightly higher, because uniform is maximally uncertain. Our $p$ is a bit "peakier" (it favors the first token), so it carries slightly less surprise per draw.

## 3. Cross-entropy: encoding $p$'s data with $q$'s beliefs

Now suppose the data truly comes from $p$, but you only have a *model* $q$ — an approximation. **Cross-entropy** $H(p, q)$ is the average surprisal you incur when you use $q$'s log-probabilities to score outcomes that actually occur with frequencies $p$:

$$
H(p, q) = E_{x \sim p}\big[-\log q(x)\big] = -\sum_x p(x) \log q(x).
$$

The difference from entropy is subtle and crucial: the *weights* are the true probabilities $p(x)$ (because that is how often each outcome really happens), but the *surprisal* uses the model's belief $q(x)$ (because $q$ is what you are using to encode / predict). You are charged $-\log q(x)$ every time $x$ occurs, and $x$ occurs with frequency $p(x)$.

Cross-entropy is minimized, and equals the entropy $H(p)$, exactly when $q = p$. Any mismatch makes it strictly larger. So "make $q$ match $p$" and "minimize $H(p, q)$" are the same goal — which is the whole reason cross-entropy is the training loss.

### 3.1 Worked example: cross-entropy of $q$ against $p$

Keep $p = (0.5, 0.25, 0.25)$ and let the model be a *different* distribution

$$
q = (0.25,\ 0.25,\ 0.5).
$$

Then (weights from $p$, logs from $q$):

$$
H(p, q) = -\big[0.5\ln 0.25 + 0.25\ln 0.25 + 0.25\ln 0.5\big] = -\big[0.5(-1.3863) + 0.25(-1.3863) + 0.25(-0.6931)\big].
$$

$$
H(p, q) = -[-0.69315 - 0.34657 - 0.17329] = 1.21301 \text{ nats}.
$$

Note $H(p, q) = 1.21301 > H(p) = 1.03972$: using the wrong distribution $q$ costs more than the theoretical minimum $H(p)$. The excess, $1.21301 - 1.03972 = 0.17329$ nats, is exactly the KL divergence we define next.

## 4. KL divergence: how far $q$ is from $p$

The **Kullback–Leibler divergence** $\mathrm{KL}(p \parallel q)$ measures how much $q$ differs from $p$ — the extra average surprisal you pay for using $q$ instead of the true $p$:

$$
\mathrm{KL}(p \parallel q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = E_{x \sim p}\Big[\log \frac{p(x)}{q(x)}\Big].
$$

Each term compares the model's belief to the truth at outcome $x$, log-scaled, weighted by how often $x$ really happens. Two key properties:

1. **Non-negativity:** $\mathrm{KL}(p \parallel q) \ge 0$ always, with equality **iff** $q = p$. (This is Gibbs' inequality.) So KL is a "distance-like" measure — zero exactly when the model is perfect.
2. **Asymmetry:** $\mathrm{KL}(p \parallel q) \ne \mathrm{KL}(q \parallel p)$ in general. It is *not* a true distance; the order of the arguments matters. "$p \parallel q$" reads "KL of $p$ from $q$", with $p$ the true distribution supplying the weights.

### 4.1 Worked example

With the same $p$ and $q$, using $\ln(0.5/0.25) = \ln 2 = 0.6931$, $\ln(0.25/0.25) = 0$, $\ln(0.25/0.5) = -0.6931$:

$$
\mathrm{KL}(p \parallel q) = 0.5\ln\frac{0.5}{0.25} + 0.25\ln\frac{0.25}{0.25} + 0.25\ln\frac{0.25}{0.5} = 0.5(0.6931) + 0.25(0) + 0.25(-0.6931).
$$

$$
\mathrm{KL}(p \parallel q) = 0.34657 + 0 - 0.17329 = 0.17329 \text{ nats}.
$$

Positive, as it must be, and equal to the excess we spotted in section 3.1.

## 5. The identity $H(p, q) = H(p) + \mathrm{KL}(p \parallel q)$

The three quantities are tied together by one identity, and it is worth deriving because it explains *what the loss is actually made of*. Start from cross-entropy and split the log using $\log q(x) = \log p(x) - \log \frac{p(x)}{q(x)}$:

$$
H(p, q) = -\sum_x p(x)\log q(x) = -\sum_x p(x)\Big[\log p(x) - \log\tfrac{p(x)}{q(x)}\Big].
$$

Distribute the sum:

$$
H(p, q) = \underbrace{-\sum_x p(x)\log p(x)}_{H(p)} + \underbrace{\sum_x p(x)\log\tfrac{p(x)}{q(x)}}_{\mathrm{KL}(p \parallel q)} = H(p) + \mathrm{KL}(p \parallel q).
$$

Check with our numbers: $H(p) + \mathrm{KL}(p \parallel q) = 1.03972 + 0.17329 = 1.21301 = H(p, q)$. Exactly.

<div class="callout key"><p>Cross-entropy = entropy + KL divergence: $H(p, q) = H(p) + \mathrm{KL}(p \parallel q)$. The first term $H(p)$ is fixed by the data and cannot be reduced by any model — it is the irreducible uncertainty of the language itself. The second term $\mathrm{KL}(p \parallel q) \ge 0$ is the model's fault, and it is zero exactly when $q = p$. <strong>Minimizing cross-entropy is minimizing KL</strong>, because $H(p)$ is a constant the optimizer cannot touch.</p></div>

### 5.1 Verify the whole thing in Python

```python
import torch, math

p = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
q = torch.tensor([0.25, 0.25, 0.5], dtype=torch.float64)

Hp  = -(p * p.log()).sum()               # entropy of p
Hpq = -(p * q.log()).sum()               # cross-entropy H(p, q): weights p, logs q
KL  = (p * (p / q).log()).sum()          # KL(p || q)

print(Hp.item())                          # 1.0397207708399179
print(Hpq.item())                         # 1.2130075659799042
print(KL.item())                          # 0.17328679513998632
print((Hp + KL).item())                   # 1.2130075659799042  == H(p, q)
print(math.log(2))                         # 0.6931471805599453  (nats-per-bit factor)
```

The natural log (`.log()` in PyTorch is $\ln$) gives values in nats, and `Hp + KL` reproduces `Hpq` to machine precision — the identity holds numerically.

## 6. Why cross-entropy against a one-hot target is the LM training signal

Here is where it all lands. When training a language model, the "true" next token at a position is a *single known token* — say token index $t$. The true distribution $p$ at that position is therefore **one-hot**: probability 1 on the true token $t$ and 0 everywhere else,

$$
p = (0, \dots, 0, \underbrace{1}_{\text{index } t}, 0, \dots, 0).
$$

Plug this $p$ into cross-entropy. Every term with $p(x) = 0$ vanishes, leaving only the term at $x = t$ where $p(t) = 1$:

$$
H(p, q) = -\sum_x p(x)\log q(x) = -1 \cdot \log q(t) = -\log q(t).
$$

The cross-entropy collapses to the **negative log-probability the model assigned to the true token** — exactly the per-token negative log-likelihood from lesson 01.2. So minimizing cross-entropy against the one-hot target is identical to maximizing the log-likelihood of the observed text. Two things worth noticing:

- The entropy $H(p)$ of a one-hot distribution is 0 (a certain outcome carries no surprise), so the identity of section 5 becomes $H(p, q) = 0 + \mathrm{KL}(p \parallel q)$. For one-hot targets, cross-entropy *is* the KL divergence — the model's entire loss is its distance from the truth.
- Because only $q(t)$ appears, the model is rewarded purely for putting probability mass on the *correct* token; it is never explicitly told how to distribute the remaining mass among the wrong tokens (softmax handles that implicitly, as lesson 01.4 shows).

<div class="callout key"><p>True next token = one-hot $p$. Cross-entropy $H(p, q)$ then reduces to $-\log q(t)$, the negative log-probability of the correct token. Minimizing it over the corpus = maximum likelihood = pushing the model's distribution $q$ toward the truth. This single scalar per position is the language-model loss.</p></div>

## Common mistakes

- **Mixing bits and nats.** An entropy quoted "in bits" is $\ln 2 \approx 0.6931$ times smaller than the same quantity in nats. If your hand number is off by a factor of $0.6931$ from PyTorch, this is why. We use nats to match PyTorch.
- **Weighting cross-entropy by $q$ instead of $p$.** In $H(p, q) = -\sum p(x)\log q(x)$ the *weights* are the true $p(x)$ and the *logs* are the model $q(x)$. Swapping them computes something else entirely.
- **Treating KL as symmetric.** $\mathrm{KL}(p\parallel q) \ne \mathrm{KL}(q\parallel p)$. It is not a distance and the argument order encodes which distribution supplies the weights. (This asymmetry matters later in RLHF/DPO, Module 15.)
- **Forgetting $0\log 0 = 0$.** Outcomes with zero true probability drop out of entropy and cross-entropy. But note: if the *model* assigns $q(x) = 0$ to a token that actually occurs ($p(x) > 0$), then $-\log q(x) = +\infty$ — an infinite loss. This is why models must never assign exactly zero probability, which softmax guarantees.

## Exercise

Let $p = (0.7, 0.3)$ (the true distribution over two tokens) and $q = (0.5, 0.5)$ (a model that guesses uniformly). Compute $H(p)$, $H(p, q)$, and $\mathrm{KL}(p \parallel q)$ in nats, and confirm the identity.

<details><summary>Optional hint</summary>

Use $\ln 0.7 = -0.3567$, $\ln 0.3 = -1.2040$, $\ln 0.5 = -0.6931$. Entropy uses $p$'s logs; cross-entropy uses $q$'s logs with $p$'s weights.

</details>

<details><summary>Stronger hint</summary>

$H(p) = -[0.7\ln 0.7 + 0.3\ln 0.3]$. $H(p,q) = -[0.7\ln 0.5 + 0.3\ln 0.5] = -\ln 0.5$ (since $q$ is uniform, both logs are $\ln 0.5$). $\mathrm{KL} = H(p,q) - H(p)$.

</details>

<details><summary>Solution</summary>

$H(p) = -[0.7(-0.3567) + 0.3(-1.2040)] = -[-0.24967 - 0.36119] = 0.61086$ nats.

$H(p, q) = -[0.7\ln 0.5 + 0.3\ln 0.5] = -\ln 0.5 = 0.69315$ nats (a uniform $q$ over 2 outcomes always costs $\ln 2$).

$\mathrm{KL}(p \parallel q) = 0.69315 - 0.61086 = 0.08229$ nats, and indeed $H(p) + \mathrm{KL} = 0.61086 + 0.08229 = 0.69315 = H(p, q)$.

```python
import torch
p = torch.tensor([0.7, 0.3], dtype=torch.float64)
q = torch.tensor([0.5, 0.5], dtype=torch.float64)
Hp  = -(p*p.log()).sum();  Hpq = -(p*q.log()).sum();  KL = (p*(p/q).log()).sum()
print(Hp.item(), Hpq.item(), KL.item())   # 0.6108643020548935 0.6931471805599453 0.08228287850505185
```

</details>

## Check yourself

<details><summary>What is the difference between entropy $H(p)$ and cross-entropy $H(p, q)$, term by term?</summary>

Both are probability-weighted sums of negative log-probabilities. In entropy $H(p) = -\sum p(x)\log p(x)$ the weights and the logs both come from $p$. In cross-entropy $H(p, q) = -\sum p(x)\log q(x)$ the weights still come from the true $p$ but the logs come from the model $q$. They are equal exactly when $q = p$; otherwise $H(p,q) > H(p)$.

</details>

<details><summary>When the true next-token distribution is one-hot on token $t$, what does the cross-entropy loss reduce to, and why?</summary>

It reduces to $-\log q(t)$, the negative log-probability the model assigned to the true token. Every term in $-\sum_x p(x)\log q(x)$ with $p(x) = 0$ vanishes, and only the $x = t$ term survives with weight $p(t) = 1$. This is exactly the per-token negative log-likelihood, so minimizing it is maximum likelihood.

</details>

<details><summary>The identity $H(p,q) = H(p) + \mathrm{KL}(p\parallel q)$ says training cannot reduce $H(p)$. What is $H(p)$ for a one-hot target, and what does that mean for the loss?</summary>

For a one-hot $p$, $H(p) = 0$ (a certain outcome has no surprise). So the identity becomes $H(p, q) = \mathrm{KL}(p \parallel q)$: with one-hot targets the entire cross-entropy loss *is* the KL divergence from the model to the truth, and driving the loss to 0 means driving $q$ to exactly match the one-hot target.

</details>

<details><summary>Your hand-computed entropy is 0.693 but a textbook says the same distribution has entropy 1.0. What likely went wrong?</summary>

Unit mismatch. $0.693 = 1.0 \times \ln 2$, so the textbook is in bits ($\log_2$) and you (or it) used the other base. We use nats ($\ln$) throughout to match PyTorch; multiply nats by $1/\ln 2 \approx 1.4427$ to get bits, or divide bits by that to get nats.

</details>

## Next

You now have entropy, cross-entropy, and KL divergence, the identity that binds them, and the key fact that cross-entropy against a one-hot true token is the negative log-likelihood — the language-model training signal. The final lesson of this module assembles these pieces into the concrete objective a language model minimizes, shows the per-token loss on a worked softmax example, and introduces perplexity as the human-readable version of the loss. It also connects to the `llmre` code you will use for the rest of the course.

Continue to [01.4 · The language-modeling objective & perplexity](lessons/module-01/lesson-04.md).
