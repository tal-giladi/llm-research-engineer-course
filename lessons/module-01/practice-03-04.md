# 01.3–01.4 · Practice set: entropy, cross-entropy, KL, softmax, the LM loss

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-01/lesson-03">01.3 · Entropy, cross-entropy, KL divergence</a> and <a href="#/lessons/module-01/lesson-04">01.4 · The language-modeling objective &amp; perplexity</a>.</p>
<p><strong>How to use this page:</strong> solve every exercise before scrolling to the <a href="#/lessons/module-01/practice-03-04?id=solutions">Solutions</a>. Exercises marked ✍️ are for pen and paper (a calculator for $e^x$ and $\ln x$ is fine). Exercises marked 🐍 are for PyTorch. Use the natural log ($\ln$) everywhere, so every answer is in nats, unless the exercise says bits. Round to 4 decimals.</p>
<p><strong>Why this matters for ML:</strong> every number here is a piece of the loss that every language model in this course is trained on. If you can do these without looking, you own the loss.</p>
</div>

<div class="hw"><p><strong>Hardware:</strong> CPU-only. Every 🐍 exercise runs in under a second on a laptop with <code>torch</code> installed.</p></div>

**Formula sheet** (the only things you need):

$$
H(p) = -\sum_x p(x)\ln p(x), \qquad
H(p, q) = -\sum_x p(x)\ln q(x), \qquad
\mathrm{KL}(p\parallel q) = \sum_x p(x)\ln\frac{p(x)}{q(x)}
$$

$$
H(p, q) = H(p) + \mathrm{KL}(p\parallel q), \qquad 0\ln 0 = 0
$$

$$
q_i = \operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad
\ln q_i = z_i - \ln\sum_j e^{z_j}, \qquad
\ell = -\ln q(t), \qquad
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} -\ln q_i(t_i), \qquad
\mathrm{PPL} = e^{\mathcal{L}}
$$

---

## Part A — Entropy, cross-entropy, KL (lesson 01.3)

**A1 ✍️ — Fair coin.** Let $p = (0.5, 0.5)$. Compute $H(p)$ in nats, then convert it to bits.

**A2 ✍️ — The most uncertain distribution.** Let $p = (0.25, 0.25, 0.25, 0.25)$. Compute $H(p)$. Then answer in one sentence: over 4 outcomes, can any distribution have a higher entropy? And which distribution has the lowest possible entropy, and what is it?

**A3 ✍️ — The full identity.** True distribution $p = (0.8, 0.2)$, model $q = (0.5, 0.5)$. Compute:

1. $H(p)$
2. $H(p, q)$
3. $\mathrm{KL}(p\parallel q)$
4. Check that $H(p) + \mathrm{KL}(p\parallel q) = H(p, q)$.

**A4 ✍️ — KL is not symmetric.** Same $p$ and $q$ as A3. Compute $\mathrm{KL}(q\parallel p)$ and compare it with $\mathrm{KL}(p\parallel q)$. In the formula, which distribution supplies the *weights* in each case?

**A5 ✍️ — Derive the identity yourself.** Without looking at the lesson:

1. Starting from $\ln\frac{p(x)}{q(x)} = \ln p(x) - \ln q(x)$, isolate $\ln q(x)$.
2. Substitute it into $H(p, q) = -\sum_x p(x)\ln q(x)$ and show that $H(p, q) = H(p) + \mathrm{KL}(p\parallel q)$.
3. Given that $\mathrm{KL} \ge 0$ always, what is the smallest value the cross-entropy loss can reach, and when is it reached?

---

## Part B — Zeros and infinities (lesson 01.3, "Forgetting $0\ln 0 = 0$")

**B1 ✍️ — A zero in $p$.** $p = (0.5, 0.5, 0)$, $q = (0.5, 0.4, 0.1)$. Compute $H(p)$, $H(p, q)$, and $\mathrm{KL}(p\parallel q)$. What happens to the third outcome in each formula?

**B2 ✍️ — A zero in $q$, harmless version.** $p = (0.5, 0.5, 0)$, $q = (0.6, 0.4, 0)$. Compute $H(p, q)$. Is it finite? Why?

**B3 ✍️ — A zero in $q$, catastrophic version.** $p = (0.5, 0.3, 0.2)$, $q = (0.6, 0.4, 0)$. Compute $H(p, q)$. Which term causes the problem, and what does it mean in words?

**B4 🐍 — The `nan` trap.** Predict what each line prints *before* running it, then run it:

```python
import torch
p = torch.tensor([0.5, 0.5, 0.0])
print(-(p * p.log()).sum())
print(-torch.xlogy(p, p).sum())
```

Why does the first line not give $0.6931$?

---

## Part C — Softmax and log-softmax (lesson 01.4, sections 2 and 6)

**C1 ✍️ — Softmax with the max trick.** Logits $z = (3, 1, 0)$.

1. Subtract the max. Write the shifted row $z'$.
2. Exponentiate $z'$, sum, and normalize to get $q$.
3. Compute $\ln q$ two ways: as $\ln$ of each entry of $q$, and as $z' - \ln\sum_j e^{z'_j}$. They must agree.

**C2 ✍️ — Huge logits.** Logits $z = (1000, 999)$.

1. What happens if you compute $e^{1000}$ in float32 (largest value $\approx 3.4 \times 10^{38}$)? What would naive softmax return?
2. Compute the softmax correctly by hand. (Hint: which smaller row has the same softmax?)

**C3 ✍️ — Designing a logit.** Two-token vocabulary, logits $z = (a, 0)$. Find the value of $a$ that makes the model assign probability exactly $0.9$ to token 0. (Use the formula, not trial and error.)

**C4 ✍️ — The loss as logsumexp minus one logit.**

1. Starting from $\ell = -\ln q(t)$ and the softmax definition, prove that $\ell = \ln\sum_j e^{z_j} - z_t$.
2. Use it to compute the loss for $z = (2, 1, 0, -1)$ with true token $t = 1$. (Lesson 01.4 computed $\ln\sum_j e^{z_j} = 2.4402$ for this row.)
3. Without computing anything new: what is the loss for $t = 2$ and $t = 3$?

**C5 ✍️ — Prove shift invariance again.** Without looking: show that $\operatorname{softmax}(z - c) = \operatorname{softmax}(z)$ for any constant $c$, naming the identity you use at each step. Then explain in one sentence why the code picks $c = \max_j z_j$ and not, say, $c = 0$ or $c = $ the mean.

---

## Part D — The loss over many positions (lesson 01.4, sections 3–4 and 7)

**D1 ✍️ — One sentence, every number.** A model predicts a 4-token sentence and assigns the true tokens the probabilities $0.5,\ 0.25,\ 0.125,\ 0.5$. Compute:

1. the per-token losses $-\ln q_i(t_i)$;
2. the sum of the losses;
3. the mean loss $\mathcal{L}$;
4. the likelihood of the sentence (product of the four probabilities), and check that $-\ln(\text{likelihood})$ equals the sum from step 2;
5. the perplexity $e^{\mathcal{L}}$, and check that it also equals $\text{likelihood}^{-1/4}$.

**D2 ✍️ — Uniform model.** A model outputs logits $(0, 0, 0, 0)$ at every position of a 10-token text, $V = 4$. What is the loss at each position? The mean loss? The perplexity? Does it depend on which tokens are true?

**D3 ✍️ — Compare two models fairly.** Model A reports a *mean* loss of $2.0$ nats over 100 tokens. Model B reports a *summed* loss of $150$ nats over 60 tokens. Which model predicts better per token? What are the two perplexities?

---

## Part E — Shapes (lesson 01.4, section 5)

**E1 ✍️ — From tokens to targets.**

```text
tokens = [[5, 3, 7, 1],
          [2, 2, 9, 4]]
```

1. Write `inputs = tokens[:, :-1]` and `targets = tokens[:, 1:]`, with their shapes.
2. How many predictions (rows of the loss) does this batch produce?
3. For sequence 0, list each prediction as "has seen …, must predict …".
4. Which tokens are never used as context by any position? Which token is never a target?

**E2 ✍️ — Real-size shapes.** A GPT-2-sized run: batch of $B = 8$ sequences, each loaded as 129 tokens, vocabulary $V = 50257$.

1. Shapes and dtypes of `tokens`, `inputs`, `targets`, `logits`.
2. Shapes after `logits.reshape(-1, V)` and `targets.reshape(-1)`.
3. Shape of the loss.
4. How much memory does the float32 `logits` tensor alone take? In bf16?

**E3 ✍️ — The `keepdim` bug.** Someone writes `log_softmax` with `m = logits.max(dim=-1).values` (no `keepdim=True`) and keeps everything else the same.

1. For `logits` of shape `(4, 3)`, what are the shapes of `m` and of `logits - m`? What happens?
2. For `logits` of shape `(3, 3)`, what happens? Use `logits = [[2, 1, 0], [0, 0, 0], [1, 2, 0]]` and compute row 0 of `logits - m`. Which token does row 0 now prefer? Why is this case worse than case 1?

---

## Part F — PyTorch

**F1 🐍 — The whole loss from scratch.**

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)
logits  = torch.randn(2, 3, 4)            # (B, T, V)
targets = torch.randint(0, 4, (2, 3))     # (B, T)
```

Compute the mean cross-entropy **without** `F.cross_entropy`, `F.log_softmax`, `torch.logsumexp`, or `.softmax` — only `max`, `exp`, `sum`, `log`, indexing, and `mean`. Print the shape after every step. Check that your answer equals `F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))`.

**F2 🐍 — Verify the identity on random distributions.** Build two random distributions over 5 outcomes with `torch.randn(5).softmax(0)`. Compute $H(p)$, $H(p, q)$, $\mathrm{KL}(p\parallel q)$, and $\mathrm{KL}(q\parallel p)$. Assert that $H(p, q) = H(p) + \mathrm{KL}(p\parallel q)$ to $10^{-6}$, that $\mathrm{KL} \ge 0$, and that the two KLs differ. Then set one entry of $p$ to 0 (and renormalize) and make it still work — use `torch.xlogy`.

**F3 🐍 — One-hot vs index.** For the logits of F1 (reshaped to `(6, 4)`), compute the per-row loss two ways: with `F.one_hot(targets, 4)` multiplied into the log-probabilities and summed over the vocabulary, and with direct indexing `logp[rows, targets]`. Show they are equal.

**F4 🐍 — Perplexity of a uniform model.** For $V \in \{2, 10, 50257\}$, build logits of zeros with shape `(16, V)` and random targets. Show that the loss is $\ln V$ and the perplexity is $V$, whatever the targets are.

**F5 🐍 — The gradient of the loss.** For `z = torch.tensor([[2., 1., 0.]], requires_grad=True)` and target `0`, compute `F.cross_entropy`, call `.backward()`, and print `z.grad`. Compare it with `softmax(z) - one_hot(target)` computed by hand. What does the sign of each gradient entry tell the optimizer to do to that logit?

---

## Part G — Debugging

**G1 ✍️🐍 — Softmax twice.** A colleague writes

```python
loss = F.cross_entropy(logits.softmax(-1), targets)
```

For `logits = [[2., 1., 0.]]` and target `0`: what is the correct loss, and what does this line return? Why is the wrong value closer to $\ln 3 = 1.0986$ than the correct one? Why does this bug hurt training even though no error is raised?

**G2 ✍️ — Forgot the shift.** A colleague builds `targets = tokens[:, :-1]` (the same slice as `inputs`) instead of `tokens[:, 1:]`. Training loss drops to almost 0 within a few steps. What did the model learn, and why is it useless?

**G3 🐍 — Underflow.** Run

```python
z = torch.tensor([0.0, -200.0])
print(z.softmax(0).log())
print(z.log_softmax(0))
```

Explain the difference in one sentence, and say which of the two the training loss must use.

---


## Solutions

Stop here if you have not tried the exercises yet.

### A1

$H(p) = -(0.5\ln 0.5 + 0.5\ln 0.5) = -\ln 0.5 = \ln 2 = 0.6931$ nats. To convert nats to bits, divide by $\ln 2$: $0.6931 / 0.6931 = 1$ bit. A fair coin carries exactly one bit of uncertainty; that is the definition of a bit.

### A2

$H(p) = -4 \times 0.25\ln 0.25 = -\ln 0.25 = \ln 4 = 1.3863$ nats. No distribution over 4 outcomes has higher entropy: the uniform distribution is the most uncertain, and its entropy is $\ln V$. The lowest is a one-hot distribution such as $(1, 0, 0, 0)$: $H = -(1\cdot\ln 1 + 0 + 0 + 0) = 0$, since $\ln 1 = 0$ and the zeros drop out by $0\ln 0 = 0$. No uncertainty at all.

### A3

1. $H(p) = -(0.8\ln 0.8 + 0.2\ln 0.2) = -(0.8 \times (-0.2231) + 0.2 \times (-1.6094)) = 0.1785 + 0.3219 = 0.5004$.
2. $H(p, q) = -(0.8\ln 0.5 + 0.2\ln 0.5) = -\ln 0.5 = 0.6931$. The weights come from $p$ (0.8, 0.2); the logs come from $q$. Because $q$ is uniform, both logs are the same and the weights sum to 1.
3. $\mathrm{KL}(p\parallel q) = 0.8\ln\frac{0.8}{0.5} + 0.2\ln\frac{0.2}{0.5} = 0.8\ln 1.6 + 0.2\ln 0.4 = 0.8 \times 0.4700 + 0.2 \times (-0.9163) = 0.3760 - 0.1833 = 0.1927$.
4. $0.5004 + 0.1927 = 0.6931 = H(p, q)$. ✓

Reading: the true text has 0.5004 nats of built-in uncertainty that no model can remove; this model wastes another 0.1927 nats by being uniform when the truth is 80/20.

### A4

$\mathrm{KL}(q\parallel p) = 0.5\ln\frac{0.5}{0.8} + 0.5\ln\frac{0.5}{0.2} = 0.5\ln 0.625 + 0.5\ln 2.5 = 0.5 \times (-0.4700) + 0.5 \times 0.9163 = 0.2231$.

$\mathrm{KL}(p\parallel q) = 0.1927 \ne 0.2231 = \mathrm{KL}(q\parallel p)$. The first argument always supplies the weights: in $\mathrm{KL}(p\parallel q)$ the weights are $p = (0.8, 0.2)$; in $\mathrm{KL}(q\parallel p)$ they are $q = (0.5, 0.5)$. Different weights on different log-ratios give different numbers, so KL is not a distance.

### A5

1. Add $\ln q(x)$ to both sides and subtract $\ln\frac{p(x)}{q(x)}$ from both sides: $\ln q(x) = \ln p(x) - \ln\frac{p(x)}{q(x)}$.
2. $H(p, q) = -\sum_x p(x)\Big[\ln p(x) - \ln\frac{p(x)}{q(x)}\Big] = -\sum_x p(x)\ln p(x) + \sum_x p(x)\ln\frac{p(x)}{q(x)} = H(p) + \mathrm{KL}(p\parallel q)$. The first step distributes $p(x)$ over the bracket; the second splits one sum into two.
3. Since $\mathrm{KL} \ge 0$, $H(p, q) \ge H(p)$. The loss can never go below the entropy of the data, and it reaches it exactly when $\mathrm{KL} = 0$, which happens only when $q = p$ (the model has learned the true distribution).

### B1

- $H(p) = -(0.5\ln 0.5 + 0.5\ln 0.5 + 0\ln 0) = 0.6931 + 0 = 0.6931$.
- $H(p, q) = -(0.5\ln 0.5 + 0.5\ln 0.4 + 0\cdot\ln 0.1) = 0.3466 + 0.4581 + 0 = 0.8047$.
- $\mathrm{KL}(p\parallel q) = 0.5\ln\frac{0.5}{0.5} + 0.5\ln\frac{0.5}{0.4} + 0 = 0 + 0.5 \times 0.2231 = 0.1116$.
- Check: $0.6931 + 0.1116 = 0.8047$. ✓

In every formula the third term has weight $p_3 = 0$, so it vanishes. The model's $q_3 = 0.1$ does not matter for the score; it only matters indirectly, because that 0.1 was taken away from the two outcomes that do happen.

### B2

$H(p, q) = -(0.5\ln 0.6 + 0.5\ln 0.4 + 0\cdot\ln 0) = 0.2554 + 0.4581 + 0 = 0.7136$. Finite. The model says outcome 3 is impossible, and it really is impossible ($p_3 = 0$), so its term has weight 0 and we use the convention $0\ln 0 = 0$. The model is never punished for an outcome that never happens.

### B3

$H(p, q) = -(0.5\ln 0.6 + 0.3\ln 0.4 + 0.2\ln 0)$. The last term is $-0.2 \times (-\infty) = +\infty$, so $H(p, q) = +\infty$. The weight $p_3 = 0.2$ is not zero, so no convention saves it. In words: the model said outcome 3 is impossible, and then it happened 20% of the time. That is infinite surprise, and the loss is infinite. This is why a language model must never output exactly $q = 0$ for any token.

### B4

The first line prints `tensor(nan)`. The second prints `tensor(0.6931)`. PyTorch evaluates `p.log()` first, which gives `[-0.6931, -0.6931, -inf]`, and then multiplies: $0 \times (-\infty)$ is `nan` in floating point. One `nan` makes the sum `nan`. The convention $0\ln 0 = 0$ exists on paper, not in the hardware. `torch.xlogy(x, y)` computes $x\ln y$ and returns exactly 0 whenever $x = 0$, so it implements the convention.

### C1

1. $\max = 3$, so $z' = (3-3, 1-3, 0-3) = (0, -2, -3)$.
2. $e^{z'} = (1, 0.1353, 0.0498)$, sum $= 1.1851$. $q = (1/1.1851,\ 0.1353/1.1851,\ 0.0498/1.1851) = (0.8438, 0.1142, 0.0420)$. Check: $0.8438 + 0.1142 + 0.0420 = 1$.
3. Directly: $\ln q = (\ln 0.8438, \ln 0.1142, \ln 0.0420) = (-0.1698, -2.1698, -3.1698)$. Via the formula: $\ln 1.1851 = 0.1698$, so $z' - 0.1698 = (0, -2, -3) - 0.1698 = (-0.1698, -2.1698, -3.1698)$. Same. ✓ Notice that the log-probabilities are just the shifted logits moved down by one shared number; the gaps between them (2 and 1) are exactly the gaps between the logits.

### C2

1. $e^{1000} \approx 10^{434}$, far above $3.4 \times 10^{38}$, so float32 gives `inf`. Both exponentials are `inf`, so naive softmax computes `inf / (inf + inf)` = `nan` for both entries.
2. Subtract the max, 1000: $z' = (0, -1)$. Softmax only depends on differences, so $\operatorname{softmax}(1000, 999) = \operatorname{softmax}(0, -1) = \operatorname{softmax}(1, 0)$. $e^{z'} = (1, 0.3679)$, sum $1.3679$, so $q = (0.7311, 0.2689)$.

### C3

$q_0 = \frac{e^a}{e^a + e^0} = \frac{e^a}{e^a + 1} = 0.9$. Multiply both sides by $(e^a + 1)$: $e^a = 0.9e^a + 0.9$, so $0.1e^a = 0.9$, so $e^a = 9$, so $a = \ln 9 = 2.1972$. Check: $9/(9+1) = 0.9$. ✓ In general, for two tokens the logit gap equals the log of the odds: $a - 0 = \ln\frac{0.9}{0.1}$.

### C4

1. $\ell = -\ln q(t) = -\ln\frac{e^{z_t}}{\sum_j e^{z_j}} = -\Big[\ln e^{z_t} - \ln\sum_j e^{z_j}\Big] = -\Big[z_t - \ln\sum_j e^{z_j}\Big] = \ln\sum_j e^{z_j} - z_t$. The identities: $\ln\frac{a}{b} = \ln a - \ln b$, then $\ln e^{x} = x$.
2. $\ell = 2.4402 - z_1 = 2.4402 - 1 = 1.4402$.
3. The logsumexp is shared by the whole row, so only $z_t$ changes: $t = 2$ gives $2.4402 - 0 = 2.4402$; $t = 3$ gives $2.4402 - (-1) = 3.4402$. (Lesson 01.4's table rounds these to 1.44, 2.44, 3.44.) Each 1-unit drop in the true token's logit costs exactly 1 more nat.

### C5

$\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} \overset{(1)}{=} \frac{e^{z_i}e^{-c}}{\sum_j e^{z_j}e^{-c}} \overset{(2)}{=} \frac{e^{z_i}e^{-c}}{e^{-c}\sum_j e^{z_j}} \overset{(3)}{=} \frac{e^{z_i}}{\sum_j e^{z_j}}$.

(1) $e^{a-b} = e^a e^{-b}$. (2) $e^{-c}$ does not depend on $j$, so it factors out of the sum. (3) The same positive number on top and bottom cancels.

Why the max: with $c = \max_j z_j$, every shifted logit is $\le 0$, so every $e^{z'_j} \le 1$ (no overflow), and the largest one is exactly $e^0 = 1$, so the sum is $\ge 1$ (its log is never $-\infty$). $c = 0$ does nothing (logits of 1000 still overflow). The mean still leaves positive values above it (for $(1000, 0)$ the mean is 500, and $e^{500}$ overflows).

### D1

1. $-\ln 0.5 = 0.6931$, $-\ln 0.25 = 1.3863$, $-\ln 0.125 = 2.0794$, $-\ln 0.5 = 0.6931$.
2. Sum $= 4.8520$.
3. $\mathcal{L} = 4.8520 / 4 = 1.2130$ nats per token.
4. Likelihood $= 0.5 \times 0.25 \times 0.125 \times 0.5 = 0.0078125$. $-\ln 0.0078125 = 4.8520$. ✓ The sum of the per-token losses *is* the negative log-likelihood of the sentence.
5. $\mathrm{PPL} = e^{1.2130} = 3.3636$. Also $0.0078125^{-1/4} = 128^{1/4} = 3.3636$. ✓ Perplexity is the likelihood turned into "per token" by a geometric mean. Reading: on average the model is as uncertain as if it were choosing uniformly among about 3.4 tokens.

### D2

Softmax of four equal logits is uniform: $q = (0.25, 0.25, 0.25, 0.25)$. Whatever the true token, $q(t) = 0.25$, so each position's loss is $-\ln 0.25 = \ln 4 = 1.3863$. The mean over 10 positions is the same, $1.3863$. Perplexity $= e^{\ln 4} = 4$. It does not depend on the true tokens at all. This is the loss of a model that knows nothing, and a freshly initialized LM starts near $\ln V$ (for GPT-2's vocabulary, $\ln 50257 = 10.82$).

### D3

Convert B to a mean: $150 / 60 = 2.5$ nats per token. A: $2.0$. Lower is better, so **A** predicts better per token, even though B's number of tokens is smaller. Perplexities: A: $e^{2.0} = 7.39$; B: $e^{2.5} = 12.18$. Sums are not comparable across different token counts; means are.

### E1

1. `inputs = [[5, 3, 7], [2, 2, 9]]`, shape `(2, 3)`. `targets = [[3, 7, 1], [2, 9, 4]]`, shape `(2, 3)`, int64.
2. $2 \times 3 = 6$ predictions.
3. Sequence 0: has seen `[5]` → must predict `3`; has seen `[5, 3]` → must predict `7`; has seen `[5, 3, 7]` → must predict `1`.
4. The last token of each row (`1` and `4`) is never context: nothing comes after it. The first token of each row (`5` and `2`) is never a target: nothing comes before it to predict it. Every other token is both the target of the position before it and context for the positions after it.

### E2

1. `tokens`: `(8, 129)` int64. `inputs`: `(8, 128)` int64. `targets`: `(8, 128)` int64. `logits`: `(8, 128, 50257)` float32 (or bf16).
2. `(1024, 50257)` and `(1024,)`, because $8 \times 128 = 1024$.
3. `()`, a scalar.
4. $8 \times 128 \times 50257 = 51{,}463{,}168$ numbers. $\times 4$ bytes $= 205{,}852{,}672$ bytes $\approx 196$ MiB in float32; $\times 2$ bytes $\approx 98$ MiB in bf16. That is for the logits alone, before gradients (the backward pass needs a tensor of the same size). The vocabulary axis is why the output layer is one of the biggest memory costs in LM training.

### E3

1. `m` has shape `(4,)`. Broadcasting aligns the *last* axes: `(4, 3) - (4,)` compares 3 with 4, so PyTorch raises a shape error. Annoying, but safe: you see the bug immediately.
2. With `(3, 3)`, `m = [2, 0, 2]` has shape `(3,)`, and `(3, 3) - (3,)` broadcasts silently. But it subtracts `m[j]` from **column** `j`, not each row's own max from that row. Row 0 becomes `[2-2, 1-0, 0-2] = [0, 1, -2]`. The original row `[2, 1, 0]` preferred token 0; the corrupted row prefers token 1. No error, wrong model predictions, and a loss that trains toward the wrong answer. Worse than case 1 because nothing tells you. `keepdim=True` makes `m` shape `(3, 1)`, a column, so each row subtracts its own max.

### F1

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)
logits  = torch.randn(2, 3, 4)                      # (2, 3, 4)
targets = torch.randint(0, 4, (2, 3))               # (2, 3)

z = logits.reshape(-1, 4);            print(z.shape)          # (6, 4)
t = targets.reshape(-1);              print(t.shape)          # (6,)
m = z.max(dim=-1, keepdim=True).values; print(m.shape)        # (6, 1)
shifted = z - m;                      print(shifted.shape)    # (6, 4), max of each row is 0
lse = shifted.exp().sum(dim=-1, keepdim=True).log(); print(lse.shape)  # (6, 1)
logp = shifted - lse;                 print(logp.shape)       # (6, 4), rows exp-sum to 1
picked = logp[torch.arange(6), t];    print(picked.shape)     # (6,)
loss = -picked.mean();                print(loss.shape)       # ()

print(loss)                                                   # tensor(1.8831)
print(F.cross_entropy(z, t))                                  # tensor(1.8831)
```

The shapes go `(2, 3, 4)` → `(6, 4)` → `(6, 4)` → `(6,)` → `()`. The vocabulary axis disappears at the indexing step and the position axis at the mean.

### F2

```python
import torch
torch.manual_seed(1)
p = torch.randn(5).softmax(0)
q = torch.randn(5).softmax(0)

H_p   = -(p * p.log()).sum()
H_pq  = -(p * q.log()).sum()
KL_pq =  (p * (p / q).log()).sum()
KL_qp =  (q * (q / p).log()).sum()

assert torch.allclose(H_pq, H_p + KL_pq, atol=1e-6)
assert KL_pq >= 0 and KL_qp >= 0
print(KL_pq.item(), KL_qp.item())      # two different numbers

# with a zero in p: xlogy(x, y) = x*log(y), and exactly 0 when x == 0
p[2] = 0.0
p = p / p.sum()
H_p   = -torch.xlogy(p, p).sum()
H_pq  = -torch.xlogy(p, q).sum()
KL_pq =  torch.xlogy(p, p).sum() - torch.xlogy(p, q).sum()   # sum p log p - sum p log q
assert torch.allclose(H_pq, H_p + KL_pq, atol=1e-6)
```

The KL line uses $\sum p\ln\frac{p}{q} = \sum p\ln p - \sum p\ln q$, so it never divides by zero or takes $\ln 0$ with a nonzero weight.

### F3

```python
logp = z.log_softmax(-1)                                  # (6, 4)
rows = torch.arange(6)
by_index  = -logp[rows, t]                                # (6,)
by_onehot = -(F.one_hot(t, 4) * logp).sum(dim=-1)         # (6, 4) * (6, 4) -> sum -> (6,)
print(torch.allclose(by_index, by_onehot))                # True
```

The one-hot row is $p$ in the formula $-\sum_x p(x)\ln q(x)$; three of its four entries are 0, so three terms vanish. Indexing reads the one surviving term directly, without building a `(6, 4)` tensor of mostly zeros. At $V = 50257$ that difference is large.

### F4

```python
import math, torch, torch.nn.functional as F
for V in (2, 10, 50257):
    z = torch.zeros(16, V)
    t = torch.randint(0, V, (16,))
    loss = F.cross_entropy(z, t)
    print(V, loss.item(), math.log(V), loss.exp().item())   # loss == ln V, perplexity == V
```

For $V = 50257$: loss $= 10.8249$, perplexity $= 50257$. A model with perplexity $V$ has learned nothing.

### F5

```python
import torch, torch.nn.functional as F
z = torch.tensor([[2., 1., 0.]], requires_grad=True)
loss = F.cross_entropy(z, torch.tensor([0]))
loss.backward()
print(z.grad)                                            # tensor([[-0.3348,  0.2447,  0.0900]])
print(z.softmax(-1) - F.one_hot(torch.tensor([0]), 3))   # the same numbers
```

The gradient is $q - p$: $(0.6652 - 1,\ 0.2447 - 0,\ 0.0900 - 0) = (-0.3348, 0.2447, 0.0900)$. Gradient descent moves each logit *against* its gradient. The true token's entry is negative, so its logit goes **up**. The wrong tokens' entries are positive, so their logits go **down**, each in proportion to how much probability it wrongly took. When $q = p$ the gradient is 0 and training stops pushing.

### G1

Correct: $-\ln 0.6652 = 0.4076$. Buggy: the line softmaxes first, giving $(0.6652, 0.2447, 0.0900)$, and then `F.cross_entropy` treats those probabilities as *logits* and softmaxes them again. It returns $0.7972$. Because the probabilities all lie between 0 and 1, their differences are small, and a second softmax of nearly equal values is close to uniform, so the loss is pulled toward $\ln 3 = 1.0986$. No error is raised because the shapes and dtypes are legal. But the loss can never go near 0 (even a perfect model's second softmax of $(1, 0, 0)$ gives $q(0) = e/(e+2) = 0.576$, loss $0.551$), and the gradients are squashed, so training is slow and stalls at a wrong value. `F.cross_entropy` always takes raw logits.

### G2

With `targets = inputs`, position $i$'s target is the token at position $i$, which is the token the model is currently reading. The model learns to copy its input, which makes the loss nearly 0. It has learned nothing about what comes *next*, so at generation time it just repeats. The shift `tokens[:, 1:]` is what makes the target at position $i$ be token $i+1$, the one the causal mask hides.

### G3

The first prints `tensor([0., -inf])`: `softmax` computes $e^{-200} \approx 10^{-87}$, which is below float32's smallest value and rounds to exactly 0, and then `log(0) = -inf`. The second prints `tensor([0., -200.])`: `log_softmax` computes $z' - \operatorname{logsumexp}(z')$ directly and never builds the tiny probability. The training loss must use the log-softmax path, which is what `F.cross_entropy` does internally when you give it logits.

---

Back to [01.3](lessons/module-01/lesson-03.md) · [01.4](lessons/module-01/lesson-04.md). Continue to [02.1 · Derivatives, partials, chain rule, gradients](lessons/module-02/lesson-01.md).
