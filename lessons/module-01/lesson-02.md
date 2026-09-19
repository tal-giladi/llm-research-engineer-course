# 01.2 · Conditional probability, Bayes, likelihood

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-01.md">01.1 · Random variables, expectation, variance</a> — especially the PMF, the "sums to 1" rule, and the categorical distribution as a language model's per-position output.</p>
<p><strong>You will learn:</strong> conditional probability $p(A \mid B)$, joint and marginal probability, Bayes' rule (with a worked medical-test example), and the <strong>chain rule of probability</strong> — the exact factorization that <em>defines</em> language modeling: a sentence's probability is the product of its next-token probabilities. Then the difference between likelihood and probability, maximum likelihood estimation, and why we sum log-probabilities instead of multiplying raw probabilities (a worked underflow example).</p>
<p><strong>Why this matters for ML:</strong> a language model does not assign a probability to a sentence in one shot — it predicts one token at a time. The chain rule is the theorem that says doing so is not an approximation: the product of per-token conditional probabilities <em>is</em> the exact probability of the whole sequence. Training the model is maximum likelihood estimation on that product, and the log-likelihood you will maximize is the direct ancestor of the cross-entropy loss in lesson 01.4.</p>
</div>

## 1. Conditional probability: updating on information

A **conditional probability** $p(A \mid B)$ is the probability that $A$ happens *given that you already know* $B$ happened. Read the bar "$\mid$" as "given". Knowing $B$ can change the odds of $A$: the probability that a sentence's next word is `Paris` is small in general, but given the preceding words are `The capital of France is`, it becomes large.

The definition is

$$
p(A \mid B) = \frac{p(A, B)}{p(B)}, \qquad \text{valid when } p(B) > 0,
$$

where $p(A, B)$ is the **joint probability** that $A$ *and* $B$ both happen, and $p(B)$ is the probability of $B$ alone. The intuition: knowing $B$ restricts the world to just the outcomes where $B$ is true, so we rescale by dividing by $p(B)$ — that is why the denominator appears, and it is why the formula needs $p(B) > 0$ (you cannot condition on something impossible).

Rearranging the definition gives the **product rule**, which we will use constantly:

$$
p(A, B) = p(A \mid B)\, p(B).
$$

The joint probability of two things equals the probability of one times the conditional probability of the other given the first.

### 1.1 Joint and marginal

Two more names, both simple:

- The **joint** $p(A, B)$ is the probability of a *combination* of outcomes happening together.
- The **marginal** $p(A)$ is the probability of one variable on its own, obtained by summing the joint over all values of the other: $p(A) = \sum_b p(A, B = b)$. This summing-out is called **marginalization** ("we marginalize out $B$"). It works because the events $B = b$ are exhaustive and mutually exclusive, so summing their joints with $A$ recovers $A$'s total probability.

## 2. Bayes' rule: flipping the condition

Often you know $p(B \mid A)$ but want $p(A \mid B)$ — the condition points the wrong way. **Bayes' rule** flips it. Since $p(A, B) = p(A \mid B)p(B) = p(B \mid A)p(A)$ (the product rule written both ways), equate the two right-hand sides and divide by $p(B)$:

$$
p(A \mid B) = \frac{p(B \mid A)\, p(A)}{p(B)}.
$$

Naming the pieces (the standard vocabulary): $p(A)$ is the **prior** (what you believed about $A$ before seeing $B$), $p(B \mid A)$ is the **likelihood** (how probable the evidence $B$ is if $A$ were true), $p(A \mid B)$ is the **posterior** (your updated belief after seeing $B$), and $p(B)$ is the **evidence** or normalizing constant, computable by marginalizing: $p(B) = p(B \mid A)p(A) + p(B \mid \lnot A)p(\lnot A)$.

### 2.1 Worked example: a medical test

A disease affects $1\%$ of people, so the prior is $p(D) = 0.01$ and $p(\lnot D) = 0.99$. A test is $99\%$ sensitive — it is positive $99\%$ of the time when the disease is present: $p(\text{pos} \mid D) = 0.99$. It has a $5\%$ false-positive rate — positive $5\%$ of the time when the disease is absent: $p(\text{pos} \mid \lnot D) = 0.05$.

You test positive. What is the probability you actually have the disease, $p(D \mid \text{pos})$?

First the evidence $p(\text{pos})$, by marginalizing over having / not having the disease:

$$
p(\text{pos}) = p(\text{pos} \mid D)p(D) + p(\text{pos} \mid \lnot D)p(\lnot D) = (0.99)(0.01) + (0.05)(0.99) = 0.0099 + 0.0495 = 0.0594.
$$

Now Bayes' rule:

$$
p(D \mid \text{pos}) = \frac{p(\text{pos} \mid D)\, p(D)}{p(\text{pos})} = \frac{(0.99)(0.01)}{0.0594} = \frac{0.0099}{0.0594} = 0.1667.
$$

The answer is a famously counterintuitive $\approx 16.7\%$ — despite a "$99\%$ accurate" test, a positive result means only a one-in-six chance of disease. The reason is the tiny prior: the disease is so rare that the false positives from the healthy $99\%$ of people ($0.0495$ worth of probability) swamp the true positives from the sick $1\%$ ($0.0099$). Bayes' rule forces you to weigh the likelihood against the prior instead of reading the test in isolation.

```python
pD, pND = 0.01, 0.99
sens, fpr = 0.99, 0.05                      # p(pos|D), p(pos|not D)
p_pos = sens*pD + fpr*pND                    # evidence, marginal p(pos)
post  = sens*pD / p_pos                       # Bayes: p(D|pos)
print(p_pos, post)                            # 0.0594  0.16666666666666669
```

<div class="callout key"><p>Bayes' rule: posterior $\propto$ likelihood $\times$ prior. A strong likelihood ("the test is 99% accurate") does not override a strong prior ("the disease is very rare"). We revisit the prior/likelihood split when we discuss maximum likelihood below.</p></div>

## 3. The chain rule of probability — this is language modeling

Everything so far builds to this. The product rule $p(A, B) = p(A \mid B)p(B)$ extends to any number of variables. For a sequence of $n$ variables $x_1, x_2, \dots, x_n$, applying the product rule repeatedly gives the **chain rule of probability**:

$$
p(x_1, x_2, \dots, x_n) = \prod_{i=1}^{n} p(x_i \mid x_1, x_2, \dots, x_{i-1}).
$$

Read it left to right: the probability of the whole sequence is the probability of the first element, times the probability of the second given the first, times the probability of the third given the first two, and so on. Each factor conditions on *everything that came before it*. For $i = 1$ the condition is empty, so $p(x_1 \mid \varnothing) = p(x_1)$.

To see it is exact, expand $n = 3$ by the product rule twice:

$$
p(x_1, x_2, x_3) = p(x_3 \mid x_1, x_2)\, p(x_1, x_2) = p(x_3 \mid x_1, x_2)\, p(x_2 \mid x_1)\, p(x_1).
$$

No approximation anywhere — just the definition of conditional probability applied repeatedly.

### 3.1 The language-modeling factorization

Now let the $x_i$ be **tokens** and the sequence be a sentence. The chain rule says

$$
p(\text{sentence}) = p(x_1, \dots, x_n) = \prod_{i=1}^{n} p(x_i \mid x_{<i}),
$$

where $x_{<i}$ is shorthand for "all tokens before position $i$", the **context** or **prefix**. This *is* what a language model computes. At position $i$ the model reads the prefix $x_{<i}$ and outputs a categorical distribution over the vocabulary (lesson 01.1) — that categorical is exactly the factor $p(x_i \mid x_{<i})$. Multiply the factors for the actual tokens in the sentence and you get the sentence's probability.

<div class="callout key"><p>A language model is a function that produces $p(x_i \mid x_{<i})$ for every position. The chain rule guarantees that the product of these per-token conditionals is the exact probability of the whole sequence — so "predict the next token, one at a time" is not a heuristic, it is the full joint distribution factored into pieces the model can actually compute.</p></div>

For example, "the capital of France is Paris" factors as

$$
p(\text{the})\, p(\text{capital} \mid \text{the})\, p(\text{of} \mid \text{the capital}) \cdots p(\text{Paris} \mid \text{the capital of France is}).
$$

A good model makes that last factor large; a model that had never seen the fact would spread its probability thinly and make it small.

## 4. Probability vs likelihood, and maximum likelihood

These two words describe the *same* number $p(\text{data} \mid \text{parameters})$ read in two directions, and keeping them straight matters.

- **Probability**: the parameters (the model) are fixed, and you ask how probable various *data* are. "Given this model, how likely is the sentence `Paris`?" You vary the data.
- **Likelihood**: the *data* are fixed (you observed them), and you ask how well various *parameters* explain them. "Given that I observed this sentence, how good is this model?" You vary the parameters.

Training a model is the second view. We hold the observed training text fixed and search for the model parameters $\theta$ that make that text as probable as possible. This is **maximum likelihood estimation** (MLE):

$$
\theta^\star = \arg\max_\theta \; p(\text{training data} \mid \theta) = \arg\max_\theta \; \prod_{i} p(x_i \mid x_{<i}; \theta).
$$

The likelihood of the whole corpus is, by the chain rule, the product of every next-token probability the model assigns to the true tokens. MLE turns "learn a language model" into "adjust $\theta$ so the model assigns high probability to the text that actually occurred". Lesson 01.4 turns this maximization into the loss we minimize.

## 5. Why we sum log-probabilities instead of multiplying

The likelihood is a product of many probabilities, each between 0 and 1. Multiplying thousands of sub-1 numbers drives the result toward zero fast — and in finite-precision floating point it eventually hits *exactly* zero, a catastrophe called **underflow**. Once the product is zero, its logarithm is $-\infty$ and every gradient is garbage.

The fix is to work in **log space**. Because $\log(ab) = \log a + \log b$, the log of a product is the sum of the logs:

$$
\log \prod_{i=1}^{n} p(x_i \mid x_{<i}) = \sum_{i=1}^{n} \log p(x_i \mid x_{<i}).
$$

The right-hand side is the **log-likelihood**. Each $\log p$ is a moderate negative number (since $0 < p \le 1$ means $\log p \le 0$), and summing them stays comfortably within floating-point range no matter how long the sequence. Maximizing the likelihood and maximizing the log-likelihood give the same $\theta^\star$, because $\log$ is monotonically increasing — so we always optimize the log-likelihood.

### 5.1 Worked example: a tiny product vs its log-sum

Take five token probabilities from a sentence: $0.1, 0.2, 0.05, 0.3, 0.15$. Their product is

$$
0.1 \times 0.2 \times 0.05 \times 0.3 \times 0.15 = 4.5 \times 10^{-5},
$$

already small for only five tokens. The log-likelihood is

$$
\ln 0.1 + \ln 0.2 + \ln 0.05 + \ln 0.3 + \ln 0.15 = -10.0088,
$$

a tame number. (Exponentiating it, $e^{-10.0088} = 4.5 \times 10^{-5}$, recovers the product — the two are the same quantity in different spaces.)

Now watch a realistic product underflow. Real sentences have hundreds of tokens with small probabilities:

```python
import math, numpy as np

# five tokens: product vs log-sum agree
ps = [0.1, 0.2, 0.05, 0.3, 0.15]
prod = 1.0
for v in ps: prod *= v
print(prod)                                   # 4.500000000000001e-05
print(sum(math.log(v) for v in ps))           # -10.008848068193954
print(math.exp(sum(math.log(v) for v in ps))) # 4.5000000000000016e-05  (same number)

# underflow: multiply twenty very small float32 probabilities
small = np.float32(1e-30)
prodf = np.float32(1.0)
for _ in range(20):
    prodf = np.float32(prodf * small)
print(prodf)                                   # 0.0   <- underflowed to exactly zero
print(20 * math.log(1e-30))                    # -1381.5510557964274  <- log-sum survives
```

The float32 product of twenty $10^{-30}$ probabilities is **exactly `0.0`** — the true value $10^{-600}$ is far below the smallest positive float32 (about $1.2 \times 10^{-38}$), so it rounds to zero and all information is lost. The log-sum, $-1381.55$, is perfectly representable and carries the full information. This is why every language-model loss is computed as a sum of log-probabilities, never a product of raw probabilities.

<div class="callout warn"><p><strong>Pitfall:</strong> even a few hundred tokens with typical probabilities like $10^{-2}$ give a product around $10^{-400}$, well past float32 <em>and</em> float64 underflow (float64 bottoms out near $10^{-308}$). Never compute a sequence probability by multiplying; always sum log-probabilities. Correspondingly, take the log of a probability with a numerically stable path (log-softmax, lesson 01.4), not <code>log(softmax(x))</code>, which can take the log of a rounded-to-zero value and return $-\infty$.</p></div>

## Common mistakes

- **Reading a test/likelihood without the prior.** The medical example shows a $99\%$ test giving a $16.7\%$ posterior. Ignoring the prior $p(D)$ is the classic base-rate fallacy.
- **Writing $p(A \mid B) = p(B \mid A)$.** They are equal only in special cases. Bayes' rule relates them, with the prior ratio as the correction; they are not interchangeable.
- **Multiplying probabilities in code.** Any loop that does `prob *= p_i` over a real sequence will underflow. Accumulate `logprob += log(p_i)` instead.
- **Confusing the direction of conditioning in the chain rule.** Each factor is $p(x_i \mid x_{<i})$ — the token given its *past*, never its future. Conditioning on future tokens would let the model cheat, which is exactly what the causal mask in the attention module prevents.

## Exercise

A spam filter sees the word `free`. Historically $p(\texttt{free} \mid \text{spam}) = 0.6$ and $p(\texttt{free} \mid \text{ham}) = 0.05$, and $40\%$ of mail is spam ($p(\text{spam}) = 0.4$). Given an email contains `free`, what is $p(\text{spam} \mid \texttt{free})$?

<details><summary>Optional hint</summary>

This is Bayes' rule. First compute the evidence $p(\texttt{free})$ by marginalizing over spam and ham.

</details>

<details><summary>Stronger hint</summary>

$p(\texttt{free}) = p(\texttt{free}\mid\text{spam})p(\text{spam}) + p(\texttt{free}\mid\text{ham})p(\text{ham})$, with $p(\text{ham}) = 1 - 0.4 = 0.6$. Then divide the spam contribution by that total.

</details>

<details><summary>Solution</summary>

Evidence: $p(\texttt{free}) = (0.6)(0.4) + (0.05)(0.6) = 0.24 + 0.03 = 0.27$.

Posterior: $p(\text{spam} \mid \texttt{free}) = \dfrac{(0.6)(0.4)}{0.27} = \dfrac{0.24}{0.27} \approx 0.8889$.

```python
p_free = 0.6*0.4 + 0.05*0.6
print(0.6*0.4 / p_free)      # 0.888888888888889
```

So seeing `free` pushes the spam probability from the $40\%$ prior up to about $88.9\%$ — here the likelihood is strong enough to move the posterior a lot, the opposite of the rare-disease case.

</details>

## Check yourself

<details><summary>State the chain rule of probability and explain in one sentence why it makes next-token prediction exact rather than approximate.</summary>

$p(x_1,\dots,x_n) = \prod_{i=1}^{n} p(x_i \mid x_{<i})$. It is a direct consequence of the definition of conditional probability applied repeatedly, so the product of per-token conditional probabilities equals the exact joint probability of the whole sequence — no approximation is introduced by factoring the sentence into next-token steps.

</details>

<details><summary>Why do we maximize the sum of log-probabilities instead of the product of probabilities?</summary>

Two reasons, one numerical and one about equivalence. Numerically, a product of many sub-1 probabilities underflows to zero in floating point, whereas the sum of their logs stays in range. And because $\log$ is monotonically increasing, the parameters that maximize the log-likelihood are the same ones that maximize the likelihood, so we lose nothing by optimizing the log.

</details>

<details><summary>In the medical example, why is the posterior only 16.7% when the test is 99% sensitive?</summary>

Because the disease is rare (prior $1\%$). The healthy $99\%$ of people generate a large absolute number of false positives ($0.05 \times 0.99 = 0.0495$) that dwarfs the true positives from the sick $1\%$ ($0.99 \times 0.01 = 0.0099$). Bayes' rule weighs the likelihood against the prior, and the small prior keeps the posterior low.

</details>

<details><summary>A sequence has 300 tokens each with probability about 0.01. Roughly what is the product, and why is that a problem in float32?</summary>

About $0.01^{300} = 10^{-600}$. The smallest positive float32 is around $1.2 \times 10^{-38}$ (float64 bottoms out near $10^{-308}$), so $10^{-600}$ underflows to exactly $0.0$ in either precision, destroying the value. The log-likelihood, $300 \times \ln(0.01) \approx -1382$, is perfectly representable — which is why we work in log space.

</details>

## Next

You now have the chain rule — the exact factorization of a sentence into next-token probabilities — and you know training a language model is maximum likelihood on that product, done in log space. The next lesson introduces the tools that measure the *quality* of a predicted distribution: entropy, cross-entropy, and KL divergence. Cross-entropy is, up to a sign and an average, the negative log-likelihood you just met — and it is the loss you will minimize.

Continue to [01.3 · Entropy, cross-entropy, KL divergence](lesson-03.md).
