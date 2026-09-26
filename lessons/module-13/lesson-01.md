# 13.1 · Loss, perplexity, zero-/few-shot

<div class="prereq">
<p><strong>Prerequisites:</strong> the language-modeling objective and perplexity from <a href="#/lessons/module-01/lesson-04">01.4 · The language-modeling objective &amp; perplexity</a>; cross-entropy from <a href="#/lessons/module-01/lesson-03">01.3 · Entropy, cross-entropy, KL</a>; the assembled model and its <code>forward(idx, targets) -&gt; (logits, loss)</code> from <a href="#/lessons/module-06/lesson-01">06.1 · Assembling GPT-2</a>; softmax over a vocabulary from <a href="#/lessons/module-05/lesson-02">05.2 · scaled dot-product attention</a>.</p>
<p><strong>You will learn:</strong> the difference between training and validation loss and what their gap tells you; why <strong>perplexity</strong> is the reportable number and how it relates to loss; what a held-out set is and how to build one honestly; the difference between <strong>zero-shot</strong> and <strong>few-shot</strong> (in-context) evaluation, and why few-shot is a prompting method rather than training; and how to grade a multiple-choice question by scoring each option's <strong>length-normalized log-likelihood</strong> — worked by hand on tiny numbers.</p>
<p><strong>Why this matters for ML:</strong> you cannot improve what you cannot measure. Every architecture change, data change, and hyperparameter sweep in the rest of this course is judged by a number produced here. Getting evaluation right — an honest held-out set, the correct metric, a fair scoring rule — is the difference between real progress and fooling yourself.</p>
</div>

## 1. Training loss vs validation loss

You already know the training signal from <a href="#/lessons/module-01/lesson-04">01.4</a>: the model outputs a probability distribution over the vocabulary at each position, and the loss is the **mean cross-entropy** between that distribution and the true next token, measured in nats. During training we compute this loss on the batches the optimizer is learning from — the **training loss**.

But the training loss alone cannot tell you whether the model is *learning the language* or merely *memorizing the training tokens*. A model with enough parameters can drive training loss arbitrarily low by memorizing, while getting no better at text it has not seen. To detect that, we hold out a second stream of text the optimizer never touches — the **validation set** — and periodically measure the loss on it. That is the **validation loss**.

- **Training loss** falls steadily as the optimizer fits the training tokens.
- **Validation loss** falls too, as long as the model is learning generalizable structure.
- When validation loss stops falling and starts to *rise* while training loss keeps falling, the model has begun **overfitting**: it is now fitting quirks specific to the training tokens that do not transfer.

<div class="callout key"><p>Report and make decisions on the <strong>validation</strong> (held-out) loss, never the training loss. Training loss measures memorization plus learning tangled together; validation loss measures only what generalizes, which is the thing you actually care about.</p></div>

For large-scale LM pretraining the story has a twist: models are trained on so much data for so few passes (often a single epoch) that classic overfitting — validation loss turning upward — is rare during the main run. The training and validation curves sit almost on top of each other. That does *not* make the held-out set optional; it is still the only honest measurement, and it is what catches bugs (a broken data pipeline shows up as a validation loss that will not move) and what you compare across experiments.

## 2. Perplexity: the reportable metric

Loss in nats is the right quantity to differentiate and optimize, but it is not intuitive to report. The convention is to exponentiate it into **perplexity**. You built this in <a href="#/lessons/module-01/lesson-04">01.4</a>; here is the one-line recap.

If the mean cross-entropy over a held-out set is $L$ nats, the perplexity is

$$
\mathrm{PPL} = \exp(L).
$$

Because $L = -\frac{1}{N}\sum_{i} \ln q(x_i)$ is the mean negative log-probability the model assigns to the true tokens, its exponential reads as an **effective branching factor**: "on average, at each position the model is as uncertain as if it were choosing uniformly among $\mathrm{PPL}$ equally likely tokens."

Two anchors worth memorizing:

- A **perfect** model assigns probability $1$ to every true token, so $L = 0$ and $\mathrm{PPL} = 1$ — no uncertainty, one choice.
- A **uniform** model over a vocabulary of size $V$ assigns $1/V$ to every token, so $L = \ln V$ and $\mathrm{PPL} = V$ — maximal uncertainty, all $V$ tokens equally likely.

A tiny numerical check, verified in Python:

| mean loss $L$ (nats) | perplexity $\exp(L)$ |
| --- | --- |
| $0.0$ | $1.0$ |
| $\ln 2 \approx 0.6931$ | $2.0$ |
| $3.0$ | $20.0855$ |
| $\ln 50257 \approx 10.8249$ | $50257.0$ |

The last row is the sanity check every practitioner runs: a freshly initialized model, before any training, should score a loss of about $\ln V$ and a perplexity of about $V$ (here GPT-2's vocabulary, $V = 50257$). If your untrained model reports a wildly different number, something in the loss or the data pipeline is wrong — catch it before you waste GPU-hours.

<div class="callout warn"><p>Perplexity is only comparable between models that share the <strong>same tokenizer and vocabulary</strong>. A model with a bigger vocabulary spreads probability over more tokens and needs fewer tokens per word, so its per-token perplexity is not comparable to a smaller-vocab model's. Never compare perplexities across tokenizers; compare loss/perplexity only within a fixed vocabulary, or switch to bits-per-byte if you must cross tokenizers.</p></div>

We turn this recap into a reusable `evaluate_perplexity` function over a held-out stream in <a href="#/lessons/module-13/lesson-03">13.3</a>.

## 3. Held-out evaluation, done honestly

A held-out set is only meaningful if the model genuinely never saw it. Three rules:

1. **Split before you train.** Carve the validation (and test) stream off the corpus before the training run begins, and keep it fixed for the life of the project. A validation set that changes between experiments cannot be used to compare experiments.
2. **No leakage.** If the same document, or a near-duplicate, appears in both training and validation, the validation loss is measuring memorization, not generalization. This is the deduplication problem of <a href="#/lessons/module-11/lesson-02">11.2</a> applied across the split, and its evaluation cousin — benchmark contamination — is the subject of <a href="#/lessons/module-13/lesson-02">13.2</a>.
3. **Validation vs test.** You tune hyperparameters against the *validation* set, so over many experiments you slowly fit to it too. Keep a separate **test** set that you look at rarely (ideally once, at the end) for an unbiased final number.

## 4. Zero-shot vs few-shot: evaluating without fine-tuning

So far "evaluation" has meant measuring loss on held-out text. But we also want to know whether the model can *do tasks* — answer questions, complete analogies, pick the right ending. A pretrained language model was never explicitly trained to do these; it was only trained to predict the next token. The discovery of GPT-2 and especially GPT-3 is that task ability emerges anyway, and you elicit it purely through the **prompt** — the text you condition on — with **no weight updates at all**.

There are two ways to prompt:

- **Zero-shot.** You give the model only a description of the task and the input, then read its continuation. Example prompt:
  > `Translate English to French.\nEnglish: cheese\nFrench:`

  The model must infer what to do from the instruction alone.

- **Few-shot (in-context learning).** You prepend $k$ worked examples ("shots") of the task before the real input, all in the prompt:
  > `Translate English to French.\nEnglish: sea otter\nFrench: loutre de mer\nEnglish: cheese\nFrench:`

  The model reads the pattern from the examples and continues it. One example is "one-shot"; $k$ examples is "$k$-shot".

<div class="callout key"><p>Few-shot learning is a <strong>prompting method, not training</strong>. The $k$ examples live in the context window and influence only this one forward pass; no gradient is computed and no weight changes. The instant the prompt ends, the model has "learned" nothing permanent. This is why it is called <em>in-context</em> learning — the learning is entirely inside the context, not in the parameters.</p></div>

Why does putting examples in the prompt help? Pretraining on a huge, varied corpus has taught the model to recognize and continue patterns; a few demonstrations pin down *which* pattern (which task, which format) you mean, sharpening the distribution over the next tokens. More shots generally help up to a point, and — a central result of GPT-3 — the *benefit* of few-shot examples grows with model scale: larger models make far better use of the same in-context examples.

<div class="callout paper"><p>This behavior is the subject of GPT-3, "Language Models are Few-Shot Learners" (Brown et al., 2020) — see the <a href="#/papers/index">paper curriculum</a>. Look at their figure showing few-shot accuracy rising with parameter count, and the curve of accuracy vs number of in-context examples.</p></div>

Both zero- and few-shot evaluation change only the prompt. The scoring machinery below is identical for both — the number of shots is just how much text sits in front of the question.

## 5. Scoring a multiple-choice question by log-likelihood

Many benchmarks (ARC, HellaSwag, MMLU, and friends) are multiple-choice: a context, and several candidate answers, exactly one correct. A pretrained model does not "click" an option; it assigns probabilities to token sequences. So we grade it by asking a precise question of each option:

> Given the context, how likely does the model think *this option's tokens* are?

Then we pick the option with the highest likelihood. That is the whole method. Two details make it fair.

### 5.1 Score the continuation, not the context

For option $o$ we form the full sequence `[context tokens] + [option tokens]`, run it through the model once, and read off the log-probability the model assigned to each **option** token given everything before it:

$$
\ell_o = \sum_{t \in \text{option}} \ln q\bigl(x_t \mid x_{<t}\bigr).
$$

We include only the option tokens in the sum. The context is shared by every option, so its log-probability is the same constant for all of them and would not change which option wins — but summing it in would drown the signal, so we leave it out. (This is exactly what `sequence_loglikelihood` computes for a whole sequence; the multiple-choice scorer restricts the sum to the continuation. Both are in <a href="#/lessons/module-13/lesson-03">13.3</a>.)

### 5.2 Length-normalize

Every term $\ln q(\cdot)$ is negative (a log of a probability $\le 1$). So a **longer** option accumulates more negative terms and looks less likely purely for being longer — even when it is the better answer. If the options have different token lengths, comparing raw sums is biased toward short options.

The fix is **length normalization**: divide each option's summed log-likelihood by its token count, comparing the **mean per-token log-probability** instead of the sum:

$$
\bar\ell_o = \frac{1}{|o|}\sum_{t \in \text{option}} \ln q\bigl(x_t \mid x_{<t}\bigr),
\qquad \hat o = \arg\max_o \bar\ell_o .
$$

### 5.3 Worked example — why normalization matters

Take a question with two options that tokenize to different lengths. Suppose the model's per-token log-probabilities (nats) for the option tokens, given the context, come out as:

- **Option A** — 1 token: $[-1.0]$.
- **Option B** — 4 tokens: $[-0.4,\ -0.3,\ -0.4,\ -0.3]$.

Score them both ways (verified in Python):

$$
\text{sum: } \ell_A = -1.0, \quad \ell_B = -1.4 \;\Rightarrow\; \arg\max = A.
$$

$$
\text{normalized: } \bar\ell_A = \frac{-1.0}{1} = -1.0, \quad \bar\ell_B = \frac{-1.4}{4} = -0.35 \;\Rightarrow\; \arg\max = B.
$$

The two rules **disagree**. The summed rule crowns A only because A is shorter — a single mildly-confident token beats four *more*-confident tokens simply by having fewer negative terms. The normalized rule looks at confidence per token and picks B, whose tokens are each more probable. When B is the correct answer, length normalization is what lets the model get it right. This is why lm-eval-harness scores tasks like HellaSwag by length-normalized log-likelihood.

<div class="callout warn"><p>Length normalization is a heuristic, not a law. Dividing by token count assumes each token carries equal weight, which is not exactly true; some benchmarks normalize by the number of characters or bytes instead, and a few score by the raw sum on purpose. The point is not that one rule is universally correct — it is that the scoring rule is <em>part of the benchmark definition</em>, and two papers reporting "HellaSwag accuracy" with different normalizations are not directly comparable.</p></div>

## 6. In code

The `multiple_choice_score` function in `llmre.evaluation.harness` implements exactly section 5: build `[context; option]`, run the model once per option, sum the option tokens' log-probabilities from the numerically stable `log_softmax` of <a href="#/lessons/module-01/lesson-04">module 1</a>, optionally divide by the option length, and return the argmax index.

```python
import torch
from llmre.evaluation.harness import multiple_choice_score

# A toy model that assigns known per-token probabilities lets us reproduce the
# section 5.3 numbers exactly (see code/tests/test_eval.py for the full stub).
context = torch.tensor([0])                      # the shared prompt, (Lc,) long
option_a = torch.tensor([2])                     # 1 low-confidence token
option_b = torch.tensor([1, 1, 1, 1])            # 4 higher-confidence tokens

# Summed log-likelihood favors the shorter option A ...
multiple_choice_score(model, context, [option_a, option_b], length_normalize=False)  # -> 0
# ... but length normalization favors option B, whose per-token log-prob is higher.
multiple_choice_score(model, context, [option_a, option_b], length_normalize=True)   # -> 1
```

The function is deliberately model-agnostic: it calls `model(seq) -> (logits, loss)`, the same interface `GPT.forward` exposes, so it works on your from-scratch GPT and on any object that returns logits of shape $(B, T, V)$. Shapes, dtypes, and the exact index arithmetic (the logits at position $p$ predict token $p+1$, so the option's tokens are predicted by positions $L_c-1 \dots L_c+L_o-2$) are documented on the function and reused in <a href="#/lessons/module-13/lesson-03">13.3</a>.

## Exercise

You are evaluating a model on a two-option question. The context is fixed. Option X tokenizes to 2 tokens with log-probs $[-0.8, -0.9]$; option Y tokenizes to 5 tokens with log-probs $[-0.5, -0.6, -0.5, -0.5, -0.6]$. Which option wins under the summed rule, and which under length normalization?

<details><summary>Optional hint</summary>

Add up each option's log-probs for the sum; divide by the token count for the normalized score. Remember all the numbers are negative, so "larger" means "closer to zero".

</details>

<details><summary>Stronger hint</summary>

Sum X $= -0.8 - 0.9 = -1.7$. Sum Y $= -0.5 -0.6 -0.5 -0.5 -0.6 = -2.7$. For normalization, divide X's sum by 2 and Y's by 5.

</details>

<details><summary>Solution</summary>

**Summed:** $\ell_X = -1.7$, $\ell_Y = -2.7$. Since $-1.7 > -2.7$, the summed rule picks **X**.

**Normalized:** $\bar\ell_X = -1.7 / 2 = -0.85$, $\bar\ell_Y = -2.7 / 5 = -0.54$. Since $-0.54 > -0.85$, the normalized rule picks **Y**.

They disagree, and for the same reason as the worked example: Y is longer, so its summed log-likelihood is dragged down by having more (negative) terms, even though each of its tokens is more confident on average. Length normalization corrects the length bias and selects Y.

</details>

## Common mistakes

- **Reporting training loss.** If your loss curve looks suspiciously good, check you are plotting the held-out loss, not the loss on the batches you are training on.
- **Comparing perplexity across tokenizers.** Two models with different vocabularies have incomparable per-token perplexities. Fix the tokenizer or switch to bits-per-byte.
- **Including the context in the option score.** Only the option's tokens go into $\ell_o$; summing the shared context in adds a constant and buries the signal.
- **Forgetting to length-normalize** when options differ in length — you will systematically over-pick the shortest option.
- **Calling few-shot "fine-tuning".** No weights change in few-shot; the examples only live in the prompt for one forward pass.

## Debugging exercise

A colleague's zero-shot multiple-choice scorer always picks the option with the fewest tokens, regardless of content. The scoring line reads:

```python
score = sum(log_probs_of_option_tokens)      # then pick argmax over options
```

What is the bug, and what one change fixes it?

<details><summary>Solution</summary>

The scorer uses the **summed** log-likelihood without length normalization. Because every per-token log-probability is negative, options with more tokens accumulate a more negative sum, so the shortest option almost always has the largest (least negative) sum and wins regardless of correctness. The fix is to divide by the number of option tokens (or otherwise normalize by length) and take the argmax of the mean per-token log-probability:

```python
score = sum(log_probs_of_option_tokens) / len(log_probs_of_option_tokens)
```

This is exactly the `length_normalize=True` path of `multiple_choice_score`.

</details>

## Check yourself

<details><summary>A newly initialized model on a vocabulary of 32000 tokens reports a validation loss of 10.373 nats. Is that expected, and what perplexity does it correspond to?</summary>

Yes. An untrained model is roughly uniform over the vocabulary, so its loss should be about $\ln V = \ln 32000 \approx 10.373$ nats, and its perplexity about $\exp(10.373) \approx 32000 = V$. This is the standard "is my loss wired up correctly?" sanity check. A very different starting number signals a bug in the data pipeline or loss.

</details>

<details><summary>Your training loss keeps dropping but the validation loss bottomed out 2000 steps ago and is now creeping up. What is happening and what would you do?</summary>

The model is **overfitting**: it is still reducing loss on the training tokens (memorizing their quirks) but those gains no longer transfer, so held-out loss rises. Responses include early-stopping at the validation minimum, adding regularization (weight decay, dropout), getting more or less-duplicated data, or reducing model capacity. You would report and checkpoint at the validation minimum, not the training minimum.

</details>

<details><summary>Why do we score only the option's tokens and not the context tokens when grading a multiple-choice question?</summary>

The context is identical across all options, so its total log-probability is the same additive constant for every option and cannot change which option has the highest score. Including it only adds noise (a large shared magnitude) that can swamp the small differences between options. Scoring just the continuation isolates the quantity that actually distinguishes the answers.

</details>

<details><summary>Is 5-shot evaluation a form of training? Justify your answer in one sentence.</summary>

No — the five examples sit in the prompt and affect only the current forward pass; no gradient is taken and no weight is updated, so nothing is learned permanently. It is in-context (prompt-time) learning, not training.

</details>

<details><summary>Two options tokenize to lengths 1 and 3. The 1-token option has log-prob -2.0; the 3-token option has log-probs [-0.5, -0.5, -0.5]. Which wins under each rule?</summary>

Summed: $-2.0$ vs $-1.5$; the 3-token option wins ($-1.5 > -2.0$). Normalized: $-2.0/1 = -2.0$ vs $-1.5/3 = -0.5$; the 3-token option wins again ($-0.5 > -2.0$). Here both rules agree on the longer option, because its per-token confidence is high enough that even its summed score beats the short option. Normalization does not always flip the winner — it only matters when the length bias would otherwise decide the outcome.

</details>

## Next

You can now measure held-out loss, report perplexity, and grade zero-/few-shot multiple-choice tasks by length-normalized log-likelihood. But a high benchmark score can be a lie — if the test questions leaked into training, or if the model's confidence does not match its accuracy. The next lesson confronts what these numbers do and do not mean: contamination, calibration, and the families of task evaluation.

Continue to [13.2 · Contamination, calibration, task eval](lessons/module-13/lesson-02.md).
