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

Had the true token instead been token 3 (the one the model disliked), the loss would be $-\ln(0.0321) = 3.44$ nats — far higher, correctly punishing the model for putting little mass on what actually occurred.

## 4. The full objective: mean cross-entropy over all positions

One position gives one loss. A language model is trained over a whole corpus, so the **objective** is the average of the per-token cross-entropies across every position it predicts. For a sequence of tokens $x_1, \dots, x_N$ (or a batch of them), with $q_i$ the model's predicted distribution at position $i$ and $t_i$ the true token there:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \big(-\log q_i(t_i)\big) = -\frac{1}{N} \sum_{i=1}^{N} \log q_i(t_i).
$$

This is the mean negative log-likelihood — and by the chain rule of lesson 01.2, $\sum_i \log q_i(t_i)$ is the log-probability of the entire sequence. So minimizing $\mathcal{L}$ is maximizing the likelihood of the training text, averaged per token. We average (rather than sum) so the loss is comparable across sequences of different lengths and batch sizes; it is a per-token quantity in nats.

<div class="callout key"><p>The language-modeling loss is the mean, over all predicted positions, of $-\log q_i(t_i)$: the negative log-probability the model assigned to each true next token. Minimizing it = maximum likelihood = pushing the model's softmax distribution toward the observed tokens. This one formula is what every pretraining run optimizes.</p></div>

## 5. Tensor shapes: what this looks like in a real model

In code the logits are not a single vector but a batched tensor. Naming the axes:

- **Logits:** shape `(B, T, V)`, dtype float32 (often bf16 in training), on the GPU. `B` sequences, each of `T` positions, each position a length-`V` logit vector.
- **Targets:** shape `(B, T)`, dtype `int64` (`long`), same device. Each entry is the true next-token index in `[0, V)`.
- The loss reshapes logits to `(B*T, V)` and targets to `(B*T,)`, computes one cross-entropy per row, and averages to a **scalar** (shape `()`).

The `V` axis is always the one softmax normalizes over and the one the true index selects from. The reshape from `(B, T, V)` to `(N, V)` with `N = B*T` is just "treat every (sequence, position) as an independent classification example"; the chain rule guarantees averaging them is the right thing.

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

Two implementation points that matter for correctness:

**Why subtract the max.** Computing $\log \frac{\exp(z_i)}{\sum_j \exp(z_j)}$ naively exponentiates the logits, and a large logit like $z_i = 90$ makes $\exp(90) \approx 10^{39}$ overflow float32 to `inf`. Subtracting the per-row max $m$ before exponentiating means the largest value fed to `exp` is $\exp(0) = 1$, so nothing overflows. Because a constant shift cancels between the numerator and denominator of softmax, this changes nothing about the result — it is exact, not an approximation. This is the same stability trick you will see again in attention (Module 5) and FlashAttention (Module 8).

**Why log-softmax and not `log(softmax(x))`.** Taking softmax first can round a tiny probability to exactly `0.0`, and then `log(0) = -inf`. Folding the log into the computation (the `shifted - lse` form) never materializes that zero, so the log-probability stays finite. This is the code embodiment of the "sum log-probs, don't multiply probs" rule from lesson 01.2.

### 6.1 Verify the by-hand loss against PyTorch

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
- **Compute.** Softmax and the log are $O(B \cdot T \cdot V)$ elementwise-ish work; the dominant cost is usually the preceding matrix multiply that produces the logits (the `(C → V)` output projection), which is $O(B \cdot T \cdot C \cdot V)$.
- **Gradient.** The gradient of cross-entropy with respect to the logits has an elegant closed form, $\partial \mathcal{L} / \partial z_i = q_i - p_i$ — the predicted distribution minus the one-hot target. For our worked example the gradient on the true token 0 is $0.6439 - 1 = -0.3561$ (push that logit up) and on token 3 it is $0.0321 - 0 = +0.0321$ (push it down). We derive this backward pass by hand in [Module 2](lessons/module-02/lesson-03.md); for now, note the loss and its gradient are both simple functions of the softmax output.

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

## Next

You have now built the complete language-modeling objective — softmax over the vocabulary, cross-entropy against the true token, averaged over positions — verified it by hand and against PyTorch and the `llmre` code, and learned to read it as perplexity. This closes Module 1: you can state exactly what a language model computes and exactly how its predictions are scored. Module 2 opens the other half of training — how the gradient of this loss flows back through the network via backpropagation, starting with derivatives and the chain rule of calculus.

Continue to [02.1 · Derivatives, partials, chain rule, gradients](lessons/module-02/lesson-01.md).
