# 15.1 · Bradley-Terry & the reward model

<div class="prereq">
<p><strong>Prerequisites:</strong> supervised fine-tuning from <a href="../module-14/lesson-01.md">14.1 · Instruction data & chat templates</a> and the response-only loss of <a href="../module-14/lesson-02.md">14.2 · Loss masking</a>; the assembled model and its hidden states from <a href="../module-06/lesson-01.md">06.1 · Assembling GPT-2</a>; the log-softmax / cross-entropy machinery from <a href="../module-01/lesson-04.md">01.4</a> and <code>llmre.evaluation.metrics</code>.</p>
<p><strong>You will learn:</strong> why imitation (SFT) cannot rank answers by quality; what pairwise preference data <code>(x, y_w, y_l)</code> is; the <strong>Bradley-Terry</strong> model that turns a scalar reward into a probability of preference; and how to build and train a <strong>reward model</strong> — a language model with a scalar head — with the loss <code>-log &sigma;(r_w - r_l)</code>.</p>
<p><strong>Why this matters for ML:</strong> the reward model is the numerical definition of "good" that the whole RLHF pipeline (lesson 15.2) optimizes against. Get it wrong and the policy will faithfully optimize the wrong thing. Even DPO (lesson 15.3), which trains no explicit reward model, is <em>derived from</em> exactly this Bradley-Terry loss — so this lesson is the foundation of the entire module.</p>
</div>

## The causal story: imitation has a ceiling

Module 14 gave us a model that *follows instructions*. SFT works by imitation: we showed the model thousands of `(prompt, ideal response)` pairs and trained it, with next-token cross-entropy, to reproduce the response. That is a real capability jump — the model now answers questions instead of continuing them.

But imitation has a structural blind spot. **Every demonstration is treated as equally correct.** The loss only ever says "make this exact response more likely." It never says "this response is better than that one." So SFT can teach the model *a* way to answer, but it cannot teach it to prefer a *good* answer over a *mediocre* one, because it never sees the two side by side.

<div class="callout key"><p>SFT (Module 14) makes the model follow instructions, but it cannot tell a <strong>good</strong> answer from a mediocre one — it only imitates. To optimize for human preference we need a signal that <em>ranks</em> answers. That signal is a <strong>reward model</strong>, and this lesson builds it.</p></div>

Two more concrete reasons imitation is not enough:

- **We can judge more easily than we can demonstrate.** Writing the *ideal* answer to "explain quantum tunneling to a 10-year-old" is hard; deciding which of two answers is *better* is easy. Human labelers are far more reliable at comparisons than at authoring gold responses.
- **The best behavior may be off the SFT distribution.** Imitation can only reach the quality of its demonstrations. Preference optimization can push *past* them, toward whatever humans actually rank highest.

## Preference data: pairs, not scores

Humans are bad at absolute scores ("rate this answer 7.3 / 10") and good at comparisons ("A is better than B"). So preference datasets are collected as **pairwise comparisons**. Each record is a triple:

$$
(x,\; y_w,\; y_l)
$$

- $x$ — the **prompt** (an instruction, possibly with conversation history).
- $y_w$ — the **chosen** (winning) completion; the subscript $w$ is for *win*.
- $y_l$ — the **rejected** (losing) completion; $l$ is for *lose*.

A labeler saw both completions for the same prompt and marked $y_w$ as the better one. That is the whole label: a *direction*, not a magnitude. Anthropic's HH-RLHF dataset (helpful/harmless) and OpenAI's InstructGPT data are exactly this shape — thousands of `(prompt, chosen, rejected)` rows.

<div class="callout warn"><p>The two completions share the <strong>same prompt</strong> <span>$x$</span>. A preference is always relative to a context; "this answer is good" only means "good <em>for this question</em>." When we score a completion below, the reward function always takes both: <span>$r(x, y)$</span>, never <span>$r(y)$</span> alone.</p></div>

## The Bradley-Terry model

### 1. Intuition

We want a scalar function $r(x, y)$ — the **reward** — that assigns a higher number to better completions. But our data is comparisons, not numbers. The Bradley-Terry model is the bridge: it says the *probability* that $y_w$ beats $y_l$ is a smooth, increasing function of the *difference* in their rewards. If $y_w$'s reward is much higher, the win is nearly certain; if the rewards are equal, it is a coin flip.

The function that maps a real-valued difference to a probability in $(0, 1)$ is the **logistic sigmoid** $\sigma(z) = 1 / (1 + e^{-z})$ — the same sigmoid from logistic regression.

### 2. Mathematics (every symbol named)

$$
P(y_w \succ y_l \mid x) \;=\; \sigma\!\big(r(x, y_w) - r(x, y_l)\big)
\;=\; \frac{1}{1 + \exp\!\big(-(r(x,y_w) - r(x,y_l))\big)}
$$

- $P(y_w \succ y_l \mid x)$ — probability the model predicts that $y_w$ is preferred over $y_l$ given prompt $x$. The symbol $\succ$ reads "is preferred to".
- $r(x, y)$ — the scalar reward the reward model assigns to completion $y$ for prompt $x$.
- $r(x, y_w) - r(x, y_l)$ — the **reward margin**. This is the only thing the probability depends on.
- $\sigma$ — logistic sigmoid, mapping any real number to $(0,1)$, with $\sigma(0) = 0.5$.

Notice the reward appears **only through the difference**. Adding a constant $c$ to *every* reward, $r \to r + c$, leaves every margin — and hence every probability — unchanged. So the reward is only defined **up to an additive constant**: absolute reward values are meaningless, only differences are.

To train $r$, we do maximum likelihood: choose $r$ to maximize the probability of the preferences we actually observed. Maximizing $\log P$ over the dataset is the same as minimizing its negative:

$$
\mathcal{L}_{\text{RM}} \;=\; -\,\mathbb{E}_{(x, y_w, y_l)}\big[\log \sigma\big(r(x, y_w) - r(x, y_l)\big)\big]
$$

For a single pair this is $-\log \sigma(r_w - r_l)$, where we abbreviate $r_w = r(x, y_w)$, $r_l = r(x, y_l)$. This is exactly binary cross-entropy where the "logit" is the reward margin and the label is always "chosen wins."

### 3. Numerical example (worked by hand, verified)

Take a single pair with rewards $r_w = 2.0$, $r_l = 1.0$. The margin is $2.0 - 1.0 = 1.0$.

$$
\sigma(1.0) = \frac{1}{1 + e^{-1}} = \frac{1}{1 + 0.367879} = 0.731059
$$

$$
\mathcal{L} = -\log(0.731059) = 0.313262
$$

Now widen the margin. As $r_w - r_l$ grows, $\sigma \to 1$ and the loss $\to 0$; the model is confident and correct, so it is barely penalized:

| $r_w$ | $r_l$ | margin | $\sigma(\text{margin})$ | loss $-\log\sigma$ |
|------:|------:|-------:|------------------------:|-------------------:|
| 0.0   | 0.0   | 0.0    | 0.500000                | 0.693147           |
| 2.0   | 1.0   | 1.0    | 0.731059                | 0.313262           |
| 2.0   | 0.0   | 2.0    | 0.880797                | 0.126928           |
| 3.0   | -1.0  | 4.0    | 0.982014                | 0.018150           |

The loss is **strictly decreasing in the margin**: pushing the chosen reward above the rejected reward is exactly what minimizing it does. At margin $0$ the loss is $\log 2 = 0.693147$ — maximum uncertainty, the coin flip.

Verify it:

```bash
py -c "import torch,torch.nn.functional as F; \
m=torch.tensor([0.,1.,2.,4.]); \
print([round(-F.logsigmoid(x).item(),6) for x in m])"
# [0.693147, 0.313262, 0.126928, 0.01815]
```

<div class="callout warn"><p>Compute the loss as <code>-F.logsigmoid(margin)</code>, <strong>not</strong> <code>-torch.log(torch.sigmoid(margin))</code>. For a large negative margin, <code>sigmoid</code> underflows to <code>0.0</code> and <code>log(0)</code> is <span>$-\infty$</span>; <code>logsigmoid</code> is a single numerically stable kernel (it internally uses the <code>log(1+exp())</code> = softplus identity) and never overflows. Same reasoning as the stable log-softmax in <a href="../module-01/lesson-04.md">01.4</a>.</p></div>

## What *is* a reward model, mechanically?

### 1. Intuition

A reward model is just a language model (Module 6) with its head swapped. The GPT of [06.1](../module-06/lesson-01.md) ends in an `lm_head` of shape $(C \to V)$ that scores every one of the $V$ vocabulary tokens at every position. A reward model throws that away and bolts on a **scalar head** of shape $(C \to 1)$: it reads one position's hidden state and outputs a single number — the reward for the whole sequence.

Which position? The reward summarizes the *entire* prompt+completion, so we read the reward off the hidden state of the **last token**. Because attention is causal (a token attends only to itself and earlier tokens — [05.x](../module-05/lesson-01.md)), the final token's hidden state is the only one that has "seen" everything, so it is the natural place to pool the sequence into one score. This is what InstructGPT and HH-RLHF do.

### 2. Mathematics

Let the backbone map token ids $x_{1:T}$ to hidden states $h_1, \dots, h_T \in \mathbb{R}^C$. The scalar head is a linear map $w \in \mathbb{R}^{C}$, $b \in \mathbb{R}$:

$$
r(x) = w^\top h_T + b
$$

That is the entire reward model on top of the backbone: one dot product per sequence.

### 3. Tensor shapes / dtype / device

- Token ids `idx`: `(B, T)`, `int64`.
- Backbone hidden states `hidden`: `(B, T, C)`, float, on the model's device.
- Scalar head `nn.Linear(C, 1)` applied to `hidden`: `(B, T, 1)`, squeezed to `(B, T)`.
- Reward at the last position: `(B,)`, one float per sequence.

For a batch that mixes sequence lengths (right-padded), you don't want the reward read at a pad token, so you index each row at `seq_lengths - 1` instead of a blanket `-1`.

### 5. From-scratch PyTorch

The module `llmre.preference.reward_model` implements this. `RewardModel` is backbone-agnostic: give it a backbone that returns `(B, T, C)` hidden states, or hand it hidden states directly.

```python
import torch
from llmre.preference.reward_model import RewardModel, bradley_terry_loss

C = 8
rm = RewardModel(n_embd=C, backbone=None)     # scalar head only

# Pretend these came from a GPT backbone: (B=2, T=4, C=8) hidden states.
hidden = torch.randn(2, 4, C)
rewards = rm.reward_from_hidden(hidden)        # (2,) — read at the last token
```

The reward-reading logic is small and worth seeing in full — the head plus the last-token gather:

```python
def reward_from_hidden(self, hidden, seq_lengths=None):
    scores = self.score(hidden).squeeze(-1)    # (B, T, 1) -> (B, T)
    if seq_lengths is None:
        return scores[:, -1]                   # (B,) last position
    idx = (seq_lengths - 1).clamp(min=0)       # (B,) last real (non-pad) index
    rows = torch.arange(scores.shape[0], device=scores.device)
    return scores[rows, idx]                   # (B,)
```

And the Bradley-Terry loss is a one-liner over the two reward vectors:

```python
def bradley_terry_loss(reward_chosen, reward_rejected, reduction="mean"):
    margin = reward_chosen - reward_rejected   # (B,)
    per_pair = -F.logsigmoid(margin)           # (B,) = -log sigmoid(margin)
    return per_pair.mean()                     # (with reduction="mean")
```

A full training step: forward the reward model on the chosen and rejected completions, compute the loss, backprop, step.

```python
opt = torch.optim.SGD(rm.parameters(), lr=0.5)
hc, hr = torch.randn(3, 4, C), torch.randn(3, 4, C)   # chosen / rejected hidden states
for _ in range(20):
    opt.zero_grad()
    loss = bradley_terry_loss(rm.reward_from_hidden(hc), rm.reward_from_hidden(hr))
    loss.backward()
    opt.step()
# the reward margin r_w - r_l grows; the loss falls (see test_preference.py)
```

### 6. Under the hood / cost

- **Initialization.** In practice you don't train a reward model from scratch — you start from the **SFT model** (Module 14): it already understands language and the chat format, so you only need to teach the scalar head (and lightly adapt the backbone) to rank. This reuse is why the reward model and policy are usually the same size.
- **Compute.** One reward evaluation is one full forward pass through the backbone — the same FLOPs as an SFT forward. A preference pair needs *two* forwards (chosen + rejected). No generation is involved in reward-model training, so it is far cheaper than the PPO loop of lesson 15.2.
- **The scalar head is tiny.** For GPT-2 small, $C = 768$, so the head is $768 + 1$ parameters — negligible next to the backbone's 124M.

<div class="hw">
<p><strong>Hardware track.</strong> Training a real reward model from an SFT checkpoint: <strong>min</strong> 1&times; 24GB GPU (e.g. RTX 3090/4090) for a ~1B backbone with LoRA; <strong>recommended</strong> 1&times; A100 40GB for a 7B backbone. <strong>Runtime</strong> a few hours over ~100k preference pairs; <strong>GPU-hours</strong> single-digit for small models. <strong>CPU-only</strong>: the from-scratch <code>RewardModel</code> unit tests run in milliseconds on CPU; training a useful reward model on CPU is not practical.</p>
</div>

## Common mistakes

- **Reading the reward at a pad token.** With right-padded batches, `scores[:, -1]` may land on padding. Pass `seq_lengths` so the reward is read at the last *real* token.
- **Expecting absolute rewards to mean something.** Only margins are trained; a reward of `+4.2` is meaningless on its own. Never threshold on absolute reward across different prompts.
- **Using `log(sigmoid(x))`.** Numerically unstable; use `F.logsigmoid`.
- **Training the reward model and policy to convergence independently and forgetting they drift apart.** As the policy improves in RLHF it produces completions the reward model never saw — *distribution shift* (lesson 15.4). Reward models are periodically refreshed.

## Exercise

Implement a batched Bradley-Terry loss that also returns the **preference accuracy**: the fraction of pairs where the model already ranks chosen above rejected ($r_w > r_l$). This is the standard metric reported alongside reward-model loss.

*Hint:* accuracy does not need the sigmoid at all — it only needs the sign of the margin.

*Stronger hint:* `(reward_chosen > reward_rejected).float().mean()`.

<details><summary>Solution</summary>

```python
import torch.nn.functional as F

def bt_loss_and_acc(reward_chosen, reward_rejected):
    margin = reward_chosen - reward_rejected
    loss = -F.logsigmoid(margin).mean()
    acc = (margin > 0).float().mean()          # fraction already ranked correctly
    return loss, acc
```

At the start of training accuracy is ~0.5 (random); a well-trained reward model reaches ~0.65–0.75 on held-out human preferences — humans themselves only agree ~70% of the time, so that is close to the ceiling.

</details>

## Debugging exercise

A colleague's reward-model loss is stuck exactly at `0.6931` and never moves, even though the gradients are non-zero. What single bug produces *exactly* that number, and where is it?

<details><summary>Answer</summary>

`0.6931 = log 2 = -log &sigma;(0)`. The loss is pinned at the margin-zero value, which means `r_w - r_l == 0` for every pair. The most likely bug: the chosen and rejected completions are being fed to the reward model **identically** — e.g. the same tensor passed twice, or the chosen/rejected columns swapped into the same variable, or the last-token index landing on shared padding so both rewards read the same pad position. The reward model literally cannot see a difference, so every margin is 0. Check that `reward_chosen` and `reward_rejected` come from genuinely different inputs.

</details>

## Research connection

<div class="callout paper"><p><strong>Read:</strong> <a href="../../papers/index.md">InstructGPT (Ouyang et al. 2022)</a> — the reward-model section defines exactly the Bradley-Terry loss above and the "reward = value at the final token" convention; and <a href="../../papers/index.md">Anthropic HH-RLHF (Bai et al. 2022)</a> for how helpful/harmless preference data is collected at scale. Read the reward-modeling subsections closely; skim the RL details (that is lesson 15.2). Both are core Module 15 readings.</p></div>

## Check yourself

<details><summary>Why is preference data collected as pairwise comparisons instead of absolute scores?</summary>

Humans are unreliable at assigning absolute numeric scores (calibration drifts between labelers and over time) but reliable at comparisons ("A is better than B"). The Bradley-Terry model only needs comparisons: it converts a *pair* of rewards into a probability via the sigmoid of their difference, so pairwise labels are exactly the training signal it consumes.

</details>

<details><summary>The reward function is "defined only up to an additive constant." What does that mean and why?</summary>

Adding the same constant $c$ to every reward, $r \to r + c$, does not change any margin $r_w - r_l$, and the Bradley-Terry probability depends only on the margin. So $r$ and $r + c$ give identical preferences and identical loss — the model can never pin down the absolute level, only differences. Absolute reward values are therefore not comparable across prompts.

</details>

<details><summary>Where in the sequence is the scalar reward read, and why there?</summary>

At the last (real, non-pad) token's hidden state. Because attention is causal, only the final token has attended over the entire prompt+completion, so its hidden state summarizes the whole sequence — the right place to pool into a single scalar. The scalar head is an `nn.Linear(C, 1)` applied to that hidden state.

</details>

<details><summary>Compute the Bradley-Terry loss for a pair with $r_w = 1.5$, $r_l = 0.5$.</summary>

Margin $= 1.0$, so it is identical to the $r_w=2.0, r_l=1.0$ case: $\sigma(1.0) = 0.731059$, loss $= -\log(0.731059) = 0.313262$. Only the margin matters, not the absolute rewards.

</details>

## Next

We now have a function that scores completions and a way to train it from preferences. Next we treat the language model as a **policy** and use the reward model's scores to actually improve it — REINFORCE, advantages, and the PPO clipped objective that makes RLHF stable.

Continue to [15.2 · Policy gradient, advantage, PPO](lesson-02.md).
