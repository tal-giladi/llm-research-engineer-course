---
id: "16.1"
module: 16
minutes: 12
practice_minutes: 10
prerequisites: ["15.1", "15.2"]
objectives:
  - Trace sampled answers through a verifier and compute their group-relative advantages.
  - Identify which policy probabilities to cache and which to recompute during training.
  - Inspect a clipped GRPO update and diagnose a missing gradient.
volatility: concept
sources:
  - title: "DeepSeekMath, section 4.1 (v2)"
    url: https://arxiv.org/html/2402.03300v2
  - title: "PyTorch 2.15 torch.std"
    url: https://docs.pytorch.org/docs/2.15/generated/torch.std.html
last_verified: "2026-11-27"
---

# 16.1 · From PPO to GRPO

Imagine training a model to answer “3 + 4”. You choose what earns a reward. The model tries several answers, a checker scores each one, and you use those scores to make successful answers more likely. Group Relative Policy Optimization (GRPO) tells you how to turn those attempts into an update. Follow one group through the loop and you can account for every number the optimizer uses.

## Who decides what earns a reward?

As the engineer setting up the run, you choose the task and its **reward function**. For arithmetic, suppose your rule is “award one point if the final answer equals the correct result, zero otherwise.” You can implement that rule as a **verifier**. The course verifier computes the correct result from the problem and compares it with the submitted answer. A human decides the rule once. The verifier applies it automatically to every sample.

The **policy** is the model’s probability distribution over its possible choices. A language model chooses one token at a time given the prompt and the tokens so far. To make the bookkeeping easy, our arithmetic toy chooses a whole numeric answer in one draw. Suppose it samples four times for the same prompt, “3 + 4”.

| Draw | Sampled answer | Verifier’s comparison | Reward |
| --- | --- | --- | --- |
| 1 | 7 | 7 equals 7 | 1 |
| 2 | 9 | 9 differs from 7 | 0 |
| 3 | 7 | 7 equals 7 | 1 |
| 4 | 6 | 6 differs from 7 | 0 |

Even if all four draws had been 9, each would earn zero. Agreement among samples has no say in arithmetic. The **group** matters in the next step, when we ask how well each answer did *relative to the other attempts*. Also notice what the rule measures. A lucky guess of 7 earns the same point as a carefully worked answer. A final-answer verifier alone cannot certify the reasoning that produced it.

For a task such as writing a helpful explanation, the team might collect human rankings and train a **reward model** to predict those preferences. That scorer can supply one number per completion through the same interface. GRPO needs scores from a source you specify. The choice of scorer determines what behavior earns them. [15.1 · Training a reward model](../module-15/lesson-01.md)

## From a reward to an advantage

A reward measures the outcome. An **advantage** measures how much better or worse that outcome was than expected. In the simple PPO example, the **critic** estimates the return from the state before an action. If the observed return is 1 and the critic predicted 0.6, the advantage estimate is $1 - 0.6 = +0.4$. Training a separate transformer critic takes memory for its weights, gradients and optimizer state, as well as compute to fit its predictions. [15.2 · Policy gradient and PPO](../module-15/lesson-02.md)

GRPO already has several attempts at *this very prompt*. It uses their mean reward as the baseline and their spread to set the scale. Let $G$ be the group size, $i$ the draw index, and $r_i$ its reward. The group mean $\mu$, population standard deviation $\sigma$, and advantage $\hat A_i$ are

$$
\mu=\frac{1}{G}\sum_{i=1}^{G}r_i,\qquad
\sigma=\sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i-\mu)^2},\qquad
\hat A_i=\frac{r_i-\mu}{\sigma+\varepsilon}.
$$

For our four draws the mean is $2/4=0.5$. Each reward is half a point from that mean, so the mean squared deviation is $0.25$ and its square root is $0.5$. The first 7 earns an advantage of approximately $(1-0.5)/0.5=+1$. The 9 gets $(0-0.5)/0.5=-1$. In draw order the advantages are **[+1, −1, +1, −1]**. Positive advantage favors increasing the sampled answer’s probability. Negative advantage favors decreasing it.

Two edge cases help when inspecting training logs. With partial-credit rewards of [2, 0, 1, 1], the mean is 1 and the standard deviation is $\sqrt{0.5}$, giving advantages of roughly [1.414, −1.414, 0, 0]. With four identical rewards, every deviation is zero. The small constant $\varepsilon$ (here $10^{-8}$) makes the division well-defined, so every advantage is zero. That group supplies no reward gradient. A separately enabled regularizer can still supply a gradient.

## One trainable policy, seen at two times

A **rollout** is the sampling phase that produced our four answers. “Old” means *at rollout time for this group*. “Current” means *at the moment we calculate the training loss*. The two labels let you ask a concrete question about the first 7. How likely was this exact answer when we drew it, and how likely is it under the weights we have now?

| When | What the trainer does | What it keeps |
| --- | --- | --- |
| Collect the group | Sample answers and record the log-probability assigned to each sampled choice. | Prompt, sampled choices and fixed rollout log-probabilities. |
| Prepare the update | Score the answers and compute the group advantages. | One fixed advantage per answer. |
| Take an optimizer step | Run the current policy on those same choices. Compare its fresh log-probabilities with the saved ones, compute the loss and backpropagate. | Updated policy weights. |
| Reuse or refresh | For another step on this group, recompute only the current policy values. For a fresh group, sample again with the updated policy. | Each group keeps its own rollout values. |

**How many models do you need?** In the course toy, one trainable policy and an arithmetic function are enough. Saved numbers represent the old policy for these particular choices. A separate rollout worker can hold a snapshot of the policy to generate text at scale. Either implementation leaves just one policy for the optimizer to train.

There is one more name in the original GRPO objective. The **reference policy** is a frozen checkpoint, often the supervised fine-tuned starting model. It anchors a KL divergence penalty that discourages drift. The rollout policy refreshes with each new group of samples. The reference can stay fixed across many groups. The course toy omits the KL term, so it needs no reference forward pass.

## What the optimizer computes

Write $q$ for the prompt, $o_i$ for a sampled answer, $\pi_{\rm old}$ for its sampling policy and $\pi_\theta$ for the policy with trainable weights $\theta$. The **probability ratio** compares the very same choice under both versions. Log-probabilities let us compute it by subtraction followed by exponentiation.

$$
\rho_i=\frac{\pi_\theta(o_i\mid q)}{\pi_{\rm old}(o_i\mid q)}
=\exp\big(\log\pi_\theta(o_i\mid q)-\log\pi_{\rm old}(o_i\mid q)\big).
$$

Suppose the rollout probability of 7 was 0.20. Before any update, the current probability is also 0.20, giving a ratio of 1. If a later pass on the same group assigns 7 probability 0.30, its ratio is $0.30/0.20=1.5$. The original reward for 7 is still 1.

GRPO uses PPO’s **clipped surrogate objective**. For each choice we maximize the smaller of two advantage-weighted ratios. The clip half-width $\epsilon$ controls the interval. It serves a different purpose from the tiny division guard $\varepsilon$ above.

$$
J_i=\min\big(\rho_i\hat A_i,\ \operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)\hat A_i\big).
$$

For our 7, use $\hat A_i=+1$ and $\epsilon=0.2$. At ratio 1.5 the two terms are 1.5 and 1.2, so the objective is 1.2. Increasing that ratio further earns no additional benefit on this sample. For the 9 with advantage −1, a ratio of 1.5 gives terms −1.5 and −1.2. The minimum is −1.5, so the objective still penalizes making the unsuccessful answer more likely. Clipping changes the incentive. The actual ratios can lie outside the interval.

On the first pass, $J_i$ has the **value** $\hat A_i$ because the ratio equals 1. Its **gradient** can be nonzero because the current log-probability depends on the trainable weights and the saved denominator is constant. Even when the mean objective prints as zero, successful and unsuccessful choices can produce different gradients. Detaching both sides loses the signal. Differentiating through the same log-probability on both sides makes their derivatives cancel.

With reference regularization, subtract $\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\rm ref})$ from the policy objective. $D_{\mathrm{KL}}$ measures distributional drift and $\beta$ sets its cost. A mean clipped objective of 0.12, KL of 0.04 and weight of 0.1 give $0.12-0.1(0.04)=0.116$. Negate the resulting objective to obtain a loss an optimizer can minimize.

### Moving from one answer to many tokens

The arithmetic toy has one action per answer. In the original outcome-supervised LLM formulation, compute the ratio **per generated token**, conditioning both policy versions on the same prompt and preceding tokens. Give each generated token its completion’s advantage. For a padded batch with $B$ prompts, $G$ answers each and at most $T$ generated tokens, the log-probabilities have shape $(B,G,T)$. Broadcast the float32 advantages from $(B,G,1)$, mask prompt and padding positions, average over valid generated tokens within each answer, then average over answers. Keep the boolean mask and floating-point tensors on the training device. Clipping a product of token ratios for a whole answer produces a different update. [DeepSeekMath v2, §4.1.2](https://arxiv.org/html/2402.03300v2#S4.SS1.SSS2)

GRPO saves the critic’s training cost. Generation still costs time and memory as $G$ and answer length grow, and a learned scorer or KL reference adds forward passes when used. In practice, log both the fraction of correct samples and the fraction of groups with mixed rewards. A run where every answer is wrong and one where every answer is correct both produce zero group advantages, but they call for different decisions about prompt difficulty.

## Practice · Trace an update on CPU

Allow ten minutes with the course Python environment and PyTorch installed. This exercise uses a handful of CPU tensors, takes seconds to run and requires zero GPU-hours. Run the block in Python. It recreates our four recorded draws and performs one real optimizer step. Before running it, predict the initial ratio for 7 and the direction its probability will move.

```python
import torch
from llmre.reasoning.verifiers import arithmetic_verifier

# Answer columns are 6, 7, 9. Sampled column indices encode 7, 9, 7, 6.
logits = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
draws = torch.tensor([[1, 2, 1, 0]])  # (1, 4), int64
answers = [6, 7, 9]
rewards = torch.tensor([[
    arithmetic_verifier("3 + 4", answers[index]) for index in draws[0].tolist()
]])
mean = rewards.mean(dim=1, keepdim=True)
std = rewards.std(dim=1, correction=0, keepdim=True)
advantages = (rewards - mean) / (std + 1e-8)

def chosen_logprobs():
    return logits.log_softmax(dim=-1).gather(1, draws)

with torch.no_grad():
    old = chosen_logprobs()
current = chosen_logprobs()
ratio = (current - old).exp()
objective = torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)
optimizer = torch.optim.SGD([logits], lr=0.1)
optimizer.zero_grad()
(-objective.mean()).backward()
optimizer.step()

print(rewards.tolist(), advantages.tolist())
print(ratio.detach().tolist())
print((chosen_logprobs() - old).exp().detach().tolist())
```

Each row holds one prompt. Logits are float32 with shape (1, 3). Rewards, advantages, saved log-probabilities and ratios are float32 with shape (1, 4), all on CPU. The mean and standard deviation each have shape (1, 1). PyTorch broadcasts them across the four draws. The example starts all three answers equally likely, so the probability of 7 starts at one third here.

**Verify the update.** The first line should show rewards [1, 0, 1, 0] and advantages [+1, −1, +1, −1]. The second should show four ones. On the third, both entries for 7 should exceed one and both wrong-answer entries should fall below one. If every ratio still reads one after the step, check whether you accidentally refreshed the saved values along with the current ones.

**Try a unanimous group.** Change the draws to four copies of index 2 and rerun the entire block. Predict the final ratios before you run it.

<details><summary>Hint</summary>

All four draws now submit 9. Work out the group mean before taking the differences from it.

</details>

<details><summary>Stronger hint and reference result</summary>

The verifier returns four zeros. Each advantage is zero, so this reward-only step leaves every ratio at one. Four copies of index 1 also give zero advantages, although the raw rewards are now all ones.

</details>

Compare your reductions with the two functions in the [course GRPO implementation](../../code/src/llmre/rl/grpo.py). The [arithmetic training loop](../../code/src/llmre/reasoning/rlvr.py) samples fresh answers for every update. It can therefore use the current log-probabilities detached from autograd as its rollout values on that single step. With multiple steps per group, save the rollout values once before the first update. The population-standard-deviation API used above was checked against [PyTorch 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.std.html) on November 27, 2026.

## Next

Train the arithmetic policy over many fresh groups and track whether verifier-accepted answers become more frequent. [16.2 · RLVR and verifiable rewards](lesson-02.md)
