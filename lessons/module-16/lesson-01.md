# 16.1 · From PPO to GRPO

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="../module-15/lesson-02.md">15.2 · Policy gradient, advantage, PPO</a> (the clipped surrogate objective, the ratio, the advantage, the KL penalty), the <a href="../module-15/lesson-01.md">15.1 · reward model</a> it builds on, and softmax / log-softmax from <a href="../module-05/lesson-02.md">the attention lessons</a>. You should be comfortable with a tensor of shape $(B, T)$ and with <code>log_softmax</code>.</p>
<p><strong>You will learn:</strong> why PPO's learned value network (the critic) is the expensive, fragile part of RLHF for LLMs; how <strong>GRPO</strong> (Group Relative Policy Optimization) deletes the critic and replaces its baseline with the statistics of a <em>group</em> of sampled completions; the group-relative advantage $\hat A_i = (r_i - \mathrm{mean}(r)) / \mathrm{std}(r)$ worked by hand on a group of four; and the GRPO clipped objective with a KL-to-reference term. You will map every piece onto the code in <code>llmre/rl/grpo.py</code>.</p>
<p><strong>Why this matters for ML:</strong> SFT and preference tuning (Modules 14–15) make a model <em>helpful</em> — it answers in the right format and tone — but they teach it to <em>imitate</em>, and imitation does not teach a model to search for a correct multi-step chain of reasoning. To get reliable reasoning we reward the model for <em>getting the answer right</em> and let RL find the chains that do. GRPO is the algorithm that made that cheap enough to run at scale (DeepSeekMath, DeepSeek-R1), because it drops the second large network PPO needs.</p>
</div>

## 1. The story so far, and where it breaks

Walk back through what the previous modules bought you. Pretraining (Modules 6–7) gave a model that continues text. SFT (Module 14) taught it to follow instructions by imitating good answers. Preference tuning (Module 15) — RLHF with a reward model and PPO, or DPO — made its answers align with what humans prefer.

All of that is still, at heart, **imitation**. SFT copies demonstrations token by token. A reward model scores whole answers by how much they *look like* what humans rated highly. Neither one rewards the model for the one thing reasoning needs: producing a *chain of intermediate steps that actually reaches the correct final answer*. A model can imitate the surface form of a worked solution and still get the arithmetic wrong, because nothing in the loss ever checked the answer.

<div class="callout key"><p>Ordinary imitation cannot teach reliable multi-step reasoning. To learn to <em>reason</em>, the model needs a reward for reaching a <em>correct</em> answer, and RL to search for the chains that earn it. That is what Modules 16–17 are about.</p></div>

So we return to reinforcement learning — but the specific RL algorithm matters, because the version you learned in 15.2 (PPO) carries a passenger that is very expensive for an LLM.

## 2. What PPO needs, and why the critic hurts

Recall PPO from [15.2](../module-15/lesson-02.md). For each generated completion you compute an **advantage** $A$ — "how much better than expected was this?" — and push the policy toward high-advantage actions with the clipped surrogate objective. "Better than expected" needs a definition of *expected*: a **baseline**. PPO learns that baseline with a separate network, the **value function** or **critic** $V(s)$, and defines the advantage as roughly

$$
A = R - V(s),
$$

the realized return minus the critic's prediction of it.

For a small RL problem the critic is cheap. For an LLM it is brutal:

- **Memory and compute.** The critic is a second network roughly the *size of the policy* — often the same transformer with a scalar head. You now hold two large models in memory and run two forward/backward passes. On the hardware where you train a 7B policy, the critic can nearly double the footprint.
- **It is hard to train.** The critic is itself learned from noisy returns. Early in training its estimates are bad, and a bad baseline injects **bias** into every advantage, which destabilizes the policy update. Much of the folklore of "getting PPO to work" is really "getting the value function to behave."

For reasoning RL we sample *many* completions per prompt anyway (we want to see which chains land on the right answer). That opens a shortcut: if you already have a whole group of scored completions for the same prompt, you can read the baseline straight off the group — no learned critic at all.

## 3. GRPO: the group is the baseline

**GRPO** (Group Relative Policy Optimization, Shao et al. 2024, *DeepSeekMath*) makes exactly that move. For each prompt:

1. Sample a **group** of $G$ completions from the current policy.
2. Score each one, giving rewards $r_1, \dots, r_G$.
3. Use the group's own reward statistics as the baseline. The advantage of completion $i$ is how far its reward sits from the group mean, measured in units of the group's spread:

$$
\hat A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}.
$$

That is a baseline computed *from the samples themselves*. A completion that beat its group's average gets a positive advantage (push its tokens up); one that did worse gets a negative advantage (push them down). The group mean plays exactly the role PPO's $V(s)$ played — but it costs nothing to learn, because it is just an average of numbers you already have.

<div class="callout key"><p>GRPO's one idea: replace PPO's learned value baseline with the <strong>mean reward of a group of completions sampled for the same prompt</strong>, and standardize by the group's std. No critic network — half the memory, and no fragile second model to train.</p></div>

### 3.1 Mathematics: every symbol named

For one prompt, let the group of completions have scalar rewards $r_1, \dots, r_G$. Define

$$
\mu = \frac{1}{G}\sum_{i=1}^{G} r_i, \qquad
\sigma = \sqrt{\frac{1}{G}\sum_{i=1}^{G} (r_i - \mu)^2}, \qquad
\hat A_i = \frac{r_i - \mu}{\sigma + \varepsilon}.
$$

- $r_i$ — the scalar reward of completion $i$ (for us, 1 if the answer is correct, 0 if not; that is lesson [16.2](lesson-02.md)).
- $\mu$ — the group **mean**, the baseline. This is the fraction of the group that was correct.
- $\sigma$ — the group **population** standard deviation (divide by $G$, not $G-1$). It rescales the advantage so groups of different difficulty contribute comparably.
- $\varepsilon$ — a tiny constant (default `1e-8`) guarding the division when $\sigma = 0$.
- $\hat A_i$ — the **group-relative advantage**, the number that multiplies completion $i$'s log-prob in the policy-gradient step.

Two properties fall straight out of the definition and matter for debugging: within each group the advantages have **mean $\approx 0$** and **population std $\approx 1$**. Mean zero because we subtracted $\mu$; unit std because we divided by $\sigma$.

The **unanimous-group** case is worth internalizing. If every completion in the group got the same reward — all correct, or all wrong — then $\sigma = 0$ and there is *no signal*: nothing in the group was better or worse than anything else. The $\varepsilon$ guard turns the $0/0$ into $0/\varepsilon = 0$, so that group contributes zero advantage and zero gradient. That is the correct behavior: a prompt the policy already always gets right (or always gets wrong) teaches this step nothing.

### 3.2 Numerical example, worked by hand — a group of four

Take one prompt whose group of $G = 4$ completions earned rewards

$$
r = [\,1,\ 0,\ 1,\ 0\,].
$$

Mean: $\mu = (1 + 0 + 1 + 0)/4 = 0.5$. Two of four were correct.

Population variance: each deviation is $\pm 0.5$, so $(r_i - \mu)^2 = 0.25$ for all four. Variance $= (0.25 \cdot 4)/4 = 0.25$, and $\sigma = \sqrt{0.25} = 0.5$.

Advantages (taking $\varepsilon$ as negligible):

$$
\hat A = \frac{[\,1,0,1,0\,] - 0.5}{0.5} = \frac{[\,0.5,\,-0.5,\,0.5,\,-0.5\,]}{0.5} = [\,+1,\ -1,\ +1,\ -1\,].
$$

The two correct completions get $+1$, the two wrong ones get $-1$. Check the invariants: mean $= (1 - 1 + 1 - 1)/4 = 0$; population std $= 1$. The gradient step will push up the tokens of the two winners and push down the tokens of the two losers — by equal and opposite amounts, because the group was evenly split.

Now a group with a real spread, $r = [\,2,\ 0,\ 1,\ 1\,]$ (imagine a shaped reward, not just 0/1). Mean $\mu = 1$. Deviations $[\,1,-1,0,0\,]$, squares $[\,1,1,0,0\,]$, variance $= 2/4 = 0.5$, so $\sigma = \sqrt{0.5} \approx 0.7071$. Advantages:

$$
\hat A = \frac{[\,1,-1,0,0\,]}{0.7071} = [\,+1.4142,\ -1.4142,\ 0,\ 0\,].
$$

The two average completions ($r = 1 = \mu$) get advantage $0$ — dead center, no push. The standout ($r = 2$) and the failure ($r = 0$) get symmetric $\pm 1.4142$.

### 3.3 This is exactly what the code does

The whole computation is `grpo_advantages` in `code/src/llmre/rl/grpo.py`. Rows are prompts, columns are the group:

```python
def grpo_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # rewards: (num_prompts, group_size) float
    mean = rewards.mean(dim=1, keepdim=True)                # (num_prompts, 1)
    std  = rewards.std(dim=1, unbiased=False, keepdim=True) # (num_prompts, 1)  population std
    return (rewards - mean) / (std + eps)                   # (num_prompts, group_size)
```

- **Shape:** in `(num_prompts, group_size)`, out identical. Row $p$ is prompt $p$'s group; each column is one completion's reward. `dtype`/`device` are preserved (float, wherever the rewards live — CPU in our toy).
- `dim=1` means "reduce along the group axis," so `mean`/`std` are per-prompt columns of shape `(num_prompts, 1)`, which then **broadcast** back across the group when we subtract and divide.
- `unbiased=False` is the population std (divide by $G$) — the $\sigma$ in the math. Using the sample std ($G-1$) would make the unit-std property only approximate and would blow up for $G$ small.

Run the by-hand example straight through it:

```python
import torch
from llmre.rl.grpo import grpo_advantages

r = torch.tensor([[1., 0., 1., 0.],
                  [2., 0., 1., 1.]])
print(grpo_advantages(r))
# tensor([[ 1.0000, -1.0000,  1.0000, -1.0000],
#         [ 1.4142, -1.4142,  0.0000,  0.0000]])
print(grpo_advantages(r).mean(dim=1))   # tensor([0., 0.])  -> baseline removed
```

The printed numbers match section 3.2 to four decimals. (The $+1$ entries are actually `0.99999998` because of the $\varepsilon$ in the denominator — negligible, and exactly why the unit test compares with a tolerance.)

## 4. The GRPO objective: PPO's clip, fed the group advantage

The advantage is only the baseline half of the story. The *update* uses PPO's clipped surrogate — unchanged in form from [15.2](../module-15/lesson-02.md). Only the advantage fed into it is different (group-relative, not $R - V$).

Let $\pi_\theta$ be the current policy and $\pi_{\text{old}}$ the policy that generated the group. For each completion define the **probability ratio**

$$
\rho_i = \frac{\pi_\theta(o_i \mid q)}{\pi_{\text{old}}(o_i \mid q)} = \exp\big(\log \pi_\theta(o_i\mid q) - \log \pi_{\text{old}}(o_i \mid q)\big),
$$

computed in log-space for stability (this is why the code takes log-probs, not probs). The clipped objective to **maximize** is

$$
J_i = \min\Big(\rho_i\,\hat A_i,\ \operatorname{clip}(\rho_i,\,1-\epsilon,\,1+\epsilon)\,\hat A_i\Big),
$$

and full GRPO subtracts a **KL-to-reference** term to keep the policy from drifting far from the SFT model $\pi_{\text{ref}}$:

$$
J_i^{\text{GRPO}} = \min\big(\rho_i \hat A_i,\ \operatorname{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\,\hat A_i\big) - \beta\, D_{\mathrm{KL}}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big).
$$

- $\epsilon$ — clip half-width, typically $0.2$. The `min` caps how far one update can move the ratio outside $[1-\epsilon, 1+\epsilon]$, so a group can be reused for several gradient steps without the policy running away.
- $\beta$ — KL penalty weight (the `kl_penalty` from [15.2](../module-15/lesson-02.md), reused). $\pi_{\text{ref}}$ is the frozen starting model.
- On the **first on-policy step**, $\pi_\theta = \pi_{\text{old}}$, so $\rho_i = 1$ and $J_i$ collapses to $\hat A_i \cdot \log \pi_\theta(o_i \mid q)$ — the plain, group-baselined policy gradient. The clip only starts to matter once you take multiple steps on the same group and the ratio drifts from 1.

### 4.1 The objective in code

`grpo_objective` in `code/src/llmre/rl/grpo.py` is the clip, element-wise (the KL term is added by the caller, exactly as in PPO):

```python
def grpo_objective(logprobs, old_logprobs, advantages, clip_eps=0.2, reduction="mean"):
    ratio     = torch.exp(logprobs - old_logprobs)                         # rho
    unclipped = ratio * advantages
    clipped   = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    obj       = torch.minimum(unclipped, clipped)
    ...        # reduction: "mean" | "sum" | "none"
```

This returns the **objective** (a thing to maximize). Optimizers minimize, so the training loss is `-grpo_objective(...)` — you will see exactly that line in the RLVR loop next lesson.

<div class="callout warn"><p><strong>Sign trap.</strong> <code>grpo_objective</code> returns a quantity you want to go <em>up</em>. If you feed it straight to <code>loss.backward()</code> you will do gradient <em>ascent on the negative</em> of what you want — the policy gets worse. Always <code>loss = -grpo_objective(...)</code>.</p></div>

### 4.2 A clip worked by hand

Suppose after one step a completion's ratio has risen to $\rho = 1.5$ (the policy now assigns it $1.5\times$ its old probability) and its advantage is $\hat A = +2$ (it was good). With $\epsilon = 0.2$ the clip caps the ratio at $1.2$:

- unclipped: $1.5 \times 2 = 3.0$
- clipped: $\operatorname{clip}(1.5, 0.8, 1.2) \times 2 = 1.2 \times 2 = 2.4$
- objective: $\min(3.0,\ 2.4) = 2.4$.

The `min` chose the clipped value, so the gradient this step is as if the ratio were pinned at $1.2$: beyond $1+\epsilon$ the objective flattens and stops rewarding pushing the probability even higher. That is the whole point of the clip — it removes the incentive to make a big, destabilizing jump on a single group.

```python
import math, torch
from llmre.rl.grpo import grpo_objective
obj = grpo_objective(torch.tensor([math.log(1.5)]),   # logprob (new)
                     torch.tensor([0.0]),             # old logprob -> ratio 1.5
                     torch.tensor([2.0]),             # advantage
                     clip_eps=0.2, reduction="none")
print(obj)   # tensor([2.4000])
```

## 5. Under the hood: what you actually save

Per optimization step, plain PPO for an LLM runs a forward+backward on the **policy** *and* a forward+backward on the **critic**, and holds both sets of parameters and optimizer states. GRPO keeps only the policy (plus a frozen reference for the KL term, which needs a forward pass but no gradients and no optimizer state). The cost you add back is **sampling $G$ completions per prompt** instead of one — but generation is cheap relative to a full critic you also have to *train*, and you were going to sample a group for verifiable reasoning anyway.

The trade you are making, stated honestly:

- **Publicly documented (DeepSeekMath):** GRPO removes the value network and uses a group-relative, standardized advantage with a KL-to-reference term; this is what the paper describes and what powered its math results.
- **Reasonable industry practice:** using a group size $G$ in the range of 8–64, and reusing each group for a small number of inner update steps (where the clip earns its keep).
- **Inference / your mileage:** exact $G$, $\epsilon$, $\beta$, and whether to normalize by $\sigma$ at all are tuning choices; some later work drops or modifies the std-normalization. Treat the specific hyperparameters as things to sweep, not laws.

<div class="callout paper"><p><strong>Read:</strong> DeepSeekMath (Shao et al. 2024) introduces GRPO — see paper #25 in the <a href="../../papers/index.md">paper curriculum</a>. Inspect the figure contrasting PPO (with critic) against GRPO (group baseline), and the GRPO objective; both map directly onto <code>llmre/rl/grpo.py</code>.</p></div>

## Exercise

Implement, then check against the code: given one prompt's group rewards `r = [1, 1, 0, 0]`, compute the group-relative advantages by hand, and predict which tokens the update pushes up vs down.

<details><summary>Hint</summary>

Find $\mu$ and $\sigma$ first. With two 1s and two 0s the mean is $0.5$; the deviations are all $\pm 0.5$.

</details>

<details><summary>Stronger hint</summary>

$\sigma$ here is the same as the section-3.2 example ($0.5$), because the multiset of rewards is the same — order does not change mean or std. So the advantages are $\pm 1$; the two correct completions get $+1$, the two wrong ones $-1$.

</details>

<details><summary>Solution</summary>

$\mu = (1+1+0+0)/4 = 0.5$. Deviations $[+0.5, +0.5, -0.5, -0.5]$, each squared $0.25$, variance $= 1/4 \cdot (0.25 \cdot 4) = 0.25$, $\sigma = 0.5$. Advantages $\hat A = [+1, +1, -1, -1]$. The update pushes **up** the tokens of the two correct completions and **down** the tokens of the two wrong ones. Verify:

```python
import torch
from llmre.rl.grpo import grpo_advantages
print(grpo_advantages(torch.tensor([[1., 1., 0., 0.]])))
# tensor([[ 1.,  1., -1., -1.]])
```

Note this is the same advantage *vector* (up to order) as `[1,0,1,0]` — GRPO only cares about each completion's reward relative to its group, not the order they were sampled in.

</details>

## Common mistakes

- **Using the sample std ($G-1$) instead of the population std.** The code passes `unbiased=False` deliberately; the unit-std property and the hand numbers assume the population std.
- **Forgetting the $\varepsilon$ guard / the unanimous group.** All-correct or all-wrong groups have $\sigma = 0$; without the guard you get NaNs, and conceptually those groups should contribute *nothing*.
- **Feeding the objective straight into `backward()`.** It is an objective to maximize; the loss is its negative (section 4.1).
- **Passing probabilities where log-probs are expected.** The ratio is `exp(logp - old_logp)`; hand it raw probabilities and the exponent is wrong.

## Check yourself

<details><summary>Why does GRPO not need a value/critic network, and what replaces the baseline?</summary>

Because it samples a *group* of completions per prompt, it can use the group's own mean reward as the baseline. The advantage is $(r_i - \mathrm{mean}(r))/\mathrm{std}(r)$. That removes the second large network PPO trains, saving memory and the instability of a learned value estimate.

</details>

<details><summary>A group's rewards are all 1.0 (every completion correct). What advantage does each completion get, and why is that right?</summary>

All zeros. $\sigma = 0$, and the $\varepsilon$-guarded division returns $0$. It is correct: nothing in the group was better or worse than anything else, so there is no signal — the prompt teaches this step nothing.

</details>

<details><summary>For the group $r = [2, 0, 1, 1]$, what are the advantages, and which completions get zero push?</summary>

$\mu = 1$, $\sigma = \sqrt{0.5} \approx 0.7071$, so $\hat A = [+1.4142, -1.4142, 0, 0]$. The two completions with $r = 1 = \mu$ sit exactly at the baseline and get advantage $0$ — no push.

</details>

<details><summary>On the very first on-policy update, what does the GRPO objective reduce to?</summary>

The ratio $\rho = \exp(\log\pi_\theta - \log\pi_{\text{old}}) = 1$ because $\pi_\theta = \pi_{\text{old}}$, so $J_i = \hat A_i \cdot \log\pi_\theta(o_i\mid q)$ — the plain group-baselined policy gradient. The clip only bites once the ratio drifts from 1 over multiple inner steps.

</details>

<details><summary>The objective returned by <code>grpo_objective</code> is 2.4 for a ratio 1.5, advantage 2, clip 0.2. Why not 3.0?</summary>

The clip caps the ratio at $1.2$, giving a clipped value $1.2 \times 2 = 2.4$; the unclipped value is $1.5 \times 2 = 3.0$; the objective is $\min(3.0, 2.4) = 2.4$. For a positive advantage the clip prevents rewarding a large upward move in one step.

</details>

## Next

You now have the GRPO machinery: a group-relative advantage with no critic, and PPO's clip to apply it. But we glossed over *where the reward comes from*. For reasoning we do not want a learned reward model that can be gamed — we want an automatic checker that knows the right answer. That is **RLVR**, and it is what makes the whole toy loop in `llmre/reasoning/rlvr.py` actually learn to add.

Continue to [16.2 · RLVR & verifiable rewards](lesson-02.md).
