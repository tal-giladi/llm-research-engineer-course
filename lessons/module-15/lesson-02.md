# 15.2 · Policy gradient, advantage, PPO

<div class="prereq">
<p><strong>Prerequisites:</strong> the reward model and Bradley-Terry loss from <a href="#/lessons/module-15/lesson-01">15.1</a>; gradient descent, the optimizer, and log-probabilities from <a href="#/lessons/module-03/lesson-01">03.1 · Optimization</a>; the model's <code>generate</code> loop and next-token distribution from <a href="#/lessons/module-06/lesson-01">06.1</a>; log-softmax from <a href="#/lessons/module-01/lesson-04">01.4</a>.</p>
<p><strong>You will learn:</strong> how to view a language model as a <strong>policy</strong> <span>$\pi(\text{token} \mid \text{context})$</span>; the <strong>REINFORCE</strong> policy gradient <span>$\nabla J = \mathbb{E}[\nabla \log \pi \cdot R]$</span> and why it has high variance; how a <strong>baseline</strong> gives the <strong>advantage</strong> <span>$\hat{A}$</span>; the <strong>PPO clipped surrogate</strong> objective; and the <strong>KL penalty</strong> to a frozen reference that stops the policy from reward-hacking.</p>
<p><strong>Why this matters for ML:</strong> PPO is the "RL" in RLHF — the algorithm InstructGPT and the first ChatGPT used to turn a reward model into a better model. Every later method (DPO in 15.3, GRPO in Module 16) is understood by contrast with this one. You need the clipped objective and the KL term in your hands to see what DPO is quietly replacing.</p>
</div>

## The causal story: a reward is not a loss

Lesson 15.1 gave us $r(x, y)$, a number that says how good a completion is. We want to make the model produce completions with *higher* reward. The obvious move — "backprop the reward into the model" — does not work, and seeing *why* motivates everything in this lesson.

The reward comes from a **sampled** completion. The model produced tokens by sampling from its own distribution ([06.1](lessons/module-06/lesson-01.md)), and *then* the reward model scored the result. Sampling is not differentiable: there is no derivative of "which token got drawn" with respect to the model weights. So we cannot just call `reward.backward()`. We need **reinforcement learning**: a way to increase the probability of the choices that led to high reward, using only the reward *value*, not its gradient.

<div class="callout key"><p>SFT has a differentiable target (the gold token) so it uses ordinary backprop. RLHF has only a scalar reward on a <em>sampled</em> output — the sampling step blocks the gradient. RL's core trick, the <strong>policy gradient</strong>, recovers a usable gradient from just the reward value and the log-probability of the action taken.</p></div>

## The language model as a policy

Rename what we already have. At each step the model reads a context and outputs a probability distribution over the next token. In RL vocabulary:

- **state** $s$ — the context so far (the tokens generated up to now).
- **action** $a$ — the next token emitted.
- **policy** $\pi_\theta(a \mid s)$ — the probability the model (parameters $\theta$) assigns to that token. This is *exactly* the softmax over logits from [06.1](lessons/module-06/lesson-01.md); nothing new.
- **reward** $R$ — the reward-model score, given once at the **end** of the full generation (a "terminal" reward). Intermediate tokens get no reward of their own.
- **return** — the total reward attributed to the trajectory; with a single terminal reward, the return for every action in the sequence is that one $R$.

So "improve the model with RL" becomes "adjust $\theta$ so the policy puts more probability on token sequences that earn high reward."

## REINFORCE: the policy gradient

### 1. Intuition

We want to maximize the expected reward $J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[R(y)]$. We cannot differentiate through the sampling of $y$. The **policy gradient theorem** gives an identity that moves the derivative off the sample and onto the log-probability, which *is* differentiable:

> nudge each action's log-probability up in proportion to the reward that followed it.

Good outcome → push its tokens up. Bad outcome → push them down. Do it in proportion to how good/bad.

### 2. Mathematics (every symbol named)

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{y \sim \pi_\theta}\big[\, \nabla_\theta \log \pi_\theta(y) \cdot R(y) \,\big]
$$

- $\nabla_\theta J$ — gradient of expected reward w.r.t. the model parameters; we ascend it.
- $\pi_\theta(y) = \prod_t \pi_\theta(a_t \mid s_t)$ — probability of the whole sequence; its log is $\sum_t \log \pi_\theta(a_t \mid s_t)$, a sum of per-token log-probs we can already compute.
- $R(y)$ — the scalar reward for the sampled sequence. **Treated as a constant** — we do *not* differentiate through it. It only scales the gradient.

The one-line derivation of the identity uses $\nabla \pi = \pi \nabla \log \pi$ (the "log-derivative trick"):

$$
\nabla_\theta \mathbb{E}[R] = \nabla_\theta \!\int \pi_\theta(y) R(y)\, dy = \int R(y)\, \nabla_\theta \pi_\theta(y)\, dy = \int \pi_\theta(y) R(y)\, \nabla_\theta \log \pi_\theta(y)\, dy = \mathbb{E}\big[R \,\nabla \log \pi_\theta\big].
$$

In code this is beautifully simple: form the surrogate loss $-\,\overline{R \cdot \log \pi_\theta(y)}$ (negated because optimizers minimize) and call `.backward()`. Autograd differentiates the $\log \pi$ factor; $R$ rides along as a constant multiplier.

### 3. Why it has high variance

$R(y)$ can be large and is different for every sample. The gradient's magnitude swings wildly from sample to sample, so the estimate from a small batch is extremely noisy. Worse, if all rewards are positive, REINFORCE pushes *every* sampled action up (just some more than others) — it never directly says "that one was below average." Training is slow and unstable. The fix is the baseline.

## Baseline and advantage

### 1. Intuition

Subtract a **baseline** $b(s)$ — a prediction of the reward you *expected* in this state — from the reward before scaling the gradient. Then an action is pushed up only if it did **better than expected**, and pushed down if worse. Actions that merely met expectations get near-zero gradient. This does not change the gradient in expectation (the baseline term integrates to zero) but drastically cuts its variance.

### 2. Mathematics

$$
\nabla_\theta J = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a\mid s)\,\big(R - b(s)\big)\big], \qquad \hat{A} = R - b(s)
$$

- $b(s)$ — the baseline, usually a learned **value function** $V(s)$ that estimates the average return from state $s$. In LLM PPO this is a second scalar head (a "value head") on the model.
- $\hat{A} = R - V(s)$ — the **advantage**: how much better this action's return was than the baseline predicted. Positive → above expectation, push up; negative → below, push down.

The full PPO paper estimates the advantage over a whole trajectory with **Generalized Advantage Estimation (GAE)**; with a single terminal reward per sequence, $\hat{A} = R - V$ is the simple, exact case we implement.

### 3. Numerical example (verified)

Three sampled completions with rewards `[1.0, 0.0, 2.0]`, and a value head predicting `0.5` for all three:

$$
\hat{A} = R - V = [1.0, 0.0, 2.0] - [0.5, 0.5, 0.5] = [0.5, -0.5, 1.5]
$$

The middle completion (reward 0.0, below the expected 0.5) gets a **negative** advantage — its tokens are pushed *down* even though its reward was not negative. That is the baseline doing its job. Verify:

```bash
py -c "import torch; r=torch.tensor([1.,0.,2.]); v=torch.tensor([.5,.5,.5]); print((r-v).tolist())"
# [0.5, -0.5, 1.5]
```

Our helper (`llmre.rl.ppo.compute_advantages`) also offers whitening (`normalize=True`) — standardizing advantages to zero mean, unit variance across the batch, a standard PPO stabilizer.

## PPO: the clipped surrogate

### 1. Intuition

REINFORCE uses each sampled batch for **one** gradient step, then must resample (expensive — sampling means generating). We'd like to take **several** optimizer steps on the same batch. But after a step, $\pi_\theta$ has moved away from the $\pi_{\text{old}}$ that generated the data, so the data is off-policy and a big step can wreck the policy.

PPO's fix: importance-weight by the probability **ratio** $r_t = \pi_\theta / \pi_{\text{old}}$, and **clip** it so a single update can't move the policy too far. The clipping removes any incentive to push the ratio outside a trust band $[1-\epsilon, 1+\epsilon]$.

### 2. Mathematics (every term named)

$$
L^{\text{CLIP}} = \mathbb{E}_t\Big[\min\big(r_t\,\hat{A}_t,\; \operatorname{clip}(r_t,\,1-\epsilon,\,1+\epsilon)\,\hat{A}_t\big)\Big], \qquad r_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}
$$

- $r_t$ — probability ratio of the new policy to the data-generating policy for the taken action. $r_t = 1$ means unchanged. Computed as $\exp(\log \pi_\theta - \log \pi_{\text{old}})$ for numerical stability.
- $\hat{A}_t$ — the advantage from above.
- $\operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)$ — clamps the ratio into $[1-\epsilon, 1+\epsilon]$; $\epsilon$ is typically $0.2$.
- $\min(\cdot, \cdot)$ — takes the **pessimistic** of the clipped and unclipped terms. This is the crux:
  - When $\hat{A} > 0$ (good action): the objective grows with $r_t$, but the clipped branch caps it at $(1+\epsilon)\hat{A}$. Once $r_t > 1+\epsilon$ there is no further gain — no reward for shoving the probability arbitrarily high on one update.
  - When $\hat{A} < 0$ (bad action): the objective is capped once $r_t < 1-\epsilon$. The policy can't be driven to make a bad action's probability collapse in one step.

$L^{\text{CLIP}}$ is an **objective to maximize**; the training loss is $-L^{\text{CLIP}}$.

### 3. Numerical example (verified, both branches)

Fix $\epsilon = 0.2$, so the ratio band is $[0.8, 1.2]$. Work four cases by hand:

| ratio $r_t$ | $\hat{A}$ | unclipped $r_t\hat{A}$ | clipped $\operatorname{clip}(r_t)\hat{A}$ | $\min$ (objective) | which branch |
|------:|-----:|-----:|-----:|-----:|:--|
| 1.5   | +2.0 | 3.0  | $1.2\times 2 = 2.4$  | **2.4** | clipped (good action, capped) |
| 0.7   | +2.0 | 1.4  | $0.8\times 2 = 1.6$  | **1.4** | unclipped |
| 1.5   | −2.0 | −3.0 | $1.2\times(-2)=-2.4$ | **−3.0**| unclipped (bad action, not helped) |
| 0.7   | −2.0 | −1.4 | $0.8\times(-2)=-1.6$ | **−1.6**| clipped |

Read the first row: a *good* action (positive advantage) whose new probability jumped to $1.5\times$ the old — PPO caps its contribution at $2.4$ instead of the greedy $3.0$, so the optimizer has no incentive to push the ratio past $1.2$. Verify:

```bash
py -c "import torch; eps=0.2
for r,A in [(1.5,2.),(0.7,2.),(1.5,-2.),(0.7,-2.)]:
    rr=torch.tensor(r); a=torch.tensor(A)
    print(r,A,torch.minimum(rr*a, torch.clamp(rr,1-eps,1+eps)*a).item())"
# 1.5 2.0 2.4000...   0.7 2.0 1.4000...   1.5 -2.0 -3.0   0.7 -2.0 -1.6000...
```

### 4. Tensor shapes / dtype / device

Everything is elementwise over the token (or sequence) axis:

- `logprobs`, `old_logprobs`, `advantages`: all the same shape, e.g. `(N,)` over $N$ tokens, float, on the model's device. `logprobs` carries grad; `old_logprobs` and `advantages` are constants (`.detach()`ed).
- `ratio = exp(logprobs - old_logprobs)`: same shape.
- objective: reduced to a scalar with `.mean()`.

### 5. From-scratch PyTorch

`llmre.rl.ppo` implements the three pieces on tiny tensors. The clipped objective in full:

```python
def ppo_clip_objective(logprobs, old_logprobs, advantages, clip_eps=0.2):
    ratio = torch.exp(logprobs - old_logprobs)                    # pi_theta / pi_old
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return torch.minimum(unclipped, clipped).mean()               # objective (maximize)
```

```python
import torch, math
from llmre.rl.ppo import ppo_clip_objective, compute_advantages, kl_penalty

lp  = torch.tensor([math.log(1.5), math.log(0.7)])   # so exp(lp - old) = 1.5, 0.7
old = torch.zeros(2)
adv = torch.tensor([2.0, 2.0])
print(ppo_clip_objective(lp, old, adv).item())       # 1.9  = mean(2.4, 1.4)
```

## The KL penalty: don't reward-hack

### 1. Intuition

The reward model is an *imperfect* proxy for human preference. If we let PPO optimize it without limit, the policy will find and exploit its blind spots — producing text that scores high on the reward model but is actually bad (**reward hacking / over-optimization**). The guardrail is a penalty for drifting from the **frozen reference policy** $\pi_{\text{ref}}$ (the SFT model we started from). The policy may improve, but it must stay *recognizably close* to a model we trust.

### 2. Mathematics

The per-token reward fed to PPO becomes the reward-model score minus a KL penalty:

$$
R_t = r(x, y) - \beta_{\text{KL}}\, \big(\log \pi_\theta(a_t\mid s_t) - \log \pi_{\text{ref}}(a_t\mid s_t)\big)
$$

The bracket is a single-sample estimate of $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$: the mean log-ratio over the tokens the policy actually sampled. $\beta_{\text{KL}}$ trades reward against staying near the reference. Verify the estimator on two tokens where the policy is $0.5$ nats more confident than the reference each:

```bash
py -c "import torch; lp=torch.tensor([-1.,-2.]); ref=torch.tensor([-1.5,-2.5]); print((lp-ref).mean().item())"
# 0.5
```

```python
def kl_penalty(logprobs, ref_logprobs):
    return (logprobs - ref_logprobs).mean()      # E[log pi_theta - log pi_ref]
```

<div class="callout warn"><p>This plain log-ratio (the "k1" estimator) is unbiased but can go slightly negative on a finite sample. Production RLHF (e.g. TRL) often uses the always-nonnegative "k3" estimator <span>$\exp(d) - 1 - d$</span> with <span>$d = \log\pi_{\text{ref}} - \log\pi_\theta$</span>. We use the transparent log-ratio here; both estimate the same KL.</p></div>

### 6. Under the hood / cost — why PPO-on-LLMs is heavy

Full RLHF-PPO keeps **four** models in play at once:

1. the **policy** $\pi_\theta$ being trained (backbone + a value head),
2. the **reference** $\pi_{\text{ref}}$ (frozen SFT model, for the KL term),
3. the **reward model** (frozen, scores completions),
4. the **value model** (the baseline; often the value head on the policy).

Each PPO iteration must **generate** completions from the policy (slow autoregressive sampling, [06.1](lessons/module-06/lesson-01.md)), score them with the reward model, compute log-probs under policy and reference, estimate advantages, then take several clipped-objective steps. It is memory-hungry (multiple model copies) and finicky (many interacting hyperparameters: $\epsilon$, $\beta_{\text{KL}}$, learning rate, GAE $\lambda$, rollout size). This engineering weight is exactly the pain DPO removes in lesson 15.3.

<div class="callout pt"><p>The production library is <strong>TRL</strong> (<code>PPOTrainer</code>), which wires generation, reward scoring, reference log-probs, GAE, and the clipped update into one loop, with the value head and KL controller built in. We implemented the load-bearing math — the clipped objective, the advantage, the KL estimate — from scratch first so you can read that loop and know exactly what each line does.</p></div>

<div class="hw">
<p><strong>Hardware track.</strong> RLHF-PPO on a real LLM is one of the heaviest post-training jobs. <strong>Min</strong> 1&times; A100 40GB for a ~1B policy with LoRA and a small reward model, and it is slow because of generation. <strong>Recommended</strong> multi-GPU (8&times; A100/H100) for 7B+ policies. <strong>Runtime</strong> hours to days; <strong>GPU-hours</strong> tens to hundreds. <strong>CPU-only</strong>: impossible for training; the <code>llmre.rl.ppo</code> math functions in this lesson run instantly on CPU for learning and unit tests.</p>
</div>

## Common mistakes

- **Backpropagating through the reward.** $R$ (or $\hat{A}$) is a constant multiplier; detach it. Only $\log \pi_\theta$ carries grad. If the reward tensor has `requires_grad=True`, you are computing nonsense.
- **Computing the ratio as $\pi_\theta / \pi_{\text{old}}$ directly.** Do it in log space: `exp(logp - old_logp)`. Dividing raw probabilities underflows.
- **Dropping the KL term.** Without it the policy reward-hacks the reward model within a few hundred steps — reward climbs, quality collapses.
- **Confusing objective and loss sign.** `ppo_clip_objective` returns a quantity to **maximize**; the loss you `.backward()` is its negation.

## Debugging exercise

During PPO, the reward-model score reported per step keeps climbing, but human raters say the outputs are getting *worse* — repetitive, weird, gaming obvious patterns. The KL to the reference is also climbing fast. What is happening and which knob fixes it?

<details><summary>Answer</summary>

Classic **reward hacking / over-optimization**. The policy has found inputs where the (imperfect) reward model scores high but true quality is low, and it is racing toward them — which is exactly why the KL to the reference is exploding (the policy has drifted far from the trusted SFT model). Fix: **increase $\beta_{\text{KL}}$** (or add/lower a KL target so the controller pulls the policy back), and/or stop earlier. The rising KL is the tell — a healthy PPO run keeps KL bounded. Longer term, the reward model itself may need refreshing on the new on-policy samples (distribution shift, lesson 15.4).

</details>

## Check yourself

<details><summary>Why can't we just call <code>reward.backward()</code> to improve the model?</summary>

The reward is computed on a *sampled* completion, and sampling (drawing a token from the distribution) is not differentiable — there is no gradient of "which token was drawn" w.r.t. the weights. The policy gradient recovers a usable gradient using only the reward *value* times $\nabla \log \pi_\theta(a)$, where the log-prob of the chosen action *is* differentiable.

</details>

<details><summary>What problem does the baseline solve, and what is the advantage?</summary>

REINFORCE's gradient estimate has very high variance (the raw reward scales it, and with all-positive rewards every action gets pushed up). Subtracting a baseline $b(s) \approx V(s)$ — the expected return in that state — leaves the gradient unbiased but low-variance. The advantage $\hat{A} = R - V(s)$ is the leftover: how much better than expected the action did. Positive → push up, negative → push down.

</details>

<details><summary>In the clip objective, why take the <em>min</em> of the clipped and unclipped terms?</summary>

The `min` makes the objective *pessimistic*, which removes the incentive to move the probability ratio outside $[1-\epsilon, 1+\epsilon]$. For a good action ($\hat A>0$) the gain is capped once $r_t > 1+\epsilon$; for a bad action ($\hat A<0$) the objective stops improving once $r_t < 1-\epsilon$. Either way one update can't move the policy too far, so we can safely take several steps per batch.

</details>

<details><summary>Compute the PPO objective for ratio 0.7, advantage −2.0, $\epsilon=0.2$. Which branch wins?</summary>

Unclipped $= 0.7 \times (-2.0) = -1.4$. Clipped ratio $= \max(0.7, 0.8) = 0.8$, so clipped $= 0.8 \times (-2.0) = -1.6$. The `min` is $-1.6$ — the **clipped** branch. (A bad action whose probability dropped below the band is held at the clipped, more-negative value.)

</details>

<details><summary>What is the KL penalty for and what happens without it?</summary>

It penalizes the policy for drifting from the frozen reference (SFT) policy, keeping the tuned model close to one we trust. Without it, PPO over-optimizes the imperfect reward model — reward-hacking: the score climbs while real quality falls, and KL to the reference explodes.

</details>

## Next

PPO works but is heavy: four models, a generation loop, and a pile of interacting hyperparameters. Next we derive **DPO**, which starts from the exact same KL-constrained RLHF objective and the exact same Bradley-Terry model from 15.1, and collapses the whole thing into a single supervised loss on preference pairs — no reward model, no sampling.

Continue to [15.3 · DPO: derivation & implementation](lessons/module-15/lesson-03.md).
