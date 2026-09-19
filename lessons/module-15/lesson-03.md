# 15.3 · DPO: derivation & implementation

<div class="prereq">
<p><strong>Prerequisites:</strong> the Bradley-Terry model and reward loss from <a href="lesson-01.md">15.1</a>; the KL-constrained RLHF objective, the reference policy, and sequence log-probabilities from <a href="lesson-02.md">15.2</a>; log-softmax and gathering token log-probs from <a href="../module-01/lesson-04.md">01.4</a>; the response-only masking of <a href="../module-14/lesson-02.md">14.2</a>.</p>
<p><strong>You will learn:</strong> how to derive <strong>Direct Preference Optimization</strong> from the RLHF objective — the closed-form optimal policy, inverting it to express the reward through the policy, and substituting into Bradley-Terry so the reward model and the partition function <em>disappear</em>; and how to implement <code>sequence_logprob</code> and <code>dpo_loss</code> from scratch, verified numerically.</p>
<p><strong>Why this matters for ML:</strong> DPO turned preference tuning from a fragile four-model RL loop into a stable supervised loss you can run like SFT. It is the default alignment method for most open-weight models today. And its derivation is the single most instructive piece of algebra in post-training: it shows the RLHF objective and a simple classification loss are two views of the same thing.</p>
</div>

## The causal story: keep the goal, drop the machinery

Lesson 15.2 optimized the right objective but paid a brutal price: a reward model, a value model, a frozen reference, and a generation loop, all interacting through touchy hyperparameters. DPO's insight is that the RLHF objective has a **closed-form solution**, and if you write the loss in terms of that solution, the reward model and the sampling both cancel out. What's left is a loss you compute directly from preference pairs — hence *Direct* Preference Optimization.

We derive it in four steps. Every step is algebra you can follow.

## Step 1 — the KL-constrained RLHF objective

RLHF (lesson 15.2) maximizes reward while staying close (in KL) to the reference policy $\pi_{\text{ref}}$:

$$
\max_{\pi}\; \mathbb{E}_{x,\, y\sim\pi}\big[r(x,y)\big] \;-\; \beta\, \mathrm{KL}\!\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)
$$

- $r(x, y)$ — the reward for completion $y$ on prompt $x$.
- $\beta$ — the KL weight; larger $\beta$ forces the policy to hug the reference.
- $\mathrm{KL}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{y\sim\pi}[\log(\pi(y|x)/\pi_{\text{ref}}(y|x))]$ — the penalty for drifting.

This is precisely the objective PPO chases with its reward term and KL penalty. It is not new; DPO just refuses to solve it with RL.

## Step 2 — the closed-form optimal policy

For a *fixed* reward $r$, this objective has an exact maximizer. Rewrite it (per prompt $x$) as a single KL to a target distribution:

$$
\max_\pi\; \mathbb{E}_{y\sim\pi}\!\left[r(x,y) - \beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]
= \min_\pi\; \mathbb{E}_{y\sim\pi}\!\left[\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)}\right].
$$

The thing inside is a KL divergence between $\pi$ and the (unnormalized) distribution $\pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$. A KL is minimized (to zero) exactly when the two distributions are equal, so the optimum is:

$$
\boxed{\;\pi^{*}(y\mid x) \;=\; \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\Big(\tfrac{1}{\beta}\, r(x,y)\Big)\;}
$$

- $\pi^{*}$ — the optimal policy for this reward and $\beta$.
- $Z(x) = \sum_{y}\pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$ — the **partition function** that renormalizes it to sum to 1. It depends on $x$ but **not** on $y$, and summing over all possible completions $y$ is intractable. This intractable $Z(x)$ is exactly what killed earlier attempts to use this formula directly — and exactly what DPO makes cancel.

Intuitively: the best policy is the reference *reweighted* by $\exp(\text{reward}/\beta)$ — bump up completions the reward likes, in proportion to how much, tempered by $\beta$.

## Step 3 — invert it: reward *in terms of* the policy

Solve the boxed equation for $r(x, y)$. Take logs of both sides and rearrange:

$$
\log \pi^{*}(y|x) = \log \pi_{\text{ref}}(y|x) + \tfrac{1}{\beta} r(x,y) - \log Z(x)
$$

$$
\boxed{\;r(x,y) \;=\; \beta\,\log\frac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \;+\; \beta\log Z(x)\;}
$$

This is the key move. The reward — the thing we trained a whole separate network for in 15.1 — is *expressible* as $\beta$ times the log-ratio of the optimal policy to the reference, plus a term that depends only on the prompt. **The optimal policy already encodes the reward.** We don't need a separate reward network; we just need the policy.

## Step 4 — substitute into Bradley-Terry; $Z(x)$ cancels

Recall the Bradley-Terry preference probability from [15.1](lesson-01.md):

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big).
$$

Plug the boxed reward expression into the **difference** $r(x, y_w) - r(x, y_l)$:

$$
r(x,y_w) - r(x,y_l) = \Big(\beta\log\tfrac{\pi^{*}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \beta\log Z(x)\Big) - \Big(\beta\log\tfrac{\pi^{*}(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + \beta\log Z(x)\Big)
$$

The $\beta\log Z(x)$ terms are **identical** (both use the same prompt $x$) and **subtract to zero**. The intractable partition function is gone:

$$
r(x,y_w) - r(x,y_l) = \beta\log\frac{\pi^{*}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi^{*}(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

Now replace the unknown optimal $\pi^{*}$ with our trainable policy $\pi_\theta$ (we are *fitting* $\theta$ so that $\pi_\theta$ becomes $\pi^{*}$), put this margin inside the Bradley-Terry $-\log\sigma$ loss, and average over the dataset:

$$
\boxed{\;\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)\right]\;}
$$

<div class="callout key"><p>DPO needs <strong>no reward model</strong> and <strong>no sampling</strong>. To compute the loss you only forward the policy and the frozen reference on the given chosen and rejected completions, gather their sequence log-probabilities, and plug into <span>$-\log\sigma$</span>. It is a supervised classification loss — as easy to run as SFT — that provably optimizes the same objective PPO does.</p></div>

### What each term means

- $\log \pi_\theta(y_w|x) - \log \pi_{\text{ref}}(y_w|x)$ — how much *more* likely the policy makes the chosen completion than the reference did. DPO's **implicit reward** for $y_w$ (times $\beta$).
- The full bracket is the **implicit reward margin**: chosen minus rejected implicit reward. Minimizing $-\log\sigma(\cdot)$ pushes this margin up — raise $\pi_\theta(y_w)$ relative to reference, lower $\pi_\theta(y_l)$ relative to reference.
- $\beta$ — same role as the RLHF KL weight. Small $\beta$ lets the policy move far from the reference; large $\beta$ keeps it close. The reference appears in the loss itself, so the KL constraint is *baked in* — no separate penalty term.

## Numerical example (worked by hand, verified)

Take one preference pair with these sequence log-probs (already summed over tokens):

- $\log\pi_\theta(y_w) = -2.0$, $\log\pi_\theta(y_l) = -3.0$
- $\log\pi_{\text{ref}}(y_w) = -2.5$, $\log\pi_{\text{ref}}(y_l) = -2.5$
- $\beta = 0.5$

Step through the loss:

$$
\text{pi\_logratios} = \log\pi_\theta(y_w) - \log\pi_\theta(y_l) = -2.0 - (-3.0) = 1.0
$$
$$
\text{ref\_logratios} = \log\pi_{\text{ref}}(y_w) - \log\pi_{\text{ref}}(y_l) = -2.5 - (-2.5) = 0.0
$$
$$
\text{margin} = \text{pi\_logratios} - \text{ref\_logratios} = 1.0 - 0.0 = 1.0
$$
$$
\beta \cdot \text{margin} = 0.5 \times 1.0 = 0.5, \qquad \sigma(0.5) = 0.622459
$$
$$
\mathcal{L}_{\text{DPO}} = -\log(0.622459) = 0.474077
$$

The policy already prefers chosen over rejected *more than the reference does* (pi\_logratios $1.0$ vs ref\_logratios $0.0$), so the loss is below $\log 2 = 0.693$. Verify:

```bash
py -c "import torch,torch.nn.functional as F; \
pcw,pcl,rcw,rcl,beta=-2.,-3.,-2.5,-2.5,0.5; \
m=(pcw-pcl)-(rcw-rcl); \
print(round(-F.logsigmoid(torch.tensor(beta*m)).item(),6))"
# 0.474077
```

## Computing the sequence log-probs: `sequence_logprob`

The DPO loss needs $\log\pi(y|x) = \sum_t \log \pi(y_t\mid y_{<t}, x)$ — the **sum** of per-token log-probs over the completion. This is the same gather-the-true-token-log-prob operation as cross-entropy ([01.4](../module-01/lesson-04.md)), but summed (not averaged) and over response tokens only (prompt masked to `ignore_index`, exactly as in [14.2](../module-14/lesson-02.md)).

### Tensor shapes / dtype / device

- `logits`: `(B, T, V)` float — the policy's logits over the completion.
- `labels`: `(B, T)` int64 — the completion token at each position, with prompt/pad set to `-100`.
- output: `(B,)` float — one summed log-prob per sequence.

### From-scratch PyTorch

```python
def sequence_logprob(logits, labels, ignore_index=-100):
    logp = F.log_softmax(logits, dim=-1)                       # (B, T, V)
    mask = labels != ignore_index                             # (B, T) bool
    safe = labels.clamp(min=0)                                # avoid gather on -100
    tok_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1) # (B, T) true-token log-prob
    tok_logp = tok_logp * mask                               # zero out ignored positions
    return tok_logp.sum(dim=-1)                              # (B,) sum over the sequence
```

Worked check on a $(1, 2, 3)$ example — two positions, vocab of 3, labels `[0, 2]`:

```bash
py -c "import torch,torch.nn.functional as F; \
lg=torch.tensor([[[2.,1.,0.],[0.,1.,2.]]]); lb=torch.tensor([[0,2]]); \
p=F.log_softmax(lg,-1); \
print(round(p[0,0,0].item(),6), round(p[0,1,2].item(),6), \
round((p[0,0,0]+p[0,1,2]).item(),6))"
# -0.407606 -0.407606 -0.815212
```

Both positions have the same log-softmax shape, so each true-token log-prob is $-0.407606$ and the sequence log-prob is their sum, $-0.815212$.

## Implementing `dpo_loss`

`llmre.preference.dpo` implements the boxed loss directly:

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    pi_logratios  = policy_chosen_logps - policy_rejected_logps   # (B,)
    ref_logratios = ref_chosen_logps   - ref_rejected_logps       # (B,) no grad
    logits = pi_logratios - ref_logratios                        # (B,) implicit margin
    return -F.logsigmoid(beta * logits).mean()
```

A full DPO step wires `sequence_logprob` to the policy and a frozen reference:

```python
import torch
from llmre.preference.dpo import dpo_loss, sequence_logprob

# policy is trainable; ref is frozen (torch.no_grad, requires_grad off).
def dpo_step(policy, ref, chosen_ids, chosen_labels, rej_ids, rej_labels, beta=0.1):
    pc = sequence_logprob(policy(chosen_ids), chosen_labels)      # (B,) grad
    pr = sequence_logprob(policy(rej_ids),    rej_labels)         # (B,) grad
    with torch.no_grad():
        rc = sequence_logprob(ref(chosen_ids), chosen_labels)    # (B,) frozen
        rr = sequence_logprob(ref(rej_ids),    rej_labels)       # (B,) frozen
    return dpo_loss(pc, pr, rc, rr, beta=beta)
```

The unit test `test_dpo_gradient_increases_the_chosen_minus_rejected_margin` starts the policy log-ratios at zero and takes a few SGD steps; the chosen-minus-rejected margin provably increases, and the **implicit rewards** (`dpo_reward_margin`, $\beta\log(\pi_\theta/\pi_{\text{ref}})$) end with chosen above rejected — the loss is doing exactly what the derivation promised.

### Under the hood / cost

- **Two models, two forwards each.** DPO forwards the policy on chosen and rejected, and the reference on chosen and rejected — four forwards per pair, no generation. The reference log-probs can be **precomputed once** and cached, dropping steady-state cost to two forwards per pair. Contrast PPO's four live models plus a sampling loop.
- **Memory.** The frozen reference must be resident (or its log-probs cached). No value head, no reward model, no replay buffer.
- **Stability.** It is a smooth supervised loss, so it trains like SFT — far fewer knobs than PPO ($\beta$ is the main one).

<div class="callout warn"><p>The reference must be the model you started DPO from (typically the SFT model), and it must be <strong>frozen</strong> — <code>eval()</code>, <code>requires_grad=False</code>, and wrapped in <code>torch.no_grad()</code> when you compute its log-probs. If the reference accidentally trains, the log-ratio collapses toward zero and DPO has no target to push against.</p></div>

<div class="hw">
<p><strong>Hardware track.</strong> DPO is roughly SFT-cost with a second (frozen) model. <strong>Min</strong> 1&times; 24GB GPU for a ~1B model with LoRA (cache reference log-probs to fit); <strong>recommended</strong> 1&times;–2&times; A100 40GB for 7B. <strong>Runtime</strong> hours over ~100k pairs; <strong>GPU-hours</strong> single-digit to low tens for small models — much cheaper than PPO. <strong>CPU-only</strong>: the <code>dpo_loss</code>/<code>sequence_logprob</code> unit tests run instantly on CPU; real DPO training needs a GPU.</p>
</div>

## Common mistakes

- **Averaging instead of summing token log-probs.** `sequence_logprob` must **sum** over the completion (it is $\log$ of a product of per-token probabilities). Averaging changes the implicit reward's scale and interacts badly with $\beta$.
- **Letting the reference train / dropping `torch.no_grad()`.** The reference is a fixed anchor; if it moves, the objective degenerates.
- **Not masking the prompt.** Only the completion tokens should contribute to $\log\pi(y|x)$. Set prompt/pad labels to `ignore_index` (see [14.2](../module-14/lesson-02.md)).
- **Sign or ratio-order slips.** It is $(\text{policy chosen} - \text{policy rejected}) - (\text{ref chosen} - \text{ref rejected})$. Swapping chosen/rejected trains the model to prefer the *worse* answer.

## Debugging exercise

Someone reports their DPO run "learns nothing" — the loss sits almost exactly at `0.6931` and the chosen/rejected implicit rewards stay equal. They insist the policy is training and gradients flow. What is the most likely bug?

<details><summary>Answer</summary>

`0.6931 = log 2 = -log &sigma;(0)`: the implicit margin is stuck at 0. The overwhelmingly common cause is that the **reference is the same object as the policy** (or the reference is being updated alongside it). Then $\log\pi_\theta - \log\pi_{\text{ref}} = 0$ for both chosen and rejected at every step, so the margin is identically zero and there is nothing to push. Fix: make the reference a separate, frozen copy of the starting model (`eval()`, `requires_grad=False`, `torch.no_grad()` for its log-probs), or precompute and cache its log-probs before training. (Contrast the same `0.6931` symptom in 15.1, where the cause was identical chosen/rejected inputs to the reward model.)

</details>

## Exercise

DPO's implicit reward for a completion is $\hat r(x,y) = \beta\log(\pi_\theta(y|x)/\pi_{\text{ref}}(y|x))$. Implement `reward_accuracy(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)` returning the fraction of pairs the *implicit* reward ranks correctly ($\hat r_w > \hat r_l$).

*Hint:* like the reward-model accuracy in 15.1, you only need the sign of a margin — and $\beta>0$ doesn't change a sign.

<details><summary>Solution</summary>

```python
def reward_accuracy(pcw, pcl, rcw, rcl, beta):
    chosen_reward   = beta * (pcw - rcw)      # implicit reward for chosen
    rejected_reward = beta * (pcl - rcl)      # implicit reward for rejected
    return (chosen_reward > rejected_reward).float().mean()
```

Since $\beta > 0$, this equals `((pcw - rcw) > (pcl - rcl)).float().mean()`, i.e. `dpo_reward_margin(...)` then compare. It is the standard "reward accuracy" logged during DPO; a healthy run climbs from ~0.5 toward 0.7+.

</details>

## Research connection

<div class="callout paper"><p><strong>Read:</strong> <a href="../../papers/index.md">DPO (Rafailov et al. 2023)</a>. Read Section 4 (the derivation above — the optimal policy, the reward reparameterization, the loss) and the gradient interpretation closely; skim the theoretical equivalence proofs on a first pass. This paper is the reason most open-weight models today are aligned with DPO rather than PPO.</p></div>

## Check yourself

<details><summary>Where does the intractable partition function $Z(x)$ go?</summary>

It appears in the reward reparameterization $r(x,y) = \beta\log(\pi^*/\pi_{\text{ref}}) + \beta\log Z(x)$. But DPO only ever uses the *difference* of rewards for the two completions of the **same prompt** $x$, and $\beta\log Z(x)$ is identical for both, so it cancels in the subtraction. That cancellation is the whole trick — it removes the one intractable piece.

</details>

<details><summary>What two networks does DPO evaluate, and what does it <em>not</em> need?</summary>

It evaluates the trainable **policy** $\pi_\theta$ and a **frozen reference** $\pi_{\text{ref}}$, forwarding each on the chosen and rejected completions. It needs **no reward model** and **no sampling/generation** — that is what makes it as cheap and stable as SFT.

</details>

<details><summary>Why sum (not average) the per-token log-probs in <code>sequence_logprob</code>?</summary>

$\log\pi(y|x) = \log\prod_t \pi(y_t|\cdot) = \sum_t \log\pi(y_t|\cdot)$ — the sequence log-probability is the sum of per-token log-probs. Averaging would compute per-token mean log-prob, which is a different quantity and would rescale the implicit reward, breaking its calibration against $\beta$.

</details>

<details><summary>Recompute the DPO loss if the policy exactly matches the reference (all four log-probs give pi_logratios = ref_logratios). What is it?</summary>

Then the margin `pi_logratios - ref_logratios = 0`, so $\beta\cdot 0 = 0$, $\sigma(0)=0.5$, and the loss is $-\log 0.5 = \log 2 = 0.693147$. A policy identical to the reference sits exactly at the maximum-uncertainty loss — there is no preference signal yet.

</details>

<details><summary>What role does $\beta$ play, and how does it relate to PPO?</summary>

$\beta$ is the same KL weight as the RLHF/PPO objective. Large $\beta$ keeps $\pi_\theta$ close to the reference (small updates); small $\beta$ lets it move far. Because the reference is inside the DPO loss itself, the KL constraint is built in — DPO needs no separate KL-penalty term the way PPO does.

</details>

## Next

We have three ways to shape behavior: SFT (imitate), RM+PPO (optimize a learned reward with RL), and DPO (optimize preferences directly). Next we compare all three head-to-head — what each optimizes, what infrastructure each demands, how each fails — and look at replacing human labels with AI feedback (Constitutional AI / RLAIF).

Continue to [15.4 · SFT vs RM+PPO vs DPO](lesson-04.md).
