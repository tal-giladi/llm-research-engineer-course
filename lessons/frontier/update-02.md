# F.2 · GRPO's instability and the fix design space: Dr. GRPO, DAPO, GVPO (Sep 2026)

<div class="callout key"><p><strong>Frontier update, not core curriculum.</strong> GVPO is <em>directional, not validated</em>: one paper line, no known frontier-lab adoption yet. Read it to understand the design space, not to swap out GRPO in your pipeline.</p></div>

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-16/lesson-01">16.1 · From PPO to GRPO</a> (group advantage, clipped ratio), <a href="#/lessons/module-15/lesson-03">15.3 · DPO</a> (the closed-form optimum of KL-constrained reward maximization and the "implicit reward"), <a href="#/lessons/frontier/update-01">F.1</a>.</p>
<p><strong>You will learn:</strong> where GRPO's instability comes from; what Dr. GRPO and DAPO change; how GVPO builds the analytical optimum into the gradient weights and drops importance sampling; a from-scratch <code>gvpo_loss</code> in <code>llmre/rl/gvpo.py</code> whose minimum is exactly $\pi^* \propto \pi_{\text{ref}}\, e^{R/\beta}$.</p>
<p><strong>Why this matters for ML:</strong> "why is GRPO unstable and what are the fixes" is common interview and practice material on the fine-tuning path. The fixes reuse ideas you already have (PPO clipping, DPO's closed form), just combined differently.</p>
</div>

## 1. Where GRPO gets unstable

Recall the GRPO per-sample objective from [16.1](lessons/module-16/lesson-01.md):

$$
\min\Big(r_i A_i,\ \text{clip}(r_i, 1-\epsilon, 1+\epsilon)\, A_i\Big), \qquad r_i = \frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)}, \qquad A_i = \frac{R_i - \bar R}{\text{std}(R)}
$$

Three weak points, each targeted by a published fix:

1. **The importance ratio $r_i$ is unbounded.** For a long completion, $\log r_i$ is a *sum* over hundreds of tokens, so small per-token drift compounds. Clipping caps the objective, not the underlying variance, and clipped samples contribute zero gradient (wasted compute).
2. **Normalizations add bias.** Dividing by std up-weights groups with low variance (nearly-all-right or nearly-all-wrong), and per-sequence length normalization $1/|y_i|$ changes how much long vs short answers are penalized.
3. **Zero-signal groups.** All-correct or all-wrong groups give $A = 0$ (seen in [F.1 §3](lessons/frontier/update-01.md)), shrinking the effective batch.

## 2. The fixes, side by side

| Method | What it changes | Status |
|---|---|---|
| **Dr. GRPO** — [Liu et al. 2025, arXiv 2503.20783](https://arxiv.org/abs/2503.20783) | Removes the $1/\lvert y_i\rvert$ length normalization and the std division. Argues both bias training (e.g. toward long wrong answers). | PUBLICLY DOCUMENTED, widely discussed |
| **DAPO** — [Yu et al. 2025, arXiv 2503.14476](https://arxiv.org/abs/2503.14476) | Decoupled clip ("clip-higher": larger upper $\epsilon$), dynamic sampling (drop all-0 / all-1 groups and refill), token-level loss averaging, overlong-response reward shaping. | PUBLICLY DOCUMENTED, used in open RL stacks |
| **GVPO** — [Zhang et al. 2025, arXiv 2504.19599](https://arxiv.org/abs/2504.19599); extended as **GVPO++**, [arXiv 2609.21432](http://arxiv.org/abs/2609.21432v1) | Replaces ratio + clip with a weight derived from the closed-form KL-constrained optimum. No importance sampling. Unique optimum. Extends to on-policy distillation. | Directional; no known frontier adoption |

Dr. GRPO and DAPO *patch* GRPO. GVPO *changes the objective*. That is why it is worth understanding even before it is worth adopting.

## 3. GVPO, six passes

### 3.1 Intuition

You already know from [15.3 · DPO](lessons/module-15/lesson-03.md) the exact answer to "maximize reward, stay close to a reference policy in KL":

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, e^{R(x, y)/\beta}.
$$

DPO used that on *preference pairs*. GVPO uses it on a *GRPO group*. The only obstacle is $Z(x)$, which you cannot compute. But $Z(x)$ is the same for every completion of the same prompt. So if you compare completions *within a group* by subtracting the group mean, $Z$ cancels. That is the whole trick.

### 3.2 Mathematics

Define the **implicit reward** of the current policy (the DPO quantity):

$$
R_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
$$

- $\pi_\theta$: policy being trained
- $\pi_{\text{ref}}$: reference policy (the paper writes $\pi_{\theta'}$)
- $\beta > 0$: KL strength
- $R(x, y)$: actual reward (e.g. 0/1 from a verifier)

At the optimum, $R_{\theta}(x, y) = R(x, y) - \beta \log Z(x)$. Centering over a group of $k$ samples $y_1, \dots, y_k$ removes $\beta \log Z(x)$. GVPO minimizes the squared mismatch of the centered quantities:

$$
\mathcal{L}_{\text{GVPO}} = \frac{1}{2} \sum_{x} \sum_{i=1}^{k} \Big[ \big(R_\theta(x, y_i) - \overline{R_\theta}\big) - \big(R(x, y_i) - \overline{R}\big) \Big]^2
$$

where bars are means over the group. Its gradient (the form in the paper) is

$$
\nabla_\theta \mathcal{L} = -\beta \sum_{x} \sum_{i=1}^{k} \Big[ (R_i - \overline{R}) - \beta\big(\ell_i - \overline{\ell}\big) \Big] \nabla_\theta \log \pi_\theta(y_i \mid x), \qquad \ell_i = \log \frac{\pi_\theta(y_i \mid x)}{\pi_{\text{ref}}(y_i \mid x)}.
$$

Read the bracket as a **weight** on each sample's log-prob gradient: "centered reward, minus how much the policy has already moved toward this sample." The weights sum to zero within a group. The paper shows this is what makes $Z(x)$ drop out, and proves the objective has a unique optimum at $\pi^*$.

Compare with GRPO: GRPO weights by $r_i A_i$ (ratio × standardized reward). GVPO weights by (centered reward − β × centered log-ratio). No ratio, no clip. The second term acts like a built-in brake: once the policy has moved enough toward a good sample, its weight goes to zero.

### 3.3 Numerical example by hand

One prompt, $k = 2$ samples, $\beta = 1$, rewards $R = [1, 0]$. Start at $\pi_\theta = \pi_{\text{ref}}$, so $\ell = [0, 0]$.

- centered reward: $[0.5, -0.5]$
- centered implicit reward: $[0, 0]$
- loss $= \tfrac{1}{2}\big[(0 - 0.5)^2 + (0 + 0.5)^2\big] = 0.25$
- gradient weight on $y_1$: $(0.5) - 1 \cdot 0 = 0.5$ → raise $\log \pi_\theta(y_1)$; on $y_2$: $-0.5$ → lower it.

Where does it stop? When centered implicit reward equals centered reward: $\ell_1 - \ell_2 = (R_1 - R_2)/\beta = 1$. So $\frac{\pi_\theta(y_1)}{\pi_\theta(y_2)} = e \cdot \frac{\pi_{\text{ref}}(y_1)}{\pi_{\text{ref}}(y_2)}$ — exactly the ratio $\pi^*$ prescribes. With $\beta = 0.1$ the target ratio would be $e^{10} \approx 22026$: small $\beta$ lets the policy move much further from the reference.

### 3.4 Shapes, dtype, device

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `logprobs` | `(P, G)` | float32 / bf16 | summed token log-probs per completion, requires grad |
| `ref_logprobs` | `(P, G)` | same | frozen reference forward, no grad |
| `rewards` | `(P, G)` | float32 | 0/1 from the verifier |
| loss | `()` or `(P,)` | float32 | `reduction="none"` → per prompt |

$P$ = prompts in the batch, $G$ = group size. All on the policy's device.

### 3.5 From-scratch implementation

`code/src/llmre/rl/gvpo.py`:

```python
def gvpo_loss(logprobs, ref_logprobs, rewards, beta=0.1, reduction="mean"):
    implicit = beta * (logprobs - ref_logprobs)                       # (P, G)
    implicit_c = implicit - implicit.mean(dim=1, keepdim=True)        # Z(x) cancels
    reward_c = rewards - rewards.mean(dim=1, keepdim=True)            # (P, G)
    per_prompt = 0.5 * ((implicit_c - reward_c) ** 2).sum(dim=1)      # (P,)
    ...
```

Check the hand example and the optimum:

```python
import torch
from llmre.rl.gvpo import gvpo_loss

ref = torch.log(torch.tensor([[0.5, 0.5]]))
R = torch.tensor([[1.0, 0.0]])
print(gvpo_loss(ref, ref, R, beta=1.0))          # tensor(0.2500)

opt = torch.log_softmax(ref + R / 1.0, dim=1)     # pi_ref * exp(R/beta) / Z
print(gvpo_loss(opt, ref, R, beta=1.0))          # tensor(0.)  (up to float error)
```

`code/tests/test_frontier_updates.py` also runs plain gradient descent on this loss from a uniform start and checks the policy converges to $\text{softmax}(\log \pi_{\text{ref}} + R/\beta)$, and that adding any per-prompt constant to the rewards leaves the loss unchanged.

### 3.6 Cost and what real frameworks would do

- **Compute:** same forward passes as GRPO with a KL term: policy log-probs (with grad) plus a frozen reference forward. No extra passes; the loss itself is $O(PG)$ elementwise ops.
- **Memory:** no old-policy log-probs to store for ratios (the reference forward replaces them). The reference model is still in memory, as in GRPO-with-KL.
- **INFERENCE:** because there is no ratio, GVPO does not by itself give the "safe to reuse this batch for several steps" guarantee that clipping gives PPO/GRPO. How it behaves under async staleness (F.1 §2.4) is an open practical question.

## 4. On-policy distillation (why the paper mentions it)

Replace the verifier reward with a teacher's log-probability of the student's own sample, $R(x, y) = \log \pi_{\text{teacher}}(y \mid x)$. The same centered-MSE loss then pushes the student toward the teacher *on the student's own samples*. Same code, different reward. The GVPO++ paper presents this as a natural extension.

## Common mistakes

- **Treating GVPO as a drop-in for production GRPO.** Unvalidated at frontier scale. Run it only as an ablation against a GRPO baseline.
- **Forgetting to center both terms.** Without centering, $Z(x)$ does not cancel and the loss is wrong.
- **Using per-token log-probs instead of per-sequence sums** for `logprobs`. The derivation is over whole completions $y$.
- **Confusing $\pi_{\text{ref}}$ with $\pi_{\text{old}}$.** GVPO's anchor is the reference in the KL constraint, not the sampling policy of the last step.

## Debugging exercise

A colleague's GVPO run: loss goes to ~0 in 50 steps, but held-out pass@1 did not change. Rewards are all 0 in 95% of groups. What happened?

<details><summary>Answer</summary>

If a group's rewards are all equal, centered reward is 0 for every sample, so the loss only asks the centered implicit reward to be 0 as well: stay at the reference on those samples. With 95% zero-signal groups the loss is minimized by barely moving. Same root cause as GRPO's zero-advantage groups. Fix the task difficulty or use DAPO-style dynamic sampling (drop unanimous groups and refill the batch).

</details>

## Exercise

With $\beta = 0.5$, $G = 3$, rewards $[1, 1, 0]$ and a uniform reference over the three samples, what probabilities does GVPO's optimum assign to them (restricted to these three)?

<details><summary>Hint</summary>

$\pi^* \propto \pi_{\text{ref}} \cdot e^{R/\beta}$, and $\pi_{\text{ref}}$ is uniform.

</details>

<details><summary>Stronger hint</summary>

The unnormalized weights are $e^{2}, e^{2}, e^{0}$.

</details>

<details><summary>Solution</summary>

$e^2 \approx 7.389$. Total $= 2 \cdot 7.389 + 1 = 15.778$. So $\pi^* \approx [0.468, 0.468, 0.063]$. Check: $\log(0.468/0.063) \approx 2 = (1 - 0)/0.5$. ✓

</details>

## Research connection / what to read

- **Read:** GVPO (arXiv 2504.19599) §3: the derivation and the "weights sum to zero ⇒ $Z$ cancels" argument. Then skim GVPO++ (arXiv 2609.21432) for the on-policy distillation extension.
- **Read:** Dr. GRPO §3 on the length and std biases; DAPO §3 for the four practical tricks.
- **Skip on first pass:** the convergence-rate appendices.

## Code this after reading

Swap `gvpo_loss` into the toy RLVR loop from [16.2](lessons/module-16/lesson-02.md) in place of the GRPO objective (log-probs of sampled answers are already `(P, G)`), and plot correct-rate vs step for both on the same seeds.

## Check yourself

<details><summary>Why can GVPO ignore the partition function Z(x)?</summary>

$Z(x)$ adds the same constant $-\beta \log Z(x)$ to every completion's implicit reward for a prompt. Subtracting the group mean removes it.

</details>

<details><summary>What does GRPO have that GVPO drops, and why does that matter?</summary>

The importance ratio $\pi_\theta/\pi_{\text{old}}$ and its clipping. The ratio is unbounded and its variance compounds over long sequences; dropping it removes that instability source.

</details>

<details><summary>What is the unique optimum of the GVPO objective?</summary>

$\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x)\, e^{R(x, y)/\beta}$, the closed-form solution of KL-constrained reward maximization — the same one DPO is built on.

</details>

<details><summary>Name one change each from Dr. GRPO and DAPO.</summary>

Dr. GRPO: remove length normalization and std normalization. DAPO: clip-higher (decoupled upper clip), dynamic sampling, token-level loss, or overlong reward shaping.

</details>

## Next

Back to the core path: [17.1 · CoT, self-consistency, STaR](lessons/module-17/lesson-01.md), or go straight to the [F.1 lab](lessons/frontier/update-01.md).
