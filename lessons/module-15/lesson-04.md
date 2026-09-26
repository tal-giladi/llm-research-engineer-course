# 15.4 · SFT vs RM+PPO vs DPO

<div class="prereq">
<p><strong>Prerequisites:</strong> SFT from <a href="#/lessons/module-14/lesson-01">14.1</a>; the reward model from <a href="#/lessons/module-15/lesson-01">15.1</a>; PPO, the KL penalty, and reward hacking from <a href="#/lessons/module-15/lesson-02">15.2</a>; the DPO derivation and loss from <a href="#/lessons/module-15/lesson-03">15.3</a>.</p>
<p><strong>You will learn:</strong> a side-by-side comparison of the three post-training pipelines — what each <em>optimizes</em>, what <em>infrastructure</em> each needs, and how each <em>fails</em> (reward hacking, over-optimization, distribution shift); a decision guide for when to reach for which; and a brief tour of replacing human labels with AI feedback — Constitutional AI and RLAIF.</p>
<p><strong>Why this matters for ML:</strong> choosing the alignment method is one of the highest-leverage decisions in a post-training project. It sets your compute budget, your data-collection strategy, and your failure surface. This lesson turns the mechanics of 14.1–15.3 into an engineering decision.</p>
</div>

## The causal story so far

Three lessons, three ways to shape a model's behavior after pretraining:

- **SFT** (14.1–14.2): imitate demonstrations with next-token cross-entropy. Teaches the *format* and *following* of instructions, but treats every demonstration as equally correct — it cannot rank.
- **RM + PPO** (15.1–15.2): train a reward model from preferences, then use RL to push the policy toward higher reward, kept near the reference by a KL penalty. Optimizes preference directly but with heavy machinery.
- **DPO** (15.3): the same preference objective, rearranged into a single supervised loss on preference pairs — no reward model, no sampling.

They are usually **stages, not rivals**: almost every aligned model does SFT *first*, then preference optimization (PPO or DPO) on top. The question is which preference method to use for the second stage.

## What each one optimizes

| | Objective | Signal | Data |
|---|---|---|---|
| **SFT** | $-\sum_t \log \pi_\theta(y_t \mid y_{<t}, x)$ on response tokens | Match one gold response | `(prompt, ideal response)` demonstrations |
| **RM+PPO** | $\mathbb{E}[r(x,y)] - \beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$ via policy gradient | Maximize a *learned* scalar reward | preference pairs → reward model → on-policy samples |
| **DPO** | $-\mathbb{E}\log\sigma\!\big(\beta\log\tfrac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\tfrac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big)$ | Increase the implicit-reward margin (chosen > rejected) | preference pairs `(prompt, chosen, rejected)` |

Key relationships, established across the module:

- SFT's target is a *point* (one gold answer); preference methods' target is an *ordering* (chosen beats rejected). That is exactly why SFT hits a quality ceiling (15.1) and preference tuning can push past it.
- **PPO and DPO optimize the same KL-constrained objective** — DPO's derivation (15.3) *starts* from PPO's objective. They differ in *how*: PPO with RL and an explicit reward model, DPO with a closed-form supervised loss.

## What infrastructure each needs

This is usually the deciding factor.

| | Models resident during training | Generation in the loop? | Main hyperparameters | Relative cost |
|---|---|---|---|---|
| **SFT** | 1 (policy) | no | LR, epochs | lowest |
| **RM+PPO** | up to 4 — policy (+value head), frozen reference, reward model | **yes** (autoregressive rollouts every step) | $\epsilon$, $\beta_{\text{KL}}$, LR, GAE $\lambda$, rollout/batch sizes | highest |
| **DPO** | 2 — policy + frozen reference (reference log-probs cacheable → effectively 1) | no | $\beta$, LR | ~SFT + a frozen forward |

PPO's cost is dominated by two things: keeping several large models in memory at once, and **generating** completions every step (slow, and 15.2's whole point). DPO removes both — no reward model, no rollouts — which is why it trains like SFT. This engineering gap, more than any quality argument, is why DPO became the default for open-weight models.

## How each one fails

Every method has a characteristic failure mode. Knowing them is half of using them well.

### Reward hacking (RM+PPO)

The reward model is a *proxy* for human preference, not the real thing. Optimize a proxy hard enough and the policy finds its blind spots — outputs that score high on the reward model but are actually bad (verbose, sycophantic, formulaic). This is **Goodhart's law**: when a measure becomes a target, it stops being a good measure. The KL penalty (15.2) is the main defense; the tell is reward climbing while KL-to-reference explodes and human ratings drop (the 15.2 debugging exercise).

### Over-optimization (RM+PPO, and DPO too)

Even without pathological hacking, pushing *too hard* against a fixed reward/preference signal eventually hurts true quality — measured quality and real quality diverge past some point. In PPO this is the KL-vs-reward tradeoff. **DPO is not immune**: with small $\beta$ or many epochs it can over-fit the preference pairs and degrade, and it has a known failure of driving *both* chosen and rejected log-probabilities down (it only optimizes the *margin*, so it can satisfy the loss by making the rejected far less likely while also lowering the chosen). Watch the absolute chosen log-prob, not just the margin.

### Distribution shift (RM+PPO especially)

The reward model was trained on completions from *some* distribution (often the SFT model's). As PPO improves the policy, it generates completions the reward model **never saw**, so the reward model's scores become unreliable exactly where it now matters — the reward model is off-distribution. Production RLHF handles this by **iterating**: periodically collect fresh preferences on the current policy's outputs and retrain the reward model. DPO on a fixed offline dataset has an analogous issue — the preference pairs may not cover the policy's new behavior — which motivates *online/iterative* DPO variants that regenerate pairs from the current policy.

### SFT's failure: it just imitates

SFT's failure is the whole reason the module exists (15.1): it cannot exceed the quality of its demonstrations and cannot express "A is better than B." Its more subtle failure is **exposure bias** — trained only on gold prefixes (teacher forcing), it never learns to recover from its own mistakes at generation time. Preference methods, which score the model's *own* style of outputs, partly address this.

## When to use which

<div class="callout key"><p><strong>Default recipe:</strong> SFT first (always), then <strong>DPO</strong> for preference tuning unless you have a specific reason to run PPO. DPO gives most of the benefit at a fraction of the infrastructure and instability.</p></div>

- **SFT only** — you have good demonstrations, need instruction-following and format adherence, and don't yet have preference data. The cheapest, always the first stage.
- **DPO** — you have offline preference pairs and want preference tuning that trains like SFT. The pragmatic default; dominates the open-weight ecosystem.
- **RM + PPO** — you can afford the infrastructure and want the ceiling: an explicit reward model you can *reuse* (for best-of-$n$ sampling, filtering, or to score many candidates), tight online control of the reward/KL tradeoff, and on-policy data. The reward model is itself a reusable asset. Frontier labs have used full RLHF-PPO for their flagship post-training.
- **Hybrid / iterative** — SFT → reward model → *iterative* DPO or online RL, regenerating preferences from the current policy to fight distribution shift. Common in recent open recipes (e.g. Llama-3-style post-training uses rejection sampling + DPO).

<div class="callout warn"><p>Do not skip SFT and jump to preference optimization on a base model. DPO and PPO both rely on a sensible <strong>reference policy</strong> and a policy that already follows the chat format; the SFT model provides both. Preference tuning <em>refines</em> behavior — it does not install it.</p></div>

## AI feedback: Constitutional AI and RLAIF

The bottleneck in all of the above is **human preference labels** — slow, expensive, and inconsistent. Two approaches replace (some of) them with AI feedback.

### Constitutional AI (CAI)

<div class="callout paper"><p><strong>Read:</strong> <a href="#/papers/index">Constitutional AI (Bai et al. 2022)</a> and <a href="#/papers/index">RLAIF (Lee et al. 2023)</a>.</p></div>

Constitutional AI (Anthropic) trains harmlessness with minimal human harm labels, in two phases (**PUBLICLY DOCUMENTED**, from the paper):

1. **Supervised phase.** The model generates a response, then is prompted to **critique** its own response against a written list of principles (the "constitution") and **revise** it. Fine-tune on the revised responses. This is self-improvement guided by stated principles.
2. **RL phase (RLAIF).** Instead of humans labeling which of two responses is better, a model does — a "**preference model** from **AI feedback**." That AI-generated preference data trains a reward model, and RL proceeds as in 15.2.

The constitution makes the values **explicit and inspectable** (a document you can read and edit) rather than implicit in thousands of human labels.

### RLAIF vs RLHF

RLAIF = RLHF with the preference labeler swapped from human to AI. The reported finding (**PUBLICLY DOCUMENTED** in the RLAIF paper): on the tasks studied, RLAIF performs **comparably** to RLHF, and it scales far more cheaply because AI labels are fast and abundant.

Honesty markers, as the brief requires:

- **PUBLICLY DOCUMENTED:** the CAI critique-revise recipe and its use of AI-generated preferences; the RLAIF comparability result on the studied tasks; that DPO optimizes the same objective as the RLHF/PPO derivation (15.3).
- **REASONABLE INDUSTRY PRACTICE:** SFT → preference optimization as the standard post-training pipeline; iterating the reward model on fresh on-policy data to combat distribution shift; using DPO as the default open-weight preference method; caching reference log-probs for DPO.
- **INFERENCE / SPECULATION:** the exact post-training recipe, data mix, or RLHF-vs-DPO choice used inside any specific frontier product is generally **not** publicly documented; do not assert that a named lab "uses X" for a current model unless they published it.

## Common mistakes

- **Treating the three as competitors.** They are stages. Preference tuning assumes an SFT starting point.
- **Reaching for PPO by default.** Unless you specifically need a reusable reward model or online control, DPO is cheaper and more stable for the same objective.
- **Ignoring absolute log-probs in DPO.** The margin can improve while both chosen and rejected probabilities fall — monitor the chosen log-prob and reward accuracy, not just the loss.
- **Using a stale reward model in long PPO runs.** Distribution shift makes it unreliable on the policy's improved outputs; refresh it.
- **Assuming AI feedback is free of bias.** RLAIF inherits the labeler model's biases and blind spots; it reduces cost, not the need for oversight.

## Debugging exercise

Team A ran DPO and reports: "reward accuracy on the training pairs is now 0.95 and the loss is tiny, but the model's actual answers got *worse* — shorter, and it sometimes refuses reasonable requests." What are the two most likely explanations, drawing on this lesson?

<details><summary>Answer</summary>

(1) **Over-optimization** with too-small $\beta$ (or too many epochs): the policy has drifted far from the reference to maximize the margin, overfitting the specific preference pairs. High training reward accuracy with degraded real quality is the signature. Fix: raise $\beta$, fewer epochs, early-stop on a held-out set. (2) The **margin-only** pathology: DPO can lower the rejected log-prob so aggressively that it also drags down the chosen log-prob and generally suppresses output (hence shorter answers / over-refusal). The margin looks great while absolute quality falls. Fix: monitor the absolute chosen log-prob; consider a variant that anchors it (e.g. adding an SFT/NLL term on the chosen response). Both are forms of over-optimization / distribution shift, not a coding bug.

</details>

## Check yourself

<details><summary>Do PPO and DPO optimize different objectives?</summary>

No — the *same* KL-constrained objective. DPO's derivation (15.3) starts from PPO's objective and rearranges it into a closed-form supervised loss. They differ in method (RL with an explicit reward model vs. a direct classification loss), not in what they ultimately optimize.

</details>

<details><summary>Name the characteristic failure mode of RM+PPO and the mechanism that defends against it.</summary>

Reward hacking / over-optimization: the policy exploits blind spots of the imperfect reward model, scoring high while true quality drops. The KL penalty to the frozen reference is the main defense — it keeps the policy near a trusted model, bounding how far it can drift to game the reward. The warning sign is KL exploding while reward rises and human ratings fall.

</details>

<details><summary>Why is DPO usually cheaper than PPO for the same goal?</summary>

DPO needs no reward model and no generation loop. It forwards the policy and a frozen reference (whose log-probs can be cached) on the given chosen/rejected completions and applies a supervised loss — effectively SFT cost plus one frozen forward. PPO keeps up to four models resident and generates completions every step, which dominates its cost.

</details>

<details><summary>What is distribution shift in RLHF and how is it mitigated?</summary>

The reward model is trained on one distribution of completions, but as the policy improves it produces completions the reward model never saw, so its scores become unreliable exactly where they now matter. Mitigation: iterate — periodically regenerate preferences from the current policy and retrain the reward model (and analogously, use online/iterative DPO instead of a fixed offline set).

</details>

<details><summary>What does the "constitution" in Constitutional AI replace, and with what?</summary>

It replaces large volumes of human harm labels with a written list of principles the model uses to critique and revise its own outputs (supervised phase), and to generate AI preference labels for a reward model (RLAIF phase). The values become an explicit, editable document rather than being implicit in thousands of human labels.

</details>

## Next

You can now pick and run a post-training method. Module 16 takes RL in a new direction: instead of a learned reward model of human preference, it uses **verifiable rewards** (does the math answer check out? does the code pass tests?) and the **GRPO** algorithm that grew out of PPO — the machinery behind modern reasoning models.

Continue to [16.1 · From PPO to GRPO](lessons/module-16/lesson-01.md).
