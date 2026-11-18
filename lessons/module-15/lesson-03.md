# 15.3 · DPO: derivation & implementation

<div class="prereq">
<p><strong>Prerequisites:</strong> the Bradley-Terry model and reward loss from <a href="#/lessons/module-15/lesson-01">15.1</a>; the KL-constrained RLHF objective, the reference policy, and sequence log-probabilities from <a href="#/lessons/module-15/lesson-02">15.2</a>; log-softmax and gathering token log-probs from <a href="#/lessons/module-01/lesson-04">01.4</a>; the response-only masking of <a href="#/lessons/module-14/lesson-02">14.2</a>.</p>
<p><strong>You will learn:</strong> how <strong>Direct Preference Optimization</strong> trains a model straight from a file of labeled answer pairs. First the practical calculation: follow one labeled pair from next-token probabilities, through sequence log-probabilities, the frozen reference, the DPO loss and backpropagation, down to the token logits. Then the derivation of <em>why</em> this loss is right: the closed-form optimal policy of the RLHF objective, inverting it to express the reward through the policy, and substituting into Bradley-Terry so the reward model and the partition function <em>disappear</em>. Finally you implement <code>sequence_logprob</code> and <code>dpo_loss</code> from scratch and verify them numerically.</p>
<p><strong>Why this matters for ML:</strong> DPO turned preference tuning from a fragile four-model RL loop into a stable supervised loss you can run like SFT. It is the default alignment method for most open-weight models today. And its derivation is the single most instructive piece of algebra in post-training: it shows the RLHF objective and a simple classification loss are two views of the same thing.</p>
</div>

<div class="hw">
<p><strong>Hardware track.</strong> DPO is roughly SFT-cost with a second (frozen) model. <strong>Hands-on in this lesson:</strong> CPU only, seconds (one-liners and the unit tests). <strong>Real training: min</strong> 1&times; 24GB GPU for a ~1B model with LoRA (cache reference log-probs to fit); <strong>recommended</strong> 1&times;–2&times; A100 40GB for 7B. <strong>Runtime</strong> hours over ~100k pairs; <strong>GPU-hours</strong> single-digit to low tens for small models — much cheaper than PPO.</p>
</div>

DPO can look like a clever classification trick: show the model a prompt, a preferred answer, and a rejected answer, then train it to score the preferred answer higher. That description is true but incomplete. It does not explain why the score is a log-probability *ratio*, why a reference model appears, or why the loss is a sigmoid of a difference of differences.

This lesson starts from the operation you already know: feed tokens into a language model and get probabilities for the next token. We follow those numbers through one complete DPO training step (sections 3–11). Only after the inputs, calculations, loss and gradients are clear do we derive why this particular loss connects to reward maximization (sections 12–14).

<div class="callout key"><p><strong>One-sentence intuition:</strong> DPO adjusts the policy so that, relative to a fixed reference model, the preferred answer becomes more likely than the rejected answer by an amount supported by the preference data.</p></div>

## 1. What problem are we solving?

A language model is a **policy** $\pi$: for each prompt $x$ it assigns a probability $\pi(y\mid x)$ to every possible answer $y$. An SFT dataset gives one target answer and trains the model to imitate it. But for many prompts there is no single correct answer. We may only know that one of two answers is **better**.

A preference record is written $(x, y_w, y_l)$:

- $x$ — the prompt.
- $y_w$ — the **winning** (preferred, "chosen") response.
- $y_l$ — the **losing** (rejected) response.

"Preferred" may reflect helpfulness, correctness, harmlessness, style, or a mix set by the annotation instructions. It is a **comparison, not an absolute score**. It says $y_w \succ y_l$ for this prompt; it does not tell us the numerical reward of either answer.

For "Explain why the sky looks blue", the chosen answer could be accurate and concise, the rejected one misleading or bloated. We know the ordering, but not whether the winner deserves reward 0.7 and the loser 0.2, or 8 and 2. Any method must use relative information without pretending the annotator supplied an absolute scale.

**Notation for the whole lesson.** $x$: prompt. $y$: answer. $\theta$: the weights of the trainable language model. **Policy** $\pi_\theta$: that language model viewed as a distribution over whole answers. **Reference** $\pi_{\text{ref}}$: a frozen language model used as a baseline. $\pi(y\mid x)$ always means the probability of the **entire** answer sequence; because the model is autoregressive, it is a product of next-token probabilities (section 4 unpacks it).

## 2. From PPO and reward models to DPO

You already know the RLHF route from 15.1–15.2, so use it as the bridge. First a reward model learns to score answers from preference comparisons. Then PPO optimizes the policy to get high reward while staying near a reference, with a KL penalty:

$$
\max_{\pi}\; \mathbb{E}_{x\sim\mathcal D}\Big[\mathbb{E}_{y\sim\pi(\cdot\mid x)}\big[r(x,y)\big] \;-\; \beta\, \mathrm{KL}\!\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)\Big]
$$

- $r(x, y)$ — the reward for answer $y$ to prompt $x$.
- $\beta$ — the KL weight; larger $\beta$ forces the policy to hug the reference.
- $\mathrm{KL}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{y\sim\pi}[\log(\pi(y|x)/\pi_{\text{ref}}(y|x))]$ — the penalty for drifting.

Read it as: choose answers that score well, but pay whenever your distribution moves away from the reference. You do not need to unpack it further yet; sections 12–14 return to it.

Lesson 15.2 optimized this objective and paid a brutal price: a reward model, a value model, a frozen reference and a generation loop, all interacting through touchy hyperparameters. DPO's key observation is that the optimum of this objective has an **exact closed form**. Write the loss in terms of that form, and the reward model and the sampling both cancel out. What is left is a loss you compute directly from the stored preference pairs — hence *Direct* Preference Optimization.

This does **not** mean "DPO is PPO with a shorter loss". PPO is an algorithm that optimizes a reward using sampled answers and a clipped policy ratio. DPO derives a supervised, pairwise loss whose solution corresponds to the KL-regularized reward optimum (under assumptions). It avoids a separate reward model, but still needs a reference model and preference data.

An analogy for an engineer: PPO is like tuning a system by repeatedly deploying it, measuring a KPI with a separate monitoring service (the reward model), and adjusting. DPO notices that the optimal configuration has a known formula in terms of the KPI, flips the formula around, and fits the configuration to the raw A/B-test labels directly — no monitoring service, no deployments.

<div class="callout key"><p><strong>Yes: we take the preference file and train our language model directly.</strong> In PPO-based RLHF, the preference examples first train a separate reward model, which then grades newly generated answers. Standard DPO skips that step: the same labeled examples train the language model itself.</p></div>

### Who decided which answer is better?

Someone made that decision **before** the DPO step: a human annotator, an AI judge, an existing reward model, or a task-specific check (lesson 15.5 covers where preference data comes from). DPO receives the finished comparison; it never discovers the label itself. One record in a JSONL file:

```json
{"prompt": "What is 2 + 2?", "chosen": "The answer is 4.", "rejected": "The answer is 5."}
```

The field names tell the trainer which answer to prefer. There is no numerical reward in the record, and none is needed. One DPO training step on it:

1. **Read the labeled pair.** The prompt and both answers are already in the dataset.
2. **Compute our model's log-probabilities for both stored answers.** Feed each answer after the prompt and measure the probability of its tokens. Nothing is generated.
3. **Compute the frozen reference's log-probabilities for the same answers.** The reference is a baseline; it does not choose the winner.
4. **Compute the DPO loss and update our model.** The loss uses these four numbers and the known chosen/rejected order to push the chosen answer up relative to the rejected one, measured against the reference.

<div class="callout warn"><p><strong>What "no reward model" means.</strong> Standard DPO needs neither a reward scorer nor a value model <em>in its training loop</em>. If a reward model produced the dataset's labels, your data pipeline used one, but DPO learns equally well from human labels. The <em>implicit reward</em> introduced later is a formula computed from policy and reference log-probabilities. It is not a separate network and it does not supply the labels.</p></div>

## 3. Start with the operation you already know

Give a language model some tokens. At each position it returns logits over the vocabulary; softmax turns them into next-token probabilities. That is the **only** language-model operation DPO needs.

<div class="callout key"><p><strong>No extra preference output.</strong> The language model never returns "I prefer answer A with probability 80%". It returns next-token probabilities. The training code computes a comparison number from them later.</p></div>

Keep the record from section 2 in front of you: prompt "What is 2 + 2?", chosen "The answer is 4.", rejected "The answer is 5."

**Giving only the prompt.** Its last position gives probabilities for the *first* answer token ("The", "4", thousands of others). It does not give the probabilities of later answer tokens: those need the answer prefix as context.

**Giving the prompt plus the stored answer.** In training we feed prompt and answer in one forward pass. The causal mask stops each prediction from seeing its own target or anything after it. We read the output at the position just *before* each answer token — the same shifted input/target alignment as ordinary next-token training ([01.4](lessons/module-01/lesson-04.md), [14.2](lessons/module-14/lesson-02.md)).

For illustration, pretend the answer splits into six tokens: `The`, `answer`, `is`, `4`, `.`, and an end-of-answer token `<end>`. (Real tokenization and chat templates differ.) Let $x$ stand for all prompt tokens, including the template's assistant prefix.

| Context visible to this prediction | Stored target token | Probability we read |
|---|---|---|
| x | The | 0.60 |
| x + The | answer | 0.50 |
| x + The answer | is | 0.80 |
| x + The answer is | 4 | 0.40 |
| x + The answer is 4 | . | 0.90 |
| x + The answer is 4. | `<end>` | 0.95 |

Every row actually holds a probability for **every** vocabulary token. We select the entry for the target token that is already in the file. At row 4 we read the entry for "4", even if another token has a larger probability. No argmax, no sampling.

The table is six prediction **positions**, not six model calls. With teacher forcing, the causal transformer computes all positions in one pass. Supplying the answer does not leak a target into its own prediction; it supplies the correct prefix for the *later* predictions.

Repeat for the rejected answer. Now the fourth target is "5", and later contexts contain "5". Suppose the selected probabilities are `0.60, 0.50, 0.80, 0.10, 0.90, 0.95`. (Identical values at the other positions are just convenient numbers for this example, not a rule.)

## 4. Turn the token probabilities into one number per answer

We have six probabilities per answer. The probability of producing **this exact answer** is the chain rule: multiply the probability of each token given the prompt and the answer tokens before it.

$$
P_{\text{chosen}} = 0.60\times0.50\times0.80\times0.40\times0.90\times0.95 = 0.08208
$$

$$
P_{\text{rejected}} = 0.60\times0.50\times0.80\times0.10\times0.90\times0.95 = 0.02052
$$

0.08208 is the probability that ordinary sampling produces exactly this token sequence, ending included. It is **not** an 8.2% probability that the answer is correct. The two numbers need not sum to 1; the model can produce many other answers.

The general notation for exactly that multiplication:

$$
\pi_\theta(y\mid x) = \prod_{t=1}^{T} p_\theta(y_t \mid x, y_{<t})
$$

- $\pi_\theta(y\mid x)$ — our model's probability of the whole answer; $\theta$ says it depends on the weights.
- $T$ — number of answer tokens (here 6).
- $y_{<t}$ — the answer tokens before position $t$.
- each factor — one selected probability from the table in section 3.

### Use log-probabilities to avoid tiny products

Long answers multiply hundreds of factors below 1 and underflow. Training code adds natural logs instead, which is the same quantity in log form:

$$
s_\theta(y) = \log\pi_\theta(y\mid x) = \sum_{t=1}^{T}\log p_\theta(y_t\mid x, y_{<t})
$$

$$
s_w = \log(0.08208) \approx -2.500061, \qquad s_l = \log(0.02052) \approx -3.886355
$$

$w$ and $l$ name the answers the **file** labels winner and loser; they do not mean our model currently ranks them correctly. A less negative log-prob means a larger probability.

In code: log-softmax over the vocabulary logits, gather the log-prob of each stored target token, mask out prompt and padding targets, and **sum** over the answer positions, including the intended end-of-answer token. Training never exponentiates these sums; the products above only show their meaning. Section 15 implements this as `sequence_logprob`.

## 5. Do the same calculation with a frozen reference

DPO needs a second set of next-token probabilities, from a **reference** model. Usually we start from an SFT checkpoint, keep a frozen copy as the reference, and train another copy as the policy. At the start their weights are identical. Our running example describes a *later* moment, after the trainable model has already changed.

The reference receives the same prompt and the same stored answers; we select the same target tokens and sum their logs the same way. It does not grade correctness or pick the winner. It only tells us how probable each answer was under the fixed baseline.

Suppose the reference matches our model everywhere except position 4, where it gives 0.20 to "4" and 0.20 to "5". Both reference sequence probabilities are then $0.60\times0.50\times0.80\times0.20\times0.90\times0.95 = 0.04104$. We now have exactly four scalars:

| Model and stored answer | Sequence probability | Sequence log-prob | Updated by training? |
|---|---|---|---|
| Our model, chosen | 0.08208 | $s_w = -2.500061$ | Yes |
| Our model, rejected | 0.02052 | $s_l = -3.886355$ | Yes |
| Reference, chosen | 0.04104 | $s_w^{\text{ref}} = -3.193208$ | No |
| Reference, rejected | 0.04104 | $s_l^{\text{ref}} = -3.193208$ | No |

Four values do not mean four separate calls: chosen and rejected answers are batched together. Reference values can be **cached** once for a fixed dataset, as long as tokenization, masking and the reference checkpoint never change. The policy's values must be recomputed every step, because its weights change.

<div class="callout"><p><strong>Where we are:</strong> we have used only normal next-token predictions. No reward-model output, no value prediction, no generated answer. Everything that follows is arithmetic on these four log-probabilities.</p></div>

## 6. Compare how each answer changed relative to the reference

First subtract the reference log-prob from the policy log-prob for each answer:

$$
a_w = s_w - s_w^{\text{ref}} = \log\frac{0.08208}{0.04104} = \log 2 \approx 0.693147
$$

$$
a_l = s_l - s_l^{\text{ref}} = \log\frac{0.02052}{0.04104} = \log 0.5 \approx -0.693147
$$

The chosen answer became twice as probable as under the reference; the rejected answer became half as probable. A log difference expresses a multiplicative change: positive = more likely than before, zero = unchanged, negative = less likely.

Then subtract the rejected change from the chosen change:

$$
m = a_w - a_l = 0.693147 - (-0.693147) = 1.386294
$$

This **margin** measures whether our model shifted the relative odds toward the labeled winner. Positive favors the chosen answer compared with the reference; negative means it moved the wrong way.

It is a **relative** comparison. If both answers become twice as probable, both changes are $\log 2$ and the margin is 0. And if the reference gave the chosen answer much less probability to begin with, a positive margin does not mean the chosen answer now has the larger raw probability.

### Multiply by a scale called beta

DPO multiplies the margin by a positive setting $\beta$. With $\beta = 0.1$:

$$
z = \beta m = 0.1 \times 1.386294 = 0.138629
$$

$z$ is the **comparison logit**: one number computed by the training code. Do not confuse it with the vocabulary logits at each token position. For now $\beta$ just scales the comparison; section 14 shows its connection to the KL constraint.

Why this particular reference-adjusted comparison? Sections 12–13 derive it. First follow its numbers through the loss, so the derivation has a concrete computation to explain.

## 7. What "probability of preferring the chosen answer" means

$z$ can be any real number. To use an ordinary binary classification loss, squash it into $(0, 1)$ with the sigmoid:

$$
q = \sigma(z) = \frac{1}{1+e^{-z}} = \frac{1}{1+e^{-0.138629}} \approx 0.534602
$$

<div class="callout key"><p><strong>This is the step people usually miss:</strong> the language model gave us token probabilities. We combined them into four sequence log-probabilities, subtracted, multiplied by beta, and applied sigmoid. <strong>That computed number <span>$q$</span></strong> is what the DPO paper calls the predicted preference probability <span>$P_\theta(y_w \succ y_l \mid x)$</span>.</p></div>

### What event is this probability about?

Name the two answers A and B before reading the label. A binary prediction asks: "would the label pick A over B?" Target 1 if A was chosen, 0 if B was. A predicted 0.53 is the comparison model's prediction for that binary outcome.

In a DPO dataset we always put the stored chosen answer in slot A, so the target is **always 1**. We still compute $q$ from the current model and compare it with the known target. This is plain supervised learning: knowing the correct target does not make the current prediction equal it.

So $P_\theta(y_w\succ y_l\mid x)$ is just a name for $q$, the prediction induced by the current policy, the reference and the DPO formula. It does not mean we are unsure which JSON field is called "chosen".

| Quantity | What it means here | Where it comes from |
|---|---|---|
| 0.40 | Probability of token "4" after its prefix | Our model's vocabulary distribution |
| 0.08208 | Probability of the exact chosen answer sequence | Product of selected token probabilities |
| 0.534602 | Predicted probability of the chosen-over-rejected label under DPO's comparison model | Sigmoid of the scaled, reference-adjusted margin |
| 1 | The observed target: the first answer is the labeled winner | The dataset |

These answer different questions. $q$ is not the chance that sampling generates the chosen answer, nor a calibrated guarantee of correctness. The chosen sequence is 4× as probable as the rejected one under our model, yet $q$ is only about 0.535, because $q$ uses the reference comparison and $\beta = 0.1$.

If the policy exactly matches the reference, both changes are 0, so $q = 0.5$ for **every** pair — even when the reference gives the two answers very different probabilities. Here 0.5 means "no margin yet", not "the pretrained model knows nothing".

## 8. Turn the comparison into a training loss

The file says the chosen answer won, and $q$ predicts that event. We want a small loss when $q \approx 1$ and a large one when $q \approx 0$. The negative log does exactly that:

$$
\ell = -\log q = -\log\sigma(z) = \log(1+e^{-z})
$$

This is binary cross-entropy with target 1. The general form for target $b$ is $\ell(b,q) = -b\log q - (1-b)\log(1-q)$; with $b = 1$ the second term disappears. Note what we are **not** doing: we are not applying next-token cross-entropy to both answers as if both should be imitated. Their log-probs are combined into one pairwise loss.

| $q$ | Loss $-\log q$ | Meaning for this labeled pair |
|---|---|---|
| 0.10 | 2.302585 | Strongly disagrees with the observed ordering |
| 0.50 | 0.693147 | No margin relative to the reference |
| 0.534602 | 0.626233 | Our running example |
| 0.90 | 0.105361 | Fits the observed ordering confidently |

Average the loss over a batch, backpropagate, step the optimizer. The reference stays frozen. Next batch, the changed weights give changed token probabilities, margins and losses.

A stable implementation uses `-F.logsigmoid(z)` (or `F.softplus(-z)`), never `torch.log(torch.sigmoid(z))`: mathematically equal, but the explicit version underflows to `log(0)` for very negative $z$.

Substituting everything, the whole standard DPO loss for one pair is:

$$
\boxed{\;\ell_{\text{DPO}} = -\log\sigma\!\Big(\beta\big[(s_w - s_w^{\text{ref}}) - (s_l - s_l^{\text{ref}})\big]\Big)\;}
$$

Every symbol inside the box was computed from the model's ordinary next-token outputs. No hidden reward or value prediction.

<div class="callout key"><p>DPO needs <strong>no reward model</strong> and <strong>no sampling</strong>. To compute the loss you forward the policy and the frozen reference on the given chosen and rejected answers, gather their sequence log-probabilities, and plug into <span>$-\log\sigma$</span>. It is a supervised classification loss — as easy to run as SFT — that (section 13 shows) optimizes the same objective PPO does.</p></div>

## 9. Follow one complete step without skipping anything

The whole example as one uninterrupted calculation (natural logs; last-digit differences are rounding):

| Step | Operation | Result |
|---|---|---|
| 1 | Read the file's chosen and rejected answers | "The answer is 4." / "The answer is 5." |
| 2 | Select each answer token's probability under our model | chosen .60, .50, .80, .40, .90, .95 <br> rejected .60, .50, .80, .10, .90, .95 |
| 3 | Sum their logs | −2.500061 / −3.886355 |
| 4 | Repeat under the frozen reference | −3.193208 / −3.193208 |
| 5 | Subtract reference from policy for each answer | +0.693147 / −0.693147 |
| 6 | Chosen change minus rejected change | $m$ = 1.386294 |
| 7 | Multiply by $\beta$ = 0.1 | $z$ = 0.138629 |
| 8 | Apply sigmoid | $q$ = 0.534602 |
| 9 | Compare with target 1 using $-\log q$ | loss = 0.626233 |
| 10 | Backpropagate and take an optimizer step | only the policy's trainable parameters change |

### Change the token probabilities and watch the calculation

Three checkpoints of the same pair. Only the probabilities of "4" and "5" after "The answer is" change; every other token probability is fixed, the reference keeps 0.20 for each, $\beta = 0.1$, and the label never changes. These are illustrative checkpoints, not real optimizer steps.

| Checkpoint | P(4) / P(5) | Sequence prob. chosen / rejected | Log change chosen / rejected | Margin $m$ | $z$ | $q$ | Loss |
|---|---|---|---|---|---|---|---|
| At initialization (policy = reference) | 0.20 / 0.20 | 0.04104 / 0.04104 | 0 / 0 | 0 | 0 | 0.500000 | 0.693147 |
| Our worked example | 0.40 / 0.10 | 0.08208 / 0.02052 | +0.693147 / −0.693147 | 1.386294 | 0.138629 | 0.534602 | 0.626233 |
| Moved the wrong way | 0.10 / 0.40 | 0.02052 / 0.08208 | −0.693147 / +0.693147 | −1.386294 | −0.138629 | 0.465398 | 0.764862 |

At initialization the comparison starts at 0.5 even though the label is already known. In the worked example the chosen answer doubled and the rejected one halved, so $q$ moves toward the target 1. Moved the wrong way, the loss is larger because the comparison went against the label.

For a batch, each record contributes this same computation with its own numbers. A long answer simply contributes more token log-probs to its sum; it needs no different kind of preference model.

Notice where the supervision enters: the file decides which sum is called chosen and which rejected. Swap the labels and the sign of $z$ flips, and the loss trains the opposite preference, even though every forward pass is identical.

## 10. Backpropagate through the arithmetic

The target tokens and the label are fixed data. The probabilities assigned to those tokens are differentiable functions of the weights. Selecting the entry for a fixed target token keeps its gradient, exactly as in normal language-model training. There is **no sampling decision** to differentiate through — the reason DPO needs no policy-gradient trick.

Start from the last operation and go backward. Since $\ell = \log(1+e^{-z})$:

$$
\frac{d\ell}{dz} = \frac{-e^{-z}}{1+e^{-z}} = -\frac{1}{1+e^{z}} = \sigma(z) - 1 = q - 1
$$

In our example, $0.534602 - 1 = -0.465398$. Negative: increasing $z$ lowers the loss. That makes sense — the label says chosen should win, so a bigger margin fits it better.

Now expand $z = \beta\,(s_w - s_w^{\text{ref}} - s_l + s_l^{\text{ref}})$. Its derivatives with respect to the two policy log-probs are $+\beta$ and $-\beta$. By the chain rule:

$$
\frac{\partial\ell}{\partial s_w} = \beta(q-1) = 0.1\times(-0.465398) \approx -0.046540
$$

$$
\frac{\partial\ell}{\partial s_l} = -\beta(q-1) = \beta(1-q) \approx +0.046540
$$

If these were independent knobs, gradient descent would push the chosen log-prob up and the rejected one down. In a real network they share weights, so these are *contributions* to the update, not a guarantee that each sequence probability moves the right way every step.

Let $\theta$ be all trainable weights. The reference does not depend on $\theta$, so:

$$
\nabla_\theta\ell = \beta(q-1)\big[\nabla_\theta s_w - \nabla_\theta s_l\big]
$$

Using $1 - \sigma(z) = \sigma(-z)$:

$$
\boxed{\;\nabla_\theta \ell = -\beta\,\sigma(-z)\,\Big[\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\Big]\;}
$$

With plain gradient descent the update is $\theta \leftarrow \theta - \eta\nabla_\theta\ell$ ($\eta$ = learning rate); Adam rescales the same gradient with its running statistics. The optimizer changes $\theta$ only. It never writes new labels into the JSON and never updates a reward model.

### The push is weighted by "how wrong is the model on this pair?"

The multiplier $\beta\,\sigma(-z) = \beta(1-q)$ is large when the comparison fits the label poorly and small when it already fits confidently. With $\beta = 0.5$:

| raw margin $m$ → $z = \beta m$ | loss | push weight $\sigma(-z)$ |
|---|---|---|
| −4 → $z = -2$ (ranked the wrong way) | 2.126928 | 0.880797 |
| 0 → $z = 0$ (undecided) | 0.693147 | 0.500000 |
| +4 → $z = 2$ (already right) | 0.126928 | 0.119203 |

So DPO spends its effort on pairs it gets wrong and nearly ignores pairs it already gets right. This weighting stops it from crushing the rejected answer forever — compare the plain "unlikelihood" idea of always lowering bad answers, which the DPO paper shows degenerates. A small gradient on an easy pair says nothing about the model being good in general; it only describes this pair under this objective.

## 11. How that gradient reaches the token predictions

A sequence log-prob is a sum of selected token log-probs, and each summand has derivative 1 with respect to the sum. So every selected token in the chosen answer receives the scalar coefficient $-\beta(1-q)$, and every selected token in the rejected answer receives $+\beta(1-q)$.

Connect that to the vocabulary output. Call the logits at one position $u_1,\ldots,u_V$ ($V$ = vocabulary size). For target token $k$:

$$
p_k = \frac{e^{u_k}}{\sum_j e^{u_j}}, \qquad \log p_k = u_k - \log\sum_j e^{u_j}
$$

Differentiating with respect to any logit $u_j$:

$$
\frac{\partial\log p_k}{\partial u_j} = \mathbf{1}[j=k] - p_j
$$

The indicator is 1 for the target token and 0 otherwise — the same `onehot - p` you met in 01.4. Let $c = \beta(1-q) > 0$. The two branches of the DPO loss contribute:

$$
\frac{\partial\ell}{\partial u_j^{(w)}} = c\,\big(p_j^{(w)} - \mathbf{1}[j=k_w]\big), \qquad
\frac{\partial\ell}{\partial u_j^{(l)}} = c\,\big(\mathbf{1}[j=k_l] - p_j^{(l)}\big)
$$

The chosen branch is ordinary target-token cross-entropy scaled by $c$. The rejected branch is the same with the **opposite sign**. Backpropagation carries both through the transformer and adds them wherever they reach shared parameters.

**Tiny numeric check.** Take the "4" vs "5" position. Both branches have exactly the same preceding text ("x + The answer is"), so with deterministic forward passes their softmax distributions $p^{(w)} = p^{(l)} = p$ are identical there. Adding the two branches, the $p_j$ terms cancel: the "4" logit gets $-c$ (gradient descent raises it), the "5" logit gets $+c$ (lowered), and every other logit gets 0. In our example $c = 0.046540$.

The shared earlier tokens "The answer is" have identical contexts *and* identical targets, so their direct contributions cancel completely in the pair difference. Later tokens have different prefixes ("…4." vs "…5."), so they need not cancel. Dropout or inconsistent preprocessing between the two branches can break the exact cancellation.

<div class="callout warn"><p><strong>Masking still matters.</strong> Prompt tokens are context, not scored targets (their representations still receive gradients through their influence on answer predictions). Padding targets contribute nothing. Chosen and rejected answers each use their own length, with identical boundary conventions for policy and reference.</p></div>

## 12. Why the reference ratio? Derive it after seeing the process

We now know how to *run* DPO. The remaining question is why this particular comparison. To answer it, temporarily imagine we had a reward function $r$ that scores every possible answer. This is a mathematical starting point, not a component we execute during DPO training.

### A tiny world we will reuse

To check every step concretely, imagine a prompt with only **three** possible answers A, B, C (a real LLM has astronomically many, but the algebra is identical):

| answer | $\pi_{\text{ref}}(y)$ | reward $r(y)$ |
|---|---|---|
| A | 0.5 | 0 |
| B | 0.3 | 1 |
| C | 0.2 | 2 |

The reference likes A most, but the reward says C is best. Use $\beta = 1$ unless stated otherwise.

### The whole idea in plain English first

Hold on to these five sentences; every step below makes one of them precise.

1. RLHF wants a policy that earns high reward but does not drift far from the reference (step 1).
2. If someone handed you the reward function, you would not need RL at all: the best policy is "the reference, with every answer's probability multiplied by $\exp(\text{reward}/\beta)$, then renormalized" (step 2).
3. Read that backwards: if you know the best policy, you can recover the reward — $\beta$ times "how much more likely the policy makes this answer than the reference does", plus a per-prompt constant (step 3).
4. Preference data only ever compares two answers to the **same** prompt, so that per-prompt constant appears on both sides and cancels (step 4, section 13).
5. So instead of training a reward model and then running RL, train the policy *directly* so its "how much more likely than the reference" score ranks chosen above rejected. That is the classification loss you just ran.

### Step 1 — the KL-constrained objective for one prompt

Fix one prompt $x$ and drop "$\mid x$" to keep things short. Written as a sum over all complete answers:

$$
J(\pi) = \sum_y \pi(y)\,r(y) \;-\; \beta\sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}
$$

The first sum is the expected reward under $\pi$; the second is the KL divergence from the reference, times $\beta$, subtracted as a penalty. This is precisely the objective PPO chases in 15.2. We assume the reference is positive wherever the policy puts mass, and that all sums are finite.

### Step 2 — the closed-form optimal policy

For a *fixed* reward, $J$ has an exact maximizer. We find it by rewriting $J$ until it becomes "minimize a KL divergence", because we know where a KL is smallest: zero, when the two distributions are equal.

**(a) Put everything under one expectation.** The KL is itself an expectation over $y\sim\pi$, so both terms combine:

$$
J = \mathbb{E}_{y\sim\pi}\Big[r(y) - \beta\log\tfrac{\pi(y)}{\pi_{\text{ref}}(y)}\Big].
$$

**(b) Pull out $-\beta$.** Factor $-\beta$ out of the bracket and write $r/\beta$ as a log, $\tfrac{r}{\beta} = \log\exp(\tfrac{r}{\beta})$:

$$
J = -\beta\,\mathbb{E}_{y\sim\pi}\Big[\log\tfrac{\pi(y)}{\pi_{\text{ref}}(y)} - \log\exp\!\big(\tfrac{r(y)}{\beta}\big)\Big]
= -\beta\,\mathbb{E}_{y\sim\pi}\Big[\log\frac{\pi(y)}{\pi_{\text{ref}}(y)\exp(r(y)/\beta)}\Big].
$$

**(c) Turn the denominator into a real distribution.** $\pi_{\text{ref}}(y)\exp(r(y)/\beta)$ is a positive weight per answer, but the weights do not sum to 1. Divide by their total $Z = \sum_y \pi_{\text{ref}}(y)\exp(r(y)/\beta)$ to get a proper distribution $\pi^{*}$:

$$
\pi^{*}(y) = \frac{\pi_{\text{ref}}(y)\exp(r(y)/\beta)}{Z}
\quad\Longleftrightarrow\quad
\pi_{\text{ref}}(y)\exp(r(y)/\beta) = Z\,\pi^{*}(y).
$$

Substitute and split the log:

$$
J = -\beta\,\mathbb{E}_{y\sim\pi}\Big[\log\frac{\pi(y)}{\pi^{*}(y)} - \log Z\Big]
= -\beta\,\Big(\mathrm{KL}(\pi\,\|\,\pi^{*}) - \log Z\Big).
$$

**(d) Read off the answer.** $Z$ is built only from the reference and the reward, so it is a constant for the optimization. $-\beta$ is negative, so *maximizing* $J$ means *minimizing* $\mathrm{KL}(\pi\,\|\,\pi^{*})$, which is never negative and is zero only when the distributions match. So the optimum is $\pi = \pi^{*}$:

$$
\boxed{\;\pi^{*}(y\mid x) \;=\; \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\Big(\tfrac{1}{\beta}\, r(x,y)\Big)\;}
$$

- $\pi^{*}$ — the optimal policy for this reward and $\beta$.
- $Z(x) = \sum_{y}\pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$ — the **partition function** that renormalizes it. It depends on $x$ but **not** on $y$, and summing over every possible completion is intractable. This $Z(x)$ is what blocked earlier attempts to use the formula directly, and exactly what DPO makes cancel.

Intuitively: the best policy is the reference *reweighted* by $\exp(\text{reward}/\beta)$ — bump up answers the reward likes, in proportion to how much, tempered by $\beta$.

<details><summary>Same result by calculus (Lagrange multiplier), if you prefer</summary>

Treat each answer probability $\pi_y$ as a variable and add a Lagrange multiplier $\lambda$ for $\sum_y \pi_y = 1$:

$$
\mathcal J = \sum_y\pi_y r_y - \beta\sum_y\pi_y\log\frac{\pi_y}{\pi_{\text{ref},y}} + \lambda\Big(\sum_y\pi_y - 1\Big)
$$

Using $\frac{d}{dp}\big[p\log(p/a)\big] = \log(p/a) + 1$ for constant $a$, set the derivative for one answer to zero:

$$
r_y - \beta\Big(\log\frac{\pi_y}{\pi_{\text{ref},y}} + 1\Big) + \lambda = 0
\quad\Longrightarrow\quad
\log\frac{\pi_y}{\pi_{\text{ref},y}} = \frac{r_y}{\beta} + \frac{\lambda}{\beta} - 1
$$

Exponentiate. The $\lambda$ term is the same for every answer to this prompt, so normalization fixes it, and it becomes $1/Z$. Same boxed formula.

</details>

**Check it on the tiny world** ($\beta = 1$). Multiply each reference probability by $e^{r}$:

| answer | $\pi_{\text{ref}}$ | $e^{r}$ | weight $\pi_{\text{ref}}\,e^{r}$ | $\pi^{*} = \text{weight}/Z$ |
|---|---|---|---|---|
| A | 0.5 | 1.000000 | 0.500000 | 0.179000 |
| B | 0.3 | 2.718282 | 0.815485 | 0.291944 |
| C | 0.2 | 7.389056 | 1.477811 | 0.529056 |

$Z = 0.5 + 0.815485 + 1.477811 = 2.793296$. The optimal policy moved probability from A (reward 0) to C (reward 2), but did *not* put everything on C — the reference still holds it back. With $\beta = 0.5$ the reward counts twice as much ($e^{r/0.5} = e^{2r}$) and $\pi^{*}$ becomes $[0.036667,\ 0.162559,\ 0.800774]$: smaller $\beta$, more aggressive move toward the reward's favorite.

This is a distribution-level optimum. A real network, with finite data and finitely many steps, need not reach it exactly.

Why not train with this formula directly? With three answers, $Z$ is a sum of three numbers. For an LLM, $y$ ranges over *every possible completion* — more strings than atoms in the universe — so $Z$ cannot be computed. And we would still need $r$, i.e. a reward model. Steps 3 and 4 remove both problems.

### Step 3 — invert it: reward in terms of the policy

Take logs of the boxed equation and solve for $r$:

$$
\log \pi^{*}(y|x) = \log \pi_{\text{ref}}(y|x) + \tfrac{1}{\beta} r(x,y) - \log Z(x)
$$

$$
\boxed{\;r(x,y) \;=\; \beta\,\log\frac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \;+\; \beta\log Z(x)\;}
$$

(Move $\log\pi_{\text{ref}}$ and $\log Z$ to the left, then multiply both sides by $\beta$.)

This is the key move, and the source of the log-ratio we used in section 6. The reward — the thing we trained a whole separate network for in 15.1 — is *expressible* as $\beta$ times the log-ratio of the optimal policy to the reference, plus a term that depends only on the prompt. **The optimal policy already encodes the reward.**

Read $\log\frac{\pi^{*}(y)}{\pi_{\text{ref}}(y)}$ as "how much did the optimal policy *boost* this answer compared with where it started?" Boosted answers had high reward; suppressed answers had low reward. The reward is visible in *how the policy moved*, not in its raw probabilities.

**Check it on the tiny world** ($\beta = 1$, $\log Z = \log 2.793296 = 1.027222$):

| answer | $\pi^{*}/\pi_{\text{ref}}$ | $\log(\pi^{*}/\pi_{\text{ref}})$ | $+\log Z$ | recovered $r$ |
|---|---|---|---|---|
| A | 0.179000 / 0.5 | −1.027222 | +1.027222 | **0** |
| B | 0.291944 / 0.3 | −0.027222 | +1.027222 | **1** |
| C | 0.529056 / 0.2 | 0.972778 | +1.027222 | **2** |

We got back exactly the rewards we started with. Notice also: even *without* adding $\log Z$, the log-ratios $[-1.027,\ -0.027,\ 0.973]$ are the rewards shifted by one shared constant. They already rank A < B < C, and their **differences** equal the reward differences (B − A = 1, C − A = 2). That is step 4.

## 13. Why sigmoid? Substitute into Bradley-Terry and $Z(x)$ cancels

One modeling choice remains: how reward differences turn into pair labels. The **Bradley-Terry** model from [15.1](lessons/module-15/lesson-01.md) assumes two answers with scores $r_A$, $r_B$ produce a preference probability

$$
P(A\succ B\mid x) = \frac{e^{r_A}}{e^{r_A}+e^{r_B}}.
$$

Divide numerator and denominator by $e^{r_A}$:

$$
P(A\succ B\mid x) = \frac{1}{1+e^{r_B-r_A}} = \sigma(r_A - r_B).
$$

A two-option softmax, written as a sigmoid of the score difference. Equal scores give 0.5; a higher score for A gives more than 0.5. It is an assumption about how a scalar score represents comparisons, not a guarantee about every human judgment.

### Step 4 — plug in the reward from step 3

Put the boxed reward into the difference $r(x, y_w) - r(x, y_l)$:

$$
r(x,y_w) - r(x,y_l) = \Big(\beta\log\tfrac{\pi^{*}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \beta\log Z(x)\Big) - \Big(\beta\log\tfrac{\pi^{*}(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + \beta\log Z(x)\Big)
$$

The two $\beta\log Z(x)$ terms are **identical** (same prompt $x$) and **subtract to zero**. The intractable partition function is gone:

$$
r(x,y_w) - r(x,y_l) = \beta\log\frac{\pi^{*}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi^{*}(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

On the tiny world: $r(C) - r(A) = 0.972778 - (-1.027222) = 2$, the same number with or without $\log Z$. We never needed $Z$, and DPO never enumerates all answers.

### The last leap: every policy is a reward model in disguise

This is the step people find most slippery. Step 3 said: *for the optimal policy*, reward $= \beta\log(\pi^{*}/\pi_{\text{ref}})$ + a prompt-only constant. Turn it around and **define** a reward from *any* policy, dropping the constant (it cannot affect a comparison):

$$
\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
$$

This is the **implicit reward**. It adds no trainable head; our next-token probabilities already determine it. In our running example, $\hat r_\theta(y_w) = 0.1\times0.693147 = 0.069315$ and $\hat r_\theta(y_l) = -0.069315$.

In 15.1 we fitted a reward network to preference pairs with the Bradley-Terry loss. Here we do the very same fit, except the "reward network" is $\hat r_\theta$, i.e. the policy itself. Whichever $\theta$ fits the preferences best defines a reward $\hat r_\theta$ — and by step 3, $\pi_\theta$ is *already the optimal RLHF policy for that reward*. Fitting the reward and solving the RL problem happen in one go.

Substitute the implicit-reward difference into Bradley-Terry:

$$
q_\theta = \sigma\big(\hat r_\theta(x,y_w) - \hat r_\theta(x,y_l)\big) = \sigma(z)
$$

That is the same $q$ we computed numerically in section 7, now with its motivation. Maximize the likelihood of the observed labels — equivalently, minimize the negative log-likelihood averaged over the dataset (in a minibatch, the mean of the per-pair losses):

$$
\boxed{\;\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x,y_w,y_l)\sim\mathcal D}\!\left[\log\sigma\!\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)\right]\;}
$$

We have arrived at the objective we already ran in section 8, without adding any new model outputs.

### What each term means

- $\log \pi_\theta(y_w|x) - \log \pi_{\text{ref}}(y_w|x)$ — how much *more* likely the policy makes the chosen answer than the reference did. Times $\beta$, DPO's implicit reward for $y_w$.
- The full bracket is the **implicit reward margin** (chosen minus rejected). Minimizing $-\log\sigma(\cdot)$ pushes it up: raise $\pi_\theta(y_w)$ relative to the reference, lower $\pi_\theta(y_l)$ relative to the reference.
- $\beta$ — the RLHF KL weight. The reference sits inside the loss itself, so the KL constraint is *baked in*; there is no separate penalty term.

### Why compare against the reference at all?

Why not just push $\log\pi_\theta(y_w)$ above $\log\pi_\theta(y_l)$? Because raw probabilities mix *quality* with *length* and *style*. A 300-token answer almost always has a much lower total log-prob than a 20-token answer, even when it is better. Subtracting the reference's log-prob cancels those built-in biases: the policy is rewarded only for the *change* it made relative to where it started. If the reference already preferred the chosen answer by 5 nats, the policy gets no credit for that — only for any extra preference it adds. (The theory says the same thing: the reward lives in the log-ratio, step 3.)

## 14. Understand beta without confusing its two appearances

**In the training calculation**, $\beta$ scales $z$ before the sigmoid. For our fixed margin $m = 1.386294$, $\beta = 0.1$ gives $q \approx 0.535$, while $\beta = 1$ gives $q = \sigma(1.386294) = 0.8$. Two different comparison predictions from the same four log-probs.

**In the theoretical objective**, $\beta$ multiplies the KL penalty. For a fixed reward, larger $\beta$ keeps the optimal policy closer to the reference (on the tiny world, $\beta = 1$ gave C 0.53; $\beta = 0.5$ gave C 0.80). The two roles agree: the optimum depends on $r/\beta$, so solving backward for the reward produces $\beta$ times a log-ratio.

Do **not** conclude that a larger $\beta$ shrinks every gradient. At initialization $z = 0$ and the gradient coefficient is $\beta\sigma(0) = \beta/2$, which *grows* with $\beta$. Later the coefficient is $\beta\,\sigma(-\beta m)$, so both its scale and the sigmoid's saturation matter. Actual movement also depends on learning rate, data, model capacity and training length.

$\beta$ is not a measured KL divergence and not a hard cap on drift. Monitor held-out behavior (and the actual KL, if you log it) instead of assuming the coefficient alone guarantees a distance from the starting model. Typical values are 0.1–0.5; TRL's `DPOConfig` default is 0.1.

## 15. Implement it: `sequence_logprob` and `dpo_loss`

### `sequence_logprob`

The DPO loss needs $\log\pi(y|x) = \sum_t \log \pi(y_t\mid y_{<t}, x)$ — the **sum** of per-token log-probs over the answer (section 4). It is the same gather-the-true-token-log-prob operation as cross-entropy ([01.4](lessons/module-01/lesson-04.md)), but summed (not averaged) and over answer tokens only (prompt masked to `ignore_index`, exactly as in [14.2](lessons/module-14/lesson-02.md)).

Tensor shapes / dtype / device:

- `logits`: `(B, T, V)` float, on the model's device — the policy's logits, already shifted so position `t` scores `labels[:, t]`.
- `labels`: `(B, T)` int64, same device — the answer token at each position, with prompt/pad set to `-100`.
- output: `(B,)` float, same dtype/device as `logits` — one summed log-prob per sequence.

```python
def sequence_logprob(logits, labels, ignore_index=-100):
    logp = F.log_softmax(logits, dim=-1)                        # (B, T, V)
    mask = labels != ignore_index                               # (B, T) bool
    safe = labels.clamp(min=0)                                  # avoid gather on -100
    tok_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)  # (B, T) true-token log-prob
    tok_logp = tok_logp * mask                                  # zero out ignored positions
    return tok_logp.sum(dim=-1)                                 # (B,) sum over the sequence
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

Cost: the log-softmax over `(B, T, V)` is the memory peak — for $B = 8$, $T = 1024$, $V = 50{,}257$ in float32 that is about 1.6 GB, the same as SFT's cross-entropy. Production code computes it in chunks or fuses it with the gather for that reason.

### `dpo_loss`

`llmre.preference.dpo` implements the boxed loss directly. The four inputs are `(B,)` float tensors on one device; the reference ones carry no gradient.

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    pi_logratios  = policy_chosen_logps - policy_rejected_logps   # (B,)
    ref_logratios = ref_chosen_logps   - ref_rejected_logps       # (B,) no grad
    logits = pi_logratios - ref_logratios                        # (B,) margin m
    return -F.logsigmoid(beta * logits).mean()                   # scalar
```

Note the regrouping: the code computes $(s_w - s_l) - (s_w^{\text{ref}} - s_l^{\text{ref}})$, the paper writes $(s_w - s_w^{\text{ref}}) - (s_l - s_l^{\text{ref}})$. Same four terms, same margin $m$. (The real function also takes `reduction="none"` to return per-pair losses, and `dpo_reward_margin` returns the implicit rewards $\hat r_\theta$ for chosen and rejected.)

A full DPO step wires `sequence_logprob` to the policy and a frozen reference:

```python
import torch
from llmre.preference.dpo import dpo_loss, sequence_logprob

# policy is trainable; ref is frozen (eval(), requires_grad off).
def dpo_step(policy, ref, chosen_ids, chosen_labels, rej_ids, rej_labels, beta=0.1):
    pc = sequence_logprob(policy(chosen_ids), chosen_labels)      # (B,) grad
    pr = sequence_logprob(policy(rej_ids),    rej_labels)         # (B,) grad
    with torch.no_grad():
        rc = sequence_logprob(ref(chosen_ids), chosen_labels)    # (B,) frozen
        rr = sequence_logprob(ref(rej_ids),    rej_labels)       # (B,) frozen
    return dpo_loss(pc, pr, rc, rr, beta=beta)
```

### Under the hood / cost

- **Two models, two forwards each.** The policy forwards chosen and rejected, the reference forwards chosen and rejected — four forwards per pair, no generation. The reference log-probs can be **precomputed once** and cached, dropping the steady-state cost to two forwards per pair (one backward). Contrast PPO's four live models plus a sampling loop.
- **Memory.** The frozen reference must be resident (or its log-probs cached). No value head, no reward model, no replay buffer.
- **Stability.** It is a smooth supervised loss, so it trains like SFT — far fewer knobs than PPO ($\beta$ is the main one).

<div class="callout warn"><p>The reference must be the model you started DPO from (typically the SFT model), and it must be <strong>frozen</strong> — <code>eval()</code>, <code>requires_grad=False</code>, and wrapped in <code>torch.no_grad()</code> when you compute its log-probs. If the reference accidentally trains (or is the same object as the policy), the log-ratio collapses to zero and DPO has no target to push against.</p></div>

## 16. Hands-on: verify the numbers and the code

About 10 minutes, CPU only. You need the course repo installed (`cd code && pip install -e .`).

**1. Recompute the running example.** Feed the four sequence log-probs from section 5 into the real `dpo_loss`:

```bash
py -c "import math,torch; from llmre.preference.dpo import dpo_loss; \
s=lambda *p: torch.tensor([sum(map(math.log,p))]); \
pc=s(.6,.5,.8,.4,.9,.95); pr=s(.6,.5,.8,.1,.9,.95); \
rc=s(.6,.5,.8,.2,.9,.95); rr=rc.clone(); \
print(round(dpo_loss(pc,pr,rc,rr,beta=0.1).item(),6))"
# 0.626233
```

You know it worked when it prints `0.626233`, the loss from section 9. Change `beta=0.1` to `beta=1.0` and predict the output before you run it (hint: $q = 0.8$).

**2. Check the hand example from the original paper's notation.** A second pair with $\log\pi_\theta(y_w) = -2.0$, $\log\pi_\theta(y_l) = -3.0$, $\log\pi_{\text{ref}}(y_w) = \log\pi_{\text{ref}}(y_l) = -2.5$, $\beta = 0.5$. By hand: margin $= (-2.0 - (-3.0)) - (-2.5 - (-2.5)) = 1.0$, $\beta m = 0.5$, $\sigma(0.5) = 0.622459$, loss $= -\log 0.622459 = 0.474077$. Verify:

```bash
py -c "import torch,torch.nn.functional as F; \
pcw,pcl,rcw,rcl,beta=-2.,-3.,-2.5,-2.5,0.5; \
m=(pcw-pcl)-(rcw-rcl); \
print(round(-F.logsigmoid(torch.tensor(beta*m)).item(),6))"
# 0.474077
```

The policy prefers chosen over rejected *more than the reference does* (1.0 vs 0.0), so the loss is below $\log 2 = 0.693$.

**3. Run the unit tests.**

```bash
cd code && pytest tests/test_preference.py -k "dpo or sequence_logprob" -v
```

Three tests should pass in a few seconds: `test_sequence_logprob_sums_token_logps_and_skips_ignore_index`, `test_dpo_loss_matches_hand_computed_value`, and `test_dpo_gradient_increases_the_chosen_minus_rejected_margin`. The last one starts the policy log-ratios at zero and takes a few SGD steps; the chosen-minus-rejected margin increases, and the implicit rewards from `dpo_reward_margin` end with chosen above rejected — the loss does exactly what the derivation promised.

## 17. Connect this to SFT and PPO

| Question | SFT | PPO-based RLHF | Standard offline DPO |
|---|---|---|---|
| What answers are used? | Stored target answers | Answers sampled from the policy during rollouts | Stored chosen and rejected answers |
| What supplies the learning signal? | Target-token labels | Reward scores and advantage estimates | The pair's existing preference label |
| What is the loss built from? | Target-token log-probs | Policy ratios and advantages, plus auxiliary terms | Both answers' policy/reference log-probs |
| Separate reward model? | No | Usually trained first, frozen during PPO | No (one may have labeled the data beforehand) |
| Value model / critic? | No | Usually learned alongside the policy | No |

In SFT, a chosen answer alone gives the usual negative log-likelihood loss. DPO adds the rejected answer and the reference comparison. Training on just the chosen half of a preference file with cross-entropy is **SFT on chosen answers, not DPO**.

The DPO reference is also not PPO's "old policy". PPO's old policy is the behavior policy that generated a rollout and supplies the denominator of its importance ratio; it changes every iteration. The DPO reference is one fixed baseline for the whole run.

### Keep the labels separate from the model's predictions

If a record labels a worse answer as chosen, DPO trains toward that label. It does not verify the facts. A winner can also be merely "less bad" than its partner. Inspect data quality and evaluate generated answers; a lower preference loss alone does not prove better accuracy.

### Remember what a relative objective can and cannot say

A pair's margin can improve even if **both** answer probabilities fall, as long as the rejected one falls more. DPO does not guarantee an absolute increase in every chosen answer's probability; shared weights and other examples affect each update. And offline training cannot ask for fresh judgments on new behaviors the updated model develops. Iterative/online variants generate and label new pairs between rounds, but that is an extra data-collection loop on top of the same per-pair calculation.

## Common mistakes

- **Averaging instead of summing token log-probs.** `sequence_logprob` must **sum** over the answer (it is the log of a product). Averaging by length changes the objective, rescales the implicit reward, and interacts badly with $\beta$. Length patterns can still leak into what the model learns, even with a reference.
- **Letting the reference train / dropping `torch.no_grad()`.** The reference is a fixed anchor; if it moves, the objective degenerates.
- **Not masking the prompt, or mismatched scoring.** Only answer tokens count toward $\log\pi(y|x)$. Use the same chat template and target boundaries for policy and reference, include the intended end token, and don't truncate away the part where the two answers differ.
- **Sign or ratio-order slips.** It is $(\text{policy chosen} - \text{policy rejected}) - (\text{ref chosen} - \text{ref rejected})$. Swapping chosen/rejected trains the model to prefer the *worse* answer.
- **`torch.log(torch.sigmoid(z))` instead of `-F.logsigmoid(z)`.** Underflows to `-inf` for very negative $z$.
- **Reading $q$ as a generation probability or a correctness score.** It is the comparison model's prediction for the pair label, nothing more (section 7).

## Debugging exercise

Someone reports their DPO run "learns nothing" — the loss sits almost exactly at `0.6931` and the chosen/rejected implicit rewards stay equal. They insist the policy is training and gradients flow. What is the most likely bug?

<details><summary>Answer</summary>

`0.6931 = log 2 = -log σ(0)`: the implicit margin is stuck at 0. The overwhelmingly common cause is that the **reference is the same object as the policy** (or the reference is updated alongside it). Then $\log\pi_\theta - \log\pi_{\text{ref}} = 0$ for both chosen and rejected at every step, so the margin is identically zero. Fix: make the reference a separate, frozen copy of the starting model (`eval()`, `requires_grad=False`, `torch.no_grad()` for its log-probs), or precompute and cache its log-probs before training. (Contrast the same `0.6931` symptom in 15.1, where the cause was identical chosen/rejected inputs to the reward model.)

</details>

## Exercise

DPO's implicit reward for an answer is $\hat r(x,y) = \beta\log(\pi_\theta(y|x)/\pi_{\text{ref}}(y|x))$. Implement `reward_accuracy(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)` returning the fraction of pairs the *implicit* reward ranks correctly ($\hat r_w > \hat r_l$). On the running example it should return 1.0; on the "moved the wrong way" checkpoint, 0.0.

<details><summary>Hint</summary>

Like the reward-model accuracy in 15.1, you only need the sign of a margin.

</details>

<details><summary>Stronger hint</summary>

$\beta > 0$ does not change a sign, so you can compare $(s_w - s_w^{\text{ref}})$ with $(s_l - s_l^{\text{ref}})$ directly.

</details>

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

<div class="callout paper"><p><strong>Read:</strong> <a href="https://arxiv.org/abs/2305.18290">Rafailov et al. 2023, <em>Direct Preference Optimization: Your Language Model is Secretly a Reward Model</em></a> (also in the <a href="#/papers/index">paper index</a>). Read Sections 3–4 (the optimal policy, the reward reparameterization, the loss) and the gradient interpretation in 4 closely, plus Appendix A.1 for the derivation of the optimal policy. Skim the theoretical equivalence proofs (Section 5, Appendix A.5–A.6) on a first pass. This paper is the reason most open-weight models today are aligned with DPO rather than PPO. The token tables and numbers in this lesson are illustrative teaching examples, not from the paper.</p></div>

## Check yourself

<details><summary>What exactly does the language model return in a DPO step?</summary>

Vocabulary logits at each position, turned into next-token probabilities or log-probs. We select the entries for the stored answer tokens. It never returns a separate preference judgment.

</details>

<details><summary>Why feed the stored answer to the model if we want to train it?</summary>

The answer supplies the correct prefixes for teacher forcing. Each target's prediction comes from the position before it, under a causal mask, so no target is leaked to its own prediction. We measure how much probability the current weights give those target tokens.

</details>

<details><summary>Where does the "probability of preferring the chosen answer" come from?</summary>

From arithmetic in the training code: sum selected token log-probs for each answer under policy and reference; subtract reference from policy; take chosen minus rejected; multiply by $\beta$; apply sigmoid. That scalar $q$ predicts the binary pair label under DPO's comparison model.

</details>

<details><summary>We already know which answer won. Why predict it?</summary>

Because it is a supervised target. We know the label, but the current model's probabilities may not fit it. The gap between prediction and label is the training signal, just as with known next-token targets in ordinary training.

</details>

<details><summary>Does $q = 0.9$ mean the model will generate the chosen answer 90% of the time?</summary>

No. The generation probability is the product of that answer's token probabilities (section 4). $q$ is a different number, computed for one comparison against the reference.

</details>

<details><summary>Where does the intractable partition function $Z(x)$ go?</summary>

It appears in the reward reparameterization $r(x,y) = \beta\log(\pi^*/\pi_{\text{ref}}) + \beta\log Z(x)$. But DPO only uses the *difference* of rewards for two answers to the **same prompt** $x$, and $\beta\log Z(x)$ is identical for both, so it cancels. That cancellation is the whole trick.

</details>

<details><summary>What two networks does DPO evaluate, and what does it <em>not</em> need?</summary>

The trainable **policy** $\pi_\theta$ and a **frozen reference** $\pi_{\text{ref}}$, each forwarded on the chosen and rejected answers. It needs **no reward model** and **no sampling/generation** — that is what makes it as cheap and stable as SFT.

</details>

<details><summary>Why sum (not average) the per-token log-probs in <code>sequence_logprob</code>?</summary>

$\log\pi(y|x) = \log\prod_t \pi(y_t|\cdot) = \sum_t \log\pi(y_t|\cdot)$. Averaging computes the per-token mean, a different quantity that rescales the implicit reward and breaks its calibration against $\beta$.

</details>

<details><summary>Recompute the loss if the policy exactly matches the reference. What is it?</summary>

Both log changes are 0, so the margin is 0, $\sigma(0) = 0.5$, and the loss is $-\log 0.5 = \log 2 = 0.693147$ — for every pair, whatever the reference probabilities. No preference signal yet.

</details>

<details><summary>What role does $\beta$ play, and how does it relate to PPO?</summary>

It is the same KL weight as in the RLHF/PPO objective: large $\beta$ keeps $\pi_\theta$ close to the reference, small $\beta$ lets it move far. Because the reference is inside the DPO loss, the KL constraint is built in — no separate KL-penalty term. In the training arithmetic it scales the margin before the sigmoid, and it does not simply shrink gradients (at the start the coefficient is $\beta/2$).

</details>

<details><summary>What changes after backpropagation?</summary>

Only the trainable model's weights (or adapters). The labeled answers and the reference weights stay fixed. New weights give new token probabilities on the next step.

</details>

<details><summary>Say the whole training step in one breath.</summary>

"I load a prompt with two already-labeled answers. My model and a frozen reference give me next-token probabilities for those stored answers. I combine them into a reference-adjusted comparison, compare it with the known label using $-\log\sigma$, and backpropagate the loss into my language model."

</details>

## Next

We have three ways to shape behavior: SFT (imitate), RM+PPO (optimize a learned reward with RL), and DPO (optimize preferences directly). Next we compare all three head-to-head — what each optimizes, what infrastructure each demands, how each fails — and look at replacing human labels with AI feedback (Constitutional AI / RLAIF).

Continue to [15.4 · SFT vs RM+PPO vs DPO](lessons/module-15/lesson-04.md).
