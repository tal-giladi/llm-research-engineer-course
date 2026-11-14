# 15.2 · Using the reward model: policy gradient & PPO

<div class="prereq">
<p><strong>Prerequisites:</strong> the SFT model and its response-only loss from <a href="#/lessons/module-14/lesson-02">14.2</a>; the reward model you trained and saved in <a href="#/lessons/module-15/lesson-01">15.1</a>; sampling with <code>generate</code> from <a href="#/lessons/module-06/lesson-03">06.3</a>; the softmax / cross-entropy gradient <code>p - onehot</code> from <a href="#/lessons/module-01/lesson-04">01.4</a>.</p>
<p><strong>You will learn:</strong> how to change the SFT model's weights so that its answers get higher grades from the reward model. Step by step: why we cannot simply backpropagate the grade; the fix, which is "SFT on the model's own answers, weighted by how good they were"; why the weight is the grade minus what we expected (the <strong>advantage</strong>), and the <strong>value model</strong> that supplies the expectation; the <strong>KL penalty</strong> that keeps the model close to the SFT model; and the <strong>PPO clip</strong> that lets us reuse each batch of answers safely. Then you run all of it on your own SFT model and reward model.</p>
<p><strong>Why this matters for ML:</strong> this loop is the "RL" in RLHF. InstructGPT and the first ChatGPT were trained this way. DPO (15.3) and GRPO (16.1) are both explained as "PPO, with one part removed", so every part you meet here comes back.</p>
</div>

<div class="hw">
<p><strong>Hardware track.</strong> <strong>Toy lab:</strong> any CPU, about one second. <strong>Wiring tests:</strong> CPU, about 25 seconds. <strong>Your SFT + reward model:</strong> CPU works for the small course model (minutes per iteration, depending on its size and answer length); a GPU makes generation much faster. For a real 1B model see the memory estimate in section 8: about 40 GB before activations in float32, so one 80 GB GPU or the sharding from Modules 7–9. Checked on 2026-11-14 with Python 3.14 and PyTorch 2.14.0+cpu.</p>
</div>

## 1. Where we are: two models and one goal

At the end of 15.1 you had two separate models:

| Model | What it does | Where it came from | Changes in this lesson? |
|---|---|---|---|
| **SFT model** | Writes answers, one token at a time, with its `lm_head` | Module 14 | **Yes.** This is the model we improve |
| **Reward model** | Reads a prompt + answer and returns one number, the grade | 15.1 (`best.pt`) | **No.** It stays frozen |

The goal of this lesson in one sentence: **change the SFT model's weights so that the answers it writes get higher grades from the reward model.**

The plan is a loop you can already picture:

```text
           +-------------------------------------------------------+
           |                                                       |
           v                                                       |
  prompt --> SFT model writes an answer --> reward model grades it --> adjust the SFT model's weights
             (sampling, as in 06.3)         (one number, as in 15.1)   (this lesson)
```

Every part except the last arrow is something you have already built. This lesson is about the last arrow: **given an answer and its grade, how exactly do we change the weights?**

### Three new words, and nothing else new

Papers about this loop use the vocabulary of reinforcement learning (RL). It only renames things you know:

| RL word | What it means here |
|---|---|
| **policy** | The model that is being trained to write answers. It starts as an exact copy of the SFT model. "Policy" just means "the thing that chooses what to do"; here it chooses the next token |
| **action** | One token the policy picks |
| **reward** | The grade. In this lesson the grade comes from your reward model, and later we subtract a penalty from it |

The policy's probability for a token is written $\pi_\theta(\text{token} \mid \text{text so far})$. Read $\pi$ as "the policy's probability" (it has nothing to do with 3.14). $\theta$ is all of its weights. This number is exactly what you already compute: `softmax(logits)` at the last position, the same distribution `generate` samples from.

## 2. Why we cannot just backpropagate the grade

In 15.1, training was easy because every step from the text to the loss was a calculation: the text was fixed data, the reward model computed a score, the loss was computed from scores, and `loss.backward()` could follow every step back to the weights.

Now the text is **not** fixed data. The policy *chooses* it. Look at what `generate` does for each token:

```python
probs = F.softmax(logits, dim=-1)                   # e.g. [0.62, 0.30, 0.08]: a calculation
idx_next = torch.multinomial(probs, num_samples=1)  # picks token 0: a random choice
```

The first line is a calculation with a gradient. The second line is a dice roll. Suppose a weight changes slightly, so the probabilities become `[0.63, 0.29, 0.08]`. The roll still picks token 0, so the text, and therefore the grade, stays exactly the same. With a larger change the roll may suddenly pick token 1 and the whole answer changes. The grade does not change *smoothly* with the weights, so there is no slope to follow. (And `generate` runs under `torch.no_grad()` anyway.)

So we need another way to turn "this answer got 2.1" into "change the weights in this direction".

## 3. The idea: SFT on the model's own answers, weighted by the grade

### 3.1 Start from the SFT loss you know

In Module 14 the loss on one training answer was: for every answer token, take the model's log-probability of that token, add them up, and negate:

$$
\text{SFT loss} = -\sum_{t=1}^{n} \log \pi_\theta(y_t \mid x, y_{<t})
$$

| Symbol | Meaning |
|---|---|
| $x$ | the prompt (all tokens before the answer, including chat markers) |
| $y_1, \dots, y_n$ | the answer's tokens; $n$ is the answer length |
| $y_{<t}$ | the answer tokens before position $t$ |
| $\pi_\theta(y_t \mid x, y_{<t})$ | the probability the model gives to token $y_t$ after reading the prompt and the earlier answer tokens |

(In 14.2 we divided by the number of tokens to get a mean. That only changes the step size, so we leave it out here.)

One gradient step on this loss makes **this exact answer more likely**. That is all SFT ever does: "write more like this".

**The sum of log-probabilities is the log-probability of the whole answer.** The model writes the answer token by token, so the probability of the whole answer is the product of the token probabilities, and the log of a product is the sum of the logs. Example with a three-token answer whose tokens got probabilities $0.5$, $0.4$ and $0.9$:

$$
\log 0.5 + \log 0.4 + \log 0.9 = -0.6931 - 0.9163 - 0.1054 = -1.7148
$$

$$
e^{-1.7148} = 0.18 = 0.5 \times 0.4 \times 0.9
$$

So we write $\log \pi_\theta(y \mid x)$ for "the log-probability of the whole answer $y$", and the SFT loss is $-\log \pi_\theta(y \mid x)$.

### 3.2 Now the trick

The policy writes an answer, and the reward model grades it. Then:

- If the answer was **good**, do an SFT step on it: make it more likely.
- If the answer was **bad**, do the *opposite* of an SFT step: make it less likely.
- Push harder the more good (or bad) it was.

All three rules fit into one loss, with a **weight** that is positive for good answers and negative for bad ones:

$$
\text{loss} = -\,\text{weight} \times \log \pi_\theta(y \mid x)
$$

With weight $+1$ this is exactly the SFT loss. With weight $-1$ the sign flips and gradient descent makes the answer *less* likely. With weight $+3$ it is an SFT step three times as large.

Notice what this avoids. The grade is just a number we multiply by. We never need its gradient. The gradient comes only from $\log \pi_\theta(y \mid x)$, which is an ordinary calculation, exactly as in SFT. The dice roll happened *before* the loss: the sampled answer is treated as fixed text, like an SFT training example.

This method is called the **policy gradient**, and this simplest version is called **REINFORCE** (Williams, 1992). It is publicly documented and is the base of everything else in this module.

### 3.3 Worked example by hand: a policy with three possible answers

Make the policy as small as possible so every number is visible. The prompt is "Ask Maya to review my PR". The policy can write only three answers, chosen from three logits, which are its only weights:

| Answer | Grade from the reward model | Logit $\theta$ | Probability |
|---|---:|---:|---:|
| A: polite request | 2.0 | 0 | 0.3333 |
| B: neutral request | 1.0 | 0 | 0.3333 |
| C: pushy request | 0.0 | 0 | 0.3333 |

From 01.4 you know the gradient of $-\log \text{softmax}(\theta)_a$ with respect to the logits: it is $p - \text{onehot}(a)$. Our loss is that times the weight, so

$$
\frac{\partial\, \text{loss}}{\partial \theta} = \text{weight} \times \big(p - \text{onehot}(a)\big)
$$

Suppose the policy writes **A** and we use weight $+1$:

$$
\text{gradient} = 1 \times \big([0.3333, 0.3333, 0.3333] - [1, 0, 0]\big) = [-0.6667,\ 0.3333,\ 0.3333]
$$

Take one gradient-descent step with learning rate 1 (new = old − gradient):

$$
\theta_{\text{new}} = [0.6667,\ -0.3333,\ -0.3333], \qquad \text{softmax}(\theta_{\text{new}}) = [0.5761,\ 0.2119,\ 0.2119]
$$

The probability of the polite answer rose from 0.333 to 0.576. Nothing new happened: this is one SFT step on answer A.

<details><summary>Why does this push toward higher grades on average? (optional, short)</summary>

The quantity we want to increase is the average grade of the answers the policy writes, $J = \sum_y \pi_\theta(y)\,R(y)$, where $R(y)$ is the grade of answer $y$. Its gradient is $\sum_y R(y)\,\nabla \pi_\theta(y)$. Use the identity $\nabla \pi = \pi \,\nabla \log \pi$ (the derivative of $\log \pi$ is $\nabla\pi/\pi$):

$$
\nabla J = \sum_y \pi_\theta(y)\; R(y)\; \nabla \log \pi_\theta(y)
$$

A sum of the form $\sum_y \pi_\theta(y) \cdot (\dots)$ is an average over answers the policy writes. So: write an answer, compute $R(y)\nabla\log\pi_\theta(y)$, and on average you get exactly $\nabla J$. That is our loss's gradient with the sign flipped (we *minimize* the loss, which *increases* $J$). In RL papers it is written

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\big[\, R(y)\, \nabla_\theta \log \pi_\theta(y \mid x) \,\big]
$$

where $\mathbb{E}_{y \sim \pi_\theta}$ means "average over answers $y$ written by the policy" and $\nabla_\theta$ means "gradient with respect to the weights". One answer gives a noisy estimate; averaging over a batch reduces the noise.

</details>

## 4. What should the weight be? The grade minus what we expected

### 4.1 Using the raw grade does not work

The obvious choice is weight = grade. Try it on the example. The policy writes **B**, the neutral answer, grade $1.0$:

$$
\text{gradient} = 1.0 \times \big([0.3333, 0.3333, 0.3333] - [0, 1, 0]\big) = [0.3333,\ -0.6667,\ 0.3333]
$$

$$
\theta_{\text{new}} = [-0.3333,\ 0.6667,\ -0.3333], \qquad \text{probabilities} = [0.2119,\ 0.5761,\ 0.2119]
$$

The **mediocre** answer B was pushed up, and the best answer A was pushed *down*, from 0.333 to 0.212. That is backwards.

There is a deeper reason, and you met it in 15.1: **the reward model's numbers only mean something as differences.** Adding 100 to every grade leaves the reward model's loss unchanged, so a reward model could just as well have given A, B and C the grades 102, 101 and 100. With weight = grade, every answer would then get a large push up, whichever answer was written. The size of the push would depend on an arbitrary constant.

### 4.2 The fix: compare with what we expected

Ask a different question: **was this answer better or worse than what the policy usually writes for this prompt?**

What the policy usually gets is the average grade over its answers, weighted by how likely each one is:

$$
\text{expected grade} = 0.3333 \times 2.0 + 0.3333 \times 1.0 + 0.3333 \times 0.0 = 1.0
$$

Now use weight = grade − expected grade:

| Policy wrote | Grade | Expected | Weight | After one step (learning rate 1) |
|---|---:|---:|---:|---|
| A: polite | 2.0 | 1.0 | **+1.0** | A rises: $[0.5761, 0.2119, 0.2119]$ |
| B: neutral | 1.0 | 1.0 | **0.0** | nothing changes |
| C: pushy | 0.0 | 1.0 | **−1.0** | C falls: $[0.4223, 0.4223, 0.1554]$ |

Every row now does the right thing. And adding 100 to all grades adds 100 to the expected grade too, so the weights do not change at all.

This weight has a name: the **advantage**, written $A$. The expected grade is called the **baseline**.

$$
A = R - b
$$

| Symbol | Meaning |
|---|---|
| $R$ | the reward this answer got (so far: its grade; in section 5 we subtract a penalty) |
| $b$ | the baseline: the reward we expected for this prompt before seeing the answer |
| $A$ | the advantage: how much better than expected the answer was. Positive: push it up. Negative: push it down |

Subtracting the baseline does not change the *average* direction of the gradient. The reason: averaged over the policy's own answers, $\nabla \log \pi_\theta(y)$ is zero (the probabilities always add up to 1, so if one rises, others must fall), so subtracting the same number $b$ from every answer's weight changes nothing on average. What it removes is the noise: the large pushes in arbitrary directions that you saw with answer B. Lab 1 in section 8 shows how bad that noise can get.

### 4.3 Where the expected grade comes from: the value model

In the toy example we could compute the expected grade exactly, because we knew all three answers. A real policy can write billions of different answers, so we need to *estimate* it. And it depends on the prompt:

| Prompt | Typical grade of the policy's answers |
|---|---|
| "Write a polite Slack message to Maya" (easy) | around +1.9 |
| "Explain caching to a nontechnical coworker, with one limitation" (harder) | around −1.2 |

An answer to the hard prompt with grade −0.9 is a **good** answer (better than the usual −1.2), even though its grade is negative. A single global average could not see that.

PPO estimates the expected grade with a second network, the **value model**. Its job: read a prompt and predict the reward the policy's answer will get. That is "text in, one number out", exactly the shape of your reward model. So we make the value model **a trainable copy of the reward model**, and train it with the plainest regression loss there is:

$$
\text{value loss} = \big(V - R\big)^2
$$

where $V$ is the value model's prediction and $R$ is the reward the answer actually got. Then the advantage is $A = R - V$:

| Prompt | Value model predicts $V$ | Answer's reward $R$ | Advantage $A = R - V$ |
|---|---:|---:|---:|
| Easy | 1.9 | 2.1 | **+0.2**: a little better than usual |
| Hard | −1.2 | −0.9 | **+0.3**: better than usual |

Starting the value model from the reward model, and the policy from the SFT model, is what InstructGPT did (publicly documented in its appendix C.4).

Two practical details, both in our code:

- **One prediction per position.** The value model reads the prompt plus the answer so far and predicts the final reward at every answer position, so each token gets its own advantage $A_t = R - V_t$. Halfway through a polite message the value model may already predict a high reward, so the later tokens get less credit for it. When we say "the value model predicts", it does this for every prefix in one forward pass, like the reward model computing a score at every position before 15.1 picked the last one.
- **Rescaling.** Before using them, the advantages of a batch are shifted and scaled to mean 0 and standard deviation 1 ("whitening"). This keeps the step size the same whether the reward model's grades are around 1 or around 50.

<div class="callout key"><p>GRPO (16.1) removes the value model: it writes several answers per prompt and uses their average grade as the baseline. That is the same idea as our exact "expected grade" in the toy example, estimated by sampling instead of by a second network.</p></div>

## 5. Staying close to the SFT model: the KL penalty

### 5.1 The problem: the reward model can be fooled

Your reward model learned from a limited set of comparisons. It is a good grader for answers that look like the ones it was trained on, and it can be wrong about anything else. The policy is trained to get high grades, and it will find *whatever* gets high grades, including the reward model's mistakes.

An illustrative example. Suppose the reward model learned "polite words → higher grade". Then

```text
Answer 1: Hi Maya, could you review my PR today when you have a moment? Thanks!      grade 2.1
Answer 2: Thanks! Thanks! Thanks! Thanks! Thanks! Thanks! Thanks! Thanks! Thanks!     grade 2.6
```

The policy will drift toward Answer 2. This is called **reward hacking**: the grade goes up, the real quality goes down. 15.4 shows how common it is.

### 5.2 The fix: a penalty for writing what the SFT model would never write

Keep one more frozen copy of the original SFT model, called the **reference**. The SFT model already writes reasonable text. We allow the policy to change, but we charge it for every token it now likes much more than the reference does.

For each token of the answer, compare the two log-probabilities:

$$
d_t = \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t})
$$

| Symbol | Meaning |
|---|---|
| $\pi_\theta$ | the policy (being trained) |
| $\pi_{\text{ref}}$ | the reference: the frozen SFT model |
| $d_t$ | how much more the policy likes this token than the SFT model does, in log units. 0 means "same as SFT" |

Add $d_t$ over the answer to get the answer's **drift**, and subtract it from the grade, scaled by a coefficient $\beta$ (beta):

$$
R = \text{grade} - \beta \sum_{t=1}^{n} d_t
$$

Work it out with $\beta = 0.1$ and three tokens of each answer:

| | Policy probabilities | SFT (reference) probabilities | $d_t$ | Drift $\sum d_t$ | Grade | $R$ |
|---|---|---|---|---:|---:|---:|
| "Thanks! Thanks! Thanks!" | 0.9, 0.9, 0.9 | 0.30, 0.05, 0.02 | 1.0986, 2.8904, 3.8067 | 7.7956 | 2.6 | **1.8204** |
| A normal polite message | 0.5, 0.4, 0.6 | 0.45, 0.40, 0.50 | 0.1054, 0.0, 0.1823 | 0.2877 | 2.1 | **2.0712** |

Example: $d_1 = \log 0.9 - \log 0.3 = \log 3 = 1.0986$. The SFT model would almost never repeat "Thanks!" ten times, so those tokens carry a large drift. After the penalty, the normal message has the higher reward, so the policy is pushed toward it.

$\beta$ sets the price of drifting. Too low and reward hacking wins. Too high and the policy cannot move away from SFT, so nothing improves. InstructGPT used $\beta = 0.02$; TRL's default is $0.05$.

### 5.3 The name: KL divergence

The average drift over answers the policy writes is a standard quantity called the **KL divergence** from the policy to the reference, written $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$. It is 0 when the two models give the same probabilities, and grows as they diverge. The drift of one answer is a one-answer estimate of it.

This gives the whole goal of RLHF in one line, which 15.3 starts from:

$$
\max_\theta\;\; \mathbb{E}_{x,\; y \sim \pi_\theta}\big[r(x, y)\big] \;-\; \beta\,\mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)
$$

Read it as: "choose weights $\theta$ that make the average grade $r(x,y)$ of the policy's answers high, minus $\beta$ times how far the policy has drifted from the SFT model." $\mathbb{E}_{x,\,y \sim \pi_\theta}$ is the average over prompts $x$ and answers $y$ the policy writes.

<div class="callout warn"><p>The drift of a <em>single</em> answer can be slightly negative: the policy may have sampled a token it now likes less than the reference. Only the average is guaranteed to be at least 0. You will see small negative drifts in the training log; that is normal. This simple estimate is called "k1". TRL also offers "k3", <code>exp(-d) - 1 + d</code> per token, which is never negative and has the same average.</p></div>

## 6. Reusing each batch safely: the PPO clip

### 6.1 Why reuse a batch

Writing answers is the slow part: one forward pass per generated token, for every answer. The gradient step itself is one forward and one backward pass. So after paying for a batch of answers, we want to take **several** gradient steps on it, not one.

### 6.2 What goes wrong if we just repeat

Each step on the same answers pushes the same tokens in the same direction again. Suppose a token in a good answer had probability 0.20 when the answer was written. After step 1 it is 0.30, after step 2 0.50, after step 3 0.80. A handful of answers have now rewritten the model's behaviour, and the advantage was only a noisy estimate from those few answers.

### 6.3 The clip, in plain words

For each answer token, measure how much its probability has changed since the answer was written. Call the probabilities at writing time the **old** policy, $\pi_{\text{old}}$ (the policy's weights frozen when the batch was written, not the SFT model). The change is the **ratio**:

$$
\rho_t = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})}
$$

A ratio of 1.5 means "this token is now 1.5 times as likely as when the answer was written". (In code we compute it as `exp(new_logprob - old_logprob)`, which is the same number and avoids dividing tiny probabilities.)

PPO's rule, with $\epsilon = 0.2$ (epsilon):

- **Stop pushing a token once it has moved 20% in the direction its advantage asked for.** Good tokens stop rising at ratio 1.2; bad tokens stop falling at ratio 0.8. That token then gets **zero** gradient for the rest of this batch.
- **Never stop correcting a move in the wrong direction.** If a good token became *less* likely, or a bad token *more* likely, it is pushed back at full strength.

Check it on five tokens. The objective column is what PPO maximizes for that token; the gradient column is its derivative with respect to the token's log-probability (computed by autograd with the course function below):

| Token | Ratio $\rho$ | Advantage $A$ | Unclipped $\rho A$ | Clipped (ratio limited to $[0.8, 1.2]$) $\times A$ | PPO takes the smaller | Gradient | What happens |
|---|---:|---:|---:|---:|---:|---:|---|
| good, moved a little | 1.1 | +1 | 1.1 | 1.1 | 1.1 | **1.1** | keep pushing up |
| good, moved a lot | 1.5 | +1 | 1.5 | 1.2 | 1.2 | **0** | stop: moved enough |
| good, moved wrong way | 0.5 | +1 | 0.5 | 0.8 | 0.5 | **0.5** | push back up |
| bad, moved a lot | 0.5 | −1 | −0.5 | −0.8 | −0.8 | **0** | stop: moved enough |
| bad, moved wrong way | 1.5 | −1 | −1.5 | −1.2 | −1.5 | **−1.5** | push back down |

Why "take the smaller" gives exactly these rules: the clipped value is a constant once the ratio is outside $[0.8, 1.2]$, so it has no gradient. Taking the smaller of the two picks the clipped constant exactly when the token has already moved far enough in the helpful direction, and picks the unclipped term (which has a gradient) in every other case.

### 6.4 The formula, every symbol named

$$
\text{PPO objective} = \frac{1}{\text{number of answer tokens}} \sum_{\text{answer tokens } t} \min\Big(\rho_t A_t,\;\; \operatorname{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon)\, A_t\Big)
$$

| Symbol | Meaning |
|---|---|
| $\rho_t$ | ratio of the token's probability now to its probability when the answer was written |
| $A_t$ | the token's advantage, $R - V_t$, rescaled (section 4) |
| $\operatorname{clip}(\rho, 0.8, 1.2)$ | $\rho$ limited to the range 0.8 to 1.2 |
| $\min$ | the smaller of the two numbers |
| $\epsilon$ | how far a token may move per batch; 0.2 is the value the PPO paper found best, and the value InstructGPT and TRL use |

We **maximize** this, so the training loss is its negative. On the first gradient step of a batch the policy has not moved yet, every ratio is exactly 1, and the loss is the plain "weight × log-probability" loss of section 3 (its gradient is $A_t$ times the gradient of the log-probability). The clip only matters from the second step on.

The method is **PPO**, Proximal Policy Optimization (Schulman et al., 2017). "Proximal" means "staying near": near the policy that wrote the batch. The KL penalty keeps the policy near the SFT model across the whole run; the clip keeps each update near the policy that wrote the current batch. They solve different problems, and RLHF uses both.

## 7. The whole algorithm, and the code that runs it

### 7.1 The four models

| Model | Starts as | Trained? | Used for | Shipped? |
|---|---|---|---|---|
| **Policy** $\pi_\theta$ | copy of the SFT model | yes, with the PPO loss | writes the answers | **yes**: this is the improved model |
| **Reference** $\pi_{\text{ref}}$ | copy of the SFT model | no, frozen | the drift (KL) penalty | no |
| **Reward model** | your `best.pt` from 15.1 | no, frozen | the grade | no |
| **Value model** | copy of the reward model | yes, with the squared-error loss | the expected reward $V_t$ | no |

What ships is one ordinary GPT with its `lm_head`, exactly like the SFT model, only with different weights.

### 7.2 One iteration, step by step

1. Take a few prompts. Format each with the SFT chat template, ending with the open `<|assistant|>` header.
2. **Write.** The policy writes several answers per prompt, sampling one token at a time, until it writes the end-of-turn marker `<|end|>` or reaches a length limit.
3. **Grade.** The reward model scores each prompt + answer: one number per answer.
4. **Freeze the numbers PPO needs**, with no gradients: the policy's log-probability of every answer token (the "old" log-probabilities), the reference's log-probabilities, the drift, the reward $R = \text{grade} - \beta \cdot \text{drift}$, the value model's $V_t$, and the rescaled advantages $A_t = R - V_t$.
5. **Update, several times.** For a few passes over the batch, in small groups of answers: compute the new log-probabilities, the PPO loss, one optimizer step for the policy; compute $V_t$ again, the value loss, one optimizer step for the value model.
6. Log the average grade, drift and other numbers, save the policy, and go back to 1.

### 7.3 Lining up tokens, log-probabilities and values

This is the one fiddly part, so here is a tiny example. A prompt has 3 tokens `[p0, p1, p2]` and the answer has 2 tokens `[a0, a1]`:

```text
position:          0     1     2     3     4
token:             p0    p1    p2    a0    a1
the model's output at position j predicts token j+1:
column j:          0     1     2     3
predicts:          p1    p2    a0    a1
answer mask:       0     1     1     0      <- columns whose prediction is an answer token
```

So for a sequence of length $T$ we get $T-1$ columns. Column 2 (the output after reading `p2`, the last prompt token) holds the log-probability of `a0`, and the value model's prediction made **before** `a0` was chosen. The answer mask is 1 on exactly the columns that predict answer tokens. Log-probabilities, values, advantages and the mask all use these same columns, so they can be multiplied element by element. Padding columns get mask 0.

### 7.4 Tensor shapes

For one iteration with 2 prompts × 4 answers = 8 answers, padded to the longest prompt + answer, $T$ tokens, vocabulary $V$:

| Tensor | Shape | dtype | Gradient? |
|---|---|---|---|
| `ids` (prompt + answer, right-padded) | `(8, T)` | int64 | – |
| policy logits | `(8, T, V)` | float32 | during the update only |
| `old_logprobs`, `ref_logprobs` | `(8, T-1)` | float32 | no |
| `mask` | `(8, T-1)` | float32, 0 or 1 | no |
| `scores` (grades), `drift`, `total_reward` | `(8,)` | float32 | no |
| `values`, `advantages` | `(8, T-1)` | float32 | no (values: yes during the value update) |
| PPO loss, value loss | `()` | float32 | yes |

Everything lives on the same device as the models.

### 7.5 The code

The trainer is the new file `code/src/llmre/rl/ppo_train.py`. It reuses the pieces you already have: the `GPT` and its bundle format, the chat template, `load_reward_model` from 15.1, and the three small functions in `llmre.rl.ppo` (`ppo_clip_objective`, `kl_penalty`, `compute_advantages`). Here are its parts, in the order of the algorithm.

**Writing answers.** All answers to one prompt start from the same prompt, so they have the same length while being written and need no padding. Each row stops when it has written the token ids of `<|end|>\n`:

```python
@torch.no_grad()
def sample_answers(policy, prompt_ids, n, max_new_tokens, end_ids):
    device = next(policy.parameters()).device
    room = policy.cfg.block_size - len(prompt_ids)
    if room < 1:
        raise ValueError("prompt leaves no room for an answer in the context")
    idx = torch.tensor([prompt_ids] * n, dtype=torch.long, device=device)  # (n, T0)
    answers = [[] for _ in range(n)]
    finished = [False] * n
    for _ in range(min(max_new_tokens, room)):
        logits, _ = policy(idx)                             # (n, T, V)
        probs = F.softmax(logits[:, -1, :].float(), dim=-1)  # (n, V)
        next_ids = torch.multinomial(probs, num_samples=1)  # (n, 1)
        idx = torch.cat([idx, next_ids], dim=1)
        for row in range(n):
            if not finished[row]:
                answers[row].append(int(next_ids[row]))
                finished[row] = answers[row][-len(end_ids):] == end_ids
        if all(finished):
            break
    return answers, finished
```

This is `generate` from 06.3 with a stop condition and without temperature or top-k: we sample from the model's own distribution, because the log-probabilities we compute later must describe the same distribution the tokens came from.

**Padding and the answer mask** (section 7.3):

```python
def collate(prompt_lengths, sequences, device):
    lengths = torch.tensor([len(s) for s in sequences], device=device)
    ids = torch.zeros((len(sequences), int(lengths.max())), dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), ids.shape[1] - 1), device=device)
    for row, (start, seq) in enumerate(zip(prompt_lengths, sequences)):
        ids[row, :len(seq)] = torch.tensor(seq, device=device)
        mask[row, start - 1:len(seq) - 1] = 1.0      # predictions of answer tokens
    return ids, lengths, mask
```

**Log-probabilities of the tokens that were actually written.** This is the SFT computation without the minus sign and without averaging:

```python
def token_logprobs(model, ids):
    logits, _ = model(ids)                                   # (N, T, V)
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)  # (N, T-1, V)
    return logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # (N, T-1)
```

`gather` picks, in each column $j$, the log-probability of the token that is actually at position $j+1$.

**The value model's predictions.** The reward model computes a score at every position and 15.1 kept only the last one. The value model keeps them all:

```python
def position_values(value_model, ids):
    hidden = value_model.backbone(ids)                  # (N, T, C)
    return value_model.score(hidden).squeeze(-1)[:, :-1].float()   # (N, T-1)
```

**The rollout: write, grade, and freeze everything PPO needs** (steps 1–4):

```python
@torch.no_grad()
def rollout(policy, reference, reward_model, value_model, prompts, tokenizer, config):
    end_ids = tokenizer.encode(END_TOKEN + "\n")
    device = next(policy.parameters()).device
    policy.eval()
    prompt_lengths, sequences, finished = [], [], []
    for prompt in prompts:
        prompt_ids = encode_prompt(prompt, tokenizer)
        answers, done = sample_answers(policy, prompt_ids, config.answers_per_prompt,
                                       config.max_new_tokens, end_ids)
        prompt_lengths += [len(prompt_ids)] * len(answers)
        sequences += [prompt_ids + answer for answer in answers]
        finished += done
    ids, lengths, mask = collate(prompt_lengths, sequences, device)

    scores = reward_model(ids, seq_lengths=lengths).float()      # (N,)  the grades
    old_logprobs = token_logprobs(policy, ids)                   # (N, T-1)
    ref_logprobs = token_logprobs(reference, ids)                # (N, T-1)
    drift = (kl_penalty(old_logprobs, ref_logprobs, reduction="none") * mask).sum(1)  # (N,)
    total_reward = scores - config.kl_coef * drift               # (N,)  R
    values = position_values(value_model, ids)                   # (N, T-1)  V_t
    advantages = whiten((total_reward[:, None] - values) * mask, mask)  # A_t, mean 0, std 1
    return {"ids": ids, "mask": mask, "prompt_lengths": prompt_lengths,
            "old_logprobs": old_logprobs, "advantages": advantages,
            "total_reward": total_reward, "scores": scores, "drift": drift,
            "finished": torch.tensor(finished, dtype=torch.float32)}
```

`reward_model(ids, seq_lengths=lengths)` is exactly the call from 15.1: it reads the score at the last real token, so the padding is ignored. `kl_penalty(..., reduction="none")` returns $d_t$ for every column; multiplying by `mask` keeps only answer tokens.

**The update** (step 5). `ppo_clip_objective` is the function of section 6, applied to every column; the mask averages it over answer tokens only:

```python
def ppo_update(policy, value_model, batch, policy_opt, value_opt, config):
    policy.eval()        # no dropout: the same answer must get the same log-probs
    value_model.train()
    n = batch["ids"].shape[0]
    for _ in range(config.ppo_epochs):                       # several passes over one batch
        order = torch.randperm(n).tolist()
        for start in range(0, n, config.minibatch_answers):
            rows = order[start:start + config.minibatch_answers]
            ids, mask = batch["ids"][rows], batch["mask"][rows]
            new_logprobs = token_logprobs(policy, ids)           # (M, T-1), has grad
            objective = ppo_clip_objective(new_logprobs, batch["old_logprobs"][rows],
                                           batch["advantages"][rows], config.clip_eps,
                                           reduction="none")      # (M, T-1)
            policy_loss = -masked_mean(objective, mask)           # maximize -> minimize negative
            policy_opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
            policy_opt.step()

            values = position_values(value_model, ids)           # (M, T-1), has grad
            target = batch["total_reward"][rows][:, None]        # (M, 1)
            value_loss = masked_mean((values - target) ** 2, mask)
            value_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), config.grad_clip)
            value_opt.step()
```

(The checked-in version also skips empty groups and collects statistics: average losses and the fraction of answer tokens that were clipped.)

`policy.eval()` during the update is deliberate. `eval()` does not stop gradients; it only turns dropout off. With dropout on, the same answer would get different log-probabilities on every forward pass, so the ratio would not be 1 even before any update.

**The outer loop** builds the four models and repeats rollout + update:

```python
policy = copy.deepcopy(sft).to(device=device, dtype=torch.float32)          # trained
reference = copy.deepcopy(sft).to(device=device, dtype=torch.float32)       # frozen
reference.eval().requires_grad_(False)
reward_model = reward_model.to(device).eval().requires_grad_(False)         # frozen
value_model = copy.deepcopy(reward_model).to(device=device, dtype=torch.float32)
value_model.requires_grad_(True)                                            # trained
policy_opt = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=0.0)
value_opt = torch.optim.AdamW(value_model.parameters(), lr=config.value_lr, weight_decay=0.0)

for iteration in range(config.iterations):
    chosen = ...                         # the next prompts_per_iteration prompts, cycling
    batch = rollout(policy, reference, reward_model, value_model, chosen, tokenizer, config)
    update = ppo_update(policy, value_model, batch, policy_opt, value_opt, config)
    ...                                  # log, save policy_bundle.pt and history.json
```

Your `sft` object is never modified: the policy and the reference are copies. After each iteration the policy is saved as `policy_bundle.pt`, in the **same bundle format as `sft_bundle.pt`** from 15.1 (config, weights, tokenizer merges), so anything that loads an SFT bundle can load the improved model.

**What our version simplifies**, compared with TRL and the InstructGPT setup (section 8): every token of an answer gets the same reward $R$, with the drift added up over the whole answer; production code charges the drift token by token and spreads credit with a method called GAE (generalized advantage estimation). We also skip value clipping, the pretraining-data mix InstructGPT added ("PPO-ptx"), adaptive $\beta$, and resuming from a checkpoint. These change efficiency and stability, not the idea.

## 8. Hands-on

### Lab 1: watch a policy learn (CPU, one second)

The three-answer policy from sections 3–4, trained for 200 steps with the advantage as the weight. Save as `toy_pg.py` and run `py toy_pg.py`:

```python
import torch

answers = ["polite request", "neutral request", "pushy request"]
grades = torch.tensor([2.0, 1.0, 0.0])      # what a reward model might say
logits = torch.zeros(3, requires_grad=True)  # the whole "policy": 3 numbers
optimizer = torch.optim.SGD([logits], lr=0.1)
torch.manual_seed(0)

for step in range(1, 201):
    probs = torch.softmax(logits, dim=-1)                 # (3,)
    choice = torch.multinomial(probs.detach(), 1).item()  # the model "writes"
    expected = (probs.detach() * grades).sum()            # what we expected
    advantage = grades[choice] - expected                 # better than expected?
    loss = -advantage * torch.log_softmax(logits, -1)[choice]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step in (1, 10, 50, 100, 200):
        p = torch.softmax(logits, -1).tolist()
        print(f"step {step:3d}  wrote {answers[choice]:15s}  advantage {advantage.item():+.2f}  "
              f"p = [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")
```

Output on the checked setup:

```text
step   1  wrote pushy request    advantage -1.00  p = [0.344, 0.344, 0.311]
step  10  wrote pushy request    advantage -1.23  p = [0.465, 0.340, 0.195]
step  50  wrote polite request   advantage +0.25  p = [0.814, 0.125, 0.061]
step 100  wrote polite request   advantage +0.11  p = [0.919, 0.056, 0.024]
step 200  wrote polite request   advantage +0.04  p = [0.968, 0.023, 0.009]
```

Read it: at step 1 the policy wrote the pushy answer, its advantage was −1, and its probability fell from 0.333 to 0.311. Later the polite answer dominates, and its advantage shrinks toward 0, because once the policy almost always writes it, it is no longer *better than expected*. Learning slows down by itself.

Try two changes and predict the result before running:

1. Add 100 to every grade: `grades = torch.tensor([2.0, 1.0, 0.0]) + 100`. The output is **identical**, because the expected grade also rises by exactly 100, so every advantage stays the same.
2. Keep the +100 and remove the baseline: `advantage = grades[choice]`. Now the first answer written gets a weight of about 100, its probability jumps to 1.000 in one step, and the policy writes only that answer forever. With seed 0 the first answer is the pushy one, and the run ends with `p = [0.000, 0.000, 1.000]`: the worst answer, locked in. With seeds 1, 2 and 3 it locks onto the polite, neutral and polite answer. Which answer wins is decided by the first dice roll, not by the grades.

### Lab 2: improve your own SFT model with your reward model

You need the two files from 15.1: `sft_bundle.pt` and the reward run's `best.pt`. The prompts file holds plain `{"prompt": ...}` rows; `code/data/rl-prompts-demo/prompts.jsonl` has six new prompts in the same style as the preference data (not the same prompts, so you can see whether the policy improves on requests the reward model was not trained on). Any JSONL with a `prompt` field works, including the 15.1 preference files.

From `code/`, after `python -m pip install -e ".[dev]"`:

```bash
python -m llmre.rl.ppo_train --sft sft_bundle.pt --reward checkpoints/reward-run/best.pt --prompts data/rl-prompts-demo/prompts.jsonl --output checkpoints/ppo-run --iterations 20 --answers-per-prompt 4 --max-new-tokens 64 --lr 0.00001 --kl-coef 0.05
```

`python -m llmre.rl.ppo_train --help` lists every setting; each matches a field of `PPOTrainConfig`. Every iteration prints one JSON line:

| Field | Meaning | What to look for |
|---|---|---|
| `score` | average grade from the reward model | should trend up; it is noisy from iteration to iteration |
| `drift` | average drift from the SFT model (the KL estimate) | small and slowly growing. Fast growth together with rising `score` means reward hacking: raise `--kl-coef` |
| `total_reward` | `score - kl_coef * drift` | what PPO actually maximizes |
| `finished` | fraction of answers that wrote the end-of-turn marker | should not fall; if it does, raise `--max-new-tokens` |
| `clipped_fraction` | fraction of answer tokens the clip stopped | 0 means the clip never acted; large values mean steps are too big: lower `--lr` |
| `value_loss` | how wrong the value model's predictions are | should fall over the run |
| `example` | the first answer of the iteration, decoded | read it: are the answers actually getting better? |

**How you know it worked.** `score` is higher in the last iterations than in the first, `drift` stays moderate, and the decoded examples read better *to you*, not only to the reward model. The output directory holds `policy_bundle.pt` (the improved model) and `history.json` (all logged lines). Load the bundle like the SFT bundle and generate answers to new prompts with both models side by side.

With the small course model and the six demo prompts, expect small, noisy changes: this run checks that the full loop works on your models. Real improvements need thousands of prompts and a reward model trained on far more comparisons.

**Without your checkpoints**, run the wiring tests (about 25 seconds on CPU):

```bash
python -m pytest tests/test_ppo_training.py
```

They check the answer mask, the log-probabilities against a hand computation, stopping at `<|end|>`, that the first rollout has zero drift and ratio 1, that clipped tokens get zero gradient, that the reward rises on a tiny model with a simple stand-in grader, that a larger $\beta$ keeps drift lower, and that the SFT model is left unchanged.

### What it costs on a real model

Generation dominates the time: every answer token is one forward pass of the policy. Memory, for a 1B-parameter model in float32: the policy and the value model are trained, so each needs weights, gradients and two AdamW buffers, 16 bytes per parameter, 16 GB each; the reference and the reward model need only weights, 4 GB each. That is about **40 GB before activations**. Mixed precision, sharding (Modules 7–9), LoRA adapters (14.3), or sharing one backbone between policy and value model reduce it. This weight is a large part of why DPO (15.3) and GRPO (16.1) exist.

### The production library: TRL

Hugging Face TRL (version 1.18.0, released 2026-11-13) has a full PPO trainer. It now lives in the experimental area, `trl.experimental.ppo.PPOTrainer` with `trl.experimental.ppo.PPOConfig`. Its settings map onto ours:

| Ours | TRL `PPOConfig` | TRL default |
|---|---|---|
| `kl_coef` ($\beta$) | `kl_coef` | 0.05 |
| `clip_eps` ($\epsilon$) | `cliprange` | 0.2 |
| `ppo_epochs` | `num_ppo_epochs` | 4 |
| the k1 drift | `kl_estimator` (`"k1"` or `"k3"`) | `"k1"` |
| (not implemented, see the exercise) | `missing_eos_penalty` | `None` |
| one reward for every token | `gamma`, `lam` (GAE) | 1.0, 0.95 |

TRL also has `vf_coef` (0.1 by default), because its loss adds the value loss to the policy loss with that weight; we use two separate optimizers instead, which gives the same kind of update. Many recent post-training recipes use GRPO or DPO instead of PPO, which is likely why PPO moved to the experimental area; the PPO loop is still the reference point those methods are explained against.

**Sources checked 2026-11-14:** [PPO paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347); [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155), appendix C.4 for $\beta = 0.02$, clip ratio 0.2, the policy initialized from SFT and the value function from the reward model; [TRL on PyPI](https://pypi.org/project/trl/) and the [TRL PPO trainer docs](https://huggingface.co/docs/trl/main/en/ppo_trainer) for the version, location and parameter names; [The N Implementation Details of RLHF with PPO](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo) for the details production code adds. The toy policy, the example grades and the example probabilities in this lesson are illustrative.

## Common mistakes

- **Letting the grade carry a gradient.** The reward, drift, values used in the advantage, and old log-probabilities are fixed numbers for the update. Compute them under `torch.no_grad()`. Only the new log-probabilities (and, in the value loss, the new values) carry gradients.
- **Using the raw grade as the weight.** Grades are only meaningful as differences (15.1). Subtract the expected value (section 4).
- **Averaging over padding or prompt tokens.** Every per-token quantity must be multiplied by the answer mask before averaging.
- **Sampling with temperature or top-k, then computing log-probabilities without them.** The tokens then come from a different distribution than the one in the loss. Sample from the plain softmax, as `sample_answers` does.
- **Leaving dropout on.** The first ratio is not 1 and the clip acts on noise. Keep the policy in `eval()` mode.
- **Dropping the KL penalty because "the score goes up faster without it".** That is reward hacking starting. Read the example answers.
- **Shipping the wrong model.** The policy is the product. The reference, reward model and value model are only used during training.

## Exercise: penalize answers that never finish

If an answer hits `--max-new-tokens` before writing `<|end|>`, the reward model still grades the cut-off text, and it may like it. TRL's `missing_eos_penalty` subtracts a fixed amount from the grade of such answers. Add a `missing_eos_penalty: float = 0.0` field to `PPOTrainConfig` and apply it in `rollout`.

<details><summary>Hint</summary>

`rollout` already knows which answers finished: the `finished` list returned by `sample_answers`. The penalty must change the grade *before* the reward and the advantages are computed.

</details>

<details><summary>Stronger hint</summary>

Build a float tensor from `finished` on the same device as `scores`, and subtract `config.missing_eos_penalty * (1 - finished)` from `scores` right after the reward model call.

</details>

<details><summary>Solution</summary>

```python
# in PPOTrainConfig
missing_eos_penalty: float = 0.0

# in rollout, right after: scores = reward_model(ids, seq_lengths=lengths).float()
done = torch.tensor(finished, dtype=torch.float32, device=scores.device)  # (N,)
scores = scores - config.missing_eos_penalty * (1.0 - done)
```

Everything after it (reward, value target, advantages, the logged score) now sees the penalized grade. A test: give the stand-in grader in `tests/test_ppo_training.py` a large penalty with `max_new_tokens=2`, so no answer can finish, and check that the logged score drops by exactly the penalty.

</details>

## Debugging exercise

A colleague adds dropout 0.1 to the model config and changes `ppo_update` to call `policy.train()`. On the very first gradient step of every batch, before the policy has changed at all, `clipped_fraction` is already about 0.3. Why, and what is the fix?

<details><summary>Answer</summary>

With dropout on, each forward pass randomly zeroes different activations, so the same answer gets different log-probabilities in the rollout and in the update. The ratio $\exp(\text{new} - \text{old})$ is then not 1 even though no weights changed, and many tokens land outside $[0.8, 1.2]$ by chance. The clip acts on noise and those tokens get no gradient. Fix: keep the policy in `eval()` mode during both rollout and update (gradients still flow; `eval()` only disables dropout), or set dropout to 0 for RL. The test `test_first_rollout_has_zero_drift_and_ratio_one` checks this property.

</details>

## Research connection

<div class="callout paper"><p><strong>Read:</strong> <a href="https://arxiv.org/abs/2203.02155">InstructGPT (Ouyang et al. 2022)</a>, section 3.5 (the "Reinforcement learning (RL)" part) and appendix C.4: the reward with the per-token KL penalty, and how the policy and value model are initialized. Then <a href="https://arxiv.org/abs/1707.06347">PPO (Schulman et al. 2017)</a>, sections 3 and 5 for the clipped objective and the algorithm; skip the Atari and MuJoCo results. After that, <a href="https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo">The N Implementation Details of RLHF with PPO</a> lists what production code adds to the simple loop you wrote.</p></div>

**Code this after reading:** the per-token version of the reward. Charge $\beta d_t$ at each token, add the grade only at the last token, and give each token the sum of what comes *after* it as its reward ("reward-to-go"). Compare its learning curve with ours on the stand-in grader from the tests.

## Check yourself

<details><summary>Why can't we call <code>loss.backward()</code> through the reward model's grade, as we did in 15.1?</summary>

In 15.1 the text was fixed data, and every step from text to loss was a calculation. Here the policy chooses the text by sampling, and a small change to the weights either leaves the sampled tokens (and the grade) unchanged or changes them abruptly. There is no smooth slope from weights to grade. So we treat the sampled answer as fixed text and use the grade only as a weight on the answer's log-probability, which does have a gradient.

</details>

<details><summary>In one sentence, what does the policy-gradient loss do to an answer with a positive advantage, and to one with a negative advantage?</summary>

Positive: a scaled SFT step on that answer, making it more likely. Negative: the opposite of an SFT step, making it less likely. The size of the step is proportional to the advantage.

</details>

<details><summary>The reward model gives every answer to a prompt a grade between 50 and 52. What goes wrong with weight = grade, and what fixes it?</summary>

Every sampled answer gets a large positive weight, so whatever the policy writes is pushed up, and the useful signal (the differences of up to 2) is buried. Subtract the expected grade for that prompt: the advantage $A = R - V$ is between about −1 and +1, positive for better-than-usual answers and negative for worse ones.

</details>

<details><summary>What do the reference model and the KL penalty protect against, and what happens if β is far too large?</summary>

The reward model is imperfect, so the policy can find answers it overrates (reward hacking). The penalty charges the policy for tokens the original SFT model would rarely write, so it can only improve in ways that stay close to reasonable text. If $\beta$ is far too large, any change costs more than it gains, the policy stays at the SFT model, and nothing improves.

</details>

<details><summary>A good token (advantage +2) had probability 0.10 when the answer was written and 0.15 now. With ε = 0.2, what is its PPO objective and gradient?</summary>

Ratio $= 0.15 / 0.10 = 1.5$. Unclipped $1.5 \times 2 = 3.0$; clipped $1.2 \times 2 = 2.4$; the minimum is $2.4$, the clipped constant, so its gradient is 0. The token already moved more than 20% in the helpful direction, so this batch stops pushing it.

</details>

<details><summary>Which of the four models ship, and which are trained?</summary>

Only the policy ships. The policy and the value model are trained; the reference and the reward model stay frozen.

</details>

## Next

PPO works, but it needs four models, a slow generation loop, and several settings that interact. 15.3 starts from the one-line RLHF goal in section 5.3 (average grade minus $\beta$ times KL) and the Bradley-Terry model from 15.1, and shows that the whole loop can be replaced by a single supervised loss on the preference pairs: **DPO**, with no reward model, no value model, and no sampling during training.

Continue to [15.3 · DPO: derivation & implementation](lessons/module-15/lesson-03.md).
