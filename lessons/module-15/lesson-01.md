# 15.1 · Bradley-Terry & the reward model

<div class="prereq">
<p><strong>Prerequisites:</strong> supervised fine-tuning from <a href="#/lessons/module-14/lesson-01">14.1 · Instruction data & chat templates</a> and the response-only loss of <a href="#/lessons/module-14/lesson-02">14.2 · Loss masking</a>; the assembled model and its hidden states from <a href="#/lessons/module-06/lesson-01">06.1 · Assembling GPT-2</a>; the log-softmax / cross-entropy machinery from <a href="#/lessons/module-01/lesson-04">01.4</a> and <code>llmre.evaluation.metrics</code>.</p>
<p><strong>You will learn:</strong> how pairwise comparisons add a preference signal beyond SFT demonstrations; what pairwise preference data <code>(x, y_w, y_l)</code> is; the <strong>Bradley-Terry</strong> model that turns a scalar reward into a probability of preference; and how to build and train a <strong>reward model</strong> — a language model with a scalar head — with the loss <code>-log &sigma;(r_w - r_l)</code>.</p>
<p><strong>Why this matters for ML:</strong> the reward model is the numerical definition of "good" that the whole RLHF pipeline (lesson 15.2) optimizes against. Get it wrong and the policy will faithfully optimize the wrong thing. Even DPO (lesson 15.3), which trains no explicit reward model, is <em>derived from</em> exactly this Bradley-Terry loss — so this lesson is the foundation of the entire module.</p>
</div>

## The causal story: imitation has a ceiling

Module 14 gave us a model that *follows instructions*. SFT works by imitation: we showed the model thousands of `(prompt, ideal response)` pairs and trained it, with next-token cross-entropy, to reproduce the response. That is a real capability jump — the model now answers questions instead of continuing them.

SFT can already teach good behavior when its demonstrations show it. What its ordinary next-token loss does not directly consume is a comparison: "for this prompt, response A is better than response B." Preference data supplies that additional signal. We can show the model two plausible answers and teach it which one better serves the user, even when both are grammatically fluent and contain correct information.

> Pretraining teaches broad language patterns and factual knowledge; SFT teaches behavior through demonstrations. Preference training adds explicit comparisons of candidate responses. A reward model learns to predict those comparisons so that a later policy-training loop can use its scores.

Two reasons to collect comparisons alongside demonstrations:

- **A comparison can be easier to provide than a demonstration.** A reviewer may know which of two explanations better suits a beginner without being able to write the ideal explanation from scratch. A clear labeling rubric still matters.
- **Generated candidates reveal tradeoffs.** Comparing the model's own responses can expose recurring problems such as unnecessary detail, a mismatched tone, or confident guesses. Those comparisons give a direct target for improving its choices. Neither SFT nor preference optimization guarantees improvement without evaluation.

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

### What preference adds when the model already knows the facts

Knowing that fire is hot is a poor motivating example for RLHF: a pretrained model may already know that fact. Preference training is also about **which response to choose when several responses are possible**. A response can be factually correct yet too technical for its audience, ignore a requested format, sound accusatory, or offer a confident diagnosis without enough evidence. Factual accuracy remains part of quality; it is not the only dimension.

Here are three illustrative comparisons. Their labels follow the stated user request; they are examples written for this lesson, not rows claimed to come from a human-labeling study.

**Tone and intent.** The user asks: "Write a short, polite Slack message asking Maya to review my PR today. Do not imply that she is late."

- **Chosen:** "Hi Maya, could you review my PR today when you have a moment? Thanks!"
- **Rejected:** "Hi Maya, I'm still waiting for your review of my PR; please get it done today so I can move forward."
- **Reason:** both messages ask for a PR review today, but the rejected version implies an overdue obligation and introduces pressure the user explicitly wanted to avoid. If the request were for a firm follow-up after a missed deadline, the appropriate tone could change.

**Teaching usefulness.** The user asks for gradient accumulation explained to a programmer who can fit only two examples on their GPU, with a concrete batch-size example.

- **Chosen:** "Process two examples at a time. For each microbatch, divide its mean loss by four before calling backward, and keep the accumulated gradients. After four microbatches, take one optimizer step: that gives an effective batch of eight examples. Clear the gradients before the next group."
- **Rejected:** "Gradient accumulation adds gradients from multiple microbatches before updating the parameters. It reduces the activation memory needed for a larger effective batch."
- **Reason:** both descriptions are correct. The chosen answer connects the idea to the user's constraints and supplies the requested example. Here the preferred answer is **longer**.

**Brevity and audience.** The user asks for an API defined in one sentence for a nontechnical colleague.

- **Chosen:** "An API is a set of rules that lets one piece of software ask another for information or actions."
- **Rejected:** "An API, or application programming interface, defines how software components interact. APIs may use approaches such as REST, GraphQL, or RPC, with conventions for authentication, errors, and versioning."
- **Reason:** the rejected version contains useful technical information, but exceeds the requested length and assumes vocabulary the audience may not know. Here the preferred answer is **shorter**.

The training target is "this response better meets this request," not "longer is better," "shorter is better," or "the other response must be false." Labelers need a rubric and enough context to make that judgment; different legitimate preferences can disagree. [InstructGPT's labeling description and quality criteria](https://arxiv.org/html/2203.02155v1#S3.SS4) include task intent and constraints alongside truthfulness.

SFT could also teach these behaviors through good demonstrations. RLHF is one way to use **comparative feedback** to improve the policy; it is not required to establish a basic fact, and it does not guarantee truthfulness. The pair loss below learns the supplied preferences, so poor or biased labels can teach poor choices.

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

<div class="callout warn"><p>Compute the loss as <code>-F.logsigmoid(margin)</code>, <strong>not</strong> <code>-torch.log(torch.sigmoid(margin))</code>. For a large negative margin, <code>sigmoid</code> underflows to <code>0.0</code> and <code>log(0)</code> is <span>$-\infty$</span>; <code>logsigmoid</code> is a single numerically stable kernel (it internally uses the <code>log(1+exp())</code> = softplus identity) and never overflows. Same reasoning as the stable log-softmax in <a href="#/lessons/module-01/lesson-04">01.4</a>.</p></div>

## The big picture: the reward model is a temporary grader, not part of the product

Before the mechanics, one fact that the rest of this module depends on: **the reward model is only used while training. It is never shipped, and it never runs when a user chats with the finished model.**

### Why it exists at all

RLHF wants to train the model on "humans prefer this answer". The ideal loop would be:

1. the model writes an answer;
2. a human reads it and says how good it is;
3. the model is nudged toward answers humans like.

Step 2 is the problem. RL training needs millions of judged answers, and every time the model changes, it writes new answers that nobody has judged yet. Humans cannot grade millions of fresh answers fast enough.

So we split the work in two:

- **Humans provide comparisons offline:** for example, they prefer a tactful PR-review request over an accusatory version, or a concrete beginner explanation over an abstract one. A training round uses a collected set of these judgments.
- **A model learns to copy the humans' judgment** from that dataset. That model is the **reward model**. After training it can grade any new answer in milliseconds, as many times as needed.

The reward model is a **stand-in for the human labeler**. That is its whole job.

### The full timeline, start to finish

| stage | what is trained | models that exist | lesson |
|---|---|---|---|
| 1. Pre-training | GPT learns language | GPT (with `lm_head`) | Modules 6–9 |
| 2. SFT | GPT learns the chat format | SFT model (with `lm_head`) | Module 14 |
| 3. Reward-model training | a **copy** of the SFT model learns to grade | SFT model **+** reward model (copy with scalar head) | this lesson |
| 4. RL (PPO) | the SFT model, now called the **policy**, learns to write answers the grader likes; the grader is **frozen** | policy **+** frozen reward model | 15.2 |
| 5. Ship | nothing | **policy only**: one GPT, one `lm_head` | — |

At stage 5 the reward model is put aside (archived, or thrown away). What users talk to is an ordinary GPT with its normal `lm_head` that generates text exactly as in Module 6. One model, one head, one forward pass per generated token. You are right that the shipped GPT does not carry extra heads; the scalar head lives only on the temporary grader.

### Why two forward passes during training (and only then)

During stage 4, each training step does:

1. **Policy generates** a response to "Write a polite PR-review request": an unnecessarily pushy message.
2. **Reward model scores** the complete prompt + message: an illustrative $r = 1.48$.
3. Another sampled response is a tactful message that meets the same request; suppose its score is $r = 2.14$.
4. **The policy is updated** using its sampled responses and reward signals (lesson 15.2 shows how). This sketch compares two possible responses to make the preference concrete; PPO does not require every rollout to contain a chosen/rejected pair.

Yes, that is extra compute: a grading pass for every generated answer. It is the price of not needing a human in the loop, and it is paid **only during training**, like a teacher grading homework during the course. After training, the policy can favor useful, appropriately phrased responses on its own, so this deployment does not need to score each answer with the reward model.

### Why a separate model, and not a second head on the same GPT?

Because the grader must **stay fixed while the student changes**. In stage 4 the policy's weights are updated every step. If the scalar head sat on the policy's own backbone, every update to the policy would also change the grader, and the policy could learn to make its own grader happy instead of writing better answers (this is **reward hacking**, lesson 15.4). With two separate copies, the grader is frozen and the policy cannot touch it.

### Your intuition is where the field went next

The two-model setup is heavy: two big models in memory, a grading pass per answer. Researchers asked the same question you are asking, and there are two well-known answers, both later in this module:

- **DPO** (lesson 15.3, publicly documented, Rafailov et al. 2023): uses the same preference pairs but trains the policy **directly**, with no reward model and no generation during training. The math in this lesson is still the foundation: DPO is derived from the Bradley-Terry loss.
- **Generative reward models / LLM-as-a-judge** (reasonable industry practice): the grader is an ordinary GPT with its normal `lm_head` that *writes* its verdict as text ("Answer A is better" or "Score: 8/10"). No special head at all.

Scalar-head reward models are still widely used and publicly documented (InstructGPT; Llama 2 trained separate helpfulness and safety reward models), so we build one here: it is the cleanest way to see what "learning human preferences" means numerically.

## What *is* a reward model, mechanically?

[Compare with the complete original lesson](lessons/module-15/lesson-01.backup.md).

### 1. Intuition: start with the SFT forward pass you already know

You already know how to train the SFT model. Keep its tokenizer, chat format, embeddings, transformer blocks, and final LayerNorm in mind. We will reuse that computation. The changes are **what we read from it** and **what loss we attach**.

Use the PR-review example above as an illustrative preference record. Its label expresses the requested tone and intent:

```text
prompt:    Write a short, polite Slack message asking Maya to review my PR today. Do not imply that she is late.
chosen:    Hi Maya, could you review my PR today when you have a moment? Thanks!
rejected:  Hi Maya, I'm still waiting for your review of my PR; please get it done today so I can move forward.
label:     the chosen answer better follows the requested tone
```

Both answers are already in the dataset. During reward-model training we feed them in as input; we do not ask the model to generate them.

Build **two separate sequences**, each containing the same prompt and one answer:

```text
chosen row:
<|user|>
Write a short, polite Slack message asking Maya to review my PR today. Do not imply that she is late.
<|end|>
<|assistant|>
Hi Maya, could you review my PR today when you have a moment? Thanks!
<|end|>

rejected row:
<|user|>
Write a short, polite Slack message asking Maya to review my PR today. Do not imply that she is late.
<|end|>
<|assistant|>
Hi Maya, I'm still waiting for your review of my PR; please get it done today so I can move forward.
<|end|>
```

Now score each row using **the same reward model**, with the same backbone weights and the same scalar-head weights:

```text
prompt + chosen message   --> shared reward model --> r_chosen   --+
                                                                  +--> difference --> one pair loss
prompt + rejected message --> shared reward model --> r_rejected --+
```

**Yes, we run the two sequences independently.** You can call `rm(chosen)` and then `rm(rejected)`, or stack them as two batch rows and call `rm(two_rows)` once. Batching evaluates both rows together for efficiency; attention operates within each row, so neither answer can attend to the other. No weights are updated between the two scores.

The reward model does not receive the words "chosen" or "rejected" as input. It only sees each prompt-and-answer sequence. The training code knows which score belongs to the human's winner and puts that score first in the subtraction. **The comparison happens in the loss, after scoring.**

Here is the exact change from SFT:

| Part | SFT from Module 14 | Reward-model training here |
|---|---|---|
| Inputs for this record | Prompt + desired answer | Two rows: prompt + chosen, prompt + rejected |
| Backbone output | A hidden vector at every token position | A hidden vector at every token position |
| Output head | `lm_head`: each vector becomes $V$ vocabulary logits | `score`: each vector becomes a scalar; we keep the final real position |
| Supervision | Correct next-token ids on the response | Human preference: chosen should score higher |
| Loss | Cross-entropy at response prediction positions | One loss from the difference of two scores |

We make a **separate copy** of the SFT backbone for this scorer. The original SFT model keeps its language head and will become the policy in 15.2. This section trains the scorer, not that policy.

### 2. Mathematics: what exactly is the "last token" readout?

#### Attention weights and hidden states are different tensors

Inside a transformer block, attention weights say how strongly each position reads other positions. With $N_h$ attention heads and sequence length $T$, their shape is `(B, N_h, T, T)`. They are intermediate values used to mix value vectors. Here $B$ counts sequences in the forward pass.

After attention, its output projection, residual connections, the MLPs, all the remaining blocks, and final LayerNorm, the backbone returns **hidden states** of shape `(B, T, C)`. Each position has a vector of $C$ features. This is the tensor that fed `lm_head` in 06.1.

```text
token ids                             (B, T)
    |
embeddings + all transformer blocks
    |   attention is computed for all positions, as before
final LayerNorm
    |
hidden states                         (B, T, C)
    |
scalar head at every position         (B, T, 1)
    |
remove the size-1 output axis          (B, T)
    |
select final real score per row       (B,)
```

For a shape illustration, suppose a tokenizer produces full-sequence lengths 256 and 320 for a pair; these are assumed sizes, not measured token counts for the Slack messages. After right-padding, $B=2$ and $T=320$. With hidden width $C=16$, the backbone returns `(2, 320, 16)`, the head produces `(2, 320, 1)`, and the readout keeps two final real-position rewards `(2,)`. Conceptually, those rewards depend on two final hidden vectors of shape `(2, 16)`. We do **not** select an attention head, an attention matrix, or the last feature of a vector. We select one **position**, keeping all $C$ features there.

#### We still compute every position; we attach the output at the end

Causal attention gives each position access to its own token and the earlier tokens in **its row**:

| Position in the chosen row | Its hidden state can depend on |
|---|---|
| `review` inside the user prompt | The user marker and prompt text up to that position; no answer yet |
| `Thanks` near the end of the chosen message | The full prompt, assistant marker, and answer text up to that position |
| Final `<end>` | The whole prompt, the answer, and `<end>` |

The role markers are abbreviated in this table. The important fact is the direction: a prompt position cannot read the answer that comes after it. The final position can read all of it.

That makes the final hidden state a useful place to learn a whole-response score. It is **not guaranteed to perfectly summarize everything** just because it is last. It has access to the whole sequence, and preference training teaches the backbone and head to extract information useful for judging it. Last-position readout is an architectural choice; other pooling choices are possible.

Why can two identical `<end>` tokens give different rewards? Their **input token embeddings** are identical, but their **contextual hidden states** need not be:

```text
<end> after the tactful message --> hidden vector influenced by that full message
<end> after the pushy message   --> hidden vector influenced by that different message
```

We read the vector **after the backbone**, not the fixed embedding-table row for `<end>`. The end marker itself is not essential: a model can instead read the last answer token. What matters is a consistent final real position after the complete answer.

You have also used a final-position readout before: `GPT.generate` in [06.3](lessons/module-06/lesson-03.md) takes `logits[:, -1, :]` to predict the next token. During SFT, we supervised many response positions because each was a different next-token prediction. Here the label judges **one complete answer**, so we use one output for that answer.

#### The scalar head

Let $h_T(x,y)$ be the final hidden vector after processing the formatted prompt $x$ and answer $y$. The subscript $T$ means the final real token position; the two answers can have different lengths. With learned head weights $w$ and bias $b$:

$$
r_\theta(x,y) = w^\top h_T(x,y) + b.
$$

Here $h_T$ and $w$ each contain $C$ numbers; $b$ and the resulting reward contain one number each. The symbol $\theta$ means all trainable reward-model parameters: the backbone and the head. For example, with $h_T=(2,1)$, $w=(-0.3,0.4)$, and $b=0.1$, the score is $(-0.3)(2)+(0.4)(1)+0.1=-0.1$.

There is no vocabulary softmax here. The scalar can be any real number. The sigmoid comes **later**, when we compare the two rewards.

### 3. Numerical example: two forwards, one loss, one update

To focus on the new mechanism, suppose the backbone has already produced these final hidden vectors:

```text
chosen:    h_w = (2, 1)
rejected:  h_l = (1, 2)
```

These are illustrative backbone outputs for the two messages, not token embeddings and not a claim that a particular phrase always has the same hidden vector. We do not need to rederive attention to follow what happens next.

#### Score each answer with the same head

Use the head above, $w=(-0.3,0.4)$ and $b=0.1$:

$$
r_w = (-0.3)(2) + (0.4)(1) + 0.1 = -0.1
$$

$$
r_l = (-0.3)(1) + (0.4)(2) + 0.1 = 0.6.
$$

The model currently ranks the rejected answer higher. The human's label does not force the forward pass to be right; it tells the loss which direction should be right.

#### Join the two scores into one supervised prediction

The Bradley-Terry prediction is the probability that the human prefers the chosen answer:

$$
m = r_w-r_l = -0.1-0.6=-0.7,
\qquad p=\sigma(m)=0.331812.
$$

The target for this comparison is **1: chosen wins**. Using the cross-entropy you already know, the loss for this observed outcome is:

$$
\mathcal L = -\log p = -\log(0.331812) = 1.103186.
$$

This is where both answers enter the same loss. There is no separate correct target such as "chosen must score 10" and "rejected must score 0." We only supervise their ordering. Adding 100 to both scores would still give margin $-0.7$ and exactly the same loss.

#### One backward pass reaches both scoring branches

The derivative of this loss with respect to the margin is $p-1$. Since the chosen score is added to the margin and the rejected score is subtracted:

$$
\frac{\partial\mathcal L}{\partial r_w}=p-1=-0.668188,
\qquad
\frac{\partial\mathcal L}{\partial r_l}=1-p=+0.668188.
$$

Gradient descent moves opposite to a gradient: the loss asks for a higher chosen score and a lower rejected score. The scores are outputs, not two independent parameters that the optimizer directly edits. It edits their **shared weights**, combining the contributions from both branches.

For a visible hand calculation, hold the backbone vectors fixed and update only the head. Because both scores use the same $w$:

$$
\frac{\partial\mathcal L}{\partial w}
= (p-1)h_w+(1-p)h_l
= (p-1)(h_w-h_l)
= -0.668188(1,-1)
= (-0.668188,+0.668188).
$$

The shared bias cancels in the difference, so this pair loss gives it zero gradient. Using learning rate $1$ just to make the arithmetic visible:

$$
w_{\text{new}}=(-0.3,0.4)-1\,(-0.668188,+0.668188)
=(0.368188,-0.268188).
$$

Score the same two vectors again with that **one updated head**:

| Quantity | Before the update | After the update |
|---|---:|---:|
| Chosen reward | -0.100000 | 0.568188 |
| Rejected reward | 0.600000 | -0.068188 |
| Margin | -0.700000 | 0.636376 |
| Pair loss | 1.103186 | 0.424749 |

The ranking has flipped and the loss has fallen. This large learning rate is for the tiny head-only calculation, not a recommendation for fine-tuning an LLM.

In full fine-tuning the same backward pass also reaches the backbone through both final vectors. Earlier tokens can influence the final vector through attention, so they can receive gradient even though we attach no separate reward loss to their positions. **Selecting the final output does not freeze the earlier computation.** With shared parameters and many examples, an optimizer step need not improve every individual pair; the goal is to reduce the training objective across the data.

### 4. Tensor shapes: count pairs separately from sequences

Let $P$ be the number of preference pairs in a minibatch. There are $2P$ input sequences. Continue the assumed length example above: one pair with real lengths 256 and 320, right-padded to width 320, and hidden width 16:

| Tensor | General shape | This example | dtype |
|---|---|---|---|
| Chosen ids, including padding | `(P, T)` | `(1, 320)` | `int64` |
| Rejected ids | `(P, T)` | `(1, 320)` | `int64` |
| Both groups, stacked | `(2*P, T)` | `(2, 320)` | `int64` |
| Backbone hidden states | `(2*P, T, C)` | `(2, 320, 16)` | `float32` |
| Position scores before readout | `(2*P, T)` | `(2, 320)` | `float32` |
| Scalar rewards | `(2*P,)` | `(2,)` | `float32` |
| Chosen / rejected rewards | Each `(P,)` | Each `(1,)` | `float32` |
| Margins / per-pair losses | Each `(P,)` | Each `(1,)` | `float32` |
| Mean minibatch loss | `()` | One scalar | `float32` |

#### Unequal answer lengths: pad first, then stack

Real answers usually have different lengths. The table above shows equal stored widths because padding has already been applied; the real lengths remain 256 and 320. The following smaller, assumed lengths make the padding counts easy to inspect by hand. **Before stacking, right-pad every prompt + answer sequence to the longest sequence in that minibatch.** The real text stays unchanged; padding makes the tensor rectangular.

Suppose a minibatch contains two preference pairs. These lengths count tokens in the **entire formatted prompt + answer**, including chat markers, not just the answer:

| Row order | Real tokens | Right-padding added | Stored width |
|---|---:|---:|---:|
| Chosen answer 1 | 8 | 5 | 13 |
| Chosen answer 2 | 11 | 2 | 13 |
| Rejected answer 1 | 13 | 0 | 13 |
| Rejected answer 2 | 7 | 6 | 13 |

The longest sequence has 13 tokens, so this minibatch uses `T = 13`. "Both groups, stacked" now means one tensor `ids` of shape **`(4, 13)`**: four sequences, thirteen columns each. Each chosen row still shares its prompt with its corresponding rejected row; their answer lengths can differ.

```text
row 0: [8 real tokens ] [5 padding tokens]   chosen 1
row 1: [11 real tokens] [2 padding tokens]   chosen 2
row 2: [13 real tokens] [no padding      ]   rejected 1
row 3: [7 real tokens ] [6 padding tokens]   rejected 2
```

We keep the real lengths separately: `[8, 11, 13, 7]`. The final real token indices are therefore `[7, 10, 12, 6]`, because tensor indices start at zero. Each row's reward comes from its own final real position, even though all rows occupy the same stored width.

For this batch, the existing functions are used as follows inside the full training loop shown below:

```python
ids, lengths = collate_pairs(batch, device)
# ids: (4, 13), int64; lengths: (4,), int64 = [8, 11, 13, 7]
# Both tensors and the model are on device.

rewards = model(ids, seq_lengths=lengths)       # (4,), float32
chosen, rejected = rewards.chunk(2)           # each (2,)
# chosen = [r_chosen_1, r_chosen_2]
# rejected = [r_rejected_1, r_rejected_2]
loss = bradley_terry_loss(chosen, rejected)    # () mean of the two pair losses
```

`collate_pairs` arranges all chosen rows first, then all rejected rows. Splitting the rewards in half preserves pair alignment: chosen 1 is compared with rejected 1, and chosen 2 with rejected 2. Attention still operates independently within each row.

**Use one shared padding width across both groups.** Padding chosen rows to their own maximum of 11 and rejected rows to their own maximum of 13 would leave tensors of shapes `(2, 11)` and `(2, 13)`; they cannot be concatenated along the batch axis. Pad both groups to 13 first. Another minibatch can use a different maximum width.

We do not read a padding position as the answer's reward. In this causal GPT, padding placed after the real text also cannot affect the earlier real-token hidden states. A different backbone may require a padding attention mask as well; selecting `seq_lengths - 1` handles the reward readout, not every model's masking requirements.

All tensors and model parameters below are on **CPU**. In GPU training, move both the model and input tensors to the same device; hidden states follow the model's compute dtype.

For several pairs, subtract **matching** entries, then average their losses. For example, three pairs with margins $0,1,2$ have losses $0.693147,0.313262,0.126928$, whose mean is $0.377779$. We average over pairs, not over response tokens. Preserve pair order when you split the batch.

### 5. From-scratch PyTorch: make the two branches visible

#### First, reproduce the hand calculation

Run this in the course's Python environment with PyTorch installed. Every intermediate here is a CPU `float32` tensor; `head.weight` is `(1, 2)` and `head.bias` is `(1,)`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_last = torch.tensor([[2., 1.], [1., 2.]])  # (2, 2): chosen, rejected
head = nn.Linear(2, 1)
with torch.no_grad():
    head.weight.copy_(torch.tensor([[-0.3, 0.4]]))
    head.bias.fill_(0.1)

rewards = head(hidden_last).squeeze(-1)          # (2,)
rewards.retain_grad()                           # only to inspect output gradients
margin = rewards[0] - rewards[1]                # () chosen minus rejected
loss = -F.logsigmoid(margin)                    # () numerically stable -log sigmoid
loss.backward()                                # ONE call, BOTH rewards involved

print("rewards:", rewards.detach())             # [-0.1000, 0.6000]
print("loss:", round(loss.item(), 6))            # 1.103186
print("reward gradients:", rewards.grad)        # [-0.6682, +0.6682]
print("weight gradient:", head.weight.grad)      # [[-0.6682, +0.6682]]

with torch.no_grad():
    head.weight -= head.weight.grad             # one SGD step, learning rate 1
    head.bias -= head.bias.grad                 # zero gradient from this pair loss
    after = head(hidden_last).squeeze(-1)
    after_loss = -F.logsigmoid(after[0] - after[1])
print("new rewards:", after)                     # [0.5682, -0.0682]
print("new loss:", round(after_loss.item(), 6))   # 0.424749
```

The subtraction keeps both rewards in the autograd graph. Do not use `.item()`, `.detach()`, or `torch.no_grad()` while building the training loss: those would break the path to the weights. We use them above only for initialization, reporting, and the manual optimizer update.

#### Use the existing `reward_model.py`, not a second model implementation

The hand calculation above isolates the new head and loss. For actual training, use **the class already in `code/src/llmre/preference/reward_model.py`**:

```python
from llmre.preference.reward_model import RewardModel, bradley_terry_loss
```

Its three pieces have different jobs:

| Existing piece | What it does | Used in the training loop as |
|---|---|---|
| `RewardModel.forward` | Runs the backbone, then calls `reward_from_hidden` | `rewards = model(ids, seq_lengths=lengths)` |
| `RewardModel.reward_from_hidden` | Turns hidden vectors into position scores and selects the last real score | Called inside `forward`; do not call it yourself when you have token ids |
| `bradley_terry_loss` | Combines chosen and rejected rewards into the pair loss | `bradley_terry_loss(chosen, rejected, reduction="none")` |

Here is the existing readout, with its docstring omitted:

```python
def reward_from_hidden(self, hidden, seq_lengths=None):
    scores = self.score(hidden).squeeze(-1)          # (B, T, 1) -> (B, T)
    if seq_lengths is None:
        return scores[:, -1]                          # (B,)
    idx = (seq_lengths - 1).clamp(min=0)             # (B,) last real index
    rows = torch.arange(scores.shape[0], device=scores.device)
    return scores[rows, idx]                          # (B,)
```

Notice the exact order: **score every position, then select the final real score**. Our earlier conceptual explanation selected the final hidden vector before applying the linear head. Both give the same number because the same linear head acts independently at every position. The code in this lesson now follows the repository's score-then-select implementation throughout. Intermediate position scores have no separate training targets and are not averaged into the answer reward.

The missing connection was the SFT backbone. `RewardModel` deliberately accepts a backbone returning hidden states, whereas `GPT.forward` returns vocabulary logits. We now provide this adapter **in that same file**:

```python
class GPTBackbone(nn.Module):
    """An independent copy of the course GPT, stopping before ``lm_head``.

    Input: ``idx`` (B, T) int64 on the copied parameters' device.
    Output: final hidden states (B, T, C), in the backbone's floating dtype.
    Right-padding is supported when the caller reads only real positions;
    causal attention prevents trailing pads from changing real-token states.
    """

    def __init__(self, sft):
        super().__init__()
        self.transformer = copy.deepcopy(sft.transformer)
        self.block_size = sft.cfg.block_size

    def forward(self, idx):
        T = idx.shape[1]
        if T > self.block_size:
            raise ValueError("sequence exceeds the SFT context length")
        pos = torch.arange(T, device=idx.device)
        tr = self.transformer
        token_embeddings = tr.wte(idx)             # (B, T, C): keep the input embedding
        position_embeddings = tr.wpe(pos)          # (T, C): broadcast over batch
        hidden = token_embeddings + position_embeddings
        hidden = tr.drop(hidden)                  # nn.Dropout on activations; deletes no weights
        for block in tr.h:
            hidden = block(hidden)
        return tr.ln_f(hidden)
```

It copies `sft.transformer`, runs the same computation as 06.1, and stops before `lm_head`. Copying the transformer also copies the input embeddings, including the weights that were tied to the SFT language head. The reward model has its own parameters; training it cannot change the original SFT model.

The factory method on the existing `RewardModel` connects that adapter to its existing scalar head:

```python
@classmethod
def from_sft(cls, sft):
    """Copy a course GPT's SFT backbone and add a new scalar head.

    No weights are shared with ``sft``. The language head stays on the
    original model; its tied input embeddings are retained in the copy.
    The returned model uses the SFT backbone's device and floating dtype.
    """
    parameter = next(sft.parameters())
    return cls(sft.cfg.n_embd, backbone=GPTBackbone(sft)).to(
        device=parameter.device, dtype=parameter.dtype
    )
```

So the actual construction is simply:

```python
rm = RewardModel.from_sft(sft)  # sft is your ALREADY TRAINED GPT from Module 14
```

The backbone adapter supplies the connection to SFT; the scalar-head model and Bradley-Terry loss are the existing course implementations. Import and reuse them in your training program.

#### Why `drop` is in `forward`: dropout keeps the embedding weights

In the original compact expression `tr.drop(tr.wte(idx) + tr.wpe(pos))`, **`drop` is the name of an `nn.Dropout` module**. It is not a command to remove `wte`. The original `GPT.__init__` from 06.1 created it with `drop=nn.Dropout(cfg.dropout)`; the reward backbone copies that module along with the other transformer components.

Read the expanded lines above in order: look up this batch's token embeddings, look up its position embeddings, add them, then apply dropout to the resulting **activation tensor**. The learned `wte.weight` table remains in the model and participates in reward-model fine-tuning. Dropout does not erase entries from that table or remove the embedding layer.

For example, suppose one combined embedding vector is `[1, 2, 3, 4]` and dropout probability is `p=0.5`. During training, one possible dropout result is `[0, 4, 0, 8]`: two activation values were temporarily zeroed, and the retained values were multiplied by `1 / (1 - 0.5) = 2`. This scaling preserves each activation's expected value. The original embedding weights are unchanged by applying dropout; the optimizer may later update them using gradients.

**Why run it in `forward`?** Each batch has new token ids and therefore new embedding activations. During training with nonzero dropout, each forward call samples a fresh mask for those activations. Creating or copying the module belongs in `__init__`; applying it to the current batch belongs in `forward`, just like applying attention or a linear layer. Applying dropout once in `__init__` would neither process future inputs nor provide the changing training masks.

| Model mode | What `tr.drop(hidden)` does |
|---|---|
| `model.train()`, `p > 0` | Temporarily masks activation values using a fresh random mask and rescales retained values |
| `model.eval()` | Returns the activations unchanged; dropout is disabled |
| Either mode, `p = 0` | Returns the activations unchanged; the course `GPTConfig` defaults to zero dropout |

The same distinction applies to the other operations: `__init__` creates or copies persistent layers and weights once; `forward` uses the current weights to compute new activations for each input batch. We keep the input embeddings. The language-model output head is excluded once when the wrapper keeps only `sft.transformer`; that has nothing to do with the per-forward dropout call.

#### The complete path from SFT to a trained grader

The training program below performs this whole stage:

```text
Your SFT weights + their original tokenizer
                    |
Human preference JSONL --> split by prompt into train / validation
                    |
Same Module 14 chat template and segment tokenization
                    |
Two full sequences per pair, right-padded with real lengths recorded
                    |
Copied SFT backbone + existing RewardModel scalar head
                    |
Shuffled epochs -> microbatches -> two rewards -> pair losses
                    |
Accumulate gradients -> clip -> set learning rate -> optimizer update
                    |
Held-out validation -> save best.pt and resumable last.pt
                    |
Reload best.pt -> frozen grader for policy training in 15.2
```

**The SFT weights are the starting point, not something this stage retrains from scratch.** We train both their copied backbone and the new head on preferences. We do not use the SFT next-token targets or response-only cross-entropy in this stage. The preference strings are inputs; their chosen/rejected ordering supplies the labels.

This is a complete **single-device, float32** baseline for the course's own GPT and byte-level BPE tokenizer. The batching, optimizer, evaluation, and checkpoint flow applies to a real run. Large-model execution additionally needs the memory and distributed techniques from Modules 7–9. Changing `vocab_size` does not make this loader compatible with a Hugging Face model: those architectures need their corresponding backbone and tokenizer adapter.

#### Input 1: export the SFT model and its actual tokenizer

At the end of your existing SFT program, with the trained `sft` and its `tokenizer` still in memory, save this small bundle convention once. The config fixes architecture; the state dict contains learned weights; the merge list preserves **the same token-to-id mapping** used for SFT:

```python
from dataclasses import asdict
import torch

torch.save({
    "config": asdict(sft.cfg),
    "model": sft.state_dict(),
    "tokenizer_merges": [[a, b, new_id]
                         for (a, b), new_id in tokenizer.merges.items()],
}, "sft_bundle.pt")
```

This is the input format of the reward-training entry point, not a claim that earlier checkpoints already use these key names. If you saved only `sft.state_dict()` earlier, first reconstruct the same `GPTConfig`, load those weights into `sft`, and recover its original tokenizer; then export this bundle. If SFT used LoRA, merge its learned updates into the GPT weights first. Do not export the frozen base model without its trained adapters.

This loader deliberately supports the course `BPETokenizer`. A tokenizer from another library must be saved and loaded through its own API and used with a compatible backbone, not reconstructed from these merge triples. Never train a fresh tokenizer for the reward stage.

#### Input 2: realistic preference records, with a held-out split

The repository includes six hand-written comparisons in `code/data/preferences-demo/train.jsonl` and three held-out-prompt examples in `code/data/preferences-demo/validation.jsonl`. These illustrate the workflow; they are not a human-labeling study or enough data to train a useful grader. Each line is one JSON object with plain string fields. Two training examples are:

```json
{"prompt": "Write a short, polite Slack message asking Maya to review my PR today. Do not imply that she is late.", "chosen": "Hi Maya, could you review my PR today when you have a moment? Thanks!", "rejected": "Hi Maya, I'm still waiting for your review of my PR; please get it done today so I can move forward."}
{"prompt": "Explain gradient accumulation to a programmer who understands mini-batches but only has GPU memory for two examples at a time. Include one concrete batch-size example.", "chosen": "Process two examples at a time. For each microbatch, divide its mean loss by four before calling backward, and keep the accumulated gradients. After four microbatches, take one optimizer step: that gives an effective batch of eight examples. Clear the gradients before the next group.", "rejected": "Gradient accumulation adds gradients from multiple microbatches before updating the parameters. It reduces the activation memory needed for a larger effective batch."}
```

The validation file contains different prompts, including:

```json
{"prompt": "Explain caching to a nontechnical coworker using one analogy, then mention one limitation.", "chosen": "A cache is like keeping a frequently used document on your desk instead of fetching it from a filing cabinet each time. It is faster, but your desk copy can become outdated when the original changes.", "rejected": "Caching stores previously computed results so later requests can reuse them. Cache entries may be indexed by keys and removed according to an expiration or eviction policy."}
```

These rows explain the file format; they are not enough data to train a useful grader. In a real dataset, collect or load human-labeled comparisons, clean them, group all comparisons for the same prompt, then split the **prompt groups**. Keep a final test split untouched while selecting the best checkpoint on validation. Repeated or paraphrased prompts need appropriate deduplication too. The program rejects exact train/validation prompt overlap, empty fields, identical answers, and sequences exceeding the SFT context limit. It does not silently cut off an answer while keeping its original preference label.

For each row, `encode_pairs` uses the existing `render_with_response_mask` from 14.1 to encode `user prompt + assistant answer`. It discards the response mask because the reward loss needs no per-token labels, but preserves **the exact segment tokenization from SFT**. The course template closes each turn with `<|end|>\n`, so its final real token may be the closing newline or a BPE piece; the readout consistently uses the final real position of that complete formatted sequence.

This baseline accepts one user prompt and one assistant response per side. If your data has system messages or conversation history, preserve that full shared history in both sides and extend the record formatter accordingly; do not discard it and change the task being judged.

#### Full runnable program: the actual training module

The following is the complete file **`code/src/llmre/preference/train.py`**. It imports the existing model and loss above. No training step is hidden behind a framework trainer, and there are no omitted lines inside the loop. You can read it here and run the checked-in module directly.

Read `collate_pairs`, `evaluate`, and the nested loops in `fit_reward_model` first. The tokenizer and checkpoint helpers keep the run reproducible and reloadable. `main` connects your files to that training function.

```python
"""Lesson 15.1: complete single-device reward-model training from an SFT GPT.

Run from code/: python -m llmre.preference.train --help
Uses the existing RewardModel and Bradley-Terry loss, not a demo replacement.
All model computation is float32. Token ids and lengths are int64. This is an
in-memory, single-device baseline; distributed/mixed-precision execution is a
separate scaling concern. Resume is supported at completed epoch boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.optim.schedule import cosine_warmup_lr
from llmre.preference.reward_model import RewardModel, bradley_terry_loss
from llmre.sft.chat_template import render_with_response_mask
from llmre.tokenizer.bpe import BPETokenizer


@dataclass
class RewardTrainConfig:
    epochs: int = 1
    pairs_per_batch: int = 4         # microbatch size in PAIRS, not sequences
    grad_accum_steps: int = 1
    lr: float = 1e-5
    warmup_steps: int = 0
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0


def read_jsonl(path):
    """Read {prompt, chosen, rejected} string records; no tensor/device yet."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def encode_pairs(rows, tokenizer, model_config):
    """Return [(chosen_ids, rejected_ids)], lists of ints on the Python side.

    Reuse exactly Module 14's segment-by-segment tokenization. Do not retrain
    the tokenizer or silently truncate answers and retain their old labels.
    """
    if not rows:
        raise ValueError("each data split needs at least one preference pair")
    pairs = []
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ("prompt", "chosen", "rejected")):
            raise ValueError("prompt, chosen, rejected must be nonempty strings")
        sequences = []
        for answer in (row["chosen"], row["rejected"]):
            messages = [{"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": answer}]
            ids, _ = render_with_response_mask(messages, tokenizer.encode)
            if not 0 < len(ids) <= model_config.block_size:
                raise ValueError("complete preference sequence exceeds context; filter/review it")
            if min(ids) < 0 or max(ids) >= model_config.vocab_size:
                raise ValueError("tokenizer ids do not fit the SFT vocabulary")
            sequences.append(ids)
        if sequences[0] == sequences[1]:
            raise ValueError("identical answers cannot express a strict preference")
        pairs.append(tuple(sequences))
    return pairs


def collate_pairs(pairs, device):
    """Right-pad chosen-first, rejected-second rows on device.

    Return ids (2P, T) int64 and real lengths (2P,) int64. Pad id 0 is a
    valid embedding id, not a real part of the response. With this causal GPT,
    trailing padding cannot influence the selected earlier real-token states.
    """
    sequences = [p[0] for p in pairs] + [p[1] for p in pairs]
    lengths = torch.tensor([len(s) for s in sequences], device=device)
    ids = torch.zeros((len(sequences), int(lengths.max())),
                      dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        ids[row, :len(sequence)] = torch.tensor(sequence, device=device)
    return ids, lengths


@torch.no_grad()
def evaluate(model, pairs, pairs_per_batch, device):
    """Held-out pair loss/accuracy as Python floats; no gradients retained."""
    model.eval()
    loss_sum, correct = 0.0, 0
    for start in range(0, len(pairs), pairs_per_batch):
        batch = pairs[start:start + pairs_per_batch]
        ids, lengths = collate_pairs(batch, device)
        chosen, rejected = model(ids, seq_lengths=lengths).chunk(2)
        loss_sum += bradley_terry_loss(chosen, rejected, reduction="sum").item()
        correct += (chosen > rejected).sum().item()
    return {"loss": loss_sum / len(pairs), "accuracy": correct / len(pairs)}


def tokenizer_from_merges(merges):
    """Reconstruct the SAME course BPE tokenizer from saved integer triples."""
    tokenizer = BPETokenizer()
    for a, b, new_id in merges:
        if new_id != len(tokenizer.vocab):
            raise ValueError("tokenizer merges must be in training order")
        tokenizer.merges[a, b] = new_id
        tokenizer.vocab[new_id] = tokenizer.vocab[a] + tokenizer.vocab[b]
    return tokenizer


def load_reward_model(path, device="cpu"):
    """Load a frozen scorer and its tokenizer; no SFT checkpoint needed.

    The scorer accepts ids (B,T) and optional lengths (B,), int64 on device,
    and returns float32 rewards (B,). Its whole backbone is in the checkpoint.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = RewardModel.from_sft(GPT(GPTConfig(**saved["model_config"])))
    model.load_state_dict(saved["model"])
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return model, tokenizer_from_merges(saved["tokenizer_merges"])


def fit_reward_model(sft, tokenizer, train_rows, val_rows, output_dir,
                     config=None, device="cpu", resume=None):
    """Train a separate scorer, save best/last checkpoints, return best frozen RM.

    sft is the already trained course GPT. tokenizer is its unchanged BPE
    tokenizer. Data rows are plain {prompt, chosen, rejected} strings.
    All tensors created here use device; parameters/activations use float32.
    The SFT model is not optimized. Resume restarts at the next saved epoch,
    using the same config and encoded data; interrupted partial epochs replay.
    """
    config = config or RewardTrainConfig()
    if min(config.epochs, config.pairs_per_batch, config.grad_accum_steps) < 1:
        raise ValueError("epochs, batch size, accumulation must be positive")
    if config.lr <= 0 or config.grad_clip <= 0 or config.weight_decay < 0:
        raise ValueError("invalid optimizer settings")
    train_pairs = encode_pairs(train_rows, tokenizer, sft.cfg)
    val_pairs = encode_pairs(val_rows, tokenizer, sft.cfg)
    if {r["prompt"].strip() for r in train_rows} & {r["prompt"].strip() for r in val_rows}:
        raise ValueError("split by prompt: train and validation prompts overlap")
    fingerprint = hashlib.sha256(json.dumps([train_pairs, val_pairs]).encode()).hexdigest()
    effective_pairs = config.pairs_per_batch * config.grad_accum_steps
    total_steps = config.epochs * math.ceil(len(train_pairs) / effective_pairs)
    if not 0 <= config.warmup_steps <= total_steps:
        raise ValueError("warmup_steps must be between zero and total updates")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume is None and any((output_dir / name).exists() for name in ("best.pt", "last.pt")):
        raise FileExistsError("choose a new output directory or resume the existing run")

    torch.manual_seed(config.seed)
    model = RewardModel.from_sft(sft).to(device=device, dtype=torch.float32)
    model.requires_grad_(True)                    # full backbone + head fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    shuffle_rng = torch.Generator().manual_seed(config.seed)
    start_epoch, step, best_loss = 0, 0, float("inf")
    merges = [[a, b, i] for (a, b), i in tokenizer.merges.items()]
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if (saved["train_config"] != asdict(config) or
                saved["data_fingerprint"] != fingerprint or
                saved["model_config"] != asdict(sft.cfg) or
                saved["tokenizer_merges"] != merges):
            raise ValueError("resume requires the same config, tokenizer, and encoded data")
        if not (output_dir / "best.pt").exists():
            raise ValueError("resume in the original run directory containing best.pt")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, step, best_loss = saved["epoch"], saved["step"], saved["best_loss"]
        shuffle_rng.set_state(saved["shuffle_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available() and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])

    print("initial validation:", evaluate(model, val_pairs, config.pairs_per_batch, device))
    for epoch in range(start_epoch, config.epochs):
        model.train()                            # re-enable training-time dropout
        order = torch.randperm(len(train_pairs), generator=shuffle_rng).tolist()
        train_loss_sum, train_correct = 0.0, 0
        for start in range(0, len(order), effective_pairs):
            group = [train_pairs[i] for i in order[start:start + effective_pairs]]
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(group), config.pairs_per_batch):
                batch = group[micro_start:micro_start + config.pairs_per_batch]
                ids, lengths = collate_pairs(batch, device)
                rewards = model(ids, seq_lengths=lengths)   # (2P,)
                chosen, rejected = rewards.chunk(2)       # each (P,)
                pair_losses = bradley_terry_loss(chosen, rejected, reduction="none")
                if not torch.isfinite(pair_losses).all():
                    raise FloatingPointError("non-finite reward loss")
                # Sum/N gives each PAIR equal weight, even in a short last group.
                loss = pair_losses.sum() / len(group)
                loss.backward()                 # both branches; accumulate gradients
                train_loss_sum += pair_losses.detach().sum().item()
                train_correct += (chosen.detach() > rejected.detach()).sum().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                           error_if_nonfinite=True)
            lr = cosine_warmup_lr(step, config.warmup_steps, total_steps,
                                  config.lr, config.lr * 0.1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.step()                    # ONE update per effective pair group
            step += 1

        val = evaluate(model, val_pairs, config.pairs_per_batch, device)
        if not math.isfinite(val["loss"]):
            raise FloatingPointError("non-finite validation loss")
        improved = val["loss"] < best_loss
        best_loss = min(best_loss, val["loss"])
        print({"epoch": epoch + 1, "step": step,
               "train_loss": train_loss_sum / len(train_pairs),
               "train_accuracy": train_correct / len(train_pairs), "validation": val})
        saved = {"model": model.state_dict(), "model_config": asdict(sft.cfg),
                 "tokenizer_merges": merges, "optimizer": optimizer.state_dict(),
                 "epoch": epoch + 1, "step": step, "best_loss": best_loss,
                 "train_config": asdict(config), "data_fingerprint": fingerprint,
                 "shuffle_rng": shuffle_rng.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        # Write then rename: an interrupted save leaves the prior checkpoint intact.
        for name in (["best.pt", "last.pt"] if improved else ["last.pt"]):
            temporary = output_dir / (name + ".tmp")
            torch.save(saved, temporary)
            temporary.replace(output_dir / name)

    best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    return model.eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", required=True, help="SFT bundle: config, model, tokenizer_merges")
    parser.add_argument("--train", required=True, help="training preferences JSONL")
    parser.add_argument("--validation", required=True, help="held-out preferences JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pairs-per-batch", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--resume", help="last.pt from the original output directory")
    args = parser.parse_args()
    saved = torch.load(args.sft, map_location="cpu", weights_only=True)
    sft = GPT(GPTConfig(**saved["config"]))
    sft.load_state_dict(saved["model"])
    tokenizer = tokenizer_from_merges(saved["tokenizer_merges"])
    config = RewardTrainConfig(epochs=args.epochs, pairs_per_batch=args.pairs_per_batch,
                               grad_accum_steps=args.grad_accum_steps,
                               lr=args.lr, warmup_steps=args.warmup_steps)
    fit_reward_model(sft, tokenizer, read_jsonl(args.train), read_jsonl(args.validation),
                     args.output, config, args.device, args.resume)


if __name__ == "__main__":
    main()
```

The two loop levels have distinct meanings. An **effective batch** is the group of pairs contributing to one optimizer update. A **microbatch** is the part that fits in memory at once. With `pairs_per_batch=2` and `grad_accum_steps=3`, one update uses up to six preference pairs, processing up to four sequences in each forward call. We clear gradients once for that group, backpropagate each microbatch, clip once, and update once. The last group can be shorter; dividing each microbatch's summed losses by the actual group size gives every pair equal weight.

The right-padding value is id 0, an existing vocabulary entry. It is excluded by the **readout lengths**, not by a next-token loss mask. This works for our causal GPT because real positions cannot attend to padding placed after them. A different backbone may also need an attention mask and explicit position ids; `seq_lengths` alone is not a general substitute for those.

Training metrics describe predictions made while weights were changing and dropout was active. Validation uses `model.eval()` and no gradient tracking, sums losses and correct comparisons over the whole held-out split, and divides by the actual pair count. The next training epoch switches back to `model.train()`.

#### Run it against your SFT checkpoint

From `code/`, after installing the course package with `python -m pip install -e ".[dev]"`, run:

```bash
python -m llmre.preference.train --sft sft_bundle.pt --train data/preferences-demo/train.jsonl --validation data/preferences-demo/validation.jsonl --output checkpoints/reward-run --device cpu --epochs 1 --pairs-per-batch 2 --grad-accum-steps 4 --lr 0.00001
```

For a GPU run, use `--device cuda` and choose microbatch size according to available memory. The example learning rate and epoch count are starting settings to validate, not universal best values. The program prints an initial validation baseline and then training and validation loss/accuracy after each epoch. Watch held-out performance: increasing the training margin alone does not prove a better grader.

The output directory contains:

| File | Purpose |
|---|---|
| `best.pt` | Full scorer weights with the lowest validation pair loss, plus model config and tokenizer state |
| `last.pt` | Latest completed epoch, including model and AdamW states, step, config, shuffle state, and torch RNG states |

If a run stops, repeat the **same command and settings** with `--resume checkpoints/reward-run/last.pt` added. Resume starts at the next completed-epoch boundary; work from an interrupted partial epoch is replayed. It requires the original run directory with its `best.pt`, the same encoded data, tokenizer, and training config. For reproducibility, keep the same software and hardware setup; cross-device bit-for-bit continuation is not promised. This trainer uses PyTorch AdamW checkpoints rather than the earlier custom optimizer's checkpoint helper because their optimizer-state formats differ.

#### Reload the trained reward model and score a new answer

This is a separate program; it needs the reward checkpoint, not the original SFT model:

```python
import torch
from llmre.preference.train import load_reward_model
from llmre.sft.chat_template import render_with_response_mask

rm, tokenizer = load_reward_model("checkpoints/reward-run/best.pt", device="cpu")
messages = [
    {"role": "user", "content": "Write a short, polite Slack message asking Maya to review my PR today. Do not imply that she is late."},
    {"role": "assistant", "content": "Hi Maya, would you be able to take a look at my PR today? Thank you!"},
]
token_ids, _ = render_with_response_mask(messages, tokenizer.encode)
ids = torch.tensor([token_ids], dtype=torch.long)  # (1, T), CPU
with torch.no_grad():
    reward = rm(ids)                              # (1,), CPU float32; no padding
print(reward.item())                              # a learned score, not a probability
```

This call needs **one answer**. Pairing was how we supervised the grader during training. To compare two new answers, score each in its own full prompt-and-answer sequence and subtract their scores, just as before.

At this point the original SFT policy still generates text through `lm_head`. The saved reward model is a separate, frozen scorer. Lesson 15.2 supplies the next loop: the policy generates responses, this scorer grades them, and PPO updates the policy. Completing reward-model training alone does not improve the text-generating policy.

### 6. Under the hood: what this costs and where it fits

A pair requires processing **two sequences** through the backbone. That can be one batched call or two calls; "two scores" does not require two separate Python calls. The transformer still does its usual work at all token positions. Selecting the final output saves no transformer layers; it changes where we attach supervision.

The scalar head is small. With hidden width $C=768$, `nn.Linear(768, 1)` has 768 weights and one bias: 769 parameters. We also avoid the large vocabulary projection used for next-token logits. Full fine-tuning still needs backbone activations for backpropagation, and batching both answers together requires activation memory for both rows.

After reward-model training, freeze **this whole scorer** for the PPO stage in 15.2. At that stage the policy generates a new answer and the scorer evaluates its prompt-and-answer sequence. It can score that answer alone: pairs were needed to teach the scorer, not to make every later scoring call. The policy retains its own `lm_head` and is trained separately.

**Hardware and scale.** Minimum for the tiny verification tests: an ordinary CPU laptop, no GPU, and zero GPU-hours; the targeted suite took under 10 seconds in the checked environment. Allow about 20 minutes to prepare the files and inspect a first run, plus the training time for your actual checkpoint and data. For a real model, this baseline's float32 parameters, gradients, and two Adam moment buffers alone require roughly 16 bytes per parameter: about 16 GB for a 1B model, before activations and temporary buffers. The original SFT copy, tokenized dataset, and checkpoint loading also need host RAM. Choose GPU capacity after measuring a microbatch; reducing microbatch size and accumulating gradients helps activation memory but does not shrink optimizer state. For a larger model, apply the mixed-precision and sharding techniques from Modules 7–9, or the adapter training from 14.3, with a compatible implementation. Runtime and GPU-hours depend on checkpoint size, context length, pair count, and hardware; this tiny CPU check does not estimate a useful large-model run.

**Hands-on, about 20 minutes after SFT plus training time:** run the hand calculation, export your existing SFT bundle, prepare training and held-out JSONL, and run the full training entry point. Confirm that checkpoints appear, reload `best.pt`, and score a new answer. For a wiring check without your own checkpoint, run `python -m pytest tests/test_reward_training.py tests/test_preference.py` from `code/`; those tests use explicitly synthetic tiny models, not a pretrained SFT checkpoint. The tests verify SFT isolation, padded versus unpadded scores, reload, gradient accumulation, and completed-epoch resume.

**Sources checked 2026-11-13.** The implementation and end-to-end training tests were checked with Python 3.14 and PyTorch 2.14.0+cpu; the tests exercise the checked-in training module. The historical SFT-to-scalar-scorer recipe and pairwise loss are publicly documented in [InstructGPT, section 3.5](https://arxiv.org/html/2203.02155v1#S3.SS5). For the final-real-token readout, see [the GPT-2 sequence-classification documentation](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2ForSequenceClassification). For the production counterpart after understanding this implementation, read [TRL's reward-model preprocessing and loss explanation](https://huggingface.co/docs/trl/reward_trainer#looking-deeper-into-the-training-method); its trainer wraps the same independent scores and Bradley-Terry comparison. The toy hidden vectors and head-only update above are an educational example.

## Common mistakes

- **Reading the reward at a pad token.** With right-padded batches, `scores[:, -1]` may land on padding. Pass `seq_lengths` so the reward is read at the last *real* token.
- **Expecting absolute rewards to mean something.** Only margins are trained; a reward of `+4.2` is meaningless on its own. Never threshold on absolute reward across different prompts.
- **Using `log(sigmoid(x))`.** Numerically unstable; use `F.logsigmoid`.
- **Training the reward model and policy to convergence independently and forgetting they drift apart.** As the policy improves in RLHF it produces completions the reward model never saw — *distribution shift* (lesson 15.4). Reward models are periodically refreshed.

## Exercise

Implement a batched Bradley-Terry loss that also returns the **preference accuracy**: the fraction of pairs where the model already ranks chosen above rejected ($r_w > r_l$). This is the standard metric reported alongside reward-model loss.

<details><summary>Hint</summary>

Accuracy does not need the sigmoid at all — it only needs the sign of the margin.

</details>

<details><summary>Stronger hint</summary>

`(reward_chosen > reward_rejected).float().mean()`.

</details>

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

<div class="callout paper"><p><strong>Read:</strong> <a href="#/papers/index">InstructGPT (Ouyang et al. 2022)</a> — the reward-model section defines exactly the Bradley-Terry loss above and the "reward = value at the final token" convention; and <a href="#/papers/index">Anthropic HH-RLHF (Bai et al. 2022)</a> for how helpful/harmless preference data is collected at scale. Read the reward-modeling subsections closely; skim the RL details (that is lesson 15.2). Both are core Module 15 readings.</p></div>

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

Continue to [15.2 · Policy gradient, advantage, PPO](lessons/module-15/lesson-02.md).
