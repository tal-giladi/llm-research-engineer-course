# 15.1 · Bradley-Terry & the reward model

<div class="prereq">
<p><strong>Prerequisites:</strong> supervised fine-tuning from <a href="#/lessons/module-14/lesson-01">14.1 · Instruction data & chat templates</a> and the response-only loss of <a href="#/lessons/module-14/lesson-02">14.2 · Loss masking</a>; the assembled model and its hidden states from <a href="#/lessons/module-06/lesson-01">06.1 · Assembling GPT-2</a>; the log-softmax / cross-entropy machinery from <a href="#/lessons/module-01/lesson-04">01.4</a> and <code>llmre.evaluation.metrics</code>.</p>
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

- **Humans judge once, offline:** a fixed dataset of ~100k pairs ("Yes is better than No").
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

1. **Policy generates** an answer to a prompt: "Is fire hot?" → "No".
2. **Reward model scores** prompt + answer, one forward pass: $r = 1.48$.
3. Another try: "Yes" → $r = 2.14$.
4. **The policy is updated** to make the higher-scoring answer more likely (lesson 15.2 shows how).

Yes, that is extra compute: a grading pass for every generated answer. It is the price of not needing a human in the loop, and it is paid **only during training**, like a teacher grading homework during the course. After the course the student answers alone: the policy has learned to say "Yes" by itself, so nobody needs to score it any more.

### Why a separate model, and not a second head on the same GPT?

Because the grader must **stay fixed while the student changes**. In stage 4 the policy's weights are updated every step. If the scalar head sat on the policy's own backbone, every update to the policy would also change the grader, and the policy could learn to make its own grader happy instead of writing better answers (this is **reward hacking**, lesson 15.4). With two separate copies, the grader is frozen and the policy cannot touch it.

### Your intuition is where the field went next

The two-model setup is heavy: two big models in memory, a grading pass per answer. Researchers asked the same question you are asking, and there are two well-known answers, both later in this module:

- **DPO** (lesson 15.3, publicly documented, Rafailov et al. 2023): uses the same preference pairs but trains the policy **directly**, with no reward model and no generation during training. The math in this lesson is still the foundation: DPO is derived from the Bradley-Terry loss.
- **Generative reward models / LLM-as-a-judge** (reasonable industry practice): the grader is an ordinary GPT with its normal `lm_head` that *writes* its verdict as text ("Answer A is better" or "Score: 8/10"). No special head at all.

Scalar-head reward models are still widely used and publicly documented (InstructGPT; Llama 2 trained separate helpfulness and safety reward models), so we build one here: it is the cleanest way to see what "learning human preferences" means numerically.

## What *is* a reward model, mechanically?

### 1. Intuition

A reward model is just a language model (Module 6) with its head swapped. The GPT of [06.1](lessons/module-06/lesson-01.md) ends in an `lm_head` of shape $(C \to V)$ that scores every one of the $V$ vocabulary tokens at every position. A reward model throws that away and bolts on a **scalar head** of shape $(C \to 1)$: it reads one position's hidden state and outputs a single number — the reward for the whole sequence.

Which position? The reward summarizes the *entire* prompt+completion, so we read the reward off the hidden state of the **last token**. Because attention is causal (a token attends only to itself and earlier tokens — [05.x](lessons/module-05/lesson-01.md)), the final token's hidden state is the only one that has "seen" everything, so it is the natural place to pool the sequence into one score. This is what InstructGPT and HH-RLHF do.

### 2. Mathematics

Let the backbone map token ids $x_{1:T}$ to hidden states $h_1, \dots, h_T \in \mathbb{R}^C$. The scalar head is a linear map $w \in \mathbb{R}^{C}$, $b \in \mathbb{R}$:

$$
r(x) = w^\top h_T + b
$$

That is the entire reward model on top of the backbone: one dot product per sequence.

### 3. Full numerical example: one preference pair, from start to finish

This walks through **everything** that happens to one preference pair, with numbers small enough to check by hand. Nothing is assumed: every operation (dot product, softmax, sigmoid, log, gradient) is explained the first time it appears, and every number is computed in front of you. Numbers are rounded to 4 decimals; the last digit can differ by 1 because of rounding.

> **What is real and what is shrunk.** The *steps* below are exactly the steps a real reward model runs (InstructGPT, HH-RLHF). The *sizes* are shrunk so you can follow them: each token is described by **2 numbers** instead of 768, the vocabulary has **9 tokens** instead of 50,257, there is **one** attention layer instead of 12, and we skip LayerNorm and the MLP. A real model does the same thing with bigger tables and more layers.

#### Step 0 — Where the reward model comes from

We start from a model that has already been through both earlier stages:

1. **Pre-training** (Modules 6–9): the model read a lot of text and learned to predict the next token. In doing so it learned a list of numbers for every token (its **embedding**) and learned which earlier words to pay attention to.
2. **SFT** (Module 14): the model was trained on example conversations, so it knows the chat format `<|user|> question <|assistant|> answer <|end|>`.

**Tokens and embeddings.** The model cannot read letters. Text is first cut into **tokens** (words or word pieces), each token has a number called its **id**, and each id has a row of numbers called its **embedding**. The embedding is the model's learned description of that token. Ours has 2 numbers per token ($C = 2$, $C$ = "hidden size"). These are the values our model ended up with after pre-training + SFT; we take them as given:

| id | token | embedding $e$ |
|---:|---|---|
| 0 | `<\|user\|>` | (0.0, 0.5) |
| 1 | `Is` | (0.5, 0.0) |
| 2 | `fire` | (1.0, 0.5) |
| 3 | `hot` | (0.5, 1.0) |
| 4 | `?` | (0.0, 0.0) |
| 5 | `<\|assistant\|>` | (0.0, 1.0) |
| 6 | `Yes` | (2.0, 1.0) |
| 7 | `No` | (0.0, −1.0) |
| 8 | `<\|end\|>` | (1.0, 0.0) |

**What a "head" is.** A language model has two parts. The **backbone** (embeddings + all the transformer layers) reads the text and produces, for each position, a list of $C$ numbers that describes it: the **hidden state**. The **head** is the last layer, sitting on top of the backbone, that turns a hidden state into the output you actually want. Same backbone, different head, different job. A **scalar head** is a head whose output is one single number (a "scalar" is just one number, as opposed to a list).

**Swapping the head.** The SFT model ends in an `lm_head`: a table of $2 \times 9 = 18$ weights that turns the final 2 numbers into 9 scores, one per possible next token ("what word comes next?"). A reward model does not want to predict the next word; it wants **one number that says how good the answer is**. So:

- we **delete** the `lm_head`;
- we add a **scalar head**: 2 weights $w = (w_1, w_2)$ and 1 bias $b$ (3 numbers in total). Its job: take the 2 numbers that describe the whole conversation and turn them into one score.

The new head has not learned anything yet, so it starts with small random values. Ours starts at

$$
w = (-0.3,\; 0.4), \qquad b = 0.1
$$

Everything else (embeddings, attention) is copied unchanged from the SFT model. The model still *understands* the text; it just does not yet know what "a better answer" means. Teaching it that is what training does.

#### Step 1 — The preference record

One row of preference data, $(x, y_w, y_l)$:

- prompt $x$: "Is fire hot?"
- chosen $y_w$: "Yes" (a human labeler picked this one as better)
- rejected $y_l$: "No"

The only label is "Yes is better than No". No score, no reason.

#### Step 2 — Build two full sequences and turn them into ids

The reward model never scores the answer alone; it always reads **prompt + answer** in the chat format, because "Yes" is only a good answer *to this question*. So we build two sequences that are identical except for the answer:

| position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| chosen | `<\|user\|>` | `Is` | `fire` | `hot` | `?` | `<\|assistant\|>` | **`Yes`** | `<\|end\|>` |
| rejected | `<\|user\|>` | `Is` | `fire` | `hot` | `?` | `<\|assistant\|>` | **`No`** | `<\|end\|>` |
| chosen ids | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
| rejected ids | 0 | 1 | 2 | 3 | 4 | 5 | 7 | 8 |

In PyTorch the two rows go in together as one tensor `idx` of shape `(2, 8)` (2 sequences, 8 tokens each), dtype `int64` (whole numbers). $T = 8$ is the sequence length.

#### Step 3 — Replace every id with its embedding

Each id is looked up in the embedding table above. Now each sequence is 8 rows of 2 numbers; together a `(2, 8, 2)` float tensor. The two sequences are identical everywhere except position 7: `Yes` = (2.0, 1.0) in the chosen one, `No` = (0.0, −1.0) in the rejected one.

#### Step 4 — Let the last token read the whole sequence (attention)

We need **one** description of the whole conversation, so we take it from the **last** token, `<|end|>`. Attention is how that token reads everything before it. It works in four small steps: (a) score how relevant each token is, (b) turn the scores into percentages, (c) mix the tokens using those percentages, (d) add the token's own embedding back.

In a real model each token's embedding is first multiplied by three learned matrices to make its *query* (what am I looking for?), *key* (what do I contain?) and *value* (what do I pass on?). In our tiny model those matrices are the identity (multiply by 1), so query, key and value are all just the embedding itself. That keeps the arithmetic readable; the steps are the same.

**4a. Score each token with a dot product.** A **dot product** of two lists of numbers means: multiply the matching numbers and add the results. For example $(1, 2) \cdot (3, 4) = 1 \times 3 + 2 \times 4 = 11$. A big dot product means "these two point the same way", which attention reads as "relevant".

The last token's query is $q = e_{\text{end}} = (1.0, 0.0)$. Its score for each position $s$ is

$$
\text{score}_s = \frac{q \cdot e_s}{\sqrt{C}} = \frac{1.0 \times e_{s,1} + 0.0 \times e_{s,2}}{\sqrt 2}
$$

where $e_{s,1}$ and $e_{s,2}$ are the first and second numbers of token $s$'s embedding. Dividing by $\sqrt{C} = \sqrt 2 = 1.4142$ is a fixed rule of attention that keeps scores from growing large when $C$ is large. Because $q = (1.0, 0.0)$, each score is simply the token's first number divided by 1.4142. For `fire`: $(1.0 \times 1.0 + 0.0 \times 0.5) / 1.4142 = 0.7071$.

**4b. Turn scores into percentages (softmax).** Softmax does two things:

1. Take $e^{\text{score}}$ for every score ($e = 2.71828$). This makes every value positive and makes bigger scores stand out more. For example $e^{0} = 1$, $e^{0.7071} = 2.0281$.
2. Divide each of those by their total, so they add up to 1. These are the **attention weights** $a_s$: what fraction of its attention the last token gives to each position.

| position | token | first number $e_{s,1}$ | score | $e^{\text{score}}$ |
|---:|---|---:|---:|---:|
| 1 | `<\|user\|>` | 0.0 | 0.0000 | 1.0000 |
| 2 | `Is` | 0.5 | 0.3536 | 1.4241 |
| 3 | `fire` | 1.0 | 0.7071 | 2.0281 |
| 4 | `hot` | 0.5 | 0.3536 | 1.4241 |
| 5 | `?` | 0.0 | 0.0000 | 1.0000 |
| 6 | `<\|assistant\|>` | 0.0 | 0.0000 | 1.0000 |
| 7 | `Yes` (chosen) | 2.0 | 1.4142 | 4.1133 |
| 7 | `No` (rejected) | 0.0 | 0.0000 | 1.0000 |
| 8 | `<\|end\|>` | 1.0 | 0.7071 | 2.0281 |

The seven positions that both sequences share add up to $1 + 1.4241 + 2.0281 + 1.4241 + 1 + 1 + 2.0281 = 9.9044$. So the totals are:

- chosen: $9.9044 + 4.1133 = 14.0177$
- rejected: $9.9044 + 1.0000 = 10.9044$

Each weight is $e^{\text{score}}$ divided by its sequence's total. For example `Yes` gets $4.1133 / 14.0177 = 0.2934$, and `<|user|>` in the chosen sequence gets $1 / 14.0177 = 0.0713$.

**4c. Mix the tokens.** Multiply each token's embedding by its weight (both of its numbers), then add everything up:

| token | chosen: $a_s$ | chosen: $a_s \times e_s$ | rejected: $a_s$ | rejected: $a_s \times e_s$ |
|---|---:|---|---:|---|
| `<\|user\|>` (0.0, 0.5) | 0.0713 | (0.0000, 0.0357) | 0.0917 | (0.0000, 0.0459) |
| `Is` (0.5, 0.0) | 0.1016 | (0.0508, 0.0000) | 0.1306 | (0.0653, 0.0000) |
| `fire` (1.0, 0.5) | 0.1447 | (0.1447, 0.0723) | 0.1860 | (0.1860, 0.0930) |
| `hot` (0.5, 1.0) | 0.1016 | (0.0508, 0.1016) | 0.1306 | (0.0653, 0.1306) |
| `?` (0.0, 0.0) | 0.0713 | (0.0000, 0.0000) | 0.0917 | (0.0000, 0.0000) |
| `<\|assistant\|>` (0.0, 1.0) | 0.0713 | (0.0000, 0.0713) | 0.0917 | (0.0000, 0.0917) |
| answer: `Yes` (2.0, 1.0) / `No` (0.0, −1.0) | **0.2934** | (0.5869, 0.2934) | 0.0917 | (0.0000, −0.0917) |
| `<\|end\|>` (1.0, 0.0) | 0.1447 | (0.1447, 0.0000) | 0.1860 | (0.1860, 0.0000) |
| **sum** | 1.0000 | **(0.9778, 0.5744)** | 1.0000 | **(0.5026, 0.2694)** |

Notice what happened: in the chosen sequence the last token put 29% of its attention on `Yes`, more than on any other token, because `Yes` has a large first number. In the rejected sequence `No` got only 9%.

**4d. Add the token's own embedding back (the residual connection).** Every transformer layer adds its input back to its output, so the token keeps its own information. $e_{\text{end}} = (1.0, 0.0)$:

$$
h_w = (1.0 + 0.9778,\; 0.0 + 0.5744) = (1.9778,\; 0.5744)
$$

$$
h_l = (1.0 + 0.5026,\; 0.0 + 0.2694) = (1.5026,\; 0.2694)
$$

$h$ is called the **hidden state**. $h_w$ is the model's 2-number summary of "Is fire hot? Yes", and $h_l$ of "Is fire hot? No". They differ only because the last token read a different answer.

#### Step 5 — The scalar head turns each summary into one number

The reward is $r = w \cdot h + b$: a dot product with the head's weights, plus the bias. With $w = (-0.3, 0.4)$ and $b = 0.1$:

$$
r_w = (-0.3 \times 1.9778) + (0.4 \times 0.5744) + 0.1 = -0.5933 + 0.2298 + 0.1 = -0.2636
$$

$$
r_l = (-0.3 \times 1.5026) + (0.4 \times 0.2694) + 0.1 = -0.4508 + 0.1078 + 0.1 = -0.2430
$$

The untrained head gives the **wrong** answer a slightly higher score ($-0.2430$ is bigger than $-0.2636$). That is expected: the head is random, so it has no idea yet what humans prefer.

#### Step 6 — How likely does the model think "Yes" is the better answer?

**The margin** is the gap between the two rewards:

$$
m = r_w - r_l = -0.2636 - (-0.2430) = -0.0206
$$

**The sigmoid** turns any number into a probability between 0 and 1: $\sigma(m) = \dfrac{1}{1 + e^{-m}}$. A margin of 0 gives exactly 0.5 (a coin flip); a big positive margin gives close to 1; a big negative margin gives close to 0. Step by step:

1. $-m = 0.0206$
2. $e^{0.0206} = 1.0208$
3. $1 + 1.0208 = 2.0208$
4. $1 / 2.0208 = 0.4949$ (more exactly 0.49485)

So the model says "the chosen answer is better with probability 49.5%": it leans, slightly, the wrong way.

#### Step 7 — The loss: one number for "how wrong"

The loss is $\mathcal{L} = -\log \sigma(m)$, where $\log$ is the natural logarithm. Why this form:

- if the model gives the true winner probability 1, then $-\log 1 = 0$: no penalty;
- probability 0.5 gives $-\log 0.5 = 0.6931$;
- the smaller the probability, the bigger the penalty ($-\log 0.1 = 2.30$, $-\log 0.01 = 4.61$).

Here:

$$
\mathcal{L} = -\log(0.49485) = 0.7035
$$

A bit above 0.6931 (the coin-flip value), because the model leans the wrong way.

#### Step 8 — The gradient: which way should each weight move?

The **gradient** of the loss with respect to a number answers: "if I increase this number by a tiny amount, how much does the loss change, and in which direction?" Training then moves each number the opposite way, to make the loss smaller. We need it for the head's weights $w_1$, $w_2$ and the bias $b$. We get there in two links.

**Link 1 — loss vs. margin.** Calculus gives the slope of $-\log\sigma(m)$ as $\sigma(m) - 1$:

$$
\frac{\partial \mathcal{L}}{\partial m} = 0.4949 - 1 = -0.5051 \quad (\text{more exactly } -0.5052)
$$

You can check it without calculus: nudge the margin by $0.001$ from $-0.0206$ to $-0.0196$ and recompute the loss; it goes from $0.703500$ to $0.702995$, a change of $-0.000505$. Dividing by the nudge, $-0.000505 / 0.001 = -0.505$. Same number. The minus sign means: **making the margin bigger makes the loss smaller**.

**Link 2 — margin vs. each head weight.** Write the margin out in full:

$$
m = r_w - r_l = (w_1 h_{w,1} + w_2 h_{w,2} + b) - (w_1 h_{l,1} + w_2 h_{l,2} + b) = w_1 (h_{w,1} - h_{l,1}) + w_2 (h_{w,2} - h_{l,2})
$$

The two $b$'s cancel. Plugging in the hidden states:

$$
m = w_1 \times (1.9778 - 1.5026) + w_2 \times (0.5744 - 0.2694) = w_1 \times 0.4752 + w_2 \times 0.3050
$$

So raising $w_1$ by a tiny amount raises the margin by $0.4752$ times that amount; raising $w_2$ raises it by $0.3050$ times; and changing $b$ does nothing.

**Multiply the two links** (this is the **chain rule**: how loss reacts to margin, times how margin reacts to the weight):

$$
\frac{\partial \mathcal{L}}{\partial w_1} = -0.5052 \times 0.4752 = -0.2401, \qquad
\frac{\partial \mathcal{L}}{\partial w_2} = -0.5052 \times 0.3050 = -0.1540, \qquad
\frac{\partial \mathcal{L}}{\partial b} = 0
$$

$\partial\mathcal{L}/\partial b = 0$ is the earlier statement "rewards are only defined up to a constant", appearing as a number: adding the same amount to both rewards never changes the gap, so the bias never learns anything.

**The backbone gets a signal too.** The same chain rule says how the loss reacts to each number of the two hidden states: $\partial\mathcal{L}/\partial h_w = -0.5052 \times w = (0.1516, -0.2021)$ and $\partial\mathcal{L}/\partial h_l = +0.5052 \times w = (-0.1516, 0.2021)$. Moving opposite to these pushes $h_w$ in the direction of $w$ (raising $r_w$) and $h_l$ away from it (lowering $r_l$). From there the signal continues back into the attention and into the `Yes` and `No` embeddings. In full fine-tuning those are updated too; to keep the example readable we update only the head.

#### Step 9 — Update the weights

**Gradient descent:** new value = old value − learning rate × gradient. The learning rate $\eta$ controls the step size. We use a large one, $\eta = 1.0$, so the change is visible after a single step (real reward-model training uses around $10^{-5}$):

$$
w_1 \leftarrow -0.3 - 1.0 \times (-0.2401) = -0.0599
$$

$$
w_2 \leftarrow 0.4 - 1.0 \times (-0.1540) = 0.5540
$$

$$
b \leftarrow 0.1 - 1.0 \times 0 = 0.1 \;(\text{unchanged})
$$

Both weights went **up**, because both gradients were negative.

**Run the pair again.** Steps 2–4 give the same hidden states as before (we did not change the backbone); only Step 5 changes:

$$
r_w = (-0.0599 \times 1.9778) + (0.5540 \times 0.5744) + 0.1 = -0.1185 + 0.3182 + 0.1 = 0.2997
$$

$$
r_l = (-0.0599 \times 1.5026) + (0.5540 \times 0.2694) + 0.1 = -0.0900 + 0.1493 + 0.1 = 0.1592
$$

$$
m = 0.2997 - 0.1592 = 0.1405, \qquad \sigma(0.1405) = \frac{1}{1 + e^{-0.1405}} = \frac{1}{1 + 0.8689} = 0.5351, \qquad \mathcal{L} = -\log 0.5351 = 0.6254
$$

After **one** step the reward model already ranks "Yes" above "No", and the loss fell from 0.7035 to 0.6254.

#### Step 10 — Keep training

Repeating Steps 5–9 on the same pair:

| step | $w$ | $r_w$ | $r_l$ | margin | $P(y_w \succ y_l)$ | loss |
|---:|---|---:|---:|---:|---:|---:|
| 0 | (−0.3000, 0.4000) | −0.2636 | −0.2430 | −0.0206 | 0.4948 | 0.7035 |
| 1 | (−0.0599, 0.5540) | 0.2997 | 0.1592 | 0.1405 | 0.5351 | 0.6254 |
| 2 | (0.1610, 0.6958) | 0.8181 | 0.5294 | 0.2887 | 0.5717 | 0.5592 |
| 3 | (0.3646, 0.8264) | 1.2958 | 0.8705 | 0.4253 | 0.6047 | 0.5030 |
| 4 | (0.5524, 0.9469) | 1.7365 | 1.1852 | 0.5513 | 0.6344 | 0.4550 |
| 5 | (0.7262, 1.0584) | 2.1442 | 1.4763 | 0.6678 | 0.6610 | 0.4140 |

Every step the margin grows and the loss falls. The *absolute* rewards drift (from about −0.25 to about +2); only the gap between them means anything. In real training this loop runs over ~100k different pairs, not one, so the model learns a general sense of "better", not just "Yes beats No for this question". Real training also processes many pairs at once (a batch) and averages their losses.

#### Step 11 — What the trained reward model is used for

After training, the reward model takes **one** sequence (prompt + any answer) and returns **one** number, using exactly Steps 2–5 above. It is then frozen and used only as the automatic grader during RL training (lesson 15.2): the policy writes an answer, the reward model scores it, and that number is the reward the policy is trained to increase. When RL training ends, the reward model is put aside and only the policy is shipped (see "The big picture" above).

#### Verify every number

```bash
py -c "
import torch, torch.nn.functional as F, math
E={'u':(0,.5),'Is':(.5,0),'fire':(1,.5),'hot':(.5,1),'?':(0,0),'a':(0,1),'Yes':(2,1),'No':(0,-1),'end':(1,0)}
P=['u','Is','fire','hot','?','a']
def h(seq):
    X=torch.tensor([E[t] for t in seq],dtype=torch.float32); q=X[-1]
    a=F.softmax(X@q/math.sqrt(2),0); return X[-1]+a@X
hw,hl=h(P+['Yes','end']),h(P+['No','end'])
w=torch.tensor([-.3,.4],requires_grad=True); b=torch.tensor(.1,requires_grad=True)
for step in range(6):
    m=(w@hw+b)-(w@hl+b); L=-F.logsigmoid(m); w.grad=b.grad=None; L.backward()
    print(step,[round(v,4) for v in w.tolist()],round(m.item(),4),round(L.item(),4),[round(v,4) for v in w.grad.tolist()],b.grad.item())
    with torch.no_grad(): w-=w.grad
"
```

### 4. Tensor shapes / dtype / device

- Token ids `idx`: `(B, T)`, `int64`.
- Backbone hidden states `hidden`: `(B, T, C)`, float, on the model's device.
- Scalar head `nn.Linear(C, 1)` applied to `hidden`: `(B, T, 1)`, squeezed to `(B, T)`.
- Reward at the last position: `(B,)`, one float per sequence.

For a batch that mixes sequence lengths (right-padded), you don't want the reward read at a pad token, so you index each row at `seq_lengths - 1` instead of a blanket `-1`.

### 5. From-scratch PyTorch

#### What a scalar head is, in code

A scalar head is one ordinary linear layer with **one output**:

```python
score = nn.Linear(C, 1)    # weight shape (1, C), bias shape (1,)  ->  C + 1 numbers in total
```

`nn.Linear(C, 1)` computes exactly $w \cdot h + b$ from the numerical example: it multiplies the $C$ numbers of a hidden state by $C$ weights, adds them up, and adds the bias. Compare it with the head the SFT model already has:

| head | layer | input per position | output per position | job |
|---|---|---|---|---|
| `lm_head` (SFT model) | `nn.Linear(C, V)` | $C$ numbers | $V$ numbers, one per vocabulary token | "which token comes next?" |
| `score` (reward model) | `nn.Linear(C, 1)` | $C$ numbers | 1 number | "how good is this conversation?" |

For GPT-2 small, $C = 768$ and $V = 50{,}257$: the `lm_head` is $768 \times 50{,}257 \approx 38.6$M weights; the scalar head is $768 + 1 = 769$.

#### Attaching it to the SFT model, step by step

Our `GPT` from [06.1](lessons/module-06/lesson-01.md) runs: embeddings → transformer blocks → final LayerNorm `ln_f` → `lm_head`. We want everything **up to and including `ln_f`**, and then our scalar head instead of `lm_head`.

**Step A — Load the SFT model.** It is a normal `GPT` with SFT weights loaded.

**Step B — Make a separate copy for the reward model** with `copy.deepcopy`. The reward model will be trained, and we do not want that training to change the SFT model, because the SFT model is also the starting point of the *policy* in lesson 15.2. Two independent copies, two jobs.

**Step C — Wrap the copy so it stops before `lm_head`.** We write a small module that runs the same steps as `GPT.forward` but returns the hidden states `(B, T, C)` instead of the logits `(B, T, V)`.

Why not just `del model.lm_head`? Because of **weight tying** (06.1): in our GPT, `lm_head.weight` and the token-embedding table `wte.weight` are *the same tensor*. The embedding table is still needed to read the input, so we don't delete anything; we simply never call `lm_head`. The wrapper below keeps only `transformer` (which holds `wte`), so `lm_head` is not part of the reward model at all.

**Step D — Put the backbone and the scalar head together.** `RewardModel` (from `llmre.preference.reward_model`) holds the backbone and creates `self.score = nn.Linear(C, 1)`. Its `forward` runs the backbone, applies the head at every position, and returns the value at the last token.

```python
import copy
import torch
import torch.nn as nn
from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.preference.reward_model import RewardModel, bradley_terry_loss

# Step A: the SFT model (tiny config so it runs on CPU in a second)
cfg = GPTConfig(vocab_size=9, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
sft = GPT(cfg)
# real use: sft.load_state_dict(torch.load("sft.pt"))

# Step C: the GPT up to ln_f, stopping BEFORE lm_head
class GPTBackbone(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.transformer = gpt.transformer        # wte, wpe, blocks, ln_f -- but NOT lm_head

    def forward(self, idx):                       # idx: (B, T) int64
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)  # (T,)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))  # (B, T, C)
        for block in self.transformer.h:
            x = block(x)                          # (B, T, C)
        return self.transformer.ln_f(x)           # (B, T, C) hidden states, no lm_head

# Step B + D: separate copy, wrapped, with a new scalar head on top
rm = RewardModel(n_embd=cfg.n_embd, backbone=GPTBackbone(copy.deepcopy(sft)))

print(rm.score)                  # Linear(in_features=16, out_features=1, bias=True)
print(rm.score.weight.shape)     # torch.Size([1, 16])
```

The head starts with random values (PyTorch's default initialization for `nn.Linear`); the backbone starts with the SFT weights.

**Step E — Score the two sequences and train.** Same ids as the numerical example ("Is fire hot?" + Yes / No):

```python
chosen   = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 8]])   # (1, 8)  ... Yes <|end|>
rejected = torch.tensor([[0, 1, 2, 3, 4, 5, 7, 8]])   # (1, 8)  ... No  <|end|>

print(rm.backbone(chosen).shape)     # torch.Size([1, 8, 16])  hidden states
print(rm(chosen), rm(rejected))      # two (1,) rewards, random at first

opt = torch.optim.AdamW(rm.parameters(), lr=1e-2)   # trains the head AND the backbone copy
for step in range(50):
    loss = bradley_terry_loss(rm(chosen), rm(rejected))
    opt.zero_grad()
    loss.backward()
    opt.step()

print(loss.item())                   # close to 0
print(rm(chosen) > rm(rejected))     # tensor([True])

torch.save(rm.state_dict(), "reward_model.pt")       # the trained reward model
```

`rm.parameters()` includes the backbone copy, so this is full fine-tuning: head and backbone both learn. To train only the head (as in the hand example), pass `rm.score.parameters()` to the optimizer instead.

#### "How do I take the head out later?" You don't: you throw the whole grader away

The scalar head is never removed and never moved onto the model you ship. During RL training there are two separate models:

| model | built from | head | used for | after RL training |
|---|---|---|---|---|
| policy | the SFT model (`sft`) | `lm_head`, untouched | writing answers, token by token | **shipped** to users |
| reward model | a copy of the SFT model (`rm`) | scalar head `score` | grading each answer during training | **put aside**, not shipped |

In code, the RL loop of lesson 15.2 looks like this (sketch):

```python
rm.eval()
for p in rm.parameters():
    p.requires_grad_(False)          # the grader is frozen during RL

# one RL step
full_ids = sft.generate(prompt_ids, max_new_tokens=20)     # policy writes; returns prompt + answer ids
reward = rm(full_ids)                                       # grader scores prompt + answer
# ... update sft's weights so high-reward answers become more likely (15.2) ...

# after RL training
torch.save(sft.state_dict(), "final_chat_model.pt")         # this is the product
# rm is not saved with it and never runs when users chat
```

That is why we made a `deepcopy` in Step B: the policy keeps changing during RL, and the grader must not change with it.

> **Production equivalent.** In Hugging Face Transformers this whole recipe is `AutoModelForSequenceClassification.from_pretrained(sft_path, num_labels=1)`: it loads the SFT backbone, drops the language head, and adds a fresh linear layer with one output (named `score` in Llama-style models, without a bias) read at the last non-pad token. Same mechanism as above.

#### The two pieces of `RewardModel`, in full

The reward-reading logic: the head applied at every position, then the value at the last token is picked:

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
