# 17.4 · Logit distillation: forward vs reverse KL &amp; on-policy distillation

<div class="callout key"><p><strong>Added 2026-10-04 by the weekly curriculum review</strong> (<a href="#/research/CHANGELOG">changelog</a>). This extends 17.3; nothing earlier is replaced. The distillation you met in <a href="#/lessons/module-17/lesson-03">17.3</a> (sample the teacher's answers, then plain-SFT the student on them) is still correct and still used. This lesson adds the two knobs modern pipelines turn on top of it: <em>what</em> the student matches (the teacher's whole next-token distribution, through a forward or a reverse KL) and <em>where</em> it matches it (on the teacher's text, or on the student's own).</p></div>

<div class="prereq">
<p><strong>Prerequisites:</strong> next-token cross-entropy and the response mask from <a href="#/lessons/module-14/lesson-02">14.2 · Loss masking &amp; packing</a>; the KL penalty and the log-ratio estimator from <a href="#/lessons/module-15/lesson-02">15.2 · Policy gradient, advantage, PPO</a>; sampling rollouts from a policy in <a href="#/lessons/module-16/lesson-01">16.1 · From PPO to GRPO</a>; sequence-level distillation of R1 into small dense models from <a href="#/lessons/module-17/lesson-03">17.3 · DeepSeek-R1 case study</a>.</p>
<p><strong>You will learn:</strong> why "SFT on teacher samples" throws most of the teacher's information away; the forward KL $\mathrm{KL}(p_T\,\|\,p_S)$ and reverse KL $\mathrm{KL}(p_S\,\|\,p_T)$ between teacher and student next-token distributions, their gradients, and why one <strong>covers</strong> modes and the other <strong>seeks</strong> one; the generalized JSD that interpolates them (GKD); <strong>on-policy distillation</strong>, where the student generates and the teacher grades every token; a from-scratch implementation in <code>llmre.reasoning.distill</code> with two exact toy experiments; and what production trainers (TRL's <code>DistillationTrainer</code>) do and what it costs.</p>
<p><strong>Why this matters for ML:</strong> almost every small open-weight model you will use was distilled from a bigger one. Qwen3's small models are documented as trained by distillation from the flagship models: first off-policy on teacher responses, then on-policy with logit (KL) matching, which the report says beat RL at about a tenth of the GPU-hours. If you train or fine-tune small models, choosing the divergence and the rollout policy is a design decision you will make, and the wrong choice fails silently.</p>
</div>

<div class="hw">
<p><strong>Hardware track:</strong> CPU-only is enough. Both toy experiments are exact computations on a 20-bin and an 8-token problem; the whole test file runs in about a second, using a few MB of memory, 0 GPU-hours. Real logit distillation of a 0.5B student from a 1.5B teacher (the TRL quick-start shape) needs one GPU with 24 GB or more for both models plus generation. That is not needed here.</p>
</div>

## 1. Intuition: a sampled token is one bit of a whole distribution

### 1.1 The foundational technique (recap of 17.3)

In [17.3](lessons/module-17/lesson-03.md), DeepSeek ran R1 to generate reasoning traces and then fine-tuned small dense models on them with ordinary SFT cross-entropy. This is **sequence-level knowledge distillation** (Kim &amp; Rush 2016). It works, it is simple, and it needs only the teacher's *text*: you can distill from a model you can sample from but cannot inspect.

### 1.2 The problem at modern scale

Two problems, one per knob.

**What is matched.** At each position the teacher has a full distribution over ~150K tokens. Sampling throws all of it away except one token. Suppose the teacher, after "The answer is", puts 0.55 on "12", 0.40 on "twelve" and 0.05 elsewhere. SFT on one sample tells the student "12 is right, everything else is wrong". The teacher knew more: "twelve" is almost as good. This extra information is often called **dark knowledge**; matching soft teacher distributions is the original knowledge-distillation idea of Hinton et al. (2015). Matching the full distribution passes it on, and it gives a dense training signal at every position.

**Where it is matched.** The student is trained on prefixes the *teacher* wrote. At inference the student continues *its own* prefixes. A small student makes mistakes the teacher never makes. After a mistake, the student is in a prefix that never appeared in training, so it never learned how to recover. Errors compound. This is **exposure bias**: the train/inference mismatch of any model trained only on prefixes it did not write (teacher forcing).

### 1.3 The modern technique

- **Logit distillation:** at every position, minimize a divergence between the teacher's and the student's next-token distributions.
- **On-policy distillation:** let the student generate the prefixes, then have the teacher score every token the student produced. The student sees its own mistakes and the teacher tells it, token by token, what to do there. Agarwal et al. (2023) call this **GKD** ("learning from self-generated mistakes").

It sits between SFT and RL. Like RL (16.x), the data is on-policy. Like SFT, the signal is dense: a full target distribution at every token, instead of one scalar reward per sequence.

<div class="callout key"><p>SFT on teacher samples = forward KL, estimated with one sample per position, on the teacher's prefixes. Logit distillation removes the "one sample". On-policy distillation removes "the teacher's prefixes". The divergence direction (forward or reverse) decides what a student that cannot copy the teacher exactly gives up.</p></div>

## 2. Mathematics

Notation. Vocabulary size $V$. A prefix (prompt plus tokens so far) $x_{<t}$. Teacher next-token distribution $p_T(v \mid x_{<t})$, student $p_S(v \mid x_{<t}) = \mathrm{softmax}(z)_v$ where $z \in \mathbb{R}^V$ are the student's logits at that position. Below we drop the conditioning and write $p_T(v)$, $p_S(v)$.

### 2.1 Forward KL: the student must cover the teacher

$$
\mathrm{KL}(p_T \,\|\, p_S) = \sum_{v=1}^{V} p_T(v)\,\big(\log p_T(v) - \log p_S(v)\big)
$$

The sum is weighted by the **teacher**. A token the teacher likes and the student ignores ($p_T(v)$ large, $p_S(v) \to 0$) costs $p_T(v)\cdot(-\log p_S(v)) \to \infty$. A token the student likes and the teacher ignores costs nothing directly. So the student is forced to put mass everywhere the teacher does. This is **mode covering** (also called "mean seeking").

**SFT is a special case.** If the teacher is one-hot on the observed token $y$ ($p_T(y) = 1$), the sum has one term and $\log p_T(y) = 0$:

$$
\mathrm{KL}(\text{one-hot}_y \,\|\, p_S) = -\log p_S(y),
$$

which is exactly the cross-entropy of 14.2. Sequence-level distillation is the Monte-Carlo version: sampling $y \sim p_T$ and averaging $-\log p_S(y)$ gives $\mathbb{E}_{y\sim p_T}[-\log p_S(y)] = \mathrm{KL}(p_T\|p_S) + H(p_T)$, and the teacher entropy $H(p_T)$ does not depend on the student. Same expected gradient, much higher variance.

**Gradient.** With respect to the student logits $z$:

$$
\frac{\partial\, \mathrm{KL}(p_T\|p_S)}{\partial z_v} = p_S(v) - p_T(v).
$$

The same "prediction minus target" as cross-entropy, but the target is the teacher's soft distribution.

### 2.2 Reverse KL: the student must not invent

$$
\mathrm{KL}(p_S \,\|\, p_T) = \sum_{v=1}^{V} p_S(v)\,\big(\log p_S(v) - \log p_T(v)\big)
$$

Now the sum is weighted by the **student**. A token the student likes and the teacher ignores ($p_S(v) > 0$, $p_T(v) \to 0$) costs $p_S(v)\cdot(-\log p_T(v)) \to \infty$. A teacher mode the student ignores costs nothing directly. So a student that cannot fit everything picks the modes it can fit well and drops the rest. This is **mode seeking**. MiniLLM (Gu et al. 2023) argues this is the better default for generative LMs: a student that puts mass where the teacher has none produces text the teacher would never write.

**Gradient.** Let $\ell_v = \log p_S(v) - \log p_T(v)$ and $K = \mathrm{KL}(p_S\|p_T) = \sum_v p_S(v)\ell_v$. Then

$$
\frac{\partial\, \mathrm{KL}(p_S\|p_T)}{\partial z_v} = p_S(v)\,\big(\ell_v - K\big).
$$

Read it as a policy gradient. The per-token "reward" is $-\ell_v$ (how much more the teacher likes $v$ than the student does), and $K$ is the baseline. Tokens whose log-ratio is worse than average are pushed down, better than average are pushed up. If you sample one token $v \sim p_S$ instead of summing over the vocabulary, $\ell_v$ is exactly the one-sample "k1" KL estimator of 15.2, now used as a per-token negative reward. That sampled form is how some on-policy distillation recipes plug into an RL trainer.

### 2.3 Generalized JSD: a dial between the two

GKD uses a mixture $M = (1-\beta)\,p_S + \beta\,p_T$ and

$$
\mathrm{JSD}_\beta(p_T, p_S) = \beta\,\mathrm{KL}(p_T\|M) + (1-\beta)\,\mathrm{KL}(p_S\|M), \qquad \beta \in [0,1].
$$

Each term compares a distribution to a mixture that contains it, so it is bounded: $\mathrm{JSD}_\beta \le$ a constant (at $\beta = 0.5$ it is at most $\ln 2$). Taken literally the formula is $0$ at $\beta = 0$ and $\beta = 1$, but $\mathrm{JSD}_\beta/\beta \to \mathrm{KL}(p_T\|p_S)$ as $\beta \to 0$ and $\mathrm{JSD}_\beta/(1-\beta) \to \mathrm{KL}(p_S\|p_T)$ as $\beta \to 1$. TRL therefore special-cases the ends: `beta=0` means forward KL and `beta=1` means reverse KL (**publicly documented** in TRL's docs; TRL's `DistillationTrainer` defaults to `beta=1.0`, reverse KL).

### 2.4 On-policy vs off-policy: which prefixes the loss is averaged over

For a whole completion, the loss is the per-token divergence $D$ averaged over prefixes. The only difference between the two regimes is the distribution the prefixes come from:

$$
\mathcal{L}_{\text{off}} = \mathbb{E}_{x \sim \mathcal{D}}\; \mathbb{E}_{y \sim p_T(\cdot\mid x)} \Big[\tfrac{1}{|y|}\textstyle\sum_t D\big(p_T(\cdot\mid x, y_{<t}),\, p_S(\cdot\mid x, y_{<t})\big)\Big]
$$

$$
\mathcal{L}_{\text{on}} = \mathbb{E}_{x \sim \mathcal{D}}\; \mathbb{E}_{y \sim \mathrm{sg}[p_S](\cdot\mid x)} \Big[\tfrac{1}{|y|}\textstyle\sum_t D\big(p_T(\cdot\mid x, y_{<t}),\, p_S(\cdot\mid x, y_{<t})\big)\Big]
$$

$x$ is a prompt from dataset $\mathcal{D}$, $y$ a completion, $D$ the forward KL, reverse KL or $\mathrm{JSD}_\beta$. $\mathrm{sg}[\cdot]$ is stop-gradient: in GKD the student's sampling is treated as fixed data, and gradients flow only through $p_S$ inside $D$. GKD also mixes the two with a fraction $\lambda$ of on-policy batches (TRL's experimental `GKDTrainer` exposes it as `lmbda`).

## 3. Numerical example by hand

Three-token vocabulary. Teacher $p_T = (0.6,\ 0.3,\ 0.1)$, student $p_S = (0.3,\ 0.3,\ 0.4)$.

**Forward KL.** Term by term, $p_T(v)\ln\frac{p_T(v)}{p_S(v)}$:

- token 1: $0.6 \ln(0.6/0.3) = 0.6\ln 2 = 0.4159$
- token 2: $0.3 \ln(0.3/0.3) = 0$
- token 3: $0.1 \ln(0.1/0.4) = -0.1\ln 4 = -0.1386$

$\mathrm{KL}(p_T\|p_S) = 0.4159 + 0 - 0.1386 = 0.2773$ nats.

**Reverse KL.** $p_S(v)\ln\frac{p_S(v)}{p_T(v)}$:

- token 1: $0.3\ln(0.3/0.6) = -0.3\ln 2 = -0.2079$
- token 2: $0$
- token 3: $0.4\ln(0.4/0.1) = 0.4\ln 4 = 0.5545$

$\mathrm{KL}(p_S\|p_T) = 0.3466 = 0.5\ln 2$ nats. The two directions are not equal: KL is not symmetric. Reverse KL is larger here because the student's biggest mistake is putting 0.4 on token 3, which the teacher barely likes, and reverse KL weights mistakes by the student's own mass.

**Gradients on the student logits.**

- Forward: $p_S - p_T = (-0.3,\ 0,\ +0.3)$. Raise token 1, lower token 3, leave token 2.
- Reverse: $\ell = (-\ln 2,\ 0,\ \ln 4) = (-0.6931,\ 0,\ 1.3863)$, $K = 0.3466$, so $p_S(\ell - K) = (0.3\cdot(-1.0397),\ 0.3\cdot(-0.3466),\ 0.4\cdot 1.0397) = (-0.3119,\ -0.1040,\ +0.4159)$.

Under reverse KL, token 2 also moves (its logit goes *up*, because a negative gradient means gradient descent raises it), even though $p_S(2) = p_T(2)$: its log-ratio $0$ is better than the average $K$. Both gradient vectors sum to zero, as any gradient through a softmax must. The test `test_gradients_match_closed_forms` checks these numbers against autograd.

## 4. Tensor shapes, dtype, device

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `teacher_logits` | `(B, T, V)` | bf16 in production, `float64` in our tests | computed under `no_grad`; we `.detach()` inside the loss |
| `student_logits` | `(B, T, V)` | same | the only thing with gradient |
| `mask` | `(B, T)` | bool/int | 1 on completion tokens, 0 on prompt and padding (the 14.2 response mask) |
| per-token divergence | `(B, T)` | float32 at least | `reduction="none"` |
| loss | `()` | float32 | masked mean over completion tokens |

Teacher and student **must share a tokenizer**: the KL is a sum over the same $V$ token ids. Distilling across tokenizers needs extra alignment machinery, outside this lesson.

Device: both models live on the training GPUs in a simple setup. In larger setups the teacher is served separately and returns log-probs (TRL has an async variant that queries a teacher over HTTP). In our code everything is CPU.

## 5. From-scratch PyTorch

`code/src/llmre/reasoning/distill.py` holds the divergences and two toy experiments. The divergences are a few lines each:

```python
def forward_kl(teacher_logits, student_logits, mask=None, reduction="mean"):
    log_pt = F.log_softmax(teacher_logits.detach(), dim=-1)   # (..., V)
    log_ps = F.log_softmax(student_logits, dim=-1)            # (..., V)
    kl = (log_pt.exp() * (log_pt - log_ps)).sum(-1)           # (...)  sum over vocab
    return kl if reduction == "none" else _masked_mean(kl, mask)

def reverse_kl(teacher_logits, student_logits, mask=None, reduction="mean"):
    log_pt = F.log_softmax(teacher_logits.detach(), dim=-1)
    log_ps = F.log_softmax(student_logits, dim=-1)
    kl = (log_ps.exp() * (log_ps - log_pt)).sum(-1)           # weighted by the STUDENT
    return kl if reduction == "none" else _masked_mean(kl, mask)
```

Work in log space (`log_softmax`, never `log(softmax(...))`) so tiny probabilities do not become `log(0)`. `generalized_jsd` computes $\log M$ with a `logsumexp` of $\log(1-\beta) + \log p_S$ and $\log\beta + \log p_T$ for the same reason.

### 5.1 Experiment A: mode covering vs mode seeking

`bimodal_teacher_probs()` is a teacher over 20 bins with two bumps at bins 4 and 15. The student, `discretized_gaussian_logits(mu, log_sigma, V)`, has only two parameters, so it can make **one** bump. This is the small-student problem in miniature: the student cannot copy the teacher, so the divergence decides what it gives up.

```python
from llmre.reasoning.distill import bimodal_teacher_probs, fit_unimodal_student
tp = bimodal_teacher_probs()
for kind in ["forward", "reverse", "jsd:0.5"]:
    r = fit_unimodal_student(tp, kind)
    print(kind, round(r["mu"], 2), round(r["sigma"], 2), round(r["probs"][7:12].sum().item(), 3))
```

Output (deterministic; produced by running the code):

```
forward 9.02 14.64 0.268
reverse 4.0 1.0 0.005
jsd:0.5 4.0 1.0 0.005
```

- **Forward KL** spreads into a nearly flat bump centred between the modes. It puts **26.8%** of its mass on bins 7–11, where the teacher has **0.24%**. For a language model, that is the student generating text the teacher would essentially never produce.
- **Reverse KL** moves onto the nearer mode and matches it exactly ($\mu = 4$, $\sigma = 1$, the teacher's own bump). Its loss is $\ln 2 = 0.693$: it gets half of the teacher's behaviour right and ignores the other half. No mass in the gap.
- $\mathrm{JSD}_{0.5}$ behaves like reverse KL here.

Neither is "correct". Forward KL is safer when you need the student to keep every behaviour the teacher has (diversity, coverage, pass@$k$). Reverse KL is safer when you need each sample to be good (precision, pass@1). For generation, the field has mostly moved toward reverse KL or JSD (MiniLLM, TRL's default); the 2026 study discussed in section 8 shows the choice interacts with the rollout policy.

### 5.2 Experiment B: on-policy vs off-policy

To make on-/off-policy exact, the "language model" is a **bigram chain**: a `(V, V)` table where row $s$ is the next-token distribution after token $s$. For such a chain we do not need to sample rollouts: the probability of being at each token at step $t$ is $d_{t+1} = d_t P$ (`state_visitation`), so the expected loss over all rollouts is an exact weighted sum. Real systems sample; the expectation is the same.

- The **teacher** (`make_teacher_chain`) cycles $0\to1\to2\to3\to0$ with probability 0.9999 and almost never visits the "off-road" tokens 4–7. From an off-road token it returns to 0 with probability 0.9. It knows how to recover, but its own text almost never shows that.
- The **student** (`slipping_student_logits`) can never be perfect: a fraction `slip = 0.05` of its mass is always spread uniformly. That stands in for limited capacity or sampling temperature. It will step off-road sometimes, whatever it learns.

`distill_markov` trains the same student, with the same steps and learning rate, twice: once weighting each row's divergence by the **teacher's** visitation (off-policy) and once by the **student's own**, recomputed every step (on-policy).

```python
from llmre.reasoning.distill import make_teacher_chain, distill_markov
t, start = make_teacher_chain()
for kind in ["forward", "reverse"]:
    for policy in ["off", "on"]:
        r = distill_markov(t, start, policy=policy, kind=kind)
        print(kind, policy, round(r["eval_on_policy_rkl"], 3),
              round(r["off_road_time"], 3), round(r["recovery_prob"], 2))
```

```
forward off 0.366 0.05 0.13
forward on 0.294 0.031 0.77
reverse off 0.327 0.046 0.13
reverse on 0.26 0.028 0.75
```

Columns: reverse KL to the teacher measured on the **student's own** rollouts (what a user experiences), fraction of time spent off-road, and probability of returning to token 0 from an off-road token (teacher 0.9, untrained $1/8 = 0.125$).

Off-policy training never puts weight on the off-road rows, so the student's recovery probability stays at chance, $0.125$. Every slip turns into a long detour. On-policy training visits those rows because the *student* visits them, so it learns to recover (0.75–0.77). It spends about 40% less time lost, and its on-policy KL is about 20% lower. Both runs fit the rows they train on; the difference is entirely *which rows*. That is exposure bias, and its fix, in eight tokens.

<div class="callout warn"><p>This toy is built to show the mechanism cleanly: a teacher that almost never errs, and a student that always slips. With a student that <em>can</em> copy the teacher exactly (set <code>slip=0</code> and train long enough), both regimes converge to the teacher and the gap closes. On-policy distillation helps when the student cannot be perfect, which is the usual case for small students, but it is not a law. Section 8 gives a 2026 study where its advantage was smaller than expected.</p></div>

### 5.3 Unit tests

`code/tests/test_distill.py` (12 tests) checks: the hand-example values; agreement with `torch.nn.functional.kl_div`; that forward KL with a one-hot teacher equals `F.cross_entropy` (the foundational SFT loss); both closed-form gradients against autograd; that the teacher gets no gradient; the $\mathrm{JSD}_\beta$ endpoints and limits; prompt masking; and both experiments' qualitative results.

```bash
cd code && PYTHONPATH=src python -m pytest -q tests/test_distill.py
```

## 6. Under the hood: what production trainers do, and the cost

TRL's `DistillationTrainer` (**publicly documented** in TRL's docs) does three things you now recognise:

1. **Generate** completions from the student for each prompt, optionally with vLLM, exactly like an online RL trainer (16.1).
2. **Score** them with one teacher forward pass: prefill only, no generation, which is much cheaper than generating.
3. **Loss**: the generalized JSD of section 2.3 over completion tokens (`beta`, default `1.0` = reverse KL), computed in **chunks** so the full `(B, T, V)` logits tensor is never in memory at once.

**Why chunking is needed.** Logits are big. With Qwen-size vocabulary $V = 151{,}936$ and a batch of $8 \times 4096 = 32{,}768$ tokens, one fp32 logits tensor is $32{,}768 \times 151{,}936 \times 4 \approx 19.9$ GB. Logit distillation needs the teacher's and the student's, plus the student's gradient. Chunking the vocabulary projection and the divergence over positions keeps peak memory at one chunk. That is the same trick as fused or chunked cross-entropy in pretraining.

**Offline logit distillation.** Storing full teacher distributions for a dataset is impossible at that size, so offline pipelines usually store only the teacher's **top-$k$ log-probs** per position and renormalise (**reasonable industry practice**; the exact $k$ and handling of the tail vary). Off-policy forward KL works with stored logits. On-policy distillation cannot be precomputed: the teacher must be live, because the prefixes do not exist until the student writes them.

**Compute compared with RL.** Per step, on-policy distillation costs about the same as an RLVR step: student generation dominates. The difference is the signal: RL gets one scalar reward per sequence, distillation gets a full target distribution at every token. The Qwen3 report states that on-policy distillation of Qwen3-8B reached better AIME and coding scores than RL from the same starting point, using about 1,800 GPU-hours versus 17,920 for RL (**company claim**, Qwen3 Technical Report, Table 21). It needs a stronger teacher to exist. RL does not, which is why frontier labs still need RL for their largest models and use distillation to pass the result down.

## 7. Side by side: when to use which

| | Sequence-level KD (17.3) | Off-policy logit KD | On-policy distillation |
|---|---|---|---|
| Needs from teacher | text only (API is enough) | log-probs on its own text | live log-probs on the student's text |
| Signal per token | 1 sampled token | full distribution | full distribution |
| Prefixes seen | teacher's | teacher's | student's |
| Fixes exposure bias | no | no | yes |
| Cost per step | SFT | SFT + teacher logits (or stored top-$k$) | student generation + teacher prefill |
| Typical role | first stage; teacher behind an API; different tokenizers | cheap first stage when logits are available | final stage for a small student |

**When you would still use the original.** Sequence-level KD remains the right tool when you only have the teacher's text (a closed API), when teacher and student use different tokenizers, or as a cheap first stage. DeepSeek's R1 distillation (17.3) and Qwen3's first, off-policy distillation stage both start this way. A common pipeline is: sequence-level or off-policy KD first, so the student's own samples are sensible, then on-policy distillation. Running on-policy from a random student wastes teacher compute grading gibberish.

## Experiment: switch the knobs

Run in `code/` with `PYTHONPATH=src python`:

1. In Experiment A, start the student in the empty gap: `fit_unimodal_student(tp, "reverse", init_mu=9.5)`. **Expected:** reverse KL gets stuck at a wide, flat solution (loss about 2.87, far above $\ln 2$), because from the gap no small move reaches a mode. Start it at `init_mu=13.0` and it finds the *other* mode ($\mu = 15$). Forward KL ends at essentially the same wide solution (loss 0.881) from any start. Reverse KL is non-convex in the student's parameters.
2. In Experiment B, set `slip=0.0` and `steps=2000`. **Expected:** both on-policy KLs drop to about 0.001 and the gap almost disappears, because a student that can copy the teacher eventually stops leaving the road and rarely needs to recover.
3. Raise `slip` to `0.1`. **Expected:** the gap in recovery probability stays wide; both on-policy KLs rise, because a sloppier student is more often lost.

## Common mistakes

- **Swapping the KL arguments.** `F.kl_div(input, target)` computes $\mathrm{KL}(\text{target}\,\|\,\text{input})$ with `input` in log space. Passing the student as `target` silently turns your forward KL into reverse KL. Test against a hand example.
- **Using `log(softmax(z))`.** Underflows for small probabilities and gives `-inf`, so `0 * -inf = nan`. Use `log_softmax`.
- **Letting gradient flow into the teacher.** Compute teacher logits under `torch.no_grad()` and detach them; otherwise you are fine-tuning the teacher toward the student.
- **Forgetting the mask.** Prompt tokens are given, not generated. Including them in the distillation loss trains the student to imitate the teacher's opinion of the *user's* text (the same mistake as 14.2).
- **Mismatched tokenizers.** Vocabulary index 1234 must mean the same string for both models. Otherwise the KL is meaningless even though it runs.
- **Temperature mismatch.** If you sample student rollouts at temperature $\tau$ but compute the loss at temperature 1 (or soften only one side), the target and the trained distribution differ. Pick one temperature and apply it to both logits (TRL uses its `temperature` setting for both sampling and the loss).
- **Claiming on-policy always wins.** It fixes exposure bias; it does not guarantee a better student. Measure on your task.

## Exercise

Implement the **sampled** on-policy reverse-KL update (the form that plugs into an RL trainer) for the bigram toy, and check it against the exact gradient. For one row $s$: sample $N$ tokens $v_i \sim p_S(\cdot\mid s)$, use reward $r_i = -(\log p_S(v_i) - \log p_T(v_i))$, and a REINFORCE loss $-\frac{1}{N}\sum_i (r_i - \bar r)\,\log p_S(v_i)$ with $\bar r$ the mean reward. Show that as $N$ grows, its gradient approaches the exact gradient $p_S(\ell - K)$ from section 2.2.

<details><summary>Hint</summary>

Detach the rewards. The exact gradient is `torch.autograd.grad(reverse_kl(t_row, z), z)`. Compare cosine similarity of the two gradients for $N = 10, 100, 10{,}000$.

</details>

<details><summary>Stronger hint</summary>

$\nabla_z \mathrm{KL}(p_S\|p_T) = \mathbb{E}_{v\sim p_S}\big[\ell_v \nabla_z \log p_S(v)\big]$, because the extra term $\mathbb{E}_{v\sim p_S}[\nabla_z \log p_S(v)]$ is zero. Minimizing $\mathrm{KL}$ means following $-\ell_v$ as a reward. Subtracting a baseline does not change the expectation.

</details>

<details><summary>Solution</summary>

```python
import torch, torch.nn.functional as F
from llmre.reasoning.distill import reverse_kl
g = torch.Generator().manual_seed(0)
t_row = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64).log()
z = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64).log().requires_grad_(True)
exact, = torch.autograd.grad(reverse_kl(t_row, z), z)
for N in [10, 100, 10_000]:
    log_ps = F.log_softmax(z, -1)
    v = torch.multinomial(log_ps.exp().detach(), N, replacement=True, generator=g)
    r = -(log_ps[v] - t_row[v]).detach()
    loss = -((r - r.mean()) * log_ps[v]).mean()
    est, = torch.autograd.grad(loss, z)
    print(N, F.cosine_similarity(est, exact, dim=0).item())
```

The cosine similarity rises toward 1 as $N$ grows (about 0.986, 0.9996, 0.99999 with this seed). On a 3-token vocabulary 10 samples are already close; with a real 150K-token vocabulary a handful of samples per position is very noisy, which is why the full-vocabulary version is preferred whenever the teacher's whole distribution is available. The sampled version needs only the teacher's log-prob of the sampled token, so it works when the teacher returns per-token log-probs but not full distributions.

</details>

## Debugging exercise

A colleague's distillation run reports a training loss that is **negative** and falling. Their loss line is:

```python
loss = F.kl_div(F.softmax(student_logits, -1), F.log_softmax(teacher_logits, -1), log_target=True, reduction="batchmean")
```

What is wrong?

<details><summary>Answer</summary>

`F.kl_div` expects its first argument (`input`) to be **log**-probabilities. They passed probabilities, so the formula computes $\sum p_T(\log p_T - p_S)$, which is not a KL and can be negative. A true KL is never negative: a negative KL value always means a bug. The fix is `F.kl_div(F.log_softmax(student_logits, -1), F.log_softmax(teacher_logits, -1), log_target=True, reduction="batchmean")`, which is forward $\mathrm{KL}(p_T\|p_S)$. Then check the direction you wanted, mask the prompt, and note that `batchmean` divides by the first dimension only. With `(B, T, V)` inputs that is $B$, not $B \cdot T$, so reshape to `(B*T, V)` and mask first.

</details>

## Research connection

<div class="callout paper"><p><strong>On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes</strong> (GKD) — Agarwal et al., ICLR 2024, <a href="https://arxiv.org/abs/2306.13649">arXiv 2306.13649</a>. Defines on-policy distillation with the generalized JSD and the on-policy fraction $\lambda$. <strong>MiniLLM: On-Policy Distillation of Large Language Models</strong> — Gu et al., ICLR 2024, <a href="https://arxiv.org/abs/2306.08543">arXiv 2306.08543</a>. Argues for reverse KL and optimizes it with a policy-gradient method. Background: <strong>Distilling the Knowledge in a Neural Network</strong> — Hinton, Vinyals, Dean 2015, <a href="https://arxiv.org/abs/1503.02531">arXiv 1503.02531</a> (soft targets, "dark knowledge"); <strong>Sequence-Level Knowledge Distillation</strong> — Kim &amp; Rush, EMNLP 2016, <a href="https://arxiv.org/abs/1606.07947">arXiv 1606.07947</a> (the 17.3 technique). Production: <strong>Qwen3 Technical Report</strong>, <a href="https://arxiv.org/abs/2505.09388">arXiv 2505.09388</a>, "Strong-to-Weak Distillation" section (off-policy distillation on teacher responses, then on-policy distillation that aligns student and teacher logits by KL, for the small models; GPU-hour comparison is a company claim). Implementation: TRL's <a href="https://huggingface.co/docs/trl/main/en/distillation_trainer"><code>DistillationTrainer</code></a> (on-policy generation, chunked generalized JSD). Current research: <strong>On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics</strong> — Piskorz, Berthon, van der Schaar, 2026, <a href="https://arxiv.org/abs/2609.35259">arXiv 2609.35259</a>.</p></div>

**What the 2026 study adds (single paper, not yet reproduced).** It varies rollout policy, KL direction and learning rate separately on Llama 3 and Qwen 2.5 students. It reports that the KL direction mattered more than the rollout policy for task performance: forward KL was robust to whether rollouts were on- or off-policy, while reverse KL depended on on-policy rollouts. Learning rate, not rollout policy, governed forgetting. On-policy data helped generalization on harder variants, but the advantage did not reliably survive later RLVR. Treat it as a reason to ablate both knobs on your own task, not as a settled rule.

**What to read / what to skip.** GKD: read the background on KD divergences (forward/reverse KL, generalized JSD) and the GKD algorithm itself. Skim the task-by-task results; read the ablations on the on-policy fraction and the divergence. Skip the RLHF-combination part on a first pass. MiniLLM: read the argument for reverse KL and its mode-seeking illustration; skip the policy-gradient variance-reduction details unless you implement them. Qwen3 report: read only the Strong-to-Weak Distillation section and its distillation-vs-RL table.

**Code this after reading:** add a `mixed` policy to `distill_markov` that uses the student's visitation with probability $\lambda$ and the teacher's otherwise (GKD's $\lambda$), and plot `eval_on_policy_rkl` against $\lambda \in \{0, 0.25, 0.5, 0.75, 1\}$.

**Documented vs inferred.** *Publicly documented:* the GKD and MiniLLM methods; Qwen3's two-stage (off-policy, then on-policy) strong-to-weak distillation and its reported GPU-hour comparison (company claim); TRL's loss, defaults and chunking. *Reasonable industry practice:* top-$k$ stored logits for offline KD; staging sequence-level or off-policy KD before on-policy distillation. *Inference:* that a given frontier lab's small models are distilled this way, unless its report says so.

## Check yourself

<details><summary>Why is SFT on teacher samples a special case of forward-KL distillation, and what does it lose?</summary>

With a one-hot teacher on the sampled token $y$, $\mathrm{KL}(\text{one-hot}_y\|p_S) = -\log p_S(y)$, the SFT cross-entropy. Averaged over $y \sim p_T$ it equals the forward KL plus the teacher's entropy, a constant, so the expected gradient is the same. It loses the rest of the teacher's distribution at each position (the relative preferences among non-sampled tokens), so its gradient is far noisier.

</details>

<details><summary>In the hand example, why does reverse KL change token 2's logit even though $p_S(2) = p_T(2)$?</summary>

The reverse-KL gradient is $p_S(v)(\ell_v - K)$. Token 2 has $\ell_2 = 0$, but the baseline $K = 0.3466 &gt; 0$, so $\ell_2 - K &lt; 0$ and the gradient is $-0.1040$. Gradient descent raises the logit. Relative to the student's average mistake, token 2 is a good token, so the student moves mass toward it (and away from token 3).

</details>

<details><summary>A one-bump student is fit to a two-bump teacher. Which divergence gives a student that sometimes produces outputs the teacher would almost never produce, and why?</summary>

Forward KL. It is weighted by the teacher, so missing either teacher mode is very costly. A one-bump student can only cover both by spreading wide, which puts mass between the modes where the teacher has almost none (26.8% vs 0.24% in Experiment A). Reverse KL is weighted by the student and punishes exactly that mass, so it commits to one mode.

</details>

<details><summary>Why can off-policy distillation never teach the bigram student to recover from an off-road token, however long it trains?</summary>

The off-policy loss weights each row by how often the teacher visits it. The teacher almost never visits off-road tokens, so those rows get almost zero weight and almost zero gradient; recovery stays at chance, 1/8. The student, which slips, does visit them at inference. On-policy weighting uses the student's own visitation, so exactly the rows the student needs get trained.

</details>

<details><summary>Why can you precompute teacher outputs for off-policy distillation but not for on-policy distillation, and what does that mean for infrastructure?</summary>

Off-policy prefixes come from the teacher or a fixed dataset, so teacher log-probs (often top-$k$) can be computed once and stored. On-policy prefixes are written by the current student, which changes every step, so the teacher must score them live. On-policy distillation therefore needs the teacher running alongside training, like a reward model in RLHF, plus a fast generation engine for the student.

</details>

<details><summary>When would you still choose plain sequence-level distillation?</summary>

When you only have the teacher's text (closed API), when the tokenizers differ, or as a cheap first stage before on-policy distillation so the student's own samples are good enough to be worth grading. DeepSeek's R1 distillation used it, and it is still a standard first stage.

</details>

## Next

You can now distill a reasoning model three ways and say what each one gives up. The course turns from *how a model reasons* to *how a model acts*. Continue to [18.1 · Tool use](lessons/module-18/lesson-01.md).
