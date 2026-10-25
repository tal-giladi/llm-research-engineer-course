# 16.4 · Training–inference mismatch: when on-policy RL is secretly off-policy

<div class="callout key"><p><strong>Added 2026-10-25 by the weekly curriculum review</strong> (<a href="#/research/CHANGELOG">changelog</a>). This extends 16.1–16.3; nothing earlier is replaced. The GRPO objective you implemented in <a href="#/lessons/module-16/lesson-01">16.1</a> is still correct. It assumes that the probabilities in <code>old_logprobs</code> belong to the distribution your completions were sampled from. In a real RL stack that assumption is quietly false. This lesson shows why, how to measure it, and the two production answers: make the numbers agree, or correct for the difference.</p></div>

<div class="prereq">
<p><strong>Prerequisites:</strong> the GRPO ratio $\rho = \pi_\theta / \pi_{\text{old}}$ and its clip from <a href="#/lessons/module-16/lesson-01">16.1 · From PPO to GRPO</a>; the RLVR loop from <a href="#/lessons/module-16/lesson-02">16.2 · RLVR &amp; verifiable rewards</a>; the KL penalty and the log-ratio estimator from <a href="#/lessons/module-15/lesson-02">15.2 · Policy gradient, advantage, PPO</a>; floating-point non-associativity and <code>torch.use_deterministic_algorithms</code> from <a href="#/lessons/module-07/lesson-03">07.3 · Checkpointing, resuming, seeds, reproducibility</a>; bf16 from <a href="#/lessons/module-08/lesson-03">08.3 · Profiling &amp; mixed precision</a>; temperature and top-k/top-p sampling from <a href="#/lessons/module-06/lesson-03">06.3 · Generation</a>.</p>
<p><strong>You will learn:</strong> why the inference engine that samples rollouts and the trainer that updates the policy assign <em>different</em> probabilities to the same tokens with the same weights; why a request's numbers can depend on which other requests share its batch (<strong>batch variance</strong>), and how <strong>batch-invariant</strong> kernels remove that; how to <strong>measure</strong> the mismatch (log-ratio statistics, the K3 KL estimator, sequence-level ratios); how to <strong>correct</strong> for it with truncated importance sampling (TIS), sequence-level weights and rejection masks, in the "decoupled" three-policy form of PPO/GRPO; and, on an exactly solvable toy, what goes wrong if you do neither. Code: <code>llmre.rl.mismatch</code>.</p>
<p><strong>Why this matters for ML:</strong> every production RL stack for LLMs (verl, slime, OpenRLHF, TorchTitan's TitanRL, the Qwen team's) separates generation from training, because generation dominates the cost and needs a serving engine. Since 2025 the mismatch between the two has been documented as a cause of unstable or collapsing RL runs. When your rewards differ from a baseline, you need to know whether you changed the algorithm or the numbers.</p>
</div>

<div class="hw">
<p><strong>Hardware track:</strong> CPU-only is enough. <code>tests/test_rl_mismatch.py</code> (16 tests) runs in about a second; <code>experiments/rl_mismatch.py</code> runs in about 3 seconds on a laptop CPU, under 200 MB RAM, 0 GPU-hours. Everything is a toy: a 64×50 "LM head" and a 4-armed bandit. Measuring the real mismatch between vLLM and a training framework needs a GPU, both stacks and a real model (e.g. one 24 GB GPU for a 1.5B model). That is not needed here; the mechanism is the same.</p>
</div>

## 1. Intuition: two programs, one set of weights

In 16.2 one Python object did everything: it sampled a group of completions, you scored them, and you computed `old_logprobs` with the same model, the same code and the same numbers. At scale this is split in two:

- an **inference engine** (vLLM, SGLang) samples the rollouts. It is built for throughput: continuous batching, fused kernels, CUDA graphs, often lower precision. It can return the log-probability of each token it sampled;
- a **trainer** (FSDP, Megatron, TorchTitan) holds the master weights and computes gradients with its own kernels and its own batch layout.

After each update the trainer copies the new weights into the engine. The weights are identical. The probabilities are not, because the two programs do the arithmetic differently. Call the engine's distribution $\mu$ (the **behavior** or rollout policy) and the trainer's $\pi_{\text{old}}$. GRPO's derivation needs $\mu = \pi_{\text{old}}$: "the samples came from the policy we are improving." When the two differ, the samples came from a slightly different policy, so the update is **off-policy**, and nothing in 16.1's code notices.

Where the differences come from, roughly from smallest to largest:

1. **Reduction order.** A matmul or a softmax sums many numbers. Floating-point addition is not associative (07.3). Kernels choose how to split a sum (across threads, tiles, "split-K" chunks) from the problem shape, so the engine and the trainer, which use different shapes, add in different orders.
2. **Batch composition.** Inside the engine, the shape that decides the split is the *batch*, and the batch is whatever requests happened to arrive together. The same prompt gets slightly different logits depending on its neighbours. This is **batch variance**.
3. **Different kernels.** The engine's fused attention or linear-attention kernel (12.5) and the trainer's are different programs with different rounding.
4. **Precision.** FP8 or other quantized rollouts against a bf16 trainer, or a bf16 LM head in one and fp32 in the other.
5. **Sampling transforms.** The engine samples from the distribution *after* temperature, top-k and top-p (06.3). If the trainer computes plain `log_softmax(logits)`, it is scoring a different distribution, and top-k/top-p even give $\mu(a)=0$ for tokens where $\pi(a)>0$.
6. **MoE routing.** A tiny logit difference can flip which expert a token is routed to (12.4), which is a large difference in the computation.
7. **Staleness.** In asynchronous RL the engine samples with weights that are one or more updates old. This is a different cause (policy lag rather than numerics) with the same mathematical shape.

<div class="callout key"><p><strong>Deterministic is not the same as batch-invariant, and neither is the same as "engine equals trainer".</strong> <code>torch.use_deterministic_algorithms(True)</code> from 07.3 promises the same output for the same inputs <em>and the same batch</em>. Batch invariance promises that one request's output does not depend on the rest of the batch. Engine–trainer agreement is a third, stronger property: two different programs produce the same bits. You can have the first without the second, and the second without the third.</p></div>

## 2. Mathematics

### 2.1 Three policies

For a prompt $q$ the engine samples a completion $o = (o_1, \dots, o_T)$ token by token from $\mu$. Write $s_t = (q, o_{<t})$ for the prefix. Three policies give a probability to the same token $o_t$:

- $\mu(o_t \mid s_t)$ — the **behavior** policy: what the engine actually sampled from (after temperature/top-k/top-p);
- $\pi_{\text{old}}(o_t \mid s_t)$ — the **proximal** policy: the trainer's recomputation with the same weights, before the update. The PPO/GRPO clip is anchored here;
- $\pi_\theta(o_t \mid s_t)$ — the policy being optimized, which changes during the update.

Per token define the **mismatch ratio**

$$
\rho_t = \frac{\pi_{\text{old}}(o_t \mid s_t)}{\mu(o_t \mid s_t)}, \qquad \log \rho_t = \log \pi_{\text{old}}(o_t \mid s_t) - \log \mu(o_t \mid s_t).
$$

- $\rho_t = 1$ — the trainer and the engine agree on this token.
- $\rho_t > 1$ — the trainer thinks the token was more likely than the engine did.

### 2.2 Why the uncorrected gradient is wrong

The quantity we want to increase is $J(\theta) = \mathbb{E}_{o \sim \pi_\theta}[R(o)]$. Its gradient (15.2) is

$$
\nabla_\theta J = \mathbb{E}_{o \sim \pi_\theta}\big[R(o)\,\nabla_\theta \log \pi_\theta(o)\big].
$$

The expectation is over $\pi_\theta$, but the samples come from $\mu$. Importance sampling rewrites an expectation under one distribution as an expectation under another (exact whenever $\mu(o) > 0$ wherever $\pi_\theta(o) > 0$):

$$
\mathbb{E}_{o \sim \pi}[f(o)] = \sum_o \mu(o)\,\frac{\pi(o)}{\mu(o)}\,f(o) = \mathbb{E}_{o \sim \mu}\!\left[\frac{\pi(o)}{\mu(o)}\,f(o)\right].
$$

Using samples from $\mu$ *without* the factor $\pi/\mu$ computes $\mathbb{E}_{o\sim\mu}[R\,\nabla \log \pi_\theta]$, a different vector. GRPO's ratio $\pi_\theta/\pi_{\text{old}}$ only corrects for the change *during* the update; it never sees $\mu$.

### 2.3 Decoupled objective and truncated importance sampling

The fix keeps the PPO/GRPO machinery and adds one constant weight per sample. This "decoupled" form, which separates the proximal policy from the behavior policy, comes from Hilton, Cobbe and Schulman's work on batch-size-invariant PPO, and is what verl's rollout-correction module implements:

$$
J^{\text{dec}} = \mathbb{E}_{o \sim \mu}\Big[\, w \cdot \min\big(r\,\hat A,\ \operatorname{clip}(r, 1-\epsilon, 1+\epsilon)\,\hat A\big)\Big], \qquad r = \frac{\pi_\theta}{\pi_{\text{old}}},\quad w = \min(\rho, C).
$$

- $\hat A$ — the group-relative advantage from 16.1.
- $r$ — exactly 16.1's ratio; the clip still limits how far one update moves.
- $w$ — the importance weight that converts "samples from $\mu$" into "samples from $\pi_{\text{old}}$". It is treated as a **constant** (no gradient flows through it).
- $C$ — the truncation cap. Plain importance weights can be huge for a token the engine thought unlikely; capping them trades a little bias for bounded variance. This is **truncated importance sampling (TIS)**. verl's documentation lists typical caps of 1.5–5 for token-level weights and 2–10 for sequence-level ones.

Two granularities:

$$
w_t^{\text{tok}} = \min(\rho_t, C), \qquad w^{\text{seq}} = \min\Big(\prod_{t=1}^{T} \rho_t,\ C\Big).
$$

- **Token-level** weights have low variance but ignore that earlier tokens changed which prefix you are in, so they are biased; verl's notes state the bias grows like $O(T^2 \Delta_{\max})$ with length $T$ and largest per-token mismatch $\Delta_{\max}$.
- **Sequence-level** weights are the exact importance weight of the whole completion (unbiased before truncation) but are a product of $T$ numbers, so their variance explodes with length.

Instead of down-weighting, you can **reject**: drop a sequence whose ratio $\prod_t \rho_t$, or its per-token geometric mean $\exp(\frac{1}{T}\sum_t \log\rho_t)$, falls outside $[C_{\text{low}}, C_{\text{high}}]$.

### 2.4 Measuring the mismatch

On the sampled tokens of a batch, with $m$ response tokens in total:

- mean and max of $|\log \rho_t|$;
- the **K3 estimator** of $\mathrm{KL}(\mu \,\|\, \pi_{\text{old}})$: $\ \hat k_3 = \frac{1}{m}\sum_t (\rho_t - \log \rho_t - 1)$. Every term is $\ge 0$ (because $x - 1 \ge \log x$), and its expectation under $\mu$ is the KL, because the tokens were sampled from $\mu$;
- the largest $|\sum_t \log \rho_t|$ over sequences, the log of the sequence-level weight.

## 3. Worked examples with tiny numbers

**Non-associativity in bf16.** bf16 keeps 8 significant bits, so the gap between $1$ and the next representable number is $2^{-7} = 0.0078125$. Let $s = 2^{-8}$:

- $(1 + s) + s$: $1 + 2^{-8}$ is exactly halfway between $1$ and $1 + 2^{-7}$; round-to-even gives $1$. Adding $s$ again gives $1$ again. Result: $1$.
- $1 + (s + s) = 1 + 2^{-7} = 1.0078125$, which is representable. Result: $1.0078125$.

Same three numbers, different order, different answer. A kernel that splits a 4096-long sum into 4 chunks instead of 1 performs a different sequence of roundings.

**Token-level weights.** Three sampled tokens, $C = 2$:

| token | $\mu$ (engine) | $\pi_{\text{old}}$ (trainer) | $\rho_t$ | $w_t = \min(\rho_t, 2)$ |
|---|---|---|---|---|
| 1 | 0.5 | 0.6 | 1.2 | 1.2 |
| 2 | 0.001 | 0.02 | 20 | **2** (capped) |
| 3 | 0.3 | 0.3 | 1 | 1 |

Token 2 is the dangerous case: the engine almost never samples it, so when it does, a full importance weight of 20 would let one token dominate the batch's gradient. The cap bounds that.

**K3 by hand.** For $\rho = 1.25$: $1.25 - \ln 1.25 - 1 = 1.25 - 0.2231 - 1 = 0.0269$. For $\rho = 0.5$: $0.5 + 0.6931 - 1 = 0.1931$. Both positive, as promised.

**Length makes small noise large.** Suppose every token's $\log \rho_t$ is independent noise with standard deviation $\sigma = 0.01$. The sequence log-ratio is a sum of $T$ such terms, so its standard deviation is $\sigma\sqrt{T}$: $0.1$ at $T = 100$, $0.32$ at $T = 1000$, $1.0$ at $T = 10{,}000$. At $T = 10{,}000$ a typical sequence weight is $e^{\pm 1}$, i.e. between $0.37$ and $2.7$. That is why long reasoning traces make sequence-level weights noisy, and why the cap fires more often on long responses (part B of the experiment: 0% capped at $T = 100$, 24% at $T = 10{,}000$).

**The decoupled objective's gradient.** Two completions with $\pi_\theta = \pi_{\text{old}}$ (first step, $r = 1$), $\hat A = 1$ each, engine probabilities $\mu/\pi_{\text{old}} = 0.8$ and $0.1$. Then $\rho = 1.25$ and $10$, $w = 1.25$ and $\min(10, 2) = 2$. At $r = 1$ the clipped surrogate's gradient with respect to $\log \pi_\theta$ is $\hat A$, so the gradients are $w \hat A = 1.25$ and $2$. Without correction both would be $1$.

## 4. Tensor shapes, dtype, device

| Tensor | Shape | dtype | Device | Note |
|---|---|---|---|---|
| `rollout_logp` ($\log\mu$) | $(B, T)$ | float32 | GPU, from the engine, then sent to the trainer | returned with the sampled tokens; store in fp32 |
| `train_logp` ($\log\pi_{\text{old}}$) | $(B, T)$ | float32 | trainer GPU | one extra forward pass, no grad |
| `logprobs` ($\log\pi_\theta$) | $(B, T)$ | float32 (from bf16 logits) | trainer GPU | requires grad |
| `mask` | $(B, T)$ | bool | trainer GPU | True on response tokens, False on prompt and padding |
| weights $w$ | $(B, T)$ | float32 | trainer GPU | detached; sequence-level is $(B, 1)$ broadcast |
| keep mask | $(B,)$ | bool | trainer GPU | from `ratio_rejection_mask` |

Course code note: 16.1's `grpo_objective` works with one log-probability per completion (shape $(B,)$, the sum over tokens). `decoupled_grpo_objective` keeps that convention, so its weight is the sequence-level one. `truncated_is_weights` and `mismatch_metrics` take per-token $(B, T)$ tensors.

Two things to get right on real systems: compute the softmax and the log-probabilities in **fp32** even when the logits come from a bf16 matmul; and make sure `rollout_logp` and `train_logp` are aligned token for token (same tokenizer, same response mask, and the engine's log-prob of the token it sampled, *after* its sampling transforms).

## 5. From scratch: `llmre.rl.mismatch`

The code is in `code/src/llmre/rl/mismatch.py`, tested in `code/tests/test_rl_mismatch.py`; the experiment is `code/experiments/rl_mismatch.py` (`cd code && PYTHONPATH=src python experiments/rl_mismatch.py`). `llmre.rl.grpo` is unchanged and is imported.

### 5.1 A toy engine with batch variance

`split_k_matvec(x, W, num_splits)` computes `x @ W` for `x` of shape $(B, d)$ and `W` of shape $(d, V)$ with an explicit loop: the $d$ hidden indices are cut into `num_splits` contiguous chunks, each chunk is accumulated one multiply-add at a time, and the chunk sums are added left to right. Because every row goes through the same element-wise operations, the order is exactly known.

```python
def engine_num_splits(batch_size):   # toy kernel heuristic
    if batch_size <= 4:
        return 4                     # little parallel work -> split the reduction more
    if batch_size <= 16:
        return 2
    return 1

def toy_logprobs(x, W, tokens, batch_invariant=False, fixed_splits=1, acc_dtype=torch.float32):
    splits = fixed_splits if batch_invariant else engine_num_splits(x.shape[0])
    logits = split_k_matvec(x, W, splits, acc_dtype=acc_dtype)
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, tokens.long().unsqueeze(-1)).squeeze(-1)
```

Score the same 4 requests alone and inside a batch of 64 (experiment part A):

```text
fp32 accumulate  batch_invariant=False  1.431e-06
fp32 accumulate  batch_invariant=True   0.000e+00
bf16 accumulate  batch_invariant=False  7.741e-02
bf16 accumulate  batch_invariant=True   0.000e+00
```

With fp32 accumulation one matmul moves a log-probability by about $10^{-6}$. That looks harmless, but a real model stacks hundreds of matmuls, norms and attention reductions, low-precision activations round those differences up, and section 3 showed how sums over long responses grow. With bf16 accumulation the difference is already $0.08$ in a single layer. Fixing the split makes the result independent of the batch: the bits are identical.

<div class="callout pt"><p><strong>What the production systems do.</strong> <em>Publicly documented:</em> vLLM has a beta batch-invariance mode (<code>VLLM_BATCH_INVARIANT=1</code>) that uses deterministic kernels for attention and other operations, disables some non-deterministic optimizations and pins the all-reduce reduction order under tensor parallelism; its documentation lists RL among the use cases and warns that it may cost performance (<a href="https://docs.vllm.ai/en/stable/features/batch_invariance/">vLLM docs</a>). The approach was popularized by Horace He's Thinking Machines post "Defeating Nondeterminism in LLM Inference" (September 2025), which argues that the cause of nondeterminism in served LLMs is mainly batch-size-dependent kernels, not GPU concurrency, and builds batch-invariant RMSNorm, matmul and attention. TorchTitan's TitanRL keeps <em>one</em> model definition for pretraining, post-training and inside vLLM, and states that with its batch-invariant mode "in supported on-policy configurations, the trainer and generator produce bitwise identical log probabilities" (<a href="https://pytorch.org/blog/torchtitan-update-new-sota-models-performance-at-scale-titanrl/">PyTorch blog, October 2026</a>, <strong>(company claim)</strong>). Batch invariance in the engine alone does not make the engine agree with the trainer; TitanRL's agreement comes from both sides running the same model code and kernels.</p></div>

### 5.2 Measuring and correcting

```python
def truncated_is_weights(train_logp, rollout_logp, cap=2.0, level="token", mask=None):
    m = mask.bool() if mask is not None else torch.ones_like(train_logp, dtype=torch.bool)
    delta = (train_logp - rollout_logp).detach().masked_fill(~m, 0.0)    # log rho_t, (B, T)
    if level == "token":
        w = torch.exp(delta).clamp(max=cap)
    else:  # "sequence": one weight per sequence, broadcast to its tokens
        w = torch.exp(delta.sum(dim=-1, keepdim=True)).clamp(max=cap).expand_as(delta)
    return w.masked_fill(~m, 0.0)

def decoupled_grpo_objective(logprobs, old_logprobs, rollout_logprobs, advantages,
                             clip_eps=0.2, cap=2.0, reduction="mean"):
    w = torch.exp((old_logprobs - rollout_logprobs).detach()).clamp(max=cap)
    obj = w * grpo_objective(logprobs, old_logprobs, advantages, clip_eps=clip_eps, reduction="none")
    ...  # reduction
```

(The file's version also validates its arguments.) The `.detach()` is the "constant" in section 2.3. When `rollout_logprobs == old_logprobs`, $w = 1$ and the function returns exactly `grpo_objective`; `test_decoupled_reduces_to_grpo` checks that. `mismatch_metrics` returns the four statistics of section 2.4, and `ratio_rejection_mask` returns which sequences to keep.

### 5.3 The consequence, computed exactly

To see what the missing weight does, take a one-step "bandit" with $K = 4$ actions, policy $\pi = \operatorname{softmax}(\theta)$, rewards $r = (1, 0, 0, 0)$, uniform reference policy and the KL-regularized objective of 15.2:

$$
J(\theta) = \mathbb{E}_{a\sim\pi}[r(a)] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}}), \qquad \pi^*(a) \propto \pi_{\text{ref}}(a)\,e^{r(a)/\beta}.
$$

With $\beta = 0.5$ the optimum is $\pi^* = (0.711, 0.096, 0.096, 0.096)$: $e^{2} = 7.39$ against $1, 1, 1$, divided by $10.39$. The "engine" samples from $\mu = \operatorname{softmax}(\theta + \delta)$ with a fixed bias $\delta = (-1, +1, 0, 0)$: it under-samples the good action. `softmax_bandit_gradient` sums over all four actions instead of sampling, so there is no noise:

$$
g = \sum_a \mu(a)\, w(a)\, \tilde R(a)\, (e_a - \pi), \qquad \tilde R(a) = r(a) - \beta\big(\log\pi(a) - \log\pi_{\text{ref}}(a)\big),
$$

where $e_a - \pi$ is $\nabla_\theta \log\pi(a)$ for a softmax. With $w = \pi/\mu$ this is exactly $\nabla_\theta J$ (`test_is_gradient_equals_true_gradient` checks it against autograd). After 2000 exact ascent steps (experiment part C):

```text
optimum pi* = [0.711 0.096 0.096 0.096]
no correction  pi = [0.486 0.209 0.153 0.153]  KL = 1.18e-01
TIS cap 1.0    pi = [0.541 0.153 0.153 0.153]  KL = 6.49e-02
TIS cap 1.5    pi = [0.638 0.121 0.121 0.121]  KL = 1.23e-02
TIS cap 2.0    pi = [0.711 0.096 0.096 0.096]  KL = 1.03e-07
full IS        pi = [0.711 0.096 0.096 0.096]  KL = 1.03e-07
```

Without correction the update converges, smoothly and confidently, to the **wrong** policy: it puts too much mass on the action the engine over-samples. Nothing diverges, and no loss curve would warn you. Full importance weights reach $\pi^*$. TIS sits in between: at $\pi^*$ the largest ratio is $\pi^*(0)/\mu(0) = 1.95$, so a cap of 2 never fires and is exact here; caps of 1.5 or 1 fire on the good action and leave a bias.

The last lines show what importance sampling cannot fix. If the engine can never produce an action ($\mu(a) = 0$, as with top-k or top-p truncation), no weight can bring its term back:

```text
support mismatch, optimum pi* = [0.61  0.083 0.083 0.225]
no correction  pi = [0.786 0.106 0.106 0.002]  KL = 2.44e-01
full IS        pi = [0.786 0.106 0.106 0.002]  KL = 2.44e-01
```

The fix there is not a weight. Either score the tokens under the same truncated distribution in the trainer, or sample without truncation during RL.

<div class="callout warn"><p>The toy uses a fixed, systematic engine bias so that the effect is exact and visible. Real numerical mismatch is mostly small and noisy, with occasional large outliers on low-probability tokens. Its effect on a real run is not a clean fixed point but extra bias and variance that compound over many updates. The 2026 study cited in section 7 reports that this alone can collapse training. The toy shows the direction of the error, not its size in practice.</p></div>

## 6. What the frameworks do, and what it costs

**Recomputing $\pi_{\text{old}}$.** The trainer runs one extra forward pass, without gradients, over every rollout. With the 07.4 accounting (forward $\approx 2PN$ FLOPs, forward plus backward $\approx 6PN$), that adds about a third of a training step's compute, plus a full pass over the rollouts' activations in memory (no saved activations, since there is no backward). Many stacks pay this anyway, because PPO with several updates per batch needs a fixed $\pi_{\text{old}}$.

**Engine log-probabilities.** The engine already has the logits when it samples, so returning $\log\mu$ of the sampled token is nearly free: $4$ bytes per generated token in fp32. For $B = 512$ completions of $T = 8192$ tokens that is $512 \cdot 8192 \cdot 4 = 16.8$ MB.

**The weights and the metrics.** Element-wise operations on $(B, T)$ tensors: negligible next to the forward pass.

**Batch-invariant or bitwise-aligned kernels.** They give up some of the shape-dependent tuning that makes kernels fast. vLLM documents a performance cost without a single number, because it depends on the model and hardware. You pay it on every rollout token, so it is the expensive option; its value is that it removes the mismatch instead of correcting it, which is what you want when you are debugging.

**In production:** *publicly documented:* verl's rollout-correction module implements the decoupled mode (three policies), a "bypass" mode that uses $\mu$ as the PPO anchor, token- and sequence-level truncated weights, and token/sequence/geometric rejection with K2/K3 statistics ([verl docs source](https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr_math.md)). The Qwen team's 2025 RL-stability study ([arXiv 2512.01374](https://arxiv.org/abs/2512.01374), 30B MoE, hundreds of thousands of GPU-hours, **(company claim)**) argues that the token-level surrogate is valid only when training–inference discrepancy and staleness are both small, finds that the basic policy gradient with importance-sampling correction was the most stable on-policy setup, and that MoE models additionally need "Routing Replay" (in our summary: keeping each token's expert choices fixed between the passes, so a tiny logit difference cannot reroute it). *Reasonable industry practice:* log `mismatch_metrics` every step and alert on them; keep the LM head and the softmax in fp32 on both sides. *Inference:* that the mismatch matters more for MoE and linear-attention models than for dense softmax-attention models (plausible from causes 3 and 6 above, not established in general).

## 7. Research connection

The mismatch was named and measured in 2025 by Feng Yao, Liyuan Liu and co-authors (blog posts "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training" and "FlashRL: 8bit rollouts, full power RL", and the NeurIPS 2025 contribution "On the Rollout-Training Mismatch in Modern RL Systems"). They proposed TIS and used it to make FP8 rollouts train as well as bf16 ones. We could not open those pages from the build environment, so they are named, not linked; papers that cite them are linked below.

<div class="callout paper"><p><strong>Batch size-invariance for policy optimization</strong> — Hilton, Cobbe, Schulman, NeurIPS 2022, <a href="https://arxiv.org/abs/2110.00641">arXiv 2110.00641</a>. Introduces the decoupled objective: a proximal policy for the clip, a behavior policy for the importance correction. <strong>Diagnosing Training Inference Mismatch in LLM Reinforcement Learning via a Zero-Mismatch Reference</strong> — Zhong et al., 2026, <a href="https://arxiv.org/abs/2605.14220">arXiv 2605.14220</a>. Builds a rollout engine that is bit-for-bit aligned with the FSDP trainer, so the mismatch can be switched off, and reports that small token-level disagreements alone can collapse training. <strong>Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch RL</strong> — Chen et al., ICLR 2026, <a href="https://arxiv.org/abs/2602.06107">arXiv 2602.06107</a>. Rollouts from a deliberately cheaper model, corrected by rejection sampling instead of weights. <strong>AIS: Adaptive Importance Sampling for Quantized RL</strong> — Zhou et al., 2026, <a href="https://arxiv.org/abs/2605.13907">arXiv 2605.13907</a>. FP8 rollouts; blends uncorrected and importance-weighted gradients with a per-batch coefficient. <strong>Trust Region Masking for Long-Horizon LLM RL</strong> — Li et al., 2025–26, <a href="https://arxiv.org/abs/2512.23075">arXiv 2512.23075</a>. Masks whole sequences that leave a trust region; the analysis behind sequence-level rejection.</p></div>

Open questions (2026): how to pick the cap and the granularity for very long agentic rollouts, where sequence weights are hopeless and token weights are biased; whether removing the mismatch (bitwise alignment) or correcting it (weights) gives better results per GPU-hour; and how the correction interacts with deliberate approximations such as quantized or speculative rollouts.

## Exercise

A response has three tokens. The engine reports $\log\mu = (-0.5, -2.0, -0.1)$; the trainer recomputes $\log\pi_{\text{old}} = (-0.4, -1.0, -0.1)$. With $C = 2$, compute (a) the token-level weights, (b) the sequence-level weight, (c) $\hat k_3$, and (d) whether `ratio_rejection_mask` with $[0.5, 2]$ keeps the sequence.

<details><summary>Hint</summary>

Start with $\log\rho_t = \log\pi_{\text{old}} - \log\mu$ for each token. Everything else is an exponential, a cap or a sum.

</details>

<details><summary>Stronger hint</summary>

$\log\rho = (0.1, 1.0, 0)$, so $\rho = (e^{0.1}, e^{1}, 1) = (1.105, 2.718, 1)$. The sequence ratio is $e^{1.1}$.

</details>

<details><summary>Solution</summary>

(a) $w^{\text{tok}} = (1.105,\ \min(2.718, 2) = 2,\ 1)$.
(b) $\prod\rho = e^{1.1} = 3.004$, so $w^{\text{seq}} = \min(3.004, 2) = 2$ for all three tokens.
(c) $\rho - \log\rho - 1$: $1.105 - 0.1 - 1 = 0.0052$; $2.718 - 1 - 1 = 0.718$; $0$. Mean $= 0.723/3 = 0.241$.
(d) $3.004 > 2$, so the sequence is rejected. The geometric mean $e^{1.1/3} = 1.443$ would pass $[0.5, 2]$ but fail a tight band such as $[0.999, 1.001]$.
Check: `truncated_is_weights`, `mismatch_metrics` and `ratio_rejection_mask` on these tensors.

</details>

## Implementation exercise

The engine samples at temperature $\tau = 0.8$: it draws from $\operatorname{softmax}(z/\tau)$. The trainer computes `log_softmax(z)`. Using `toy_logprobs`-style code, (1) measure the mismatch this alone causes on random logits with `mismatch_metrics`, and (2) write `tempered_logprobs(logits, tokens, temperature)` so that the trainer scores the distribution the engine actually sampled from. Then show the mismatch is gone.

<details><summary>Hint</summary>

This mismatch has nothing to do with rounding. It is a different distribution. Which distribution did the sampled tokens come from?

</details>

<details><summary>Stronger hint</summary>

Divide the logits by $\tau$ *before* `log_softmax`, in fp32. Temperature is a property of $\mu$, so it must be applied wherever $\log\mu$ or $\log\pi_{\text{old}}$ is computed.

</details>

<details><summary>Solution</summary>

```python
def tempered_logprobs(logits, tokens, temperature):
    logp = torch.log_softmax(logits.float() / temperature, dim=-1)
    return logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
```

With $\tau = 0.8$ on random logits of unit scale, the untempered trainer gives a mean $|\log\rho|$ of about $0.2$ (it grows with the logit scale): five orders of magnitude above the fp32 rounding effect of section 5.1. `tempered_logprobs` gives $0$ difference against an engine that uses the same formula. If the engine also applies top-k or top-p, the trainer must apply the same truncation and renormalize, otherwise you are back to the support mismatch of section 5.3.

</details>

## Debugging exercise

A colleague's RL loop logs a mean token weight of exactly $1.000$ and zero K3 every step, yet rewards plateau below the baseline. This is their code:

```python
out = engine.generate(prompts, temperature=0.8, return_logprobs=True)
old_logprobs = out.logprobs                  # from the engine
rollout_logprobs = out.logprobs
obj = decoupled_grpo_objective(policy_logprobs, old_logprobs, rollout_logprobs, adv)
```

What is wrong, and what should they log?

<details><summary>Solution</summary>

They anchor the clip at the engine's numbers and compute the weight from two copies of the same tensor, so $w \equiv 1$ by construction. The metric cannot see the mismatch they are trying to measure. $\pi_{\text{old}}$ must be the **trainer's** recomputation (one no-grad forward pass with the current weights, at the same temperature 0.8). This is what verl's "bypass" mode does deliberately, using $\mu$ as the anchor and computing the weight against $\pi_\theta$ instead. Then `mismatch_metrics(train_logp, rollout_logp)` reports the real disagreement. Separately, check that `policy_logprobs` are also computed at temperature 0.8.

</details>

## Common mistakes

- **Computing the weight from engine numbers on both sides**, as in the debugging exercise: the metric is always perfect.
- **Letting gradients flow through the weight.** $w$ corrects which data you have; it is not something to optimize. Forgetting `.detach()` adds a spurious gradient term.
- **Ignoring sampling transforms.** Temperature, top-k and top-p change $\mu$. Score with the same transforms or the "mismatch" is a different distribution.
- **Uncapped sequence weights on long responses.** Section 3: at $T = 10{,}000$ even tiny per-token noise gives weights spread over a factor of 7. Cap them, use token-level weights, or reject.
- **Treating `torch.use_deterministic_algorithms` as the fix.** It makes a run repeatable; it does not make the engine agree with the trainer.
- **Comparing log-probabilities computed in bf16.** Do the log-softmax in fp32, or rounding in the comparison itself will dominate what you measure.

## What to read / what to skip

- **vLLM batch-invariance page**: read all of it; it is short and lists exactly which kernels change.
- **TorchTitan update post**: read the TitanRL "Correctness and debugging" section; the rest is about pretraining performance.
- **verl rollout-correction math page**: read the three policies, the decoupled objective and the TIS formulas; skim the long list of rejection variants.
- **Qwen 2512.01374**: read the formulation section (why the token-level surrogate needs small discrepancy and staleness) and the Routing Replay discussion; skip the hyperparameter tables.
- **Hilton et al. 2110.00641**: read the decoupled-objective section; skip the Atari experiments.

## Code this after reading

Add mismatch logging to 16.2's toy RLVR loop: compute $\log\pi_{\text{old}}$ twice, once with the batch as sampled and once one sequence at a time with the bf16 path of `split_k_matvec`, and log `mismatch_metrics` per step. Then switch to `decoupled_grpo_objective` and compare the reward curves with and without the weight.

## Check yourself

<details><summary>The engine and the trainer load the same weights after every update. Name three reasons they can still assign different probabilities to the same token.</summary>

Different reduction orders (kernels split sums by shape; floating-point addition is not associative); batch composition (the engine's kernels choose splits from the batch, so neighbours change your numbers); different kernels or precisions (fused attention, FP8 rollouts, bf16 vs fp32 LM head); sampling transforms (temperature/top-k/top-p); MoE routing flips; staleness in async setups.

</details>

<details><summary>What does batch invariance guarantee, and what does it not?</summary>

It guarantees that one request's output does not depend on which other requests share its batch, so the engine is reproducible under changing load. It does not make the engine agree with the trainer, which runs different code. TitanRL's bitwise agreement comes from the same model code and kernels on both sides (in supported configurations, company claim).

</details>

<details><summary>Why is the importance weight a constant in the objective?</summary>

It converts an expectation under $\mu$ into one under $\pi_{\text{old}}$. Both are fixed when the batch was produced, so the weight is a property of the data, not of the parameters being optimized. Only $\pi_\theta$, through the ratio $r$, carries the gradient.

</details>

<details><summary>Token-level or sequence-level weights for 30,000-token agent trajectories? Why?</summary>

The sequence weight is a product of 30,000 ratios; even $\sigma = 0.01$ per token gives a log-weight spread of $0.01\sqrt{30000} \approx 1.7$, so most weights are far from 1 and the cap fires constantly. Token-level weights (biased, low variance), geometric-mean rejection, or removing the mismatch at the source are the practical choices.

</details>

<details><summary>In the toy bandit, why did the uncorrected update converge to the wrong policy instead of diverging?</summary>

Its expected gradient is a smooth vector field with its own fixed point, where $\sum_a \mu(a)\tilde R(a)(e_a - \pi) = 0$, which is not where $\nabla J = 0$. It converges confidently to that point. That is what makes the bug dangerous: no instability, just a worse answer.

</details>

<details><summary>Can importance sampling correct for top-p truncation in the engine? What should you do instead?</summary>

No. Tokens with $\mu(a) = 0$ are never sampled, so no weight can reintroduce their contribution (section 5.3). Apply the same truncation and renormalization when computing $\log\pi_{\text{old}}$ and $\log\pi_\theta$, or sample without truncation during RL.

</details>

## Next

Module 16 now ends with the systems fact that every RL algorithm in it relies on: the numbers you train on must describe the distribution you sampled from. Module 17 builds modern reasoning models on top of this RL machinery.

Continue to [17.1 · CoT, self-consistency, STaR](lessons/module-17/lesson-01.md).
