"""Training-inference mismatch in RL: when "on-policy" is secretly off-policy.

This is Module 16, lesson 16.4. It extends :mod:`llmre.rl.grpo` (lesson 16.1),
which is unchanged. ``grpo_objective(logprobs, old_logprobs, advantages)``
assumes that ``old_logprobs`` are the log-probabilities of the distribution
the completions were *sampled from*. In a real RL stack the sampler is a
separate inference engine (vLLM, SGLang) and the learner is a training stack
(FSDP, Megatron, TorchTitan). Even with identical weights the two compute
slightly different numbers: different kernels, different reduction orders,
different batch compositions, different precisions, or different sampling
transforms (temperature, top-k, top-p). Then the sampling distribution ``mu``
is not the trainer's ``pi_old`` and the "on-policy" update is off-policy.

The module has three parts, all plain tensor math so every bit is visible:

1. **Where the mismatch comes from.** :func:`split_k_matvec` computes
   ``x @ W`` with the reduction over the hidden dimension split into chunks,
   the way GPU matmul kernels do ("split-K"). Floating-point addition is not
   associative, so the result depends on the split. :func:`engine_num_splits`
   is a toy kernel heuristic that picks the split from the batch size, which
   makes a request's logits depend on *which other requests share its batch*.
   :func:`toy_logprobs` with ``batch_invariant=True`` fixes the split, so a
   row's result no longer depends on the batch (the idea behind vLLM's
   ``VLLM_BATCH_INVARIANT=1`` and TorchTitan's batch-invariant mode).
2. **How to measure it.** :func:`mismatch_metrics`.
3. **How to correct for it.** :func:`truncated_is_weights` (TIS),
   :func:`ratio_rejection_mask`, and :func:`decoupled_grpo_objective`, which
   multiplies the unchanged GRPO surrogate by a truncated importance weight
   ``min(pi_old / mu, C)``. When ``mu == pi_old`` it is exactly
   :func:`llmre.rl.grpo.grpo_objective`.

A last group (:func:`softmax_bandit_gradient`, :func:`train_kl_bandit`) is an
exactly-enumerable toy that shows the *consequence*: with a KL-regularized
objective the uncorrected update converges to the wrong policy, and the
importance-weighted one converges to the analytical optimum
``pi* ∝ pi_ref * exp(r / beta)``.
"""

from __future__ import annotations

import torch

from llmre.rl.grpo import grpo_objective

# --------------------------------------------------------------------------
# 1. Where the mismatch comes from
# --------------------------------------------------------------------------


def split_k_matvec(
    x: torch.Tensor,
    W: torch.Tensor,
    num_splits: int,
    acc_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``x @ W`` with the reduction over ``d`` split into ``num_splits`` chunks.

    x: (B, d), W: (d, V)  ->  (B, V) float32 (values rounded to ``out_dtype``).

    Each chunk is accumulated sequentially (one multiply-add per hidden index,
    in ``acc_dtype``), then the chunk partials are added left to right. The
    sum is the same in exact arithmetic for every ``num_splits``; in floating
    point it is not. The result is rounded to ``out_dtype`` (pass bf16 to
    mimic a kernel that writes bf16 outputs) and returned as float32 so callers
    can do the softmax in full precision. With fp32 accumulation the split
    changes results by about 1e-6; with bf16 accumulation by about 1e-1.

    The explicit loop makes the summation order exact and independent of the
    batch size ``B``: every row is computed by the same sequence of
    element-wise operations.
    """
    if x.dim() != 2 or W.dim() != 2 or x.shape[1] != W.shape[0]:
        raise ValueError(f"expected x (B, d) and W (d, V); got {tuple(x.shape)} and {tuple(W.shape)}")
    d = x.shape[1]
    if not 1 <= num_splits <= d:
        raise ValueError(f"num_splits must be in [1, {d}], got {num_splits}")
    xa, Wa = x.to(acc_dtype), W.to(acc_dtype)
    bounds = torch.linspace(0, d, num_splits + 1).round().long().tolist()
    total = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        part = torch.zeros(x.shape[0], W.shape[1], dtype=acc_dtype)
        for k in range(lo, hi):
            part = part + xa[:, k : k + 1] * Wa[k]
        total = part if total is None else total + part
    return total.to(out_dtype).to(torch.float32)


def engine_num_splits(batch_size: int) -> int:
    """A toy split-K heuristic: small batches split the reduction more.

    Real kernel libraries choose tile sizes and split-K factors from the
    problem shape to keep every streaming multiprocessor busy. With few rows
    there is little parallel work, so the reduction dimension is split to
    create more of it. The exact thresholds here are invented; the point is
    that the choice depends on ``batch_size``.
    """
    if batch_size <= 4:
        return 4
    if batch_size <= 16:
        return 2
    return 1


def toy_logprobs(
    x: torch.Tensor,
    W: torch.Tensor,
    tokens: torch.Tensor,
    batch_invariant: bool = False,
    fixed_splits: int = 1,
    acc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Log-probabilities of ``tokens`` under ``softmax(x @ W)`` from a toy engine.

    x: (B, d) hidden states, W: (d, V) LM-head weights, tokens: (B,) int64.
    Returns (B,) float32.

    ``batch_invariant=False``: the split is ``engine_num_splits(B)``, so the
    same row can get different logits depending on the batch it is in.
    ``batch_invariant=True``: the split is always ``fixed_splits``.
    """
    splits = fixed_splits if batch_invariant else engine_num_splits(x.shape[0])
    logits = split_k_matvec(x, W, splits, acc_dtype=acc_dtype)
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, tokens.long().unsqueeze(-1)).squeeze(-1)


# --------------------------------------------------------------------------
# 2. Measuring the mismatch
# --------------------------------------------------------------------------


def _mask_like(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return torch.ones_like(x, dtype=torch.bool) if mask is None else mask.bool()


def mismatch_metrics(
    train_logp: torch.Tensor,
    rollout_logp: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summary statistics of the trainer/sampler disagreement on sampled tokens.

    train_logp, rollout_logp: (B, T) log-probs of the *sampled* tokens under the
    trainer (``pi_old``) and the sampler (``mu``); mask: (B, T) bool, True on
    response tokens.

    - ``mean_abs_diff`` / ``max_abs_diff``: of ``log pi_old - log mu`` per token.
    - ``k3_kl``: the K3 estimator of KL(mu || pi_old), the mean of
      ``rho - log rho - 1`` with ``rho = pi_old / mu``. It is >= 0 per token and
      unbiased because the tokens were sampled from ``mu``.
    - ``max_seq_log_ratio``: the largest |sum_t log rho_t| over sequences, the
      log of the sequence-level importance weight. Small per-token noise adds up
      over long responses.
    """
    m = _mask_like(train_logp, mask)
    delta = (train_logp - rollout_logp).masked_fill(~m, 0.0)
    n = m.sum().clamp_min(1)
    rho = torch.exp(delta)
    k3 = (rho - delta - 1.0).masked_fill(~m, 0.0)
    return {
        "mean_abs_diff": float(delta.abs().sum() / n),
        "max_abs_diff": float(delta.abs().max()),
        "k3_kl": float(k3.sum() / n),
        "max_seq_log_ratio": float(delta.sum(dim=-1).abs().max()),
    }


# --------------------------------------------------------------------------
# 3. Correcting for it
# --------------------------------------------------------------------------


def truncated_is_weights(
    train_logp: torch.Tensor,
    rollout_logp: torch.Tensor,
    cap: float = 2.0,
    level: str = "token",
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Truncated importance weights ``min(pi_old / mu, cap)`` (TIS), detached.

    train_logp, rollout_logp, mask: (B, T). Returns (B, T) float32, zero where
    ``mask`` is False.

    - ``level="token"``: ``w_t = min(exp(log pi_old_t - log mu_t), cap)``.
      Low variance, but it ignores how earlier tokens changed the prefix
      distribution, so it is biased.
    - ``level="sequence"``: ``w = min(prod_t rho_t, cap)``, the same value
      broadcast to every token of the sequence. Unbiased before truncation,
      high variance on long sequences.

    The weights are constants for the gradient: they correct *which data we
    have*, they are not something to optimize.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    m = _mask_like(train_logp, mask)
    delta = (train_logp - rollout_logp).detach().masked_fill(~m, 0.0)
    if level == "token":
        w = torch.exp(delta).clamp(max=cap)
    elif level == "sequence":
        w = torch.exp(delta.sum(dim=-1, keepdim=True)).clamp(max=cap).expand_as(delta)
    else:
        raise ValueError(f"level must be 'token' or 'sequence', got {level!r}")
    return w.masked_fill(~m, 0.0)


def ratio_rejection_mask(
    train_logp: torch.Tensor,
    rollout_logp: torch.Tensor,
    low: float = 0.5,
    high: float = 2.0,
    mask: torch.Tensor | None = None,
    geometric: bool = False,
) -> torch.Tensor:
    """Which sequences to keep: True where the sequence ratio is in [low, high].

    Returns (B,) bool. The ratio is ``prod_t rho_t`` or, with
    ``geometric=True``, its per-token geometric mean
    ``exp(mean_t log rho_t)``, which does not grow with length and so needs
    much tighter bounds (e.g. 0.999-1.001). Dropping a sequence is the blunt
    alternative to down-weighting it: no bias from truncation, but less data.
    """
    m = _mask_like(train_logp, mask)
    delta = (train_logp - rollout_logp).detach().masked_fill(~m, 0.0)
    log_r = delta.sum(dim=-1)
    if geometric:
        log_r = log_r / m.sum(dim=-1).clamp_min(1)
    r = torch.exp(log_r)
    return (r >= low) & (r <= high)


def decoupled_grpo_objective(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    cap: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """GRPO surrogate with a truncated importance correction for the sampler.

    All inputs have the same shape (one entry per completion, as in
    :func:`llmre.rl.grpo.grpo_objective`). Three policies are involved:

    - ``rollout_logprobs``: ``log mu``, reported by the inference engine.
    - ``old_logprobs``: ``log pi_old``, recomputed by the trainer before the
      update; the PPO clip is anchored here.
    - ``logprobs``: ``log pi_theta``, the policy being optimized.

    ``objective = min(pi_old / mu, cap) * grpo_objective(logprobs, old_logprobs, A)``.
    When ``rollout_logprobs == old_logprobs`` the weight is 1 and this is
    exactly ``grpo_objective``. Returns an objective to *maximize*.
    """
    w = torch.exp((old_logprobs - rollout_logprobs).detach()).clamp(max=cap)
    obj = w * grpo_objective(logprobs, old_logprobs, advantages, clip_eps=clip_eps, reduction="none")
    if reduction == "mean":
        return obj.mean()
    if reduction == "sum":
        return obj.sum()
    if reduction == "none":
        return obj
    raise ValueError(f"unknown reduction {reduction!r}")


# --------------------------------------------------------------------------
# 4. Consequence: an exactly enumerable KL-regularized bandit
# --------------------------------------------------------------------------


def kl_regularized_value(theta: torch.Tensor, rewards: torch.Tensor, ref_logp: torch.Tensor, beta: float) -> torch.Tensor:
    """``J(theta) = E_pi[r] - beta * KL(pi || pi_ref)`` with ``pi = softmax(theta)``."""
    logp = torch.log_softmax(theta, dim=-1)
    p = logp.exp()
    return (p * rewards).sum() - beta * (p * (logp - ref_logp)).sum()


def kl_regularized_optimum(rewards: torch.Tensor, ref_logp: torch.Tensor, beta: float) -> torch.Tensor:
    """Log-probs of the maximizer of :func:`kl_regularized_value`: ``pi* ∝ pi_ref e^{r/beta}``."""
    return torch.log_softmax(ref_logp + rewards / beta, dim=-1)


def softmax_bandit_gradient(
    theta: torch.Tensor,
    rewards: torch.Tensor,
    ref_logp: torch.Tensor,
    beta: float,
    sampler_logp: torch.Tensor,
    correction: str = "none",
    cap: float = 2.0,
) -> torch.Tensor:
    """Expected policy-gradient update when actions come from the sampler ``mu``.

    Enumerates every action ``a`` of a single-step softmax policy
    ``pi = softmax(theta)`` (no sampling noise):

        g = sum_a mu(a) * w(a) * R(a) * (e_a - pi),
        R(a) = r(a) - beta * (log pi(a) - log pi_ref(a)),

    where ``e_a - pi`` is ``grad_theta log pi(a)`` and ``w`` is

    - ``"none"``: 1 (pretend the samples came from pi; the bug),
    - ``"is"``: ``pi(a) / mu(a)`` (exact; equals ``grad J``),
    - ``"tis"``: ``min(pi(a) / mu(a), cap)``.

    An action with ``mu(a) = 0`` is never sampled, so no weight can bring its
    term back: importance sampling corrects *how often* the sampler produced
    something, not *whether* it could.
    """
    logp = torch.log_softmax(theta, dim=-1)
    p = logp.exp()
    mu = sampler_logp.exp()
    R = rewards - beta * (logp - ref_logp)
    # mu(a) * w(a), written so that actions the sampler can never produce
    # (mu = 0, e.g. cut by top-k/top-p) contribute 0 instead of 0 * inf.
    if correction == "none":
        mw = mu
    elif correction == "is":
        mw = torch.where(mu > 0, p, torch.zeros_like(p))
    elif correction == "tis":
        mw = torch.minimum(p, cap * mu)  # = mu * min(pi / mu, cap)
    else:
        raise ValueError(f"correction must be 'none', 'is' or 'tis', got {correction!r}")
    coef = mw * R  # (K,)
    # sum_a coef[a] * (e_a - p) = coef - p * sum(coef)
    return coef - p * coef.sum()


def train_kl_bandit(
    rewards: torch.Tensor,
    ref_logp: torch.Tensor,
    beta: float,
    sampler_bias: torch.Tensor,
    correction: str = "none",
    cap: float = 2.0,
    lr: float = 0.5,
    steps: int = 2000,
) -> torch.Tensor:
    """Gradient ascent with :func:`softmax_bandit_gradient`; returns final log pi.

    The sampler's logits are the policy's logits plus a fixed ``sampler_bias``
    (K,), a stand-in for a systematic engine difference (another kernel,
    precision, or sampling transform). Starts from ``theta = ref_logp``.
    """
    theta = ref_logp.clone().to(torch.float64)
    rewards = rewards.to(torch.float64)
    ref = ref_logp.to(torch.float64)
    bias = sampler_bias.to(torch.float64)
    for _ in range(steps):
        sampler_logp = torch.log_softmax(theta + bias, dim=-1)
        theta = theta + lr * softmax_bandit_gradient(theta, rewards, ref, beta, sampler_logp, correction, cap)
    return torch.log_softmax(theta, dim=-1)
