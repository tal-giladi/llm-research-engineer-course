"""Multi-teacher on-policy distillation, MOPD (lesson 17.4, advanced extension).

This extends :mod:`llmre.reasoning.distill` (lesson 17.4), which distills **one**
teacher into a student on the student's own rollouts. Nothing there changes. MOPD
adds two things:

1. **Several teachers, one per domain.** Each prompt carries a domain label (TRL calls
   it ``teacher_id``). The student's rollout for that prompt is scored, token by
   token, only by that domain's teacher. The teachers are usually RL-trained
   specialists that started from the same checkpoint as the student, so they share
   its tokenizer.
2. **The distillation signal written as an RL advantage.** MiMo-V2-Flash (Xiaomi,
   https://arxiv.org/abs/2601.02780, section 4.4) writes the per-token advantage as

       A_t = sg[ log pi_{teacher(x)}(y_t | x, y_<t) - log pi_theta(y_t | x, y_<t) ]
             + alpha * A_ORM

   and plugs it into a policy-gradient surrogate with a per-token, truncated
   importance weight w_t = sg[pi_theta / mu_theta] (set to 0 outside
   [eps_low, eps_high]) that corrects for the inference engine mu_theta having
   sampled the rollout (lesson 16.4). With alpha = 0 and w_t = 1 the expected
   gradient is exactly the gradient of the reverse KL of lesson 17.4: this is the
   "sampled" on-policy reverse-KL update of that lesson's exercise, routed per
   domain.

The module has two halves.

* Tensor functions with real ``(B, T)`` shapes, the pieces a trainer needs:
  :func:`route_teacher_logprobs`, :func:`mopd_advantages`,
  :func:`truncated_is_weights`, :func:`mopd_loss`, :func:`teacher_token_share`,
  :func:`token_share_weights`.
* An exact two-domain toy (bigram "language models", as in ``distill.py``) that
  shows routing vs a single teacher, and how the loss normalization (per token vs
  per sequence) decides which domain wins when the student's capacity is shared:
  :func:`make_two_domain_teachers`, :func:`domain_rkl`, :func:`distill_two_domains`.

Everything is small CPU float32/float64 and runs in well under a second.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llmre.reasoning.distill import divergence, state_visitation


# --------------------------------------------------------------------------- #
# Tensor pieces of an MOPD trainer
# --------------------------------------------------------------------------- #
def route_teacher_logprobs(all_teacher_logprobs: torch.Tensor, teacher_ids: torch.Tensor) -> torch.Tensor:
    """Pick, for each sequence, the per-token log-probs of its own domain's teacher.

    Args:
        all_teacher_logprobs: ``(K, B, T)`` log-prob that each of ``K`` teachers gives
            to the token the student actually sampled at each position. A real
            trainer only queries the routed teacher (one teacher server per
            ``teacher_id``); the dense ``K`` axis here keeps the toy simple.
        teacher_ids: ``(B,)`` integer domain label per sequence, in ``[0, K)``.

    Returns:
        ``(B, T)`` routed teacher log-probs.
    """
    K, B, T = all_teacher_logprobs.shape
    if teacher_ids.shape != (B,):
        raise ValueError("teacher_ids must have shape (B,)")
    if teacher_ids.min() < 0 or teacher_ids.max() >= K:
        raise ValueError("teacher_ids out of range")
    idx = teacher_ids.long().view(1, B, 1).expand(1, B, T)
    return all_teacher_logprobs.gather(0, idx).squeeze(0)


def mopd_advantages(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    orm_advantages: torch.Tensor | None = None,
    alpha: float = 0.0,
) -> torch.Tensor:
    """Per-token MOPD advantage ``sg[log p_T(y_t) - log p_S(y_t)] + alpha * A_ORM``.

    Args:
        teacher_logprobs: ``(B, T)`` routed teacher log-probs of the sampled tokens.
        student_logprobs: ``(B, T)`` student log-probs of the same tokens.
        orm_advantages: optional ``(B,)`` sequence-level outcome advantage (e.g. the
            group-normalized GRPO advantage of lesson 16.1), broadcast to every token.
        alpha: weight of the outcome term. ``0`` = pure distillation.

    Returns:
        ``(B, T)`` detached advantages. Positive where the teacher likes the sampled
        token more than the student does.
    """
    adv = (teacher_logprobs - student_logprobs).detach()
    if orm_advantages is not None and alpha != 0.0:
        adv = adv + alpha * orm_advantages.detach().unsqueeze(-1)
    return adv


def truncated_is_weights(
    student_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    eps_low: float = 0.5,
    eps_high: float = 2.0,
) -> torch.Tensor:
    """Per-token weight ``sg[pi_theta / mu]``, set to 0 when the ratio leaves the band.

    ``mu`` is the inference engine's probability of the sampled token (the policy
    that actually generated the rollout). Tokens whose ratio is outside
    ``[eps_low, eps_high]`` are dropped from the loss rather than clipped, as in
    MiMo-V2-Flash's Eq. 8. The report gives no numeric band; the defaults here are
    illustrative. Returns a detached ``(B, T)`` tensor.
    """
    ratio = torch.exp(student_logprobs.detach() - behavior_logprobs.detach())
    keep = (ratio >= eps_low) & (ratio <= eps_high)
    return torch.where(keep, ratio, torch.zeros_like(ratio))


def mopd_loss(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor,
    behavior_logprobs: torch.Tensor | None = None,
    orm_advantages: torch.Tensor | None = None,
    alpha: float = 0.0,
    normalize: str = "sequence",
    seq_weights: torch.Tensor | None = None,
    eps_low: float = 0.5,
    eps_high: float = 2.0,
) -> torch.Tensor:
    """Policy-gradient surrogate of MOPD on a batch of student rollouts.

    ``loss = - sum_b c_b sum_t mask_bt * w_bt * A_bt * log pi_theta(y_bt)``

    Args:
        student_logprobs: ``(B, T)`` student log-probs of the sampled tokens, **with**
            gradient (the only tensor that carries one).
        teacher_logprobs: ``(B, T)`` routed teacher log-probs (constants).
        mask: ``(B, T)`` 1 for completion tokens, 0 for prompt/padding.
        behavior_logprobs: optional ``(B, T)`` sampler log-probs. ``None`` means the
            trainer sampled the rollout itself, so every weight is 1.
        orm_advantages, alpha: optional outcome term, see :func:`mopd_advantages`.
        normalize: how per-token terms are averaged, which decides how the gradient
            budget is shared between sequences (and so between domains):
            ``"sequence"`` = mean over each sequence's tokens, then mean over
            sequences (MiMo-V2-Flash Eq. 7, the ``1/|y|`` factor);
            ``"token"`` = one mean over all completion tokens in the batch (TRL's
            token-averaged loss). Long answers get more weight under ``"token"``.
        seq_weights: optional ``(B,)`` extra per-sequence weights, e.g. from
            :func:`token_share_weights`.

    Returns:
        Scalar loss.
    """
    mask = mask.to(student_logprobs.dtype)
    adv = mopd_advantages(teacher_logprobs, student_logprobs, orm_advantages, alpha)
    if behavior_logprobs is None:
        w = torch.ones_like(adv)
    else:
        w = truncated_is_weights(student_logprobs, behavior_logprobs, eps_low, eps_high)
    per_token = -(w * adv * student_logprobs) * mask  # (B, T)
    sw = torch.ones(mask.shape[0], dtype=mask.dtype) if seq_weights is None else seq_weights.to(mask.dtype)
    if normalize == "sequence":
        per_seq = per_token.sum(-1) / mask.sum(-1).clamp_min(1.0)
        return (sw * per_seq).sum() / sw.sum().clamp_min(1e-12)
    if normalize == "token":
        return (sw.unsqueeze(-1) * per_token).sum() / (sw.unsqueeze(-1) * mask).sum().clamp_min(1.0)
    raise ValueError("normalize must be 'sequence' or 'token'")


def teacher_token_share(mask: torch.Tensor, teacher_ids: torch.Tensor, num_teachers: int) -> torch.Tensor:
    """Fraction of the batch's completion tokens that each teacher scores: ``(K,)``.

    This is TRL's ``teacher_token_frac/<id>`` metric. Compare it with the prompt
    share: a domain with short answers (instruction following) can hold 20% of the
    prompts but only a few percent of the tokens.
    """
    tokens = mask.to(torch.float64).sum(-1)  # (B,)
    per_teacher = torch.zeros(num_teachers, dtype=torch.float64).index_add_(0, teacher_ids.long(), tokens)
    return per_teacher / per_teacher.sum().clamp_min(1.0)


def token_share_weights(
    mask: torch.Tensor,
    teacher_ids: torch.Tensor,
    num_teachers: int,
    target_share: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sequence weights that give each teacher a chosen share of the token budget.

    With ``normalize="token"`` in :func:`mopd_loss`, a teacher's share of the loss is
    its share of the (weighted) tokens. These weights rescale every sequence of
    teacher ``k`` by ``target_share[k] / token_share[k]`` so that, after weighting,
    the shares equal ``target_share`` (default: equal across the teachers present).
    A simple form of the "token-share balancing" that Open-MOPD
    (https://arxiv.org/abs/2608.19098) names as one of its fixes; their exact
    schedule is not reproduced here. Returns ``(B,)`` float64.
    """
    share = teacher_token_share(mask, teacher_ids, num_teachers)
    present = share > 0
    if target_share is None:
        target_share = present.to(torch.float64) / present.sum()
    target_share = target_share.to(torch.float64)
    scale = torch.where(present, target_share / share.clamp_min(1e-12), torch.zeros_like(share))
    return scale[teacher_ids.long()]


# --------------------------------------------------------------------------- #
# Exact two-domain toy
# --------------------------------------------------------------------------- #
def make_two_domain_teachers(p: float = 0.95) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Two specialist bigram teachers over ``V = 4`` tokens, and the domains' prompts.

    Tokens: 0 = domain-A prompt, 1 = domain-B prompt, 2 and 3 = "content" tokens
    that both domains write. From either prompt, both teachers go to token 2.

    * Teacher A (the "alternate" specialist): from 2 go to 3, from 3 go to 2.
    * Teacher B (the "repeat" specialist): from 2 stay on 2, from 3 stay on 3.

    So on rows 2 and 3 the specialists want opposite things. A bigram student sees
    only the previous token, not which prompt started the rollout, so it cannot do
    both: its capacity on those rows is shared, and the training loss decides the
    compromise. (In a real model the prompt is in context; the shared capacity is
    finite parameters rather than a hard limit, but the budget question is the same.)

    Returns ``(teacher_A_logits (V, V), teacher_B_logits (V, V), [start_A, start_B])``,
    float64. Each teacher is also a sensible model of the *other* domain's prompt row,
    so a single teacher can be used for both domains (the "one teacher" baseline).
    """
    V = 4
    q = (1.0 - p) / (V - 1)

    def chain(next_of: dict[int, int]) -> torch.Tensor:
        P = torch.full((V, V), q, dtype=torch.float64)
        for s, nxt in next_of.items():
            P[s, nxt] = p
        return P.log()

    teacher_a = chain({0: 2, 1: 2, 2: 3, 3: 2})
    teacher_b = chain({0: 2, 1: 2, 2: 2, 3: 3})
    return teacher_a, teacher_b, [0, 1]


def domain_rkl(teacher_logits: torch.Tensor, student_logits: torch.Tensor, start: int, T: int) -> torch.Tensor:
    """Expected per-token reverse KL to ``teacher_logits`` on the student's own
    length-``T`` rollouts from ``start``: the on-policy objective of 17.4 for one domain.

    Visitation weights are detached (gradients flow only through the next-token
    distributions), exactly as in :func:`llmre.reasoning.distill.expected_sequence_divergence`.
    """
    per_state = divergence(teacher_logits, student_logits, "reverse", "none")  # (V,)
    d = state_visitation(F.softmax(student_logits.detach(), -1), start, T)  # (T, V)
    return (d * per_state).sum(-1).mean()


def distill_two_domains(
    teachers: list[torch.Tensor],
    starts: list[int],
    lengths: list[int],
    prompt_share: list[float],
    routing: str = "routed",
    normalize: str = "sequence",
    steps: int = 400,
    lr: float = 1.0,
) -> dict:
    """Distill one bigram student from two domains with exact on-policy reverse KL.

    Args:
        teachers: ``[teacher_A_logits, teacher_B_logits]``, each ``(V, V)``.
        starts: prompt token of each domain.
        lengths: answer length ``T_d`` of each domain (B's answers can be short).
        prompt_share: fraction of prompts from each domain (sums to 1).
        routing: ``"routed"`` = domain ``d`` is scored by teacher ``d`` (MOPD);
            ``"teacher0"`` = every domain is scored by teacher 0 (single teacher).
        normalize: ``"sequence"`` weights domain ``d`` by ``prompt_share[d]``;
            ``"token"`` by ``prompt_share[d] * T_d`` (its share of tokens), the
            same two options as :func:`mopd_loss`.
        steps, lr: plain gradient descent on the ``(V, V)`` student table.

    Returns a dict with the learned ``student_logits``, the per-domain on-policy
    reverse KL to the domain's **own** specialist (``rkl``, a list; lower is better;
    TRL logs the analogous ``teacher_jsd/<id>``), the loss weight each domain got
    (``domain_weight``), and the student's probability of the "alternate" move from
    token 2 (``p_alternate``; teacher A wants ``p``, teacher B wants ``(1-p)/3``).
    """
    V = teachers[0].shape[0]
    share = torch.tensor(prompt_share, dtype=torch.float64)
    if normalize == "sequence":
        w = share
    elif normalize == "token":
        w = share * torch.tensor(lengths, dtype=torch.float64)
    else:
        raise ValueError("normalize must be 'sequence' or 'token'")
    w = w / w.sum()
    scorer = list(range(len(teachers))) if routing == "routed" else [0] * len(teachers)
    s = torch.zeros(V, V, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([s], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = sum(w[d] * domain_rkl(teachers[scorer[d]], s, starts[d], lengths[d]) for d in range(len(teachers)))
        loss.backward()
        opt.step()
    with torch.no_grad():
        rkl = [domain_rkl(teachers[d], s, starts[d], lengths[d]).item() for d in range(len(teachers))]
        p_alt = F.softmax(s, -1)[2, 3].item()
    return {"student_logits": s.detach(), "rkl": rkl, "domain_weight": w.tolist(), "p_alternate": p_alt}
