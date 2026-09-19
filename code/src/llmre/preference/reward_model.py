"""Reward modeling from scratch: the Bradley-Terry preference model.

This is Module 15, lesson 15.1. SFT (Module 14) teaches the model to *follow*
instructions by imitating demonstrations, but imitation cannot tell a **good**
answer from a mediocre one — every demonstration is treated as equally correct.
To optimize for human preference we first need a function that *scores* a
completion: a **reward model**.

Human preference data does not come as absolute scores; it comes as **pairwise
comparisons**. A labeler sees a prompt ``x`` and two completions and marks one as
better. So each record is ``(x, y_w, y_l)`` — a prompt, the *chosen* (winning)
completion ``y_w``, and the *rejected* (losing) completion ``y_l``.

The **Bradley-Terry** model turns a scalar reward ``r(x, y)`` into a probability
that one item beats another::

    P(y_w > y_l) = sigmoid( r(x, y_w) - r(x, y_l) )

Only the *difference* of rewards matters, so the reward is defined up to an
additive constant. Training the reward model = maximizing the log-likelihood of
the observed preferences, which is the loss below:

    loss = -log sigmoid( r(x, y_w) - r(x, y_l) )     (averaged over the batch)

The reward model itself is just a language model (Module 6) whose per-token
language head is replaced by a **scalar head**: a single linear that maps the
last hidden state to one number, the reward for the whole sequence.

Papers: InstructGPT (Ouyang et al. 2022, https://arxiv.org/abs/2203.02155),
Anthropic HH-RLHF (Bai et al. 2022, https://arxiv.org/abs/2204.05862).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """A transformer backbone with a scalar reward head.

    The backbone maps token ids to a sequence of hidden states, exactly like the
    GPT of Module 6 up to (but not including) the ``lm_head``. Instead of a
    ``(C -> V)`` language head that scores every vocabulary token, a reward model
    has a ``(C -> 1)`` **scalar head** that reads a single position's hidden state
    and outputs one number: the scalar reward for the sequence ending there.

    We keep this class deliberately small and backbone-agnostic so it can wrap a
    tiny GPT in tests *or* be handed pre-computed hidden states. Two ways to use
    it:

    * pass a ``backbone`` module whose ``forward(idx)`` returns hidden states of
      shape ``(B, T, C)`` (e.g. a GPT truncated before its head), and call
      :meth:`forward` with token ids ``idx`` of shape ``(B, T)``; or
    * skip the backbone entirely and call :meth:`reward_from_hidden` with hidden
      states you already have.

    Args:
        n_embd: hidden width ``C`` the scalar head reads from.
        backbone: optional module mapping ``idx`` ``(B, T)`` int64 to hidden
            states ``(B, T, C)`` float. If ``None``, only
            :meth:`reward_from_hidden` is usable.

    Shapes:
        the scalar head is ``nn.Linear(C, 1, bias=...)``; a reward is a scalar per
        sequence, so :meth:`forward` returns ``(B,)`` float on the parameters'
        device.
    """

    def __init__(self, n_embd: int, backbone: nn.Module | None = None, bias: bool = True):
        super().__init__()
        self.backbone = backbone
        self.n_embd = n_embd
        # Scalar reward head: (B, T, C) -> (B, T, 1). One output feature.
        self.score = nn.Linear(n_embd, 1, bias=bias)

    def reward_from_hidden(
        self, hidden: torch.Tensor, seq_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Score a batch of sequences from their hidden states.

        The reward is read off the **last real token** of each sequence: that is
        the position that has attended over the whole prompt+completion, so its
        hidden state summarizes the entire sequence. (This matches InstructGPT /
        HH-RLHF, where the reward is the value at the final token.)

        Args:
            hidden: shape ``(B, T, C)`` float, any device. Per-position hidden
                states from the backbone.
            seq_lengths: optional ``(B,)`` int64 giving the number of real
                (non-pad) tokens in each row; the reward is taken at index
                ``seq_lengths - 1``. If ``None``, the last position ``T - 1`` is
                used for every row (assumes no right-padding).

        Returns:
            ``(B,)`` float rewards, one scalar per sequence, on ``hidden``'s
            device.
        """
        scores = self.score(hidden).squeeze(-1)          # (B, T, 1) -> (B, T)
        if seq_lengths is None:
            return scores[:, -1]                          # (B,)
        idx = (seq_lengths - 1).clamp(min=0)             # (B,) last real index
        rows = torch.arange(scores.shape[0], device=scores.device)
        return scores[rows, idx]                          # (B,)

    def forward(
        self, idx: torch.Tensor, seq_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Token ids ``(B, T)`` -> scalar rewards ``(B,)`` via ``backbone`` + head.

        Requires a ``backbone`` (raises if none was given). See
        :meth:`reward_from_hidden` for the ``seq_lengths`` convention.
        """
        if self.backbone is None:
            raise RuntimeError(
                "RewardModel has no backbone; call reward_from_hidden(...) with "
                "precomputed hidden states instead."
            )
        hidden = self.backbone(idx)                       # (B, T, C)
        return self.reward_from_hidden(hidden, seq_lengths)


def bradley_terry_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Bradley-Terry preference loss ``-log sigmoid(r_w - r_l)``.

    This is the negative log-likelihood of the observed preference under the
    Bradley-Terry model ``P(y_w > y_l) = sigmoid(r_w - r_l)``. Driving it down
    pushes the chosen reward above the rejected reward: the loss depends only on
    the **margin** ``r_w - r_l``, so the reward is learned up to a constant.

    Implemented with ``-F.logsigmoid`` (a numerically stable ``log sigmoid``)
    rather than ``-torch.log(torch.sigmoid(...))`` so large-magnitude margins do
    not overflow/underflow.

    Args:
        reward_chosen: ``r_w``, shape ``(B,)`` float, the reward of the winning
            completion in each pair.
        reward_rejected: ``r_l``, shape ``(B,)`` float, same shape/device.
        reduction: ``"mean"`` (default) or ``"sum"`` over the batch, or
            ``"none"`` to return the per-pair loss ``(B,)``.

    Returns:
        Scalar loss (``"mean"``/``"sum"``) or ``(B,)`` (``"none"``), same
        dtype/device as the inputs.
    """
    margin = reward_chosen - reward_rejected             # (B,)
    per_pair = -F.logsigmoid(margin)                     # (B,), = -log sigmoid(margin)
    if reduction == "none":
        return per_pair
    if reduction == "sum":
        return per_pair.sum()
    if reduction == "mean":
        return per_pair.mean()
    raise ValueError(f"unknown reduction {reduction!r}")
