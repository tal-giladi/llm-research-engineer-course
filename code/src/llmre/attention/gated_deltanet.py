"""Linear attention, the delta rule and Gated DeltaNet (Module 12, lesson 05).

Extends :mod:`llmre.attention.gqa` (lesson 12.3). GQA shrinks the KV cache by
sharing key/value heads, but the cache still grows linearly with context length
``T``. The layers here replace the growing cache with a **fixed-size state**
``S`` of shape ``(d_v, d_k)`` per head that is updated once per token:

* **Linear attention** (Katharopoulos et al. 2020): drop the softmax, then
  ``o_t = sum_{s<=t} (q_t . k_s) v_s = S_t q_t`` with ``S_t = S_{t-1} + v_t k_t^T``.
  Optional per-token decay ``alpha_t`` (Mamba2/GLA-style gating):
  ``S_t = alpha_t S_{t-1} + v_t k_t^T``.
* **Gated delta rule** (Yang, Kautz & Hatamizadeh, ICLR 2025):
  ``S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T``
  ``    = alpha_t S_{t-1} + beta_t (v_t - alpha_t S_{t-1} k_t) k_t^T``.
  The state is corrected toward ``v_t`` along ``k_t`` instead of having
  ``v_t k_t^T`` blindly added, so writing a new value under an old key
  overwrites instead of accumulating.

Production models (Qwen3-Next, Kimi Linear) interleave these layers with
ordinary softmax attention, typically 3 linear layers : 1 full-attention layer;
:func:`hybrid_layer_pattern` and :func:`hybrid_decode_cache_bytes` capture that.

Everything here is the slow, readable token-by-token recurrence. Production
kernels (``flash-linear-attention``) compute the same function in chunks.

Layout convention for every function: ``q, k`` are ``(B, H, T, d_k)``,
``v`` is ``(B, H, T, d_v)``, ``alpha, beta`` are ``(B, H, T)`` and the state is
``(B, H, d_v, d_k)``. The output is ``(B, H, T, d_v)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmre.model.config import GPTConfig
from llmre.model.rmsnorm import RMSNorm


def causal_linear_attention_parallel(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor | None = None
) -> torch.Tensor:
    """Quadratic "attention-like" form of (decayed) linear attention.

    ``o_t = sum_{s<=t} D[t, s] (q_t . k_s) v_s`` with ``D[t, s] = prod_{r=s+1}^{t} alpha_r``
    (all ones if ``alpha`` is ``None``). This is the reference that the recurrent
    form must match: it materializes a ``(T, T)`` matrix, exactly like softmax
    attention, but with no softmax.

    Returns:
        ``(B, H, T, d_v)``.
    """
    T = q.shape[-2]
    scores = q @ k.transpose(-2, -1)  # (B, H, T, T)
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    if alpha is not None:
        # log D[t, s] = c_t - c_s with c = cumsum(log alpha); only s <= t is used.
        c = torch.cumsum(torch.log(alpha), dim=-1)  # (B, H, T)
        log_d = c.unsqueeze(-1) - c.unsqueeze(-2)  # (B, H, T, T)
        decay = torch.exp(log_d.masked_fill(~causal, float("-inf")))
        scores = scores * decay
    scores = scores.masked_fill(~causal, 0.0)
    return scores @ v


def linear_attention_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear attention as an RNN: ``S_t = alpha_t S_{t-1} + v_t k_t^T``, ``o_t = S_t q_t``.

    Args:
        q, k: ``(B, H, T, d_k)``; v: ``(B, H, T, d_v)``.
        alpha: optional per-token decay in ``(0, 1]``, ``(B, H, T)``. ``None`` = no decay.
        initial_state: optional ``(B, H, d_v, d_k)`` state carried from earlier tokens.

    Returns:
        ``(o, S_T)``: outputs ``(B, H, T, d_v)`` and the final state ``(B, H, d_v, d_k)``.
    """
    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    S = q.new_zeros(B, H, d_v, d_k) if initial_state is None else initial_state
    outs = []
    for t in range(T):
        if alpha is not None:
            S = alpha[..., t, None, None] * S
        S = S + v[..., t, :, None] * k[..., t, None, :]  # outer product v_t k_t^T
        outs.append((S @ q[..., t, :, None]).squeeze(-1))  # (B, H, d_v)
    return torch.stack(outs, dim=-2), S


def gated_delta_rule_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated delta rule, one token at a time.

    ``S_t = alpha_t S_{t-1} + beta_t (v_t - alpha_t S_{t-1} k_t) k_t^T`` and ``o_t = S_t q_t``.

    ``alpha = 1`` gives plain DeltaNet; ``beta = 0`` leaves the state untouched
    except for decay. Keys are expected to be L2-normalized (as in Gated DeltaNet)
    so that ``I - beta k k^T`` is a contraction for ``beta`` in ``[0, 1]``.

    Args:
        q, k: ``(B, H, T, d_k)``; v: ``(B, H, T, d_v)``.
        alpha: decay gate in ``(0, 1]``, ``(B, H, T)``.
        beta: write strength in ``[0, 1]``, ``(B, H, T)``.
        initial_state: optional ``(B, H, d_v, d_k)``.

    Returns:
        ``(o, S_T)``: outputs ``(B, H, T, d_v)`` and the final state ``(B, H, d_v, d_k)``.
    """
    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    S = q.new_zeros(B, H, d_v, d_k) if initial_state is None else initial_state
    outs = []
    for t in range(T):
        k_t = k[..., t, :]  # (B, H, d_k)
        a_t = alpha[..., t, None, None]  # (B, H, 1, 1)
        b_t = beta[..., t, None]  # (B, H, 1)
        S = a_t * S  # forget
        pred = (S @ k_t[..., None]).squeeze(-1)  # what the memory currently returns for k_t, (B, H, d_v)
        err = b_t * (v[..., t, :] - pred)  # delta: move part of the way toward v_t
        S = S + err[..., :, None] * k_t[..., None, :]
        outs.append((S @ q[..., t, :, None]).squeeze(-1))
    return torch.stack(outs, dim=-2), S


class GatedDeltaNet(nn.Module):
    """A Gated DeltaNet token mixer over ``(B, T, C)``, a drop-in for an attention layer.

    Simplified from Yang et al. (2025) / Qwen3-Next: no short causal convolution
    and one scalar decay per head. Per head ``h`` with ``d = C // n_head``:

    * ``q, k = l2norm(silu(W_q x)), l2norm(silu(W_k x))``; ``q`` is scaled by ``d^-0.5``.
    * ``v = W_v x``.
    * ``alpha = exp(-softplus(W_a x))`` in ``(0, 1)``; ``beta = sigmoid(W_b x)`` in ``(0, 1)``.
    * ``o = RMSNorm(gated_delta_rule(q, k, v, alpha, beta)) * silu(W_g x)``, then ``W_o``.

    Shapes (forward): input ``(B, T, C)`` -> output ``(B, T, C)``. The recurrent
    state is ``(B, n_head, d, d)`` and does not depend on ``T``.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.d_h = cfg.n_embd // cfg.n_head
        C = cfg.n_embd
        self.q_proj = nn.Linear(C, C, bias=False)
        self.k_proj = nn.Linear(C, C, bias=False)
        self.v_proj = nn.Linear(C, C, bias=False)
        self.a_proj = nn.Linear(C, self.n_head, bias=True)  # one decay per head per token
        self.b_proj = nn.Linear(C, self.n_head, bias=True)  # one write strength per head per token
        self.g_proj = nn.Linear(C, C, bias=False)  # output gate
        self.o_norm = RMSNorm(self.d_h)
        self.o_proj = nn.Linear(C, C, bias=cfg.bias)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, self.n_head, self.d_h).transpose(1, 2)  # (B, H, T, d)

    def forward_with_state(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the layer and also return the final recurrent state.

        Args:
            x: ``(B, T, C)``.
            state: optional ``(B, n_head, d, d)`` state from a previous call
                (e.g. the prompt), so decoding continues where it stopped.

        Returns:
            ``(y, state)``: ``y`` is ``(B, T, C)``; ``state`` is ``(B, n_head, d, d)``.
        """
        B, T, C = x.shape
        q = F.normalize(F.silu(self._heads(self.q_proj(x))), dim=-1) * self.d_h**-0.5
        k = F.normalize(F.silu(self._heads(self.k_proj(x))), dim=-1)
        v = self._heads(self.v_proj(x))
        alpha = torch.exp(-F.softplus(self.a_proj(x))).transpose(1, 2)  # (B, H, T)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)  # (B, H, T)
        o, state = gated_delta_rule_recurrent(q, k, v, alpha, beta, initial_state=state)
        o = self.o_norm(o)  # per-head RMSNorm over d
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(o * F.silu(self.g_proj(x)))
        return y, state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer to ``x`` of shape ``(B, T, C)``; returns ``(B, T, C)``."""
        return self.forward_with_state(x)[0]


def hybrid_layer_pattern(n_layer: int, linear_per_full: int = 3) -> list[str]:
    """Layer types for a hybrid stack: ``linear_per_full`` linear layers, then one full.

    ``hybrid_layer_pattern(8, 3)`` -> ``['linear', 'linear', 'linear', 'full'] * 2``
    (the Qwen3-Next / Kimi Linear 3:1 layout).
    """
    period = linear_per_full + 1
    return ["full" if (i + 1) % period == 0 else "linear" for i in range(n_layer)]


def kv_cache_bytes(
    n_layer: int, n_kv_head: int, d_head: int, T: int, bytes_per: int = 2, batch: int = 1
) -> int:
    """KV-cache bytes for ``n_layer`` softmax-attention layers (lesson 12.3 formula)."""
    return 2 * n_layer * n_kv_head * d_head * T * bytes_per * batch


def linear_state_bytes(
    n_layer: int, n_head: int, d_k: int, d_v: int, bytes_per: int = 4, batch: int = 1
) -> int:
    """Recurrent-state bytes for ``n_layer`` linear-attention layers. Independent of ``T``.

    States are usually kept in fp32 (``bytes_per=4``) because they accumulate over
    the whole sequence.
    """
    return n_layer * n_head * d_k * d_v * bytes_per * batch


def hybrid_decode_cache_bytes(
    pattern: list[str],
    T: int,
    n_kv_head: int,
    d_head: int,
    n_lin_head: int,
    d_k: int,
    d_v: int,
    kv_bytes_per: int = 2,
    state_bytes_per: int = 4,
    batch: int = 1,
) -> int:
    """Total per-sequence decode memory of a hybrid stack: KV cache for 'full' layers + states."""
    n_full = sum(p == "full" for p in pattern)
    n_lin = sum(p == "linear" for p in pattern)
    return kv_cache_bytes(n_full, n_kv_head, d_head, T, kv_bytes_per, batch) + linear_state_bytes(
        n_lin, n_lin_head, d_k, d_v, state_bytes_per, batch
    )
