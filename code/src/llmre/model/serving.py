"""Inference serving foundations: KV cache, prefill vs decode, batching, speculative decoding.

Lesson 08.6 (added 2026-11-01 by the weekly curriculum review). This module extends
:meth:`llmre.model.gpt.GPT.generate` (lesson 06.3), which re-runs the whole prefix every
step. That method is unchanged and stays the reference: everything here must produce the
same logits (and, for greedy decoding, the same tokens) as the plain loop.

Contents, in lesson order:

* :class:`KVCache` and :func:`forward_with_cache` — run the course's own ``GPT`` weights
  on only the *new* tokens, reading past keys/values from a cache. A call with many new
  tokens is a **prefill**; a call with one new token is a **decode** step.
  :func:`generate_cached` is ``GPT.generate`` rebuilt on top of it.
* :func:`kv_cache_bytes`, :func:`matmul_intensity`, :func:`roofline_time` — the cost model
  that explains why prefill is compute-bound and decode is memory-bound (08.2's roofline).
* :func:`simulate_static_batching`, :func:`simulate_continuous_batching` — step-level
  scheduling (Orca's iteration-level scheduling) vs request-level batching.
* :class:`PagedKVAllocator` — fixed-size KV blocks with a per-sequence block table
  (vLLM's PagedAttention memory manager, without the kernel).
* :func:`residual_distribution`, :func:`speculative_accept`,
  :func:`speculative_first_token_distribution`, :func:`expected_tokens_per_round`,
  :func:`speculative_generate` — lossless draft-and-verify speculative decoding
  (Leviathan et al. 2023; Chen et al. 2023).

Everything is plain PyTorch for clarity; production engines (vLLM, SGLang, TensorRT-LLM)
implement the same ideas with custom kernels.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from llmre.attention.attention import scaled_dot_product_attention

# --------------------------------------------------------------------------- KV cache


class KVCache:
    """Per-layer keys and values of every token processed so far.

    Layer ``l`` stores ``k[l]`` and ``v[l]``, each ``(B, n_head, S, d_h)`` where ``S`` is the
    number of cached positions (the same for every layer). Tensors keep the dtype/device of
    the model's activations.

    Args:
        n_layer: number of transformer blocks in the model the cache belongs to.
    """

    def __init__(self, n_layer: int):
        self.n_layer = n_layer
        self.k: list[torch.Tensor | None] = [None] * n_layer
        self.v: list[torch.Tensor | None] = [None] * n_layer

    @property
    def length(self) -> int:
        """Number of cached positions ``S`` (0 for an empty cache)."""
        return 0 if self.k[0] is None else self.k[0].size(2)

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Add the new positions' ``k, v`` (``(B, nh, T_new, d_h)``) for one layer.

        Returns the full ``(K, V)`` for that layer, ``(B, nh, S + T_new, d_h)`` each.
        ``torch.cat`` copies; real engines write into preallocated (paged) storage instead.
        """
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]

    def truncate(self, length: int) -> None:
        """Keep only the first ``length`` positions (used to roll back rejected drafts)."""
        if length >= self.length:
            return
        for layer in range(self.n_layer):
            if length == 0:
                self.k[layer] = self.v[layer] = None
            else:
                self.k[layer] = self.k[layer][:, :, :length]
                self.v[layer] = self.v[layer][:, :, :length]

    def nbytes(self) -> int:
        """Bytes currently held by all cached keys and values."""
        total = 0
        for k, v in zip(self.k, self.v):
            if k is not None:
                total += k.numel() * k.element_size() + v.numel() * v.element_size()
        return total


def _attention_with_cache(attn, x, cache: KVCache, layer: int, start: int):
    """``CausalSelfAttention.forward`` for new positions ``start .. start+T-1`` only.

    Uses the module's own ``c_attn``/``c_proj`` weights. Query ``i`` (absolute position
    ``start + i``) may attend to cached and new keys at positions ``<= start + i``.
    """
    B, T, C = x.shape
    q, k, v = attn.c_attn(x).split(attn.n_embd, dim=2)          # 3 x (B, T, C)
    q = q.view(B, T, attn.n_head, attn.d_h).transpose(1, 2)     # (B, nh, T, d_h)
    k = k.view(B, T, attn.n_head, attn.d_h).transpose(1, 2)
    v = v.view(B, T, attn.n_head, attn.d_h).transpose(1, 2)
    K, V = cache.append(layer, k, v)                            # (B, nh, S, d_h), S = start + T
    S = K.size(2)
    q_pos = torch.arange(start, start + T, device=x.device)[:, None]   # (T, 1)
    k_pos = torch.arange(S, device=x.device)[None, :]                  # (1, S)
    mask = torch.zeros(T, S, dtype=x.dtype, device=x.device)
    mask = mask.masked_fill(k_pos > q_pos, float("-inf"))              # causal, offset by start
    y = scaled_dot_product_attention(q, K, V, mask=mask)               # (B, nh, T, d_h)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return attn.c_proj(y)


def forward_with_cache(model, idx_new: torch.Tensor, cache: KVCache) -> torch.Tensor:
    """Run ``model`` (a :class:`~llmre.model.gpt.GPT`) on new tokens only.

    Args:
        model: the course GPT, in eval mode (dropout is not applied here).
        idx_new: ``(B, T_new)`` int64 ids of the positions *after* the cached ones.
        cache: the :class:`KVCache` for this batch; it grows by ``T_new`` positions.

    Returns:
        ``(B, T_new, V)`` logits, equal (up to float rounding) to the last ``T_new``
        positions of ``model(full_sequence)[0]``.
    """
    B, T = idx_new.shape
    start = cache.length
    assert start + T <= model.cfg.block_size, (
        "learned absolute positions: the cached sequence cannot exceed block_size"
    )
    pos = torch.arange(start, start + T, dtype=torch.long, device=idx_new.device)
    x = model.transformer.wte(idx_new) + model.transformer.wpe(pos)    # (B, T, C)
    for layer, block in enumerate(model.transformer.h):
        x = x + _attention_with_cache(block.attn, block.ln_1(x), cache, layer, start)
        x = x + block.mlp(block.ln_2(x))
    x = model.transformer.ln_f(x)
    return model.lm_head(x)                                            # (B, T, V)


def _sample(logits: torch.Tensor, temperature: float, top_k, greedy: bool, generator):
    """Pick the next token from ``(B, V)`` logits exactly as ``GPT.generate`` does."""
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens, temperature=1.0, top_k=None,
                    greedy=False, generator=None, return_cache=False):
    """``GPT.generate`` with a KV cache: one prefill, then one-token decode steps.

    Args:
        model: the course GPT (eval mode).
        idx: ``(B, T0)`` int64 prompt.
        max_new_tokens: tokens to append. ``T0 + max_new_tokens`` must be ``<= block_size``
            (no context cropping: with learned absolute positions a cropped window would
            shift every position, so the whole cache would have to be recomputed).
        temperature, top_k: as in ``GPT.generate``.
        greedy: take the argmax instead of sampling.
        generator: optional ``torch.Generator`` for reproducible sampling.
        return_cache: also return the final :class:`KVCache`.

    Returns:
        ``(B, T0 + max_new_tokens)`` int64 (and the cache if requested).
    """
    cache = KVCache(model.cfg.n_layer)
    logits = forward_with_cache(model, idx, cache)[:, -1, :]          # prefill: (B, V)
    for step in range(max_new_tokens):
        nxt = _sample(logits, temperature, top_k, greedy, generator)  # (B, 1)
        idx = torch.cat([idx, nxt], dim=1)
        if step < max_new_tokens - 1:
            logits = forward_with_cache(model, nxt, cache)[:, -1, :]  # decode: 1 new token
    return (idx, cache) if return_cache else idx


# --------------------------------------------------------------------------- cost model


def kv_cache_bytes(n_layer, n_kv_head, d_head, seq_len, batch=1, bytes_per_elem=2) -> int:
    """KV-cache size ``2 * n_layer * n_kv_head * d_head * seq_len * batch * bytes`` (12.3)."""
    return 2 * n_layer * n_kv_head * d_head * seq_len * batch * bytes_per_elem


def matmul_intensity(tokens: int, d_in: int, d_out: int, bytes_per_elem=2) -> float:
    """Arithmetic intensity (FLOPs/byte) of ``(tokens, d_in) @ (d_in, d_out)``.

    FLOPs ``2 * tokens * d_in * d_out``; bytes = read the weight, read the input, write the
    output. For ``tokens << d_in, d_out`` the weight read dominates and the intensity tends
    to ``2 * tokens / bytes_per_elem``, i.e. about ``tokens`` FLOPs/byte in bf16: decode (``tokens = batch``) is memory-bound,
    prefill (``tokens = batch * prompt_len``) is compute-bound.
    """
    flops = 2 * tokens * d_in * d_out
    nbytes = bytes_per_elem * (d_in * d_out + tokens * d_in + tokens * d_out)
    return flops / nbytes


def roofline_time(flops: float, nbytes: float, peak_flops: float, bandwidth: float) -> float:
    """Lower-bound time of a step: ``max(flops / peak, bytes / bandwidth)`` (08.2)."""
    return max(flops / peak_flops, nbytes / bandwidth)


# --------------------------------------------------------------------------- batching


def simulate_static_batching(lengths, max_batch):
    """Request-level batching: take ``max_batch`` requests, decode until *all* finish.

    Args:
        lengths: number of output tokens each request needs, in arrival order.
        max_batch: number of sequences decoded together.

    Returns:
        dict with ``steps`` (decode iterations) and ``utilization`` = useful token slots /
        total slots (``sum(lengths) / (steps * max_batch)``).
    """
    steps = 0
    for s in range(0, len(lengths), max_batch):
        steps += max(lengths[s:s + max_batch])
    return {"steps": steps, "utilization": sum(lengths) / (steps * max_batch)}


def simulate_continuous_batching(lengths, max_batch):
    """Iteration-level batching: a finished sequence's slot is refilled at the next step.

    Same arguments and return keys as :func:`simulate_static_batching`. Prefill cost is
    ignored (both schedulers would pay it); each step every active sequence emits one token.
    """
    waiting = list(lengths)
    active: list[int] = []
    steps = 0
    while waiting or active:
        while waiting and len(active) < max_batch:
            active.append(waiting.pop(0))
        steps += 1
        active = [r - 1 for r in active if r - 1 > 0]
    return {"steps": steps, "utilization": sum(lengths) / (steps * max_batch)}


class PagedKVAllocator:
    """Block-table bookkeeping of a paged KV cache (vLLM's PagedAttention manager).

    GPU KV memory is cut into ``num_blocks`` blocks of ``block_size`` token slots. Each
    sequence owns a *block table*: the list of physical block ids holding its tokens in
    order. A new block is taken only when the last one is full, so the waste per sequence is
    at most ``block_size - 1`` slots, instead of reserving ``max_len`` slots up front.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.tables: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}

    def append_token(self, seq_id: int) -> tuple[int, int]:
        """Reserve the slot for one more token; returns its ``(physical_block, offset)``.

        Raises ``MemoryError`` when no block is free (a real scheduler would preempt).
        """
        n = self.lengths.get(seq_id, 0)
        table = self.tables.setdefault(seq_id, [])
        if n % self.block_size == 0:
            if not self.free_blocks:
                raise MemoryError("out of KV blocks")
            table.append(self.free_blocks.pop(0))
        self.lengths[seq_id] = n + 1
        return table[n // self.block_size], n % self.block_size

    def free(self, seq_id: int) -> None:
        """Return all of a finished sequence's blocks to the free list."""
        self.free_blocks.extend(self.tables.pop(seq_id, []))
        self.lengths.pop(seq_id, None)

    def reserved_slots(self) -> int:
        """Token slots held by live sequences (used + internal fragmentation)."""
        return sum(len(t) for t in self.tables.values()) * self.block_size


# --------------------------------------------------------------------------- speculative


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``norm(max(p - q, 0))`` over the last axis: where to resample after a rejection."""
    r = torch.clamp(p - q, min=0.0)
    return r / r.sum(dim=-1, keepdim=True)


def speculative_first_token_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Exact distribution of the first token emitted by one draft-and-verify round.

    ``P(x) = q(x) * min(1, p(x)/q(x)) + (1 - sum_y min(p(y), q(y))) * residual(x)``,
    which simplifies to ``p(x)``: the procedure is lossless. Computed term by term (not
    simplified) so the test can check the algebra numerically.
    """
    accept = torch.minimum(p, q)                      # q(x) * min(1, p/q) = min(p, q)
    reject_mass = 1.0 - accept.sum(dim=-1, keepdim=True)
    if float(reject_mass.max()) <= 0:                 # p == q: never rejects
        return accept
    return accept + reject_mass * residual_distribution(p, q)


def expected_tokens_per_round(alpha: float, k: int) -> float:
    """Expected tokens emitted per target forward, ``(1 - alpha^(k+1)) / (1 - alpha)``.

    Assumes each draft token is accepted independently with probability ``alpha``
    (``alpha = sum_x min(p(x), q(x))``). ``alpha = 1`` gives ``k + 1``.
    """
    if alpha >= 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speculative_accept(p: torch.Tensor, q: torch.Tensor, draft: torch.Tensor,
                       generator=None, greedy: bool = False):
    """Verify ``k`` drafted tokens against the target.

    Args:
        p: ``(k + 1, V)`` target probabilities; row ``i`` is the target's distribution for
            the position of draft token ``i`` (row ``k`` is the bonus position after all).
        q: ``(k, V)`` draft probabilities the drafts were sampled from.
        draft: ``(k,)`` int64 drafted token ids.
        generator: optional ``torch.Generator`` for the uniform draws / resampling.
        greedy: accept iff the draft equals the target argmax; correct with the argmax.

    Returns:
        ``(n_accepted, next_token)``: the first ``n_accepted`` drafts are kept and
        ``next_token`` (an ``int``) is appended after them (a correction or the bonus).
    """
    k = draft.numel()
    for i in range(k):
        x = int(draft[i])
        if greedy:
            if x == int(p[i].argmax()):
                continue
            return i, int(p[i].argmax())
        r = torch.rand((), generator=generator)
        if float(r) < min(1.0, float(p[i, x]) / max(float(q[i, x]), 1e-30)):
            continue
        resid = residual_distribution(p[i], q[i])
        return i, int(torch.multinomial(resid, 1, generator=generator))
    if greedy:
        return k, int(p[k].argmax())
    return k, int(torch.multinomial(p[k], 1, generator=generator))


@torch.no_grad()
def speculative_generate(target, draft_model, idx, max_new_tokens, k=4,
                         temperature=1.0, greedy=False, generator=None):
    """Draft-and-verify generation with two course GPTs sharing a tokenizer (batch 1).

    Each round the draft proposes ``k`` tokens one at a time (``k`` cheap decode steps); the
    target scores all of them in **one** forward over ``k + 1`` new positions; then
    :func:`speculative_accept` keeps a prefix and appends one more token. Both KV caches are
    truncated back to the committed sequence after every round.

    Invariant: each cache holds a prefix of the committed tokens; before a forward we feed
    the committed tokens it is missing.

    Returns:
        ``(tokens, stats)``: ``(1, T0 + max_new_tokens)`` int64 and a dict with
        ``rounds``, ``drafted`` and ``accepted`` counts.
    """
    assert idx.size(0) == 1, "batch 1 keeps the bookkeeping readable"
    seq = idx[0].tolist()
    goal = len(seq) + max_new_tokens
    assert goal + k <= min(target.cfg.block_size, draft_model.cfg.block_size)
    t_cache = KVCache(target.cfg.n_layer)
    d_cache = KVCache(draft_model.cfg.n_layer)
    stats = {"rounds": 0, "drafted": 0, "accepted": 0}

    def probs(logits):
        if greedy:
            return F.one_hot(logits.argmax(-1), logits.size(-1)).to(logits.dtype)
        return F.softmax(logits / temperature, dim=-1)

    def as_ids(tokens):
        return torch.tensor([tokens], dtype=torch.long, device=idx.device)

    while len(seq) < goal:
        # 1. Draft k tokens autoregressively with the small model.
        logits = forward_with_cache(draft_model, as_ids(seq[d_cache.length:]), d_cache)[0, -1]
        drafts, q_rows = [], []
        for i in range(k):
            q_i = probs(logits)
            x = int(q_i.argmax()) if greedy else int(torch.multinomial(q_i, 1, generator=generator))
            drafts.append(x)
            q_rows.append(q_i)
            if i < k - 1:
                logits = forward_with_cache(draft_model, as_ids([x]), d_cache)[0, -1]
        # 2. Score every draft with ONE target forward over the missing tokens + drafts.
        new = seq[t_cache.length:] + drafts
        t_logits = forward_with_cache(target, as_ids(new), t_cache)[0]      # (len(new), V)
        p = probs(t_logits[-(k + 1):])                                     # (k + 1, V)
        # 3. Accept a prefix, append the correction / bonus token.
        n_acc, nxt = speculative_accept(p, torch.stack(q_rows), torch.tensor(drafts),
                                        generator=generator, greedy=greedy)
        committed = len(seq)
        seq = seq + drafts[:n_acc] + [nxt]
        t_cache.truncate(committed + n_acc)       # drop rejected drafts' keys/values
        d_cache.truncate(committed + n_acc)
        stats["rounds"] += 1
        stats["drafted"] += k
        stats["accepted"] += n_acc
    return as_ids(seq[:goal]), stats
