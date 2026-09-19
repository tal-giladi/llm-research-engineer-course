"""The assembled GPT-2 model: embeddings + blocks + head, with generation.

This is Module 6. Module 5 built the pieces — the token/positional embeddings,
causal multi-head attention (:mod:`llmre.attention.attention`), and the pre-norm
:class:`~llmre.model.block.Block`. Here we stack them into the full model:

    idx (B, T) long
      -> token embedding wte + positional embedding wpe            (B, T, C)
      -> dropout
      -> n_layer x Block                                          (B, T, C)
      -> final LayerNorm ln_f                                     (B, T, C)
      -> lm_head (tied to wte)                       logits        (B, T, V)

Two design choices carry real weight (both explained in lessons 06.1 and 06.2):

* **Weight tying** — ``transformer.wte.weight`` and ``lm_head.weight`` are the
  *same* tensor. The input embedding maps a token id to a vector; the output head
  maps a vector back to a score per token. Tying them says "the code for a token
  is the same going in and coming out", saving ~40M parameters in GPT-2 small and
  usually improving quality.
* **Scaled residual init** — the projections that *write into* the residual
  stream (``attn.c_proj`` and ``mlp.c_proj``) are initialised with a smaller std,
  ``0.02 / sqrt(2 * n_layer)``, so the residual stream's variance does not grow
  as we stack more layers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmre.model.block import Block, LayerNorm
from llmre.model.config import GPTConfig


class GPT(nn.Module):
    """A decoder-only GPT-2 style language model.

    Args:
        cfg: a :class:`~llmre.model.config.GPTConfig` fixing every tensor shape
            (``vocab_size`` ``V``, ``block_size`` ``T_max``, ``n_layer``,
            ``n_head``, ``n_embd`` ``C``, ``dropout``, ``bias``).

    Submodules (grouped under ``self.transformer`` to mirror nanoGPT / HF names):
        ``wte``: token embedding, ``nn.Embedding(V, C)``.
        ``wpe``: positional embedding, ``nn.Embedding(T_max, C)``.
        ``drop``: input dropout.
        ``h``: ``ModuleList`` of ``n_layer`` :class:`Block`.
        ``ln_f``: final :class:`LayerNorm` over ``C``.
    And ``self.lm_head``: ``nn.Linear(C, V, bias=False)`` whose ``.weight`` is the
    same tensor as ``transformer.wte.weight`` (weight tying).

    Shapes:
        forward input ``idx`` is ``(B, T)`` int64 (token ids); output logits are
        ``(B, T, V)`` float on the same device as the parameters.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.vocab_size is not None
        assert cfg.block_size is not None
        self.cfg = cfg

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=LayerNorm(cfg.n_embd, bias=cfg.bias),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: the un-embedding IS the transpose of the embedding. We
        # make them the *same* Parameter object so they share storage and one
        # gradient. (nn.Linear stores weight as (V, C); nn.Embedding as (V, C);
        # same shape, so we can point one at the other directly.)
        self.transformer.wte.weight = self.lm_head.weight

        # GPT-2 initialisation: N(0, 0.02) for all Linear/Embedding weights,
        # zeros for biases.
        self.apply(self._init_weights)
        # Scaled residual init: shrink the std of the projections that write into
        # the residual stream by 1/sqrt(2 * n_layer). There are 2 such adds per
        # layer (attn + mlp), hence the factor 2.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise one submodule in place (called by ``self.apply``)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        """Total number of trainable parameters.

        Args:
            non_embedding: if ``True``, subtract the positional-embedding table
                ``wpe`` (nanoGPT reports this "non-embedding" figure). The token
                table ``wte`` is *not* subtracted because it is tied into the
                ``lm_head`` and so is genuinely used by the output.

        Returns:
            The parameter count. ``model.parameters()`` de-duplicates tied
            tensors, so the shared ``wte``/``lm_head`` weight is counted once.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        """Run the model, returning logits and (optionally) the loss.

        Args:
            idx: token ids, shape ``(B, T)`` int64, with ``T <= block_size``, on
                any device the parameters live on.
            targets: optional next-token ids, shape ``(B, T)`` int64. Entry
                ``targets[b, t]`` is the token that should follow ``idx[b, t]``.
                Positions equal to ``-1`` are ignored in the loss.

        Returns:
            ``(logits, loss)`` where ``logits`` is ``(B, T, V)`` float and
            ``loss`` is a scalar mean cross-entropy tensor when ``targets`` is
            given, else ``None``.
        """
        B, T = idx.shape
        assert (
            T <= self.cfg.block_size
        ), f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        # Positions 0..T-1, one per column; shape (T,) on the same device as idx.
        pos = torch.arange(T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)   # (B, T) -> (B, T, C): id -> vector
        pos_emb = self.transformer.wpe(pos)   # (T,)   -> (T, C):    position -> vector
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, C); pos_emb broadcasts over B
        for block in self.transformer.h:
            x = block(x)                      # (B, T, C) -> (B, T, C)
        x = self.transformer.ln_f(x)          # final norm, (B, T, C)
        logits = self.lm_head(x)              # (B, T, C) -> (B, T, V)

        loss = None
        if targets is not None:
            # Flatten batch and time into one axis of examples for cross-entropy:
            # logits (B*T, V) vs targets (B*T,). ignore_index=-1 skips pad targets.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively extend a context by sampling one token at a time.

        Args:
            idx: seed context, shape ``(B, T0)`` int64. The generated tokens are
                appended to it.
            max_new_tokens: how many tokens to append.
            temperature: softmax temperature applied to the last-position logits.
                ``T < 1`` sharpens the distribution (more greedy), ``T > 1``
                flattens it (more random), ``T == 1`` leaves it unchanged.
            top_k: if given, keep only the ``k`` highest-logit tokens as
                candidates (the rest get probability 0) before sampling.

        Returns:
            ``(B, T0 + max_new_tokens)`` int64 — the seed with the new tokens
            appended. The context fed to the model is cropped to the last
            ``block_size`` tokens each step, so it never exceeds ``block_size``.
        """
        for _ in range(max_new_tokens):
            # Crop the context so we never feed more than block_size positions.
            idx_cond = (
                idx
                if idx.size(1) <= self.cfg.block_size
                else idx[:, -self.cfg.block_size :]
            )
            logits, _ = self(idx_cond)          # (B, T, V)
            # Only the last position predicts the next token; scale by temperature.
            logits = logits[:, -1, :] / temperature  # (B, V)
            if top_k is not None:
                # Keep the k largest logits; set everything below the k-th to -inf.
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)        # v: (B, k), sorted descending
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)       # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # grow by one column
        return idx
