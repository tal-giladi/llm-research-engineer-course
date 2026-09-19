"""GPTConfig — the single source of truth for model dimensions.

This dataclass is the stable contract shared by every module that touches the
model (Module 5's attention/block, Module 6's assembled GPT, Module 7's
training loop, ...). It carries no tensors and no logic: it is a plain bundle
of the hyperparameters that fix the *shapes* of every weight in the network.

The defaults are GPT-2 small (124M): ``vocab_size=50257`` BPE tokens, a
``block_size=1024`` maximum context, ``n_layer=12`` transformer blocks,
``n_head=12`` attention heads, and an embedding width of ``n_embd=768``
(so the per-head dimension is ``n_embd // n_head = 64``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Hyperparameters that determine every tensor shape in the GPT.

    Attributes:
        vocab_size: number ``V`` of distinct token ids the model can embed and
            predict. Sets the row count of the token-embedding table ``wte``
            (``V x n_embd``) and the output vocabulary of the final logits.
        block_size: maximum context length ``T`` (number of positions). Sets the
            row count of the positional-embedding table ``wpe``
            (``block_size x n_embd``) and the size of the causal mask.
        n_layer: number of stacked transformer ``Block``s.
        n_head: number of attention heads ``nh`` per block. Must divide
            ``n_embd`` evenly; the per-head dimension is ``n_embd // n_head``.
        n_embd: embedding / residual-stream width ``C`` (channels per position).
        dropout: dropout probability used inside attention and the MLP
            (``0.0`` disables it — the default for from-scratch study).
        bias: whether ``Linear`` and ``LayerNorm`` layers carry a bias term.
            GPT-2 uses ``True``; some modern models drop biases for speed.

    Note:
        This object holds only Python scalars — no tensors, no dtype, no device.
        It is cheap to copy and safe to pass everywhere the model shape is
        needed.
    """

    vocab_size: int = 50257
    block_size: int = 1024  # max context length T
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
