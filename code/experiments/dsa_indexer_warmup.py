"""DSA dense warm-up on CPU: train a lightning indexer to imitate a frozen attention layer.

Runnable companion to lesson 12.6 (`lessons/module-12/lesson-06.md`). A ``DSAttention``
layer (d=32, 2 heads, one shared KV head) with random, frozen main weights (queries scaled
up so its attention is peaked) reads sequences of 64 tokens drawn from a 16-token
vocabulary. Only the indexer (4 heads x 8 dims) is trained, with DSA's dense-warm-up loss
KL(p_t || softmax(I_t)) on fresh random sequences each step. Every 100 steps it prints, on
held-out sequences:

* the indexer KL,
* recall@8: the fraction of the attention's own top-8 keys that the indexer also selects,
* the relative error of the sparse (top-8) layer output against the dense one, and
* the same error for an *oracle* selection (the attention's true top-8): the floor that a
  perfect indexer could reach with only 8 of 64 keys.

It is NOT a pytest test::

    cd code && PYTHONPATH=src python experiments/dsa_indexer_warmup.py

Runs in a few seconds on a laptop CPU. The digits depend on the seed and PyTorch version.
"""

from __future__ import annotations

import torch

from llmre.attention.sparse import DSAttention, recall_at_k, sparse_attention, topk_selection_mask
from llmre.model.config import GPTConfig

torch.manual_seed(0)
C, V, T, K = 32, 16, 64, 8
emb = torch.randn(V, C)  # fixed token embeddings: the indexer can learn token-level structure
layer = DSAttention(GPTConfig(n_embd=C, n_head=2, bias=False), n_kv_head=1, n_index_head=4, d_index=8, topk=K)
with torch.no_grad():
    layer.q_proj.weight.mul_(6.0)  # make the frozen attention peaked, like a trained layer
for name, p in layer.named_parameters():
    p.requires_grad_(name.startswith("idx_"))  # stage 1: everything frozen but the indexer
opt = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=1e-2)
x_test = emb[torch.randint(V, (16, T))]


def oracle_error(x: torch.Tensor) -> float:
    """Relative error when the selection is the attention's own top-K (a perfect indexer)."""
    _, aux = layer(x, dense=True, return_aux=True)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    select = topk_selection_mask(aux["target"].masked_fill(future, float("-inf")), K)
    q, k, v = layer._qkv(x)
    y = sparse_attention(q, k, v, select).transpose(1, 2).reshape(x.shape)
    y_dense = layer(x, dense=True)
    return ((layer.c_proj(y) - y_dense).norm() / y_dense.norm()).item()


def report(step: int) -> None:
    with torch.no_grad():
        _, aux = layer(x_test, dense=True, return_aux=True)
        y_dense, y_sparse = layer(x_test, dense=True), layer(x_test)
        rel = ((y_sparse - y_dense).norm() / y_dense.norm()).item()
        rec = recall_at_k(aux["index_scores"], aux["target"], K)
        print(f"step {step:3d}  KL {aux['indexer_loss'].item():6.2f}  recall@{K} {rec:.2f}  "
              f"sparse rel.err {rel:.3f}  oracle rel.err {oracle_error(x_test):.3f}")


for step in range(301):
    if step % 100 == 0:
        report(step)
    x = emb[torch.randint(V, (16, T))]  # fresh random token sequences each step
    _, aux = layer(x, dense=True, return_aux=True)
    opt.zero_grad()
    aux["indexer_loss"].backward()
    opt.step()
