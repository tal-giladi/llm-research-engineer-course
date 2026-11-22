"""Tests for lesson 12.6: DeepSeek Sparse Attention (lightning indexer + top-k selection)."""

import math

import torch

from llmre.attention.gqa import GroupedQueryAttention
from llmre.attention.sparse import (
    DSAttention,
    attention_target_distribution,
    indexer_kl_loss,
    lightning_indexer_scores,
    per_pair_macs,
    recall_at_k,
    selected_pairs,
    sparse_attention,
    sparse_attention_gather,
    topk_selection_mask,
)
from llmre.model.config import GPTConfig


def _hand_example():
    """The 4-token example of lesson 12.6 section 3 (query = last token)."""
    k_idx = torch.tensor([[2.0, 0.0], [-1.0, 1.0], [0.5, 0.0], [1.0, 1.0]])[None]  # (1, 4, 2)
    q_last = torch.tensor([[1.0, 0.0], [0.0, 2.0]])  # H_I = 2 heads, d_I = 2
    q_idx = torch.zeros(1, 4, 2, 2)
    q_idx[0, 3] = q_last
    w = torch.zeros(1, 4, 2)
    w[0, 3] = torch.tensor([1.0, 0.5])
    return q_idx, k_idx, w


def test_hand_worked_index_scores_and_selection():
    I = lightning_indexer_scores(*_hand_example())
    assert torch.allclose(I[0, 3], torch.tensor([2.0, 1.0, 0.5, 2.0]))
    sel = topk_selection_mask(I, 2)
    assert sel[0, 3].tolist() == [True, False, False, True]


def test_hand_worked_main_attention_dense_vs_sparse():
    # Main attention logits for the last query are (ln 4, 0, 0, ln 4): q.k / sqrt(2).
    q = torch.zeros(1, 1, 4, 2)
    q[0, 0, 3] = torch.tensor([math.log(4) * math.sqrt(2), 0.0])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]])[None, None]
    v = torch.tensor([4.0, 8.0, 0.0, 2.0])[None, None, :, None]
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))[None]
    dense = sparse_attention(q, k, v, causal)[0, 0, 3, 0]
    assert abs(dense.item() - 3.2) < 1e-5
    I = lightning_indexer_scores(*_hand_example())
    sparse = sparse_attention(q, k, v, topk_selection_mask(I, 2))[0, 0, 3, 0]
    assert abs(sparse.item() - 3.0) < 1e-5


def test_hand_worked_indexer_kl():
    p = torch.tensor([0.4, 0.1, 0.1, 0.4])
    I = torch.tensor([2.0, 1.0, 0.5, 2.0])
    kl = indexer_kl_loss(p[None, None], I[None, None])
    expected = (p * (p / I.softmax(-1)).log()).sum()
    assert abs(kl.item() - expected.item()) < 1e-6
    assert abs(kl.item() - 0.0085) < 1e-4


def test_selection_is_causal_and_has_min_t_plus_1_k_entries():
    torch.manual_seed(0)
    T, k = 12, 5
    I = lightning_indexer_scores(torch.randn(2, T, 3, 4), torch.randn(2, T, 4), torch.randn(2, T, 3))
    sel = topk_selection_mask(I, k)
    assert not (sel & torch.triu(torch.ones(T, T, dtype=torch.bool), 1)).any()
    assert sel.sum(-1).tolist() == [[min(t + 1, k) for t in range(T)]] * 2


def test_gather_kernel_equals_mask_reference():
    torch.manual_seed(1)
    B, H, T, d, k = 2, 3, 10, 4, 4
    q, kk, v = torch.randn(B, H, T, d), torch.randn(B, H, T, d), torch.randn(B, H, T, d)
    I = lightning_indexer_scores(torch.randn(B, T, 2, 3), torch.randn(B, T, 3), torch.rand(B, T, 2))
    ref = sparse_attention(q, kk, v, topk_selection_mask(I, k))
    assert torch.allclose(sparse_attention_gather(q, kk, v, I, k), ref, atol=1e-6)


def test_topk_at_least_T_reduces_to_dense_gqa():
    torch.manual_seed(2)
    cfg = GPTConfig(n_embd=16, n_head=4, bias=False)
    dsa = DSAttention(cfg, n_kv_head=2, topk=64)
    gqa = GroupedQueryAttention(cfg, n_kv_head=2)
    gqa.load_state_dict({n: p for n, p in dsa.state_dict().items() if not n.startswith("idx_")})
    x = torch.randn(2, 9, 16)
    assert torch.allclose(dsa(x), gqa(x), atol=1e-6)
    assert torch.allclose(dsa(x, dense=True), gqa(x), atol=1e-6)


def test_layer_is_causal():
    torch.manual_seed(3)
    layer = DSAttention(GPTConfig(n_embd=16, n_head=4, bias=False), topk=3)
    x = torch.randn(1, 8, 16)
    y = layer(x)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 3, 16)
    assert torch.allclose(layer(x2)[:, :5], y[:, :5], atol=1e-6)


def test_gradients_are_separated():
    """LM loss trains only the main attention; indexer KL trains only the indexer."""
    torch.manual_seed(4)
    layer = DSAttention(GPTConfig(n_embd=16, n_head=4, bias=False), topk=3)
    x = torch.randn(2, 8, 16, requires_grad=True)
    y, aux = layer(x, return_aux=True)
    y.pow(2).mean().backward(retain_graph=True)
    assert all(p.grad is None for n, p in layer.named_parameters() if n.startswith("idx_"))
    assert layer.q_proj.weight.grad is not None
    layer.zero_grad(set_to_none=True)
    x.grad = None
    aux["indexer_loss"].backward()
    assert all(p.grad is None for n, p in layer.named_parameters() if not n.startswith("idx_"))
    assert x.grad is None
    assert layer.idx_q.weight.grad is not None


def test_target_rows_sum_to_one_and_kl_zero_on_match():
    probs = torch.rand(2, 3, 5, 5).tril()
    target = attention_target_distribution(probs / probs.sum(-1, keepdim=True))
    assert torch.allclose(target.sum(-1), torch.ones(2, 5))
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), 1)
    I = target.clamp_min(1e-12).log().masked_fill(future, float("-inf"))
    assert indexer_kl_loss(target, I).abs().item() < 1e-5


def test_dense_warmup_trains_indexer_to_imitate_attention():
    torch.manual_seed(5)
    layer = DSAttention(GPTConfig(n_embd=32, n_head=4, bias=False), n_index_head=4, d_index=8, topk=4)
    for n, p in layer.named_parameters():
        p.requires_grad_(n.startswith("idx_"))  # stage 1: everything frozen but the indexer
    opt = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(8, 24, 32)
    _, aux = layer(x, dense=True, return_aux=True)
    loss0, rec0 = aux["indexer_loss"].item(), recall_at_k(aux["index_scores"], aux["target"], 4)
    for _ in range(150):
        _, aux = layer(x, dense=True, return_aux=True)
        opt.zero_grad()
        aux["indexer_loss"].backward()
        opt.step()
    _, aux = layer(x, dense=True, return_aux=True)
    assert aux["indexer_loss"].item() < 0.5 * loss0
    assert recall_at_k(aux["index_scores"], aux["target"], 4) > rec0 + 0.2


def test_pair_and_cost_arithmetic():
    assert selected_pairs(4, 2) == (10, 7)
    dense, sparse = selected_pairs(131072, 2048)
    assert dense / sparse > 30
    # DeepSeek-V3.2 per-pair cost: MLA in MQA mode (128 heads, 576-dim keys, 512-dim values)
    # vs the indexer (64 heads x 128).
    assert per_pair_macs(128, 576, 512) == 139264
    assert per_pair_macs(64, 128, 0) == 8192
