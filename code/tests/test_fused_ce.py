"""Tests for llmre.training.fused_ce (lesson 08.5): every path equals the naive loss."""

import math

import pytest
import torch
import torch.nn.functional as F

from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.training.fused_ce import (
    chunked_linear_cross_entropy,
    fused_logprobs_and_entropy,
    logits_bytes,
    naive_linear_cross_entropy,
    online_logsumexp_linear,
    vocab_tiled_linear_cross_entropy,
)


def _problem(N=37, C=16, V=101, seed=0, n_ignored=5):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(N, C, generator=g, dtype=torch.float64)
    w = torch.randn(V, C, generator=g, dtype=torch.float64) * 0.5
    t = torch.randint(0, V, (N,), generator=g)
    t[:n_ignored] = -100
    return h, w, t


def _loss_and_grads(fn, h, w, t, **kw):
    h = h.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    loss = fn(h, w, t, **kw)
    (2.5 * loss).backward()  # non-unit upstream gradient checks the grad_out scaling
    return loss.detach(), h.grad, w.grad


def test_hand_worked_example():
    # Lesson 08.5 section 3: h=[1,2], four vocab rows, target 1, two tiles of 2.
    h = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    t = torch.tensor([1])
    assert online_logsumexp_linear(h, w, block_v=2).item() == pytest.approx(3.419717, abs=1e-6)
    loss, gh, gw = _loss_and_grads(vocab_tiled_linear_cross_entropy, h, w, t, block_v=2)
    assert loss.item() == pytest.approx(1.419717, abs=1e-6)
    assert torch.allclose(gh / 2.5, torch.tensor([[0.734142, -0.100984]], dtype=torch.float64), atol=1e-5)
    lp, ent = fused_logprobs_and_entropy(h, w, t, block_v=2)
    assert lp.item() == pytest.approx(-1.419717, abs=1e-5)
    assert ent.item() == pytest.approx(0.887543, abs=1e-5)


@pytest.mark.parametrize("chunk", [1, 7, 37, 1000])
def test_chunked_matches_naive_loss_and_grads(chunk):
    h, w, t = _problem()
    ref = _loss_and_grads(naive_linear_cross_entropy, h, w, t)
    got = _loss_and_grads(chunked_linear_cross_entropy, h, w, t, chunk_size=chunk)
    for a, b in zip(ref, got):
        assert torch.allclose(a.double(), b.double(), atol=1e-6)


@pytest.mark.parametrize("block_v", [1, 8, 50, 101, 4096])
def test_vocab_tiled_matches_naive_loss_and_grads(block_v):
    h, w, t = _problem()
    ref = _loss_and_grads(naive_linear_cross_entropy, h, w, t)
    got = _loss_and_grads(vocab_tiled_linear_cross_entropy, h, w, t, block_v=block_v)
    for a, b in zip(ref, got):
        assert torch.allclose(a.double(), b.double(), atol=1e-6)


def test_naive_matches_torch_reference():
    h, w, t = _problem()
    ref = F.cross_entropy(h @ w.t(), t, ignore_index=-100)
    assert torch.allclose(naive_linear_cross_entropy(h, w, t), ref)


def test_online_logsumexp_is_exact_and_stable():
    h, w, _ = _problem()
    h = h * 40.0  # logits in the hundreds: naive exp would overflow fp32
    ref = torch.logsumexp(h @ w.t(), dim=-1)
    for bv in (3, 64, 1000):
        assert torch.allclose(online_logsumexp_linear(h, w, bv).double(), ref, atol=1e-4)
    h32, w32 = h.float(), w.float()
    assert torch.isfinite(online_logsumexp_linear(h32, w32, 16)).all()


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_logprobs_and_entropy_match_full_softmax(temperature):
    h, w, t = _problem(n_ignored=0)
    logits = (h @ w.t()) / temperature
    logp = torch.log_softmax(logits, dim=-1)
    ref_lp = logp.gather(-1, t[:, None]).squeeze(-1)
    ref_ent = -(logp.exp() * logp).sum(-1)
    lp, ent = fused_logprobs_and_entropy(h, w, t, block_v=16, temperature=temperature)
    assert torch.allclose(lp.double(), ref_lp, atol=1e-5)
    assert torch.allclose(ent.double(), ref_ent, atol=1e-5)


def test_float32_inputs_keep_dtype_and_match():
    h, w, t = _problem()
    h, w = h.float(), w.float()
    ref = naive_linear_cross_entropy(h, w, t)
    a = chunked_linear_cross_entropy(h, w, t, chunk_size=8)
    b = vocab_tiled_linear_cross_entropy(h, w, t, block_v=16)
    assert a.dtype == b.dtype == torch.float32
    assert torch.allclose(a, ref, atol=1e-5) and torch.allclose(b, ref, atol=1e-5)


def test_reduces_to_gpt_loss_with_tied_weights():
    # The foundational loss in llmre.model.gpt, recomputed from the final hidden
    # states and the tied lm_head weight, with GPT's ignore_index=-1.
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=61, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT(cfg).eval()
    idx = torch.randint(0, 61, (2, 8))
    tgt = torch.randint(0, 61, (2, 8))
    tgt[0, :3] = -1
    _, ref = model(idx, tgt)
    feats = {}
    model.transformer.ln_f.register_forward_hook(lambda m, i, o: feats.__setitem__("x", o))
    model(idx)
    x = feats["x"].reshape(-1, 16)
    w = model.lm_head.weight
    for loss in (
        chunked_linear_cross_entropy(x, w, tgt.reshape(-1), chunk_size=5, ignore_index=-1),
        vocab_tiled_linear_cross_entropy(x, w, tgt.reshape(-1), block_v=7, ignore_index=-1),
    ):
        assert loss.item() == pytest.approx(ref.item(), abs=1e-5)


def test_logits_bytes_matches_lesson_numbers():
    # Qwen-size vocabulary, 8 x 4096 tokens, fp32: about 19.9 GB (lesson 17.4 / 08.5).
    assert logits_bytes(8 * 4096, 151_936, 4) / 1e9 == pytest.approx(19.91, abs=0.01)
    assert logits_bytes(1024, 151_936, 4) / 1e6 == pytest.approx(622.3, abs=0.1)


def test_bad_arguments_raise():
    h, w, t = _problem()
    with pytest.raises(ValueError):
        chunked_linear_cross_entropy(h, w, t, chunk_size=0)
    with pytest.raises(ValueError):
        vocab_tiled_linear_cross_entropy(h, w[:, :3], t)
    with pytest.raises(ValueError):
        fused_logprobs_and_entropy(h, w, t.clamp(min=0), temperature=0.0)
    assert math.isfinite(chunked_linear_cross_entropy(h, w, t).item())
