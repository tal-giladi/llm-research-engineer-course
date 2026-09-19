"""Tests for llmre.model.gpt — the assembled GPT-2 model.

These back Module 6 (`lessons/module-06/lesson-01.md` .. `lesson-03.md`):

* forward returns logits ``(B, T, V)`` and a scalar loss when targets are given;
* the token embedding and the LM head share storage (weight tying);
* the parameter count matches a closed-form formula;
* at initialisation the loss on random targets is about ``ln(vocab_size)``;
* ``generate`` extends the context by exactly ``max_new_tokens`` and never feeds
  more than ``block_size`` positions to the model.
"""

from __future__ import annotations

import math

import torch

from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT


def _cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=100, block_size=16, n_layer=3, n_head=4, n_embd=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def _expected_params(cfg: GPTConfig) -> int:
    """Closed-form parameter count for the (weight-tied) GPT.

    C = n_embd, V = vocab_size, b = 1 if bias else 0.
      embeddings : C*(V + block_size)          # wte (tied into lm_head) + wpe
      per block  : 12*C^2 + 2*C + 11*b*C        # 12C^2 weights, biases, 2 LayerNorms
      final      : C*(1 + b)                    # ln_f
    """
    C = cfg.n_embd
    V = cfg.vocab_size
    b = 1 if cfg.bias else 0
    emb = C * (V + cfg.block_size)
    per_block = 12 * C * C + 2 * C + 11 * b * C
    ln_f = C * (1 + b)
    return emb + cfg.n_layer * per_block + ln_f


def test_forward_shapes_and_loss():
    torch.manual_seed(0)
    cfg = _cfg()
    model = GPT(cfg).eval()
    B, T = 2, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0

    logits_only, none_loss = model(idx)
    assert logits_only.shape == (B, T, cfg.vocab_size)
    assert none_loss is None


def test_weight_tying_shares_storage():
    cfg = _cfg()
    model = GPT(cfg)
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
    assert model.lm_head.weight.shape == (cfg.vocab_size, cfg.n_embd)


def test_param_count_matches_formula():
    cfg = _cfg()
    model = GPT(cfg)
    assert model.num_params() == _expected_params(cfg)
    # Sanity: the tied weight is counted exactly once.
    assert sum(p.numel() for p in model.parameters()) == _expected_params(cfg)


def test_bias_free_param_count_matches_formula():
    cfg = _cfg(bias=False)
    model = GPT(cfg)
    assert model.num_params() == _expected_params(cfg)


def test_init_loss_is_about_ln_vocab():
    torch.manual_seed(0)
    cfg = _cfg(vocab_size=200)
    model = GPT(cfg).eval()
    B, T = 4, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    _, loss = model(idx, targets)
    # An untrained model predicts roughly uniform, so loss ~= ln(V).
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5


def test_generate_extends_and_respects_block_size():
    torch.manual_seed(0)
    cfg = _cfg(block_size=8)
    model = GPT(cfg).eval()
    B, T0 = 2, 3
    idx = torch.randint(0, cfg.vocab_size, (B, T0))
    out = model.generate(idx, max_new_tokens=10, temperature=1.0, top_k=None)
    assert out.shape == (B, T0 + 10)
    # All generated ids are valid token ids.
    assert out.min().item() >= 0 and out.max().item() < cfg.vocab_size
    # The seed is preserved as the prefix.
    assert torch.equal(out[:, :T0], idx)


def test_generate_greedy_is_deterministic():
    torch.manual_seed(0)
    cfg = _cfg()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    # top_k=1 forces the argmax token, so two runs must agree.
    a = model.generate(idx, max_new_tokens=5, top_k=1)
    b = model.generate(idx, max_new_tokens=5, top_k=1)
    assert torch.equal(a, b)


def test_generate_never_exceeds_block_size_context(monkeypatch):
    torch.manual_seed(0)
    cfg = _cfg(block_size=8)
    model = GPT(cfg).eval()
    seen_lengths = []
    real_forward = model.forward

    def spy(idx, targets=None):
        seen_lengths.append(idx.size(1))
        return real_forward(idx, targets)

    monkeypatch.setattr(model, "forward", spy)
    idx = torch.randint(0, cfg.vocab_size, (1, 6))
    model.generate(idx, max_new_tokens=20)
    assert max(seen_lengths) <= cfg.block_size
