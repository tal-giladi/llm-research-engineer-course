"""Tests for lesson 08.6: inference serving foundations (llmre.model.serving).

The foundational reference is ``GPT.forward`` / ``GPT.generate`` from Module 6; every
cached path must reproduce it.
"""

from __future__ import annotations

import math

import pytest
import torch

from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.model.serving import (
    KVCache,
    PagedKVAllocator,
    expected_tokens_per_round,
    forward_with_cache,
    generate_cached,
    kv_cache_bytes,
    matmul_intensity,
    residual_distribution,
    roofline_time,
    simulate_continuous_batching,
    simulate_static_batching,
    speculative_accept,
    speculative_first_token_distribution,
    speculative_generate,
)


def _model(seed=0, **kw):
    base = dict(vocab_size=50, block_size=64, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    base.update(kw)
    torch.manual_seed(seed)
    m = GPT(GPTConfig(**base)).double().eval()
    return m


def _greedy_reference(model, idx, n):
    """The plain no-cache loop of 06.3, greedy: re-run the whole prefix each step."""
    for _ in range(n):
        logits, _ = model(idx)
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return idx


def test_prefill_matches_full_forward():
    m = _model()
    idx = torch.randint(0, 50, (3, 10))
    full, _ = m(idx)
    cache = KVCache(m.cfg.n_layer)
    assert torch.allclose(forward_with_cache(m, idx, cache), full, atol=1e-10)
    assert cache.length == 10


def test_prefill_then_decode_matches_full_forward():
    m = _model()
    idx = torch.randint(0, 50, (2, 12))
    full, _ = m(idx)
    cache = KVCache(m.cfg.n_layer)
    forward_with_cache(m, idx[:, :5], cache)                       # prefill 5 tokens
    for t in range(5, 12):                                         # decode one at a time
        step = forward_with_cache(m, idx[:, t:t + 1], cache)
        assert torch.allclose(step[:, 0], full[:, t], atol=1e-10)
    # a multi-token chunk on top of a cache (chunked prefill / verification) also matches
    cache.truncate(8)
    chunk = forward_with_cache(m, idx[:, 8:12], cache)
    assert torch.allclose(chunk, full[:, 8:12], atol=1e-10)


def test_generate_cached_greedy_equals_reference_loop():
    m = _model(seed=1)
    idx = torch.randint(0, 50, (2, 4))
    ref = _greedy_reference(m, idx, 20)
    out = generate_cached(m, idx, 20, greedy=True)
    assert torch.equal(out, ref)


def test_generate_cached_sampling_shape_and_cache_size():
    m = _model()
    idx = torch.randint(0, 50, (2, 3))
    g = torch.Generator().manual_seed(0)
    out, cache = generate_cached(m, idx, 7, top_k=5, generator=g, return_cache=True)
    assert out.shape == (2, 10)
    assert torch.equal(out[:, :3], idx)
    # last sampled token is never fed back, so 3 + 7 - 1 positions are cached
    assert cache.length == 9
    d_h = m.cfg.n_embd // m.cfg.n_head
    expected = kv_cache_bytes(m.cfg.n_layer, m.cfg.n_head, d_h, 9, batch=2, bytes_per_elem=8)
    assert cache.nbytes() == expected


def test_kv_cache_bytes_matches_lesson_12_3():
    # Llama-2-70B geometry from 12.3: MHA 10.737 GB, GQA-8 1.342 GB at T=4096, fp16
    assert kv_cache_bytes(80, 64, 128, 4096) / 1e9 == pytest.approx(10.737, abs=1e-3)
    assert kv_cache_bytes(80, 8, 128, 4096) / 1e9 == pytest.approx(1.342, abs=1e-3)


def test_matmul_intensity_decode_vs_prefill():
    d = 4096
    decode = matmul_intensity(1, d, d)
    prefill = matmul_intensity(4096, d, d)
    assert decode == pytest.approx(1.0, rel=1e-3)        # ~1 FLOP/byte in bf16
    assert prefill > 1000                                 # far right of an A100 ridge (~153)
    assert matmul_intensity(64, d, d) == pytest.approx(64 * d / (d + 2 * 64), rel=1e-9)  # t*d/(d+2t)


def test_roofline_time_takes_the_binding_ceiling():
    assert roofline_time(1e12, 1e9, 1e12, 1e12) == pytest.approx(1.0)      # compute-bound
    assert roofline_time(1e9, 1e12, 1e12, 1e12) == pytest.approx(1.0)      # memory-bound


def test_continuous_batching_beats_static_on_mixed_lengths():
    lengths = [10, 1, 1, 1, 10, 1, 1, 1]
    s = simulate_static_batching(lengths, 4)
    c = simulate_continuous_batching(lengths, 4)
    assert s["steps"] == 20
    assert c["steps"] == 11
    assert c["utilization"] > s["utilization"]
    # equal lengths: nothing to gain
    eq = [5] * 8
    assert simulate_static_batching(eq, 4)["steps"] == simulate_continuous_batching(eq, 4)["steps"] == 10


def test_paged_allocator_block_tables_and_free():
    a = PagedKVAllocator(num_blocks=4, block_size=4)
    slots = [a.append_token(0) for _ in range(5)]
    assert slots[0] == (0, 0) and slots[3] == (0, 3) and slots[4] == (1, 0)
    assert a.tables[0] == [0, 1]
    assert a.reserved_slots() == 8                       # 5 used + 3 internal fragmentation
    for _ in range(2):
        a.append_token(1)
    assert a.tables[1] == [2]
    a.free(0)
    assert sorted(a.free_blocks) == [0, 1, 3]
    a.append_token(1); a.append_token(1); a.append_token(1)   # fills block 2, needs a new one
    assert a.tables[1] == [2, 3]
    with pytest.raises(MemoryError):
        b = PagedKVAllocator(num_blocks=1, block_size=2)
        for _ in range(3):
            b.append_token(0)


def test_speculative_first_token_is_exactly_target():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        p = torch.softmax(torch.randn(7, generator=g, dtype=torch.float64) * 2, -1)
        q = torch.softmax(torch.randn(7, generator=g, dtype=torch.float64) * 2, -1)
        assert torch.allclose(speculative_first_token_distribution(p, q), p, atol=1e-12)
    p = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    assert torch.allclose(speculative_first_token_distribution(p, p), p)


def test_residual_hand_example():
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.2, 0.6, 0.2])
    assert torch.allclose(residual_distribution(p, q), torch.tensor([1.0, 0.0, 0.0]))


def test_speculative_accept_monte_carlo_matches_target():
    p = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    q = torch.tensor([0.2, 0.6, 0.2], dtype=torch.float64)
    g = torch.Generator().manual_seed(123)
    counts = torch.zeros(3)
    n = 40000
    for _ in range(n):
        x = torch.multinomial(q, 1, generator=g)
        n_acc, nxt = speculative_accept(torch.stack([p, p]), q[None], x, generator=g)
        first = int(x) if n_acc == 1 else nxt
        counts[first] += 1
    assert torch.allclose(counts / n, p.float(), atol=0.01)


def test_expected_tokens_per_round():
    assert expected_tokens_per_round(1.0, 4) == 5
    assert expected_tokens_per_round(0.0, 4) == 1
    assert expected_tokens_per_round(0.8, 4) == pytest.approx((1 - 0.8 ** 5) / 0.2)


def test_speculative_greedy_is_lossless_vs_target_greedy():
    target = _model(seed=1)
    draft = _model(seed=2, n_layer=1, n_embd=16, n_head=2)
    idx = torch.randint(0, 50, (1, 4))
    ref = _greedy_reference(target, idx, 25)
    out, stats = speculative_generate(target, draft, idx, 25, k=3, greedy=True)
    assert torch.equal(out, ref)
    assert stats["accepted"] <= stats["drafted"]


def test_speculative_with_self_draft_accepts_everything():
    target = _model(seed=3)
    idx = torch.randint(0, 50, (1, 3))
    out, stats = speculative_generate(target, target, idx, 20, k=4, greedy=True)
    assert torch.equal(out, _greedy_reference(target, idx, 20))
    assert stats["accepted"] == stats["drafted"]
    assert stats["rounds"] == math.ceil(20 / 5)


def test_speculative_sampling_runs_and_respects_length():
    target = _model(seed=4)
    draft = _model(seed=5, n_layer=1)
    idx = torch.randint(0, 50, (1, 2))
    g = torch.Generator().manual_seed(0)
    out, stats = speculative_generate(target, draft, idx, 15, k=3, generator=g)
    assert out.shape == (1, 17)
    assert torch.equal(out[:, :2], idx)
    assert 0 <= stats["accepted"] <= stats["drafted"]
