"""Tests for llmre.evaluation.metrics.

We check that our from-scratch cross-entropy matches PyTorch's fused
``torch.nn.functional.cross_entropy`` on a random batch, and that perplexity of
a uniform distribution over V classes equals V.
"""
import math

import torch
import torch.nn.functional as F

from llmre.evaluation.metrics import cross_entropy, perplexity


def test_cross_entropy_matches_pytorch():
    torch.manual_seed(0)
    N, V = 8, 10
    logits = torch.randn(N, V, dtype=torch.float64)
    targets = torch.randint(0, V, (N,))

    ours = cross_entropy(logits, targets)
    ref = F.cross_entropy(logits, targets)  # default reduction is mean, in nats

    assert torch.allclose(ours, ref, atol=1e-5), (ours.item(), ref.item())


def test_perplexity_of_uniform_is_V():
    # Uniform logits (all equal) => uniform softmax => loss = ln(V) => ppl = V,
    # for any targets, since every class has probability 1/V.
    N, V = 5, 7
    logits = torch.zeros(N, V)
    targets = torch.randint(0, V, (N,))

    ce = cross_entropy(logits, targets)
    assert math.isclose(ce.item(), math.log(V), rel_tol=1e-6)

    ppl = perplexity(logits, targets)
    assert math.isclose(ppl.item(), float(V), rel_tol=1e-5)


def test_perfect_prediction_low_loss():
    # One class given an overwhelmingly large logit => near-zero loss, ppl ~ 1.
    logits = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
    targets = torch.tensor([0])
    assert cross_entropy(logits, targets).item() < 1e-6
    assert abs(perplexity(logits, targets).item() - 1.0) < 1e-5
