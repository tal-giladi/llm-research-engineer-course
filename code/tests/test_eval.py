"""Tests for the evaluation harness and calibration.

We use small deterministic toy models (returning hand-built logits) so the
assertions do not depend on training randomness, plus one tiny real GPT trained
a few steps to confirm the harness runs end to end on the real model.
"""
import math

import torch

from llmre.evaluation.calibration import expected_calibration_error
from llmre.evaluation.harness import (
    evaluate_perplexity,
    multiple_choice_score,
    sequence_loglikelihood,
)
from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT


class UniformModel:
    """Toy model: uniform predictions (all-zero logits) over ``vocab_size``.

    forward returns ``(logits, loss)`` like :class:`llmre.model.gpt.GPT`. With
    equal logits every class has probability ``1/V``, so the cross-entropy is
    exactly ``ln(V)`` and perplexity is ``V``.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        logits = torch.zeros(B, T, self.vocab_size)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1)
            )
        return logits, loss

    def eval(self):
        return self

    def parameters(self):
        return iter(())


class FavorTokenModel:
    """Toy model that puts a large logit on one fixed token at every position."""

    def __init__(self, vocab_size: int, favored: int, strength: float = 10.0):
        self.vocab_size = vocab_size
        self.favored = favored
        self.strength = strength

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        logits = torch.zeros(B, T, self.vocab_size)
        logits[..., self.favored] = self.strength
        return logits, None

    def eval(self):
        return self

    def parameters(self):
        return iter(())


class FixedProbsModel:
    """Toy model returning the same fixed probability distribution everywhere.

    Constructed so ``log p`` of each token id is exactly the given value: the
    logits are just ``log(probs)``, and ``softmax(log(probs)) == probs``.
    """

    def __init__(self, probs: torch.Tensor):
        self.probs = probs / probs.sum()
        self.logits_row = torch.log(self.probs)
        self.vocab_size = self.probs.numel()

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        logits = self.logits_row.view(1, 1, -1).expand(B, T, self.vocab_size).clone()
        return logits, None

    def eval(self):
        return self

    def parameters(self):
        return iter(())


def test_evaluate_perplexity_uniform_is_vocab_size():
    V = 23
    model = UniformModel(V)
    data = torch.randint(0, V, (500,))
    ppl = evaluate_perplexity(model, data, block_size=16, batch_size=4)
    # Uniform predictions => perplexity == V, exactly (up to float error).
    assert math.isclose(ppl, V, rel_tol=1e-5), ppl


def test_multiple_choice_picks_favored_option():
    V = 50
    favored = 7
    model = FavorTokenModel(V, favored=favored)
    context = torch.tensor([1, 2, 3])
    # Option 0 is the token the model loves; option 1 is a different token.
    options = [torch.tensor([favored]), torch.tensor([favored + 1])]
    assert multiple_choice_score(model, context, options) == 0

    # And the ordering is by likelihood: swapping the options flips the answer.
    options_swapped = [torch.tensor([favored + 1]), torch.tensor([favored])]
    assert multiple_choice_score(model, context, options_swapped) == 1


def test_length_normalization_changes_the_winner():
    # Build a model whose per-token log-probs are exactly known:
    #   token 1 -> log p = -1  (high confidence)
    #   token 2 -> log p = -3  (low confidence)
    # Option A is one low-confidence token: sum = -3, mean/token = -3.
    # Option B is four high-confidence tokens: sum = -4, mean/token = -1.
    # Summed log-likelihood favors the *shorter* A (-3 > -4); length
    # normalization favors B, whose per-token confidence is higher (-1 > -3).
    p = torch.full((6,), 0.145584)   # filler mass for the other 4 tokens
    p[1] = math.exp(-1.0)            # log p(token 1) = -1
    p[2] = math.exp(-3.0)            # log p(token 2) = -3
    model = FixedProbsModel(p)
    context = torch.tensor([0])
    option_a = torch.tensor([2])              # 1 low-confidence token
    option_b = torch.tensor([1, 1, 1, 1])     # 4 high-confidence tokens

    assert (
        multiple_choice_score(model, context, [option_a, option_b], length_normalize=False)
        == 0
    )
    assert (
        multiple_choice_score(model, context, [option_a, option_b], length_normalize=True)
        == 1
    )


def test_sequence_loglikelihood_is_negative_and_summed():
    V = 12
    model = UniformModel(V)
    ids = torch.tensor([1, 2, 3, 4, 5])
    ll = sequence_loglikelihood(model, ids)
    # Uniform model: each of the (L-1)=4 scored tokens has log-prob ln(1/V).
    expected = 4 * math.log(1.0 / V)
    assert math.isclose(ll, expected, rel_tol=1e-6), (ll, expected)


def test_ece_zero_for_perfectly_calibrated():
    # 100 predictions at confidence 0.5, exactly half correct: in the bin holding
    # 0.5, mean confidence 0.5 == accuracy 0.5, so ECE is 0.
    conf = [0.5] * 100
    correct = [1] * 50 + [0] * 50
    ece = expected_calibration_error(conf, correct, n_bins=10)
    assert math.isclose(ece, 0.0, abs_tol=1e-9), ece


def test_ece_positive_for_miscalibrated():
    # 100 predictions at confidence 0.9, all wrong: |acc - conf| = |0 - 0.9| = 0.9.
    conf = [0.9] * 100
    correct = [0] * 100
    ece = expected_calibration_error(conf, correct, n_bins=10)
    assert math.isclose(ece, 0.9, abs_tol=1e-9), ece


def test_harness_runs_on_a_tiny_real_gpt():
    # Train a tiny GPT a handful of steps on a fixed repeating pattern, then check
    # the harness produces a finite perplexity and a sane multiple-choice answer.
    torch.manual_seed(0)
    V, T = 16, 8
    cfg = GPTConfig(
        vocab_size=V, block_size=T, n_layer=2, n_head=2, n_embd=16, dropout=0.0
    )
    model = GPT(cfg)
    # Repeating pattern 0,1,2,...,V-1,0,1,... so "next token" is fully learnable.
    stream = torch.arange(V).repeat(60)  # length 960
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(60):
        i = torch.randint(0, stream.numel() - T - 1, (1,)).item()
        x = stream[i : i + T].unsqueeze(0)
        y = stream[i + 1 : i + 1 + T].unsqueeze(0)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    ppl = evaluate_perplexity(model, stream, block_size=T, batch_size=8)
    assert math.isfinite(ppl) and ppl > 0.0
    # After training on the pattern, the true continuation should be far more
    # likely than a wrong one. Context ends at token 4 -> next should be 5.
    context = torch.tensor([0, 1, 2, 3, 4])
    right = torch.tensor([5])
    wrong = torch.tensor([11])
    assert multiple_choice_score(model, context, [right, wrong]) == 0
