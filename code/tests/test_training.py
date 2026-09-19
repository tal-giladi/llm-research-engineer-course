"""Tests for llmre.training — the loop, metrics, and checkpointing (Module 7).

Four claims the lessons make, each pinned to an assertion:

1. Learning works end to end: a tiny GPT trained on a repeated toy sequence
   drives its loss well below the untrained ``ln(vocab_size)`` baseline.
2. Gradient accumulation over ``G`` micro-batches reproduces the gradient of one
   full batch of the same total size (to floating-point tolerance).
3. A checkpoint round-trips model + optimizer + step, so a parameter equals its
   pre-save value after save/load.
4. ``tokens_per_step`` is just ``b * G * W * T``.
"""

from __future__ import annotations

import math
import os
import tempfile

import torch

from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.optim.adamw import AdamW
from llmre.training.checkpoint import load_checkpoint, save_checkpoint
from llmre.training.loop import TrainConfig, train
from llmre.training.metrics import (
    mfu,
    model_flops_per_token,
    non_embedding_params,
    tokens_per_step,
)


def _tiny_config(vocab_size: int = 16, block_size: int = 8) -> GPTConfig:
    """A GPT small enough to train in a fraction of a second on CPU."""
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True,
    )


def test_tiny_gpt_overfits_toy_sequence():
    """Loss must fall well below the ln(vocab) init baseline within ~200 steps."""
    torch.manual_seed(0)
    cfg = _tiny_config(vocab_size=16, block_size=8)
    model = GPT(cfg)

    # A short, highly repetitive stream the model can memorize. Cycling a fixed
    # pattern means every window is predictable, so an overfit is expected.
    pattern = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2]
    stream = torch.tensor(pattern * 64, dtype=torch.long)

    def get_batch(split: str):
        b, T = 4, cfg.block_size
        ix = torch.randint(0, stream.numel() - T - 1, (b,))
        x = torch.stack([stream[i : i + T] for i in ix])
        y = torch.stack([stream[i + 1 : i + 1 + T] for i in ix])
        return x, y

    tcfg = TrainConfig(
        max_steps=200,
        micro_batch_size=4,
        grad_accum_steps=1,
        warmup_steps=20,
        max_lr=1e-2,
        min_lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        seed=0,
        device="cpu",
        log_interval=10,
    )
    history = train(model, get_batch, tcfg)

    init_baseline = math.log(cfg.vocab_size)  # ~2.7726 for V=16
    final_loss = history["loss"][-1]
    # Well below the untrained baseline: the model has clearly learned the pattern.
    assert final_loss < 0.5 * init_baseline, (final_loss, init_baseline)
    # And it went down monotone-ish overall (first logged >> last logged).
    assert history["loss"][0] > final_loss


def test_grad_accumulation_matches_full_batch():
    """Sum of G scaled micro-batch grads == grad of the full-batch mean loss."""
    torch.manual_seed(0)
    cfg = _tiny_config(vocab_size=16, block_size=8)
    model = GPT(cfg).double()  # float64 so the equality is about algebra, not fp

    G, b, T = 4, 3, cfg.block_size
    total = G * b
    torch.manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (total, T))
    y = torch.randint(0, cfg.vocab_size, (total, T))

    params = list(model.parameters())

    # (a) One full batch of size G*b.
    model.zero_grad(set_to_none=True)
    _, loss_full = model(x, y)
    loss_full.backward()
    full_grads = [p.grad.detach().clone() for p in params]

    # (b) G micro-batches of size b, each scaled by 1/G and summed.
    model.zero_grad(set_to_none=True)
    for g in range(G):
        xm = x[g * b : (g + 1) * b]
        ym = y[g * b : (g + 1) * b]
        _, loss_m = model(xm, ym)
        (loss_m / G).backward()
    accum_grads = [p.grad.detach().clone() for p in params]

    for gf, ga in zip(full_grads, accum_grads):
        assert torch.allclose(gf, ga, atol=1e-5, rtol=0), (gf - ga).abs().max().item()


def test_checkpoint_roundtrips_model_optimizer_step():
    """save -> mutate -> load restores a parameter, the step, and extra."""
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = GPT(cfg)
    opt = AdamW(list(model.parameters()), lr=1e-3)

    # Take a couple of real steps so optimizer moments are non-zero.
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    for _ in range(3):
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()

    # Snapshot a specific parameter before saving.
    ref_name = "transformer.h.0.mlp.c_fc.weight"
    saved_param = dict(model.named_parameters())[ref_name].detach().clone()
    saved_t = opt.t

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, opt, step=3, extra={"note": "hello", "val": 1.5})

        # Corrupt the live model + optimizer to prove load actually restores.
        with torch.no_grad():
            dict(model.named_parameters())[ref_name].add_(1.0)
        opt.t = 999

        step, extra = load_checkpoint(path, model, opt, map_location="cpu")

    restored_param = dict(model.named_parameters())[ref_name].detach()
    assert torch.allclose(restored_param, saved_param, atol=0, rtol=0)
    assert step == 3
    assert opt.t == saved_t == 3
    assert extra == {"note": "hello", "val": 1.5}


def test_resumed_optimizer_state_is_restored():
    """After load, the very next update matches a run that never stopped."""
    torch.manual_seed(0)
    cfg = _tiny_config()

    def run(save_and_reload: bool):
        torch.manual_seed(0)
        model = GPT(cfg)
        opt = AdamW(list(model.parameters()), lr=1e-3)
        x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
        y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
        # Two steps.
        for _ in range(2):
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            opt.step()
        if save_and_reload:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "c.pt")
                save_checkpoint(path, model, opt, step=2, extra={})
                model2 = GPT(cfg)
                opt2 = AdamW(list(model2.parameters()), lr=1e-3)
                load_checkpoint(path, model2, opt2, map_location="cpu")
                model, opt = model2, opt2
        # One more step; return the resulting parameter.
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        return dict(model.named_parameters())[
            "transformer.h.0.mlp.c_fc.weight"
        ].detach().clone()

    p_continuous = run(save_and_reload=False)
    p_resumed = run(save_and_reload=True)
    assert torch.allclose(p_continuous, p_resumed, atol=1e-6, rtol=0)


def test_tokens_per_step_arithmetic():
    assert tokens_per_step(micro_bs=4, grad_accum=8, world_size=1, block_size=1024) == 32768
    assert tokens_per_step(micro_bs=12, grad_accum=40, world_size=8, block_size=2048) == 12 * 40 * 8 * 2048
    assert tokens_per_step(1, 1, 1, 1) == 1


def test_flops_and_mfu():
    cfg = GPTConfig()  # GPT-2 small
    N = non_embedding_params(cfg)
    assert N == 12 * 12 * 768**2  # 84,934,656
    fpt = model_flops_per_token(cfg)
    assert fpt == 6.0 * N
    # A100 bf16 peak ~312 TFLOP/s; a made-up throughput gives a fraction in (0,1).
    util = mfu(flops_per_token=fpt, tokens_per_sec=50_000, peak_flops=312e12)
    assert 0.0 < util < 1.0
