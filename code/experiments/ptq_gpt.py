"""Post-training quantization of the course GPT on CPU: RTN vs GPTQ vs AWQ.

Runnable companion to lesson 08.7 (`lessons/module-08/lesson-07.md`). Two parts:

A. One layer with an outlier input channel. A random (64, 128) weight sees inputs whose
   channel 0 is 20x larger than the rest and whose channels are correlated. Prints the
   layer output error ||XW^T - XWq^T||^2 for per-tensor RTN, per-channel RTN, groupwise
   RTN (g=32), AWQ and GPTQ at 4 and 3 bits.
B. A whole model. Trains a 4-layer GPT (d=128) on a small byte corpus for a few hundred
   steps, then quantizes every Linear inside the blocks with each method at 8, 4 and 3
   bits (one scale per output channel) and prints, on held-out text, the KL divergence from
   the unquantized model's next-token distribution and the top-1 agreement with it.
   (KL to the original model measures what quantization changed; held-out loss on a tiny
   corpus is too noisy for that.)

It is NOT a pytest test::

    cd code && PYTHONPATH=src python experiments/ptq_gpt.py

Runs in under a minute on a laptop CPU. The exact numbers depend on the seed; the
orderings are the point (see the lesson's "expected observations").
"""

from __future__ import annotations

import torch

from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.model.quantize import (
    awq_quantize,
    fake_quantize,
    gptq_quantize,
    layer_output_error,
    quantize_gpt_weights,
    quantized_weight_bytes,
    rtn_quantize,
)


def part_a():
    torch.manual_seed(0)
    n, d_in, d_out = 512, 128, 64
    mix = torch.randn(d_in, d_in) / d_in**0.5 + torch.eye(d_in)  # correlated channels
    x = torch.randn(n, d_in) @ mix
    x[:, 0] *= 20.0                                                # one outlier channel
    w = torch.randn(d_out, d_in) * 0.05
    print("A. One layer, outlier input channel: output error ||XW^T - XWq^T||^2")
    for bits in (4, 3):
        rows = {
            "RTN per-tensor": fake_quantize(w, bits, dim=None),
            "RTN per-channel": rtn_quantize(w, bits),
            "RTN group 32": rtn_quantize(w, bits, 32),
            "AWQ group 32": awq_quantize(w, x, bits, 32)[0],
            "GPTQ group 32": gptq_quantize(w, x, bits, 32),
        }
        print(f"   {bits}-bit")
        for k, wq in rows.items():
            print(f"     {k:16s} {layer_output_error(w, wq, x):10.3f}")


@torch.no_grad()
def _kl_to_reference(ref, model, x):
    """Mean KL(ref || model) of the next-token distributions, in nats, and top-1 agreement."""
    lp_ref = torch.log_softmax(ref(x)[0], -1)
    lp_q = torch.log_softmax(model(x)[0], -1)
    kl = (lp_ref.exp() * (lp_ref - lp_q)).sum(-1).mean().item()
    agree = (lp_ref.argmax(-1) == lp_q.argmax(-1)).float().mean().item()
    return kl, agree


def part_b():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "pack", "my",
             "box", "with", "five", "dozen", "liquor", "jugs", "how", "daft", "zebras", "jump"]
    text = " ".join(words[k] for k in torch.randint(0, len(words), (3000,), generator=g).tolist())
    data = torch.tensor(list(text.encode()), dtype=torch.long)
    split = int(0.9 * data.numel())
    train, val = data[:split], data[split:]
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=4, n_head=4, n_embd=128, dropout=0.0)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        i = torch.randint(0, train.numel() - 65, (16,))
        x = torch.stack([train[j:j + 64] for j in i])
        y = torch.stack([train[j + 1:j + 65] for j in i])
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    calib = torch.stack([train[j:j + 64] for j in range(0, 64 * 16, 64)])  # 16 sequences
    held = torch.stack([val[j:j + 64] for j in range(0, val.numel() - 64, 64)])
    lin = [m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and ".h." in f".{n}"]
    fp_bytes = sum(m.weight.numel() * 2 for m in lin)
    print("\nB. Whole model (4 layers, d=128), block Linears only, per-channel scales")
    print(f"   bf16 weights: {fp_bytes / 1e3:6.1f} kB")
    print("   bits   kB     method  KL(fp||q) nats  top-1 agreement")
    for bits in (8, 4, 3):
        nbytes = sum(quantized_weight_bytes(*m.weight.shape, bits) for m in lin)
        for method in ("rtn", "awq", "gptq"):
            q = quantize_gpt_weights(model, calib, method, bits)
            kl, agree = _kl_to_reference(model, q, held)
            print(f"   {bits}   {nbytes / 1e3:6.1f}   {method:5s}   {kl:12.2e}    {agree:6.1%}")


if __name__ == "__main__":
    part_a()
    part_b()
