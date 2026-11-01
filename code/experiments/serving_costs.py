"""Inference serving costs: KV cache, batching and speculative decoding on CPU.

Runnable companion to lesson 08.6 (`lessons/module-08/lesson-06.md`). Three parts,
all on CPU in well under a minute:

A. KV cache. Times greedy generation of 200 tokens with the plain 06.3 loop (re-run
   the whole prefix every step) and with :func:`llmre.model.serving.generate_cached`
   (one prefill, then one-token decode steps), and checks the tokens are identical.
B. Batching. Static (request-level) vs continuous (iteration-level) batching on 64
   requests whose output lengths follow a long-tailed distribution; prints decode
   steps and slot utilization for several batch sizes.
C. Speculative decoding. A 4-layer "target" and a 1-layer "draft" trained for a few
   hundred steps on the same tiny repetitive corpus, so the draft is a decent guess.
   Prints the acceptance rate, tokens per target forward, and checks the greedy
   output equals the target's own greedy output.

It is NOT a pytest test::

    cd code && PYTHONPATH=src python experiments/serving_costs.py

Wall-clock numbers depend on the CPU; the ratios and the token checks are the point.
"""

from __future__ import annotations

import random
import time

import torch

from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.model.serving import (
    generate_cached,
    simulate_continuous_batching,
    simulate_static_batching,
    speculative_generate,
)


def plain_greedy(model, idx, n):
    for _ in range(n):
        logits, _ = model(idx)
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return idx


@torch.no_grad()
def part_a():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, block_size=256, n_layer=4, n_head=4, n_embd=128, dropout=0.0)
    m = GPT(cfg).eval()
    idx = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        t0 = time.perf_counter(); ref = plain_greedy(m, idx, 200); t1 = time.perf_counter()
        out = generate_cached(m, idx, 200, greedy=True); t2 = time.perf_counter()
    print("A. KV cache (200 new tokens, 4-layer d=128 model, CPU)")
    print(f"   no cache : {t1 - t0:6.2f} s")
    print(f"   KV cache : {t2 - t1:6.2f} s   speedup {(t1 - t0) / (t2 - t1):.1f}x")
    print(f"   identical tokens: {torch.equal(ref, out)}")


def part_b():
    rng = random.Random(0)
    lengths = [min(1000, int(rng.paretovariate(1.2) * 20)) for _ in range(64)]
    print("\nB. Static vs continuous batching (64 requests, long-tailed lengths)")
    print(f"   lengths: mean {sum(lengths) / 64:.0f}, max {max(lengths)}")
    for b in (4, 8, 16):
        s = simulate_static_batching(lengths, b)
        c = simulate_continuous_batching(lengths, b)
        print(f"   batch {b:2d}: static {s['steps']:5d} steps ({s['utilization']:.0%} busy)"
              f" | continuous {c['steps']:5d} steps ({c['utilization']:.0%} busy)")


def _train(model, data, steps, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    T = model.cfg.block_size
    for _ in range(steps):
        i = torch.randint(0, data.numel() - T - 1, (16,))
        x = torch.stack([data[j:j + T] for j in i])
        y = torch.stack([data[j + 1:j + T + 1] for j in i])
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def part_c():
    torch.manual_seed(0)
    text = ("the cat sat on the mat. the dog sat on the log. " * 200).encode()
    data = torch.tensor(list(text), dtype=torch.long)
    big = GPT(GPTConfig(vocab_size=256, block_size=96, n_layer=4, n_head=4, n_embd=96, dropout=0.0))
    small = GPT(GPTConfig(vocab_size=256, block_size=96, n_layer=1, n_head=2, n_embd=32, dropout=0.0))
    lt = _train(big, data, 300)
    ls = _train(small, data, 300)
    big.eval(); small.eval()
    prompt = data[:8][None]
    print("\nC. Speculative decoding (target 4 layers, draft 1 layer)")
    print(f"   final train loss: target {lt:.3f}, draft {ls:.3f}")
    with torch.no_grad():
        ref = generate_cached(big, prompt, 60, greedy=True)
        for k in (2, 4, 6):
            out, st = speculative_generate(big, small, prompt, 60, k=k, greedy=True)
            print(f"   k={k}: accepted {st['accepted']}/{st['drafted']} drafts "
                  f"({st['accepted'] / st['drafted']:.0%}), {60 / st['rounds']:.2f} tokens per "
                  f"target forward, identical to target greedy: {torch.equal(out, ref)}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
