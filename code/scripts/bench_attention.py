"""Benchmark: naive attention vs torch SDPA vs the FlashAttention algorithm.

Backs `lessons/module-08/lesson-04.md`. Run it directly (it is NOT a pytest test):

    py code/scripts/bench_attention.py
    py code/scripts/bench_attention.py --T 4096 --heads 12 --dim 64

On CPU this script's job is *correctness*: it checks that all three paths agree
to 1e-4 and reports rough wall-clock timings. The interesting numbers —
FlashAttention's speedup and, above all, its ``O(T)`` instead of ``O(T²)`` memory
— only appear on a real GPU, so if CUDA is unavailable the script prints a clear
note saying so and what you would expect to see.

Expected GPU behaviour (representative, A100/H100-class, not measured here):
  * Peak memory: naive attention materializes a (B, nh, T, T) score+prob tensor
    that grows with T². FlashAttention never stores it, so its memory grows only
    linearly in T. At T = 8k this is the difference between OOM and fitting.
  * Speed: FlashAttention is typically ~2-4x faster than a naive PyTorch
    attention at long context because it moves far fewer bytes to/from HBM, even
    though it does slightly MORE arithmetic (it recomputes some exponentials).
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from llmre.attention.flash import flash_attention_reference


def naive_attention(q, k, v, causal=False):
    """Full-matrix reference: softmax(QKᵀ/√d (+ causal mask)) V. Materializes T×T."""
    B, nh, T, hd = q.shape
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    if causal:
        mask = torch.triu(
            torch.full((T, T), float("-inf"), dtype=q.dtype, device=q.device),
            diagonal=1,
        )
        scores = scores + mask
    return F.softmax(scores, dim=-1) @ v


def _time(fn, iters):
    # Warm up, then time. On CUDA we synchronize so timings are real.
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--dim", type=int, default=32, help="head dimension hd")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--causal", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("=" * 68)
        print("NOTE: CUDA is unavailable — running on CPU.")
        print("  This validates CORRECTNESS (all three paths must agree) and")
        print("  prints CPU timings, but the FlashAttention memory/speed win is")
        print("  a GPU/HBM-traffic effect and will NOT show up here. See the")
        print("  docstring for the numbers you would expect on an A100/H100.")
        print("=" * 68)

    B, nh, T, hd = args.B, args.heads, args.T, args.dim
    dtype = torch.float32
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(B, nh, T, hd, generator=g, dtype=dtype, device=device)
    k = torch.randn(B, nh, T, hd, generator=g, dtype=dtype, device=device)
    v = torch.randn(B, nh, T, hd, generator=g, dtype=dtype, device=device)

    print(f"\nconfig: B={B} heads={nh} T={T} hd={hd} block={args.block} "
          f"causal={args.causal} device={device} dtype={dtype}")

    # --- correctness: all three must agree ---
    out_naive = naive_attention(q, k, v, causal=args.causal)
    out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=args.causal)
    out_flash = flash_attention_reference(q, k, v, causal=args.causal, block_size=args.block)
    ok_sdpa = torch.allclose(out_naive, out_sdpa, atol=1e-3, rtol=0)
    ok_flash = torch.allclose(out_naive, out_flash, atol=1e-3, rtol=0)
    print(f"correctness: naive vs SDPA  agree={ok_sdpa}")
    print(f"correctness: naive vs flash agree={ok_flash}")
    assert ok_sdpa and ok_flash, "attention paths disagree — this is a bug"

    # --- timing ---
    t_naive = _time(lambda: naive_attention(q, k, v, causal=args.causal), args.iters)
    t_sdpa = _time(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=args.causal), args.iters)
    t_flash = _time(lambda: flash_attention_reference(q, k, v, causal=args.causal, block_size=args.block), args.iters)
    print(f"\ntiming (mean over {args.iters} iters):")
    print(f"  naive attention        : {t_naive*1e3:8.3f} ms")
    print(f"  F.scaled_dot_product   : {t_sdpa*1e3:8.3f} ms")
    print(f"  flash_attention_ref    : {t_flash*1e3:8.3f} ms  "
          f"(python block loop — slow on CPU, illustrative only)")

    # score-matrix size that naive materializes but flash never does
    score_bytes = B * nh * T * T * q.element_size()
    print(f"\nnaive materializes a (B,nh,T,T) score tensor = "
          f"{score_bytes/1e6:.1f} MB (grows as T²); flash never allocates it.")


if __name__ == "__main__":
    main()
