"""Fit an empirical scaling law L(N) on a CPU-sized toy sweep.

This is the runnable companion to lesson 10.2. It trains ``K`` tiny GPTs of
*increasing width* on one fixed toy corpus, records each model's final loss, and
fits the power law ``L(N) = coefficient * N**(-alpha_N)`` with
:func:`llmre.evaluation.scaling.fit_power_law`. The point is not a research-grade
result (a real scaling study needs many GPUs and a real corpus) but to *see the
mechanism*: bigger model, lower loss, straight line in log-log space.

It is deliberately NOT a pytest test — it trains real networks and takes a
minute or two on a laptop CPU. Run it directly::

    cd code && py experiments/scaling_sweep.py

Everything is seeded, so the printed exponent is reproducible run to run.
"""

from __future__ import annotations

import time

import torch

from llmre.data.loader import get_batch
from llmre.evaluation.scaling import fit_power_law
from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT

# A small, structured toy corpus. Structure (repetition, a fixed vocabulary of
# words, punctuation) is what lets a bigger model fit better and drives the
# loss-vs-size trend the fit picks up.
CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "a wizard's job is to vex chumps quickly in fog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump! "
    "the five boxing wizards jump quickly. "
) * 200


def build_data() -> tuple[torch.Tensor, int]:
    """Byte-level encode the toy corpus into one id stream. Returns (data, vocab)."""
    ids = list(CORPUS.encode("utf-8"))
    return torch.tensor(ids, dtype=torch.long), 256


def train_one(n_embd: int, data: torch.Tensor, vocab: int,
              steps: int = 600, block_size: int = 32, batch_size: int = 32,
              seed: int = 0) -> tuple[int, float]:
    """Train one tiny GPT of width ``n_embd``; return (non_embedding_params, final_loss)."""
    torch.manual_seed(seed)
    cfg = GPTConfig(
        vocab_size=vocab, block_size=block_size,
        n_layer=2, n_head=4, n_embd=n_embd, dropout=0.0, bias=True,
    )
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1,
                            betas=(0.9, 0.95))
    model.train()
    for _ in range(steps):
        x, y = get_batch(data, block_size, batch_size, device="cpu")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Final loss: average over a handful of fresh batches for a stable estimate.
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(20):
            x, y = get_batch(data, block_size, batch_size, device="cpu")
            _, loss = model(x, y)
            losses.append(loss.item())
    n_params = model.num_params(non_embedding=True)
    return n_params, float(sum(losses) / len(losses))


def main() -> None:
    data, vocab = build_data()
    widths = [8, 16, 24, 32, 48, 64]
    print(f"toy corpus: {data.numel()} tokens, vocab {vocab}")
    print(f"{'n_embd':>7} {'N (non-emb)':>13} {'final loss':>11}")

    Ns, Ls = [], []
    t0 = time.time()
    for w in widths:
        N, L = train_one(w, data, vocab)
        Ns.append(N)
        Ls.append(L)
        print(f"{w:>7} {N:>13,} {L:>11.4f}")

    exponent, coefficient = fit_power_law(Ns, Ls)
    print(f"\nfitted power law:  L(N) = {coefficient:.4g} * N**(-{exponent:.4f})")
    print(f"fitted exponent alpha_N = {exponent:.4f}")
    print(f"(sweep took {time.time() - t0:.1f}s on CPU)")
    print("\nNote: a toy exponent, not a research figure. Real scaling laws "
          "(Kaplan 2020) report alpha_N ~ 0.076 over many orders of magnitude "
          "with far larger models, data, and compute.")


if __name__ == "__main__":
    main()
