"""AdamW vs Muon (+ auxiliary AdamW) on a tiny GPT: equal steps and equal time.

Runnable companion to lesson 03.4 (`lessons/module-03/lesson-04.md`). It trains
the course's GPT (Module 6) on a byte-level corpus made of the course's own
Module 3 lesson files, once with AdamW on every parameter and once with
``MuonWithAuxAdamW`` (Muon on the hidden matrices, AdamW on embeddings / head /
biases / gains). For each it reports

* the validation loss after the same number of steps (data efficiency), and
* the training time and the share of it spent inside ``optimizer.step()``
  (the wall-clock cost the lesson's section 6 predicts).

Each optimizer gets a small learning-rate sweep and the best value is reported,
because comparing one tuned optimizer with one untuned one says nothing.
It is NOT a pytest test. A few minutes on a laptop CPU::

    cd code && PYTHONPATH=src python experiments/muon_vs_adamw.py

Everything is seeded. Exact numbers vary by CPU and torch build, the trend
should not. A toy, not a research result: see the lesson for what does and
does not transfer to scale.
"""

from __future__ import annotations

import pathlib
import time

import torch

from llmre.data.loader import get_batch
from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.optim.adamw import AdamW
from llmre.optim.muon import MuonWithAuxAdamW

ROOT = pathlib.Path(__file__).resolve().parents[2]
STEPS, BLOCK, BATCH, SEEDS = 300, 64, 16, (0, 1)


def build_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Byte-encode lessons/module-03/*.md; return (train_ids, val_ids)."""
    text = "".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / "lessons" / "module-03").glob("lesson-0[1-3].md")))
    ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    cut = int(0.9 * ids.numel())
    return ids[:cut], ids[cut:]


def make_model(seed: int) -> GPT:
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=256, block_size=BLOCK, n_layer=2, n_head=4,
                    n_embd=128, dropout=0.0, bias=True)
    return GPT(cfg)


def run(kind: str, lr: float, train: torch.Tensor, val: torch.Tensor, seed: int,
        steps: int = STEPS):
    """Train one model; return (val_loss, total_seconds, optimizer_seconds)."""
    model = make_model(seed)
    if kind == "adamw":
        opt = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    else:
        opt = MuonWithAuxAdamW(model, muon_kwargs={"lr": lr},
                               adamw_kwargs={"lr": 3e-3, "betas": (0.9, 0.95), "weight_decay": 0.0})
    torch.manual_seed(1000 + seed)  # same batches for every optimizer
    t_total = t_opt = 0.0
    for _ in range(steps):
        t0 = time.perf_counter()
        x, y = get_batch(train, BLOCK, BATCH)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        t1 = time.perf_counter()
        opt.step()
        t2 = time.perf_counter()
        t_total += t2 - t0
        t_opt += t2 - t1
    torch.manual_seed(99)
    with torch.no_grad():
        vl = sum(model(*get_batch(val, BLOCK, 64))[1].item() for _ in range(10)) / 10
    return vl, t_total, t_opt


def main() -> None:
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    train, val = build_data()
    print(f"corpus: {train.numel():,} train / {val.numel():,} val bytes; {STEPS} steps x {BATCH}x{BLOCK} tokens")
    grid = {"adamw": (3e-4, 1e-3, 3e-3), "muon": (0.01, 0.02, 0.05)}
    bests = {}
    for kind, lrs in grid.items():
        best = None
        for lr in lrs:
            res = [run(kind, lr, train, val, s) for s in SEEDS]
            vl = sum(r[0] for r in res) / len(res)
            tt = sum(r[1] for r in res) / len(res)
            to = sum(r[2] for r in res) / len(res)
            print(f"  {kind:5s} lr={lr:<6g} val loss {vl:.4f}  time {tt:5.1f}s  optimizer share {100 * to / tt:4.1f}%")
            if best is None or vl < best[1]:
                best = (lr, vl, tt, to)
        print(f"best {kind}: lr={best[0]:g}  val loss {best[1]:.4f}  "
              f"time/step {1000 * best[2] / STEPS:.1f} ms  optimizer share {100 * best[3] / best[2]:.1f}%\n")
        bests[kind] = best
    # Equal wall-clock: give AdamW as many extra steps as Muon's slower steps cost.
    ratio = bests["muon"][2] / bests["adamw"][2]
    steps = round(STEPS * ratio)
    res = [run("adamw", bests["adamw"][0], train, val, s, steps=steps) for s in SEEDS]
    vl = sum(r[0] for r in res) / len(res)
    print(f"equal time: AdamW lr={bests['adamw'][0]:g} for {steps} steps ({ratio:.2f}x) -> val loss {vl:.4f}"
          f"  vs Muon {bests['muon'][1]:.4f} in {STEPS} steps")


if __name__ == "__main__":
    main()
