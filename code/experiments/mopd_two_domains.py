"""Multi-teacher on-policy distillation on an exact two-domain toy.

Runnable companion to the MOPD extension of lesson 17.4
(`lessons/module-17/lesson-04.md`, "Advanced extension: multi-teacher on-policy
distillation"). Uses :mod:`llmre.reasoning.mopd`; runs on CPU in a few seconds.

Two specialist bigram teachers disagree on the tokens both domains write (teacher A
alternates 2,3,2,3; teacher B repeats 2,2,2). One bigram student must serve both.
The script prints, for each setting, the student's on-policy reverse KL to each
domain's own specialist (lower is better) and the loss weight each domain received:

1. single teacher vs routed teachers, equal answer lengths;
2. routed, domain A answers 24 tokens long and domain B 4 tokens long, with
   per-sequence vs per-token loss normalization;
3. the token share of a realistic three-domain batch (math/code/IF) and the
   per-sequence weights that rebalance it.

It is NOT a pytest test::

    cd code && PYTHONPATH=src python experiments/mopd_two_domains.py
"""

from __future__ import annotations

import torch

from llmre.reasoning.mopd import (
    distill_two_domains,
    make_two_domain_teachers,
    teacher_token_share,
    token_share_weights,
)


def row(name: str, r: dict) -> None:
    a, b = r["rkl"]
    wa, wb = r["domain_weight"]
    print(f"{name:<38} rkl A {a:6.3f}  rkl B {b:6.3f}  weight A/B {wa:.3f}/{wb:.3f}  p_alt {r['p_alternate']:.3f}")


def main() -> None:
    tA, tB, starts = make_two_domain_teachers()
    teachers = [tA, tB]
    print("1. Equal lengths (8, 8), 50/50 prompts")
    row("   single teacher (A scores everything)", distill_two_domains(teachers, starts, [8, 8], [0.5, 0.5], "teacher0"))
    row("   routed (MOPD)", distill_two_domains(teachers, starts, [8, 8], [0.5, 0.5], "routed"))

    print("2. Unequal lengths (24, 4), 50/50 prompts, routed")
    row("   per-sequence normalization", distill_two_domains(teachers, starts, [24, 4], [0.5, 0.5], "routed", "sequence"))
    row("   per-token normalization", distill_two_domains(teachers, starts, [24, 4], [0.5, 0.5], "routed", "token"))

    print("3. Token share of a math/code/IF batch (prompts 40/40/20)")
    lengths = [300, 300, 250, 250, 20]
    mask = torch.zeros(len(lengths), max(lengths))
    for b, n in enumerate(lengths):
        mask[b, :n] = 1
    ids = torch.tensor([0, 0, 1, 1, 2])
    share = teacher_token_share(mask, ids, 3)
    print("   token share math/code/IF:", [round(x, 4) for x in share.tolist()])
    print("   balancing weights per sequence:", [round(x, 3) for x in token_share_weights(mask, ids, 3).tolist()])


if __name__ == "__main__":
    main()
