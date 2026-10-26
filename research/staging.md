# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261026-01 · Where Rounding Happens Matters: Rounding-Location Alignment Stabilizes LLM Reinforcement Learning

- **Class:** B
- **Date discovered:** 2026-10-26
- **Date published:** 2026-10-23 (v1)
- **Source:** https://arxiv.org/abs/2610.32951 (verified resolves; v1, Fri 23 Oct 2026, cs.LG; announced in Monday's batch, having missed Friday's cutoff)
- **Organization/researchers:** Yijing Ke, Tianle Cai, Ruosong Wang (affiliations not stated on the abstract page; Tianle Cai is a named author on prior speculative-decoding/inference-systems work, general knowledge, not independently re-confirmed here).
- **Category:** RL/post-training (root cause of training-inference mismatch in LLM RL)
- **What changed:** The course's new lesson 16.4 (`lessons/module-16/lesson-04.md`, accepted 2026-10-25) teaches that an RL trainer and its inference engine can assign different probabilities to the same weights/tokens even in BF16, making a nominally on-policy update off-policy, and surveys two production fixes (remove the mismatch via batch-invariant kernels, or correct it via truncated importance sampling). This paper supplies a specific, previously-unlisted root cause and fix: rather than a global precision/kernel change, it aligns only *where inside the forward pass* BF16 rounding happens — keeping RoPE, SwiGLU, and residual-RMSNorm intermediates in FP32 until they produce BF16 outputs, matching the inference engine's rounding points without touching the inference engine, the matmul/attention kernels, or the RL objective itself.
- **Technical summary:** Across four Qwen-family models, aligning all three regions (RoPE + SwiGLU + residual RMSNorm) prevents the training collapse seen with the default BF16 baseline (authors' own runs); aligning only the RoPE path alone was also sufficient in the studied runs even though some numerical discrepancy remained elsewhere. No independent reproduction found; no code repository found on the abstract page as of this check.
- **Why it might matter:** It is a concrete, mechanism-level (not just systems-level) answer to "where exactly does the bitwise mismatch 16.4 describes actually come from inside the model's own forward pass," phrased as three specific, checkable intermediate-precision changes rather than a new kernel library or a correction term in the RL loss — a natural worked addendum to 16.4's existing "what causes the mismatch" section.
- **Evidence of adoption:** None found — single paper, three days old at discovery, no stated venue, no code repository located.
- **Major organizations using it:** None identified; no stated lab affiliation.
- **Open-source implementation:** none found on the abstract page.
- **Paper:** https://arxiv.org/abs/2610.32951
- **Code:** none found
- **Relationship to existing course material:** Directly extends `lessons/module-16/lesson-04.md` and `research/accepted/rl-training-inference-mismatch.md` (16.4, ADD 2026-10-25) and `code/src/llmre/rl/mismatch.py`. Checked via `lookup "rounding LLM reinforcement learning"` (no registry/course hit beyond an unrelated precision line already in 16.4) — new, not a duplicate of the accepted topic's existing evidence.
- **Potential course lesson:** If reproduced or adopted, a short worked addition to 16.4's mismatch-causes section: a concrete example of *where* BF16 rounding inside RoPE/SwiGLU/RMSNorm creates the mismatch, and the FP32-intermediate fix, as a third concrete mechanism alongside batch-invariant kernels and TIS correction.
- **Confidence:** medium — a specific, falsifiable mechanism with authors' own measured results across four real model sizes, directly on-topic for already-accepted material, but one paper, no code, no independent reproduction.
- **Recommendation:** monitor; revisit at the next weekly cycle given how directly it extends the just-accepted 16.4 — worth an early look even without independent reproduction, since it is cheap to spot-check against `code/src/llmre/rl/mismatch.py`'s existing toy harness.

### C-20261026-02 · When and How Does Muon Precondition? The Role of Structured Gradient Noise

- **Class:** B
- **Date discovered:** 2026-10-26
- **Date published:** 2026-10-22 (v1)
- **Source:** https://arxiv.org/abs/2610.31728 (verified resolves; v1, Thu 22 Oct 2026 15:20:40 UTC, cs.LG; comments: "62 pages, 11 figures")
- **Organization/researchers:** Xiaoran Chen, Binghui Li, Liwei Wang, Pinyan Lu (affiliations not stated on the abstract page).
- **Category:** training/optimization (theoretical analysis of Muon, already taught in lesson 03.4)
- **What changed:** The course already teaches Muon (`lessons/module-03/lesson-04.md`) and the registry already has one deferred theory paper on it (`research/deferred/muon-polar-update-theory.md`, a "quadratic lens" on when orthogonalization helps via singular-value structure). This is a different theoretical account: it asks when Muon's orthogonalized update helps at all, as a function of *gradient-noise geometry* rather than curvature. It first shows, theoretically and empirically, that under isotropic noise Muon gives no benefit beyond a learning-rate rescaling — i.e., orthogonalization alone buys nothing without structure in the noise. It then analyzes structured noise from mini-batch sampling in a toy associative-memory model with uneven task frequencies, deriving a balancing threshold (extended to many tasks) under which Muon turns the usual linear dependence on task frequency into a square-root dependence, helping rare tasks train alongside frequent ones; it also identifies a rotation-induced negative-transfer failure mode of plain SignSGD that Muon's orthogonalization avoids, and gives an account of how momentum reduces sampling noise while still tracking a moving gradient.
- **Technical summary:** Controlled experiments on the toy associative-memory model support the derived threshold and the frequency-balancing claim; the paper also reports a preliminary real LLM-pretraining study suggesting practical relevance for a fast/slow-momentum variant of Muon. No code link found on the abstract page; no stated venue (62-page preprint).
- **Why it might matter:** This is a distinct, falsifiable mechanism (noise-geometry dependent task-frequency balancing) from the already-deferred quadratic-lens paper (curvature/singular-value dependent) — together they would give 03.4 two independent, non-overlapping "why orthogonalize" stories if either is eventually corroborated, rather than one being new evidence for the other's specific claim.
- **Evidence of adoption:** None — single paper, four days old at discovery, no stated venue, no code release found, no third-party reproduction.
- **Major organizations using it:** None identified; academic authorship, no stated lab affiliation.
- **Open-source implementation:** none found on the abstract page.
- **Paper:** https://arxiv.org/abs/2610.31728
- **Code:** none found
- **Relationship to existing course material:** Bears on `lessons/module-03/lesson-04.md` and `code/src/llmre/optim/muon.py`, and is thematically adjacent to (but mechanistically independent of) `research/deferred/muon-polar-update-theory.md` — checked via `lookup "Newton-Schulz orthogonalization Muon"`, which surfaces only that existing deferred file and the taught lesson; this paper's specific claim (noise-geometry-driven task-frequency balancing) is not covered there, so it is new, not a duplicate requiring no new record.
- **Potential course lesson:** If independently reproduced, a second short theory note for 03.4 alongside the quadratic-lens one: Muon's benefit under isotropic noise is nil (rules out a naive "orthogonalizing always helps" framing), and its benefit under realistic structured mini-batch noise is a frequency-balancing effect, demonstrated on a toy task mirroring 03.4's existing worked examples.
- **Confidence:** low-medium — a real theoretical analysis (not just an ablation) with a toy-model derivation and one preliminary real-pretraining check, but single paper, no code, no venue, no reproduction, and a somewhat narrow toy setting (two-task/few-task associative memory) for the main theorem.
- **Recommendation:** monitor; revisit if a main-venue version appears, code is released, or the fast/slow-momentum pretraining claim is independently checked — otherwise keep alongside (not merged into) the quadratic-lens deferred topic, since the two papers argue from different mechanisms.
