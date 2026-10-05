# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261005-01 · Looped (recurrent-depth) language models: practical training recipes

- **Class:** B
- **Date discovered:** 2026-10-05
- **Date published:** 2026-09-30
- **Source:** https://arxiv.org/abs/2610.00673 (verified resolves; v1 timestamp "Wed, 30 Sep 2026 20:09:58 UTC")
- **Organization/researchers:** Andrei Marchenko, Viacheslav Bezrukov, Oleg Kashurin, Inessa Fedorova, Dmitry Bocharov, Yuliana Shakhvalieva, Maria Tikhonova, Valerii Ternovskii (independent/academic group, not a named major lab)
- **Category:** architecture
- **What changed:** Practical recipe for training "looped" LMs — architectures that increase effective depth by repeatedly applying a shared block of layers (recurrent-depth, in the lineage of Universal Transformer / ALBERT-style weight sharing and 2025's recurrent-depth latent-reasoning work) — claiming large reductions in the token budget needed to make the approach competitive, plus a lightweight recipe for converting an already-pretrained dense checkpoint into a looped variant.
- **Technical summary:** Combines pretraining with targeted mid-training, warmup schedules and regularization changes to cut the data needed from 7.7T to 310B tokens for a competitive LoopLM. Reports a 1.4B-parameter LoopLM beating equivalently-sized dense models (+14 GSM8K, +10 MATH, +22 DROP, per the abstract) and matching a 3.9B dense model's quality while using only 36% of its parameters. Separately proposes converting an existing pretrained dense model into a looped variant using just one learned input-mixing scalar and a smoothed exit loss, rather than training from scratch.
- **Why it might matter:** If the dense-to-looped conversion and the reduced token budget hold up, this would be a cheap way to trade extra compute-per-token for fewer parameters/tokens — directly relevant to the course's architecture and efficient-training material (module on attention/architecture variants, alongside the existing Gated DeltaNet lesson).
- **Evidence of adoption:** None yet — single paper, no independent reproduction, no major-lab or major-stack adoption found.
- **Major organizations using it:** None identified.
- **Open-source implementation:** None linked from the abstract page as of this check.
- **Paper:** https://arxiv.org/abs/2610.00673
- **Code:** none found
- **Relationship to existing course material:** New — the registry/lessons have no existing "looped / recurrent-depth / weight-shared-layer" topic (checked via `research/tools/research.py lookup`); closest existing material is module-12/lesson-05 on Gated DeltaNet/hybrid linear attention, which is a different axis (attention mechanism, not depth-sharing).
- **Potential course lesson:** Possible extension of module-12 (architecture variants) or a frontier-update note, only if a second independent source or adoption signal appears.
- **Confidence:** low
- **Recommendation:** monitor — single-paper result with concrete numbers but no replication; revisit if a second group reproduces the token-budget claim or a major lab ships a looped-architecture model/checkpoint.
