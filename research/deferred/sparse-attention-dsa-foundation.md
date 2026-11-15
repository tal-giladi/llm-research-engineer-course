# DeepSeek Sparse Attention foundation (lightning indexer + top-k selection), author-approved, queued

- **Topic ID:** sparse-attention-dsa-foundation
- **Status:** WAIT
- **Next review:** 2026-11-22
- **Course change:** none (Kimi K3.1 Pro's linear + sparse layout is cited as a dated example in 12.5, without teaching the selection mechanism)
- **Candidates:** C-20261106-01, C-20261108-05

## Summary

A short foundation lesson after 12.5, approved by the author on 2026-11-08: why dense attention is quadratic, DSA's indexer score and top-k key selection on the course GPT with a worked example, the compute and KV-memory cost, and the trade-offs. Built from DeepSeek-V3.2 and GLM-5 material, not from pending single papers. Once it exists, DoSA, PISA/RecSA, FFD, Prism/MLI, cross-layer index sharing, ADIndex and CISA are re-judged against it.

## Evidence

- SCOPE-DECISIONS.md, 2026-11-08 row (author decision).
- DeepSeek-V3.2 and GLM-5 adopt DSA (company reports; see sparse-attention-cross-layer-index-sharing.md for links).
- https://huggingface.co/moonshotai/Kimi-K3.1-Pro — third major open model family using DSA: 69 KDA + 24 Gated DSA layers, index top-k 2048, 32 index heads of dim 128 (card verified 2026-11-15; company claim for quality). Tal's instruction (2026-11-08): cite this gated-DSA config in the DSA foundation when it is written.
- https://arxiv.org/abs/2611.12956 — ADIndex (daily 2026-11-13, C class): third-party work optimizing the DSA indexer on DeepSeek-V3.2 and GLM-5.1, more evidence that DSA is the common base.

## What would change the decision

Nothing about its eligibility: it qualifies. It waits only for the weekly ADD budget. W46's ADD went to the post-training quantization foundation, approved earlier (2026-10-28) and blocking more downstream topics. Next week it competes with Keep Sampling Mask and FlexShard (see weekly/2026-W46.md for the recommended order).

## History

- 2026-11-15 — WAIT — author-approved foundation, now with a third adopter (Kimi K3.1 Pro); budget only, this week's ADD went to the older PTQ foundation; first in line with Keep Sampling Mask next week — weekly/2026-W46.md — candidates C-20261106-01, C-20261108-05
