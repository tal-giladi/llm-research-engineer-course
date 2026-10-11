# Low-bit (4-bit) AdamW optimizer-state quantization: rounding in preconditioner space (ZIP-SR / ZE-EDEN)

- **Topic ID:** low-bit-optimizer-state-quantization
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20261011-01

## Summary

Shows that small rounding error in the stored second moment does not bound the error in the preconditioner actually applied, and rounds in preconditioner space instead; compared against TorchAO's 4-bit AdamW from 130M to 2.7B parameters.

## Evidence

- https://arxiv.org/abs/2610.12444 — single paper, no code; up to 70% smaller loss gap to 32-bit AdamW vs TorchAO 4-bit (authors' numbers).

## What would change the decision

Released code, an independent reproduction, or adoption in TorchAO/bitsandbytes. The course does not yet teach 8-/4-bit optimizer states at all; if this matures, the foundational version (block-wise quantized Adam states) would need to come first as an extension of 03.x/07.4.

## History

- 2026-10-11 — WAIT — clean, teachable mechanism but one day old, single paper, no code; and the foundational technique it refines is not yet in the course — weekly/2026-W41.md — candidates C-20261011-01
