# Low-bit KV-cache quantization (rotation choices: FlagRot vs RotateKV/KIVI)

- **Topic ID:** kv-cache-low-bit-quantization
- **Status:** WAIT
- **Next review:** 2027-01-03
- **Course change:** none
- **Candidates:** C-20261105-03

## Summary

Quantizing the KV cache to 2-4 bits per value, where rotating key channels (Hadamard) before quantization usually reduces error. FlagRot chooses per 32-channel group whether to rotate, by calibrated attention error, and stores the choice in spare sign bits.

## Evidence

- https://arxiv.org/abs/2611.03992 and https://github.com/KunmingSHAO/FlagRot_releaser — 12 models, beats RotateKV on 9/12 at 2.5 bits (authors); vLLM plugin with Hopper kernels; Apache-2.0; no reproduction.

## What would change the decision

First the author-approved post-training-quantization foundation (weights) has to exist; then KV-cache quantization as such (KIVI-style per-channel keys / per-token values, supported in vLLM/SGLang) is the candidate extension of 08.6 or the PTQ lesson, with methods like FlagRot as examples only after independent reproduction.

## History

- 2026-11-08 — WAIT — a single paper with good code, but it refines a technique (low-bit KV quantization) the course has not taught yet; depends on the pending PTQ foundation — weekly/2026-W45.md — candidates C-20261105-03
