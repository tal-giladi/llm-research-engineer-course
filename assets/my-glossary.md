# My glossary

Basic definitions in my own words.

**Data parallelism** — run the same loop on W GPUs at once, each on a different slice of the data, then reduce the data back.

**Ring-reduce** — a reduce method aimed at saving traffic.

**Fusion** — merging sequential operations (like add bias → GELU → dropout) into one GPU kernel, e.g. with `torch.compile`, so the intermediate results stay in the GPU chip's fast on-chip memory instead of being written to GPU memory (HBM) and read back between every step.
