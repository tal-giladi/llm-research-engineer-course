# Automated fault attribution and recovery for large distributed training runs (NVIDIA Recovery + Attribution)

- **Topic ID:** distributed-training-fault-recovery
- **Status:** WAIT
- **Next review:** 2027-01-03
- **Course change:** none
- **Candidates:** C-20261104-01

## Summary

Operating a training run rather than coding it: a rules engine ingests XID codes, NCCL events, logs and scheduler state and names the originating fault (separating it from downstream cascading errors, or saying 'ambiguous'), and a detect -> attribute -> isolate -> restart-from-checkpoint loop acts on it through the scheduler. Complementary to the deferred bitwise-deterministic-training topic (replaying a run vs diagnosing why it died).

## Evidence

- https://developer.nvidia.com/blog/keep-distributed-training-running-with-automated-workload-recovery/ and https://developer.nvidia.com/blog/diagnose-distributed-ai-workload-failures-with-nvidia-attribution/ — NVIDIA posts (2026-11-02): ~200 jobs/day, mean detection 69 s, mean recovery action 94 s on NVIDIA's cluster (company claim); IBM Research and Verda named as users.
- https://github.com/dsx-ai-factory/NVIDIA-Recovery and https://github.com/dsx-ai-factory/attribution — Apache-2.0; the Recovery README calls the project experimental and not production-ready.

## What would change the decision

A stable (non-experimental) release or adoption in a widely used training stack (Megatron/NeMo, TorchTitan, Slurm/Kubernetes operators) with an independent write-up. Even then this is likely a short operations section in Module 09 (distributed training), not a lesson of its own; the teachable core (cascading-failure attribution, checkpoint-restart cost) needs no specific tool.

## History

- 2026-11-08 — WAIT — real open-source tooling with two named external users, but the project calls itself experimental and the numbers are NVIDIA's own; operations material that would be a small section, not a curriculum gap — weekly/2026-W45.md — candidates C-20261104-01
