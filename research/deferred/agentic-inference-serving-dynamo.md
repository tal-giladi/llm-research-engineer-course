# Session-aware agentic inference serving (NVIDIA Dynamo): session IDs, admission control, KV-cache hints

- **Topic ID:** agentic-inference-serving-dynamo
- **Status:** WAIT
- **Next review:** 2026-12-06
- **Course change:** none
- **Candidates:** C-20261009-01

## Summary

Serving infrastructure that treats a multi-turn agent session, not a request, as the unit: session-ID propagation from agent harnesses, a KV-utilization control loop that pauses and resumes agent programs at tool-call boundaries, KV-aware routing into external cache tiers, and a proposed router-to-engine KV-hint interface (Share/Prefetch/Demote/Pin/Retain) for vLLM and SGLang.

## Evidence

- https://pytorch.org/blog/session-aware-agentic-inference-with-nvidia-dynamo/ — 2026-10-08; 12-16% throughput over KV-aware routing on SWE-bench and 11.0-14.6% on agentic RL rollouts at medium concurrency (company claim); KvHint proposals open against vLLM and SGLang, only 'Share' implemented (in SGLang HiCache).

## What would change the decision

KvHint (or an equivalent session/KV-hint interface) merged into vLLM or SGLang; an independent measurement; or a second serving stack adopting session-level scheduling. Separately, the course has no serving-systems lesson (continuous batching, paged KV cache, prefix caching), which is the prerequisite such material would need first; that is a structural decision for the course author, not a weekly ADD.

## History

- 2026-10-11 — WAIT — concrete and open-source but a single vendor's blog with vendor benchmarks, the key interface is an unmerged proposal, and the course lacks the serving-systems prerequisites it builds on — weekly/2026-W41.md — candidates C-20261009-01
