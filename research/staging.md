# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261104-01 · NVIDIA Recovery + Attribution: open-source automated fault recovery and root-cause diagnosis for distributed training

- **Class:** A
- **Date discovered:** 2026-11-04
- **Date published:** 2026-11-02 (both blog posts)
- **Source:** https://developer.nvidia.com/blog/keep-distributed-training-running-with-automated-workload-recovery/ (verified resolves, 2 Nov 2026); https://developer.nvidia.com/blog/diagnose-distributed-ai-workload-failures-with-nvidia-attribution/ (verified resolves, 2 Nov 2026); https://github.com/dsx-ai-factory/NVIDIA-Recovery (verified resolves, public, Apache-2.0, 18 stars); https://github.com/dsx-ai-factory/attribution (verified resolves, public, Apache-2.0, 35 stars)
- **Organization/researchers:** NVIDIA (DSX / AI Factory infrastructure team; individual blog bylines not independently re-verified beyond the company byline).
- **Category:** distributed (training reliability/observability infrastructure)
- **What changed:** Two complementary open-source tools for large-scale training clusters. NVIDIA Attribution ingests failure symptoms from a training attempt (XID codes, NCCL events, framework logs, kernel messages, scheduler state, metrics) and returns one structured root-cause diagnosis per workload via a deterministic rules engine, explicitly distinguishing the originating fault from downstream cascading errors (e.g. one node's NVLink XID 149 vs. the NCCL errors it causes on other ranks) — reporting "ambiguous"/"unattributed" rather than forcing a guess when evidence is insufficient. NVIDIA Recovery sits above it: a detect → attribute → isolate → restart/stop loop that classifies crashes/hangs/stragglers, excludes implicated nodes from later attempts, and restarts jobs from checkpoint through a scheduler integration (Slurm now, Kubernetes on the roadmap); Attribution plugs in as the "attribute" stage but Recovery works without it.
- **Technical summary:** Blog-reported production numbers (NVIDIA's own cluster, Jul 9–Oct 9 2026 window): ~200 jobs/day, ~8,550 anomaly detections, mean detection latency 69s, mean recovery-action latency 94s (to the automated action; actual training resumption also depends on scheduling + checkpoint restore time, which the post states explicitly as a caveat, not a hidden omission). Attribution's own worked example: one RDMA-backed cluster, fabric failures traced to specific host/switch ports.
- **Why it might matter:** Distributed-training observability/fault-tolerance is explicitly in this protocol's "distributed training" search area and adjacent to the course's existing deferred `bitwise-deterministic-training.md` (Megatron-Core fingerprint-and-trace debugging, next review 2026-11-29) — a different but complementary problem (root-causing *why* a run died vs. making runs bit-reproducible so you can replay them). Nothing in the registry currently covers automated recovery/attribution as a topic.
- **Evidence of adoption:** Real external adoption beyond the vendor's own fleet: the Recovery post names IBM Research (running it on a Slurm training cluster, validated in monitor mode before enabling autonomous action) and Verda (an NVIDIA Cloud Partner, deployed with VictoriaLogs, has submitted PRs) as named users, not just "customers interested." NVIDIA's own claim of "100,000+ GPU fleet" use is a company claim. Caveat worth flagging: the Recovery *repo's own README* says the project is "experimental" and "not ready for production use" and that APIs may change — a narrower, more honest status statement than the blog post's framing, so I record both rather than only the more flattering one.
- **Major organizations using it:** NVIDIA (primary; fleet-scale per company claim), IBM Research (named, Slurm deployment), Verda (named, NVIDIA Cloud Partner).
- **Open-source implementation:** https://github.com/dsx-ai-factory/NVIDIA-Recovery (Apache-2.0); https://github.com/dsx-ai-factory/attribution (Apache-2.0).
- **Paper:** none (engineering blog + open-source release, not an academic paper).
- **Code:** https://github.com/dsx-ai-factory/NVIDIA-Recovery ; https://github.com/dsx-ai-factory/attribution
- **Relationship to existing course material:** Checked via `lookup "distributed training fault attribution"`, `lookup "NVIDIA Attribution"`, `lookup "workload recovery"` (no hits) and `lookup "bitwise deterministic"` (hits only the complementary, distinct `bitwise-deterministic-training.md`) — new topic, not a duplicate.
- **Potential course lesson:** A short addition to the distributed-training material (module 04/16 area, wherever large-scale run operations are covered) on *operating* a training run, not just coding one: detect/attribute/isolate/restart as a named pattern, contrasted with the bitwise-determinism approach to the same "a run died, now what" problem; could use the real latency numbers as a worked cost-of-failure example.
- **Confidence:** medium-high for "this is real, open-source, and has at least two named external adopters" (primary sources: NVIDIA's own blog posts, both GitHub repos independently verified); medium for "this is mature enough to teach as-is" given Recovery's own README calls it experimental/pre-production, which tempers the blog's more confident framing.
- **Recommendation:** weekly review should look at this for a short distributed-training-operations addition; not a full module on its own, but a good "what production infra actually does about training failures" grounding example, with the production-claim-vs-README-status tension noted explicitly if it is used.

### C-20261104-02 · Muon vs. AdamW, revisited: two new papers each chip at a different piece of the accepted lesson's story

- **Class:** B
- **Date discovered:** 2026-11-04
- **Date published:** 2026-11-02 (both v1)
- **Source:** https://arxiv.org/abs/2611.02529 (verified resolves — "Tune the Baseline First: Most of Muon's Scaled Weight Decay Gain Is a Tuning Gap, but Its Small Real Advantage Grows With Model Size"); https://arxiv.org/abs/2611.03633 (verified resolves — "Pretraining Data Scale Reverses the Muon-AdamW Fine-Tuning Gap")
- **Organization/researchers:** Paper 1: Esther Xin (single author; no affiliation stated on the abstract page; code+data released). Paper 2: Catherine Li, Jingchu Gai, Andrej Risteski, Jacob Mitchell Springer, Aditi Raghunathan (affiliation not stated on the page; Risteski and Raghunathan are established CMU ML-theory faculty per general knowledge of the names, not independently confirmed on the page).
- **Category:** training (optimizer — Muon vs. AdamW)
- **What changed:** The course's accepted Muon lesson (03.4) cites Moonshot's Moonlight report that scaling Muon's weight decay let them reuse AdamW's own learning rate/weight-decay settings with "no new tuning," explicitly flagged in the lesson as **(company claim)**. Paper 1 directly tests that claim: tuning learning rate and weight decay jointly (as a product) for both Muon-with-scaled-decay and plain AdamW on a small MoE ladder (1.9M–15M active params, one T4 GPU), it finds ~90% of the untuned 0.035-BPB gap disappears after proper tuning, leaving a small (0.0026–0.0051 BPB) but real and size-growing advantage, and that the optimal weight-decay product shifts faster with scale for Muon-SW than for AdamW (a law fit on two sizes overshoots at a third). Paper 2 is a different, complementary question the lesson doesn't address at all: it studies downstream *fine-tuning* accuracy (not pretraining loss) after Muon- vs AdamW-pretrained checkpoints, and finds the ranking reverses as pretraining data scale grows — AdamW overtakes Muon at large data/model scale — explained via a "margin-to-spread ratio" argument proved in a linear associative-memory model and checked in simulation.
- **Technical summary:** Paper 1: code+data released (github.com/ethxin0011/weight_decay_ladder); explicit numbers above; no stated venue, no third-party reproduction, and the scale (tens of millions of active params on a single consumer GPU) is far below anything the course's own Moonlight citation or Muon lesson example operates at, so generalization to real pretraining scale is untested by this paper itself. Paper 2: no code/data link found; no stated venue; a theoretical argument plus simulation, not a large-scale empirical reproduction either.
- **Why it might matter:** Both papers independently push back on parts of the Muon-vs-AdamW picture the course currently teaches as settled or as an unverified company claim, from two different angles (pretraining-time tuning artifact vs. fine-tuning-time scale-dependent reversal). Neither is independently strong enough alone, but together they're a signal that "is Muon's reported edge over AdamW real, and does it survive to fine-tuning?" is actively contested right now, which the lesson's existing "(company claim)" flag already anticipated.
- **Evidence of adoption:** None for either — two single/small-author academic papers, two days old, no stated venue, no third-party reproduction, no use in any production stack.
- **Major organizations using it:** None identified as using either finding; no stated institutional affiliation on either page.
- **Open-source implementation:** Paper 1: https://github.com/ethxin0011/weight_decay_ladder (author's own code+data). Paper 2: none found.
- **Paper:** https://arxiv.org/abs/2611.02529 ; https://arxiv.org/abs/2611.03633
- **Code:** https://github.com/ethxin0011/weight_decay_ladder (paper 1 only)
- **Relationship to existing course material:** Checked via `lookup "Muon scaled weight decay"` and `lookup "Muon AdamW fine-tuning gap"` (both hit only the accepted `muon-optimizer.md`/lesson 03.4, no existing deferred file on this specific nuance) — new evidence qualifying an already-taught, already-flagged company claim; not a duplicate, and not yet tracked anywhere.
- **Potential course lesson:** Not a new lesson; at most a caveat/footnote addition to 03.4's existing Moonlight citation (which already hedges with "(company claim)") noting that an independent small-scale replication found most of the claimed tuning-free benefit to be a tuning artifact, with a small real residual that grows with scale — and, separately, that Muon's pretraining-time edge is not guaranteed to transfer to downstream fine-tuning at large data scale. Low priority given both papers' small scale/no venue.
- **Confidence:** low for either paper standing alone (small scale or no empirical scale at all, no venue, no reproduction); the two together are a medium-confidence signal that this specific company claim deserves a watching brief.
- **Recommendation:** monitor; worth opening a small deferred-topic file at the weekly review to track "Muon vs AdamW: how real and how transferable is the reported edge" so future papers in this vein aren't scattered as one-offs; not an ADD on this evidence.

### C-20261104-03 · ParetoSaddler: separating intermediate-candidate promotion from final deployment choice in agent-harness optimization

- **Class:** B
- **Date discovered:** 2026-11-04
- **Date published:** 2026-11-01 (v1)
- **Source:** https://arxiv.org/abs/2611.02079 (verified resolves; cs.AI, cross-listed cs.CL/cs.LG/cs.MA)
- **Organization/researchers:** Xingzhi Qian, Sungho Park, Rongyuan Tan, Jianbo Yu, Jue Zhang, Lorenzo Cavallaro, Menglin Xia, Yanzhi Chen, Xinnuo Xu, Yu Kang, Xinran Zheng, Gagan Somashekar, Xuchao Zhang, Chetan Bansal, Qingwei Lin, Victor Rühle (affiliations not stated on the abstract page; Chetan Bansal and Qingwei Lin are established Microsoft Research systems/DevOps researchers per general knowledge of the names, and the project site is an `aka.ms` link, both consistent with but not a confirmed statement of Microsoft affiliation on the page itself).
- **Category:** agents/tools (agent-harness optimization, quality/cost trade-off)
- **What changed:** Agent-harness optimization (tuning the prompts/tools/control flow around a fixed model, not its weights) typically applies the deployment cost preference at every intermediate selection step, which the authors show can discard candidates early that would have led to better quality-cost trade-offs later. ParetoSaddler separates the two: during search, it promotes intermediate candidates using repeated runs plus a Bayesian-bootstrap Pareto test (comparing quality and cost distributions, not point estimates), deferring the cost-preference trade-off to a single final deployment-choice step over the surviving Pareto set.
- **Technical summary:** Authors' own numbers: on ClawMark, quality 70.86% → 81.35% while cost falls $1.17 → $1.01/task; on GAIA2, quality 56.22% → 62.89% while cost falls $0.78 → $0.45/task, both versus quality-only optimization baselines. Project site (aka.ms/ParetoSaddler-website) states code "will be available" there — not yet released at time of checking. No stated venue.
- **Why it might matter:** Directly new evidence for the course's already-active deferred `agent-harness-policy-co-optimization.md` cluster (Self-HPO, HarnessPro, HERD), mechanistically distinct from all three (none separate intermediate Pareto-test promotion from final single-objective deployment choice), and from a credible industrial source (plausibly Microsoft Research, given named researchers and an `aka.ms` link) with concrete, named-benchmark numbers (ClawMark, GAIA2) rather than only self-defined tasks.
- **Evidence of adoption:** None yet — single paper, three days old, no code released yet (promised, not delivered), no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified as adopters; plausible Microsoft Research authorship per the reasoning above, not confirmed on the page.
- **Open-source implementation:** none yet (project site promises a future release).
- **Paper:** https://arxiv.org/abs/2611.02079
- **Code:** https://aka.ms/ParetoSaddler-website (states code "will be available," not yet present)
- **Relationship to existing course material:** Checked via `lookup "ParetoSaddler agent harness"` (no hit) and by reading `research/deferred/agent-harness-policy-co-optimization.md` directly — same cluster/topic file as Self-HPO/HarnessPro/HERD, a new and mechanistically distinct entry, not a duplicate.
- **Potential course lesson:** If code is released and the result holds up, adds to that cluster's eventual resolution (likely inside the agents/tool-use module) as a concrete two-stage search/selection pattern for harness optimization under a quality-cost trade-off.
- **Confidence:** low-medium — concrete numbers on named benchmarks and a plausible strong-lab origin, but no code yet, no venue, no reproduction.
- **Recommendation:** add as a new candidate to `agent-harness-policy-co-optimization.md`; no change to its 2026-12-20 scheduled review, but worth re-checking for the promised code release before then.

### C-20261104-04 · SPTA: Student-Prefix Teacher Adaptation for on-policy distillation

- **Class:** B
- **Date discovered:** 2026-11-04
- **Date published:** 2026-11-02 (v1)
- **Source:** https://arxiv.org/abs/2611.03133 (verified resolves; cs.CL)
- **Organization/researchers:** Xue Zhang, Songming Zhang, Yunlong Liang, Fandong Meng, Yufeng Chen, Jinan Xu (affiliations not stated on the abstract page; Fandong Meng is an established WeChat AI/Tencent NLP researcher per general knowledge of the name, not independently confirmed on the page).
- **Category:** RL/post-training (on-policy distillation; teacher-adaptation mechanism)
- **What changed:** In on-policy distillation (OPD), the student's own rollout prefixes can contain errors or patterns the frozen teacher wasn't trained to continue from, making the teacher's per-token supervision less reliable on exactly the states the student actually reaches, worse for longer prefixes. SPTA adapts the teacher *to the student's distribution* before distillation: it collects student rollout prefixes of varying lengths and trains the teacher via RLVR to continue correctly from them (rewarding correct final answers), then runs standard, unmodified OPD with this adapted teacher. Distinct from task-level RLVR teacher adaptation (which the authors also compare against and beat) because it specifically targets the states the student reaches, not just general task competence.
- **Technical summary:** Authors report SPTA improves average student accuracy on math-reasoning benchmarks over both standard OPD and OPD with a teacher adapted via plain task-level RLVR. No specific numbers captured beyond this (abstract-level summary); no code repository found on the page; no stated venue.
- **Why it might matter:** A fourteenth mechanistically-distinct single-paper variant for the course's heavily-tracked `on-policy-distillation-variants.md` cluster (11 candidates currently in the deferred file, plus ATI-OPD and TESD staged on 2026-11-02/03 awaiting weekly merge) — distinct from all of them in adapting the teacher specifically to the student's own *rollout-prefix distribution* via RLVR, rather than weakening/rewriting/aligning-hidden-states as the other variants do.
- **Evidence of adoption:** None — single paper, two days old, no code, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; plausible WeChat AI/Tencent authorship per general knowledge of co-author Fandong Meng, not confirmed on the page.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2611.03133
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "student prefix teacher adaptation"` (no hit) and by reading `research/deferred/on-policy-distillation-variants.md` directly — same cluster as WISD/AOPD/PR-OPD/ATI-OPD/TESD etc., a new and distinct mechanism, not a restatement.
- **Potential course lesson:** If independently reproduced, folds into the cluster's eventual resolution (17.4 extension) alongside the other teacher-adaptation variants (TESD, ATI-OPD).
- **Confidence:** low — single paper, no code, no stated venue or affiliation, no reproduction.
- **Recommendation:** add as the (now) fourteenth candidate to `on-policy-distillation-variants.md`; no change to its 2026-11-29 scheduled review. Repeating yesterday's process observation more strongly: this cluster has accumulated 14 single-paper, unreplicated variants across six weeks with no ADD/WAIT/REJECT resolution yet — the weekly review may want to prioritize resolving it (even a WAIT-with-synthesis covering the whole family) over continuing to accumulate individual entries indefinitely.

### C-20261103-01 · FLy: lossy relaxed-verification speculative decoding, now a documented vLLM core feature

- **Class:** A
- **Date discovered:** 2026-11-03
- **Date published:** 2025-11-28 (v1); v3 2026-04-29; shipped in vLLM v0.33.0, released 2026-11-02
- **Source:** https://arxiv.org/abs/2511.22972 (verified resolves — title "Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match"; comments field states ICLR 2026); https://docs.vllm.ai/en/latest/features/speculative_decoding/fly/ (verified resolves, vLLM's own docs); https://github.com/vllm-project/vllm/releases/tag/v0.33.0 (verified resolves, lists "FLy" as a new opt-in lossy verification policy)
- **Organization/researchers:** Jinze Li, Yixing Xu, Guanchen Li, Shuo Yang, Jinfeng Xu, Xuanwu Yin, Dong Li, Edith C.H. Ngai, Emad Barsoum (affiliations not stated on the abstract page; Emad Barsoum is a known AMD GenAI researcher per general knowledge of the name, not independently confirmed on the page).
- **Category:** inference/serving (speculative decoding — relaxed/lossy verification)
- **What changed:** Standard speculative decoding verifies drafts with an exact-match rejection-sampling rule that preserves the target distribution exactly (this lossless rule is what the course's 08.6 already teaches). FLy ("Training-Free Loosely Speculative Decoding") relaxes this: at a high-entropy target position it can defer rejecting a draft token, checking a short window of later draft/verify agreement before finalizing the decision, using an entropy gate (top-k target probabilities) to decide where relaxation is safe. It needs no extra training. vLLM has now shipped this as an opt-in, fully documented `rejection_sample_method="fly"` policy (with `fly_window_size`, `fly_entropy_threshold`, `fly_entropy_top_k` knobs), explicitly flagged in vLLM's own docs as "intentionally lossy" — the first built-in vLLM verification policy that abandons the lossless guarantee by design (distinct from the existing lossless/exact-match and from DSpark's confidence-scheduled *lossless* adaptive verification).
- **Technical summary:** Paper's own numbers (Llama-3.1-70B/405B-Instruct): >99% of target-model accuracy retained, average speedups of 2.81x (70B) and 5.07x (405B); reports beating training-based EAGLE-3 by 1.62x on out-of-domain data. vLLM's own docs independently report measured quality retention across 29 configurations (minimum 97.9%) and separate NVIDIA/AMD single-pass evaluations (minimum accuracy retention 96.30% NVIDIA / 96.43% AMD, both corresponding to exactly one fewer correct AIME25 answer), plus measured speedups of ~1.07–1.94x in one evaluation setup and ~1.04–1.31x in another — i.e., vLLM re-measured the technique on its own stack rather than only citing the paper's numbers, which is a step beyond a company restating authors' claims.
- **Why it might matter:** This is precisely the trigger the deferred `speculative-decoding.md` cluster already names as the condition for re-judging its tracked "relaxed verification" sub-problem: *"independent reproduction, or adoption in vLLM/SGLang/TensorRT-LLM"* (added 2026-11-01, after 08.6). FLy is a different paper from the cluster's existing relaxed-verification candidate (C-20261019-03, arXiv 2610.23818, under review for NAACL 2027, no code, no adoption) — FLy is ICLR 2026 (peer-reviewed), and is now a shipped, documented, independently-remeasured vLLM feature. This is materially stronger evidence than anything currently in that cluster.
- **Evidence of adoption:** Strong for an A: merged into vLLM core (not a fork/plugin) and shipped in the current stable release (v0.33.0, 2 Nov 2026); vLLM's own docs independently re-measured quality/speedup rather than only citing the paper. Not yet found: SGLang or TensorRT-LLM adoption (checked SGLang v0.5.23 release notes directly — no mention); no third-party academic reproduction beyond vLLM's own re-measurement.
- **Major organizations using it:** vLLM (shipped as a core, documented feature). Possible AMD co-authorship (Emad Barsoum, per general knowledge of the name) is plausible but not confirmed on the paper page.
- **Open-source implementation:** https://github.com/vllm-project/vllm (core, v0.33.0+; docs at the URL above). No separate reference implementation from the paper authors was found.
- **Paper:** https://arxiv.org/abs/2511.22972
- **Code:** https://docs.vllm.ai/en/latest/features/speculative_decoding/fly/ (vLLM integration; no standalone author repo found)
- **Relationship to existing course material:** Checked via `lookup "FLy"` (no hit) and `lookup "relaxed verification speculative decoding"` (hits `research/deferred/speculative-decoding.md` and the accepted `inference-serving-foundations.md`/lesson 08.6). This is new evidence for the already-active WAIT cluster, not a duplicate of its existing relaxed-verification candidate (different paper, different authors, different venue, materially stronger adoption story) — exactly the kind of evidence the cluster's "what would change the decision" note asked for.
- **Potential course lesson:** A concrete extension of 08.6: after teaching the lossless draft-and-verify rule, FLy is a clean, real, shipped counter-example of a serving stack deliberately trading the lossless guarantee for speed, with vLLM's own measured accuracy-retention numbers as the worked "how much does lossy cost you" example — directly in the course's `.callout.guarantee`-style pattern (what it guarantees / does not guarantee / measured cost) if reused in the ai-security-course too.
- **Confidence:** high for "this is real and shipped" (primary sources: arXiv abstract page, vLLM's own current docs, vLLM's GitHub release notes all independently verified and mutually consistent); medium for "this generalizes" (single academic paper behind the shipped feature, no SGLang/TensorRT-LLM adoption yet, no third-party academic reproduction).
- **Recommendation:** weekly review should treat this as the headline update to `speculative-decoding.md` — likely an ADD as an 08.6 extension (lossy verification as a named, real trade-off), or at minimum a strong WAIT update that changes the cluster's priority.

### C-20261103-02 · XPress: causal refiner for block-diffusion speculative-decoding drafters, merged into vLLM core by the paper's own author

- **Class:** A
- **Date discovered:** 2026-11-03
- **Date published:** 2026-08-03 (v1); v2 2026-10-26; PR merged into vLLM 2026-10-22; shipped in vLLM v0.33.0, released 2026-11-02
- **Source:** https://arxiv.org/abs/2608.02438 (verified resolves — title "XPress: Parallel Refinement for Diffusion Drafters in Speculative Decoding"); https://github.com/vllm-project/vllm/pull/54448 (verified resolves, merged); https://github.com/vllm-project/vllm/releases/tag/v0.33.0 (verified resolves, lists "XPress" under Speculative decoding, citing PR #54448)
- **Organization/researchers:** Zheng Wang, Davis Wertheimer, Yu Chin Fabian Lim, Mudhakar Srivatsa, Raghu K. Ganti, Minjia Zhang, Naigang Wang (affiliations not stated on the abstract page; several of IBM Research's speculative-decoding team names recur across this group per general knowledge, not independently confirmed on the page). PR #54448 author "ZKBig" is the same Zheng Wang (first author), i.e. the paper's own author contributed the vLLM integration; merged by vLLM maintainer Benjamin Chislett.
- **Category:** inference/serving (speculative decoding — block-diffusion drafter training/refinement)
- **What changed:** Block-diffusion drafters (e.g. DFlash — already a shipped vLLM/vLLM-TPU speculator, >6x lossless speedup per its own paper, 3.13x measured by a Google TPU blog post) draft a whole block of tokens in one forward pass by sampling each position independently, so tokens in the block are not conditioned on each other — which causes early rejections by the target verifier. XPress adds a lightweight causal refiner, resolved via Jacobi iteration, that restores cross-token dependency within the drafted block without a token-by-token decoding loop. This is a refinement layered on top of existing diffusion drafters, not a new drafter itself.
- **Technical summary:** Paper's own numbers (Qwen3-8B, 7 math/code/chat benchmarks, vs. plain dFlash): ~30% average / up to 56% acceptance-length increase, ~1.3x average / up to 1.7x end-to-end throughput increase. The merged PR's own description (read directly) includes e2e GSM8K and throughput/acceptance-length benchmarks against both DFlash and DSpark baselines run on vLLM's own stack. No third-party academic reproduction found; no stated conference acceptance for the paper itself.
- **Why it might matter:** This is the second mechanism after yesterday's MDraft (C-20261102-01, staged as B) to target diffusion drafters for speculative decoding — yesterday's record noted the cluster "has no diffusion-drafter entry yet." XPress is mechanistically different from MDraft (causal post-hoc refinement of a drafter's output vs. MDraft's training-time multi-draft-block distillation target aggregation) but targets the same underlying cross-token-independence problem in block-diffusion drafting, and — unlike MDraft — it is not merely a paper: its own author merged it into vLLM core, and it is now a documented, shipped, V1 speculative-decoding method.
- **Evidence of adoption:** Strong: merged by a vLLM maintainer (not a fork), author-contributed, and shipped in the current stable release; PR includes vLLM-stack benchmarks against the two most relevant existing baselines (DFlash, DSpark) rather than only citing the paper's own numbers.
- **Major organizations using it:** vLLM (core, shipped). Possible IBM Research co-authorship (several named researchers associated with IBM's efficiency/speculative-decoding work per general knowledge) plausible but not confirmed on the page.
- **Open-source implementation:** https://github.com/vllm-project/vllm/pull/54448 (merged, commit ce9467b), shipped in v0.33.0.
- **Paper:** https://arxiv.org/abs/2608.02438
- **Code:** https://github.com/vllm-project/vllm (core, v0.33.0+)
- **Relationship to existing course material:** Checked via `lookup "XPress diffusion drafter"`, `lookup "DFlash"` (no hits) and `lookup "block diffusion drafter"` (hits only yesterday's staged MDraft record, confirming this is new, not a restatement) — new evidence for `research/deferred/speculative-decoding.md`, extending the diffusion-drafter sub-genre MDraft opened yesterday, but with materially stronger (merged/shipped) evidence than MDraft has.
- **Potential course lesson:** Together with C-20261103-01, supports extending 08.6 (or a future dedicated speculative-decoding lesson) with a short diffusion-drafter subsection — DFlash as the base mechanism (already real, TPU-validated), XPress as the "why naive parallel block drafting under-performs and how a causal refiner fixes it" worked example.
- **Confidence:** high for "this is real and shipped by the paper's own author into vLLM core"; medium for "this generalizes beyond the author's own benchmarks" (no independent third-party reproduction; comparisons are the PR author's own, albeit run directly on vLLM's stack).
- **Recommendation:** weekly review should read this alongside C-20261103-01 and yesterday's MDraft (B) as a cluster of real, current speculative-decoding developments; likely strengthens the case for an 08.6 extension covering lossy verification and diffusion drafters together.

### C-20261103-03 · ProVe: Proposer-Verifier Decoding (self-speculative, role-reversed block prediction)

- **Class:** B
- **Date discovered:** 2026-11-03
- **Date published:** 2026-10-20 (v1)
- **Source:** https://arxiv.org/abs/2610.27815 (verified resolves; cs.CL, cross-listed cs.LG)
- **Organization/researchers:** Ziqi Jin, Weiwen Xu, Jiaxi Li, Yujie Zhong, Qi She, Aixin Sun (affiliations not stated on the abstract page).
- **Category:** inference/serving (speculative decoding — proposer/verifier role reversal, self-speculative decoding)
- **What changed:** Standard speculative decoding uses a small drafter and a large verifier in separate forward passes. ProVe fine-tunes a standard autoregressive model into a "proposer" that predicts a whole block of future tokens in one forward pass (not via diffusion, unlike DFlash/XPress above), then verifies the block with a second forward pass. The verifier can be a smaller model, or the proposer itself — the latter ("ProVe-lossless") exactly reproduces the proposer's own greedy decoding output, making it self-speculative rather than a separate-draft-model scheme.
- **Technical summary:** Authors' own results across six benchmarks: with a 32B proposer, ProVe-lossless reaches 2.5x average throughput over plain autoregressive decoding without training a separate draft head; with a 3B verifier, 2.7x throughput at 2.6 accuracy points below AR average (a tunable lossy/speed trade-off). No code repository found on the page; no stated venue.
- **Why it might matter:** A third, mechanistically distinct speculative-decoding development found today, but self-speculative (same model plays both roles after fine-tuning) rather than diffusion-based or verification-relaxation-based — a different point in the same design space as the two A items above and as the cluster's existing five single papers.
- **Evidence of adoption:** None — single paper, two weeks old, no code, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; no stated affiliation.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2610.27815
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "proposer verifier decoding"` (no hit) and `lookup "self-speculative decoding"` (hits only the unrelated `continuous-flow-language-models.md` deferred topic, about flow-LM self-speculative decoding, a different setting) — genuinely new evidence for `speculative-decoding.md`, not a duplicate.
- **Potential course lesson:** If reproduced, a citation inside a future 08.6 extension on self-speculative (single-model) approaches, contrasted with the separate-draft-model and diffusion-drafter approaches.
- **Confidence:** low — single paper, no code, no stated affiliation or venue, no reproduction, though the "lossless when the model verifies itself" framing is a clean, checkable claim.
- **Recommendation:** add as a candidate to `speculative-decoding.md` alongside today's two A items and yesterday's MDraft; monitor at the cluster's 2026-11-15 scheduled review.

### C-20261103-04 · TESD: Teacher-Evolution Self-Distillation (reward-driven teacher updates in on-policy self-distillation)

- **Class:** B
- **Date discovered:** 2026-11-03
- **Date published:** 2026-10-28 (v1)
- **Source:** https://arxiv.org/abs/2610.38637 (verified resolves; cs.LG, cross-listed cs.AI)
- **Organization/researchers:** Shufan Shen, Zhongni Hou, Chao Bi, Yufei Zhang, Guojun Yin, Shuhui Wang (affiliations not stated on the abstract page).
- **Category:** RL/post-training (on-policy self-distillation; teacher-update mechanism)
- **What changed:** In on-policy self-distillation (OPSD), a frozen self-teacher (the same model, given privileged context) can give unreliable guidance on some samples, stalling the student's gains. TESD adds: (1) teacher-oriented optimization — the self-teacher's own rollouts are rewarded for task correctness and context-independence, so the teacher improves over training rather than staying frozen; (2) student-oriented optimization — per-sample guidance reliability is estimated from those same rewards, and unreliable samples are down-weighted in the student's distillation loss.
- **Technical summary:** Authors' abstract reports TESD "consistently outperforms representative token-level and sequence-level methods" across multiple datasets/scales; no specific numbers given in the abstract itself (full text not fetched beyond the abstract page). No code repository found on the page; no stated venue.
- **Why it might matter:** A 13th mechanistically-distinct single-paper variant for the course's already heavily-tracked `on-policy-distillation-variants.md` cluster (12 prior candidates, including yesterday's WISD/ATI-OPD). Distinct from the tracked variants: it specifically evolves the *teacher* via its own reward signal (task correctness + context-independence) rather than weakening a copy of the model (WISD), rewriting divergent segments (ATI-OPD), or aligning hidden states (PR-OPD). Given how saturated this cluster already is with single, unreplicated papers, this is recorded for completeness but flagged as lower-priority than the two A items today.
- **Evidence of adoption:** None — single paper, six days old, no code, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; no stated affiliation.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2610.38637
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "teacher evolution self-distillation"` (no hit) and by reading `research/deferred/on-policy-distillation-variants.md` directly (same cluster/topic file as WISD, AOPD, PR-OPD, ATI-OPD, etc.; this is a new, distinct mechanism, not a restatement).
- **Potential course lesson:** If independently reproduced, folds into the cluster's eventual resolution (17.4 extension or a short survey of "what makes a good self-teacher" tricks) alongside WISD/ATI-OPD.
- **Confidence:** low — single paper, no code, no stated venue or affiliation, no reproduction, abstract-only numbers (not even reported in the abstract itself).
- **Recommendation:** add as the thirteenth candidate to `on-policy-distillation-variants.md`; no change to its 2026-11-29 scheduled review. Given the cluster's size, the weekly review may want to consider whether it has enough evidence to resolve (ADD/WAIT/REJECT) rather than keep accumulating single-paper variants indefinitely — noted here as a process observation, not a decision this run can make.

### C-20261102-01 · MDraft: Multi-Draft On-Policy Distillation for Diffusion Drafters

- **Class:** B
- **Date discovered:** 2026-11-02
- **Date published:** 2026-10-27 (v1, 18:02:08 UTC)
- **Source:** https://arxiv.org/abs/2610.37778 (verified resolves; v1, Tue 27 Oct 2026, cs.CL)
- **Organization/researchers:** Petr Molodyk, Yuchen Zhu, Jaemoo Choi, Molei Tao, Yongxin Chen. Molei Tao and Yongxin Chen are established Georgia Tech applied-math/ML faculty per general knowledge of the named researchers; affiliation is not stated on the abstract page itself.
- **Category:** inference/serving (speculative decoding, diffusion drafters) + RL/post-training (on-policy distillation)
- **What changed:** One-step diffusion drafters propose several speculative-decoding tokens in parallel, but training them against a verifier's feedback is hard because each drafted token in a block is scored without seeing the other tokens in the same block — the authors show distilling against these per-token verifier targets separately suppresses tokens that are plausible under some prefixes, hurting draft acceptance length. MDraft (Multi-Draft On-Policy Distillation) samples several draft blocks from the same context and aggregates the verifier's full-vocabulary predictions across their prefixes before distilling, changing only training (architecture and inference procedure are unchanged).
- **Technical summary:** Authors' own results on math, coding, and conversational tasks: MDraft yields longer accepted draft lengths than prior distillation methods for diffusion drafters, beyond the pretrained drafter's acceptance rate. No independent reproduction found; no stated venue; no code repository linked on the abstract page (checked the "Code, Data and Media" section directly — only generic third-party-tool toggles, no paper-specific link).
- **Why it might matter:** Intersects two course clusters at once: the newly-added inference-serving foundation (`lessons/module-08/lesson-06.md`, 08.6, which teaches lossless draft-and-verify speculative decoding) and the accepted on-policy-distillation lesson (17.4). The course's deferred `speculative-decoding.md` cluster already tracks "RL-trained drafters" as one open sub-problem but has no diffusion-drafter entry yet — this is the first.
- **Evidence of adoption:** None — single paper, six days old at discovery, no code, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; academic authorship, no stated affiliation on the page.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2610.37778
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "diffusion drafter"` (no hit) and by reading `research/deferred/speculative-decoding.md` directly, whose "what would change the decision" explicitly asks for independent reproduction or vLLM/SGLang/TensorRT-LLM adoption before any refinement in this cluster becomes an 08.6 extension — new evidence (a new sub-genre: diffusion drafters) for that already-active WAIT topic, not a duplicate of its five tracked papers (relaxed verification, four October single papers, the MoE-speedup NeurIPS paper).
- **Potential course lesson:** If reproduced or adopted by a serving stack, a citation inside a future 08.6 extension on drafter training, alongside the cluster's other RL/distillation-trained-drafter work.
- **Confidence:** low-medium — a clean mechanistic argument (block-level target aggregation) with concrete acceptance-length results, but single paper, no code, no stated affiliation or venue, no reproduction.
- **Recommendation:** monitor alongside `speculative-decoding.md` at its 2026-11-15 scheduled review; add as a candidate to that file.

### C-20261102-02 · FAD-KV: Adaptive Dictionary KV Caching for Efficient Reasoning Decoding

- **Class:** B
- **Date discovered:** 2026-11-02
- **Date published:** 2026-10-27 (v1, 20:05:31 UTC)
- **Source:** https://arxiv.org/abs/2610.37921 (verified resolves; v1, Tue 27 Oct 2026, cs.CL)
- **Organization/researchers:** Yuxuan Zhu, David H. Yang, Parisa Shiravani, Mohammad Mohammadi Amiri, Tejaswini Pedapati, Subhajit Chaudhury, Keerthiram Murugesan, Pin-Yu Chen. Pin-Yu Chen is a well-known IBM Research scientist per general knowledge of the name; affiliation is not stated on the abstract page itself.
- **Category:** inference/serving (KV-cache compression for long-CoT reasoning decoding)
- **What changed:** Studies dictionary-based sparse KV compression (each token's KV state as a sparse combination of learned atoms) and identifies two decoding-time problems: iterative sparse coding adds overhead, and a static dictionary falls out of sync with evolving KV distributions during generation (naive adaptive updates make the dictionary grow unboundedly). FAD-KV uses a one-shot Top-n screening step plus lightweight coefficient refinement (replacing iterative sparse coding) and a fixed-size dictionary that reuses inactive atoms (bounding memory) instead of growing it.
- **Technical summary:** Authors' own results on reasoning and long-context benchmarks: a "strong quality-efficiency trade-off" with bounded memory growth. No independent reproduction found; no stated venue; no code repository linked (checked directly — only generic third-party-tool toggles).
- **Why it might matter:** The course's deferred `reasoning-kv-compression-pause-tokens.md` (next review 2026-12-13) tracks a single training-based mechanism (AbridgeKV: summarize finished reasoning blocks into learned pause-token KV entries) for the same problem (bounding KV growth during long-CoT decoding). FAD-KV is a mechanistically distinct, training-free-at-inference (dictionary/sparse-coding) approach to the same problem, and is also directly adjacent to the newly-added `lessons/module-08/lesson-06.md` (08.6) inference-serving foundation, which teaches paged KV memory but not compression.
- **Evidence of adoption:** None — single paper, six days old at discovery, no code, no stated venue, no third-party reproduction. The rejected `query-aware-kv-cache-compression.md` and `chunked-kv-compression-phase-sensitivity.md` topics are a reminder that this sub-field has a history of single-paper KV-compression claims that did not hold up — noted as a reason for caution, not for immediate rejection (this is a different mechanism from both).
- **Major organizations using it:** None identified; authors plausibly IBM-affiliated per general knowledge of co-author Pin-Yu Chen, not confirmed on the page.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2610.37921
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "KV cache compression"` (hits the two rejected topics above, plus unrelated tokenizer/MLA mentions) and by reading `reasoning-kv-compression-pause-tokens.md` directly — this is a different mechanism for the same problem the WAIT topic already tracks, not a restatement of AbridgeKV.
- **Potential course lesson:** If reproduced, could join `reasoning-kv-compression-pause-tokens.md`'s eventual resolution as a second, mechanistically distinct data point (dictionary/sparse-coding vs. learned-summary-token), or extend 08.6 directly if a KV-compression sub-lesson is ever built.
- **Confidence:** low — single paper, no code, no stated affiliation or venue, no reproduction, and the sub-field has two prior rejects for related (but distinct) KV-compression claims.
- **Recommendation:** monitor; add as a second candidate alongside AbridgeKV in `reasoning-kv-compression-pause-tokens.md`, noting the mechanism difference explicitly so the two are not conflated at the next review.

### C-20261102-03 · ATI-OPD: Adaptive Teacher Intervention for Stable Long-Sequence On-Policy Distillation

- **Class:** B
- **Date discovered:** 2026-11-02
- **Date published:** 2026-10-28 (v1, 02:00:26 UTC)
- **Source:** https://arxiv.org/abs/2610.38160 (verified resolves; v1, Wed 28 Oct 2026, cs.CL)
- **Organization/researchers:** Jiannan Guan, Youxin Jiang, Yutai Hou, Qiguang Chen, Yirong Zeng, Haitao Wang, Dengyun Peng, Zheng Yan, Jinhao Liu, Tao Ye, Dandan Tu, Wanxiang Che. Wanxiang Che and Qiguang Chen are established Harbin Institute of Technology NLP researchers per general knowledge of the names; affiliation is not stated on the abstract page itself.
- **Category:** RL/post-training (on-policy distillation; teacher-intervention mechanism)
- **What changed:** For long responses in on-policy distillation, the student's early reasoning can drift from the teacher's, weakening the teacher's token-level feedback on later tokens and destabilizing training. Existing fixes (length control, loss reweighting) limit learning from problematic responses without changing their prefixes. ATI-OPD instead replaces the earliest high-disagreement segment of the student's own generation with a short teacher-written segment, lets the student continue from there, and then removes persistently divergent suffixes from the distillation loss (rather than the whole response) to keep useful later continuations.
- **Technical summary:** Authors' own training diagnostics report reduced teacher-student disagreement, fewer gradient spikes, and fewer extremely long responses; across five math-reasoning benchmarks and two teacher-student pairs, ATI-OPD improves average student-only accuracy over standard OPD. No code repository linked on the page; no stated venue; no independent reproduction found.
- **Why it might matter:** A twelfth single-paper variant in the course's actively-tracked `on-policy-distillation-variants.md` cluster (11 candidates already tracked), but mechanistically distinct from all of them — none of the tracked variants do segment-level teacher rewriting combined with persistent-divergence loss masking specifically for long sequences (closest tracked items are AOPD's advantage-weighted teacher-following and PR-OPD's hidden-state alignment, both different mechanisms).
- **Evidence of adoption:** None — single paper, five days old at discovery, no code, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; academic authorship (HIT per general knowledge), not confirmed on the page.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2610.38160
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "on-policy distillation"` and by reading `research/deferred/on-policy-distillation-variants.md` directly — confirms this is the same cluster/topic file as WISD, AOPD, PR-OPD, etc., but a mechanistically new variant (segment-level teacher-rewrite intervention for long-sequence stability), consistent with how the cluster file has treated each new distinct mechanism (e.g. WISD on 2026-11-01).
- **Potential course lesson:** If independently reproduced or if teacher-intervention-style fixes converge across several papers, a short addition to 17.4 on long-sequence stability tricks.
- **Confidence:** low — single paper, no code, no stated venue or independently confirmed affiliation, no reproduction; this specific "teacher rewrites the divergent segment" idea also appears (under different names/papers, e.g. MAESTRO/AC-OPD-style work found via general web search but not yet in the course's tracked list) to be an increasingly crowded sub-genre, which argues for monitoring rather than urgency.
- **Recommendation:** add as the twelfth candidate to `on-policy-distillation-variants.md`; no change to its 2026-11-29 scheduled review date.

### C-20261102-04 · Artifact Distillation: Training-Free Transfer from Frontier to Small Language Models

- **Class:** B
- **Date discovered:** 2026-11-02
- **Date published:** 2026-10-12 (v1, 20:27:04 UTC) — outside the usual ~48h discovery window but not previously recorded in any daily file; surfaced via the arXiv cs.CL Oct-30 listing rescan, flagged here as newly found, not newly published.
- **Source:** https://arxiv.org/abs/2610.37680 (verified resolves; v1, Mon 12 Oct 2026, cs.CL/cs.AI/cs.LG)
- **Organization/researchers:** Mustafa Safa Ozdayi, Somin Wadhwa. Affiliation not stated on the abstract page.
- **Category:** data/distillation (training-free, inference-time capability transfer)
- **What changed:** Proposes transferring capability from a frontier model to a smaller one without any gradient update. A teacher model solves training problems and writes its problem-solving procedures into a natural-language "artifact bank"; the student model consults this bank at inference time (in-context), rather than being fine-tuned on teacher outputs as in standard distillation.
- **Technical summary:** Authors' own results across four LiveBench tasks: Claude Haiku 4.5 using an artifact bank written by Claude Opus 4.5 outscored unaided Claude Sonnet 4.5 on spatial reasoning (0.707 vs. 0.647) at roughly 60% of the inference cost; on table-joining, Haiku-with-bank matched Sonnet 4.5 at roughly 89% of the cost; the same bank raised Qwen3.5-27B's pooled score from 0.730 to 0.800, suggesting cross-model-family transfer. These are API-measured benchmark results from named, checkable models (not a self-reported training claim), but still a single paper with no independent reproduction. No code repository linked on the page (checked directly); no stated venue.
- **Why it might matter:** A genuinely new mechanism not covered by any existing course material or registry entry: "distillation" with zero parameter updates, via natural-language procedure transfer consulted at inference. The course's on-policy-distillation material (17.4) and its variants cluster are all gradient-based; this is a distinct category (closer to in-context learning / prompting research than to RL post-training), with concrete, verifiable cross-model-family numbers.
- **Evidence of adoption:** None — single paper, two unaffiliated authors, no code, no stated venue, no third-party reproduction. The use of real production models (Claude Opus/Sonnet/Haiku 4.5, Qwen3.5-27B) as test subjects is methodologically stronger than a purely synthetic benchmark, but is not itself adoption evidence.
- **Major organizations using it:** None identified as adopters of the technique; the paper merely benchmarks against commercially available models.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2610.37680
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "training-free distillation"` and `lookup "in-context distillation procedure bank"` (no hits either way) — new topic, not a duplicate of any accepted/deferred/rejected file or lesson.
- **Potential course lesson:** If reproduced or adopted, a short new topic (not an extension of 17.4, since the mechanism is training-free) contrasting gradient-based on-policy distillation with inference-time, artifact-bank-based transfer — possibly module 17 or a RAG/memory-adjacent module given the "bank consulted at inference" structure.
- **Confidence:** low — a single paper with no code, no stated affiliation, no venue, and no reproduction, but a clean, falsifiable idea with concrete numbers on real, named production models across two different model families.
- **Recommendation:** monitor; worth a from-scratch sanity check (writing a small artifact bank and testing transfer on a toy task) if the weekly review wants an early read before deciding whether to open a new deferred topic file.

### C-20261105-01 · Dory1.0-4B: NVIDIA's open-weight looped (recurrent-depth) language model

- **Class:** A
- **Date discovered:** 2026-11-05
- **Date published:** 2026-11-02 (arXiv paper, v1); open weights quietly uploaded to Hugging Face circa 2026-10-16 per third-party reporting (not independently timestamped by me beyond "updated 5–19 days ago" on the HF listing page at check time)
- **Source:** https://arxiv.org/abs/2611.03998 (verified resolves); https://huggingface.co/models?search=dory1.0 (verified resolves — confirms `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-{BF16,MXFP8,NVFP4,GGUF}` under the real `nvidia` org namespace, plus ~16 third-party derivative repos — community GGUF/MLX/CoreML/MLC conversions — as of this check)
- **Organization/researchers:** NVIDIA, Nemotron Labs team. Authors: Maxim Fishman, Simeng Sun, Shantanu Acharya, Faisal Ladhak, Brian Chmiel, Santiago Akle Serano, Ron Banner, Boris Ginsburg, Krishna C. Puvvada. Ginsburg and Banner are also authors of NVIDIA's original nGPT paper, consistent with (not independently re-confirmed beyond) the stated NVIDIA/Nemotron-Labs HF org namespace.
- **Category:** architecture (looped / recurrent-depth language model)
- **What changed:** A 4B-parameter model with an 8-layer input block, a 20-layer shared recurrent block run a variable 1–4 times at inference (elastic depth from one checkpoint), and an 8-layer output block, on an nGPT (normalized-Transformer) backbone with interleaved sliding-window/full attention. Pretrained on 6T tokens (first 4.5T in NVFP4), extended to 256K context; post-trained with SFT + RLVR using a difficulty-aware loop curriculum. KV cache is kept to a non-looped model's footprint by caching only the last loop's activations.
- **Technical summary:** Authors' own numbers: with ≥2 loops, average score across 12 benchmarks exceeds Qwen3.5-9B, Gemma 4 12B, and Gemma 4 26B-A4B **(company claim — no independent evaluation found)**. Open-sourced BF16/MXFP8/NVFP4/GGUF checkpoints; no separate training-code repo found, weights-only release.
- **Why it might matter:** `research/deferred/looped-recurrent-depth-lms.md` (WAIT, next review 2026-11-29) explicitly names "a looped model released by a major lab" as a condition that would change the decision. This is exactly that: a real, open-weight, major-lab (NVIDIA) looped model, not another single toy-scale academic paper. It also lands in the same week as two more looped-transformer papers — ARLoop (attention-based depth residuals, 1.2B scale, beats Huginn-3.5B — authors' claim, no code found) and LoopVQ (sub-1-bit KV cache quantization specifically for looped transformers) — both logged under C below, which the weekly review may read as the cluster's activity continuing to converge rather than scatter.
- **Evidence of adoption:** Real, verifiable community derivative activity: at least ~16 independent re-uploads/quantized conversions (GGUF, MLX 4-/6-/8-bit, CoreML/ANE, MLC) by unrelated HF users within roughly two to three weeks of the quiet release — genuine usage signal beyond NVIDIA's own claim. Counterweight, flagged explicitly rather than omitted: third-party commentary (several blog aggregators, not independently re-verified by me beyond their consistent agreement) reports NVIDIA made **no official announcement or developer-blog post** for the release, and that the benchmark superiority claims are **unverified against an independent evaluation** — i.e. real open-weight release, but a "quiet drop" with self-reported numbers, not a headlined, externally validated launch.
- **Major organizations using it:** NVIDIA (origin, Nemotron Labs). No other major lab has adopted the architecture; community adopters are individual/small-scale HF users building format conversions, not production deployments.
- **Open-source implementation:** https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 (and sibling MXFP8/NVFP4/GGUF repos) — weights only, no distinct training-recipe repo found.
- **Paper:** https://arxiv.org/abs/2611.03998
- **Code:** none found beyond the HF weight repos (no dedicated GitHub training/inference repo located).
- **Relationship to existing course material:** Checked via `lookup "looped transformer"` → hits `research/deferred/looped-recurrent-depth-lms.md` (WAIT) and the from-scratch optimizer lesson (unrelated hit on generic "optimizer" text). This is new evidence for that existing cluster, not a duplicate — it is the first major-lab, open-weight entry in a cluster so far built entirely from academic 100M–1.6B-scale papers.
- **Potential course lesson:** If the weekly review resolves the cluster, this is the natural real-world grounding example for a Module 12 toy looped-transformer lesson (shared block + input injection + adaptive exit, built from scratch at toy scale, then "here is what this looks like at 4B/6T tokens from a major lab") — with the self-reported-benchmark caveat stated explicitly, the same way the Muon lesson flags Moonlight's claim.
- **Confidence:** medium-high that the release itself is real and has genuine (if small-scale) community adoption (independently verified: real `nvidia` HF namespace, real third-party derivative repos); medium that it is "mature enough to teach as a settled recipe" given the no-announcement, no-independent-eval, self-reported-benchmarks caveats above.
- **Recommendation:** weekly review should treat this as the trigger event for `looped-recurrent-depth-lms.md`'s "major lab release" condition and decide whether to resolve the cluster now (ADD a short toy lesson with Dory1.0-4B as the grounding example, benchmark claims explicitly flagged) rather than deferring to 2026-11-29, given this item plus ARLoop plus LoopVQ all landing the same week.

### C-20261105-02 · SpecOPD: branch-parallel on-policy distillation for speculative-decoding drafters

- **Class:** B
- **Date discovered:** 2026-11-05
- **Date published:** 2026-11-02 (v1)
- **Source:** https://arxiv.org/abs/2611.02846 (verified resolves; cs.AI); also surfaced independently via https://huggingface.co/papers/2611.02846 (verified resolves, Hugging Face daily-papers listing for 2026-11-05)
- **Organization/researchers:** 15 authors (Tianyu Liu, Jun Zhang, Wenjie Huang, and 12 others); affiliation not stated on the abstract page.
- **Category:** inference/efficiency (speculative decoding) and RL/post-training (on-policy distillation applied to drafter training)
- **What changed:** Standard on-policy distillation (OPD) for speculative-decoding drafters requires slow, sequential draft-and-verify rollouts to get training signal. SpecOPD samples drafts and computes target supervision in parallel across independent branches from the evolving drafter, packed into proposal trees that share target-prefix computation (tree masks for full attention, grouped indexers for sparse attention, indexed state replay for linear attention), trained with an "E2E-TV" surrogate objective that directly targets acceptance length rather than token-level likelihood.
- **Technical summary:** Authors' own numbers across 8B full-attention, 27B full-linear, and 125B sparse-linear target models, three parallel drafter architectures (DFlash, DSpark, DFlash V2), and four domains (dialogue, coding, math, agentic): up to 41.45% higher mean acceptance length, up to 16.09x supervised-token throughput over sequential-rollout OPD, up to 3.56x generation speedup over autoregressive decoding. No stated venue; no code repository found.
- **Why it might matter:** Directly extends two clusters the course already tracks: `research/deferred/speculative-decoding.md` (WAIT, next review 2026-11-15, whose evidence list already includes "RL-trained drafters" as an open sub-problem on top of the lossless draft-and-verify rule taught in lesson 08.6) and the accepted on-policy-distillation material (lesson 17.4), applied here to a different target (a drafter optimized for acceptance length, not a reasoning student).
- **Evidence of adoption:** None — single paper, three days old, no code, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; no stated affiliation.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2611.02846
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "speculative decoding branch parallel on-policy distillation"` (no hit) and by reading `research/deferred/speculative-decoding.md` directly — new, mechanistically specific entry for its "RL-trained drafters" sub-problem; not a duplicate.
- **Potential course lesson:** If reproduced, a candidate extension of 08.6 (drafter training via OPD, acceptance-length-targeted objective) cross-linked to 17.4.
- **Confidence:** low-medium — large, scale-spanning experiments (8B/27B/125B) and concrete numbers, but no code, no stated venue/affiliation, no reproduction.
- **Recommendation:** add as new evidence to `speculative-decoding.md` ahead of its 2026-11-15 review; cross-reference the accepted on-policy-distillation lesson.

### C-20261105-03 · FlagRot: per-group selective rotation for low-bit KV cache quantization

- **Class:** B
- **Date discovered:** 2026-11-05
- **Date published:** 2026-11-02 (v1)
- **Source:** https://arxiv.org/abs/2611.03992 (verified resolves; cs.CL/cs.AI/cs.LG); https://github.com/KunmingSHAO/FlagRot_releaser (verified resolves, public, Apache-2.0, code present)
- **Organization/researchers:** Kunming Shao, Xijie Huang, Jierun Chen, and 7 others; affiliation not stated on the abstract page.
- **Category:** inference/efficiency (KV cache quantization)
- **What changed:** Existing low-bit KV cache methods rotate keys before quantizing (e.g. RotateKV), but the authors find rotation doesn't always reduce error. FlagRot stores each 32-channel group either rotated (Walsh-Hadamard or whole-head Hadamard) or left unrotated, whichever gives lower calibrated attention error, decided per head (for per-channel caches like KIVI) or per key (for per-token caches), with the choice packed into existing sign bits; the query is rotated to match at read time.
- **Technical summary:** Authors' own numbers: at 2.5 bits/value, reduces KIVI's KL divergence from the BF16 next-token distribution across 12 models and beats RotateKV on 9 of 12; as a vLLM 0.30 plugin with CUDA (Hopper/sm_90a) decode-attention kernels, improves RULER@128k and LongBench v1 scores specifically where 2-bit KIVI loses accuracy; also lowers error for 4-bit integer and FP4 caches. The GitHub repo (verified) contains the simulator, the vLLM plugin, CUDA kernels, reproduction scripts, and unit tests, Apache-2.0 licensed (one bundled LongBench-derived file under its own MIT terms).
- **Why it might matter:** A concrete, code-complete, vLLM-integrated extension of the KV-cache material already in the accepted `research/accepted/inference-serving-foundations.md` lesson, spanning 12 models rather than one.
- **Evidence of adoption:** Code released (Apache-2.0, functional vLLM plugin) but the repo itself shows only 1 star at check time — real engineering artifact, no external adoption yet; single paper, three days old, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; no stated affiliation.
- **Open-source implementation:** https://github.com/KunmingSHAO/FlagRot_releaser (Apache-2.0)
- **Paper:** https://arxiv.org/abs/2611.03992
- **Code:** https://github.com/KunmingSHAO/FlagRot_releaser
- **Relationship to existing course material:** Checked via `lookup "KV cache quantization"` (hits the accepted `inference-serving-foundations.md`, which teaches the KV-cache concept but not a specific low-bit quantization method) and `lookup "low-bit KV cache rotation quantization"` (no hit) — a new, more specific technique, not a duplicate of the foundation lesson. Distinct from the author's PTQ scope decision (2026-10-28, weight quantization) — this is activation/KV-cache quantization, a different target; worth flagging that distinction explicitly for whoever reviews it.
- **Potential course lesson:** Possible short addition to the inference-serving material once/if a PTQ-adjacent extension is being built, as a worked "which bits actually need rotation" example; not foundational on its own.
- **Confidence:** medium on technique soundness (real code, 12-model sweep, vLLM-integrated); low on adoption (3 days old, 1 star, no reproduction).
- **Recommendation:** monitor; no existing deferred file is an exact fit, so weekly review should decide whether to open one or fold it into a future KV-cache/PTQ-adjacent extension.

### C-20261105-04 · AEGIS loss: a geometric account of SFT capability degradation via feature interference

- **Class:** B
- **Date discovered:** 2026-11-05
- **Date published:** 2026-11-03 (v1)
- **Source:** https://arxiv.org/abs/2611.05003 (verified resolves; cs.LG)
- **Organization/researchers:** Hang Chen, Jiaying Zhu, Wenya Wang; affiliation not stated on the abstract page.
- **Category:** training / post-training (SFT catastrophic-forgetting mechanism and mitigation loss)
- **What changed:** Argues SFT-induced loss of general capability is a bottom-up geometric effect — narrow task data disrupts the near-orthogonal feature structure learned in pretraining, pushing activations across decision boundaries for unrelated tasks — rather than only a macro-level distribution-shift story. Proves standard one-hot cross-entropy cannot resolve this alone, and proposes the AEGIS loss, which bounds how far activation margins drift toward competing tokens, to separate "suppress interference" from "learn the task."
- **Technical summary:** Authors' own numbers: across 3 model families and 14 SFT tasks, AEGIS beats representative methods from each of six paradigms the authors say their framework unifies, on both the target task and retained general capability, at roughly 1.3x the compute of plain SFT. No code repository found; no stated venue.
- **Why it might matter:** A mechanistic, falsifiable account (with a concrete proposed fix) of a problem every post-training pipeline in the course already has to deal with empirically (catastrophic forgetting during SFT); distinct from the course's existing softmax/numerical-fix and weight-decay deferred topics.
- **Evidence of adoption:** None — single paper, two days old, no code, no stated venue/affiliation, no third-party reproduction.
- **Major organizations using it:** None identified.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2611.05003
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "feature interference SFT capability degradation"` and `lookup "SFT capability degradation feature interference"` (no hits) — new topic, not a duplicate.
- **Potential course lesson:** If reproduced, a candidate caveat/technique addition to whichever post-training module covers SFT trade-offs (general-capability retention vs. task performance).
- **Confidence:** low — single paper, no code, no venue, no reproduction, despite a reasonably broad experimental sweep (3 families x 14 tasks) for an academic paper at this stage.
- **Recommendation:** monitor; worth a small deferred-topic file if the weekly review judges the mechanistic account itself (not just AEGIS specifically) interesting enough to track independent of whether AEGIS as a specific loss gets reproduced.

### C-20261105-05 · NVIDIA NeMo Compass + Eval Author: trace-to-eval-to-harness-fix loop for coding/domain agents

- **Class:** B
- **Date discovered:** 2026-11-05
- **Date published:** 2026-11-04
- **Source:** https://developer.nvidia.com/blog/use-nvidia-nemo-to-turn-agent-failures-into-tests-and-harness-fixes/ (verified resolves, 4 Nov 2026); https://github.com/NVIDIA-NeMo/nemo-compass (verified linked from the post); https://github.com/NVIDIA-NeMo/eval-author (verified linked from the post)
- **Organization/researchers:** NVIDIA. Byline: Patrick Moorhead (technical marketing engineering), Sean Lopp (software engineer, data/AI/developer tooling), Moon Chung (senior product marketing manager) — flagged explicitly: this is substantially a marketing-authored post, not a research-team write-up, even though it describes and links real open-source tooling.
- **Category:** agents/tools (agent-harness iteration loop)
- **What changed:** A detect → test → fix → measure loop for domain-specific agents, distinct from RL-policy harness co-optimization: NeMo Compass analyzes stored agent traces against an expected-behavior spec and surfaces recurring failure patterns with trace-ID evidence; NeMo Eval Author turns those findings into executable evaluations (Harbor/NeMo Gym format) via a coding agent; a coding agent then proposes a harness change (system-prompt rule, validation step, middleware); results are checked on both development and held-out tasks to catch overfitting, and the loop repeats.
- **Technical summary:** Two reported cases. (1) A fictional/internal "Hermes" assistant example: development set 3/6 → 6/6 passes (tool calls 10 → 6), held-out set 4/6 → 6/6 (tool calls 28 → 15) — self-reported, very small sample (2 tasks x 3 runs), not statistically robust. (2) A named external case, Dassault Systèmes (CAD/engineering, "LEO" agent): malformed tool calls 142 → 0, 28% relative improvement in overall evaluation score, 35% fewer tool calls per run across six scenarios, optimization run ~12h with no reported regressions — the only externally-named adopter, but the specific numbers are still NVIDIA's own write-up of the case, not an independent report.
- **Why it might matter:** Same problem class as the already-tracked `research/deferred/agent-harness-policy-co-optimization.md` (WAIT, next review 2026-12-20) — closing the loop from agent failures to harness changes — but via a different mechanism (coding-agent-driven trace analysis and eval authoring, no RL policy training) and, unusually for that cluster, backed by real shipped open-source tooling and one named external production user rather than only an academic single paper.
- **Evidence of adoption:** Real open-source repos exist and are linked (`NVIDIA-NeMo/nemo-compass`, `NVIDIA-NeMo/eval-author`); one named external user (Dassault Systèmes) with concrete numbers, though NVIDIA-reported. No independent/third-party write-up of the Dassault case was found beyond NVIDIA's own post and a referenced Dassault blog/GTC video (not separately fetched this run).
- **Major organizations using it:** NVIDIA (origin); Dassault Systèmes (named external user, engineering/CAD agent).
- **Open-source implementation:** https://github.com/NVIDIA-NeMo/nemo-compass ; https://github.com/NVIDIA-NeMo/eval-author
- **Paper:** none (engineering/product blog, not an academic paper).
- **Code:** https://github.com/NVIDIA-NeMo/nemo-compass ; https://github.com/NVIDIA-NeMo/eval-author
- **Relationship to existing course material:** Checked via `lookup "agent harness co-optimization"` style terms (no new hit beyond the existing file) and by reading `research/deferred/agent-harness-policy-co-optimization.md` directly — adjacent new evidence, mechanistically distinct from Self-HPO/HarnessPro/HERD/ParetoSaddler (no RL-trained policy here), not a duplicate.
- **Potential course lesson:** If the cluster is eventually resolved (module 19 agents area), this is a good "what this looks like without RL, as a shipped product pattern" counterpoint to the RL-co-optimization papers — with the marketing-authorship and small-sample caveats stated explicitly if used.
- **Confidence:** medium that the tooling itself is real and usable (open repos, one named production case); low-medium on the reported numbers generalizing, given the marketing byline and the tiny/fictional internal example.
- **Recommendation:** add as new, mechanistically distinct evidence to `agent-harness-policy-co-optimization.md`; flag the marketing-authorship caveat explicitly so the weekly review weighs it differently from the academic RL papers already in that cluster.

### C-20261105-06 · PACT: certified commit rule for parallel decoding in diffusion language models

- **Class:** B
- **Date discovered:** 2026-11-05
- **Date published:** 2026-11-03 (v1)
- **Source:** https://arxiv.org/abs/2611.04152 (verified resolves; cs.CL); https://github.com/LiZaiyuan0619/PACT (verified resolves, public, MIT license, code present)
- **Organization/researchers:** Yanchao Li, Yusu Zhang, Jiaqing Xie, and 5 others; affiliation not stated on the abstract page.
- **Category:** other (masked-diffusion-LM decoding; falls under the 2026-11-03 author scope decision on non-autoregressive LMs)
- **What changed:** In parallel decoding of masked diffusion LMs, existing "commit rules" (which tokens to finalize per step) don't account for which tokens were actually sampled. PACT samples first, then certifies which sampled tokens are safe to commit (a certificate that holds at any temperature), and spaces out committed positions within a step — training-free.
- **Technical summary:** Authors' own numbers: across 4 models (LLaDA-8B, LLaDA-MoE, LLaDA2.1-mini, Dream-7B) and 5 benchmarks (HumanEval, MBPP, GSM8K, MATH500, IFEval), PACT matches published commit rules' quality with fewer forward passes, and at equal speed repeats consecutive words far less often than published decoders. Code repo (verified) includes PACT, the baselines it compares against (Fast-dLLM, EB-Sampler, KLASS, etc.), the four models' configs, and grading code for all five benchmarks; MIT licensed.
- **Why it might matter:** `research/SCOPE-DECISIONS.md` (2026-11-03) explicitly says non-autoregressive-LM candidates should be kept as WAIT and have evidence added to their existing deferred files rather than rejected for scope alone or used to open near-duplicate files casually. This doesn't fit the two named files there (`diffusion-lm-rl-credit-assignment.md` is about RL credit assignment, `continuous-flow-language-models.md` is about flow-matching) — it's specifically about decode-time commit-rule certification — so it's new evidence under that scope umbrella without an obvious existing home.
- **Evidence of adoption:** Code released (MIT) with a fairly complete reproduction harness (baselines + 4 models + 5 benchmarks), but the repo shows 0 stars/0 forks at check time; single paper, two days old, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; no stated affiliation.
- **Open-source implementation:** https://github.com/LiZaiyuan0619/PACT (MIT)
- **Paper:** https://arxiv.org/abs/2611.04152
- **Code:** https://github.com/LiZaiyuan0619/PACT
- **Relationship to existing course material:** Checked via `lookup "diffusion language model parallel decoding certification"` (no hit) and by reading the SCOPE-DECISIONS 2026-11-03 non-AR-LM row directly — new evidence within that scope-decided area, not a duplicate of either named deferred file.
- **Potential course lesson:** None under the current scope decision; would only become relevant if/when the author revisits the non-AR-LM foundation question.
- **Confidence:** low-medium on technique soundness (real code, 4-model/5-benchmark sweep); the course-relevance question is separately gated by the scope decision regardless of technical merit.
- **Recommendation:** per the 2026-11-03 scope decision, do not treat as ADD-eligible; stage as B so the weekly review can decide whether to open a new "diffusion-LM decoding" deferred file or fold it into one of the two existing non-AR-LM files as general-area evidence, consistent with "do not REJECT for scope alone."


### C-20261106-01 · DeepSeek Sparse Attention's indexer mechanism is now independently adopted by a second major open-weight lab (GLM-5) — possible trigger for the 2026-10-28 sparse-attention scope decision

- **Class:** A
- **Date discovered:** 2026-11-06
- **Date published:** MLI paper: 2026-11-02 (v1). GLM-5's DSA adoption: GLM-5 itself shipped circa February 2026 (arXiv:2602.15763, "GLM-5: from Vibe Coding to Agentic Engineering") — this adoption fact predates today and is recorded here because it was not found in any existing `research/` file, not because it is a new event.
- **Source:** https://arxiv.org/abs/2611.03768 (verified resolves; MLI paper, cs.LG/cs.AI); https://huggingface.co/zai-org/GLM-5 (verified resolves — model card states GLM-5 "integrates DeepSeek Sparse Attention (DSA)"); https://arxiv.org/abs/2602.15763 (verified resolves — abstract states GLM-5 "adopts DSA to significantly reduce training and inference costs")
- **Organization/researchers:** MLI paper: Serdarcan Dilbaz, Arjun Krishna, Kiran Kamble, Daniel M. Bikel (affiliation not stated on the abstract page). DSA origin: DeepSeek (DeepSeek-V3.2, Jan 2026). Adopter: Zhipu AI / Z.ai (GLM-5).
- **Category:** attention (sparse attention — directly the subject of `research/SCOPE-DECISIONS.md`'s 2026-10-28 "No, for now" row)
- **What changed:** Two things, read together. (1) New paper: in DeepSeek Sparse Attention (DSA), a per-layer "lightning indexer" scans the whole KV cache before each layer's attention, dominating decode time at long context; the authors observe adjacent layers select nearly the same tokens, so a layer can reuse the previous layer's indexer selection and run its own indexer concurrently with that layer's attention instead of sequentially before it (Multi-layer Indexer, MLI) — no retraining, runs in SGLang's production decode path, with an optional fused kernel for DeepSeek-V3.2. (2) Independently confirmed fact the paper's own cross-model benchmarking led to checking: DSA (lightning indexer + top-k selection) was not a DeepSeek-only mechanism — Zhipu AI integrated the same mechanism into GLM-5 (744B/40B-active MoE, MIT-licensed, open-weight) rather than designing its own sparse attention, per GLM-5's own model card and technical report.
- **Technical summary:** MLI paper's own numbers: combined with IndexCache (a related cross-layer reuse method), MLI decodes faster than IndexCache alone at equal retention on both DeepSeek-V3.2 and GLM-5, across 10K–195K token contexts, with LongBench/RULER/LongBench v2 accuracy within a few points of IndexCache. Code released (https://github.com/ArjunKrishnak/dsa-multi-layer-indexer, Apache-2.0, verified public with a real SGLang v0.5.16 patch, CUDA kernel, and eval scripts — 1 star, 2 commits, so not independently adopted itself yet). Accepted to the LCFM workshop at NeurIPS 2026. GLM-5's own numbers for DSA itself are the model's own technical report, not independently re-benchmarked by me beyond what the MLI paper's cross-model comparison implies.
- **Why it might matter:** `research/SCOPE-DECISIONS.md` (2026-10-28) rejected a sparse-attention foundation and told the weekly review to REJECT candidates like Prism/DoSA/PISA/FFD at their next review, "reopen only if ... one mechanism becomes standard in major open models with independent validation." DeepSeek-style sparse attention (lightning indexer + top-k) is, on today's evidence, exactly that: shipped in DeepSeek-V3.2 (Jan 2026) and independently adopted — not merely imitated conceptually, but integrated as the same DSA mechanism — by Zhipu AI for GLM-5, a second major open-weight lab, since roughly February 2026. I could not find this specific adoption fact already recorded anywhere in `research/` (lookup for "DeepSeek sparse attention indexer" hits only `sparse-attention-indexer-dosa.md` and `sparse-attention-cross-layer-index-sharing.md`, neither of which mentions GLM-5), so either it was already weighed and set aside when the 2026-10-28 decision was made, or it wasn't — the weekly review should check which, since the decision's own stated reopening condition appears to have been satisfied since February, independent of today's MLI paper.
- **Evidence of adoption:** Real, verifiable, cross-lab: DeepSeek (origin, DeepSeek-V3.2) and Zhipu AI (GLM-5, open-weight MIT license, independently confirmed via HF model card and the GLM-5 arXiv technical report's own abstract, not just secondary blogs). The MLI paper itself (today's new contribution) has code but only 1 star / 2 commits — no adoption of MLI specifically, only of the underlying DSA mechanism it optimizes.
- **Major organizations using it:** DeepSeek (origin), Zhipu AI / Z.ai (independent adopter, GLM-5). No evidence of a third lab; this is two, not "several."
- **Open-source implementation:** DSA itself: open in DeepSeek-V3.2 and GLM-5 (both open-weight). MLI specifically: https://github.com/ArjunKrishnak/dsa-multi-layer-indexer (Apache-2.0).
- **Paper:** https://arxiv.org/abs/2611.03768 (MLI); https://arxiv.org/abs/2602.15763 (GLM-5 technical report, for the adoption fact)
- **Code:** https://github.com/ArjunKrishnak/dsa-multi-layer-indexer
- **Relationship to existing course material:** Checked via `lookup "DeepSeek sparse attention indexer"` → hits `research/deferred/sparse-attention-indexer-dosa.md` (WAIT) and `research/deferred/sparse-attention-cross-layer-index-sharing.md` (WAIT), both of which treat "native sparse attention" as an unmet prerequisite and neither of which names GLM-5 or any second-lab adoption. This is new evidence for both files, and squarely for the `SCOPE-DECISIONS.md` 2026-10-28 row itself — not a duplicate, and not resolvable by the weekly review alone since that row says reopening is a question for the author.
- **Potential course lesson:** Unchanged by this item alone — still gated on the author's scope decision. If reopened: a short foundation lesson on indexer-based sparse attention (DeepSeek-style lightning indexer + top-k), which would let `sparse-attention-indexer-dosa.md`, `sparse-attention-cross-layer-index-sharing.md`, `block-sparse-attention-pisa.md`, and `sparse-attention-decoding-ffd.md` all be re-judged against it, with MLI as a concrete cross-model optimization example.
- **Confidence:** high that DSA is genuinely adopted by GLM-5 independent of DeepSeek (primary sources: GLM-5's own HF model card and its own technical report's abstract, not just aggregator commentary); medium-high that the MLI paper's own cross-model speedup numbers are real (real code, real SGLang integration) but low on MLI-the-method's own adoption (1 star, 2 commits, 4 days old).
- **Recommendation:** flag explicitly for Tal/the weekly review as a possible trigger for revisiting the 2026-10-28 sparse-attention scope decision, separate from and stronger than the MLI paper's own (otherwise ordinary single-paper) contribution — the question is whether "DeepSeek + one independent major-lab adopter" meets the decision's own "standard in major open models" bar, not whether MLI itself should be ADDed.

### C-20261106-02 · D3D: standalone diffusion drafter removes the drafter-verifier dependency for speculative decoding

- **Class:** B
- **Date discovered:** 2026-11-06
- **Date published:** 2026-11-04 (v1)
- **Source:** https://arxiv.org/abs/2611.06425 (verified resolves; cs.AI, cross-listed cs.LG); https://github.com/fudan-generative-vision/D3D (verified resolves, public, MIT license, code present)
- **Organization/researchers:** Baoyou Chen, Ruiqiao Mei, Guancong Lin, Tao Song, Weihao Yuan, Qun Liu, Siyu Zhu (affiliation not stated on the abstract page; the GitHub org name "fudan-generative-vision" suggests, but does not confirm, a Fudan University affiliation).
- **Category:** inference/efficiency (speculative decoding, diffusion-based drafters)
- **What changed:** Target-conditioned diffusion drafters (e.g. the course-tracked XPress, MDraft) predict a block of candidate tokens in parallel but each new block must wait for the target model's updated hidden states, forcing drafting and verification to alternate sequentially. D3D is a standalone diffusion drafter trained with blockwise denoising plus auxiliary autoregressive supervision so it doesn't need target hidden states to draft a good block, which lets it draft the next block during verification of the current one, in a single packed pass (asynchronous speculative decoding). The authors also prune drafter layers by their effect on accepted-prefix length to keep packed drafting cheap (removing 5 of 36 layers keeps over 99% of mean acceptance length).
- **Technical summary:** Authors' own numbers across seven benchmarks: 5.00x and 5.71x average decoding speedup with Qwen3-8B and Qwen3-32B targets respectively (vs. 4.03x/3.67x for the DFlash baseline), up to 7.48x on GSM8K with Qwen3-32B. Code repo (verified) contains inference/, training/, benchmarks/, scripts/, and a third_party/sglang integration, MIT licensed, 9 stars/0 forks at check time. No stated venue.
- **Why it might matter:** Direct new evidence for the course's actively-tracked `research/deferred/speculative-decoding.md` cluster (WAIT, next review 2026-11-15) and mechanistically distinct from the diffusion-drafter entries already staged this week (XPress: causal refiner merged into vLLM core; MDraft: multi-draft OPD) — D3D's specific contribution is removing the drafter's dependence on target hidden states to enable true pipelining, not refining candidates or multi-draft sampling.
- **Evidence of adoption:** Code released (MIT, real SGLang integration) but only 9 stars; single paper, two days old, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; no stated institutional affiliation.
- **Open-source implementation:** https://github.com/fudan-generative-vision/D3D (MIT)
- **Paper:** https://arxiv.org/abs/2611.06425
- **Code:** https://github.com/fudan-generative-vision/D3D
- **Relationship to existing course material:** Checked via `lookup "diffusion drafter speculative decoding"` (hits `research/staging.md`'s XPress entry only) and by reading `research/deferred/speculative-decoding.md` directly — new, mechanistically distinct entry for that cluster, not a duplicate of XPress/MDraft/FLy/ProVe.
- **Potential course lesson:** If reproduced, extends 08.6's speculative-decoding foundation with an async/pipelined-drafting variant, alongside the other diffusion-drafter entries already staged.
- **Confidence:** low-medium — real code and SGLang integration, concrete multi-model numbers, but no venue, no reproduction, four days old.
- **Recommendation:** add as new evidence to `speculative-decoding.md` ahead of its 2026-11-15 review, alongside this week's XPress/MDraft/FLy/ProVe/SpecOPD entries.

### C-20261106-03 · EvoOPD: evolving which privileged-information feedback to use as on-policy distillation progresses

- **Class:** B
- **Date discovered:** 2026-11-06
- **Date published:** 2026-11-04 (v1)
- **Source:** https://arxiv.org/abs/2611.06542 (verified resolves; cs.LG)
- **Organization/researchers:** Ximo Zhu, Ruiqi Liu, Yi Han, Zhengan Wu, Gengsheng Li, ZiAn Wang, Zhu Yuantian, Jin Jiang, Wenhao Wang (affiliation not stated on the abstract page).
- **Category:** RL/post-training (on-policy distillation from black-box teachers)
- **What changed:** Distilling from proprietary (black-box) teachers can't use teacher logits, so prior work treats the teacher's textual feedback as "privileged information" (PI) converted into dense supervision on the student's own rollouts. EvoOPD's contribution: which kind of feedback helps most changes over the course of training, so it maintains a PI-strategy bank that evolves with the student's state, using each feedback type's measured "repair utility" on the current student to pick focus/information-type/granularity, rather than a fixed PI strategy throughout.
- **Technical summary:** Authors' own numbers: tested on math reasoning and code generation with Qwen3-1.7B/4B/8B students, best in all six model-task settings, beating the strongest baseline by an average 2.14 points (math) and 1.30 points (code). No code repository link found on the abstract page; no stated venue.
- **Why it might matter:** A fifteenth mechanistically-distinct single-paper variant for the heavily-tracked `research/deferred/on-policy-distillation-variants.md` cluster (11 entries already in the file, plus TESD/ATI-OPD/SPTA staged this week awaiting the 2026-11-29 merge) — distinct in treating *which feedback to give* as an online-adapted choice rather than a fixed teacher-adaptation or weakening mechanism.
- **Evidence of adoption:** None — single paper, two days old, no code, no stated venue, no third-party reproduction.
- **Major organizations using it:** None identified; no stated affiliation.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2611.06542
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "on-policy distillation privileged information"` (no hit beyond an unrelated course-text match) and by reading `research/deferred/on-policy-distillation-variants.md` directly — new, distinct mechanism, not a restatement of WISD/PR-OPD/DIAL-OPD/etc.
- **Potential course lesson:** If reproduced, folds into the cluster's eventual resolution (17.4 extension).
- **Confidence:** low — single paper, no code, no stated venue or affiliation, no reproduction.
- **Recommendation:** add as the (now) fifteenth-plus candidate to `on-policy-distillation-variants.md`; repeating prior days' process note more strongly: this cluster has accumulated roughly 15 single-paper, unreplicated variants across seven+ weeks with no ADD/WAIT/REJECT resolution — the 2026-11-29 review should prioritize resolving or synthesizing the whole family rather than continuing to accumulate entries.

### C-20261106-04 · SIMPLE: a closed-form, four-parameter energy-cost model for LLM inference

- **Class:** B
- **Date discovered:** 2026-11-06
- **Date published:** 2026-11-02 (v1)
- **Source:** https://arxiv.org/abs/2611.03330 (verified resolves; cs.AI, cross-listed cs.LG/cs.PF)
- **Organization/researchers:** Hiari Pizzini Cavagna, Andrea Proia, Daniele Cesarini, Zeynep Kiziltan, Andrea Bartolini (affiliation not stated on the abstract page). Submitted to *Future Generation Computer Systems*.
- **Category:** efficiency / serving (energy-cost modeling for inference)
- **What changed:** Existing per-token energy-cost models for LLM inference are curve fits that need recalibration per model. The authors derive closed-form compute/memory-access complexity expressions from architectural hyperparameters (covering MHA and GQA) and build SIMPLE, a four-parameter model that separates prefill from decode and splits decode into KV-cache cost vs. next-token compute, with no per-model calibration needed.
- **Technical summary:** Authors' own numbers: on traces from 13 dense LLMs across 6 families on NVIDIA H100s, R²=0.969 and 11.97% MAPE, reported as 3.1–3.5x lower error than the prior state of the art. Reported findings: prefill energy grows quadratically with hidden dimension (hurts larger models more on long inputs); more aggressive GQA grouping cuts per-token energy up to 3x vs. MHA; larger reasoning-token budgets push output length past the analytic energy optimum, and "intelligence per joule" can drop up to 3.14x across a 4x budget range; agentic serving raises energy per request 13.0x vs. non-agentic; prefix caching removes ~93% of prefill cost but leaves the KV-cache term unchanged, so a cached agentic request still costs 4.8x more. No code repository link found on the abstract page.
- **Why it might matter:** A quantitative cost-modeling complement to the course's accepted `inference-serving-foundations.md` lesson and to the RL-rollout-cost framing in `rl-training-inference-mismatch.md` — the reasoning-budget and agentic-serving energy findings are directly relevant to how the course already discusses why rollout generation dominates RL cost.
- **Evidence of adoption:** None — single paper, under journal review, no code, no third-party reproduction, academic authorship only.
- **Major organizations using it:** None identified; no stated major-lab affiliation.
- **Open-source implementation:** none found on the abstract page.
- **Paper:** https://arxiv.org/abs/2611.03330
- **Code:** none found.
- **Relationship to existing course material:** Checked via `lookup "energy cost LLM inference"` (no hit) — new topic, not a duplicate of the existing serving/cost material, which doesn't currently quantify energy.
- **Potential course lesson:** Possible short addition to `inference-serving-foundations.md` or the RL-cost discussion, as a worked "where does inference energy actually go, and how does reasoning-budget/agentic serving change it" example — low priority given no code and journal-review-pending status.
- **Confidence:** medium on methodology (closed-form derivation, 13-model/6-family validation, concrete R²/MAPE); low on course-relevance priority given it's cost accounting rather than a training/inference technique.
- **Recommendation:** monitor; no existing deferred file is an exact fit, so weekly review should decide whether to open a small new deferred topic or fold the reasoning-budget/agentic-serving findings into the existing accepted lessons' caveats once/if code or a published version appears.

### C-20261107-01 · MOPD lands in TRL: `AsyncDistillationTrainer` implements multi-teacher on-policy distillation, with six labs' releases cited as prior adopters

- **Class:** A
- **Date discovered:** 2026-11-07
- **Date published:** 2026-11-03
- **Source:** https://huggingface.co/blog/sergiopaniego/mopd-trl (verified resolves)
- **Organization/researchers:** Sergio Paniego (Hugging Face). Published on the Hugging Face blog platform; TRL is a Hugging Face library.
- **Category:** RL/post-training (distillation)
- **What changed:** The course already teaches on-policy distillation (logit distillation, forward vs. reverse KL) in accepted lesson 17.4, which notes "a later review may extend 17.4 with ... multi-teacher distillation." TRL now ships a concrete, runnable implementation of exactly that: `AsyncDistillationTrainer` / `AsyncDistillationConfig` (`trl.experimental.async_distillation`) for Multi-Teacher On-Policy Distillation (MOPD) — a general SFT student samples its own rollouts, each prompt is routed to a domain-expert teacher (math/code/instruction-following) served over vLLM, and the student is trained toward that teacher's per-token distribution (reverse KL, `beta=1.0`, `teacher_top_k=1`) via a `teacher_id`-routed dataset column.
- **Technical summary:** Full worked example in the post: student `BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT`, data `BytedTsinghua-SIA/Open-MOPD-Data` (40/40/20 math/code/IF split), 5 processes on 4×H200, launched via `launcher.py` or `hf jobs uv run --flavor h200x4`. Per-teacher monitoring metrics (`teacher_jsd/<id>`, `teacher_token_frac/<id>`) show instruction-following prompts received only 1–3% of trained tokens — a real, reported imbalance, not smoothed over. A 60-step partial run (~1% of a full training budget) moved AIME24 17.5→24.2, AIME25 19.2→22.5, IFEval 67.5→73.0; author explicitly flags this as a small run. The feature is in TRL's `experimental` namespace (not yet the stable API); a synchronous `DistillationTrainer` is pending in TRL PR #7215. Runnable code (`async_distillation_mopd.py`, `launcher.py`) is linked in TRL's own GitHub examples, plus an Open-MOPD model/data collection and a Trackio dashboard Space.
- **Why it might matter:** This is adoption in a major open-source stack (TRL) of a technique the course has already flagged as a likely future extension point for 17.4, with a runnable example rather than a bare paper. It directly answers "how would I actually run this" for the existing lesson.
- **Evidence of adoption:** TRL implementation itself is primary, verifiable adoption evidence (open-source, code in TRL's repo), though still marked `experimental`. The post's table of six labs (Xiaomi, DeepSeek, Zhipu/GLM, NVIDIA, Moonshot, Liquid AI, plus Mistral Large 4) citing MOPD-style fusion in their own releases is **(company claim)** in every case — sourced from each lab's own model card/report, not independently reproduced here, and not re-verified against each primary card in this check.
- **Major organizations using it:** Hugging Face (TRL maintainer); labs *claiming* use per their own model cards: Xiaomi (MiMo-V2-Flash, MiMo-V2.6), DeepSeek (V4), Zhipu AI (GLM-5, variant), NVIDIA (Nemotron 3 Ultra, 10+ teachers), Moonshot AI (Kimi K3), Liquid AI (LFM2.5 family), Mistral AI (Mistral Large 4) — all **(company claim)**, not independently checked here.
- **Open-source implementation:** TRL, `trl.experimental.async_distillation.AsyncDistillationTrainer` (github.com/huggingface/trl).
- **Paper:** none (TRL is implementing a technique named/used across the cited model cards, not one paper); background method reference: https://huggingface.co/papers/2608.27409 (not independently re-verified in this check).
- **Code:** https://huggingface.co/blog/sergiopaniego/mopd-trl (links the example scripts and Open-MOPD collection).
- **Relationship to existing course material:** `lookup "multi-teacher on-policy distillation"` hits `research/accepted/on-policy-distillation.md` (lesson 17.4, which already anticipates this extension) and `research/deferred/multi-teacher-on-policy-distillation.md` (WAIT, next review 2026-11-08) — this is new evidence (open-source implementation) for exactly that deferred topic, due for review tomorrow.
- **Potential course lesson:** Extend 17.4 with a MOPD section (multiple domain teachers, reverse-KL routing by `teacher_id`) using TRL's own `AsyncDistillationTrainer` as the from-scratch-then-library example, with the token-imbalance finding as a worked caveat; keep the six-lab adoption table explicitly labeled (company claim).
- **Confidence:** medium-high that TRL now has a real, runnable MOPD implementation (verified directly: trainer name, config, example scripts, student/data IDs all named and specific); low-medium on the breadth of the "six labs use this" claim since those are the post's own citations of each lab's self-report, not independently cross-checked card by card.
- **Recommendation:** weekly review should treat this as new evidence resolving `multi-teacher-on-policy-distillation.md`'s next review (due 2026-11-08) toward ADD — extend 17.4 rather than open a new lesson, since the base technique is already taught there.

### C-20261107-02 · Muir: low-rank gate sharing + matched-filter readout for Gated DeltaNet-style linear attention

- **Class:** B
- **Date discovered:** 2026-11-07
- **Date published:** 2026-11-05 (v1)
- **Source:** https://arxiv.org/abs/2611.08376 (verified resolves; cs.LG)
- **Organization/researchers:** Tushaar Gangavarapu, Gil Tabak, Clemens JS Schaefer (affiliations not stated on the abstract page).
- **Category:** attention (linear attention / efficient architectures)
- **What changed:** Builds directly on Gated DeltaNet-2 (GDN-2), the family the course already teaches in accepted lesson 12.5. The authors observe state updates are far less sensitive to the erase gate than to writes, so the erase gate can be a low-rank projection shared with other gating, freeing parameters for a wider FFN; separately, slow-decaying state rows accumulate larger norms and dominate unweighted readout, motivating a "matched-filter" readout gate that re-weights state rows before readout. Also describes a parallel-update formulation meant to keep training stable at large chunk sizes.
- **Technical summary:** Self-reported numbers (166M–1.34B params): perplexity 1.8–2.0% lower than GDN-2 and 2.5–3.6% lower than a same-size Transformer; up to +1.9 points on commonsense accuracy; better extractive/multi-hop/multi-needle recall than both baselines at 1.34B; extrapolates from 4K training length to 10M-token streams without perplexity drift; near-100% passkey retrieval to 1M tokens with static readout; matches GDN-2 throughput while reaching its 100B-token validation loss 1.2x faster, and trains 2.7x faster than a Transformer at 32K context. No code repository found on the abstract page.
- **Why it might matter:** A direct architectural refinement of the exact mechanism (Gated DeltaNet) the course already teaches as a hybrid-attention foundation; if it held up it would be a natural "further reading / what's next" pointer from 12.5, not a new lesson.
- **Evidence of adoption:** None — single paper, two days old, no code link found, no third-party reproduction, no stated lab affiliation.
- **Major organizations using it:** None identified.
- **Open-source implementation:** none found.
- **Paper:** https://arxiv.org/abs/2611.08376
- **Code:** none found on the abstract page.
- **Relationship to existing course material:** `lookup "Gated DeltaNet"` hits the accepted `hybrid-linear-attention-gated-deltanet.md`/lesson 12.5 and several deferred Gated-DeltaNet-adjacent files (Kronecker Delta Rule, state quantization, ReplaySSM) — new topic, not a duplicate of any of those (none address gate-sharing or matched-filter readout).
- **Potential course lesson:** No new lesson; at most a "further reading" pointer from 12.5 once/if code appears or it is reproduced — low priority on current single-paper evidence.
- **Confidence:** low — no code, no reproduction, self-reported numbers only, despite a reasonably detailed mechanistic argument.
- **Recommendation:** monitor; open a small deferred-topic file at the weekly review tracking refinements to Gated DeltaNet-style gating/readout (distinct from the existing quantization/caching/Kronecker clusters on the same base architecture).

### C-20261107-03 · SAE Interprets MoE: sparse-autoencoder decomposition shows fine-grained expert specialization below routing-based domain labels

- **Class:** B
- **Date discovered:** 2026-11-07
- **Date published:** 2026-11-05 (v1)
- **Source:** https://arxiv.org/abs/2611.08089 (verified resolves; cs.LG)
- **Organization/researchers:** Anyuan Zhuo, Shaohua Zhang, Yuan Lin, Pinyan Lu (affiliations not stated on the abstract page; code published under the ByteDance-Seed GitHub organization).
- **Category:** interpretability / MoE
- **What changed:** Prior expert-specialization studies count routing-selection frequency on domain-labeled data (coarse, fixed categories). This paper trains a sparse autoencoder on an MoE layer's router-weighted output and shows each SAE latent's pre-activation decomposes exactly across experts, giving a per-latent measure of each expert's actual contribution rather than just how often it was selected.
- **Technical summary:** Applied to layers 4, 30, 46 of Qwen3-30B-A3B. Contribution is much more concentrated than routing-selection statistics suggest: most SAE latents have one or two dominant experts, and overlaps cluster on a few expert pairs. Auto-interpretation (via GPT-5.5-generated labels, per the released data) finds specialization finer than domain labels — e.g. separate experts for Java vs. front-end web code, or geometry vs. arithmetic/polynomials. Findings are qualitative (no numerical benchmark claims). Code + released explanations/labels confirmed at github.com/ByteDance-Seed/SAE-interprets-MoE (Apache-2.0); trained TopK SAEs released separately at huggingface.co/ByteDance-Seed/Qwen3-30B-A3B-MoE-SAE.
- **Why it might matter:** A genuinely new interpretability method for MoE models (exact per-latent expert-contribution decomposition, not a routing proxy), with code and released SAE weights rather than a bare claim — relevant to the course's interpretability material and to any future MoE-routing lesson.
- **Evidence of adoption:** None — single paper, two days old, no third-party reproduction; code/data release from a named lab (ByteDance Seed) is a form of self-adoption evidence but not external use.
- **Major organizations using it:** ByteDance (Seed team, via code/data release); no external users identified.
- **Open-source implementation:** https://github.com/ByteDance-Seed/SAE-interprets-MoE (Apache-2.0); SAE weights at https://huggingface.co/ByteDance-Seed/Qwen3-30B-A3B-MoE-SAE.
- **Paper:** https://arxiv.org/abs/2611.08089
- **Code:** https://github.com/ByteDance-Seed/SAE-interprets-MoE
- **Relationship to existing course material:** `lookup "MoE expert specialization interpretability"` — no matches; new topic, not covered by any existing lesson or registry entry.
- **Potential course lesson:** Possible interpretability-module addition or MoE-lesson extension ("what are the experts actually doing," SAE-based rather than routing-frequency-based) — low priority on single-paper evidence, though the code+weights release lowers the bar for later independent checking.
- **Confidence:** medium on methodology (exact linear decomposition is a clean, verifiable claim given the code); low on course-relevance priority until reproduced or extended beyond one model family.
- **Recommendation:** monitor; open a small deferred-topic file for MoE-interpretability methods (distinct from the existing MoE efficiency clusters — routing entropy, expert pruning, channel selection) at the weekly review.

### C-20261107-04 · Harness adapters: per-harness LoRA adapters trained with TRL's AsyncGRPOTrainer behind a tool-call safety proxy

- **Class:** B
- **Date discovered:** 2026-11-07
- **Date published:** 2026-11-06
- **Source:** https://huggingface.co/blog/harness-hub/harness-adapters (verified resolves)
- **Organization/researchers:** Younes Belkada (ybelkada), Lysandre, Nathan Habib (SaylorTwift) — Hugging Face, published under the `harness-hub` organization; acknowledgments list Aritra Roy Gosthipaty, Shaun Smith, Adithya S Kolavi, Ben Burtenshaw, Amine Dirhoussi.
- **Category:** agents/tools (RL post-training for agentic coding)
- **What changed:** One LoRA adapter per (model, agent harness) pair — pi, opencode, hermes — trained with GRPO via TRL's `AsyncGRPOTrainer` (rank 32, all linear layers) inside the harness unmodified, rather than fine-tuning the base model or modifying the harness. A reverse proxy, `safeagents`, sits between harness and model and rule-checks every tool call (`atif-scan` + custom plugins) before execution; blocked calls get a reward of 0 instead of executing. Synthetic tasks are built from open agent trajectories on the Hub plus SmolDataEnvs.
- **Technical summary:** Main runs: LFM2.5-24B-A2B and LFM3-1.7B; TwIL-8B as a non-agentic-baseline comparison. Author-labeled "smoke training tests" (full runs still in progress) report adapters making fewer tool calls than the base model at equal-or-better accuracy on held-out tasks. A 50-step safeagents-guard ablation found +0.0 ± 4.1 points held-out pass@1 — i.e., statistically indistinguishable, which the post itself does not oversell. Cost framing: evaluating nine checkpoints cost ~$5 of GPU time vs. an estimated ~$90 to merge-and-serve each separately. Task mix is skewed (253/270 "build" tasks, zero "qa" tasks), an explicit limitation in the post. Code: safeagents (github.com/huggingface/safeagents), Repo2RLEnv (github.com/huggingface/Repo2RLEnv), TRL's AsyncGRPOTrainer.
- **Why it might matter:** A concrete, code-backed pattern for per-harness RL adaptation without base-model fine-tuning or harness modification, plus a cheap tool-call safety gate for agentic RL rollouts — adjacent to both the existing agent-harness-co-optimization cluster (different mechanism: adapters, not harness edits) and to agent/tool-use material generally.
- **Evidence of adoption:** Weak/early — the post states earlier (unspecified) "early adapters" got "positive signals" and "community adoption," prompting this expanded post, but gives no adoption metrics; the quantitative ablation here shows no significant effect yet. This is a Hugging Face team post (primary, not third-party), not independent validation.
- **Major organizations using it:** Hugging Face (author); Liquid AI models (LFM2.5/LFM3) used as subjects, not stated as co-adopters of the method itself.
- **Open-source implementation:** https://github.com/huggingface/safeagents ; https://github.com/huggingface/Repo2RLEnv ; TRL `AsyncGRPOTrainer`.
- **Paper:** none (engineering blog post, not a paper).
- **Code:** https://huggingface.co/blog/harness-hub/harness-adapters (links all repos/datasets/Spaces).
- **Relationship to existing course material:** `lookup "harness adapter LoRA"` and `lookup "safeagents guardrail proxy tool call"` — no matches; `research/deferred/agent-harness-policy-co-optimization.md` (WAIT) covers a different mechanism (policy proposes harness *edits*) — related cluster, not a duplicate.
- **Potential course lesson:** No new lesson on current evidence; a possible future agents/tools-module example of per-harness LoRA adapters + cheap rule-based tool-call gating as a design pattern, once full-run results (not smoke tests) are published.
- **Confidence:** low — author-labeled smoke tests, a null-result ablation, skewed task mix, and no independent source; the engineering pattern (adapters + proxy) is clear and verifiable even though the empirical payoff is not yet shown.
- **Recommendation:** monitor; do not merge into the existing harness-co-optimization deferred file (different mechanism) — open a separate small deferred topic if the weekly review wants to track it, re-review once full (non-smoke) run results are published.

### C-20261107-05 · Coca-Cola: latent (non-text) compression of agent tool outputs, with action-preserving training — a fifth mechanism in the agent-context-compaction cluster

- **Class:** B
- **Date discovered:** 2026-11-07
- **Date published:** 2026-11-04 (v1)
- **Source:** https://arxiv.org/abs/2611.06500 (verified resolves; cs.AI)
- **Organization/researchers:** Yuetong Liu, Zhehao Zhou, Yitong Zhang, Haorui Chen, Jia Li (affiliations not stated on the abstract page; code published under the THU-Agent GitHub organization, suggesting a Tsinghua-affiliated group, not independently confirmed).
- **Category:** agents/tools (context/memory management)
- **What changed:** Unlike the existing cluster's text-level compaction/summarization/externalization mechanisms, this paper skips text entirely: a continuous latent-compression model ("Coca-Cola") turns raw tool/observation outputs directly into latent tokens embedded in the agent's input embedding space, trained with "Action-Preserving Training" that aligns the agent's next-action distribution under full vs. compressed observations.
- **Technical summary:** Self-reported, on web-search agents across four QA benchmarks and two model families: 70.01–89.24% reduction in token consumption, 24.45–55.97% reduction in end-to-end runtime, and claimed consistent task-performance improvement (underlying tables not in the abstract page). Code confirmed at github.com/THU-Agent/Coca-cola (training/eval code, baselines: Vanilla, LLMSummary, LLMLingua).
- **Why it might matter:** Adds a sixth(ish) distinct single-paper mechanism to the already-tracked `agent-context-compaction.md` cluster (next review 2026-11-29) — this time latent rather than textual compression, which is a meaningfully different approach class from the five already on file (graph/importance, RL-trained compaction policy, belief-state tracking, residual memory tokens, stateless rebuild, source-judged dropping).
- **Evidence of adoption:** None — single paper, three days old, no third-party reproduction.
- **Major organizations using it:** None identified; code release only (THU-Agent org).
- **Open-source implementation:** https://github.com/THU-Agent/Coca-cola
- **Paper:** https://arxiv.org/abs/2611.06500
- **Code:** https://github.com/THU-Agent/Coca-cola
- **Relationship to existing course material:** `lookup "context compaction"` hits `research/deferred/agent-context-compaction.md` directly (WAIT, next review 2026-11-29) — this is additional evidence for that existing cluster (new mechanism, no convergence yet), not a duplicate requiring class D.
- **Potential course lesson:** None yet — same "what would change the decision" bar as the existing file (convergence on one mechanism with reproduction/adoption, or use in a frontier agent product).
- **Confidence:** low (single paper, self-reported numbers, no reproduction) — consistent with every other entry already in this cluster.
- **Recommendation:** fold into `agent-context-compaction.md` as a new candidate/evidence line at the weekly review; do not open a separate topic file (same underlying problem, same evidentiary bar as the existing cluster).

### C-20261108-01 · Harness-S: sibling-harness pool + RL rescue training + adaptive router, a fourth distinct mechanism in the agent-harness co-optimization cluster

- **Class:** B
- **Date discovered:** 2026-11-08
- **Date published:** 2026-11-03 (v1)
- **Source:** https://arxiv.org/abs/2611.04277 (verified resolves; HTTP 200)
- **Organization/researchers:** Kangning Zhang, Wenxiang Jiao, Bowen Fang, Shuai Shao, Jiachen Zhu, Rong Shan, Shijian Wang, Hao Wang, Yuan Lu, Weiwen Liu, Weinan Zhang (affiliations not stated on the abstract page; code published under the `DeepExperience` GitHub organization, not independently identified).
- **Category:** agents/tools (agent-harness optimization)
- **What changed:** The existing deferred cluster (`agent-harness-policy-co-optimization.md`: Self-HPO, HarnessPro, HERD) has the policy propose edits to a single evolving harness. Harness-S instead keeps a fixed pool of specialized "sibling harnesses" (Directional Harness Evolution) and only calls on them when every attempt under the original base harness fails (Complementary Harness Rescue Training): the policy's rollouts under a rescuing sibling harness replace the failed group in RL, so the trained policy still has to work under the unmodified base harness at the end. An Adaptive Harness Router (behavior profiles + cross-LLM voting, refreshed during training) additionally picks the best top-k siblings at inference time.
- **Technical summary:** Authors' own numbers on AppWorld, SWE-Gym, and DBBench: the policy trained with sibling-harness rescue outperforms policy-only RL training when both are evaluated on the original base harness; the router adds further gains at inference (reported as "Rank-1") at the same rollout/executor budget. Code is released (github.com/DeepExperience/Harness-S, Apache-2.0, verified live, 57 stars at capture time) but the README itself states the first release only targets AppWorld (20-harness pool: base + 19 siblings), that "the portable GPU launcher has not yet completed an end-to-end training run," and that results are not a bitwise replay of the paper's reported scores — i.e. the authors' own release flags itself as not yet fully reproduced even by themselves.
- **Why it might matter:** A fourth mechanistically distinct approach to the same problem the course's deferred cluster is already tracking (harness/policy co-optimization for coding agents), with code released (unlike Self-HPO/HarnessPro/HERD, none of which had code at last review) — though the code's own README tempers how finished it is.
- **Evidence of adoption:** None — single paper, five days old, no third-party reproduction, no stated lab affiliation; the code release is the authors' own, and even that is self-described as incomplete (no finished end-to-end run, no bitwise replay).
- **Major organizations using it:** None identified.
- **Open-source implementation:** https://github.com/DeepExperience/Harness-S (Apache-2.0; verified live).
- **Paper:** https://arxiv.org/abs/2611.04277
- **Code:** https://github.com/DeepExperience/Harness-S
- **Relationship to existing course material:** `lookup "agent harness policy co-optimization"` hits `research/deferred/agent-harness-policy-co-optimization.md` directly (WAIT, next review 2026-12-20, candidates so far: C-20261021-03, plus C-20261104-03 ParetoSaddler already staged) — new, mechanistically distinct candidate for the same cluster, not a duplicate.
- **Potential course lesson:** No new lesson; add as a fourth candidate to the existing deferred cluster (module 19/agents area) alongside Self-HPO/HarnessPro/HERD/ParetoSaddler once/if the cluster converges or gets independent reproduction.
- **Confidence:** low — single paper, no third-party reproduction, and the authors' own code release admits its training pipeline hasn't completed an end-to-end run yet.
- **Recommendation:** add as a new candidate to `agent-harness-policy-co-optimization.md` at the weekly review; no change to its 2026-12-20 scheduled review date.

### C-20261108-02 · EvoMemState: relational + reasoning memory state for personal agents (RL-trained, beyond flat fact storage)

- **Class:** B
- **Date discovered:** 2026-11-08
- **Date published:** 2026-11-02 (v1)
- **Source:** https://arxiv.org/abs/2611.03743 (verified resolves; HTTP 200; cs.IR, cross-listed cs.AI)
- **Organization/researchers:** Zhishang Xiang, Yunbo Tang, Zhongqiao Liu, Junkai Chen, Zerui Chen, Qingguo Hu, Yuxuan Zhang, William Grant, Qinggang Zhang, Jinsong Su (affiliations not stated on the abstract page; code published under the `XMUDeepLIT` GitHub organization, consistent with but not confirming a Xiamen University affiliation).
- **Category:** memory/retrieval (agents)
- **What changed:** Most LLM personal-agent memory systems (MemGPT, Mem0, A-MEM — all named as baselines) store memory as isolated facts retrieved by similarity. EvoMemState adds two further pieces of state per retrieval: a relational state (whether a fact is still valid and how it relates to other facts as they change over time, built via ontology-guided extraction) and a reasoning state (a request-specific plan for how to apply the retrieved facts, generated by an RL-trained policy on top of a frozen response model).
- **Technical summary:** Authors' own numbers across MemSyco-Bench, MemTrapBench, LoCoMo-Refined and LoCoMo report consistent gains over MemGPT/Mem0/A-MEM-style baselines in retrieval completeness and in the accuracy of memory-grounded responses (no single headline number given in the abstract; full tables not independently re-extracted here). The reasoning-state policy is trained with GRPO. Code is released and confirmed live (github.com/XMUDeepLIT/EvoMemState, 26 stars at capture time): it includes inference code, GRPO policy training, and a vendored "slime" training backend, and the README states it reproduces the paper's main results table across the four cited benchmarks; training data/checkpoints are not bundled.
- **Why it might matter:** A structurally different mechanism from the agent-memory baselines it compares against (adds explicit relational/temporal validity tracking plus an RL-trained, request-specific reasoning step, rather than flat fact storage + similarity retrieval) — relevant to the course's agents/memory material, with a working code release rather than a bare claim.
- **Evidence of adoption:** None — single paper, six days old, no third-party reproduction; the code release is the authors' own (self-adoption, not external use), though the repo's own README claims it reproduces the paper's table.
- **Major organizations using it:** None identified; no stated institutional affiliation on the paper itself.
- **Open-source implementation:** https://github.com/XMUDeepLIT/EvoMemState (verified live).
- **Paper:** https://arxiv.org/abs/2611.03743
- **Code:** https://github.com/XMUDeepLIT/EvoMemState
- **Relationship to existing course material:** `lookup "EvoMemState"` and `lookup "agent personal memory relational state"` — no matches; not covered by any accepted/deferred/rejected topic or lesson. (Distinct from the rejected `agent-memory-benchmark-feb30.md`, which was about a benchmark's own validity, not a memory mechanism.) New topic.
- **Potential course lesson:** No lesson yet; a possible future agents/memory-module pointer ("beyond flat fact memory: relational validity + RL-trained reasoning state") if a second group reproduces it or it sees real use beyond the authors' own repo — low priority on single-paper evidence.
- **Confidence:** low-medium — single paper and self-reported numbers, but a working, described code release with a reasonably specific, verifiable mechanism (ontology extraction + GRPO-trained reasoning state).
- **Recommendation:** monitor; open a small deferred-topic file at the weekly review for agent personal-memory mechanisms (distinct from the existing context-compaction cluster, which is about trimming an agent's own working context, not personal long-term memory) if the reviewer judges it worth tracking as its own cluster.

