# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261012-01 · Thinkingbox: grading agents by terminal backend state and repeat-reliability (pass^N), not self-report

- **Class:** B
- **Date discovered:** 2026-10-12
- **Date published:** 2026-08-20 (v1); current v4 2026-10-01; publicized via a joint Microsoft/Hugging Face blog post 2026-10-03
- **Source:** https://arxiv.org/abs/2608.19741 (verified resolves; v4 "Thu, 1 Oct 2026"). A joint Microsoft/Hugging Face blog post announced this around 2026-10-03 per secondary coverage, but its exact URL could not be independently located/verified this run — the arXiv paper and its linked GitHub repos (below) are the sources actually verified.
- **Organization/researchers:** Zhuochun Li, Youngmin Ko, Ali Keramati, Tuhin Kundu, Liang-Chun Tsai, Nicola Ferri, Mirco Milletari, Jiaxiang Liu, Susana Palmaz Lopez Pelaez, Yuepeng Wang, Vadim Smolyakov, Xiang Jiang, Kjartan Olafsson, Tommy Guy — Microsoft (several authors) plus academic co-authors; affiliations not individually stated on the abstract page.
- **Category:** evaluation (agents)
- **What changed:** Thinkingbox is a sandbox (isolated MCP-compatible tool sessions) plus Thinkingbox-bench: 507 policy-governed, stateful business workflows across retail, hospitality, auto insurance, neobank IT, and consulting IT/HR. Each attempt is graded by task-specific executable tests over the *terminal backend state* (accept valid outcomes, reject wrong/missing/extra state changes) rather than by the agent's transcript or self-reported completion. The paper also reports pass^N — the same task attempted N times, checking whether the agent succeeds *every* time — as a reliability metric distinct from pass@k's coverage framing (pass@k asks "does at least one of k independent samples succeed"; pass^N asks "does it succeed on all N repeats of the identical task").
- **Technical summary:** Claude Opus 5 drops from 66.50% pass@1 to 47.53% pass^20; Kimi-K3 drops from 57.37% to 17.60%. The authors report many failed runs end "cleanly" after valid-looking state-changing actions, so response-level or tool-call-level signals are poor proxies for actual task completion. Code (sandbox) at github.com/microsoft/thinkingbox, data (benchmark) at github.com/microsoft/thinkingbox-data, both linked from the abstract page.
- **Why it might matter:** A clean, falsifiable complement to the course's existing pass@k material (`code/src/llmre/evaluation/pass_at_k.py`, from the accepted MLPerf-RLVR topic): pass^N measures repeat-reliability of one task rather than coverage across independent attempts, and the "agent claims success, backend disagrees" framing is a sharp, teachable case for grounding agent evaluation in environment side effects rather than self-report.
- **Evidence of adoption:** Integrated as an official OpenEnv environment on Hugging Face (huggingface.co/docs/openenv/environments/thinkingbox) — real infrastructure availability, though not yet adoption by another named evaluation harness or model card.
- **Major organizations using it:** Microsoft (co-author and code owner); publicized jointly with Hugging Face.
- **Open-source implementation:** https://github.com/microsoft/thinkingbox (sandbox), https://github.com/microsoft/thinkingbox-data (benchmark data)
- **Paper:** https://arxiv.org/abs/2608.19741
- **Code:** https://github.com/microsoft/thinkingbox
- **Relationship to existing course material:** New sub-topic — checked via `lookup "agent evaluation terminal state side effects benchmark"`, no match. Related to but distinct from the already-deferred `research/deferred/computer-use-agent-process-evaluation.md` (OSWorld-Pro grades intermediate process steps; Thinkingbox grades terminal backend state plus repeat-reliability) — flag both together at the next weekly review as two angles on "evaluation beyond final self-reported success."
- **Potential course lesson:** Possible extension of the evaluation module: add a pass^N reliability metric alongside the existing pass@k code, and an agent lesson contrasting self-report/transcript grading with environment-state grading.
- **Confidence:** medium-high — code and data released, concrete numbers across multiple named models, and now has HF OpenEnv infrastructure support, but still a single benchmark paper without independent reproduction or adoption by another evaluation suite.
- **Recommendation:** review at the next weekly cycle alongside OSWorld-Pro (both evaluation-methodology candidates); consider whether pass^N is a small, low-risk addition to the existing `pass_at_k.py` module even before a lesson decision (flag only — no code change made here).

### C-20261012-02 · Rethinking Relative Policy Differences Transfer of On-Policy Distillation (new evidence for deferred OPD-variants topic)

- **Class:** B
- **Date discovered:** 2026-10-12
- **Date published:** 2026-10-09
- **Source:** https://arxiv.org/abs/2610.13205 (verified resolves; v1 "Fri, 9 Oct 2026")
- **Organization/researchers:** Rong Wu, Daocheng Fu, Yu Yang, Xuemeng Yang, Siqi Li, Shu Zou, Yao Zhu, Botian Shi, Bo Zhang, Cong Zhang, Pinlong Cai, Jianbiao Mei (affiliations not stated on the abstract page)
- **Category:** RL/post-training (on-policy distillation — base technique taught in lesson 17.4)
- **What changed:** Standard on-policy distillation (OPD) assumes a much stronger teacher. This paper generalizes OPD to transfer the *relative difference* between any expert/reference policy pair (regardless of scale or training stage) by treating their log-probability ratio as an implicit reward inside KL-regularized policy optimization — standard OPD falls out as a special case. Three findings: (1) a stronger expert is not automatically a better transfer source — differences from task-specific training (RL/SFT) transfer well, differences from scale alone transfer weakly or not at all; (2) the expert supplies the capability, the reference subtracts out shared behavior to isolate the useful residual; (3) a useful difference doesn't always transfer to a given target — raw policy distance doesn't predict transferability, but choosing a better reference can rescue a failed transfer.
- **Technical summary:** No code repository linked from the abstract page; the contribution is primarily theoretical/framing plus supporting experiments (not independently reproduced).
- **Why it might matter:** Directly tests an assumption embedded in lesson 17.4's framing of OPD (bigger/stronger teacher ⇒ better source); if the scale-vs-task-specific-training distinction holds up, it is a teachable refinement of *when* OPD works, not just that it works.
- **Evidence of adoption:** None — single paper, no code, no independent reproduction.
- **Major organizations using it:** None identified (academic, no affiliations stated).
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.13205
- **Code:** none found
- **Relationship to existing course material:** New evidence for the already-deferred `research/deferred/on-policy-distillation-variants.md` — a sixth independent sub-mechanism/finding in that cluster (distinct from DCE/SRCL co-evolving teacher, the teacher-scale scaling law, NP-OPD, the "new skills not new knowledge" framing, and DIAL-OPD token selection). Checked via `lookup "on-policy distillation transfer policy difference"`, no match.
- **Potential course lesson:** Not a new lesson; a sixth data point for the deferred topic's next scheduled review (2026-11-29), specifically bearing on the "does teacher scale matter" sub-question.
- **Confidence:** low-medium — single paper, no code, no reproduction, but a conceptually clean generalization with a falsifiable scale-vs-task-specific-training distinction.
- **Recommendation:** add as a sixth data point to `on-policy-distillation-variants.md` at its 2026-11-29 review; not independently actionable before then.

### C-20261012-03 · Stateless Language Agents: externalized state + rebuilt-per-call context for long-horizon research agents (new evidence for deferred agent-context-compaction topic)

- **Class:** B
- **Date discovered:** 2026-10-12
- **Date published:** 2026-10-06
- **Source:** https://arxiv.org/abs/2610.07625 (verified resolves; v1 "Tue, 6 Oct 2026")
- **Organization/researchers:** Qizheng Zhang, Changxiu Ji, Isaac Sun, Yuetai Li, Shubhangi Upasani, Sherry Ruan, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Yoonho Lee, Yuzhen Mao, Genghan Zhang, Rulin Shao, Qiuyang Mang, Andy Dimnaku, Changran Hu, Radha Poovendran, Kunle Olukotun (18 authors; affiliations not stated on the abstract page)
- **Category:** agents/tools (long-horizon context/memory management)
- **What changed:** Argues that running agents longer doesn't automatically help: agents replay growing histories, duplicate work, or stop experimenting while still burning tokens — traced to *where research state lives and who decides what to try next*. Proposes Stateless Language Agents (SLA): "stateful search with stateless agents" — an external harness stores candidate solutions and measured outcomes and rebuilds a fresh, role-specific context for every agent call, rather than one persistent agent accumulating (and later compacting/summarizing) its own history. A stateless Advisor reviews summarized evidence and assigns experiments to parallel stateless Workers.
- **Technical summary:** Across software engineering, kernel optimization, and algorithm design, with budgets up to one billion tokens, SLA reportedly reaches the best final result on every tested task; it matches the strongest kernel baseline's final performance using over 84% fewer tokens, and the Advisor itself uses under 0.6% of total tokens. No code repository linked from the abstract page.
- **Why it might matter:** A fifth, architecturally distinct approach to the same problem already tracked in the deferred cluster below — rather than compacting or summarizing one agent's growing context (ReCAP, AutoCompact, PoS, REMORY), it externalizes all state and rebuilds context fresh per call. Strong empirical scale (up to 1B-token budgets) makes this a sharper data point than most single papers in that cluster.
- **Evidence of adoption:** None — single paper, no code, no independent reproduction.
- **Major organizations using it:** None identified; authors span several institutions but no org is named as adopting the architecture.
- **Open-source implementation:** None found linked from the abstract page.
- **Paper:** https://arxiv.org/abs/2610.07625
- **Code:** none found
- **Relationship to existing course material:** New evidence for the already-deferred `research/deferred/agent-context-compaction.md` — a fifth independent mechanism, and the first that externalizes state entirely rather than compacting it in place. Checked via `lookup "stateless agent long horizon research context"`, no match.
- **Potential course lesson:** Not a new lesson; a fifth data point for that deferred topic's 2026-11-29 review. Worth flagging to the weekly review that the cluster may now span two genuinely different design philosophies (compact-in-place vs. externalize-and-rebuild) rather than four variants of one idea — possibly a reason to consider the *design-space distinction itself* as teachable even before one mechanism is proven standard.
- **Confidence:** medium — no code or reproduction, but unusually large-scale experiments (1B-token budgets) for a single paper in this area.
- **Recommendation:** add as a fifth data point to `agent-context-compaction.md` at its 2026-11-29 review, flagging the compact-vs-externalize distinction for discussion.

### C-20261012-04 · GenCorrect: externally-graded IMO/IOI gold-level results via generate-verify-refine test-time pipeline (Nemotron)

- **Class:** B
- **Date discovered:** 2026-10-12
- **Date published:** blog 2026-10-07 (Hugging Face); underlying papers: IMO 2026 recipe https://arxiv.org/abs/2609.10712, IOI 2026/GenCorrect https://arxiv.org/abs/2609.02849
- **Source:** https://huggingface.co/blog/nvidia/nemotron-ioi-and-imo-2026 (verified loads; publish date 2026-10-07 confirmed on page)
- **Organization/researchers:** NVIDIA (Nemotron/NeMo-Skills team)
- **Category:** reasoning / RL/post-training (test-time generate-verify-refine)
- **What changed:** Two specialized Nemotron pipelines: a 550B-total/55B-active competitive-coding model (SFT on 22K curated problems) and an IMO math pipeline (SFT on 414,890 filtered examples across 15,818 proof problems plus RL on 9,597 problems), both combined at test time with GenCorrect — an iterative generate → evaluate → refine loop, no formal prover or external tools for the math track.
- **Technical summary:** IMO 2026: 30/42, above the official gold threshold of 29, with submissions graded by official IMO graders (external, human verification — not a self-reported or company-only benchmark). IOI 2026: 535.4/600 (unofficial/unsupervised run, gold threshold 361.12, beating the reported top human score of 498.27). On IOI 2025 (where progression is given), GenCorrect alone took a 30B-A3B model from 291 (after SFT+RL) to 468, above that year's 438.3 gold threshold — the single largest jump in the reported progression. Checkpoints released (NVIDIA-Nemotron-Labs-3-Competitive-Coding-550B-A55B-NVFP4; the Nemotron Labs IMO 2026 collection), IMO training data and a 200-problem IMO-Bench released, and code released (NeMo-Skills, including the IMO inference pipeline/prompts/submitted proofs).
- **Why it might matter:** The IMO result is graded by an independent human panel (official IMO graders), not just a company's own benchmark — stronger evidence than typical "company claim" releases — and GenCorrect's test-time generate-verify-refine loop is a concrete, reproducible technique (code + checkpoints + data released) directly extendable to the course's test-time-scaling/reasoning material (module 17), distinct from pure sampling-based best-of-N since it iteratively revises rather than just reselecting among independent samples.
- **Evidence of adoption:** Externally verified for the IMO track specifically (official competition grading); no third-party adoption of GenCorrect itself as a technique yet.
- **Major organizations using it:** NVIDIA (originator).
- **Open-source implementation:** https://github.com/NVIDIA-NeMo/Skills (also the IMO-specific recipe under `recipes/nemotron-imo-tts`)
- **Paper:** https://arxiv.org/abs/2609.10712 (IMO), https://arxiv.org/abs/2609.02849 (IOI/GenCorrect)
- **Code:** https://github.com/NVIDIA-NeMo/Skills
- **Relationship to existing course material:** New sub-topic — checked via `lookup "IMO gold olympiad"` and `lookup "GenCorrect generate verify refine"`, no matches. Related to but distinct from the course's existing best-of-N/self-consistency material (17.1) and process-reward material (17.2): GenCorrect is an iterative refinement loop, not a rerank-after-independent-sampling method.
- **Potential course lesson:** Possible addition to module 17's test-time-scaling material contrasting best-of-N/self-consistency (resample + rerank) with iterative generate-verify-refine (GenCorrect), using the IOI 2025 progression numbers as a worked before/after example.
- **Confidence:** medium-high — code, checkpoints, and (for IMO) externally-graded results, but still a single organization's own technical report for the overall system (company claim for the IOI "beats top human" framing specifically, flagged as such on NVIDIA's own page as an unofficial run).
- **Recommendation:** monitor for the next weekly review as a possible module-17 extension; the IMO external grading and released code/checkpoints make this stronger than a typical single-source company claim, but still single-source for the method's generality beyond these two competitions.
