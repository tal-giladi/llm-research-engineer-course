# Status — LLM Research Engineer course

**Repo:** https://github.com/tal-giladi/llm-research-engineer-course
**Live (after Pages is enabled):** https://tal-giladi.github.io/llm-research-engineer-course/

## Build order (commit + push after each — partial work is always safe)
- [x] Scaffold: docsify site, `code/` package skeleton, curriculum map, dependency graph, readiness matrix.
- [x] Paper curriculum: all 30 URLs web-verified, ordered reading guide with per-paper template.
- [x] Module 0 — PyTorch foundations
- [x] Module 1 — Probability & LM (+ evaluation/metrics)
- [x] Module 2 — Gradients & backprop (+ model/micrograd)
- [x] Module 3 — Optimization (+ optim/)
- [ ] Module 4 — Tokenization & data (+ tokenizer/, data/)
- [ ] Module 5 — Transformer (+ attention/, model/)
- [ ] Module 6 — GPT-2 (+ model/)
- [ ] Module 7 — Pretraining infra (+ training/)
- [ ] Module 8 — GPU performance (+ attention/ triton)
- [ ] Module 9 — Distributed training (+ distributed/)
- [ ] Module 10 — Scaling laws (+ experiments/)
- [ ] Module 11 — Data engineering (+ data/)
- [ ] Module 12 — Modern architectures (+ model/, attention/)
- [ ] Module 13 — Evaluation (+ evaluation/)
- [ ] Module 14 — SFT (+ sft/)
- [ ] Module 15 — RLHF/DPO (+ preference/, rl/)
- [ ] Module 16 — Reasoning RL (+ rl/, reasoning/)
- [ ] Module 17 — Reasoning models (+ reasoning/)
- [ ] Module 18 — Tool use (+ tools/)
- [ ] Module 19 — Agents & research eng (+ agents/)
- [ ] Module 20 — Capstone
- [ ] Reference: cheat sheets, glossary, Anki CSV exports
- [ ] Enable GitHub Pages (Settings → Pages → deploy from `main` / root)

## To publish the site
GitHub → repo Settings → Pages → Source: "Deploy from a branch", branch `main`, folder `/ (root)`.
`.nojekyll` is already present so docsify serves correctly.

## Notes
- All paper links verified to resolve (Sep 2026). GPT-2 → the OpenAI PDF, not an arXiv lookalike.
- `code/` is CPU-first; GPU-only tasks are marked with a hardware track in the lesson.
