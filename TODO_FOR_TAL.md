# Status — LLM Research Engineer course

**COMPLETE.** 21 modules · 67 lessons · 30 verified papers · one coherent `llmre` codebase · 159 passing tests · 3 cheat sheets + glossary + 96-card Anki deck.

**Repo:** https://github.com/tal-giladi/llm-research-engineer-course
**Live:** https://tal-giladi.github.io/llm-research-engineer-course/ (Pages enabled, HTTP 200)

Verify locally: `cd code && py -m pip install -e . && py -m pytest -q` (159 pass).
Run the whole pipeline tiny/CPU: `py code/scripts/capstone.py` (~20s; pretrain loss 5.9 → 0.07).

## Build order (commit + push after each — partial work is always safe)
- [x] Scaffold: docsify site, `code/` package skeleton, curriculum map, dependency graph, readiness matrix.
- [x] Paper curriculum: all 30 URLs web-verified, ordered reading guide with per-paper template.
- [x] Module 0 — PyTorch foundations
- [x] Module 1 — Probability & LM (+ evaluation/metrics)
- [x] Module 2 — Gradients & backprop (+ model/micrograd)
- [x] Module 3 — Optimization (+ optim/)
- [x] Module 4 — Tokenization & data (+ tokenizer/, data/)
- [x] Module 5 — Transformer (+ attention/, model/)
- [x] Module 6 — GPT-2 (+ model/gpt.py, 124.4M verified)
- [x] Module 7 — Pretraining infra (+ training/)
- [x] Module 8 — GPU performance (+ attention/flash, bench)
- [x] Module 9 — Distributed training (+ distributed/)
- [x] Module 10 — Scaling laws (+ evaluation/scaling, experiments/)
- [x] Module 11 — Data engineering (+ data/ quality,minhash,contamination)
- [x] Module 12 — Modern architectures (+ rope,rmsnorm,swiglu,gqa,moe)
- [x] Module 13 — Evaluation (+ evaluation/harness,calibration)
- [x] Module 14 — SFT (+ sft/ chat,masking,lora)
- [x] Module 15 — RLHF/DPO (+ preference/, rl/ppo)
- [x] Module 16 — Reasoning RL (+ rl/grpo, reasoning/rlvr,verifiers)
- [x] Module 17 — Reasoning models (+ reasoning/cot,r1_pipeline)
- [x] Module 18 — Tool use (+ tools/)
- [x] Module 19 — Agents & research eng (+ agents/)
- [x] Module 20 — Capstone (+ scripts/capstone.py, runs end-to-end on CPU)
- [x] Reference: cheat sheets, glossary, Anki CSV exports
- [x] GitHub Pages enabled — live at https://tal-giladi.github.io/llm-research-engineer-course/

## To publish the site
GitHub → repo Settings → Pages → Source: "Deploy from a branch", branch `main`, folder `/ (root)`.
`.nojekyll` is already present so docsify serves correctly.

## Notes
- All paper links verified to resolve (Sep 2026). GPT-2 → the OpenAI PDF, not an arXiv lookalike.
- `code/` is CPU-first; GPU-only tasks are marked with a hardware track in the lesson.
