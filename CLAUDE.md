# Build guide — LLM Research Engineer course

This repo is a **docsify static course site** plus a **runnable PyTorch codebase** (`code/`).
It teaches an experienced software engineer to work as an ML Research / LLM Training Engineer,
using Stanford CS336 as the backbone and extending into frontier-model research. Source brief:
`brief/BRIEF.md`.

## Layout
- `index.html` — docsify config (KaTeX auto-render, search, copy-code, count). Do not add a build step.
- `README.md` — docsify home page.
- `_sidebar.md` — full navigation. Every new lesson/paper MUST be linked here.
- `curriculum/course-outline.md` — module map + explicit dependency graph + readiness matrix.
- `lessons/module-XX/lesson-YY.md` — lesson content. Modules 00–20.
- `papers/` — one reading guide per paper (`NN-slug.md`) + `papers/index.md`.
- `assets/` — cheat sheets, glossary, Anki CSV exports.
- `code/` — the coherent repo built up across the course:
  `src/llmre/{tokenizer,model,attention,optim,training,distributed,data,evaluation,sft,preference,rl,reasoning,tools,agents}`,
  plus `tests/ experiments/ configs/ scripts/ notebooks/ reports/`. Package name: **llmre**.

## Lesson template (every lesson)
Open with a `<div class="prereq">` block containing **Prerequisites**, **You will learn**,
**Why this matters for ML**. Then the six-pass structure for each concept:
1. Intuition (plain English) 2. Mathematics (every symbol named) 3. Numerical example worked
by hand with tiny numbers 4. Tensor shapes/dtype/device 5. From-scratch PyTorch implementation
6. What PyTorch does under the hood / cost (compute + memory).
Include, where the brief asks: unit test, exercise (with hint → stronger hint → collapsed
solution, never spoil), research connection + linked paper, "what to read / what to skip",
"code this after reading", common mistakes, a debugging exercise, and a **Check yourself**
section of `<details>` Q&A. Close with **Next** linking the following lesson.
Add a `<div class="hw">` block stating hardware track (min/recommended HW, expected runtime,
GPU memory, GPU-hours, whether CPU-only works) for any lesson with a real compute task.

## Rules
- Never write "PyTorch does this for us." Explain what it does. Implement the mechanism from
  scratch BEFORE introducing the production framework (HF Transformers/PEFT/TRL/FSDP/DeepSpeed/
  Megatron/Triton/FlashAttention/vLLM).
- Every equation gets a concrete numerical example. Every tensor gets shape/dtype/device.
- Verify every paper URL before embedding it. GPT-2 → https://cdn.openai.com/better-language-models/language-models.pdf
- Distinguish PUBLICLY DOCUMENTED vs REASONABLE INDUSTRY PRACTICE vs INFERENCE/SPECULATION.
- Math in `$...$` / `$$...$$` (KaTeX). Never leave a bare `$` inside a code fence — it breaks rendering.
- Keep pages in small sequential chunks; err toward more technical explanation, not less.
- `code/` must stay one coherent codebase — each stage extends the previous. Tests live in `code/tests/`.

## Local preview
`python -m http.server 8080` from repo root, open http://localhost:8080.
Run code: `cd code && pip install -e . && pytest`.

## Status / handoff
Progress and any gaps are tracked in `TODO_FOR_TAL.md`. Commit + push after each module so
partial work is always preserved.

## Modernization (research/)
The course is kept current by a daily-research → weekly-review pipeline in `research/`
(read `research/README.md`). New material is only added through the weekly review
(`research/PROTOCOL-weekly.md`): additions extend existing lessons and code, never replace
them, and every change leaves an audit trail (candidate → weekly report → topic file →
CHANGELOG). Helper: `python research/tools/research.py check|index|lookup|new-day|scope`.
