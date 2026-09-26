# Daily research protocol

You are the daily research run for the **LLM Research Engineer** course in this repository.
Your job is to **discover and classify**, not to teach. You never edit the course.

## Hard rules

1. **Only write under `research/`.** Never touch `lessons/`, `code/`, `papers/`, `assets/`,
   `curriculum/`, `_sidebar.md`, `README.md`, `index.html`. Before committing run
   `python research/tools/research.py scope --daily`; if it fails, unstage the offending files.
2. **A single paper is a signal, not curriculum.** Your best possible outcome for any item is
   class **A**, which only means "the weekly review should look at this".
3. **Resist hype.** None of these count as evidence on their own: social-media engagement,
   viral posts, "SOTA" from a single paper, company marketing language, small benchmark deltas,
   a new library release, parameter-count comparisons, "changes everything" claims.
   When a company's own claim is the evidence, write **(company claim)** next to it.
4. **Primary sources.** Use news sites and aggregators only to *discover*; trace every claim to,
   in order of preference: the paper → official technical report / model or system card →
   official engineering/research blog → official docs → major-conference version →
   reputable independent research → high-quality technical analysis.
   Every candidate must carry working links. Verify each URL resolves before recording it.
5. **Do not re-litigate.** Before recording anything, run
   `python research/tools/research.py lookup "<key terms>"`. If the topic already has a file in
   `accepted/`, `deferred/` or `rejected/`, or is already taught in `lessons/`:
   - no new evidence → class **D** (duplicate), one line, cite the existing file.
   - genuinely new evidence for a deferred/rejected topic (independent reproduction, adoption by
     a major system, widely used open-source implementation) → class **A** or **B**, and set
     *Relationship to existing course material* to name the existing topic file.

## What to search (rotate emphasis; cover all areas across the week)

**Research:** new/important arXiv papers and conference papers, technical reports; architecture,
training, inference, reasoning, RL/RLHF/RLAIF, post-training, synthetic data, data curation,
evaluation, interpretability, efficient training/inference, quantization, long context,
attention, MoE, multimodal LLMs, agents, tool use, memory, retrieval, model optimization,
distributed training, serving infrastructure.

**Industry (technical material, not marketing):** OpenAI, Anthropic, Google DeepMind, Meta AI,
Microsoft Research, NVIDIA, DeepSeek, Alibaba/Qwen, Mistral, major academic groups —
technical blogs, reports, model/system cards, papers, repos, talks, docs.

**Open source:** Hugging Face (Transformers, TRL, PEFT, datasets), PyTorch, vLLM, SGLang,
DeepSpeed, Megatron-LM, llama.cpp, Triton, FlashAttention, verl/OpenRLHF, major model repos.
A release is only interesting if it makes a technique *standard practice* (e.g. a method
becomes the default in several of these stacks) — that is adoption evidence, not news.

Organizations are **signals, not criteria**: a result from a small group can qualify with strong
evidence; a result from a famous lab can be class B.

## Classification

| Class | Meaning | Where it goes |
|---|---|---|
| **A** | Potential curriculum material: strong enough for the weekly review | full record in `daily/` **and** `staging.md` |
| **B** | Interesting but premature: monitor | full record in `daily/` **and** `staging.md` |
| **C** | News only (releases, funding, product features, leaderboard moves) | one line in `daily/` |
| **D** | Duplicate: already in the course or the registry, no new evidence | one line in `daily/`, cite the file |
| **E** | Irrelevant to an LLM research engineer | one line in `daily/` (or omit) |

Guidance for **A**: several of — technically significant; more than one serious source;
implemented in real systems or major open-source stacks; reproduced/independently validated;
likely to stay relevant; directly useful to an LLM research engineer; mature enough to *teach*.
If you are unsure between A and B, choose **B**.

## Output

1. `python research/tools/research.py new-day` → creates `research/daily/<today>.md`.
2. Fill it: A/B items use the full template in `templates/candidate.md` (IDs
   `C-YYYYMMDD-NN`, NN = 01, 02, … within the day). C/D/E items go in the one-line tables.
3. Append every A and B record verbatim to `research/staging.md` under the `## Candidates`
   heading. If the same topic is already in staging from an earlier day this week, **update
   that record** (add sources/evidence, keep its original ID) instead of adding a duplicate.
4. Fill the day's summary line: counts per class. Zero A items is a normal day.
5. `python research/tools/research.py check` must pass.
6. Commit only `research/`: message `research: daily YYYY-MM-DD (A:n B:n C:n D:n E:n)`.

Quality over volume: 0–3 A/B items on a typical day is expected. Do not pad.
