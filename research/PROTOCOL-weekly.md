# Weekly curriculum review protocol

You are the weekly curriculum review for the **LLM Research Engineer** course in this repository.
You are the **only** process allowed to change the course, and you do so at most once per week.

**Optimize for long-term curriculum quality, not number of updates.** "No curriculum changes
this week" is a fully successful outcome and should be the most common one. Adding weak
material is a failure; missing a genuinely important development for a week or two is not.

The test for every decision: *"Would learning this make the student a better LLM research
engineer?"* — not "is it new", "is it popular", or "did a famous lab mention it".

**Never touch Tal's personal progress files:** `course.py`, `progress/`, `PROGRESS.md`.

## Step 0 — Read the course before judging anything

Read `CLAUDE.md` (lesson template, rules), `_sidebar.md`, `curriculum/course-outline.md`
(module map + dependency graph), `research/REGISTRY.md`, and every lesson the candidates touch.
You must know the existing structure, terminology, code layout (`code/src/llmre/...`),
prerequisites and teaching style before deciding where anything fits.

## Step 1 — Gather the week's input

- Every candidate in `research/staging.md`.
- Every file in `research/deferred/` whose **Next review** date is on or before today, or whose
  topic appears again in this week's `daily/` files.
- Skim this week's `research/daily/*.md` C-class lines in case the daily run under-classified
  something; you may promote it (record why).

## Step 2 — Six questions per candidate

Answer each in writing in the weekly report. Short, concrete, sourced.

1. **Significant?** Would an LLM research engineer materially benefit from learning it?
2. **Enough evidence?** More than one interesting paper? Look for independent reproduction,
   real implementations, adoption in major systems or widely used open-source stacks, strong
   (independently verified) benchmark evidence, repeated appearance in serious research.
   Label company self-reports as **(company claim)**.
3. **Mature enough to teach?** Is the mechanism settled enough that a lesson written now will
   still be correct in a year? Interesting ≠ teachable.
4. **Where does it fit?** Name the exact module/lesson from the real course structure
   (the spine in `curriculum/course-outline.md`). Are all its prerequisites already taught
   before that point? If not, the lesson must teach the missing prerequisite, or it waits.
5. **Genuinely new?** Does it add a concept the course does not already teach, or is it an
   existing lesson under new terminology? If mostly covered → no new lesson (at most an
   extension section, only if the delta is substantial).
6. **Implement?** Implement when the mechanism is understandable at toy scale and building it
   teaches something. Do not implement merely because code exists.

## Step 3 — Decide: ADD / WAIT / REJECT

No numeric score decides. Write a reasoned decision, e.g.

```text
ADD
Reason: moved beyond an isolated result — independently reproduced and implemented in several
modern LLM systems. Extends Lesson X; added as an advanced extension, not a replacement.
```

- **ADD** — several of: technically important; demonstrated by serious research; adopted or
  implemented by major organizations; reproducible; practically useful; relevant to modern LLM
  systems; likely to remain relevant; teaches an underlying concept; fits naturally.
  **Budget: at most one ADD per week, two only if both are clearly exceptional.** If more
  qualify, ADD the most important and WAIT the rest with "next review: next week".
- **WAIT** — promising, insufficient evidence/maturity. Set **Next review** (a date, typically
  4–8 weeks out) and write exactly what evidence would change the decision.
- **REJECT** — too narrow, immature, hype, or not useful enough for curriculum space. State what
  would have to change to reopen it.
- B-class items default to WAIT unless the review finds strong reasons otherwise.

## Step 4 — Record decisions (every candidate, every outcome)

For each topic, create or update exactly one topic file from `templates/topic.md`, in the folder
matching its status: `accepted/`, `deferred/`, `rejected/`. Use `git mv` when status changes so
history is preserved. Append a line to its **History**:

```text
- YYYY-MM-DD — WAIT — <one-sentence reason> — weekly/YYYY-Www.md — candidates C-…, C-…
```

Deferred topics can later become ADD when new evidence arrives; rejected topics are reopened
only by materially new evidence (record which).

## Step 5 — Only for ADD: build a full lesson

### Placement (in this order of preference)

1. **New lesson inside the right module**, appended after the module's last lesson
   (e.g. `lessons/module-16/lesson-04.md`). Default for a genuinely new concept.
2. **"Advanced extension" section appended at the end of an existing lesson**, just above its
   `## Next` section, headed `## Advanced extension: <topic> (added YYYY-MM-DD)`. Only when the
   topic is a natural continuation of that one lesson. Do not edit any existing sentence.
3. `lessons/frontier/update-NN.md` only for important cross-cutting developments that do not
   belong to a single module (e.g. a new industry benchmark discipline).

Never place a lesson before its prerequisites. Update `_sidebar.md` (add the link; do not
reorder or rename existing entries) and, if a new lesson, add a row/line to
`curriculum/course-outline.md` without changing existing rows.

### Progression, never replacement

The student must never read "what you learned before is obsolete". Structure the lesson as:
the foundational technique (link the existing lesson; recap in 2–3 sentences) → the concrete
problem it has at modern scale → the modern technique → a side-by-side comparison → when you
would still use the original. Example tone: *"The attention algorithm you implemented remains
correct. Here is how modern implementations make it far more memory/I-O efficient."*

### Lesson content

Follow the lesson template in `CLAUDE.md` exactly (prereq block; six-pass structure:
intuition → math with every symbol named → worked numerical example with tiny numbers →
tensor shapes/dtype/device → from-scratch implementation → what the framework does / cost).
Also include, as appropriate: motivation; the problem with the previous approach; relationship
to existing concepts; a diagram if it helps; exercises with hint → stronger hint → collapsed
solution; an implementation exercise; an experiment with expected observations; limitations;
comparison with the foundational technique; a debugging exercise; common mistakes; "what to
read / what to skip"; **Check yourself** `<details>` Q&A; a hardware-track `<div class="hw">`
if there is real compute; **Next** link; references to the original paper and important
implementations. Mark claims as PUBLICLY DOCUMENTED / REASONABLE INDUSTRY PRACTICE /
INFERENCE. Verify every URL. KaTeX math; no bare `$` inside code fences.
Links follow the repo convention docsify resolves: markdown links are root-relative
(`[16.1](lessons/module-16/lesson-01.md)`), links inside raw-HTML blocks use hash routes
(`<a href="#/lessons/module-16/lesson-01">`).
Depth as needed for the concept — not a news summary, not artificially long.

### Code policy

- **Never modify existing functions, classes, tests or exercises** in `code/`.
  The foundational implementation stays exactly as the student learned it.
- New technique → **new module file** next to the related one (e.g. `llmre/rl/gvpo.py` beside
  `llmre/rl/grpo.py`), whose docstring names the foundational module it extends.
- New tests → **new test file** in `code/tests/` (e.g. `test_<topic>.py`); include a test that
  compares against / reduces to the foundational implementation where that is meaningful.
- Shared infrastructure may change only if the change is required for the lesson, backwards
  compatible, and educationally useful — explain it in the weekly report.
- `cd code && python -m pytest -q` must pass with **all** pre-existing tests still passing.

## Step 6 — Weekly report, changelog, reset

1. Write `research/weekly/YYYY-Www.md` from `templates/weekly-report.md` (ISO week). It must
   list: candidates reviewed (with the six answers), **Added to course** (topic, why it
   qualifies, where, lesson created, code added, sources, adoption evidence), **Deferred**,
   **Rejected**. If nothing was added write exactly: **No curriculum changes this week.**
2. Append to `research/CHANGELOG.md` (newest first): the week, and either the additions with
   links or "No curriculum changes this week."
3. Move the whole content of `staging.md`'s `## Candidates` section into the weekly report's
   `## Appendix: staged candidates` section, then reset `staging.md` to the empty template.
4. `python research/tools/research.py index` then `python research/tools/research.py check`.
5. Commit: `curriculum: weekly review YYYY-Www — <ADD n / WAIT n / REJECT n>` (course changes and
   research files in one commit so the audit trail is atomic).
