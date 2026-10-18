# Question-bank generation rules

Output: one JSON file per lesson in `quiz/parts/<chapter>.json` (e.g. `quiz/parts/05.2.json`,
frontier lessons `F.1.json`, practice sets `01.P34.json` / `07.P4.json`). Each file is a JSON array of:

```json
{
  "id": "05.2-001",
  "chapter": "05.2",
  "chapter_name": "Q/K/V & scaled dot-product attention",
  "module": "5",
  "module_name": "Transformer from scratch",
  "source": "lessons/module-05/lesson-02.md",
  "section": "the ## / ### heading the fact comes from",
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "answer_index": 2,
  "explanation": "..."
}
```

Rules
1. Coverage: walk the lesson top to bottom; for almost every substantive line/claim (definitions,
   intuitions, equations and what each symbol means, shapes/dtypes, costs, code behaviour, common
   mistakes, research connections, check-yourself items, prerequisites/why-it-matters) write a
   question. Skip only the worked numerical examples' arithmetic (a question about the *idea*
   a numerical example illustrates is fine). Expect roughly one question per 3–6 lesson lines.
2. Self-contained: the question is the only thing the learner sees. Never say "the lesson",
   "above", "this example", "the code shown", "in this chapter". Name every object fully
   (e.g. "In scaled dot-product attention, why are the scores divided by sqrt(d_k)?").
3. Exactly 4 options, exactly one correct. Distractors are plausible, same length/style as the
   correct one, and wrong for a clear reason. No "all of the above"/"none of the above".
   Spread answer_index roughly evenly over 0–3.
4. Explanation: 2–5 sentences that teach the fact fully — why the right answer is right and why
   the tempting distractor is wrong — so the learner never needs to open the course.
5. Math: plain-text or KaTeX `$...$` is fine; keep it readable inside JSON strings (escape
   backslashes properly — the file must parse with `json.load`).
6. Faithful to the course content; don't invent facts the lesson doesn't support.
7. ids are `<chapter>-NNN`, sequential from 001. UTF-8, valid JSON, no trailing commas.

## Length balance (mandatory)
The correct option must NOT be identifiable by length or detail. All four options must be of
similar length (within ~20%) and equal specificity/hedging. Write distractors as full, precise,
technical-sounding statements that are wrong for a clear reason — not short strawmen. Across a
file, the correct answer should be the longest option only ~25% of the time. Check with
`py quiz/lengthcheck.py <file>`; it must report 0 flagged and correct-is-longest ≤ 35%.
