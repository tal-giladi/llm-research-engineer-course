# Question bank

`questions.json` — 4,585 self-contained multiple-choice questions covering every lesson (00.1 → 20,
practice sets, frontier updates). Built from `parts/<chapter>.json` by `py quiz/build.py`.

Each question: `id, chapter, chapter_name, module, module_name, source, section, question,
options[4], answer_index (0-3), explanation`.

Generation rules: `GENERATION.md`. Answer-length bias check: `py quiz/lengthcheck.py quiz/parts/*.json`
(correct option is the longest ~18% of the time, shortest ~26%).
