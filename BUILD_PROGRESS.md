# Lesson 15.1 mechanics revision

Goal: rewrite only "What is a reward model, mechanically?" for a learner who already understands SFT, preserve the full original lesson, and push both versions.

- [x] Read the lesson, SFT/backbone code, course instructions, and primary sources.
- [x] Preserve an exact copy of the original lesson and link it for comparison.
- [x] Rewrite the section: independent scoring, final hidden-state readout, pair loss, and shared gradients.
- [x] Execute examples; check links, math rendering, and the course import report.
- [x] Prepare the scoped revision for the requested commit and push to the existing remote branch.

Active writer: main session only. Other modules and existing user changes are outside this revision.

Resume: inspect git status and the diff, then continue the unchecked items. The original belongs at `lessons/module-15/lesson-01.backup.md`; do not overwrite it with the revision.

Decisions: keep the published lesson path and section heading stable. This is a scoped lesson edit, not an Academy format migration of the existing course.

Validation: both runnable Python examples executed with Python 3.14 / PyTorch 2.14.0+cpu. Hand-example loss moved from 1.103186 to 0.424749; GPT demo loss moved from 0.687988 to 0.003655. All three equivalence checks passed. The existing preference suite passed all nine tests. All 44 new math expressions rendered with the bundled KaTeX; new course links exist. The backup matched the original Git blob byte for byte, and content outside the rewritten section matched the backup.

The Academy dry-run importer ran and reported 616 repository-wide legacy-format problems (including missing front matter, disallowed existing HTML, and absent YAML quizzes). Those are outside this section-only revision; this course is not being declared ready for an Academy import. Git history records publication of the revision.
