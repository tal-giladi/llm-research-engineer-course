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

## Follow-up: one implementation and the full reward-training flow

- [x] Inspect existing reward model, tokenizer, chat template, and checkpoint conventions.
- [x] Remove the separate demo model; use the repository RewardModel everywhere.
- [x] Implement and explain the full SFT-to-reward-model training flow, evaluation, saving, and reload.
- [x] Test padding, pair alignment, checkpoint reload/resume, and preservation of the SFT model.
- [x] Check the revised lesson and prepare the follow-up for publication.

Keep the original lesson backup unchanged. Show a single-device full-precision training baseline with explicit scaling limits; PPO policy optimization belongs to 15.2.

Follow-up validation: 14 targeted tests pass, including exact epoch-boundary resume with dropout and unequal microbatch accumulation. The command-line entry point was also run end to end on synthetic fixture data; its saved scorer reloaded and scored independently. The lesson includes the exact checked-in training module. All 44 math expressions render; new links exist; content outside the mechanics section and the original backup remain unchanged. Academy dry-run result remains 616 legacy format problems, matching the earlier report.

Unequal-length clarification: added the concrete two-pair example next to the tensor-shapes table: real lengths 8/11/13/7, shared padded shape (4,13), final real indices 7/10/12/6, and preserved chosen/rejected alignment. Verified these shapes and indices with the existing collator. The original lesson backup remains unchanged.

Dropout clarification: expanded the backbone's compact embedding expression into token lookup, position lookup, addition, and dropout, with matching lesson code. Explained activation masking versus persistent weights, initialization versus per-batch forward computation, train/eval modes, and zero-dropout behavior. The language head is excluded during construction; input embeddings remain trainable.
