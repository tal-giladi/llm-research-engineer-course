# Joint multi-domain RL vs specialist RL + multi-teacher distillation (Inkling 2 Small, Kimi K3.1 Pro)

- **Topic ID:** joint-multidomain-rl-vs-mopd
- **Status:** WAIT
- **Next review:** 2027-01-03
- **Course change:** lessons/module-17/lesson-04.md (dated example paragraph citing the Kimi K3.1 Pro card, added 2026-11-15 on the author's instruction; no new mechanism taught)
- **Candidates:** C-20261108-04, C-20261108-05

## Summary

Two labs describe post-training as one RL run across all domains instead of separate specialist RL runs merged by multi-teacher on-policy distillation (17.4 extension). The design choice is real; no controlled comparison is published.

## Evidence

- https://huggingface.co/moonshotai/Kimi-K3.1-Pro — "large-scale joint reinforcement learning across hundreds of downstream domains" (verified 2026-11-15; company claim).
- https://thinkingmachines.ai/news/inkling-2-small/ — Inkling 2 Small: one joint RL run replacing the first Inkling's specialists + MOPD; coding 38.5% of rollouts but 65% of RL tokens; NVFP4 end to end (primary page not reachable; recorded from secondary coverage, e.g. https://latent.space/p/ainews-inkling-2-small-17a-594b-trained-in-68-hours-on-12t-tokens-beating-muse-spark/ ; company claims).

## What would change the decision

A technical report or paper comparing joint RL with specialists + MOPD at matched compute, or open weights with a documented recipe (how domains are balanced). Then a short "design choice" section in 17.4's extension with the token-share arithmetic as an exercise.

## History

- 2026-11-15 — WAIT — a real design choice from two labs, but company claims only, primary Inkling pages not read, and no comparison; Kimi's card line added to 17.4 as a dated example per Tal's instruction — weekly/2026-W46.md — candidates C-20261108-04, C-20261108-05
