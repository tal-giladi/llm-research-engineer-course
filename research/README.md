# Curriculum modernization system

The course is a **curriculum, not a news feed**. This folder is how it stays modern without
churning: many discoveries go in, very few serious candidates come out, and the course changes
at most once a week, only when something clears a high bar. **A week with zero course changes is
a successful week.**

```text
DAILY  (research only — never touches the course)
  web research ─▶ classify A/B/C/D/E ─▶ daily/YYYY-MM-DD.md (all items)
                                     └▶ staging.md (A and B only, full candidate record)

WEEKLY (the only process allowed to change the course)
  staging.md + registry ─▶ six questions per candidate ─▶ ADD / WAIT / REJECT
     ADD    ─▶ full lesson (+ separate code + tests) ─▶ sidebar ─▶ CHANGELOG.md
     WAIT   ─▶ deferred/<topic>.md   (re-reviewed when new evidence appears)
     REJECT ─▶ rejected/<topic>.md   (not reconsidered without materially new evidence)
  ─▶ weekly/YYYY-Www.md report ─▶ staging.md reset
```

## Audit trail

Every course change can be traced back, file to file:

```text
source URL ─▶ candidate (daily/…, staging.md, ID C-YYYYMMDD-NN)
           ─▶ weekly evaluation (weekly/YYYY-Www.md)
           ─▶ decision + history (accepted|deferred|rejected/<topic>.md)
           ─▶ course change (lesson path, code path, commit) + CHANGELOG.md
```

## Layout

| Path | What it holds | Written by |
|---|---|---|
| `PROTOCOL-daily.md` | Instructions for the daily research run | human |
| `PROTOCOL-weekly.md` | Instructions for the weekly curriculum review | human |
| `templates/` | Candidate, topic and weekly-report templates | human |
| `daily/YYYY-MM-DD.md` | Everything found that day, classified A–E | daily run |
| `staging.md` | "Things worth considering": this week's A/B candidates in full | daily run; reset weekly |
| `weekly/YYYY-Www.md` | Weekly review report + archived staging content | weekly run |
| `accepted/` | Topics that are in the course (with where and when) | weekly run |
| `deferred/` | WAIT topics being monitored, with decision history | weekly run |
| `rejected/` | REJECT topics, kept so they are not reconsidered for nothing | weekly run |
| `REGISTRY.md` | Generated index of every topic and its status | `tools/research.py index` |
| `CHANGELOG.md` | Student-facing log of curriculum changes (linked in the sidebar) | weekly run |
| `tools/research.py` | Stdlib helper: scaffold, validate, index, dedup lookup, scope guard | human |

## Tool

```bash
python research/tools/research.py new-day            # create today's daily file if missing
python research/tools/research.py lookup "GRPO"      # has the course or registry seen this?
python research/tools/research.py check              # validate candidates + topic files
python research/tools/research.py index              # regenerate REGISTRY.md
python research/tools/research.py scope --daily      # fail if staged changes leave research/
```

## Automation

Two cloud routines run against the GitHub repo (see `TODO_FOR_TAL.md` for their IDs):

- **Daily research** — every day, research + classify + stage. Commits only `research/`.
- **Weekly curriculum review** — once a week, decides ADD/WAIT/REJECT and, only for ADD, edits the course.

Both follow the protocols in this folder verbatim; to change behaviour, edit the protocol,
not the routine.
