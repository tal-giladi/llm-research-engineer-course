#!/usr/bin/env python3
"""course.py — track which lessons of the LLM course you have done and where you struggled.

Standard library only. Reads the lesson list from _sidebar.md (so lessons added later show
up automatically), reads/writes progress/progress.json, and regenerates PROGRESS.md on every
change. It never touches course content.

Lesson ids: 00.2 = lessons/module-00/lesson-02.md · 20.1 = the capstone · F.1 = frontier update 1

Quick reference
  python course.py status                   where am I?
  python course.py next                     what should I study next?
  python course.py start 05.2               mark a lesson in progress
  python course.py read 05.2                I read it (did not do the exercises yet)
  python course.py complete 05.2            I did it, exercises included
  python course.py skip 01.1 --reason "I know this"
  python course.py quiz 05.2 4/5            record a Check yourself score
  python course.py struggle 05.2 "why divide by sqrt(d_k)?"
  python course.py resolve 05.2             struggles on that lesson are resolved
  python course.py note 05.2 "revisit the causal mask trick"
  python course.py review                   lessons worth revisiting
  python course.py show 05.2                one lesson: status, quiz, struggles, notes
  python course.py list [05|F]              lessons with status (optionally one module)
  python course.py log [-n 20]              recent activity
  python course.py reset 05.2               forget progress on a lesson
  python course.py render                   regenerate PROGRESS.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIDEBAR = ROOT / "_sidebar.md"
PROGRESS = Path(os.environ.get("COURSE_PROGRESS_FILE", ROOT / "progress" / "progress.json"))
PROGRESS_MD = Path(os.environ.get("COURSE_PROGRESS_MD", ROOT / "PROGRESS.md"))

STATUSES = ("not-started", "in-progress", "read", "completed", "skipped")
DONE = {"completed", "skipped"}
ICON = {"not-started": "⬜", "in-progress": "🟨", "read": "📖", "completed": "✅", "skipped": "⏭️"}
LOW_QUIZ = 0.7


# --------------------------------------------------------------------------- lessons
def lesson_id(path: str) -> str | None:
    m = re.search(r"lessons/module-(\d+)/lesson-(\d+)\.md$", path)
    if m:
        return f"{int(m.group(1)):02d}.{int(m.group(2))}"
    m = re.search(r"lessons/frontier/update-(\d+)\.md$", path)
    if m:
        return f"F.{int(m.group(1))}"
    return None


def load_lessons(sidebar: Path = SIDEBAR) -> tuple[list[dict], dict]:
    """Parse _sidebar.md into ordered modules and lessons. Non-lesson links are ignored."""
    modules: list[dict] = []
    lessons: dict[str, dict] = {}
    current = None
    for line in sidebar.read_text(encoding="utf-8").splitlines():
        head = re.match(r"^- \*\*(.+?)\*\*", line)
        if head:
            title = head.group(1)
            key = title.split(" · ")[0].strip()
            current = {"key": "F" if title.lower().startswith("frontier") else key, "title": title, "lessons": []}
            modules.append(current)
            continue
        link = re.match(r"^\s+- \[(.+?)\]\((.+?)\)", line)
        if not (link and current):
            continue
        lid = lesson_id(link.group(2))
        if not lid or lid in lessons:
            continue
        text = link.group(1)
        title = text.split(" · ", 1)[1] if " · " in text else text
        lessons[lid] = {"id": lid, "title": title, "path": link.group(2), "module": current["key"],
                        "order": len(lessons)}
        current["lessons"].append(lid)
    return [m for m in modules if m["lessons"]], lessons


# --------------------------------------------------------------------------- progress data
def today() -> str:
    return dt.date.today().isoformat()


def empty_progress() -> dict:
    return {"schema": 1, "started": today(), "lessons": {}, "struggles": [], "log": []}


def load_progress() -> dict:
    base = empty_progress()
    if PROGRESS.exists():
        base.update(json.loads(PROGRESS.read_text(encoding="utf-8")))
    return base


def save_progress(p: dict, modules: list[dict], lessons: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    PROGRESS_MD.write_text(render_markdown(p, modules, lessons), encoding="utf-8")


def log(p: dict, event: str) -> None:
    p["log"].append({"date": today(), "event": event})
    p["log"] = p["log"][-500:]


def normalize_id(raw: str, lessons: dict) -> str:
    s = raw.strip().upper()
    m = re.fullmatch(r"(\d{1,2})(?:\.(\d{1,2}))?", s)
    if m:
        s = f"{int(m.group(1)):02d}.{int(m.group(2) or 1)}"
    m = re.fullmatch(r"F\.?(\d{1,2})", s)
    if m:
        s = f"F.{int(m.group(1))}"
    if s not in lessons:
        prefix = s.split(".")[0]
        close = [k for k in lessons if k.split(".")[0] == prefix]
        sys.exit(f"Unknown lesson '{raw}'." + (f" Lessons in that module: {', '.join(close)}" if close else ""))
    return s


def status(p: dict, lid: str) -> str:
    return p["lessons"].get(lid, {}).get("status", "not-started")


def set_status(p: dict, lid: str, new: str) -> None:
    entry = p["lessons"].setdefault(lid, {})
    entry["status"] = new
    entry.setdefault("history", []).append({"date": today(), "status": new})
    if new == "in-progress":
        entry.setdefault("started", today())
    else:
        entry[new] = today()


def open_struggles(p: dict, lid: str | None = None) -> list[dict]:
    return [s for s in p["struggles"] if not s.get("resolved") and (lid is None or s["id"] == lid)]


def quiz_ratio(score: str) -> float:
    right, total = (int(x) for x in score.split("/"))
    return right / total if total else 1.0


def review_items(p: dict, lessons: dict) -> list[tuple[str, str]]:
    """(lesson id, reason) for lessons worth revisiting, in course order."""
    out = []
    for lid in sorted(lessons, key=lambda k: lessons[k]["order"]):
        reasons = []
        n = len(open_struggles(p, lid))
        if n:
            reasons.append(f"{n} open struggle{'s' if n > 1 else ''}")
        q = p["lessons"].get(lid, {}).get("quiz")
        if q and quiz_ratio(q[-1]["score"]) < LOW_QUIZ:
            reasons.append(f"last quiz {q[-1]['score']}")
        if status(p, lid) == "read":
            reasons.append("read but exercises not done")
        if reasons:
            out.append((lid, ", ".join(reasons)))
    return out


def next_lessons(p: dict, lessons: dict) -> tuple[list[str], str | None]:
    ordered = sorted(lessons, key=lambda k: lessons[k]["order"])
    in_progress = [l for l in ordered if status(p, l) == "in-progress"]
    nxt = next((l for l in ordered if status(p, l) == "not-started"), None)
    return in_progress, nxt


# --------------------------------------------------------------------------- commands
def fmt(lessons: dict, lid: str) -> str:
    return f"{lid} {lessons[lid]['title']}"


def cmd_status(p, modules, lessons, args) -> None:
    done = sum(1 for l in lessons if status(p, l) in DONE)
    print(f"Lessons done: {done}/{len(lessons)} ({100 * done // max(1, len(lessons))}%)")
    for m in modules:
        ids = m["lessons"]
        if any(status(p, i) != "not-started" for i in ids):
            d = sum(1 for i in ids if status(p, i) in DONE)
            bar = "█" * round(10 * d / len(ids)) + "░" * (10 - round(10 * d / len(ids)))
            print(f"  {bar} {d}/{len(ids)}  {m['title']}")
    st = open_struggles(p)
    if st:
        print(f"Open struggles: {len(st)}  (python course.py review)")
    print()
    cmd_next(p, modules, lessons, args)


def cmd_next(p, modules, lessons, args) -> None:
    in_progress, nxt = next_lessons(p, lessons)
    for l in in_progress:
        print(f"Continue: {fmt(lessons, l)}  →  {lessons[l]['path']}")
    if nxt:
        print(f"Next:     {fmt(lessons, nxt)}  →  {lessons[nxt]['path']}")
    rev = review_items(p, lessons)
    if rev:
        l, why = rev[0]
        print(f"Revisit:  {fmt(lessons, l)}  ({why})" + (f"  +{len(rev) - 1} more: python course.py review" if len(rev) > 1 else ""))
    if not (in_progress or nxt):
        print("Every lesson is done or skipped.")


def change_status(new: str):
    def run(p, modules, lessons, args) -> None:
        lid = normalize_id(args.id, lessons)
        set_status(p, lid, new)
        if new == "skipped":
            p["lessons"][lid]["reason"] = args.reason or ""
        log(p, f"{new} {lid}")
        save_progress(p, modules, lessons)
        print(f"{ICON[new]} {fmt(lessons, lid)} → {new}")
        if new == "completed" and open_struggles(p, lid):
            print(f"You still have open struggles on {lid}. If they are clear now: python course.py resolve {lid}")
        if new in DONE:
            _, nxt = next_lessons(p, lessons)
            if nxt:
                print(f"Next: {fmt(lessons, nxt)}")
    return run


def cmd_quiz(p, modules, lessons, args) -> None:
    lid = normalize_id(args.id, lessons)
    if not re.fullmatch(r"\d+/\d+", args.score):
        sys.exit("Score looks like 4/5")
    p["lessons"].setdefault(lid, {"status": "not-started"}).setdefault("quiz", []).append(
        {"date": today(), "score": args.score})
    log(p, f"quiz {lid} {args.score}")
    save_progress(p, modules, lessons)
    print(f"Recorded {lid} Check yourself: {args.score}")
    if quiz_ratio(args.score) < LOW_QUIZ:
        print("Below 70% — it will show up in `python course.py review`. Reread the lesson or log what is unclear with `struggle`.")


def cmd_struggle(p, modules, lessons, args) -> None:
    lid = normalize_id(args.id, lessons)
    p["struggles"].append({"id": lid, "note": args.note, "date": today(), "resolved": None})
    log(p, f"struggle {lid}: {args.note}")
    save_progress(p, modules, lessons)
    total = sum(1 for s in p["struggles"] if s["id"] == lid)
    print(f"Noted. Struggles on {lid}: {len(open_struggles(p, lid))} open, {total} total.")


def cmd_resolve(p, modules, lessons, args) -> None:
    lid = normalize_id(args.id, lessons)
    n = 0
    for s in open_struggles(p, lid):
        s["resolved"] = today()
        n += 1
    log(p, f"resolve {lid}")
    save_progress(p, modules, lessons)
    print(f"Resolved {n} struggle(s) on {lid}.")


def cmd_note(p, modules, lessons, args) -> None:
    lid = normalize_id(args.id, lessons)
    p["lessons"].setdefault(lid, {"status": "not-started"}).setdefault("notes", []).append(
        {"date": today(), "note": args.note})
    log(p, f"note {lid}")
    save_progress(p, modules, lessons)
    print(f"Note saved on {lid}.")


def cmd_review(p, modules, lessons, args) -> None:
    rev = review_items(p, lessons)
    if not rev:
        print("Nothing to revisit.")
    for lid, why in rev:
        print(f"{fmt(lessons, lid)}  — {why}  →  {lessons[lid]['path']}")
        for s in open_struggles(p, lid):
            print(f"    · {s['date']}: {s['note']}")


def cmd_show(p, modules, lessons, args) -> None:
    lid = normalize_id(args.id, lessons)
    e = p["lessons"].get(lid, {})
    print(f"{ICON[status(p, lid)]} {fmt(lessons, lid)}  [{status(p, lid)}]")
    print(f"   {lessons[lid]['path']}")
    for h in e.get("history", []):
        print(f"   {h['date']}  {h['status']}")
    if e.get("reason"):
        print(f"   skipped: {e['reason']}")
    for q in e.get("quiz", []):
        print(f"   {q['date']}  quiz {q['score']}")
    for s in [s for s in p["struggles"] if s["id"] == lid]:
        print(f"   {s['date']}  struggle: {s['note']}" + (f"  (resolved {s['resolved']})" if s.get("resolved") else ""))
    for n in e.get("notes", []):
        print(f"   {n['date']}  note: {n['note']}")


def cmd_list(p, modules, lessons, args) -> None:
    want = (args.filter or "").upper()
    for m in modules:
        if want and m["key"].lstrip("0") != want.lstrip("0") and m["key"] != want:
            continue
        d = sum(1 for i in m["lessons"] if status(p, i) in DONE)
        print(f"{m['title']}  {d}/{len(m['lessons'])}")
        for i in m["lessons"]:
            flag = "  ⚠" if open_struggles(p, i) else ""
            print(f"  {ICON[status(p, i)]} {fmt(lessons, i)}{flag}")


def cmd_log(p, modules, lessons, args) -> None:
    for e in (p["log"] if args.n <= 0 else p["log"][-args.n:]):
        print(f"{e['date']}  {e['event']}")


def cmd_reset(p, modules, lessons, args) -> None:
    lid = normalize_id(args.id, lessons)
    p["lessons"].pop(lid, None)
    p["struggles"] = [s for s in p["struggles"] if s["id"] != lid]
    log(p, f"reset {lid}")
    save_progress(p, modules, lessons)
    print(f"Reset {lid}")


def cmd_render(p, modules, lessons, args) -> None:
    save_progress(p, modules, lessons)
    print(f"Wrote {PROGRESS_MD.name}")


# --------------------------------------------------------------------------- PROGRESS.md
def render_markdown(p: dict, modules: list[dict], lessons: dict) -> str:
    done = sum(1 for l in lessons if status(p, l) in DONE)
    in_progress, nxt = next_lessons(p, lessons)
    L = [
        "# Progress",
        "",
        "<!-- generated by course.py from progress/progress.json — do not edit by hand -->",
        "",
        f"_Last updated {today()}. Started {p.get('started', '?')}._",
        "",
        "## Where I am",
        "",
        f"- **Lessons done:** {done}/{len(lessons)} ({100 * done // max(1, len(lessons))}%)",
        f"- **Open struggles:** {len(open_struggles(p))}",
    ]
    if in_progress:
        L.append("- **In progress:** " + ", ".join(f"[{fmt(lessons, l)}]({lessons[l]['path']})" for l in in_progress))
    if nxt:
        L.append(f"- **Next:** [{fmt(lessons, nxt)}]({lessons[nxt]['path']})")
    L += [
        "",
        "Legend: ⬜ not started · 🟨 in progress · 📖 read (exercises not done) · ✅ completed · ⏭️ skipped · ⚠ open struggle",
        "",
        "```bash",
        "python course.py next",
        "python course.py complete 05.2",
        "python course.py struggle 05.2 \"note\"",
        "```",
        "",
        "## Lessons",
        "",
    ]
    for m in modules:
        ids = m["lessons"]
        d = sum(1 for i in ids if status(p, i) in DONE)
        active = any(status(p, i) not in ("not-started",) for i in ids) and d < len(ids)
        L.append(f"<details{' open' if active else ''}>")
        L.append(f"<summary><b>{m['title']}</b> — {d}/{len(ids)}</summary>")
        L.append("")
        for i in ids:
            e = p["lessons"].get(i, {})
            extra = ""
            if e.get("quiz"):
                extra += f" · quiz {e['quiz'][-1]['score']}"
            if open_struggles(p, i):
                extra += " · ⚠"
            box = "x" if status(p, i) in DONE else " "
            L.append(f"- [{box}] {ICON[status(p, i)]} [{fmt(lessons, i)}]({lessons[i]['path']}){extra}")
        L += ["", "</details>", ""]

    L += ["## Worth revisiting", ""]
    L += [f"- [{fmt(lessons, l)}]({lessons[l]['path']}) — {why}" for l, why in review_items(p, lessons)] or ["Nothing."]

    L += ["", "## Struggles", ""]
    if p["struggles"]:
        L += ["| Lesson | Note | Date | Resolved |", "|---|---|---|---|"]
        for s in p["struggles"]:
            title = fmt(lessons, s["id"]) if s["id"] in lessons else s["id"]
            L.append(f"| {title} | {s['note']} | {s['date']} | {s.get('resolved') or '—'} |")
    else:
        L.append("None recorded. `python course.py struggle <id> \"what was hard\"`")

    skipped = [(i, e) for i, e in p["lessons"].items() if e.get("status") == "skipped" and i in lessons]
    if skipped:
        L += ["", "## Skipped", ""] + [f"- {fmt(lessons, i)} — {e.get('reason', '')}" for i, e in skipped]

    L += ["", "## Recent activity", ""]
    L += [f"- {e['date']} — {e['event']}" for e in p["log"][-15:][::-1]] or ["Nothing yet."]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="LLM course progress tracker", epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("status", "next", "review", "render"):
        sub.add_parser(name)
    for name in ("start", "read", "complete", "resolve", "show", "reset"):
        sub.add_parser(name).add_argument("id")
    s = sub.add_parser("skip"); s.add_argument("id"); s.add_argument("--reason", default="")
    s = sub.add_parser("quiz"); s.add_argument("id"); s.add_argument("score")
    s = sub.add_parser("struggle"); s.add_argument("id"); s.add_argument("note")
    s = sub.add_parser("note"); s.add_argument("id"); s.add_argument("note")
    s = sub.add_parser("list"); s.add_argument("filter", nargs="?")
    s = sub.add_parser("log"); s.add_argument("-n", type=int, default=20)

    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    modules, lessons = load_lessons()
    p = load_progress()
    handlers = {
        "status": cmd_status, "next": cmd_next, "review": cmd_review, "render": cmd_render,
        "start": change_status("in-progress"), "read": change_status("read"),
        "complete": change_status("completed"), "skip": change_status("skipped"),
        "quiz": cmd_quiz, "struggle": cmd_struggle, "resolve": cmd_resolve, "note": cmd_note,
        "show": cmd_show, "list": cmd_list, "log": cmd_log, "reset": cmd_reset,
    }
    handlers[args.cmd](p, modules, lessons, args)


if __name__ == "__main__":
    main()
