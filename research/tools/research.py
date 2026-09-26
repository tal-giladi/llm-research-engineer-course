"""Helper for the curriculum modernization system (see research/README.md).

Stdlib only, so it runs anywhere the repo is checked out.

    python research/tools/research.py new-day [YYYY-MM-DD]
    python research/tools/research.py lookup "<terms>"
    python research/tools/research.py check
    python research/tools/research.py index
    python research/tools/research.py scope --daily | --weekly
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
STATUS_DIRS = {"accepted": "ADD", "deferred": "WAIT", "rejected": "REJECT"}

CANDIDATE_FIELDS = [
    "Class", "Date discovered", "Date published", "Source", "Organization/researchers",
    "Category", "What changed", "Technical summary", "Why it might matter",
    "Evidence of adoption", "Major organizations using it", "Open-source implementation",
    "Paper", "Code", "Relationship to existing course material", "Potential course lesson",
    "Confidence", "Recommendation",
]
TOPIC_FIELDS = ["Topic ID", "Status", "Next review", "Course change", "Candidates"]

CANDIDATE_HEAD = re.compile(r"^### (C-\d{8}-\d{2}) · (.+)$", re.M)
FIELD = re.compile(r"^- \*\*(.+?):\*\*[ \t]*(.*)$", re.M)
HISTORY_LINE = re.compile(r"^- \d{4}-\d{2}-\d{2} — (ADD|WAIT|REJECT) — .+", re.M)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_candidates(text: str) -> list[dict]:
    """Split a markdown file into candidate records keyed by field name."""
    heads = list(CANDIDATE_HEAD.finditer(text))
    out = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[h.end():end]
        fields = {m.group(1): m.group(2).strip() for m in FIELD.finditer(body)}
        out.append({"id": h.group(1), "title": h.group(2).strip(), "fields": fields})
    return out


def parse_topic(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), path.stem)
    fields = {m.group(1): m.group(2).strip() for m in FIELD.finditer(text)}
    history = HISTORY_LINE.findall(text)
    last = re.findall(r"^- (\d{4}-\d{2}-\d{2}) — ", text, re.M)
    return {"path": path, "title": title, "fields": fields, "history": history,
            "last_date": max(last) if last else ""}


def topic_files() -> list[Path]:
    return sorted(p for d in STATUS_DIRS for p in (RESEARCH / d).glob("*.md"))


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_new_day(date: str | None) -> int:
    date = date or dt.date.today().isoformat()
    dt.date.fromisoformat(date)
    path = RESEARCH / "daily" / f"{date}.md"
    if path.exists():
        print(f"exists: {path.relative_to(ROOT).as_posix()}")
        return 0
    tpl = (RESEARCH / "templates" / "daily.md").read_text(encoding="utf-8")
    path.write_text(tpl.replace("{date}", date), encoding="utf-8")
    print(f"created: {path.relative_to(ROOT).as_posix()}")
    return 0


def check_candidates(text: str, where: str) -> list[str]:
    errors = []
    for c in parse_candidates(text):
        missing = [f for f in CANDIDATE_FIELDS if not c["fields"].get(f)]
        if missing:
            errors.append(f"{where}: {c['id']} missing {', '.join(missing)}")
        cls = c["fields"].get("Class", "")
        if cls and cls not in {"A", "B"}:
            errors.append(f"{where}: {c['id']} Class must be A or B (C/D/E are one-liners), got {cls!r}")
        if not c["id"][2:10] == c["fields"].get("Date discovered", "").replace("-", ""):
            errors.append(f"{where}: {c['id']} ID date does not match Date discovered")
    return errors


def check_topic(t: dict) -> list[str]:
    errors = []
    rel = t["path"].relative_to(ROOT).as_posix()
    missing = [f for f in TOPIC_FIELDS if not t["fields"].get(f)]
    if missing:
        errors.append(f"{rel}: missing {', '.join(missing)}")
    expected = STATUS_DIRS[t["path"].parent.name]
    if t["fields"].get("Status") and t["fields"]["Status"] != expected:
        errors.append(f"{rel}: Status {t['fields']['Status']} but file is in {t['path'].parent.name}/ ({expected})")
    if t["fields"].get("Topic ID") and t["fields"]["Topic ID"] != t["path"].stem:
        errors.append(f"{rel}: Topic ID must equal file name")
    if not t["history"]:
        errors.append(f"{rel}: History needs at least one '- YYYY-MM-DD — STATUS — reason' line")
    elif t["history"][-1] != expected:
        errors.append(f"{rel}: last History status {t['history'][-1]} != {expected}")
    return errors


def cmd_check() -> int:
    errors: list[str] = []
    ids: dict[str, str] = {}
    for p in sorted((RESEARCH / "daily").glob("*.md")):
        text = p.read_text(encoding="utf-8")
        errors += check_candidates(text, p.relative_to(ROOT).as_posix())
        for c in parse_candidates(text):
            if c["id"] in ids:
                errors.append(f"duplicate candidate ID {c['id']} in {p.name} and {ids[c['id']]}")
            ids[c["id"]] = p.name
    staging = RESEARCH / "staging.md"
    errors += check_candidates(staging.read_text(encoding="utf-8"), "research/staging.md")
    seen = [c["id"] for c in parse_candidates(staging.read_text(encoding="utf-8"))]
    errors += [f"research/staging.md: duplicate {i}" for i in {i for i in seen if seen.count(i) > 1}]
    topic_ids: set[str] = set()
    for p in topic_files():
        t = parse_topic(p)
        errors += check_topic(t)
        if p.stem in topic_ids:
            errors.append(f"topic {p.stem} exists in more than one status folder")
        topic_ids.add(p.stem)
    for e in errors:
        print(e)
    print("check: OK" if not errors else f"check: {len(errors)} problem(s)")
    return 1 if errors else 0


def cmd_index() -> int:
    rows = []
    for p in topic_files():
        t = parse_topic(p)
        f = t["fields"]
        rel = p.relative_to(RESEARCH).as_posix()
        rows.append((f.get("Status", "?"), t["title"], f"research/{rel}", f.get("Next review", ""),
                     f.get("Course change", ""), t["last_date"]))
    order = {"ADD": 0, "WAIT": 1, "REJECT": 2}
    rows.sort(key=lambda r: (order.get(r[0], 9), r[1].lower()))
    lines = [
        "# Topic registry",
        "",
        "Generated by `python research/tools/research.py index` — do not edit by hand.",
        "Every topic the weekly review has decided on, so nothing is reconsidered without new evidence.",
        "",
        "| Status | Topic | Next review | Course change | Last decision |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {s} | [{t}]({p}) | {n} | {c} | {d} |" for s, t, p, n, c, d in rows]
    if not rows:
        lines.append("| — | (no topics yet) | | | |")
    (RESEARCH / "REGISTRY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"index: {len(rows)} topic(s)")
    return 0


def cmd_lookup(terms: str) -> int:
    """Case-insensitive search of the registry, staging, and the course itself."""
    words = [w for w in re.split(r"\s+", terms.lower()) if w]
    areas = [("registry", topic_files()),
             ("staging", [RESEARCH / "staging.md"]),
             ("course", sorted((ROOT / "lessons").rglob("*.md")) + sorted((ROOT / "papers").glob("*.md"))),
             ("code", sorted((ROOT / "code" / "src").rglob("*.py")))]
    hits = 0
    for label, paths in areas:
        for p in paths:
            text = p.read_text(encoding="utf-8", errors="replace")
            low = text.lower()
            if all(w in low for w in words):
                first = next((l.strip() for l in text.splitlines() if words[0] in l.lower()), "")
                print(f"[{label}] {p.relative_to(ROOT).as_posix()}: {first[:120]}")
                hits += 1
    if not hits:
        print("no matches — not covered by the course or registry")
    return 0


def cmd_scope(mode: str) -> int:
    """Guard: the daily run may only change research/."""
    out = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    if mode == "daily":
        bad = [f for f in out if not f.startswith("research/")]
        if bad:
            print("daily run may only change research/; unstage:\n  " + "\n  ".join(bad))
            return 1
    print(f"scope ({mode}): OK — {len(out)} staged file(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new-day"); p.add_argument("date", nargs="?")
    sub.add_parser("check")
    sub.add_parser("index")
    p = sub.add_parser("lookup"); p.add_argument("terms")
    p = sub.add_parser("scope"); g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--daily", action="store_true"); g.add_argument("--weekly", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "new-day":
        return cmd_new_day(a.date)
    if a.cmd == "check":
        return cmd_check()
    if a.cmd == "index":
        return cmd_index()
    if a.cmd == "lookup":
        return cmd_lookup(a.terms)
    return cmd_scope("daily" if a.daily else "weekly")


if __name__ == "__main__":
    sys.exit(main())
