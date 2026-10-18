"""Merge quiz/parts/*.json into quiz/questions.json and validate every question."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
REQUIRED = ["id", "chapter", "chapter_name", "module", "module_name", "source",
            "section", "question", "options", "answer_index", "explanation"]


def chapter_key(ch):
    if ch.startswith("F."):
        return (999, int(ch[2:]), "")
    parts = re.split(r"[.]", ch)
    minor = parts[1] if len(parts) > 1 else "0"
    m = re.match(r"(\d+)", minor)
    return (int(parts[0]), int(m.group(1)) if m else 0, minor)


def main():
    errors, questions, seen = [], [], set()
    for f in sorted((ROOT / "parts").glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errors.append(f"{f.name}: {e}")
            continue
        for q in data:
            qid = q.get("id", "?")
            missing = [k for k in REQUIRED if k not in q]
            if missing:
                errors.append(f"{qid}: missing {missing}")
                continue
            if len(q["options"]) != 4 or len(set(q["options"])) != 4:
                errors.append(f"{qid}: need 4 distinct options")
            if q["answer_index"] not in (0, 1, 2, 3):
                errors.append(f"{qid}: bad answer_index")
            if qid in seen:
                errors.append(f"{qid}: duplicate id")
            seen.add(qid)
            questions.append(q)
    questions.sort(key=lambda q: (chapter_key(q["chapter"]), q["id"]))
    out = {"version": 1, "count": len(questions), "questions": questions}
    (ROOT / "questions.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    dist = [sum(q["answer_index"] == i for q in questions) for i in range(4)]
    print(f"{len(questions)} questions, answer spread {dist}, {len(errors)} errors")
    for e in errors:
        print("  ", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
