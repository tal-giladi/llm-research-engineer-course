"""Report answer-length bias for quiz part files: py quiz/lengthcheck.py quiz/parts/05.2.json ..."""
import json
import sys

for path in sys.argv[1:]:
    qs = json.load(open(path, encoding="utf-8"))
    bad, longest = [], 0
    for q in qs:
        lens = [len(o) for o in q["options"]]
        c = lens[q["answer_index"]]
        others = [l for i, l in enumerate(lens) if i != q["answer_index"]]
        if c == max(lens):
            longest += 1
        if c > 1.25 * (sum(others) / 3) or c == max(lens) and c > 1.1 * max(others):
            bad.append(q["id"])
    print(f"{path}: {len(qs)} q, correct-is-longest {longest / len(qs):.0%}, flagged {len(bad)}")
    if bad:
        print("  ", " ".join(bad))
