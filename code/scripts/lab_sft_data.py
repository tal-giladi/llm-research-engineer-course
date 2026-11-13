"""Lab for lesson 14.4: build a small, clean SFT set from real public data.

Downloads a slice of two real datasets (streaming, nothing large is stored),
converts them to one ``messages`` schema, runs every cleaning step of
``llmre.sft.data``, decontaminates against GSM8K's test questions, mixes the
two sources to a token budget, and writes ``sft_clean.jsonl`` plus a data card.

    pip install "datasets==5.2.0"
    python scripts/lab_sft_data.py --n 3000 --out data/sft

Runs on CPU in a few minutes (most of it is download). Checked 2026-11-13 with
datasets 5.2.0 against allenai/tulu-3-sft-mixture and
databricks/databricks-dolly-15k, decontaminated against openai/gsm8k (test) and
HuggingFaceH4/MATH-500.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import islice
from pathlib import Path

from llmre.sft.data import (
    approx_tokens,
    build_benchmark_index,
    decontaminate,
    dedupe_by_prompt,
    mix_to_budget,
    near_dedupe_prompts,
    response_issues,
    to_messages,
    validate_messages,
)


SAFETY_SOURCES = ("wildguardmix", "wildjailbreak", "coconot")


def load_rows(name: str, split: str, n: int, config: str | None = None, shuffle: bool = True):
    from datasets import load_dataset  # imported here so tests never need it

    ds = load_dataset(name, config, split=split, streaming=True)
    if shuffle:
        # Mixtures are stored grouped by source: the first N rows of Tulu 3 are
        # all OASST. Shuffling picks random shards and mixes them in a buffer.
        ds = ds.shuffle(seed=0, buffer_size=10_000)
    return list(islice(ds, n))


def clean_source(rows: list[dict], label: str, report: dict, check_truncation: bool) -> list[list[dict]]:
    kept, why, samples = [], Counter(), {}
    for row in rows:
        try:
            chat = to_messages(row)
        except (ValueError, KeyError) as e:
            why["unparseable: " + str(e)[:40]] += 1
            continue
        # Safety subsets teach the model to refuse harmful requests: their
        # refusals are the signal, so the refusal filter must skip them.
        safety = any(k in row.get("source", "") for k in SAFETY_SOURCES)
        problems = validate_messages(chat) or response_issues(
            chat, check_refusals=not safety, check_truncation=check_truncation)
        if problems:
            reason = problems[0].split(": ", 1)[-1]
            why[reason] += 1
            turn = int(problems[0].split()[1].rstrip(":")) if problems[0].startswith("turn") else -1
            samples.setdefault(reason, chat[turn]["content"][-160:])
            continue
        kept.append(chat)
    report[label] = {"in": len(rows), "after_filters": len(kept), "dropped_because": dict(why),
                     "one_dropped_example_per_reason": samples}
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="rows to stream from each source")
    ap.add_argument("--budget", type=int, default=600_000, help="target tokens in the mixture")
    ap.add_argument("--out", default="data/sft")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    tulu_rows = load_rows("allenai/tulu-3-sft-mixture", "train", args.n)
    report["tulu3_sources_sampled"] = dict(Counter(r["source"].split("/")[-1] for r in tulu_rows).most_common())
    tulu = clean_source(tulu_rows, "tulu3", report,
                        check_truncation=True)
    dolly = clean_source(load_rows("databricks/databricks-dolly-15k", "train", args.n), "dolly", report,
                         check_truncation=False)  # human-written: see lesson 14.4

    for label, chats in (("tulu3", tulu), ("dolly", dolly)):
        chats, exact = dedupe_by_prompt(chats)
        chats, near = near_dedupe_prompts(chats, threshold=0.8)
        report[label].update(exact_dupes=exact, near_dupes=near)
        if label == "tulu3":
            tulu = chats
        else:
            dolly = chats

    # Benchmarks we will evaluate on later (13.3): never train on their questions.
    gsm8k = load_rows("openai/gsm8k", "test", 10_000, config="main", shuffle=False)
    math500 = load_rows("HuggingFaceH4/MATH-500", "test", 10_000, shuffle=False)
    index = build_benchmark_index([r["question"] for r in gsm8k] + [r["problem"] for r in math500], n=8)
    for label in ("tulu3", "dolly"):
        chats = tulu if label == "tulu3" else dolly
        clean, removed = decontaminate(chats, index, n=8, threshold=0.5)
        report[label]["contaminated"] = len(removed)
        if removed:
            report[label]["contaminated_example"] = removed[0][0]["content"][:200]
        if label == "tulu3":
            tulu = clean
        else:
            dolly = clean

    mixture, taken = mix_to_budget({"tulu3": tulu, "dolly": dolly}, {"tulu3": 0.7, "dolly": 0.3},
                                   total_tokens=args.budget, seed=0)
    report["mixture"] = {"chats": len(mixture), "approx_tokens_per_source": taken,
                         "approx_tokens_total": sum(approx_tokens(c) for c in mixture)}

    with open(out / "sft_clean.jsonl", "w", encoding="utf-8") as f:
        for chat in mixture:
            f.write(json.dumps({"messages": chat}, ensure_ascii=False) + "\n")
    (out / "data_card.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
