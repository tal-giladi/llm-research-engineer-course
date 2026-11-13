"""Lab for lesson 15.5: build preference pairs from real ratings and audit them.

Part 1 binarizes raw UltraFeedback (4 rated completions per prompt) two ways,
drops the TruthfulQA prompts, measures ties, margins and length bias, and
writes ``train.jsonl`` and ``validation.jsonl`` in the ``{prompt, chosen, rejected}`` format that
lesson 15.1's ``python -m llmre.preference.train`` reads.

Part 2 measures how often two human annotators agree in HelpSteer3's
preference split (percent agreement and Cohen's kappa).

    pip install "datasets==5.2.0"
    python scripts/lab_preference_data.py --n 4000 --out data/pref

CPU only, a few minutes (mostly download). Checked 2026-11-13 with datasets
5.2.0 against openbmb/UltraFeedback and nvidia/HelpSteer3 (config
"preference").
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from itertools import islice
from pathlib import Path

from llmre.preference.data import (
    binarize,
    cohens_kappa,
    length_balanced_subset,
    length_bias,
    percent_agreement,
    validate_pair,
)


def stream(name: str, n: int, config: str | None = None, split: str = "train"):
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split, streaming=True).shuffle(seed=0, buffer_size=10_000)
    return list(islice(ds, n))


def mean_rating(completion: dict) -> float | None:
    """Mean of UltraFeedback's four aspect ratings (1-5); None if any is missing."""
    vals = []
    for aspect in ("helpfulness", "honesty", "instruction_following", "truthfulness"):
        r = completion["annotations"][aspect]
        r = r[0] if isinstance(r, list) else r          # some aspects are wrapped in a list
        try:
            vals.append(float(r["Rating"]))
        except (KeyError, TypeError, ValueError):
            return None                                  # "N/A" or missing
    return sum(vals) / len(vals)


def part1_ultrafeedback(n: int, out: Path, report: dict) -> None:
    rows = stream("openbmb/UltraFeedback", n)
    report["uf_sources"] = dict(Counter(r["source"] for r in rows).most_common())
    rows = [r for r in rows if r["source"] != "truthful_qa"]   # contaminated with a benchmark
    rng = random.Random(0)
    pairs = {"best_vs_random": [], "best_vs_worst": []}
    skipped = Counter()
    disagree_with_overall = 0
    for r in rows:
        comps = [c["response"] for c in r["completions"]]
        scores = [mean_rating(c) for c in r["completions"]]
        keep = [i for i, s in enumerate(scores) if s is not None and comps[i].strip()]
        if len(keep) < 2:
            skipped["fewer than 2 rated completions"] += 1
            continue
        comps = [comps[i] for i in keep]
        scores = [scores[i] for i in keep]
        overall = [r["completions"][i]["overall_score"] for i in keep]
        if max(range(len(scores)), key=scores.__getitem__) != max(range(len(overall)), key=overall.__getitem__):
            disagree_with_overall += 1
        for strategy in pairs:
            p = binarize(r["instruction"], comps, scores, strategy=strategy, rng=rng)
            if p is None:
                skipped[f"{strategy}: all tied"] += 1
            elif validate_pair(p):
                skipped[f"{strategy}: {validate_pair(p)[0]}"] += 1
            else:
                pairs[strategy].append(p)
    report["uf_prompts_after_truthfulqa_removal"] = len(rows)
    report["uf_skipped"] = dict(skipped)
    report["uf_best_by_mean_rating_differs_from_best_by_overall_score"] = disagree_with_overall
    for strategy, ps in pairs.items():
        margins = Counter(round(p["score_chosen"] - p["score_rejected"], 2) for p in ps)
        small = sum(v for m, v in margins.items() if m <= 0.5)
        report[f"uf_{strategy}"] = {
            "pairs": len(ps),
            "margin_le_0.5": small,
            "length_bias_chars": length_bias(ps),
        }
    balanced = length_balanced_subset(pairs["best_vs_random"])
    report["uf_length_balanced"] = {"pairs": len(balanced), "length_bias_chars": length_bias(balanced)}
    # One pair per prompt, so a random split by row is also a split by prompt:
    # no prompt appears in both train and validation.
    final = pairs["best_vs_random"]
    n_val = max(1, len(final) // 20)
    for name, part in (("validation.jsonl", final[:n_val]), ("train.jsonl", final[n_val:])):
        with open(out / name, "w", encoding="utf-8") as f:
            for p in part:
                f.write(json.dumps({k: p[k] for k in ("prompt", "chosen", "rejected")}, ensure_ascii=False) + "\n")
    report["uf_written"] = {"train": len(final) - n_val, "validation": n_val}


def part2_helpsteer3(n: int, report: dict) -> None:
    rows = stream("nvidia/HelpSteer3", n, config="preference")
    sign = lambda s: "A" if s < 0 else ("B" if s > 0 else "tie")   # negative = response 1 better
    a1, a2 = [], []
    n_annot = Counter()
    for r in rows:
        prefs = r["individual_preference"]
        n_annot[len(prefs)] += 1
        if len(prefs) >= 2:
            a1.append(sign(prefs[0]["score"]))
            a2.append(sign(prefs[1]["score"]))
    report["hs3_annotators_per_row"] = dict(sorted(n_annot.items()))
    report["hs3_overall_preference"] = dict(Counter(int(r["overall_preference"]) for r in rows).most_common())
    report["hs3_two_annotator_rows"] = len(a1)
    report["hs3_percent_agreement"] = round(percent_agreement(a1, a2), 3)
    report["hs3_cohens_kappa"] = round(cohens_kappa(a1, a2), 3)
    report["hs3_label_frequencies"] = dict(Counter(a1 + a2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="data/pref")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    part1_ultrafeedback(args.n, out, report)
    part2_helpsteer3(args.n, report)
    (out / "data_card.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
