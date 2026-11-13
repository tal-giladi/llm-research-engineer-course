"""Lab for lesson 16.5: build an RL prompt set with verifiable answers.

Part 1  NuminaMath-1.5: which problems have an answer a program can check?
        Compares the dataset's own labels with llmre.rl.data.answer_is_verifiable
        and checks that math-verify can parse every kept gold answer.
Part 2  Skywork-OR1: the same problem is "too easy" for one model and "useful"
        for another. Uses the per-model difficulty that ships with the data.
Part 3  Decontaminate the kept prompts against MATH-500, AIME 2024 and AIME 2025.
Part 4  (optional, --policy) measure pass rates yourself with your own model.

    pip install "datasets==5.2.0" "math-verify==0.9.0"
    python scripts/lab_rl_data.py --n 5000 --out data/rl
    python scripts/lab_rl_data.py --n 0 --policy Qwen/Qwen2.5-0.5B-Instruct --k 8 --m 20   # part 4

Parts 1-3 run on CPU in a few minutes. Part 4 needs `transformers`; on a CPU
use a 0.5B model and a few problems, on a GPU use vLLM for real sizes.
Checked 2026-11-13 with datasets 5.2.0, math-verify 0.9.0, transformers 5.17.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import islice
from pathlib import Path

from llmre.rl.data import answer_is_verifiable, difficulty_filter, pass_rate
from llmre.sft.data import build_benchmark_index, contamination_score


def stream(name, n, split="train", config=None, shuffle=True):
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split, streaming=True)
    if shuffle:
        ds = ds.shuffle(seed=0, buffer_size=10_000)
    return list(islice(ds, n))


def mv_equal(gold: str, pred: str | None) -> float:
    """math-verify equality. Timeouts are off: math-verify implements them with
    signals (Unix) or subprocesses, which fail in some Windows sandboxes. In a
    long-running trainer, run the verifier in a worker with your own timeout."""
    from math_verify import parse, verify

    if pred is None:
        return 0.0
    g = parse(f"${gold}$", parsing_timeout=None)
    p = parse(f"${pred}$", parsing_timeout=None)
    return 1.0 if g and p and verify(g, p, timeout_seconds=None) else 0.0


def part1_numina(n, report):
    from math_verify import parse

    rows = stream("project-numina/NuminaMath-1.5", n)
    report["numina_question_type"] = dict(Counter(r["question_type"] for r in rows))
    report["numina_answer_special"] = dict(Counter(
        r["answer"] if r["answer"] in ("proof", "notfound", "") else "a value" for r in rows))
    kept, why = [], Counter()
    for r in rows:
        if r["question_type"] != "math-word-problem" or r["answer"] in ("proof", "notfound"):
            why["dataset label: " + (r["answer"] if r["answer"] in ("proof", "notfound") else r["question_type"])] += 1
            continue
        reasons = answer_is_verifiable(r["problem"], r["answer"])
        if reasons:
            why["our filter: " + reasons[0]] += 1
            continue
        if not parse(f"${r['answer']}$", parsing_timeout=None):
            why["math-verify cannot parse gold"] += 1
            continue
        if r.get("problem_is_valid") == "No" or r.get("solution_is_valid") == "No":
            why["dataset flags invalid"] += 1
            continue
        kept.append({"problem": r["problem"].strip(), "answer": r["answer"], "source": r["source"]})
    report["numina_kept"] = len(kept)
    report["numina_dropped"] = dict(why.most_common())
    return kept


def part2_skywork(n, report):
    rows = stream("Skywork/Skywork-OR1-RL-Data", n, split="math")
    out = {}
    models = ["DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-32B"]
    for m in models:
        # model_difficulty = wrong answers out of 16 samples (card: higher = harder).
        scored = [{"pass_rate": 1 - r["extra_info"]["model_difficulty"][m] / 16} for r in rows]
        _, counts = difficulty_filter(scored)
        out[m] = counts
    both = sum(1 for r in rows
               if r["extra_info"]["model_difficulty"][models[0]] == 16
               and 0 < r["extra_info"]["model_difficulty"][models[2]] < 16)
    report["skywork_rows"] = len(rows)
    report["skywork_filter_per_model"] = out
    report["skywork_too_hard_for_1.5B_but_useful_for_32B"] = both
    prompts = []
    for r in rows:
        prompt = r["prompt"][0]["content"]
        gold = json.loads(r["reward_model"]["ground_truth"])
        prompts.append({"problem": prompt, "answer": gold[0], "source": r["data_source"]})
    return prompts


def part3_decontaminate(problems, report):
    bench = [r["problem"] for r in stream("HuggingFaceH4/MATH-500", 10_000, split="test", shuffle=False)]
    bench += [r["problem"] for r in stream("HuggingFaceH4/aime_2024", 100, shuffle=False)]
    bench += [r["problem"] for r in stream("math-ai/aime25", 100, split="test", shuffle=False)]
    index = build_benchmark_index(bench, n=8)
    clean, hits = [], []
    for p in problems:
        (hits if contamination_score(p["problem"], index, 8) >= 0.5 else clean).append(p)
    report["decontam_benchmark_items"] = len(bench)
    report["decontam_removed"] = len(hits)
    report["decontam_removed_by_source"] = dict(Counter(h["source"] for h in hits).most_common())
    report["decontam_examples"] = [h["problem"][:160] for h in hits[:3]]
    return clean


def part4_policy(model_name, problems, k, report):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    instruction = "\nPlease reason step by step, and put your final answer within \\boxed{}."

    def policy(problem: str) -> str:
        msgs = [{"role": "user", "content": problem + instruction}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=512, do_sample=True, temperature=1.0, top_p=1.0)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    rows = []
    for p in problems:
        rate = pass_rate(p["problem"], p["answer"], policy, k=k, reward=lambda pred, gold: mv_equal(gold, pred))
        rows.append({**p, "pass_rate": rate})
        print(f"pass rate {rate:.2f}  {p['problem'][:70]!r}", flush=True)
    kept, counts = difficulty_filter(rows)
    report["policy"] = {"model": model_name, "k": k, "problems": len(rows), "filter": counts,
                        "pass_rates": [r["pass_rate"] for r in rows]}
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="rows to stream per dataset (0 = skip parts 1-3)")
    ap.add_argument("--out", default="data/rl")
    ap.add_argument("--policy", help="HF model id for part 4, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--k", type=int, default=8, help="samples per problem in part 4")
    ap.add_argument("--m", type=int, default=20, help="problems to score in part 4")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    problems = []
    if args.n:
        problems = part1_numina(args.n, report)
        problems = part3_decontaminate(problems, report)
        part2_skywork(args.n, report)
        with open(out / "rl_prompts.jsonl", "w", encoding="utf-8") as f:
            for p in problems:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        report["written"] = len(problems)
    if args.policy:
        pool = problems or stream("project-numina/NuminaMath-1.5", 2000)
        pool = [p for p in pool if p.get("question_type", "math-word-problem") == "math-word-problem"
                and p["answer"] not in ("proof", "notfound") and not answer_is_verifiable(p["problem"], p["answer"])]
        part4_policy(args.policy, pool[: args.m], args.k, report)
    (out / "data_card.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
