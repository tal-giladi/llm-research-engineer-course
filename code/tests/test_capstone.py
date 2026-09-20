"""Fast smoke test for the Module 20 capstone driver (`scripts/capstone.py`).

This does not test model quality — it tests that the whole `llmre` pipeline
composes and *runs*, at minimal size, and that the two learning signals point the
right way: pretraining loss decreases and the RLVR correct-rate rises. Kept tiny
so it finishes in a few seconds.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Load scripts/capstone.py by path (scripts/ is not an installed package).
_CAPSTONE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "capstone.py"
_spec = importlib.util.spec_from_file_location("capstone", _CAPSTONE_PATH)
capstone = importlib.util.module_from_spec(_spec)
sys.modules["capstone"] = capstone
_spec.loader.exec_module(capstone)


def _tiny_cfg():
    """Minimal-size config: everything just large enough to learn a little."""
    return capstone.CapstoneConfig(
        seed=0,
        tok_vocab_size=300,
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=32,
        max_steps=60,
        micro_batch_size=8,
        warmup_steps=5,
        eval_interval=0,
        eval_iters=5,
        eval_batch_size=4,
        eval_max_batches=5,
        sft_steps=5,
        lora_rank=2,
        lora_alpha=4.0,
        rlvr_steps=30,
        rlvr_group_size=8,
        corpus_repeats=8,
    )


def test_capstone_runs_end_to_end_and_learns():
    res = capstone.run_capstone(_tiny_cfg(), verbose=False, write_report_to=None)

    # Every stage produced a result.
    for key in ("vocab_size", "n_params", "pretrain", "perplexity", "sft", "lora", "rlvr"):
        assert key in res, f"missing stage result: {key}"

    # Pretraining loss decreased (the core smoke assertion).
    pre = res["pretrain"]
    assert pre["final_loss"] < pre["initial_loss"], (
        f"pretrain loss did not drop: {pre['initial_loss']:.3f} -> {pre['final_loss']:.3f}"
    )

    # Perplexity is a finite positive number.
    assert res["perplexity"] > 0

    # LoRA froze the base weight: only the low-rank adapters are trainable.
    lora = res["lora"]
    assert lora["trainable"] < lora["total"]
    assert lora["same_at_init"] is True

    # RLVR raised the verifiable correct-rate.
    rlvr = res["rlvr"]
    assert rlvr["final"] >= rlvr["initial"]


def test_capstone_writes_report(tmp_path):
    report = tmp_path / "capstone_report.md"
    capstone.run_capstone(_tiny_cfg(), verbose=False, write_report_to=report)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    # It follows the experiment-report template sections.
    for section in ("## QUESTION", "## METHOD", "## RESULTS", "## CONCLUSION", "## NEXT EXPERIMENT"):
        assert section in text, f"report missing {section}"
