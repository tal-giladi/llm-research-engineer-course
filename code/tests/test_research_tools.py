"""Tests for the curriculum modernization helper (research/tools/research.py)."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "research" / "tools" / "research.py"
_spec = importlib.util.spec_from_file_location("research_tool", _PATH)
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


def _candidate(cid="C-20260926-01", cls="A", discovered="2026-09-26", drop=()):
    lines = [f"### {cid} · Example technique", ""]
    for f in rt.CANDIDATE_FIELDS:
        if f in drop:
            continue
        value = {"Class": cls, "Date discovered": discovered}.get(f, "x")
        lines.append(f"- **{f}:** {value}")
    return "\n".join(lines) + "\n"


def test_complete_candidate_passes():
    assert rt.check_candidates(_candidate(), "t") == []


def test_missing_field_is_reported():
    errs = rt.check_candidates(_candidate(drop=("Evidence of adoption",)), "t")
    assert any("Evidence of adoption" in e for e in errs)


def test_only_a_or_b_get_full_records():
    assert any("Class must be A or B" in e for e in rt.check_candidates(_candidate(cls="C"), "t"))


def test_id_must_match_discovery_date():
    errs = rt.check_candidates(_candidate(discovered="2026-09-27"), "t")
    assert any("ID date" in e for e in errs)


def test_parse_multiple_candidates():
    text = _candidate("C-20260926-01") + "\n" + _candidate("C-20260926-02")
    assert [c["id"] for c in rt.parse_candidates(text)] == ["C-20260926-01", "C-20260926-02"]


def test_topic_status_must_match_folder(tmp_path):
    d = tmp_path / "deferred"
    d.mkdir()
    p = d / "foo.md"
    p.write_text(
        "# Foo\n\n- **Topic ID:** foo\n- **Status:** ADD\n- **Next review:** n/a\n"
        "- **Course change:** none\n- **Candidates:** C-20260926-01\n\n## History\n\n"
        "- 2026-09-26 — ADD — reason — weekly/2026-W39.md\n",
        encoding="utf-8",
    )
    t = rt.parse_topic(p)
    t["path"] = rt.RESEARCH / "deferred" / "foo.md"   # pretend it lives in the repo
    errs = rt.check_topic(t)
    assert any("Status ADD but file is in deferred/" in e for e in errs)


def test_repository_research_folder_is_valid():
    assert rt.cmd_check() == 0
