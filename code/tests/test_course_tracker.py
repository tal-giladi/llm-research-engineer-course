"""Tests for the personal progress tracker (course.py at the repo root)."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def course(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSE_PROGRESS_FILE", str(tmp_path / "progress.json"))
    monkeypatch.setenv("COURSE_PROGRESS_MD", str(tmp_path / "PROGRESS.md"))
    spec = importlib.util.spec_from_file_location("course_tracker", ROOT / "course.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.tmp = tmp_path
    return mod


def state(course):
    return json.loads((course.tmp / "progress.json").read_text(encoding="utf-8"))


def test_sidebar_parses_every_lesson(course):
    modules, lessons = course.load_lessons()
    assert "00.1" in lessons and "20.1" in lessons and "F.1" in lessons
    assert lessons["00.2"]["path"] == "lessons/module-00/lesson-02.md"
    assert all((ROOT / l["path"]).exists() for l in lessons.values())
    assert sum(len(m["lessons"]) for m in modules) == len(lessons)


def test_id_normalization(course):
    _, lessons = course.load_lessons()
    assert course.normalize_id("5.2", lessons) == "05.2"
    assert course.normalize_id("20", lessons) == "20.1"
    assert course.normalize_id("f1", lessons) == "F.1"
    with pytest.raises(SystemExit):
        course.normalize_id("99.9", lessons)


def test_complete_and_next(course, capsys):
    course.main(["complete", "00.1"])
    assert state(course)["lessons"]["00.1"]["status"] == "completed"
    capsys.readouterr()
    course.main(["next"])
    assert "Next:     00.2" in capsys.readouterr().out
    assert "✅" in (course.tmp / "PROGRESS.md").read_text(encoding="utf-8")


def test_struggle_review_resolve(course, capsys):
    course.main(["struggle", "5.2", "why sqrt(d_k)?"])
    course.main(["quiz", "03.1", "1/4"])
    course.main(["read", "01.1"])
    capsys.readouterr()
    course.main(["review"])
    out = capsys.readouterr().out
    assert "01.1" in out and "03.1" in out and "05.2" in out and "sqrt(d_k)" in out
    course.main(["resolve", "05.2"])
    capsys.readouterr()
    course.main(["review"])
    assert "05.2" not in capsys.readouterr().out


def test_skip_reset_and_notes(course):
    course.main(["skip", "01.1", "--reason", "known"])
    course.main(["note", "01.1", "nice"])
    s = state(course)["lessons"]["01.1"]
    assert s["status"] == "skipped" and s["reason"] == "known" and s["notes"][0]["note"] == "nice"
    course.main(["reset", "01.1"])
    assert "01.1" not in state(course)["lessons"]
