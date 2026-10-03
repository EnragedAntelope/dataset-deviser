"""Tests for shot-plan save/load round-tripping."""

from __future__ import annotations

from pathlib import Path

import pytest

from studio.plan_io import load_plan, save_plan
from studio.shotplan import default_plan


def test_round_trip_preserves_shots(tmp_path: Path) -> None:
    plan = default_plan()
    plan[0].outfit = "a blue jacket"  # exercise the new field
    saved = save_plan(plan, tmp_path / "my-plan")
    assert saved.suffix == ".yaml"
    loaded = load_plan(saved)
    assert [s.model_dump() for s in loaded] == [s.model_dump() for s in plan]


def test_save_adds_yaml_suffix(tmp_path: Path) -> None:
    saved = save_plan(default_plan(), tmp_path / "plan")
    assert saved.name == "plan.yaml"


def test_load_rejects_non_list(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("just: a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_plan(bad)


def test_loading_a_pre_0_17_plan_warns_about_sks_prompts() -> None:
    """Plans saved by 0.16 hold the old engine's `<sks>` grammar; the 2.1 engine
    would send it as literal text, so the load note has to say so."""
    import app
    from studio.config import settings

    old = default_plan()[:2]
    old[0].local_prompt = "<sks> front view eye-level shot medium shot"
    save_plan(old, settings.shot_plans_dir / "old")
    _, note = app.do_load_plan("old")
    assert "1 shot(s) still use the old" in note

    save_plan(default_plan()[:2], settings.shot_plans_dir / "new")
    assert "⚠️" not in app.do_load_plan("new")[1]
