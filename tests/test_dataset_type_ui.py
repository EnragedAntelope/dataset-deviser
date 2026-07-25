"""Tests for the dataset-type wiring in the UI and CLI (Phase 2).

`app.py` builds a Gradio Blocks at import time but never launches it here; the
handlers under test are plain functions over strings/DataFrames, so no server,
network or model is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

import app
import cli
from studio import user_config


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the remembered dataset type out of the developer's real cache."""
    cache = tmp_path / ".cache"
    monkeypatch.setattr(user_config, "CACHE_DIR", cache)
    monkeypatch.setattr(user_config, "USER_SETTINGS_FILE", cache / "user_settings.json")


# ---------- ② plan selection ----------

def test_plan_subject_per_type() -> None:
    from studio.shotplan import plan_subject

    assert plan_subject("", "character") == "the character"
    assert plan_subject("Sy Snootles", "character") == "character Sy Snootles"
    assert plan_subject("", "concept") == "the object"
    assert plan_subject("brass compass", "concept") == "brass compass"


def test_concept_type_builds_the_concept_plan() -> None:
    character = app.refresh_plan("", "character")
    concept = app.refresh_plan("", "concept")
    assert len(character) == 24
    assert len(concept) == 18
    assert set(concept["kind"]) == {"angle", "framing", "context"}


def test_style_has_an_empty_plan_not_a_character_one() -> None:
    """Showing 24 character shots in a tab that can't generate them would be a
    lie; the table is empty (with headers intact) and ② explains why."""
    df = app.refresh_plan("", "style")
    assert len(df) == 0
    assert list(df.columns) == app.PLAN_COLUMNS


# ---------- the header selector retunes every dependent control ----------

def _updates(dataset_type: str) -> tuple:
    return app.on_dataset_type_change(dataset_type, "")


def test_change_handler_returns_one_update_per_wired_output() -> None:
    """Guard against the outputs list and the handler drifting apart — Gradio
    silently mismatches them otherwise."""
    assert len(_updates("character")) == len(app.type_outputs)


def test_character_keeps_generation_and_wardrobe() -> None:
    (isolate, subject, note, name, refresh, plan, gen, regen,
     outfits, outfits_clear, wardrobe, props, gen_subject,
     cap_name, trigger, sparse, exp_name) = _updates("character")
    assert isolate.constructor_args["value"] is True
    assert subject.constructor_args["value"] == "character"
    assert note.constructor_args["visible"] is False
    assert gen.constructor_args["interactive"] is True
    assert outfits.constructor_args["visible"] is True
    assert wardrobe.constructor_args["visible"] is True
    assert props.constructor_args["value"] is True
    assert sparse.constructor_args["visible"] is False
    assert len(plan) == 24
    assert "Character name" in cap_name.constructor_args["label"]
    assert exp_name.constructor_args["label"] == "Character name"
    assert "character" in refresh.constructor_args["value"].lower()
    assert "subject" in trigger.constructor_args["info"].lower()
    assert "Character name" in name.constructor_args["label"]
    assert gen_subject.constructor_args["value"] == "character"


def test_style_disables_generation_and_isolation() -> None:
    (isolate, _subject, note, _name, refresh, _plan, gen, regen,
     outfits, outfits_clear, wardrobe, _props, _gen_subject,
     cap_name, _trigger, sparse, exp_name) = _updates("style")
    assert isolate.constructor_args["value"] is False
    assert gen.constructor_args["interactive"] is False
    assert regen.constructor_args["interactive"] is False
    assert refresh.constructor_args["interactive"] is False
    assert outfits.constructor_args["visible"] is False
    assert outfits_clear.constructor_args["visible"] is False
    assert wardrobe.constructor_args["visible"] is False
    assert sparse.constructor_args["visible"] is True  # style-only toggle
    assert note.constructor_args["visible"] is True
    assert "Style name" in cap_name.constructor_args["label"]
    assert exp_name.constructor_args["label"] == "Style name"


def test_concept_generates_with_object_defaults() -> None:
    (isolate, subject, note, _name, _refresh, plan, gen, _regen,
     outfits, _outfits_clear, wardrobe, props, gen_subject,
     cap_name, _trigger, sparse, exp_name) = _updates("concept")
    assert isolate.constructor_args["value"] is True
    assert subject.constructor_args["value"] == "object"
    assert gen_subject.constructor_args["value"] == "object"
    assert gen.constructor_args["interactive"] is True
    assert len(plan) == 18
    # An object has no wardrobe, and the prop-exclusion clause is character-worded.
    assert outfits.constructor_args["visible"] is False
    assert wardrobe.constructor_args["visible"] is False
    assert props.constructor_args["value"] is False
    assert sparse.constructor_args["visible"] is False
    assert note.constructor_args["visible"] is True
    assert "Concept name" in cap_name.constructor_args["label"]
    assert exp_name.constructor_args["label"] == "Concept name"


def test_change_handler_remembers_the_type() -> None:
    app.on_dataset_type_change("concept", "")
    assert user_config.get_dataset_type() == "concept"


# ---------- remembered dataset type ----------

def test_dataset_type_round_trip() -> None:
    assert user_config.get_dataset_type() == "character"  # default when unset
    user_config.set_dataset_type("style")
    assert user_config.get_dataset_type() == "style"


def test_unknown_dataset_type_is_ignored_on_write_and_read() -> None:
    user_config.set_dataset_type("concept")
    user_config.set_dataset_type("portrait")  # rejected
    assert user_config.get_dataset_type() == "concept"
    user_config.save_user_config({"dataset_type": "portrait"})  # hand-edited file
    assert user_config.get_dataset_type() == "character"


# ---------- ⑤ metadata reconciliation ----------

def _dataset(tmp_path: Path, meta: dict | str) -> Path:
    ds = tmp_path / "ds"
    ds.mkdir(exist_ok=True)
    (ds / "metadata.json").write_text(
        meta if isinstance(meta, str) else json.dumps(meta), encoding="utf-8")
    return ds


def test_no_metadata_means_no_note(tmp_path: Path) -> None:
    (tmp_path / "bare").mkdir()
    assert app.dataset_type_note(tmp_path / "bare", "character") == ""


def test_matching_type_reports_without_warning(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, {"dataset_type": "style", "caption_style": "prose"})
    note = app.dataset_type_note(ds, "style")
    assert "**style** dataset" in note
    assert "prose" in note
    assert "⚠️" not in note


def test_mismatched_type_warns_and_names_both(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, {"dataset_type": "style"})
    note = app.dataset_type_note(ds, "character")
    assert "⚠️" in note
    assert "**style**" in note and "**character**" in note


def test_corrupt_or_typeless_metadata_is_silent(tmp_path: Path) -> None:
    assert app.dataset_type_note(_dataset(tmp_path, "{ not json"), "character") == ""
    assert app.dataset_type_note(_dataset(tmp_path, {"trigger": "x"}), "character") == ""
    assert app.dataset_type_note(_dataset(tmp_path, "[1, 2]"), "character") == ""


# ---------- CLI parity ----------

def test_cli_plan_selection_matches_the_ui() -> None:
    assert len(cli._plan_for("character", "")) == 24
    assert len(cli._plan_for("concept", "")) == 18


def test_cli_refuses_to_generate_a_style() -> None:
    with pytest.raises(typer.BadParameter, match="no synthetic generation"):
        cli._plan_for("style", "")


def test_cli_prop_default_follows_the_dataset_type() -> None:
    assert cli._props_default(None, "character") is True
    assert cli._props_default(None, "concept") is False
    # An explicit flag always wins.
    assert cli._props_default(True, "concept") is True
    assert cli._props_default(False, "character") is False


def test_cli_outfit_randomizer_is_character_only() -> None:
    from studio.shotplan import concept_plan

    plan = concept_plan()
    assert cli._dress(plan, "concept") is plan  # untouched
    dressed = cli._dress(cli._plan_for("character", ""), "character")
    assert any(s.outfit for s in dressed)
