"""Event-graph wiring for the 0.15.1 controls (shot style + inline editor).

Gradio resolves handlers at build time and mismatches fail *silently* at
runtime — a control wired to nothing simply does nothing, which is exactly how
a UI bug ships green. These tests read `app.demo`'s dependency list, so they
catch a control that was added to the layout but never connected.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import app as A
from studio import user_config
from studio.shot_style import CUSTOM, MATCH

# What gr.skip() evaluates to: 'change nothing about this output'.
SKIP = {"__type__": "update"}


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / ".cache"
    monkeypatch.setattr(user_config, "CACHE_DIR", cache)
    monkeypatch.setattr(user_config, "USER_SETTINGS_FILE", cache / "user_settings.json")


def _handlers() -> set[str]:
    """Names of every python function wired to an event in the Blocks."""
    names = set()
    for dep in A.demo.fns.values():
        fn = getattr(dep, "fn", None)
        if fn is not None:
            names.add(getattr(fn, "__name__", ""))
    return names


# ---------- shot style ----------

def test_style_controls_are_actually_wired() -> None:
    handlers = _handlers()
    assert "rebuild_plan_for_style" in handlers, "the shot-style dropdown is wired to nothing"
    assert "_toggle_style_text" in handlers, "the custom-style box never shows/hides"
    assert "preview_final_prompt" in handlers


def test_changing_the_style_rebuilds_the_plan_and_says_so() -> None:
    df, note = A.rebuild_plan_for_style("character", "Sy", "anime", "")
    assert len(df) == 24
    assert df["cloud_prompt"].str.contains("Rendered as an anime illustration").all()
    # Hand edits are lost on rebuild — silence there looks like a bug.
    assert "replaced" in note


def test_custom_style_without_text_warns_instead_of_silently_matching() -> None:
    _df, note = A.rebuild_plan_for_style("character", "Sy", CUSTOM, "  ")
    assert note.startswith("⚠️")
    assert "matching the reference" in note


def test_style_change_is_a_no_op_for_style_datasets() -> None:
    """Style never generates, so there is no table to rebuild."""
    df, note = A.rebuild_plan_for_style("style", "", "anime", "")
    assert df == SKIP and note == SKIP


def test_custom_textbox_visibility_follows_the_dropdown() -> None:
    assert A._toggle_style_text(CUSTOM).constructor_args["visible"] is True
    assert A._toggle_style_text("anime").constructor_args["visible"] is False


def test_style_is_remembered_between_launches() -> None:
    A.rebuild_plan_for_style("character", "Sy", "anime", "")
    assert user_config.get_shot_style() == ("anime", "")


def test_dataset_type_handler_still_returns_the_wired_arity() -> None:
    """The style controls are inputs, not outputs — adding them to the outputs
    list would silently mismatch Gradio's expectations."""
    updates = A.on_dataset_type_change("character", "", MATCH, "")
    assert len(updates) == 17


# ---------- ② final-prompt preview ----------

def test_preview_shows_what_the_engine_actually_receives() -> None:
    df = A.refresh_plan("Sy", "character", "anime", "")
    out = A.preview_final_prompt(df, "gemini", True)
    assert "Rendered as an anime illustration" in out
    # Prop exclusion is folded in at generation time, so it must show here too.
    assert "do not include any backpacks" in out

    local = A.preview_final_prompt(df, "comfyui", False)
    assert "Local (ComfyUI" in local


def test_preview_on_an_empty_plan_is_a_friendly_error() -> None:
    import gradio as gr

    df = A.refresh_plan("", "style")
    with pytest.raises(gr.Error):
        A.preview_final_prompt(df, "gemini", False)


# ---------- ③ inline editor ----------

def test_editor_navigation_is_wired() -> None:
    handlers = _handlers()
    assert "editor_prev" in handlers, "◀ Prev is not connected"
    assert "editor_next" in handlers, "Next ▶ is not connected"
    assert "save_and_next" in handlers
    assert "_editor_context" in handlers, "the preview/position never update"
    assert "_editor_relabel" in handlers


def _folder(tmp_path: Path, n: int = 3) -> tuple[str, list[str]]:
    for i in range(n):
        Image.new("RGB", (16, 16), (10 * i, 20, 30)).save(tmp_path / f"img{i}.png")
    return str(tmp_path), [f"img{i}.png" for i in range(n)]


def test_editor_choices_marks_captioned_files(tmp_path: Path) -> None:
    folder, names = _folder(tmp_path)
    (tmp_path / "img1.txt").write_text("a caption", encoding="utf-8")
    dd, listed = A._editor_choices(folder)
    assert listed == names
    labels = dict((v, lbl) for lbl, v in dd.constructor_args["choices"])
    # Trailing, not leading: Gradio puts its own selection tick first, so a
    # leading ✓ rendered the selected+captioned row as "✓ ✓ img1.png".
    assert labels["img1.png"] == "img1.png ✓"
    assert labels["img0.png"] == "img0.png"


def test_editor_step_moves_and_clamps(tmp_path: Path) -> None:
    folder, names = _folder(tmp_path)
    nxt = A._editor_step(folder, "img0.png", names, 1)
    assert nxt["value"] == "img1.png"
    prev = A._editor_step(folder, "img2.png", names, -1)
    assert prev["value"] == "img1.png"
    # Clamps rather than wrapping — looping back to image 1 loses your place.
    assert A._editor_step(folder, "img0.png", names, -1) == SKIP
    assert A._editor_step(folder, "img2.png", names, 1) == SKIP


def test_editor_context_gives_the_image_and_position(tmp_path: Path) -> None:
    folder, names = _folder(tmp_path)
    img, pos = A._editor_context(folder, "img1.png", names)
    assert img.endswith("img1.png")
    assert "2 / 3" in pos
    # Nothing selected must not raise — the dropdown fires .change with None.
    assert A._editor_context(folder, "", names) == (None, "")
    assert A._editor_context("", "img1.png", names) == (None, "")


def test_save_and_next_writes_then_advances(tmp_path: Path) -> None:
    folder, names = _folder(tmp_path)
    note, dd = A.save_and_next(folder, "img0.png", " a caption ", names)
    assert (tmp_path / "img0.txt").read_text(encoding="utf-8") == "a caption"
    assert "Saved caption" in note
    assert dd.constructor_args["value"] == "img1.png"
    # The just-saved file is re-marked as done in the picker.
    labels = dict((v, lbl) for lbl, v in dd.constructor_args["choices"])
    assert labels["img0.png"].endswith("✓")


def test_save_and_next_on_the_last_image_stays_put(tmp_path: Path) -> None:
    folder, names = _folder(tmp_path)
    _note, dd = A.save_and_next(folder, "img2.png", "last one", names)
    assert dd.constructor_args["value"] == "img2.png"


def test_loading_a_caption_with_no_selection_is_silent(tmp_path: Path) -> None:
    """Wired to .change, which fires with None whenever the folder reloads."""
    folder, _ = _folder(tmp_path)
    assert A.load_one_caption(folder, "") == ""
    assert A.load_one_caption("", "img0.png") == ""


# ---------- stop + doctor ----------

def test_stop_and_doctor_are_wired() -> None:
    handlers = _handlers()
    assert "request_stop" in handlers
    assert "run_doctor" in handlers


def test_stop_button_bypasses_the_queue() -> None:
    """Queued, the click would wait behind the job it is meant to interrupt."""
    stop_deps = [d for d in A.demo.fns.values()
                 if getattr(getattr(d, "fn", None), "__name__", "") == "request_stop"]
    assert stop_deps, "Stop is not wired"
    assert all(getattr(d, "queue", True) is False for d in stop_deps)
