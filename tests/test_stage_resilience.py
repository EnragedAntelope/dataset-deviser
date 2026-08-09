"""Tests for ①'s partial-failure reporting in the UI layer.

`preprocess_sources` records a per-image failure instead of raising (see
`tests/test_preprocess.py`); these cover what the user is then shown. A skipped
image is only useful information if the note names it AND says what to do.
"""

from __future__ import annotations

from pathlib import Path

import app as APP
from studio.preprocess import PreprocessReport, failed_report


def _ok(name: str) -> PreprocessReport:
    return PreprocessReport(source=Path(f"src/{name}"), output=Path(f"out/{name}"),
                            original_size=(10, 10), final_size=(8, 8),
                            restored=False, reason="clean source, resize only")


# ---------- actionable hints ----------

def test_hint_for_no_subject_names_the_control_to_change() -> None:
    hint = APP._failure_hint("SAM3 found no 'character' in a.png — adjust the prompt")
    assert "Subject to keep" in hint
    assert "Isolate subject" in hint


def test_hint_for_gated_model_points_at_the_licence_and_token() -> None:
    hint = APP._failure_hint(
        "Could not load facebook/sam3 — it is a gated model: accept the license "
        "and authenticate (`hf auth login` or set HF_TOKEN).")
    assert "HF_TOKEN" in hint


def test_hint_for_comfyui_offers_the_local_fallback() -> None:
    hint = APP._failure_hint("ComfyUI is not reachable at http://127.0.0.1:8188")
    assert "Built-in" in hint or "Basic" in hint


def test_hint_for_corrupt_file_says_so() -> None:
    assert "corrupt" in APP._failure_hint("cannot identify image file 'x.png'")


def test_unknown_error_still_gets_a_hint() -> None:
    """Never return an empty hint — a bullet with nothing after the arrow reads
    as a rendering bug."""
    assert APP._failure_hint("something nobody predicted").strip()


# ---------- the result note ----------

def test_note_is_clean_when_nothing_failed() -> None:
    note = APP._preprocess_note([_ok("a.png"), _ok("b.png")], Path("out"), False)
    assert note.startswith("✅ 2 image(s)")
    assert "Skipped" not in note


def test_note_names_every_skipped_image_and_keeps_the_success_count() -> None:
    reports = [_ok("a.png"),
               failed_report(Path("src/b.png"), "SAM3 found no 'character' in b.png"),
               _ok("c.png")]
    note = APP._preprocess_note(reports, Path("out"), False)
    assert "✅ 2 image(s)" in note          # the successes are still reported
    assert "Skipped 1 of 3" in note         # with an honest denominator
    assert "b.png" in note                  # the failure is named
    assert "Subject to keep" in note        # and carries its hint


def test_note_for_alpha_cutout_still_explains_the_no_autofill_rule() -> None:
    note = APP._preprocess_note([_ok("a.png")], Path("out"), True)
    assert "transparent cutout" in note
    assert "not auto-filled" in note


def test_note_when_everything_failed_does_not_claim_a_success() -> None:
    """The all-fail head must not read '✅ 0 image(s) preprocessed into …' — that
    says a folder was produced when none was."""
    reports = [failed_report(Path("src/a.png"), "boom"),
               failed_report(Path("src/b.png"), "boom")]
    note = APP._preprocess_note(reports, Path("out"), False)
    assert note.startswith("❌")
    assert "✅" not in note
    assert "nothing was written" in note
    assert "a.png" in note and "b.png" in note


def test_note_when_everything_failed_ignores_alpha_cutout_wording() -> None:
    note = APP._preprocess_note([failed_report(Path("src/a.png"), "boom")],
                                Path("out"), True)
    assert "transparent cutout" not in note


# ---------- toasts are plain text, notes are markdown ----------

def test_plain_strips_the_markdown_a_toast_cannot_render() -> None:
    """gr.Warning/gr.Error popups are not markdown: '**bold**' and backticks
    appear literally in them."""
    assert APP._plain("adjust **Subject to keep** (try `person`)") == \
        "adjust Subject to keep (try person)"
