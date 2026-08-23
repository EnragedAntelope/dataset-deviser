"""One name box and one trigger box, read directly by ②/③/④/⑤ (0.16.0).

The bug this replaces: each tab kept its own copy, carried forward by
`_fill_if_empty` — "copy into the next tab IF that box is still blank". The
first dataset of a session seeded ④/⑤ and every later one silently kept the old
value, so a user exported a dataset stamped with a name and trigger from several
runs earlier and only noticed at the end.

Mirroring the copies live was tried first and is worse — Gradio fires `.input`
per keystroke and the unqueued responses land out of order, so the copies settle
on a *prefix* of what was typed (seen in a browser, not in a unit test). The fix
is to delete the copies: one box cannot disagree with itself.

Mis-wiring is silent in Gradio (a handler reading the wrong component just gets
someone else's value), so these read `app.demo`'s real dependency graph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import app as A

NAME_ID = "dd-name-project"
TRIGGER_ID = "dd-trigger-project"
# ⑤'s LoRA name is the trained file's name, not the subject's — deliberately
# still its own box, defaulting to the header name when left blank.
TRAIN_NAME_ID = "dd-name-train"


def _by_id(elem_id: str):
    for c in A.demo.blocks.values():
        if getattr(c, "elem_id", "") == elem_id:
            return c
    raise AssertionError(f"no component with elem_id {elem_id}")


def _handler(fn_name: str):
    for dep in A.demo.fns.values():
        if getattr(getattr(dep, "fn", None), "__name__", "") == fn_name:
            return dep
    raise AssertionError(f"{fn_name} is not wired to anything")


def _input_ids(fn_name: str) -> list[str]:
    return [getattr(c, "elem_id", "") or getattr(c, "label", "")
            for c in _handler(fn_name).inputs]


# ---------- there is exactly one of each ----------

def test_only_one_name_box_and_one_trigger_box_exist() -> None:
    """A second copy is how the value went stale; a test is what keeps it gone."""
    ids = [getattr(c, "elem_id", "") for c in A.demo.blocks.values()]
    assert ids.count(NAME_ID) == 1
    assert ids.count(TRIGGER_ID) == 1
    assert ids.count(TRAIN_NAME_ID) == 1
    strays = [i for i in ids
              if i and i.startswith(("dd-name-", "dd-trigger-"))
              and i not in {NAME_ID, TRIGGER_ID, TRAIN_NAME_ID}]
    assert not strays, f"per-tab identity boxes are back: {strays}"


def test_nothing_carries_a_name_or_trigger_between_tabs_any_more() -> None:
    """`_fill_if_empty` WAS the staleness. It must not come back."""
    assert not hasattr(A, "_fill_if_empty")


# ---------- every stage reads the shared boxes ----------

@pytest.mark.parametrize("fn_name", ["do_caption", "do_test_caption", "do_export"])
def test_the_captioning_and_export_stages_read_the_shared_boxes(fn_name: str) -> None:
    ids = _input_ids(fn_name)
    assert NAME_ID in ids, f"{fn_name} does not read the shared name box"
    assert TRIGGER_ID in ids, f"{fn_name} does not read the shared trigger box"


def test_the_caption_lint_reads_the_shared_trigger() -> None:
    """It flags captions that don't start with the trigger — the wrong trigger
    would report every caption as broken."""
    assert TRIGGER_ID in _input_ids("do_analyze_captions")


@pytest.mark.parametrize("fn_name", ["refresh_plan", "rebuild_plan_for_style"])
def test_the_shot_plan_is_built_from_the_shared_name(fn_name: str) -> None:
    assert NAME_ID in _input_ids(fn_name)


def test_the_train_config_reads_the_shared_trigger_and_name() -> None:
    ids = _input_ids("do_generate_train_config")
    assert TRIGGER_ID in ids
    assert NAME_ID in ids, "⑤ can't fall back to the header name it never receives"
    assert TRAIN_NAME_ID in ids


def test_the_type_selector_relabels_the_shared_boxes() -> None:
    """Style/Concept datasets say "Style name"/"Concept name" — on the one box."""
    outs = _handler("on_dataset_type_change").outputs
    assert _by_id(NAME_ID) in outs
    assert _by_id(TRIGGER_ID) in outs
    for dtype, label in [("character", "Character name"), ("style", "Style name"),
                         ("concept", "Concept name")]:
        updates = A.on_dataset_type_change(dtype, "")
        assert len(updates) == len(A.type_outputs)
        assert updates[-2].constructor_args["label"] == label


# ---------- ⑤'s LoRA name is the one field that stays separate ----------

def _dataset(folder: Path, n: int = 2) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        Image.new("RGB", (512, 512), (60 + i, 90, 120)).save(folder / f"{i:02d}.png")
        (folder / f"{i:02d}.txt").write_text("sysnootles, a caption", encoding="utf-8")
    return folder


def _train(tmp_path: Path, lora_name: str, project: str) -> str:
    return A.do_generate_train_config(
        "ai-toolkit", "flux-dev", str(_dataset(tmp_path / "ds")), "", lora_name,
        "sysnootles", 512, 16, 16, 2000, 1e-4, 1, False, "character",
        A.shot_style.MATCH, "", project)


def test_a_blank_lora_name_follows_the_header_name(tmp_path: Path) -> None:
    """...slugified, because this one becomes a filename."""
    out = _train(tmp_path, "  ", "Sy Snootles")
    text = (tmp_path / "ds" / "ai-toolkit.yaml").read_text(encoding="utf-8")
    assert "sy-snootles" in text
    assert "Sy Snootles" not in text, "a space in a LoRA output filename"
    assert out


def test_a_typed_lora_name_wins(tmp_path: Path) -> None:
    """Someone who typed "…-v2" meant it — the header must not overwrite it."""
    _train(tmp_path, "sy-lora-v2", "sy-snootles")
    text = (tmp_path / "ds" / "ai-toolkit.yaml").read_text(encoding="utf-8")
    assert "sy-lora-v2" in text


# ---------- the receipt at ④ ----------

def _folder(folder: Path, n: int = 2) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    picks = []
    for i in range(1, n + 1):
        rng = np.random.default_rng(seed=i)
        p = folder / f"src{i}.png"
        Image.fromarray(rng.integers(48, 208, size=(64, 64, 3), dtype=np.uint8)).save(p)
        (folder / f"src{i}.txt").write_text(f"sysnootles, a caption {i}", encoding="utf-8")
        picks.append(str(p))
    return picks


def test_export_states_the_name_and_trigger_it_used(tmp_path: Path) -> None:
    picks = _folder(tmp_path / "src")
    result, _ds, _hf = A.do_export(picks, "Sy Snootles", "sysnootles",
                                   str(tmp_path / "out"))
    assert "Sy Snootles" in result
    assert "sysnootles" in result


def test_export_without_a_trigger_says_so(tmp_path: Path) -> None:
    picks = _folder(tmp_path / "src")
    result, _ds, _hf = A.do_export(picks, "Sy Snootles", "  ", str(tmp_path / "out"))
    assert "No trigger word" in result
