"""Event-graph wiring for the 0.16.0 controls (Idiot LoRa Builder hand-off).

Same reason as `test_ui_wiring_0_15_1.py`: Gradio resolves handlers at build
time, so a checkbox added to the layout but left out of the click's input list
fails *silently* — the box does nothing and the export looks fine. These read
`app.demo`'s real dependency list.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import app as A
from studio import handoff


def _export_inputs() -> list[str]:
    """Labels of the inputs wired to ④'s export button, in order."""
    for dep in A.demo.fns.values():
        fn = getattr(dep, "fn", None)
        if getattr(fn, "__name__", "") == "do_export":
            return [getattr(i, "label", "") for i in dep.inputs]
    raise AssertionError("do_export is not wired to anything")


def _captioned_folder(folder: Path, n: int = 3) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        rng = np.random.default_rng(seed=i)
        Image.fromarray(rng.integers(48, 208, size=(96, 96, 3), dtype=np.uint8)).save(
            folder / f"src{i}.png")
        (folder / f"src{i}.txt").write_text(f"a caption {i}", encoding="utf-8")
    return folder


def test_the_handoff_checkbox_is_actually_wired() -> None:
    labels = _export_inputs()
    assert "Prepare for Idiot LoRa Builder" in labels, \
        "the hand-off checkbox is in the layout but reaches do_export as nothing"


def test_handoff_checkbox_is_the_last_input_matching_the_signature() -> None:
    """`ilb_handoff` is do_export's last parameter, so it must be the last input."""
    assert _export_inputs()[-1] == "Prepare for Idiot LoRa Builder"


def test_export_with_the_box_ticked_writes_the_sidecar(tmp_path: Path) -> None:
    src = _captioned_folder(tmp_path / "src")
    picks = [str(p) for p in sorted(src.glob("*.png"))]
    result, ds_dir, _hf = A.do_export(picks, "Sy", "sysnootles", str(tmp_path / "out"),
                                      ilb_handoff=True)
    assert "Idiot LoRa Builder" in result
    data = json.loads(handoff.ratings_path(Path(ds_dir)).read_text(encoding="utf-8"))
    assert len(data["ratings"]) == 3


def test_export_with_the_box_clear_writes_nothing_extra(tmp_path: Path) -> None:
    """Off by default means off: no sidecar, no mention of it in the result."""
    src = _captioned_folder(tmp_path / "src")
    picks = [str(p) for p in sorted(src.glob("*.png"))]
    result, ds_dir, _hf = A.do_export(picks, "Sy", "sysnootles", str(tmp_path / "out"))
    assert "Idiot LoRa Builder" not in result
    assert not (Path(ds_dir) / handoff.ILB_DIR).exists()
