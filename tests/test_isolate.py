"""Tests for subject isolation mask algebra.

These never load SAM3 (gated, ~3.4 GB): `_segment` is monkeypatched with
synthetic masks so the *algorithm* is what gets tested.

The scenario mirrors the reference image that exposed the bug: a character with
a backpack the segmenter already keeps OUT of the subject mask, plus a prop that
genuinely overlaps it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from studio import isolate as iso


def test_split_terms_separates_comma_list() -> None:
    # SAM3 scores a prompt as ONE concept, so "backpack, walkie talkie" asks for
    # a single object that is both. Each term must get its own pass.
    assert iso.split_terms("backpack, walkie talkie") == ["backpack", "walkie talkie"]


def test_split_terms_ignores_blanks_and_whitespace() -> None:
    assert iso.split_terms("  backpack ,, walkie talkie ,  ") == ["backpack", "walkie talkie"]
    assert iso.split_terms("") == []


def test_split_terms_single_term_unchanged() -> None:
    assert iso.split_terms("microphone") == ["microphone"]


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    img = Image.new("RGB", (40, 40), (10, 20, 30))
    path = tmp_path / "scene.png"
    img.save(path)
    return path


def _masks() -> dict[str, np.ndarray]:
    """Subject occupies rows 0-19. The 'backpack' sits entirely outside it
    (rows 30-39) — what SAM3 actually does. The 'microphone' is genuinely merged
    into the subject (rows 15-19)."""
    subject = np.zeros((40, 40), dtype=bool)
    subject[0:20, :] = True

    backpack = np.zeros((40, 40), dtype=bool)
    backpack[30:40, :] = True

    microphone = np.zeros((40, 40), dtype=bool)
    microphone[15:20, 0:10] = True

    return {"character": subject, "backpack": backpack, "microphone": microphone}


@pytest.fixture
def fake_segment(monkeypatch: pytest.MonkeyPatch):
    lookup = _masks()

    def _fake(image, prompt, threshold=0.5):
        return lookup[prompt].copy()

    monkeypatch.setattr(iso, "_segment", _fake)
    return lookup


def _kept(path: Path) -> int:
    arr = np.asarray(Image.open(path).convert("RGB"))
    return int((~(arr == 255).all(axis=-1)).sum())


def test_prop_outside_subject_is_reported_not_subtracted(
    scene: Path, fake_segment, tmp_path: Path
) -> None:
    """The regression this whole change exists for.

    A prop the segmenter already excluded is composited away regardless, so
    subtracting it can only remove real subject pixels. Measured on the
    reference image: the old code destroyed 2.75% of the character for no gain.
    """
    log: list[str] = []
    out = tmp_path / "out.png"
    iso.isolate_builtin(scene, out, subject_prompt="character",
                        exclude_prompt="backpack", progress=log.append)

    # Subject fully intact: 20 rows x 40 cols.
    assert _kept(out) == 800
    assert any("already outside the subject mask" in m for m in log)


def test_merged_prop_is_subtracted(scene: Path, fake_segment, tmp_path: Path) -> None:
    """The case the exclusion feature was actually built for: a prop the
    segmenter merged INTO the subject still gets removed."""
    log: list[str] = []
    out = tmp_path / "out.png"
    iso.isolate_builtin(scene, out, subject_prompt="character",
                        exclude_prompt="microphone", progress=log.append)

    # 800 subject px minus the 5x10 merged microphone.
    assert _kept(out) == 800 - 50
    assert any("merged into" in m for m in log)


def test_exclude_terms_are_unioned_not_scored_as_one(
    scene: Path, fake_segment, tmp_path: Path
) -> None:
    """A comma list must remove BOTH props, not a single fused concept."""
    out = tmp_path / "out.png"
    iso.isolate_builtin(scene, out, subject_prompt="character",
                        exclude_prompt="backpack, microphone")
    # backpack is outside the subject; microphone overlaps it and is removed.
    assert _kept(out) == 800 - 50


def test_missing_prop_reports_rather_than_silently_passing(
    scene: Path, fake_segment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookup = _masks()
    lookup["unicorn"] = np.zeros((40, 40), dtype=bool)
    monkeypatch.setattr(iso, "_segment", lambda i, p, threshold=0.5: lookup[p].copy())

    log: list[str] = []
    out = tmp_path / "out.png"
    iso.isolate_builtin(scene, out, subject_prompt="character",
                        exclude_prompt="unicorn", progress=log.append)
    assert _kept(out) == 800
    assert any("not found" in m for m in log)


def test_no_subject_raises_actionable_error(scene: Path, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(iso, "_segment",
                        lambda i, p, threshold=0.5: np.zeros((40, 40), dtype=bool))
    with pytest.raises(iso.IsolationError, match="adjust the subject prompt"):
        iso.isolate_builtin(scene, tmp_path / "out.png", subject_prompt="character")


def test_dilation_defaults_to_zero() -> None:
    """Growing the exclusion mask only ever eats the subject; it is opt-in."""
    from studio.config import settings

    assert settings.exclude_dilate_px == 0


def test_dilate_grows_when_explicitly_asked(scene: Path, fake_segment, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch still works for a genuinely merged prop."""
    from studio.config import settings

    monkeypatch.setattr(settings, "exclude_dilate_px", 2)
    out = tmp_path / "out.png"
    iso.isolate_builtin(scene, out, subject_prompt="character",
                        exclude_prompt="microphone")
    assert _kept(out) < 800 - 50  # dilation removed more than the raw mask


def test_alpha_cutout_produces_transparent_background(scene: Path, fake_segment,
                                                       tmp_path: Path) -> None:
    """alpha_cutout=True keeps subject alpha at 255 and zeroes it elsewhere."""
    out = tmp_path / "out.png"
    iso.isolate_builtin(scene, out, subject_prompt="character", alpha_cutout=True)
    img = Image.open(out)
    assert img.mode == "RGBA"
    alpha = np.asarray(img.getchannel("A"))
    assert (alpha[0:20, :] == 255).all()   # subject rows (see _masks())
    assert (alpha[20:40, :] == 0).all()    # background rows


def test_alpha_cutout_off_by_default_stays_rgb_white(scene: Path, fake_segment,
                                                      tmp_path: Path) -> None:
    out = tmp_path / "out.png"
    iso.isolate_builtin(scene, out, subject_prompt="character")
    assert Image.open(out).mode == "RGB"


def test_crop_to_content_uses_alpha_channel_when_rgba() -> None:
    """A transparent canvas has no 'non-white' pixels to crop by — must use alpha."""
    arr = np.zeros((40, 40, 4), dtype=np.uint8)
    arr[10:20, 5:15, :3] = 200
    arr[10:20, 5:15, 3] = 255
    img = Image.fromarray(arr, "RGBA")
    cropped = iso.crop_to_content(img, margin_frac=0.0)
    assert cropped.size == (10, 10)


def test_alpha_cutout_rejected_for_comfyui_backend(scene: Path, tmp_path: Path) -> None:
    """Never silently return white or ignore the flag — comfyui alpha isn't built."""
    with pytest.raises(iso.IsolationError, match="builtin"):
        iso.isolate_subject(scene, tmp_path / "out.png", backend="comfyui",
                            alpha_cutout=True)
