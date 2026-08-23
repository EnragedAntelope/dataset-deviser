"""Tests for the Idiot LoRa Builder hand-off sidecar.

The file this writes is read by a *different* program, so the contract is its
schema, not ours: `{"ratings": {relative/posix/path: "good"|"needs_edit"}}`.
Its `ImageRating::from_str` maps anything else to "none", which would silently
drop the whole triage — so the value vocabulary is asserted, not assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from studio import handoff


def _dataset(folder: Path, n: int = 3) -> Path:
    """A packaged-looking dataset: flat NN.png + NN.txt, all of them clean.

    Seeded mid-grey noise, because the fixture has to clear every advisory check
    it is not testing: it is sharp (high Laplacian variance), mid-exposure and
    high-contrast, and each image's dHash is far from the others'. A gradient or
    a flat fill trips blur, contrast or the duplicate finder and quietly turns
    every fixture image into a "needs edit".
    """
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        rng = np.random.default_rng(seed=i)
        pixels = rng.integers(48, 208, size=(96, 96, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(folder / f"{i:02d}.png")
        (folder / f"{i:02d}.txt").write_text(f"caption {i}", encoding="utf-8")
    return folder


def _load(ds: Path) -> dict:
    return json.loads(handoff.ratings_path(ds).read_text(encoding="utf-8"))


# ---------- schema ----------

def test_sidecar_lands_where_the_other_app_looks(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "d")
    written = handoff.write_ilb_ratings(ds)
    assert written == ds / ".lora-studio" / "ratings.json"
    assert written.is_file()


def test_schema_is_exactly_what_its_serde_expects(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "d")
    handoff.write_ilb_ratings(ds)
    data = _load(ds)
    assert set(data) == {"ratings"}, "an extra top-level key would fail its deserializer"
    assert all(isinstance(k, str) and isinstance(v, str)
               for k, v in data["ratings"].items())


def test_values_stay_inside_the_known_rating_vocabulary(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "d")
    handoff.write_ilb_ratings(ds, {"02.png": ["blurry"]})
    assert set(_load(ds)["ratings"].values()) <= {handoff.GOOD, handoff.NEEDS_EDIT}


def test_keys_are_relative_and_forward_slashed(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "d")
    handoff.write_ilb_ratings(ds)
    keys = list(_load(ds)["ratings"])
    assert keys == ["01.png", "02.png", "03.png"]
    assert not any(Path(k).is_absolute() or "\\" in k or ".." in k for k in keys)


def test_every_exported_image_gets_a_rating(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "d", n=5)
    handoff.write_ilb_ratings(ds)
    assert len(_load(ds)["ratings"]) == 5


# ---------- triage ----------

def test_flagged_images_become_needs_edit_and_the_rest_good(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "d")
    handoff.write_ilb_ratings(ds, {"02.png": ["blurry", "dark"]})
    ratings = _load(ds)["ratings"]
    assert ratings["02.png"] == handoff.NEEDS_EDIT
    assert ratings["01.png"] == ratings["03.png"] == handoff.GOOD


def test_triage_flags_a_blurry_shot(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "d", n=1)
    # A uniform fill has no edges at all, so its Laplacian variance is 0.
    Image.new("RGB", (256, 256), (120, 120, 120)).save(ds / "01.png")
    assert "blurry" in handoff.triage(ds).get("01.png", [])


def test_triage_flags_the_copies_not_the_original(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "d", n=1)
    src = Image.open(ds / "01.png")
    for name in ("02.png", "03.png"):
        src.save(ds / name)  # byte-identical copies -> one near-duplicate group
    flagged = handoff.triage(ds)
    assert "near-duplicate" not in flagged.get("01.png", []), "the keeper was flagged"
    assert "near-duplicate" in flagged["02.png"]
    assert "near-duplicate" in flagged["03.png"]


def test_triage_survives_a_check_that_blows_up(tmp_path: Path, monkeypatch) -> None:
    """Advisory means advisory: a broken check costs its own findings, nothing more."""
    ds = _dataset(tmp_path / "d")
    import studio.quality as quality

    monkeypatch.setattr(quality, "is_blurry",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    flagged = handoff.triage(ds)
    assert not any("blurry" in reasons for reasons in flagged.values())
    # ...and the hand-off still completes.
    handoff.write_ilb_ratings(ds)
    assert set(_load(ds)["ratings"].values()) == {handoff.GOOD}


def test_triage_of_an_empty_folder_is_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert handoff.triage(empty) == {}


# ---------- never clobber ----------

def test_an_existing_ratings_file_is_left_alone(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "d")
    target = handoff.ratings_path(ds)
    target.parent.mkdir(parents=True)
    target.write_text('{"ratings": {"01.png": "bad"}}', encoding="utf-8")
    with pytest.raises(FileExistsError):
        handoff.write_ilb_ratings(ds)
    assert _load(ds)["ratings"] == {"01.png": "bad"}, "someone's triage was overwritten"


def test_a_missing_folder_is_an_error_not_a_stray_write(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        handoff.write_ilb_ratings(tmp_path / "nope")
    assert not (tmp_path / "nope").exists()


# ---------- containment ----------

def test_nothing_is_written_outside_the_dataset_folder(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "root" / "d")
    before = {p for p in tmp_path.rglob("*")}
    handoff.write_ilb_ratings(ds, handoff.triage(ds))
    new = {p for p in tmp_path.rglob("*")} - before
    assert new and all(ds in p.parents for p in new)


def test_no_temp_file_is_left_behind(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "d")
    handoff.write_ilb_ratings(ds)
    assert not list((ds / ".lora-studio").glob("*.tmp"))


# ---------- the one-liner both front ends print ----------

def test_prepare_handoff_reports_the_folder_and_the_count(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "d")
    line = handoff.prepare_handoff(ds)
    assert str(ds) in line
    assert handoff.ratings_path(ds).is_file()


def test_prepare_handoff_reports_a_skip_instead_of_raising(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "d")
    handoff.write_ilb_ratings(ds)
    line = handoff.prepare_handoff(ds)  # second time round
    assert "skipped" in line
    assert "untouched" in line


def test_prepare_handoff_reports_a_bad_folder_instead_of_raising(tmp_path: Path) -> None:
    line = handoff.prepare_handoff(tmp_path / "nope")
    assert line.startswith("⚠️")
