"""Tests for dataset introspection and the derived training numbers."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from studio import dataset_stats as ds


def _make(folder: Path, sizes: list[tuple[int, int]], captions: int = 0) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, size in enumerate(sizes):
        p = folder / f"{i:02d}.png"
        Image.new("RGB", size, (128, 128, 128)).save(p)
        if i < captions:
            p.with_suffix(".txt").write_text("a caption", encoding="utf-8")
    return folder


def test_counts_images_and_captions(tmp_path: Path) -> None:
    folder = _make(tmp_path / "d", [(1024, 1024)] * 5, captions=3)
    stats = ds.inspect(folder)
    assert stats.n_images == 5
    assert stats.n_captioned == 3


def test_empty_folder(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    stats = ds.inspect(tmp_path / "empty")
    assert stats.n_images == 0
    assert "No images" in stats.summary()


def test_steps_scale_with_image_count(tmp_path: Path) -> None:
    # The bug being fixed: a flat 2000 steps regardless of dataset size.
    small = ds.inspect(_make(tmp_path / "s", [(1024, 1024)] * 24))
    large = ds.inspect(_make(tmp_path / "l", [(1024, 1024)] * 40))
    assert small.suggested_steps == 24 * ds.STEPS_PER_IMAGE
    assert large.suggested_steps == 40 * ds.STEPS_PER_IMAGE
    assert large.suggested_steps > small.suggested_steps


def test_steps_are_clamped_at_both_ends(tmp_path: Path) -> None:
    tiny = ds.inspect(_make(tmp_path / "t", [(1024, 1024)] * 2))
    huge = ds.inspect(_make(tmp_path / "h", [(1024, 1024)] * 200))
    assert tiny.suggested_steps == ds.MIN_STEPS
    assert huge.suggested_steps == ds.MAX_STEPS


def test_aspect_ratios_are_labelled(tmp_path: Path) -> None:
    folder = _make(tmp_path / "d", [(1024, 1024), (1024, 1024), (1536, 1024)])
    stats = ds.inspect(folder)
    assert stats.aspect_counts["square"] == 2
    assert stats.aspect_counts["3:2"] == 1


def test_long_side_range(tmp_path: Path) -> None:
    stats = ds.inspect(_make(tmp_path / "d", [(512, 512), (1536, 1024)]))
    assert stats.min_long_side == 512
    assert stats.max_long_side == 1536


def test_buckets_never_exceed_available_resolution(tmp_path: Path) -> None:
    # Bucketing above what the images contain would just upscale and invent detail.
    stats = ds.inspect(_make(tmp_path / "d", [(768, 768)] * 3))
    assert stats.buckets_for(1024) == [512, 768]


def test_buckets_include_requested_resolution(tmp_path: Path) -> None:
    stats = ds.inspect(_make(tmp_path / "d", [(2048, 2048)] * 3))
    assert stats.buckets_for(1024) == [512, 768, 1024]


def test_unreadable_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    folder = _make(tmp_path / "d", [(1024, 1024)] * 2)
    (folder / "broken.png").write_bytes(b"not an image")
    stats = ds.inspect(folder)
    assert stats.n_images == 2


def test_summary_flags_uncaptioned(tmp_path: Path) -> None:
    stats = ds.inspect(_make(tmp_path / "d", [(1024, 1024)] * 4, captions=1))
    assert "3 image(s) have no" in stats.summary()


# ---------- upscale advisory (0.16.0) ----------

def test_no_advisory_when_every_image_clears_the_resolution(tmp_path: Path) -> None:
    stats = ds.inspect(_make(tmp_path / "big", [(1024, 1024), (1536, 1024)]))
    assert stats.undersized(1024) == []
    assert stats.max_upscale(1024) == 1.0
    assert stats.upscale_note(1024) == "", "a clean dataset must not raise a false alarm"


def test_undersized_counts_by_long_side(tmp_path: Path) -> None:
    # 512x768 has a 768 long side: short of 1024, but not by its width.
    stats = ds.inspect(_make(tmp_path / "d", [(1024, 1024), (512, 768), (640, 480)]))
    assert len(stats.undersized(1024)) == 2
    assert stats.undersized(512) == []


def test_max_upscale_is_driven_by_the_smallest_image(tmp_path: Path) -> None:
    stats = ds.inspect(_make(tmp_path / "d", [(1024, 1024), (512, 512), (768, 768)]))
    assert stats.max_upscale(1024) == pytest.approx(2.0)


def test_upscale_note_names_the_count_and_the_factor(tmp_path: Path) -> None:
    stats = ds.inspect(_make(tmp_path / "d", [(1024, 1024)] * 3 + [(512, 512)]))
    note = stats.upscale_note(1024)
    assert "1 of 4" in note
    assert "2.0×" in note
    assert "bucket_no_upscale" in note


def test_upscale_advisory_is_silent_on_an_empty_dataset(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    stats = ds.inspect(tmp_path / "empty")
    assert stats.max_upscale(1024) == 1.0
    assert stats.upscale_note(1024) == ""


def test_the_train_tab_actually_shows_the_advisory(tmp_path: Path) -> None:
    """The note is only worth having if ⑤ prints it — assert through the handler."""
    import app as A

    folder = _make(tmp_path / "small", [(512, 512)] * 4, captions=4)
    out = A.do_generate_train_config(
        "ai-toolkit", "flux-dev", str(folder), "", "lora", "sks",
        1024, 16, 16, 2000, 1e-4, 1, True)
    assert "below 1024px" in out
    assert "2.0×" in out


def test_the_train_tab_stays_quiet_when_the_dataset_is_big_enough(tmp_path: Path) -> None:
    import app as A

    folder = _make(tmp_path / "big", [(1024, 1024)] * 4, captions=4)
    out = A.do_generate_train_config(
        "ai-toolkit", "flux-dev", str(folder), "", "lora", "sks",
        1024, 16, 16, 2000, 1e-4, 1, True)
    assert "below 1024px" not in out
