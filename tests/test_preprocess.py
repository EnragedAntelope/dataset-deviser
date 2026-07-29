"""Tests for the preprocess stage (I/O-light paths: no restore, no isolation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from studio.preprocess import preprocess


def _img(path: Path, size: tuple[int, int] = (64, 48),
         color: tuple[int, int, int] = (120, 120, 120)) -> None:
    Image.new("RGB", size, color).save(path)


def test_preprocess_same_stem_different_ext_no_clobber(tmp_path: Path) -> None:
    """cat.jpg + cat.png both map to `cat_prepped.png` naively; the second must
    not silently overwrite the first — both images have to survive."""
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    a, b = src / "cat.jpg", src / "cat.png"
    _img(a, color=(200, 50, 50))
    _img(b, color=(50, 50, 200))

    r1 = preprocess(a, out, target=32, force_restore=False, isolate=False)
    r2 = preprocess(b, out, target=32, force_restore=False, isolate=False)

    assert r1.output != r2.output
    assert r1.output.exists() and r2.output.exists()
    assert r1.output.name == "cat_prepped.png"
    assert r2.output.name == "cat_prepped_2.png"
    assert {p.name for p in out.glob("*.png")} == {"cat_prepped.png", "cat_prepped_2.png"}


def test_preprocess_alpha_cutout_skips_rgb_flatten(tmp_path: Path, monkeypatch) -> None:
    """alpha_cutout=True must not flatten the isolated RGBA output to white RGB."""
    import studio.preprocess as pp

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "cat.png"
    _img(src)
    out = tmp_path / "out"

    def fake_isolate(image_path, out_path, subject_prompt="character", exclude_prompt="",
                      backend="", progress=None, front=False, alpha_cutout=False):
        assert alpha_cutout is True
        w, h = Image.open(image_path).size
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 3] = 128
        Image.fromarray(rgba, "RGBA").save(out_path, "PNG")
        return out_path

    monkeypatch.setattr(pp, "isolate_subject", fake_isolate)
    r = pp.preprocess(src, out, target=32, force_restore=False, isolate=True,
                      alpha_cutout=True)
    assert Image.open(r.output).mode == "RGBA"


def test_preprocess_alpha_cutout_ignored_without_isolate(tmp_path: Path) -> None:
    """alpha_cutout only means anything alongside isolation; without it, RGB as before."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "cat.png"
    _img(src)
    out = tmp_path / "out"

    r = preprocess(src, out, target=32, force_restore=False, isolate=False,
                  alpha_cutout=True)
    assert Image.open(r.output).mode == "RGB"
