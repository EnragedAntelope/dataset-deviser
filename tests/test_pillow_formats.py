"""Pillow only decodes the formats the app accepts (studio/__init__.py)."""
from __future__ import annotations

import io

import pytest
from PIL import Image, UnidentifiedImageError

import studio  # noqa: F401  (applies the decoder allowlist)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "BMP", "TIFF"])
def test_accepted_formats_still_open(fmt: str) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, fmt)
    buf.seek(0)
    assert Image.open(buf).size == (4, 4)


@pytest.mark.parametrize("fmt", ["GIF", "PPM", "PCX", "TGA"])
def test_other_formats_are_refused(fmt: str) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, fmt)  # encoders still exist
    buf.seek(0)
    with pytest.raises(UnidentifiedImageError):
        Image.open(buf)


def test_eps_never_reaches_ghostscript() -> None:
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 4 4\nshowpage\n"
    with pytest.raises(UnidentifiedImageError):
        Image.open(io.BytesIO(eps))
