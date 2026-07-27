"""Caption sidecars are user files — reading them must never crash a stage.

Every sidecar read used strict `encoding="utf-8"`. Captions written by another tool
(or a Windows text editor) are often cp1252, which raised a raw UnicodeDecodeError
and killed the whole stage; a UTF-8 BOM was worse, silently prefixing "\\ufeff" to
the trigger word so every caption in the dataset was subtly wrong.
"""

from __future__ import annotations

from pathlib import Path

from studio.config import read_caption


def _sidecar(tmp_path: Path, data: bytes) -> Path:
    img = tmp_path / "01.png"
    img.write_bytes(b"not-a-real-image")
    img.with_suffix(".txt").write_bytes(data)
    return img


def test_reads_plain_utf8(tmp_path: Path) -> None:
    img = _sidecar(tmp_path, "trig, a café at dusk".encode("utf-8"))
    assert read_caption(img) == "trig, a café at dusk"


def test_strips_a_utf8_bom_instead_of_corrupting_the_trigger(tmp_path: Path) -> None:
    """A BOM used to end up inside the first word: "\\ufefftrig"."""
    img = _sidecar(tmp_path, b"\xef\xbb\xbf" + b"trig, standing")
    caption = read_caption(img)
    assert caption == "trig, standing"
    assert "﻿" not in caption


def test_does_not_raise_on_a_cp1252_file(tmp_path: Path) -> None:
    """0x92 is a smart quote in cp1252 and invalid UTF-8."""
    img = _sidecar(tmp_path, b"trig, the dog\x92s collar")
    caption = read_caption(img)
    assert caption.startswith("trig, the dog")


def test_missing_sidecar_is_empty_not_an_error(tmp_path: Path) -> None:
    img = tmp_path / "01.png"
    img.write_bytes(b"x")
    assert read_caption(img) == ""


def test_whitespace_is_stripped(tmp_path: Path) -> None:
    img = _sidecar(tmp_path, b"  trig, standing  \r\n\r\n")
    assert read_caption(img) == "trig, standing"


def test_accepts_the_txt_path_directly(tmp_path: Path) -> None:
    """Callers hold either the image or its sidecar; both must work."""
    img = _sidecar(tmp_path, b"trig, standing")
    assert read_caption(img.with_suffix(".txt")) == "trig, standing"


def test_a_blank_sidecar_reads_as_empty(tmp_path: Path) -> None:
    img = _sidecar(tmp_path, b"   \n  \n")
    assert read_caption(img) == ""


# --- the stages that read sidecars all survive a cp1252 file ---------------

def test_export_resolution_survives_a_cp1252_caption(tmp_path: Path) -> None:
    from studio.package import resolve_export_items

    img = _sidecar(tmp_path, b"trig, the dog\x92s collar")
    res = resolve_export_items([img])
    assert len(res.items) == 1 and not res.empties and not res.missing


def test_caption_lint_survives_a_cp1252_caption(tmp_path: Path) -> None:
    from studio.caption_lint import analyze_folder

    _sidecar(tmp_path, b"trig, the dog\x92s collar")
    report, _ = analyze_folder(tmp_path, trigger="trig")
    assert report is not None


def test_skip_existing_check_survives_a_cp1252_caption(tmp_path: Path) -> None:
    from studio.captioner import _has_caption

    img = _sidecar(tmp_path, b"trig, the dog\x92s collar")
    assert _has_caption(img)
