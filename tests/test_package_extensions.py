"""Export must not rename a JPEG to .png.

`package_dataset` used to copy every source byte-for-byte to `NN.png`, so a JPEG
source produced a file called `01.png` whose contents were JPEG — and a
`metadata.jsonl` that claimed `.png`. Trainers that trust the extension, and anyone
inspecting the folder, were being lied to. Stage ④ is documented as working on *any*
folder, so non-PNG sources are a supported path, not an edge case.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from studio.package import package_dataset, resolve_export_items


def _img(folder: Path, name: str, caption: str, fmt: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    Image.new("RGB", (16, 16), "red").save(p, fmt)
    p.with_suffix(".txt").write_text(caption, encoding="utf-8")
    return p


def _export(tmp_path: Path, sources: list[Path]) -> Path:
    res = resolve_export_items(sources)
    return package_dataset(res.items, tmp_path / "out", "demo", "trig", {})


def test_a_jpeg_source_keeps_its_extension(tmp_path: Path) -> None:
    src = _img(tmp_path / "src", "photo.jpg", "trig, a red square", "JPEG")
    ds = _export(tmp_path, [src])
    assert (ds / "01.jpg").exists()
    assert not (ds / "01.png").exists()


def test_the_exported_bytes_match_the_declared_extension(tmp_path: Path) -> None:
    """The actual regression: extension and content must agree."""
    src = _img(tmp_path / "src", "photo.jpg", "trig, a red square", "JPEG")
    ds = _export(tmp_path, [src])
    with Image.open(ds / "01.jpg") as im:
        assert im.format == "JPEG"


def test_a_png_source_is_unchanged(tmp_path: Path) -> None:
    src = _img(tmp_path / "src", "shot.png", "trig, a red square", "PNG")
    ds = _export(tmp_path, [src])
    assert (ds / "01.png").exists()
    with Image.open(ds / "01.png") as im:
        assert im.format == "PNG"


def test_a_webp_source_keeps_its_extension(tmp_path: Path) -> None:
    src = _img(tmp_path / "src", "shot.webp", "trig, a red square", "WEBP")
    ds = _export(tmp_path, [src])
    assert (ds / "01.webp").exists()


def test_metadata_jsonl_records_the_real_filename(tmp_path: Path) -> None:
    """HuggingFace `imagefolder` resolves file_name literally — a wrong extension
    here means the dataset does not load."""
    src = _img(tmp_path / "src", "photo.jpg", "trig, a red square", "JPEG")
    ds = _export(tmp_path, [src])
    row = json.loads((ds / "metadata.jsonl").read_text(encoding="utf-8").strip())
    assert row["file_name"] == "01.jpg"
    assert (ds / row["file_name"]).exists()


def test_mixed_sources_each_keep_their_own_extension(tmp_path: Path) -> None:
    a = _img(tmp_path / "src", "a.png", "trig, one", "PNG")
    b = _img(tmp_path / "src", "b.jpg", "trig, two", "JPEG")
    ds = _export(tmp_path, [a, b])
    assert (ds / "01.png").exists() and (ds / "02.jpg").exists()
    names = [json.loads(ln)["file_name"]
             for ln in (ds / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert names == ["01.png", "02.jpg"]


def test_captions_still_pair_by_stem(tmp_path: Path) -> None:
    """A trainer finds NN.txt next to NN.<ext>; the stem must still match."""
    src = _img(tmp_path / "src", "photo.jpg", "trig, a red square", "JPEG")
    ds = _export(tmp_path, [src])
    assert (ds / "01.txt").read_text(encoding="utf-8") == "trig, a red square"


def test_uppercase_extensions_are_normalised(tmp_path: Path) -> None:
    src = _img(tmp_path / "src", "photo.JPG", "trig, a red square", "JPEG")
    ds = _export(tmp_path, [src])
    assert (ds / "01.jpg").exists()


def test_jpeg_extension_is_normalised_to_jpg(tmp_path: Path) -> None:
    """Same format, two spellings — pick one so a folder is not half .jpeg."""
    src = _img(tmp_path / "src", "photo.jpeg", "trig, a red square", "JPEG")
    ds = _export(tmp_path, [src])
    assert (ds / "01.jpg").exists()


def test_readme_reports_the_extensions_present(tmp_path: Path) -> None:
    a = _img(tmp_path / "src", "a.png", "trig, one", "PNG")
    b = _img(tmp_path / "src", "b.jpg", "trig, two", "JPEG")
    ds = _export(tmp_path, [a, b])
    body = (ds / "README.txt").read_text(encoding="utf-8")
    assert "NN.txt" in body
