"""0.18.0: ① provenance, ④ held-out photos, and ⑤'s exposure line.

Provenance is a PNG text chunk ① writes, so a later stage can tell a real photo
(and how small it started) from a generated shot without any side file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import app as A
from studio import dataset_stats as ds
from studio.package import package_dataset
from studio.preprocess import preprocess


def _photo(path: Path, size: tuple[int, int], seed: int = 1) -> Path:
    rng = np.random.default_rng(seed=seed)
    Image.fromarray(rng.integers(48, 208, size=(size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


def _prep(tmp_path: Path, sizes: list[tuple[int, int]]) -> list:
    src = tmp_path / "src"
    src.mkdir()
    return [preprocess(_photo(src / f"p{i}.png", s, i), tmp_path / "prepped", target=256,
                       force_restore=False, isolate=False).output
            for i, s in enumerate(sizes, 1)]


def test_preprocess_records_the_source(tmp_path: Path) -> None:
    out = _prep(tmp_path, [(120, 80)])[0]
    with Image.open(out) as im:
        origin = ds.provenance(im)
    assert origin["name"] == "p1.png"
    assert (origin["w"], origin["h"]) == (120, 80)
    assert origin["restored"] is False and origin["isolated"] is False


def test_provenance_reads_from_the_header_without_decoding(tmp_path: Path) -> None:
    """The chunk sits before IDAT, so a truncated pixel stream still yields it."""
    out = _prep(tmp_path, [(120, 80)])[0]
    data = out.read_bytes()
    cut = tmp_path / "cut.png"
    cut.write_bytes(data[:data.index(b"IDAT") + 16])
    with Image.open(cut) as im:
        assert ds.provenance(im)["name"] == "p1.png"


def test_generated_and_corrupt_chunks_read_as_empty(tmp_path: Path) -> None:
    plain = _photo(tmp_path / "gen.png", (64, 64))
    with Image.open(plain) as im:
        assert ds.provenance(im) == {}
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text(ds.PROVENANCE_KEY, "{not json")
    bad = tmp_path / "bad.png"
    Image.new("RGB", (8, 8)).save(bad, pnginfo=info)
    with Image.open(bad) as im:
        assert ds.provenance(im) == {}


def test_upscaled_sources_are_named_at_train_time(tmp_path: Path) -> None:
    _prep(tmp_path, [(120, 80), (400, 300)])  # target 256: only the first grew
    stats = ds.inspect(tmp_path / "prepped")
    assert stats.upscaled_sources == [("p1.png", 120)]
    note = stats.upscale_note(256)
    assert "p1.png (120px)" in note and "p2.png" not in note


def test_export_carries_provenance_into_metadata(tmp_path: Path) -> None:
    prepped = _prep(tmp_path, [(120, 80)])
    gen = _photo(tmp_path / "prepped" / "shot.png", (256, 256), 9)
    items = [(prepped[0], "a photo of trg"), (gen, "trg, side view")]
    out = package_dataset(items, tmp_path / "out", "Sy", "trg", {})
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["provenance"] == [{"file": "01.png", **{
        "name": "p1.png", "w": 120, "h": 80, "captured": "",
        "restored": False, "isolated": False}}]
    with Image.open(out / "01.png") as im:  # copy2 kept the chunk
        assert ds.provenance(im)["name"] == "p1.png"


def test_held_out_photos_leave_the_training_folder(tmp_path: Path) -> None:
    prepped = _prep(tmp_path, [(300, 300), (900, 600), (500, 500)])
    gen = _photo(tmp_path / "prepped" / "shot.png", (256, 256), 9)
    items = [(p, f"caption {i}") for i, p in enumerate(prepped)] + [(gen, "shot")]
    out = package_dataset(items, tmp_path / "out", "Sy", "trg", {}, holdout=2)

    held = out.parent / f"{out.name}-heldout"
    # Largest originals first; numbered, with their captions alongside.
    assert sorted(p.name for p in held.iterdir()) == [
        "01-p2_prepped.png", "01-p2_prepped.txt", "02-p3_prepped.png", "02-p3_prepped.txt"]
    assert len(list(out.glob("*.png"))) == 2  # p1 + the generated shot
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["heldout"] == {"dir": str(held), "files": ["01-p2_prepped.png",
                                                           "02-p3_prepped.png"]}


def test_holdout_never_empties_the_dataset_or_takes_generated_shots(tmp_path: Path) -> None:
    prepped = _prep(tmp_path, [(300, 300)])
    out = package_dataset([(prepped[0], "c")], tmp_path / "out", "Sy", "trg", {}, holdout=3)
    assert not (out.parent / f"{out.name}-heldout").exists()
    gen = _photo(tmp_path / "g.png", (64, 64))
    out2 = package_dataset([(gen, "c"), (gen, "d")], tmp_path / "out2", "Sy", "trg", {},
                           holdout=1)
    assert not (out2.parent / f"{out2.name}-heldout").exists()


def test_a_stale_heldout_folder_is_never_reused(tmp_path: Path) -> None:
    (tmp_path / "out" / "sy-dataset-heldout").mkdir(parents=True)
    out = package_dataset([(_photo(tmp_path / "g.png", (64, 64)), "c")],
                          tmp_path / "out", "Sy", "trg", {})
    assert out.name == "sy-dataset-2"


def test_do_export_reports_the_holdout(tmp_path: Path) -> None:
    prepped = _prep(tmp_path, [(300, 300), (500, 500)])
    for p in prepped:
        p.with_suffix(".txt").write_text("a caption", encoding="utf-8")
    result, _ds, _hf = A.do_export([str(p) for p in prepped], "Sy", "trg",
                                   str(tmp_path / "out"), holdout=1)
    assert "Held out 1" in result


def test_optimizer_reaches_the_config_and_fizgig_ignores_prodigy(tmp_path: Path) -> None:
    folder = tmp_path / "d"
    folder.mkdir()
    for i in range(4):
        Image.new("RGB", (64, 64)).save(folder / f"{i:02d}.png")
    out = A.do_generate_train_config("kohya", "sdxl", str(folder), "", "lora", "trg",
                                     1024, 16, 16, 10, 1e-4, 1, False, optimizer="prodigy")
    assert "--optimizer_type Prodigy" in out and "pip install prodigyopt" in out
    out = A.do_generate_train_config("fizgig", "krea2", str(folder), "", "lora", "trg",
                                     1024, 8, 8, 30, 1e-4, 1, False, optimizer="prodigy")
    assert "Prodigy" not in out and "--optimizer_type adamw8bit" in out
    # The ⑤ dropdown is the last generate input and follows the trainer.
    deps = [d for d in A.demo.fns.values()
            if getattr(getattr(d, "fn", None), "__name__", "") == "do_generate_train_config"]
    assert deps[0].inputs[-1].label == "Optimizer"
    update = A.on_trainer_change("fizgig")[-1]
    assert update["choices"] == [("AdamW8bit (recommended)", "adamw8bit")]


def test_exposure_line_auto_repeats(tmp_path: Path) -> None:
    folder = tmp_path / "d"
    folder.mkdir()
    for i in range(20):
        Image.new("RGB", (64, 64)).save(folder / f"{i:02d}.png")
    # 20 x 75 = 1500 target steps; 16 epochs of 20 images -> ~5 repeats.
    line = A.exposure_line("ai-toolkit", str(folder), 16, 0, 1)
    assert "20 images × 5 repeats (auto)" in line and "**1600 steps**" in line
    assert "⚠️" not in line
    # Fizgig's guidance is repeats 1; a typed value always wins.
    assert "× 1 repeats (auto)" in A.exposure_line("fizgig", str(folder), 30, 0, 1)
    assert "× 2 repeats ÷" in A.exposure_line("fizgig", str(folder), 30, 2, 1)
    assert "Few steps" in A.exposure_line("ai-toolkit", str(folder), 4, 1, 4)
    assert "Few steps" not in A.exposure_line("fizgig", str(folder), 4, 1, 4)
    assert A.exposure_line("ai-toolkit", "", 16, 0, 1) == ""
