"""⚡ Quick build: one tab that runs ①→② then ③→④→⑤ with defaults. Every stage stubbed."""

from __future__ import annotations

from pathlib import Path

import pytest

import app as A
from studio import pipeline
from studio.preprocess import PreprocessReport
from studio.shotplan import default_plan


def _start(tmp_path: Path) -> pipeline.BuildStart:
    def png(name: str) -> Path:
        p = tmp_path / name
        p.write_bytes(b"x")
        return p

    big = PreprocessReport(source=tmp_path / "big.jpg", output=png("big_prepped.png"),
                           original_size=(3000, 4000), final_size=(768, 1024),
                           restored=False, reason="ok")
    small = PreprocessReport(source=tmp_path / "small.jpg", output=png("small_prepped.png"),
                             original_size=(580, 580), final_size=(1024, 1024),
                             restored=True, reason="upscaled")
    shots = default_plan()[:2]
    results = [pipeline.GenResult(shots[0], png("angle-front.png"), 1),
               pipeline.GenResult(shots[1], None, 2, error="refused")]
    return pipeline.BuildStart(tmp_path, [big, small], [big.output], results)


def test_trigger_comes_from_the_name_when_blank() -> None:
    assert A.quick_trigger("Sy Snootles", "") == "sysnootles"
    assert A.quick_trigger("Sy Snootles", " sy_s ") == "sy_s"


def test_an_upscaled_photo_starts_unticked_and_failures_are_named(tmp_path: Path) -> None:
    start = _start(tmp_path)
    rows, keep = A._quick_rows(start)
    assert [Path(v).name for _, v, _ in rows] == ["big_prepped.png", "small_prepped.png",
                                                 "angle-front.png"]
    assert [Path(k).name for k in keep] == ["big_prepped.png", "angle-front.png"]
    assert "upscaled ×1.8 from 580px" in rows[1][2]
    note = A._quick_note(start)
    assert "refused" in note and "Unticked: small.jpg" in note


def test_a_failed_isolation_is_named_in_the_note(tmp_path: Path) -> None:
    start = _start(tmp_path)
    start.reports[0].reference = start.reports[0].output
    start.reports[0].isolated = False
    assert "Not isolated:** big.jpg" in A._quick_note(start)


def test_build_derives_the_trigger_and_returns_the_picker(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_start(images, name, engine, **kw):
        seen.update(images=images, name=name, engine=engine, **kw)
        return _start(tmp_path)

    monkeypatch.setattr(pipeline, "build_start", fake_start)
    out = A.do_quick_build([str(tmp_path / "a.jpg")], "Ann Lee", "", "character",
                           "identity", "comfyui", " plate ", False, progress=lambda *a, **k: None)
    state, rows, gallery, boxes, note, log, trigger = out
    assert trigger == "annlee" and seen["exclude_prompt"] == "plate"
    assert seen["identity"] == "identity" and state.refs
    assert len(gallery) == 3 and boxes.value == [r for _, r, _ in rows if "big" in r
                                                 or "angle" in r]


def test_build_needs_images_and_a_name() -> None:
    import gradio as gr

    with pytest.raises(gr.Error, match="Drop"):
        A.do_quick_build([], "Ann", "", "character", "identity", "gemini", "", False)
    with pytest.raises(gr.Error, match="name"):
        A.do_quick_build(["x.png"], " ", "", "character", "identity", "gemini", "", False)


def test_finish_captions_tags_for_a_tag_model_and_writes_the_config(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_finish(start, images, **kw):
        seen.update(images=images, **kw)
        return tmp_path / "ds", "train --go"

    monkeypatch.setattr(pipeline, "build_finish", fake_finish)
    start = _start(tmp_path)
    keep = [str(start.reports[0].output)]
    note, _log, ds = A.do_quick_finish(start, keep, "Ann", "", "character", "identity",
                                       "comfyui", "qwen3vl", "kohya", "sdxl",
                                       progress=lambda *a, **k: None)
    assert seen["caption_style"] == "tags" and seen["trainer"] == "kohya"
    assert seen["trigger"] == "ann" and [p.name for p in seen["images"]] == ["big_prepped.png"]
    assert "train --go" in note and ds.endswith("ds")


def test_the_quick_tab_is_first_and_wired() -> None:
    names = [getattr(d.fn, "__name__", "") for d in A.demo.fns.values()]
    assert "do_quick_build" in names and "do_quick_finish" in names
    assert ("dd-gallery-quick", "dd-picks-quick", "dd-zoom-quick") in A.PICKER_IDS


def test_engine_note_prices_cloud_and_skips_style(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A.settings, "gemini_api_key", "k")
    assert "$" in A.quick_engine_note("gemini", "character")
    assert "generate nothing" in A.quick_engine_note("gemini", "style")
