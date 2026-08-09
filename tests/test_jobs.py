"""Tests for cooperative stop across the three long-running stages.

Stopping must behave like a pause-and-keep, never like a failure: partial
results come back, nothing raises, and the work already done stays on disk.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from studio.jobs import JobControl, should_stop_now


def _img(path: Path) -> None:
    Image.new("RGB", (32, 32), (120, 120, 120)).save(path)


# ---------- the token ----------

def test_control_starts_unset_and_sets() -> None:
    c = JobControl()
    assert not c.stopped and not c()
    c.request_stop()
    assert c.stopped and c()


def test_start_rearms_after_a_previous_stop() -> None:
    """Without this, one stop would silently cancel every later run."""
    c = JobControl()
    c.request_stop()
    c.start()
    assert not c.stopped


def test_should_stop_now_treats_none_as_never() -> None:
    assert should_stop_now(None) is False
    assert should_stop_now(lambda: True) is True


# ---------- ① preprocess ----------

def test_preprocess_stops_between_images_and_keeps_what_finished(
        tmp_path: Path, monkeypatch) -> None:
    from studio import pipeline
    from studio.preprocess import PreprocessReport

    control = JobControl()
    done: list[str] = []

    def fake_preprocess(source, work_dir, **kwargs):
        done.append(source.stem)
        if len(done) == 2:
            control.request_stop()  # user clicks Stop while image 2 is running
        return PreprocessReport(source=source, output=work_dir / f"{source.stem}.png",
                                original_size=(1, 1), final_size=(1, 1),
                                restored=False, reason="test")

    monkeypatch.setattr(pipeline, "preprocess", fake_preprocess)
    sources = []
    for stem in ("a", "b", "c", "d"):
        p = tmp_path / f"{stem}.png"
        _img(p)
        sources.append(p)

    reports = pipeline.preprocess_sources(sources, tmp_path / "out",
                                          should_stop=control,
                                          progress=lambda _m: None)

    # The in-flight image is allowed to finish; c and d are never started.
    assert done == ["a", "b"]
    assert [r.source.stem for r in reports] == ["a", "b"]
    assert all(not r.error for r in reports)


def test_preprocess_announces_the_stop(tmp_path: Path, monkeypatch) -> None:
    from studio import pipeline
    from studio.preprocess import PreprocessReport

    control = JobControl()

    def fake_preprocess(source, work_dir, **kwargs):
        control.request_stop()
        return PreprocessReport(source=source, output=work_dir / "x.png",
                                original_size=(1, 1), final_size=(1, 1),
                                restored=False, reason="test")

    monkeypatch.setattr(pipeline, "preprocess", fake_preprocess)
    for stem in ("a", "b"):
        _img(tmp_path / f"{stem}.png")
    lines: list[str] = []

    pipeline.preprocess_sources([tmp_path / "a.png", tmp_path / "b.png"],
                                tmp_path / "out", should_stop=control,
                                progress=lines.append)

    assert any("Stopped" in ln for ln in lines)


# ---------- ② generate ----------

def test_generate_stops_between_shots_and_returns_finished_shots(
        tmp_path: Path, monkeypatch) -> None:
    from studio import pipeline
    from studio.shotplan import default_plan

    control = JobControl()
    made: list[str] = []

    class FakeEngine:
        def generate(self, sources, shot, out_path, seed):
            made.append(shot.id)
            out_path.write_bytes(b"x")
            if len(made) == 3:
                control.request_stop()
            return out_path

    monkeypatch.setattr(pipeline, "make_engine", lambda *a, **kw: FakeEngine())
    src = tmp_path / "ref.png"
    _img(src)

    results = pipeline.generate_shots([src], default_plan()[:8], "comfyui",
                                      tmp_path / "gen", should_stop=control,
                                      progress=lambda _m: None)

    assert len(made) == 3
    # Only the finished shots come back — no fake failures for the ones skipped.
    assert len(results) == 3
    assert all(r.path and not r.error for r in results)


def test_generate_without_a_token_runs_everything(tmp_path: Path, monkeypatch) -> None:
    """The stop parameter must default to off for the CLI and every old caller."""
    from studio import pipeline
    from studio.shotplan import default_plan

    made: list[str] = []

    class FakeEngine:
        def generate(self, sources, shot, out_path, seed):
            made.append(shot.id)
            out_path.write_bytes(b"x")
            return out_path

    monkeypatch.setattr(pipeline, "make_engine", lambda *a, **kw: FakeEngine())
    src = tmp_path / "ref.png"
    _img(src)
    pipeline.generate_shots([src], default_plan()[:5], "comfyui", tmp_path / "gen",
                            progress=lambda _m: None)
    assert len(made) == 5


# ---------- ③ caption ----------

def test_caption_stops_between_images_and_keeps_written_captions(
        tmp_path: Path, monkeypatch) -> None:
    from studio import captioner as C

    control = JobControl()
    seen: list[str] = []

    class Stub:
        spec = type("S", (), {"backend": "gemini", "label": "stub"})()

        def caption(self, path, **kw):
            seen.append(path.name)
            if len(seen) == 2:
                control.request_stop()
            return f"a caption for {path.name}"

        def load(self) -> None: ...
        def unload(self) -> None: ...

    monkeypatch.setattr(C, "Captioner", lambda *a, **kw: Stub())
    images = []
    for stem in ("a", "b", "c", "d"):
        p = tmp_path / f"{stem}.png"
        _img(p)
        images.append(p)

    written: list[Path] = []
    items = C.caption_images(images, "gemini", trigger="trg",
                             should_stop=control, progress=lambda _m: None,
                             on_item=lambda img, cap: written.append(img))

    assert seen == ["a.png", "b.png"]
    assert len(items) == 2
    assert len(written) == 2  # both already persisted, none discarded
