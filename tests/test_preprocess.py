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
                      backend="", progress=None, front=False, alpha_cutout=False,
                      label=""):
        assert alpha_cutout is True
        w, h = Image.open(image_path).size
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 3] = 128
        Image.fromarray(rgba).save(out_path, "PNG")
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


def test_preprocess_sources_forwards_alpha_cutout(tmp_path: Path, monkeypatch) -> None:
    """The batch wrapper must not drop the flag on the way to preprocess()."""
    from studio import pipeline

    seen = {}

    def fake_preprocess(source, work_dir, **kwargs):
        seen["alpha_cutout"] = kwargs.get("alpha_cutout")
        from studio.preprocess import PreprocessReport
        return PreprocessReport(source=source, output=work_dir / "x.png",
                                original_size=(1, 1), final_size=(1, 1),
                                restored=False, reason="test")

    monkeypatch.setattr(pipeline, "preprocess", fake_preprocess)
    src = tmp_path / "a.png"
    _img(src)
    pipeline.preprocess_sources([src], tmp_path / "out", alpha_cutout=True,
                                progress=lambda _m: None)
    assert seen["alpha_cutout"] is True


# ---------- a failed stage must leave nothing behind ----------

def test_preprocess_removes_the_partial_output_when_isolation_fails(
        tmp_path: Path, monkeypatch) -> None:
    """Restoration writes out_path BEFORE isolation runs. When isolation then
    fails, that restored-but-not-isolated, not-resized file must not survive —
    `list_images` would serve it to ②/③ as a finished source."""
    import studio.preprocess as pp
    from studio.isolate import IsolationError

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "cat.png"
    _img(src, size=(300, 300))
    out = tmp_path / "out"

    def fake_restore(source, out_path):
        Image.new("RGB", (512, 512), (10, 20, 30)).save(out_path, "PNG")
        return out_path

    def fake_isolate(image_path, out_path, *a, **kw):
        raise IsolationError("SAM3 found no 'character'")

    monkeypatch.setattr(pp, "_restore_comfyui", fake_restore)
    monkeypatch.setattr(pp, "isolate_subject", fake_isolate)

    try:
        pp.preprocess(src, out, target=256, force_restore=True, isolate=True,
                      restore_backend="comfyui")
    except IsolationError:
        pass
    else:
        raise AssertionError("expected the isolation failure to propagate")

    assert list(out.glob("*.png")) == [], "a half-processed file was left behind"


def test_isolation_error_names_the_users_file_not_the_intermediate(
        tmp_path: Path, monkeypatch) -> None:
    """preprocess passes the source name through: naming the deleted
    'cat_prepped.png' intermediate sent users hunting for a file that is gone."""
    import studio.preprocess as pp
    from studio.isolate import IsolationError

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "cat.png"
    _img(src)

    def fake_isolate(image_path, out_path, subject_prompt="character", *a, label="", **kw):
        raise IsolationError(f"SAM3 found no '{subject_prompt}' in {label}")

    monkeypatch.setattr(pp, "isolate_subject", fake_isolate)
    try:
        pp.preprocess(src, tmp_path / "out", target=64, force_restore=False, isolate=True)
    except IsolationError as e:
        assert "cat.png" in str(e)
        assert "prepped" not in str(e)
    else:
        raise AssertionError("expected the isolation failure to propagate")


# ---------- batch resilience: one bad image must not kill the run ----------
#
# A real report: SAM3 found no subject in image 3 of a batch, IsolationError
# propagated out of preprocess_sources, and app.py turned it into a gr.Error —
# which discards the outputs, so every image already preprocessed was lost.

def _ok_report(source: Path, work_dir: Path):
    from studio.preprocess import PreprocessReport

    return PreprocessReport(source=source, output=work_dir / f"{source.stem}.png",
                            original_size=(1, 1), final_size=(1, 1),
                            restored=False, reason="test")


def test_preprocess_sources_continues_past_a_failing_image(tmp_path: Path,
                                                           monkeypatch) -> None:
    """One raising source is recorded and skipped; the rest still run."""
    from studio import pipeline
    from studio.isolate import IsolationError

    def fake_preprocess(source, work_dir, **kwargs):
        if source.stem == "b":
            raise IsolationError("SAM3 found no 'character' in b.png")
        return _ok_report(source, work_dir)

    monkeypatch.setattr(pipeline, "preprocess", fake_preprocess)
    sources = []
    for stem in ("a", "b", "c"):
        p = tmp_path / f"{stem}.png"
        _img(p)
        sources.append(p)

    reports = pipeline.preprocess_sources(sources, tmp_path / "out",
                                          progress=lambda _m: None)

    assert len(reports) == 3
    ok = [r for r in reports if not r.error]
    failed = [r for r in reports if r.error]
    assert [r.source.stem for r in ok] == ["a", "c"]
    assert len(failed) == 1
    assert failed[0].source.stem == "b"
    assert failed[0].output is None
    assert "SAM3 found no 'character'" in failed[0].error


def test_preprocess_sources_reports_every_failure_when_all_fail(tmp_path: Path,
                                                                monkeypatch) -> None:
    """All-fail is still a return, not a raise — the caller decides how loud to be."""
    from studio import pipeline

    def fake_preprocess(source, work_dir, **kwargs):
        raise RuntimeError(f"boom {source.stem}")

    monkeypatch.setattr(pipeline, "preprocess", fake_preprocess)
    sources = []
    for stem in ("a", "b"):
        p = tmp_path / f"{stem}.png"
        _img(p)
        sources.append(p)

    reports = pipeline.preprocess_sources(sources, tmp_path / "out",
                                          progress=lambda _m: None)

    assert len(reports) == 2
    assert all(r.error and r.output is None for r in reports)
    assert "boom a" in reports[0].error


def test_preprocess_sources_failure_is_announced_through_progress(tmp_path: Path,
                                                                  monkeypatch) -> None:
    """A skipped image must be visible in the log, not swallowed silently."""
    from studio import pipeline

    def fake_preprocess(source, work_dir, **kwargs):
        raise ValueError("cannot identify image file")

    monkeypatch.setattr(pipeline, "preprocess", fake_preprocess)
    src = tmp_path / "a.png"
    _img(src)
    lines: list[str] = []

    pipeline.preprocess_sources([src], tmp_path / "out", progress=lines.append)

    assert any("SKIPPED" in ln and "a.png" in ln for ln in lines)
