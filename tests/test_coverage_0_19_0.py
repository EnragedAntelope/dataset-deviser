"""0.19.0: framing tiers, the ② coverage line, and the cut-off reference warning."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

import app as A
from studio.isolate import touches_bottom
from studio.preprocess import PreprocessReport
from studio.shotplan import concept_plan, coverage, default_plan


def test_the_character_plan_has_every_framing_tier() -> None:
    plan = default_plan()
    assert len(plan) == 24
    assert Counter(s.framing for s in plan) == {
        "full": 15, "close-up": 5, "waist-up": 2, "tight-face": 2}
    tight = next(s for s in plan if s.framing == "tight-face")
    assert "tight close-up of the face" in tight.local_prompt
    # Framing survives the ② table round trip.
    assert [s.framing for s in A._df_to_shots(A._shots_to_df(plan))] == \
        [s.framing for s in plan]
    assert not any(s.framing for s in concept_plan())


def test_coverage_warns_on_gaps_and_dominance() -> None:
    plan = default_plan()
    assert "⚠️" not in coverage(plan)
    culled = [s for s in plan if s.kind != "emotion" and "back" not in s.id]
    note = coverage(culled)
    assert "no close-up shots" in note and "no back views" in note
    assert "view 'front' 3/6 = 50%" in note
    assert "view 'front'" not in coverage(culled, threshold=0.6)
    assert coverage([]) == ""


def test_coverage_note_counts_only_kept_shots_that_exist() -> None:
    plan = default_plan()
    results = [A.pipeline.GenResult(s, Path(f"{s.id}.png"), seed=1) for s in plan]
    results[0] = A.pipeline.GenResult(plan[0], None, seed=1, error="boom")
    note = A.coverage_note(results, [s.id for s in plan[:5]], 40)
    assert note.startswith("**Coverage of 4 kept shot(s)**")


def _noise(path: Path, seed: int) -> Path:
    import numpy as np

    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(path)
    return path


def test_find_gen_duplicates_names_reference_copies_and_twins(tmp_path: Path) -> None:
    ref = _noise(tmp_path / "ref.png", 1)
    plan = default_plan()[:4]
    paths = [tmp_path / f"{s.id}.png" for s in plan]
    for p, seed in zip(paths, (1, 2, 2, 3), strict=True):  # copy, twin, twin, unique
        _noise(p, seed)
    results = [A.pipeline.GenResult(s, p, seed=1) for s, p in zip(plan, paths, strict=True)]
    note = A.find_gen_duplicates(results, [s.id for s in plan], [str(ref)], "", 5)
    assert "1 shot group(s) look like a copy of a reference" in note
    assert f"{plan[0].id}.png" in note
    assert "1 near-duplicate group(s)" in note
    assert A.primary_reference([str(ref)], "") == str(ref)
    assert A.primary_reference([], "") is None


def test_bursts_group_photos_taken_seconds_apart(tmp_path: Path) -> None:
    import json

    from PIL.PngImagePlugin import PngInfo

    from studio.dataset_stats import PROVENANCE_KEY
    from studio.dedupe import find_bursts

    def photo(name: str, captured: str) -> Path:
        info = PngInfo()
        info.add_text(PROVENANCE_KEY, json.dumps({"name": name, "captured": captured}))
        Image.new("RGB", (8, 8)).save(tmp_path / name, pnginfo=info)
        return tmp_path / name

    shots = [photo("a.png", "2026:05:01 10:00:00"), photo("b.png", "2026:05:01 10:00:02"),
             photo("c.png", "2026:05:01 10:05:00"), photo("d.png", "")]
    assert find_bursts(shots) == [shots[:2]]


def test_touches_bottom_on_white_and_alpha() -> None:
    img = Image.new("RGB", (100, 100), "white")
    img.paste((0, 0, 0), (40, 10, 60, 90))
    assert not touches_bottom(img)
    img.paste((0, 0, 0), (40, 10, 60, 100))
    assert touches_bottom(img)
    rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    rgba.paste((9, 9, 9, 255), (40, 50, 60, 100))
    assert touches_bottom(rgba)


def test_preprocess_flags_a_cut_off_reference(tmp_path: Path, monkeypatch) -> None:
    import studio.preprocess as pp

    src = tmp_path / "a.png"
    Image.new("RGB", (64, 64), (90, 90, 90)).save(src)

    def fake_isolate(image_path, out_path, *a, **kw):
        img = Image.new("RGB", (64, 64), "white")
        img.paste((0, 0, 0), (20, 10, 40, 64))  # legs run off the bottom
        img.save(out_path)
        return out_path

    monkeypatch.setattr(pp, "isolate_subject", fake_isolate)
    r = pp.preprocess(src, tmp_path / "out", target=64, force_restore=False, isolate=True,
                      tighten_crop=True)
    assert r.cut_off
    note = A._preprocess_note([r], tmp_path / "out", False)
    assert "Every reference is cut off" in note
    whole = PreprocessReport(source=Path("b.png"), output=Path("b.png"), original_size=(1, 1),
                             final_size=(1, 1), restored=False, reason="ok")
    assert "other reference(s)" in A._preprocess_note([r, whole], tmp_path / "out", False)
