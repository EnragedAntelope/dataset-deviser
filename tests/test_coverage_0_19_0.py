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
