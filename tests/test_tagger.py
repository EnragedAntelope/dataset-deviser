"""Tests for the ONNX tagger backend (pure logic + glue, no ONNX/download)."""

from __future__ import annotations

from pathlib import Path

from studio.captioner import Captioner
from studio.config import CAPTIONERS_BY_KEY
from studio.tagger import (
    Tagger,
    _read_selected_tags,
    format_tag,
    select_rating,
    select_tag_names,
)

# ---------- pure selection / formatting ----------

def test_select_tag_names_danbooru_thresholds_and_order() -> None:
    labels = [
        ("long_hair", 0, 0.9),        # general
        ("smile", 0, 0.5),            # general
        ("blurry", 0, 0.1),           # general, below threshold
        ("hatsune_miku", 4, 0.95),    # character
        ("some_char", 4, 0.5),        # character, below threshold
        ("explicit", 9, 0.8),         # rating -> dropped
    ]
    out = select_tag_names(labels, 0.35, 0.85, scheme="danbooru")
    assert out == ["long_hair", "smile", "hatsune_miku"]


def test_select_tag_names_e621_species_is_general() -> None:
    labels = [
        ("wolf", 5, 0.9),        # species -> general threshold in e621
        ("solo", 0, 0.8),        # general
        ("jay_naylor", 1, 0.95),  # artist -> character threshold
        ("meta_tag", 7, 0.99),   # meta -> dropped
        ("faint_species", 5, 0.2),  # species below general threshold
    ]
    out = select_tag_names(labels, 0.35, 0.85, scheme="e621")
    # general(+species) by confidence, then character-like (artist); meta dropped.
    assert out == ["wolf", "solo", "jay_naylor"]


def test_format_tag_underscores_to_spaces_but_keeps_kaomoji() -> None:
    assert format_tag("long_hair") == "long hair"
    assert format_tag("^_^") == "^_^"


def test_format_tag_keep_underscores_leaves_raw_tokens() -> None:
    assert format_tag("long_hair", keep_underscores=True) == "long_hair"
    assert format_tag("^_^", keep_underscores=True) == "^_^"  # kaomoji unchanged


# ---------- rating selection ----------

def test_select_rating_picks_top_danbooru_rating() -> None:
    labels = [
        ("long_hair", 0, 0.9),
        ("questionable", 9, 0.3),
        ("explicit", 9, 0.7),   # highest rating -> chosen
        ("general", 9, 0.1),
    ]
    assert select_rating(labels, "danbooru") == "explicit"


def test_select_rating_empty_when_no_rating_category() -> None:
    # e621/Z3D carries no rating category, so the opt-in is a no-op there.
    labels = [("wolf", 5, 0.9), ("explicit", 9, 0.9)]
    assert select_rating(labels, "e621") == ""


def test_read_selected_tags_handles_both_csv_layouts(tmp_path: Path) -> None:
    wd = tmp_path / "selected_tags.csv"
    wd.write_text("tag_id,name,category,count\n1,long_hair,0,100\n2,miku,4,50\n",
                  encoding="utf-8")
    z3d = tmp_path / "tags-selected.csv"
    z3d.write_text("id,name,category,post_count\n1,wolf,5,999\n2,solo,0,888\n",
                   encoding="utf-8")
    assert _read_selected_tags(wd) == (["long_hair", "miku"], [0, 4])
    assert _read_selected_tags(z3d) == (["wolf", "solo"], [5, 0])


# ---------- registry ----------

def test_tagger_specs_registered() -> None:
    for key in ("wd-eva02", "wd-vit"):
        spec = CAPTIONERS_BY_KEY[key]
        assert spec.backend == "wd_tagger"
        assert spec.hf_id.startswith("SmilingWolf/")
        assert spec.tag_scheme == "danbooru"


def test_e621_tagger_spec_registered() -> None:
    spec = CAPTIONERS_BY_KEY["z3d-e621"]
    assert spec.backend == "wd_tagger"
    assert spec.hf_id == "toynya/Z3D-E621-Convnext"
    assert spec.tags_file == "tags-selected.csv"
    assert spec.tag_scheme == "e621"


# ---------- glue (fake session, no onnxruntime import) ----------

class _FakeInput:
    name = "input"
    shape = [1, 448, 448, 3]


class _FakeSession:
    def __init__(self, preds: list[float]) -> None:
        self._preds = preds

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, _outputs, _inputs):
        return [[self._preds]]  # shape (1, n_tags)


def test_tagger_tag_pipeline(monkeypatch) -> None:
    tg = Tagger("fake/repo")
    tg._session = _FakeSession([0.9, 0.5, 0.95, 0.1])  # short-circuits load()
    tg._tag_names = ["long_hair", "smile", "hatsune_miku", "blurry"]
    tg._categories = [0, 0, 4, 0]
    monkeypatch.setattr(tg, "_preprocess", lambda _p: None)
    assert tg.tag(Path("x.png")) == ["long hair", "smile", "hatsune miku"]


def test_tagger_tag_pipeline_rating_and_underscores(monkeypatch) -> None:
    tg = Tagger("fake/repo", include_rating=True, keep_underscores=True)
    tg._session = _FakeSession([0.9, 0.95, 0.7])  # long_hair, character, explicit
    tg._tag_names = ["long_hair", "hatsune_miku", "explicit"]
    tg._categories = [0, 4, 9]  # 9 = rating
    monkeypatch.setattr(tg, "_preprocess", lambda _p: None)
    # underscores kept; the top rating appended after general+character tags.
    assert tg.tag(Path("x.png")) == ["long_hair", "hatsune_miku", "explicit"]


def test_captioner_wd_tagger_returns_joined_tags(monkeypatch) -> None:
    cap = Captioner("wd-eva02")

    class FakeTagger:
        def tag(self, _p):
            return ["long hair", "smile"]

    monkeypatch.setattr(cap, "_load_tagger", lambda: FakeTagger())
    assert cap.caption(Path("x.png"), subject="whatever", style="prose") == "long hair, smile"
