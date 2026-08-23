"""Tests for the advisory caption lint + tag-frequency report."""

from __future__ import annotations

from pathlib import Path

from studio.caption_lint import (
    analyze_folder,
    analyze_pairs,
    lint_captions,
    looks_like_tags,
    markdown_summary,
    tag_frequency,
    ubiquitous_tags,
)

# ---------- health lint ----------

def test_lint_flags_empty_short_and_missing_trigger() -> None:
    pairs = [
        ("00.png", "trig, a person standing in a park"),
        ("01.png", ""),                    # empty
        ("02.png", "trig"),                # short (1 word)
        ("03.png", "a person sitting"),    # missing trigger
    ]
    r = lint_captions(pairs, trigger="trig")
    assert r.total == 4
    assert r.empty == ["01.png"]
    assert r.short == ["02.png"]
    assert r.missing_trigger == ["03.png"]
    assert not r.clean


def test_lint_detects_identical_caption_groups() -> None:
    pairs = [
        ("00.png", "trig, standing"),
        ("01.png", "trig, standing"),   # byte-identical
        ("02.png", "Trig,  Standing"),  # same after normalization
        ("03.png", "trig, sitting"),
    ]
    r = lint_captions(pairs, trigger="trig")
    assert len(r.duplicates) == 1
    _, names = r.duplicates[0]
    assert names == ["00.png", "01.png", "02.png"]


def test_lint_empty_caption_is_not_also_a_duplicate() -> None:
    # Two empties are reported as empty, not as an identical-caption group.
    r = lint_captions([("a.png", ""), ("b.png", "")], trigger="")
    assert r.empty == ["a.png", "b.png"]
    assert r.duplicates == []


def test_lint_clean_when_all_good() -> None:
    pairs = [("00.png", "trig, standing outdoors"),
             ("01.png", "trig, sitting indoors")]
    assert lint_captions(pairs, trigger="trig").clean


# ---------- tag detection + frequency ----------

def test_looks_like_tags_true_for_short_comma_segments() -> None:
    assert looks_like_tags(["trig, from side, full body", "trig, standing, park"])


def test_looks_like_tags_false_for_prose() -> None:
    prose = ["trig, a person standing in a sunlit park wearing a long coat",
             "trig, the figure sits on a wooden bench near tall trees"]
    assert not looks_like_tags(prose)


def test_tag_frequency_counts_per_image_and_excludes_trigger() -> None:
    caps = ["trig, solo, standing", "trig, solo, sitting", "trig, solo, standing, standing"]
    freq = dict(tag_frequency(caps, trigger="trig"))
    assert freq["solo"] == 3          # once per caption despite the repeat
    assert freq["standing"] == 2
    assert "trig" not in freq          # trigger never counted


def test_ubiquitous_tags_needs_min_images() -> None:
    caps = ["trig, solo", "trig, solo", "trig, solo"]  # only 3 -> skipped
    assert ubiquitous_tags(caps, trigger="trig", min_images=4) == []


def test_ubiquitous_tags_surfaces_near_universal_tags() -> None:
    caps = [
        "trig, solo, standing",
        "trig, solo, sitting",
        "trig, solo, walking",
        "trig, solo, running",
        "trig, standing",
    ]
    # solo is in 4/5 images. At 0.9 the threshold is ceil(4.5)=5, so it's excluded;
    # at 0.8 the threshold is ceil(4.0)=4, so it surfaces with its count.
    assert dict(ubiquitous_tags(caps, trigger="trig", min_fraction=0.9)) == {}
    assert dict(ubiquitous_tags(caps, trigger="trig", min_fraction=0.8)).get("solo") == 4


# ---------- analyze + folder + summary ----------

def test_analyze_pairs_skips_tag_report_for_prose() -> None:
    prose = [("a.png", "trig, a person standing in a wide green field at noon"),
             ("b.png", "trig, the figure walks along a quiet city street at dusk")]
    _, ubiquitous = analyze_pairs(prose, trigger="trig")
    assert ubiquitous == []  # prose -> no tag-frequency noise


def test_analyze_folder_reads_sidecars(tmp_path: Path) -> None:
    for i, cap in enumerate(["trig, solo, a", "trig, solo, b"]):
        (tmp_path / f"{i}.png").write_bytes(b"x")
        (tmp_path / f"{i}.txt").write_text(cap, encoding="utf-8")
    (tmp_path / "2.png").write_bytes(b"x")  # no sidecar -> counts as empty
    report, _ = analyze_folder(tmp_path, trigger="trig")
    assert report.total == 3
    assert report.empty == ["2.png"]


def test_markdown_summary_clean_and_dirty() -> None:
    clean = markdown_summary(lint_captions([("a.png", "trig, x y z")], "trig"), [])
    assert "look OK" in clean
    dirty = markdown_summary(
        lint_captions([("a.png", ""), ("b.png", "trig, standing")], "trig"), [])
    assert "empty" in dirty
    with_tags = markdown_summary(
        lint_captions([("a.png", "trig, solo")], "trig"), [("solo", 10)])
    assert "nearly every image" in with_tags and "solo" in with_tags
