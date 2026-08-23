"""Tests for the 0.9.0-review features shipped in 0.10.0:

- CLIP 77-token truncation estimate + advisory (caption_lint)
- caption-kind detection + ④→⑤ caption/model sanity check (caption_lint + trainer)
- type-aware trainer sample prompt (trainer_configs)
- tighten-to-subject crop after isolation (isolate)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from studio import caption_lint as CL
from studio.isolate import crop_to_content
from studio.trainer_configs import (
    TRAINER_MODELS,
    TrainConfig,
    _sample_prompt,
    caption_mismatch_warning,
)

# ---------- CLIP token estimate ----------

def test_estimate_clip_tokens_grows_with_length() -> None:
    short = CL.estimate_clip_tokens("trig, standing")
    long = CL.estimate_clip_tokens(", ".join(f"tag{i} word" for i in range(40)))
    assert short < long
    assert short < CL.CLIP_TOKEN_LIMIT < long


def test_analyze_pairs_flags_long_tag_captions_only() -> None:
    long_tags = ", ".join(f"tag{i}" for i in range(90))  # tag-like, very long
    pairs = [("a.png", long_tags), ("b.png", "trig, standing, smiling")]
    report, _ = CL.analyze_pairs(pairs)
    assert "a.png" in report.too_long
    assert "b.png" not in report.too_long


def test_analyze_pairs_does_not_flag_long_prose() -> None:
    # Prose (long clauses) targets T5/Flux encoders (no 77-token cap): never nagged.
    prose = " ".join(["the subject stands in a wide open field"] * 30) + "."
    report, _ = CL.analyze_pairs([("a.png", prose)])
    assert report.too_long == []


def test_folder_caption_kind(tmp_path: Path) -> None:
    assert CL.folder_caption_kind(tmp_path) == ""  # no captions
    for i in range(3):
        (tmp_path / f"{i}.png").write_bytes(b"x")
    (tmp_path / "0.txt").write_text("trig, from side, standing, forest", encoding="utf-8")
    (tmp_path / "1.txt").write_text("trig, from above, sitting, indoors", encoding="utf-8")
    (tmp_path / "2.txt").write_text("trig, close up, smiling, studio", encoding="utf-8")
    assert CL.folder_caption_kind(tmp_path) == "tags"
    for i in range(3):
        (tmp_path / f"{i}.txt").write_text(
            "trig, the subject stands in a sunlit field looking toward the horizon.",
            encoding="utf-8")
    assert CL.folder_caption_kind(tmp_path) == "prose"


# ---------- ④→⑤ caption/model sanity check ----------

def test_sdxl_presets_expect_tags() -> None:
    ai_sdxl = next(p for p in TRAINER_MODELS["ai-toolkit"] if p.key == "sdxl")
    kohya_sdxl = next(p for p in TRAINER_MODELS["kohya"] if p.key == "sdxl")
    flux = next(p for p in TRAINER_MODELS["ai-toolkit"] if p.key == "flux-dev")
    assert ai_sdxl.expects_tags is True
    assert kohya_sdxl.expects_tags is True
    assert flux.expects_tags is False


def test_caption_mismatch_warning_both_directions() -> None:
    sdxl = next(p for p in TRAINER_MODELS["ai-toolkit"] if p.key == "sdxl")
    flux = next(p for p in TRAINER_MODELS["ai-toolkit"] if p.key == "flux-dev")
    # Tag model fed prose -> warn; prose model fed tags -> warn.
    assert caption_mismatch_warning(sdxl, "prose")
    assert caption_mismatch_warning(flux, "tags")
    # Matching styles or unknown -> no warning.
    assert caption_mismatch_warning(sdxl, "tags") == ""
    assert caption_mismatch_warning(flux, "prose") == ""
    assert caption_mismatch_warning(sdxl, "") == ""


# ---------- type-aware sample prompt ----------

def test_sample_prompt_varies_by_dataset_type(tmp_path: Path) -> None:
    def cfg(dt: str) -> TrainConfig:
        return TrainConfig(trainer="ai-toolkit", model=TRAINER_MODELS["ai-toolkit"][0],
                           dataset_dir=tmp_path, trigger="tok", dataset_type=dt)

    char = _sample_prompt(cfg("character"))
    style = _sample_prompt(cfg("style"))
    concept = _sample_prompt(cfg("concept"))
    assert char != style != concept
    assert char.startswith("a photo of tok")
    assert style.startswith("tok,")  # style names content the trigger renders
    assert concept == "a photo of tok"


# ---------- tighten-to-subject crop ----------

def test_crop_to_content_trims_white_border() -> None:
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    arr[40:60, 30:70] = 0  # a black 20x40 subject block
    img = Image.fromarray(arr)
    cropped = crop_to_content(img, margin_frac=0.0)
    assert cropped.size == (40, 20)  # (width, height) of the block, no margin


def test_crop_to_content_adds_margin() -> None:
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    arr[45:55, 45:55] = 0
    img = Image.fromarray(arr)
    cropped = crop_to_content(img, margin_frac=0.05)  # +5px each side
    assert cropped.size == (20, 20)


def test_crop_to_content_all_white_unchanged() -> None:
    img = Image.new("RGB", (64, 48), (255, 255, 255))
    assert crop_to_content(img).size == (64, 48)


# ---------- export records detected caption style ----------

def test_package_records_detected_caption_style(tmp_path: Path) -> None:
    import json

    from studio.package import package_dataset

    src = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(src)
    ds = package_dataset([(src, "trig, from side, standing, forest")],
                         tmp_path / "out", "Sy", "trig", {"dataset_type": "character"})
    meta = json.loads((ds / "metadata.json").read_text(encoding="utf-8"))
    assert meta["caption_style"] == "tags"
    assert meta["dataset_type"] == "character"


def test_package_keeps_caller_supplied_caption_style(tmp_path: Path) -> None:
    import json

    from studio.package import package_dataset

    src = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(src)
    ds = package_dataset([(src, "trig, tag")], tmp_path / "out", "Sy", "trig",
                         {"caption_style": "prose"})  # caller wins, no override
    meta = json.loads((ds / "metadata.json").read_text(encoding="utf-8"))
    assert meta["caption_style"] == "prose"
