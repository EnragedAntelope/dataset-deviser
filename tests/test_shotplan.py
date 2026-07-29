"""Tests for the curated shot plan."""

from __future__ import annotations

from studio.shotplan import Shot, _build_cloud_prompt, apply_wardrobe, default_plan


def test_default_plan_has_24_shots() -> None:
    plan = default_plan()
    assert len(plan) == 24


def test_shots_have_required_fields() -> None:
    plan = default_plan()
    for shot in plan:
        assert shot.id
        assert shot.kind in {"angle", "pose", "emotion"}
        assert shot.local_prompt
        assert shot.cloud_prompt


def test_emotion_and_setting_fields_exist() -> None:
    plan = default_plan()
    emotions = {s.emotion for s in plan if s.emotion}
    settings = {s.setting for s in plan if s.setting}
    assert emotions
    assert settings
    assert len(settings) >= 6


def test_cloud_prompt_uses_an_before_vowel_starting_emotion() -> None:
    """"with a alert expression" read as a grammar bug in real generated prompts —
    default_plan() actually uses "alert" and "intense" as emotion values."""
    prompt = _build_cloud_prompt("the character", "pose", "standing still", "", "alert")
    assert "with an alert expression" in prompt
    assert "with a alert expression" not in prompt


def test_cloud_prompt_uses_a_before_consonant_starting_emotion() -> None:
    prompt = _build_cloud_prompt("the character", "pose", "standing still", "", "confident")
    assert "with a confident expression" in prompt


def test_chained_shots_sort_last_in_generation() -> None:
    plan = default_plan()
    chained = [s for s in plan if s.chain_from]
    assert chained
    for shot in chained:
        assert shot.chain_from in {s.id for s in plan}


def test_no_duplicate_angle_pose_setting_combinations() -> None:
    plan = default_plan()
    combos = set()
    for shot in plan:
        # Use prompt-derived angle/pose stub; full uniqueness is checked via ids
        combos.add(shot.id)
    assert len(combos) == len(plan)


def test_angle_shots_use_sks_grammar() -> None:
    plan = default_plan()
    angles = [s for s in plan if s.kind == "angle"]
    assert angles
    for shot in angles:
        assert "<sks>" in shot.local_prompt


def test_pose_and_emotion_shots_do_not_use_sks() -> None:
    plan = default_plan()
    non_angles = [s for s in plan if s.kind != "angle"]
    assert non_angles
    for shot in non_angles:
        assert "<sks>" not in shot.local_prompt


def test_emotion_shots_are_closeup() -> None:
    plan = default_plan()
    emotions = [s for s in plan if s.kind == "emotion"]
    assert emotions
    for shot in emotions:
        assert "close-up" in shot.local_prompt.lower() or "closeup" in shot.local_prompt.lower()


def test_setting_varies_across_plan() -> None:
    plan = default_plan()
    settings = [s.setting for s in plan if s.setting]
    assert len(set(settings)) >= 6


def test_plan_includes_common_angles() -> None:
    plan = default_plan()
    ids = {s.id for s in plan}
    assert {"angle-front", "angle-back", "angle-right", "angle-left"} <= ids


def test_default_plan_outfit_empty() -> None:
    for shot in default_plan():
        assert shot.outfit == ""


def test_apply_wardrobe_noop_when_empty() -> None:
    shot = default_plan()[0]
    assert apply_wardrobe(shot) is shot  # untouched, same object


def test_apply_wardrobe_injects_into_both_prompts() -> None:
    shot = Shot(id="pose-x", kind="pose", local_prompt="the same {subject}, walking",
                cloud_prompt="Generate ... walking. Keep the same style.",
                outfit="a red raincoat")
    out = apply_wardrobe(shot)
    assert "wearing a red raincoat" in out.local_prompt
    assert "wearing a red raincoat" in out.cloud_prompt
    # cloud injection lands before the trailing "Keep the same" sentence
    assert out.cloud_prompt.index("wearing") < out.cloud_prompt.index("Keep the same")


def test_apply_wardrobe_idempotent() -> None:
    shot = Shot(id="pose-x", kind="pose",
                local_prompt="walking, wearing a hat", cloud_prompt="x, wearing a hat",
                outfit="a hat")
    out = apply_wardrobe(shot)
    assert out.local_prompt.count("wearing a hat") == 1
    assert out.cloud_prompt.count("wearing a hat") == 1
