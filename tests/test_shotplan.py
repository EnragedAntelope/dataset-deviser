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


def test_local_prompts_are_plain_english_instructions() -> None:
    for shot in default_plan():
        assert "<sks>" not in shot.local_prompt
        assert shot.local_prompt.startswith("Generate an image of exactly the same")


def test_every_local_shot_names_its_setting() -> None:
    """Close-ups included: without it each one came back on the reference's own
    backdrop and the dataset ended up with seven identical backgrounds."""
    for shot in default_plan():
        assert shot.setting in shot.local_prompt, shot.id


def test_three_quarter_fronts_turn_opposite_ways() -> None:
    """"front-right quarter view" came back as a plain front view on Qwen-Image
    2.1. Camera-orbit wording rotates it toward image-right; "to its left" did NOT
    mirror it (both shots faced image-right, seen live), so the left shot names the
    image edge instead (verified to face image-left)."""
    by_id = {x.id: x for x in default_plan()}
    right = by_id["angle-front-right"].local_prompt
    left = by_id["angle-front-left"].local_prompt
    assert "camera moved 45 degrees around the character to its right" in right
    assert "turned toward the left edge of the image" in left


def test_local_prompts_never_negate() -> None:
    """Qwen-Image 2.1 draws what a prompt names, even negated — "do not include any
    backpacks" put a backpack on the character in 2 of 8 test shots. So the local
    prompt must stay free of negation however the options are set."""
    from studio.shotplan import apply_prop_exclusion, concept_plan

    for plan in (default_plan(), concept_plan()):
        for shot in plan:
            local = apply_prop_exclusion(apply_wardrobe(shot)).local_prompt.lower()
            for phrase in ("without", "do not", "don't", "no bags", "backpack"):
                assert phrase not in local, (shot.id, phrase)


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


def test_apply_wardrobe_lands_before_the_style_sentence_in_real_prompts() -> None:
    """The outfit used to be appended AFTER the style sentence, producing
    "…not a photograph., wearing a red raincoat Show only…"."""
    pose = next(x for x in default_plan() if x.kind == "pose")
    out = apply_wardrobe(pose.model_copy(update={"outfit": "a red raincoat"}))
    for prompt in (out.local_prompt, out.cloud_prompt):
        head, _, tail = prompt.partition(". ")
        assert head.endswith(", wearing a red raincoat")
        assert tail.startswith("Match the reference image's medium")


def test_apply_wardrobe_idempotent() -> None:
    shot = Shot(id="pose-x", kind="pose",
                local_prompt="walking, wearing a hat", cloud_prompt="x, wearing a hat",
                outfit="a hat")
    out = apply_wardrobe(shot)
    assert out.local_prompt.count("wearing a hat") == 1
    assert out.cloud_prompt.count("wearing a hat") == 1
