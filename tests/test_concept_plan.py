"""Tests for the Concept shot plan (② for object/action datasets) and for the
shared prompt builders it exercises.

The Character plan is the tuned default, so these also pin the parts of it that
the concept work touched: the subject is interpolated (never left as a literal
`{subject}` placeholder), settings are not double-prefixed with "in", and the
mood clause only appears when a shot actually has an emotion.
"""

from __future__ import annotations

from studio.shotplan import (
    Shot,
    apply_prop_exclusion,
    apply_wardrobe,
    concept_plan,
    default_plan,
    plan_for_type,
)

# ---------- the shared selection seam ----------

def test_plan_for_type_picks_the_right_plan() -> None:
    assert len(plan_for_type("character")) == 24
    assert len(plan_for_type("concept")) == 18
    assert plan_for_type("style") == []          # a style is never generated
    assert len(plan_for_type("unknown")) == 24   # unknown falls back to the default


def test_plan_for_type_threads_the_name_into_the_prompts() -> None:
    shot = next(s for s in plan_for_type("concept", "brass compass") if s.kind == "context")
    assert "brass compass" in shot.local_prompt
    assert "brass compass" in shot.cloud_prompt


# ---------- shape ----------

def test_concept_plan_has_18_shots_with_unique_ids() -> None:
    plan = concept_plan()
    assert len(plan) == 18
    assert len({s.id for s in plan}) == 18


def test_concept_kinds_are_angle_framing_context() -> None:
    kinds = {s.kind for s in concept_plan()}
    assert kinds == {"angle", "framing", "context"}


def test_concept_covers_the_turnaround() -> None:
    ids = {s.id for s in concept_plan()}
    assert {"angle-front", "angle-back", "angle-left", "angle-right"} <= ids


def test_every_concept_shot_has_both_prompts() -> None:
    for shot in concept_plan():
        assert shot.local_prompt
        assert shot.cloud_prompt
        assert shot.setting


# ---------- an object has no emotion and no wardrobe ----------

def test_concept_shots_carry_no_emotion_or_outfit() -> None:
    for shot in concept_plan():
        assert shot.emotion == ""
        assert shot.outfit == ""


def test_concept_prompts_have_no_mood_or_expression_clause() -> None:
    for shot in concept_plan():
        assert "mood" not in shot.local_prompt
        assert "expression" not in shot.cloud_prompt


def test_wardrobe_is_a_noop_on_concept_shots() -> None:
    for shot in concept_plan():
        assert apply_wardrobe(shot) is shot


# ---------- local prompts are plain-English instructions ----------

def test_concept_local_prompts_are_plain_english_instructions() -> None:
    plan = concept_plan()
    assert len([s for s in plan if s.kind == "angle"]) == 10
    for shot in plan:
        assert "<sks>" not in shot.local_prompt
        assert shot.local_prompt.startswith("Generate an image of exactly the same")
        assert shot.setting in shot.local_prompt


def test_concept_three_quarter_fronts_turn_opposite_ways() -> None:
    """Same wording split as the character plan (see test_shotplan): orbit wording
    for the right, the image edge for the left, because "to its left" never mirrored."""
    by_id = {s.id: s for s in concept_plan()}
    assert "camera moved 45 degrees around it to its right" in \
        by_id["angle-front-right"].local_prompt
    assert "turned toward the left edge of the image" in by_id["angle-front-left"].local_prompt


def test_rear_concept_views_chain_off_a_generated_side_view() -> None:
    plan = concept_plan()
    ids = {s.id for s in plan}
    chained = [s for s in plan if s.chain_from]
    assert chained
    for shot in chained:
        assert shot.chain_from in ids


def test_concept_settings_vary() -> None:
    settings = [s.setting for s in concept_plan()]
    assert len(set(settings)) >= 6


# ---------- subject interpolation ----------

def test_subject_is_interpolated_not_left_as_a_placeholder() -> None:
    """Nothing downstream formats the local prompt — a `{subject}` placeholder
    went to ComfyUI verbatim."""
    for plan in (default_plan("character Sy Snootles"), concept_plan("a brass compass")):
        for shot in plan:
            assert "{subject}" not in shot.local_prompt
            assert "{subject}" not in shot.cloud_prompt


def test_named_subject_reaches_both_prompts() -> None:
    shot = next(s for s in concept_plan("a brass compass") if s.kind == "context")
    assert "the same a brass compass" in shot.local_prompt
    assert "the same a brass compass" in shot.cloud_prompt


def test_leading_article_is_not_doubled() -> None:
    """"the same the object" reads badly and wastes tokens."""
    for shot in concept_plan("the object"):
        assert "the same the " not in shot.local_prompt
        assert "the same the " not in shot.cloud_prompt
    for shot in default_plan("the character"):
        assert "the same the " not in shot.local_prompt
        assert "the same the " not in shot.cloud_prompt


# ---------- character regressions ----------

def test_character_setting_is_not_prefixed_twice() -> None:
    """Settings are complete phrases ("in a warmly lit interior room"), so the
    builder must not add its own "in"."""
    for shot in default_plan():
        assert "in in a" not in shot.local_prompt
        assert "in in a" not in shot.cloud_prompt
        assert "in against a" not in shot.local_prompt
        assert "in against a" not in shot.cloud_prompt
        assert "in outdoors" not in shot.local_prompt
        assert "in outdoors" not in shot.cloud_prompt


def test_character_pose_shots_keep_their_mood_and_medium_clause() -> None:
    pose = next(s for s in default_plan() if s.kind == "pose")
    assert pose.emotion
    assert f"with a {pose.emotion} expression" in pose.local_prompt
    # The medium clause used to be a hard-coded "photorealistic"; it now comes
    # from the shot style, whose default preserves the reference's own medium.
    assert "Match the reference image's medium" in pose.local_prompt


def test_character_closeups_do_not_repeat_the_expression() -> None:
    """Emotion shots describe the expression in their own description; the
    builder must not append a second "with a … expression"."""
    for shot in default_plan():
        if shot.kind == "emotion":
            assert "with a " not in shot.cloud_prompt
            # The local phrases are vivid ("with a stern, serious expression"), so
            # only the builder's own appended clause is forbidden here.
            for article in ("a", "an"):
                assert f"with {article} {shot.emotion} expression" not in shot.local_prompt


def test_concept_prop_exclusion_still_composes() -> None:
    """`--exclude-props` defaults off for concepts, but forcing it on must not
    break the prompt (it is character-worded, never fatal)."""
    shot = apply_prop_exclusion(concept_plan()[0])
    assert "do not include any backpacks" in shot.cloud_prompt


def test_concept_plan_shots_are_valid_shot_models() -> None:
    for shot in concept_plan():
        assert isinstance(shot, Shot)
        # Round-trips through the ② dataframe / YAML plan files unchanged.
        assert Shot(**shot.model_dump()) == shot
