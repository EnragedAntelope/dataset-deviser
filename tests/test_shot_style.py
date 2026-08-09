"""Shot style: the medium the ② prompts ask for.

The bug behind this: an illustrated reference came back photographic, because
both prompt builders hard-coded "photorealistic". These tests pin the three
rules that make the fix correct, not just present.
"""

from __future__ import annotations

from pathlib import Path

from studio import shot_style
from studio.shot_style import CUSTOM, MATCH, SHOT_STYLES, resolve
from studio.shotplan import concept_plan, default_plan, plan_for_type

REPO = Path(__file__).resolve().parent.parent


# ---------- the registry ----------

def test_match_is_the_default_everywhere() -> None:
    assert plan_for_type("character")[0].cloud_prompt == \
        default_plan(style=SHOT_STYLES[MATCH])[0].cloud_prompt
    assert resolve("").key == MATCH
    assert resolve("no-such-style").key == MATCH


def test_every_preset_has_all_three_renderings() -> None:
    for key, style in SHOT_STYLES.items():
        assert style.label
        if key == CUSTOM:
            continue  # filled in from the user's text by resolve()
        assert style.local and style.cloud and style.sample_lead


# ---------- rule 1: `match` instructs, it does not merely stay silent ----------

def test_match_actively_tells_the_model_to_keep_the_reference_medium() -> None:
    """Deleting the medium word is not enough — the model drifts to its own
    prior (usually photographic) unless told to preserve the reference's."""
    prompt = default_plan()[9].cloud_prompt  # a pose shot
    assert "medium" in prompt
    assert "not a photograph" in prompt


def test_match_never_asserts_a_medium_of_its_own() -> None:
    for shot in default_plan() + concept_plan():
        for text in (shot.local_prompt, shot.cloud_prompt):
            assert "photorealistic" not in text.lower()


# ---------- rule 2: the style is introduced, not appended ----------

def test_custom_style_is_introduced_with_a_connective() -> None:
    """A style appended as a bare noun phrase gets drawn as scene CONTENT — the
    model puts the object in the picture instead of rendering in that style."""
    style = resolve(CUSTOM, "a View-Master stereo slide with a cardboard mount")
    assert style.cloud.startswith("Rendered as ")
    shot = default_plan(style=style)[9]
    assert "Rendered as a View-Master" in shot.cloud_prompt


def test_custom_style_lowercases_its_first_word_but_spares_proper_nouns() -> None:
    assert resolve(CUSTOM, "A gouache painting").cloud == \
        "Rendered as a gouache painting."
    assert "Rendered as PS1-era" in resolve(CUSTOM, "PS1-era low-poly graphics").cloud


def test_empty_custom_text_falls_back_to_match() -> None:
    """An empty custom style is `match` with extra steps — and must not emit a
    dangling 'Rendered as .' fragment."""
    assert resolve(CUSTOM, "   ").key == MATCH
    assert resolve(CUSTOM, "").key == MATCH


# ---------- rule 3: photographic is not "photorealistic" ----------

def test_photographic_uses_camera_vocabulary_not_photorealistic() -> None:
    """They are different aesthetic targets: 'photorealistic' trends toward an
    idealized CG-adjacent render, a photograph has real optics."""
    style = SHOT_STYLES["photographic"]
    assert "photorealistic" not in (style.local + style.cloud).lower()
    assert "hyperrealistic" not in (style.local + style.cloud).lower()
    for term in ("camera", "lens", "depth of field", "texture"):
        assert term in style.cloud.lower()


def test_no_generated_prompt_anywhere_says_photorealistic() -> None:
    """The regression guard with teeth. The word used to be on all 24 cloud
    prompts and 15 local ones; this sweeps every style x both plans x both
    fields, which is the whole surface that reaches a model.

    Scoped to generated prompts on purpose — a repo-wide grep would also flag
    the comments that exist to explain the ban.
    """
    offenders = []
    for key in SHOT_STYLES:
        for dtype in ("character", "concept"):
            for shot in plan_for_type(dtype, "Sy", key, "a woodcut print"):
                for field in ("local_prompt", "cloud_prompt"):
                    text = getattr(shot, field).lower()
                    for word in ("photorealistic", "hyperrealistic"):
                        if word in text:
                            offenders.append(f"{key}/{dtype}/{shot.id}/{field}: {word}")
    assert not offenders, f"banned style wording reached a prompt: {offenders[:5]}"


# ---------- threading into the plans ----------

def test_style_reaches_both_plans_and_both_prompt_fields() -> None:
    shots = plan_for_type("character", "Sy Snootles", "anime")
    pose = next(s for s in shots if s.kind == "pose")
    assert "anime illustration" in pose.local_prompt
    assert "Rendered as an anime illustration" in pose.cloud_prompt

    concept = plan_for_type("concept", "brass compass", "render3d")
    framing = next(s for s in concept if s.kind == "framing")
    assert "3D render" in framing.local_prompt
    assert "Rendered as a 3D CGI render" in framing.cloud_prompt


def test_angle_shots_keep_the_sks_grammar_clean() -> None:
    """The Multiple-Angles LoRA is trained on clean splat renders; appending
    prose degrades it (same reason prop exclusion skips angle local prompts)."""
    for style_key in SHOT_STYLES:
        for shot in plan_for_type("character", "X", style_key, "a woodcut print"):
            if shot.kind != "angle":
                continue
            assert shot.local_prompt.startswith("<sks> ")
            body = SHOT_STYLES[style_key].local
            if body:
                assert body not in shot.local_prompt
            assert "Rendered as" not in shot.local_prompt


def test_style_never_generates_regardless_of_style() -> None:
    assert plan_for_type("style", "x", "anime") == []


def test_prompts_have_no_double_punctuation_or_dangling_commas() -> None:
    for key in SHOT_STYLES:
        for shot in plan_for_type("character", "Sy", key, "an oil painting"):
            for text in (shot.local_prompt, shot.cloud_prompt):
                assert ", ," not in text
                assert ".." not in text
                assert not text.strip().endswith(",")


# ---------- ⑤ sample prompt ----------

def test_sample_prompt_names_the_right_medium() -> None:
    from studio.trainer_configs import TRAINER_MODELS, TrainConfig, _sample_prompt

    preset = TRAINER_MODELS["ai-toolkit"][0]

    def prompt(style: str, text: str = "", dtype: str = "character") -> str:
        return _sample_prompt(TrainConfig(
            trainer="ai-toolkit", model=preset, dataset_dir=Path("."),
            trigger="trg", name="n", dataset_type=dtype,
            shot_style=style, shot_style_text=text))

    # match keeps the historical wording — no regression for existing configs.
    assert prompt(MATCH) == "a photo of trg, standing outdoors in daylight"
    assert prompt("anime").startswith("an anime illustration of trg")
    assert prompt("render3d").startswith("a 3D render of trg")
    assert prompt("anime", dtype="concept") == "an anime illustration of trg"
    # Style datasets describe content, not a medium — unchanged.
    assert prompt("anime", dtype="style") == "trg, a mountain landscape at sunset"


def test_sample_prompt_handles_a_custom_style() -> None:
    from studio.trainer_configs import TRAINER_MODELS, TrainConfig, _sample_prompt

    out = _sample_prompt(TrainConfig(
        trainer="ai-toolkit", model=TRAINER_MODELS["ai-toolkit"][0],
        dataset_dir=Path("."), trigger="trg", name="n",
        shot_style=CUSTOM, shot_style_text="a woodcut print"))
    assert "woodcut print" in out and "trg" in out


# ---------- persistence ----------

def test_shot_style_round_trips_through_user_config(tmp_path, monkeypatch) -> None:
    import studio.user_config as uc

    monkeypatch.setattr(uc, "USER_SETTINGS_FILE", tmp_path / "s.json")
    monkeypatch.setattr(uc, "CACHE_DIR", tmp_path)
    assert uc.get_shot_style() == (MATCH, "")
    uc.set_shot_style("anime")
    assert uc.get_shot_style() == ("anime", "")
    uc.set_shot_style(CUSTOM, "a woodcut print")
    assert uc.get_shot_style() == (CUSTOM, "a woodcut print")
    # A hand-edited/unknown value degrades to the safe default rather than
    # putting the dropdown in a state it has no entry for.
    uc.save_user_config({"shot_style": "nonsense"})
    assert uc.get_shot_style()[0] == MATCH


def test_unknown_style_key_is_not_persisted(tmp_path, monkeypatch) -> None:
    import studio.user_config as uc

    monkeypatch.setattr(uc, "USER_SETTINGS_FILE", tmp_path / "s.json")
    monkeypatch.setattr(uc, "CACHE_DIR", tmp_path)
    uc.set_shot_style("anime")
    uc.set_shot_style("bogus")
    assert uc.get_shot_style()[0] == "anime"


def test_style_choices_are_ui_ready() -> None:
    labels = [label for label, _ in shot_style.STYLE_CHOICES]
    keys = [key for _, key in shot_style.STYLE_CHOICES]
    assert keys[0] == MATCH and "default" in labels[0].lower()
    assert keys[-1] == CUSTOM
    assert len(set(keys)) == len(keys)
