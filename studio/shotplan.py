"""Shot plans: curated camera angles, poses, emotions, and settings.

Instead of generating standalone "scene/lighting" shots that repeat the same
standing pose with different backgrounds, each shot here is a unique
combination of angle/pose/emotion/setting. This keeps the dataset size
manageable (24 shots) while maximizing the diversity the LoRA actually learns.

Two plans, one shot model:
- `default_plan()`   — Character datasets (24 shots: angles / poses / emotions).
- `concept_plan()`   — Concept datasets (18 shots: angles / framing / context).
  Objects have no emotions and no wardrobe, so those fields simply stay empty
  and every downstream consumer (the ② table, `plan_io` YAML, `apply_wardrobe`)
  keeps working unchanged.

Style datasets never generate — an aesthetic can't be synthesized from a
reference the way an identity or an object can.

The roadmap (done + deferred items) lives in `docs/ARCHITECTURE.md` under
"Roadmap / deferred", not here.
"""

from __future__ import annotations

from pydantic import BaseModel

from studio import shot_style
from studio.shot_style import ShotStyle


class Shot(BaseModel):
    id: str
    # Character plan: "angle" | "pose" | "emotion".
    # Concept plan:   "angle" | "framing" | "context".
    # Only "angle" is special downstream (it drives the Multiple-Angles LoRA
    # strength in the ComfyUI engine and the "isolate angle shots" option).
    kind: str
    local_prompt: str  # Qwen-Image-Edit-2511 prompt (angles use the <sks> LoRA grammar)
    cloud_prompt: str  # plain-English instruction for Nano Banana
    # Rear views hallucinate when generated straight from a front reference;
    # chain them off a generated side view instead (stepwise rotation).
    chain_from: str = ""
    # Emotion and setting are stored explicitly so the dataframe is readable
    # and so future tooling can filter/group shots by these dimensions.
    emotion: str = ""
    setting: str = ""
    # Wardrobe/outfit override. Empty = keep the reference's default clothing
    # (no identity drift). When set, "wearing {outfit}" is injected into both
    # the local and cloud prompts so clothing can vary across the dataset.
    outfit: str = ""


# Each tuple is: (id_suffix, kind, <sks> grammar or pose stub, plain-English
# description, chain_from, emotion, setting).
#
# Design goals:
# - 9 angles: the core turnaround, each with a different setting/lighting so no
#   two are the same generic standing shot.
# - 8 poses: each pose is paired with a setting and emotion; lighting is part
#   of the setting rather than a separate repeated pose.
# - 7 emotions: close-up expression shots with varied angles and settings.
# - Total: 24 shots (down from 28) while improving per-shot diversity.
#
# Settings are written as "natural" lighting/environment phrases so both the
# local and cloud prompts read like plain English.
_SHOTS = [
    # ---------- angles ----------
    (
        "front",
        "angle",
        "front view eye-level shot medium shot",
        "seen directly from the front at eye level, full body visible",
        "",
        "neutral",
        "against a plain neutral gray studio background with soft even lighting",
    ),
    (
        "front-right",
        "angle",
        "front-right quarter view eye-level shot medium shot",
        "seen from a front-right three-quarter angle at eye level, full body visible",
        "",
        "neutral",
        "outdoors in daylight in an open field",
    ),
    (
        "right",
        "angle",
        "right side view eye-level shot medium shot",
        "seen directly from the right side in full profile, full body visible",
        "",
        "neutral",
        "in a warmly lit interior room",
    ),
    (
        "back-right",
        "angle",
        "back-right quarter view eye-level shot medium shot",
        "seen from a back-right three-quarter angle, full body visible",
        "angle-right",
        "neutral",
        "on a city street at dusk",
    ),
    (
        "back",
        "angle",
        "back view eye-level shot medium shot",
        "seen directly from behind, full body visible",
        "angle-right",
        "neutral",
        "against a plain neutral gray studio background with soft even lighting",
    ),
    (
        "back-left",
        "angle",
        "back-left quarter view eye-level shot medium shot",
        "seen from a back-left three-quarter angle, full body visible",
        "angle-left",
        "neutral",
        "standing in a forest with dappled sunlight",
    ),
    (
        "left",
        "angle",
        "left side view eye-level shot medium shot",
        "seen directly from the left side in full profile, full body visible",
        "",
        "neutral",
        "outdoors at golden hour with warm backlighting",
    ),
    (
        "front-left",
        "angle",
        "front-left quarter view eye-level shot medium shot",
        "seen from a front-left three-quarter angle at eye level, full body visible",
        "",
        "neutral",
        "lit by dramatic hard side lighting against a dark background",
    ),
    (
        "low",
        "angle",
        "front view low-angle shot medium shot",
        "seen from a low camera angle looking up",
        "",
        "confident",
        "outdoors at night under cool moonlight",
    ),
    # ---------- poses ----------
    (
        "seated",
        "pose",
        "sitting down on a simple wooden stool, hands resting naturally",
        "sitting down on a simple wooden stool, hands resting naturally",
        "",
        "relaxed",
        "in a warmly lit interior room",
    ),
    (
        "lying",
        "pose",
        "lying down on the ground on its side, relaxed",
        "lying down on the ground on its side, relaxed",
        "",
        "peaceful",
        "outdoors in daylight in an open field",
    ),
    (
        "walking",
        "pose",
        "walking forward mid-stride",
        "walking forward mid-stride",
        "",
        "determined",
        "on a city street at dusk",
    ),
    (
        "crouching",
        "pose",
        "crouching low to the ground",
        "crouching low to the ground",
        "",
        "alert",
        "standing in a forest with dappled sunlight",
    ),
    (
        "arms-raised",
        "pose",
        "with both arms raised overhead",
        "with both arms raised overhead",
        "",
        "triumphant",
        "lit by dramatic hard side lighting against a dark background",
    ),
    (
        "leaning",
        "pose",
        "leaning against a wall casually",
        "leaning against a wall casually",
        "",
        "casual",
        "on a city street at dusk",
    ),
    (
        "action",
        "pose",
        "in a dynamic action pose, mid-movement",
        "in a dynamic action pose, mid-movement",
        "",
        "intense",
        "outdoors in daylight in an open field",
    ),
    (
        "looking-back",
        "pose",
        "standing and looking back over one shoulder",
        "standing and looking back over one shoulder",
        "",
        "playful",
        "outdoors at golden hour with warm backlighting",
    ),
    # ---------- emotions (close-ups) ----------
    (
        "smiling",
        "emotion",
        "close-up of the face, smiling expression",
        "a close-up of the face and upper shoulders, smiling warmly",
        "",
        "smiling",
        "against a soft neutral studio background",
    ),
    (
        "serious",
        "emotion",
        "close-up of the face, serious expression",
        "a close-up of the face and upper shoulders, serious expression",
        "",
        "serious",
        "lit by dramatic hard side lighting against a dark background",
    ),
    (
        "surprised",
        "emotion",
        "close-up of the face, surprised expression",
        "a close-up of the face and upper shoulders, surprised expression",
        "",
        "surprised",
        "in a warmly lit interior room",
    ),
    (
        "laughing",
        "emotion",
        "close-up of the face, laughing expression",
        "a close-up of the face and upper shoulders, laughing openly",
        "",
        "laughing",
        "outdoors in daylight in an open field",
    ),
    (
        "contemplative",
        "emotion",
        "close-up of the face, contemplative expression",
        "a close-up of the face and upper shoulders, contemplative gaze",
        "",
        "contemplative",
        "outdoors at night under cool moonlight",
    ),
    (
        "confident",
        "emotion",
        "close-up of the face, confident expression",
        "a close-up of the face and upper shoulders, confident expression",
        "",
        "confident",
        "outdoors at golden hour with warm backlighting",
    ),
    (
        "sad",
        "emotion",
        "close-up of the face, sad expression",
        "a close-up of the face and upper shoulders, sad expression",
        "",
        "sad",
        "in a warmly lit interior room",
    ),
]


# Concept plan. Each tuple is: (id_suffix, kind, <sks> grammar or phrase,
# plain-English description, chain_from, setting).
#
# Design goals (mirroring the character plan's, minus identity):
# - 10 angles: the turnaround the Multiple-Angles LoRA is actually good at,
#   reusing the SAME <sks> grammar (view + camera height + shot size). No
#   invented grammar terms — a "top view" the LoRA was never trained on would
#   silently produce a normal front view.
# - 4 framing shots: scale variation (extreme detail -> tiny in a wide shot) so
#   the LoRA isn't locked to one distance.
# - 4 context shots: where the thing sits and how it is used, including a hand
#   for scale.
# - Settings vary per shot for the same reason as the character plan: 18 images
#   of the same gray backdrop teach the backdrop.
#
# No emotions, no wardrobe — an object has neither.
_CONCEPT_SHOTS = [
    # ---------- angles (turnaround) ----------
    (
        "front",
        "angle",
        "front view eye-level shot medium shot",
        "seen directly from the front at eye level, the whole subject in frame",
        "",
        "on a plain neutral gray studio backdrop with soft even lighting",
    ),
    (
        "front-right",
        "angle",
        "front-right quarter view eye-level shot medium shot",
        "seen from a front-right three-quarter angle at eye level",
        "",
        "on a wooden tabletop in a warmly lit room",
    ),
    (
        "right",
        "angle",
        "right side view eye-level shot medium shot",
        "seen directly from the right side in full profile",
        "",
        "outdoors in daylight on flat open ground",
    ),
    (
        "back-right",
        "angle",
        "back-right quarter view eye-level shot medium shot",
        "seen from a back-right three-quarter angle",
        "angle-right",
        "on a concrete surface under overcast daylight",
    ),
    (
        "back",
        "angle",
        "back view eye-level shot medium shot",
        "seen directly from behind",
        "angle-right",
        "on a plain neutral gray studio backdrop with soft even lighting",
    ),
    (
        "back-left",
        "angle",
        "back-left quarter view eye-level shot medium shot",
        "seen from a back-left three-quarter angle",
        "angle-left",
        "against a dark background with dramatic hard side lighting",
    ),
    (
        "left",
        "angle",
        "left side view eye-level shot medium shot",
        "seen directly from the left side in full profile",
        "",
        "outdoors at golden hour with warm backlighting",
    ),
    (
        "front-left",
        "angle",
        "front-left quarter view eye-level shot medium shot",
        "seen from a front-left three-quarter angle at eye level",
        "",
        "on a wooden tabletop in a warmly lit room",
    ),
    (
        "low",
        "angle",
        "front view low-angle shot medium shot",
        "seen from a low camera angle looking up at it",
        "",
        "outdoors in daylight on flat open ground",
    ),
    (
        "high",
        "angle",
        "front view high-angle shot medium shot",
        "seen from a high camera angle looking down on it",
        "",
        "on a concrete surface under overcast daylight",
    ),
    # ---------- framing / scale ----------
    (
        "detail",
        "framing",
        "an extreme close-up of one distinctive detail of it, filling the frame",
        "an extreme close-up of one distinctive detail of it, filling the frame",
        "",
        "under soft even lighting",
    ),
    (
        "close",
        "framing",
        "a close-up that fills the frame, showing its surface texture",
        "a close-up that fills the frame, showing its surface texture",
        "",
        "on a wooden tabletop in a warmly lit room",
    ),
    (
        "full",
        "framing",
        "shown in full with clear empty space around it",
        "shown in full with clear empty space around it",
        "",
        "on a plain neutral gray studio backdrop with soft even lighting",
    ),
    (
        "wide",
        "framing",
        "small in the distance in a wide establishing shot",
        "small in the distance in a wide establishing shot",
        "",
        "outdoors in daylight in an open landscape",
    ),
    # ---------- context / use ----------
    (
        "table",
        "context",
        "resting on a table beside ordinary everyday objects",
        "resting on a table beside ordinary everyday objects",
        "",
        "in a warmly lit interior room",
    ),
    (
        "ground",
        "context",
        "placed on the ground outdoors",
        "placed on the ground outdoors",
        "",
        "outdoors at golden hour with warm backlighting",
    ),
    (
        "held",
        "context",
        "held in a person's hand, showing its real-world scale",
        "held in a person's hand, showing its real-world scale",
        "",
        "in a warmly lit interior room",
    ),
    (
        "in-use",
        "context",
        "being used for its normal purpose in its natural surroundings",
        "being used for its normal purpose in its natural surroundings",
        "",
        "outdoors in daylight",
    ),
]


def _subject_phrase(subject: str) -> str:
    """Subject text that reads correctly after "the same …".

    Callers pass a natural noun phrase ("the character", "the object", "character
    Sy Snootles"), so a leading article would double up ("the same the object").
    """
    subject = subject.strip() or "subject"
    return subject[4:] if subject[:4].lower() == "the " else subject


def _indefinite_article(word: str) -> str:
    """"a" or "an" for `word` — emotions are a small curated vocabulary

    (neutral/confident/alert/intense/…), so a plain first-letter check is
    enough; no need for a phonetic library. Without this, vowel-starting
    emotions like "alert"/"intense" produced "with a alert expression".
    """
    return "an" if word[:1].lower() in "aeiou" else "a"


def _build_local_prompt(
    kind: str, grammar_or_pose: str, setting: str, emotion: str,
    outfit: str = "", subject: str = "subject", style: ShotStyle | None = None
) -> str:
    """Build the ComfyUI/Qwen-Edit prompt.

    Angle shots keep the tight <sks> Multiple-Angles LoRA grammar so the LoRA
    can do its job; every other kind is plain English with setting/lighting
    folded in. The emotion is appended so it influences expression without
    breaking the LoRA grammar for angles. An explicit outfit, when given, is
    appended after the grammar/pose so clothing can vary.

    `subject` is interpolated here, not left as a `{subject}` placeholder:
    nothing downstream formats the local prompt, so a placeholder went to
    ComfyUI verbatim. Shots with no emotion (the Concept plan) drop the mood
    clause instead of emitting a dangling "  mood".

    **Angle shots get no style clause.** Their prompt is the <sks> grammar the
    Multiple-Angles LoRA was trained on (clean splat renders); appending prose
    degrades it, which is the same reason `apply_prop_exclusion` skips them.
    They carry no medium claim today either, so nothing is lost — Qwen-Edit
    follows the reference image's medium on its own.
    """
    wardrobe = f", wearing {outfit}" if outfit else ""
    if kind == "angle":
        prompt = f"<sks> {grammar_or_pose}"
        if emotion and emotion != "neutral":
            prompt += f", {emotion} expression"
        return prompt + wardrobe
    # Settings are complete phrases ("in a warmly lit interior room", "outdoors
    # at golden hour") — they carry their own preposition.
    medium = (style or shot_style.SHOT_STYLES[shot_style.MATCH]).local
    tail = f"{emotion} mood, " if emotion else ""
    tail += f"{medium}, " if medium else ""
    tail += "consistent identity" if emotion else "unchanged form"
    return (f"the same {_subject_phrase(subject)}, {grammar_or_pose}{wardrobe}, "
            f"{setting}, {tail}")


def _build_cloud_prompt(
    subject: str, kind: str, description: str, setting: str, emotion: str,
    outfit: str = "", style: ShotStyle | None = None
) -> str:
    """Build the plain-English Nano Banana instruction.

    The medium is stated once, at the END, as its own sentence — never as an
    adjective on "image" (that used to read "Generate a photorealistic image
    of …", which turned every illustrated reference into a photograph).
    """
    style = style or shot_style.SHOT_STYLES[shot_style.MATCH]
    parts = [
        f"Generate an image of exactly the same {_subject_phrase(subject)} "
        "from the reference image(s), identical in every physical detail",
        f", {description}",
    ]
    # Close-up expression shots carry their own framing AND their expression in
    # the description; every other kind names the setting (already a complete
    # phrase) and, for characters, the mood.
    if kind != "emotion":
        if setting:
            parts.append(f", {setting}")
        if emotion and emotion != "neutral":
            parts.append(f", with {_indefinite_article(emotion)} {emotion} expression")
    if outfit:
        parts.append(f", wearing {outfit}")
    parts.append(f". {style.cloud}")
    return "".join(parts)


def apply_wardrobe(shot: Shot) -> Shot:
    """Return a copy of `shot` with its outfit folded into the prompts.

    The outfit column is the source of truth: whatever the user types there is
    injected as "wearing {outfit}" at generation time, so the column stays
    functional even if the prompt cells were edited by hand. Idempotent — a
    prompt that already mentions the outfit is left untouched.
    """
    if not shot.outfit:
        return shot
    phrase = f"wearing {shot.outfit}"
    local = shot.local_prompt
    cloud = shot.cloud_prompt
    if phrase.lower() not in local.lower():
        local = f"{local}, {phrase}"
    if phrase.lower() not in cloud.lower():
        # Insert before the trailing "Keep the same..." sentence when present.
        if ". Keep the same" in cloud:
            head, _, tail = cloud.partition(". Keep the same")
            cloud = f"{head}, {phrase}. Keep the same{tail}"
        else:
            cloud = f"{cloud}, {phrase}"
    return shot.model_copy(update={"local_prompt": local, "cloud_prompt": cloud})


# Props carried in a reference image get copied into every generated shot, and a
# dataset where 20/24 images show the same backpack teaches the LoRA that the
# backpack IS the character. These clauses ask the generator to drop them.
#
# Deliberately NOT applied to `kind="angle"` local prompts: those use the <sks>
# Multiple-Angles LoRA grammar, which is trained on clean splat renders and
# degrades when prose is appended (see ARCHITECTURE.md). Diffusion models also
# handle negation poorly in a positive prompt — naming "backpack" can summon one.
# Angle shots rely on isolation instead, which removes props from the reference
# itself and is the mechanism that actually works.
_CLOUD_NO_PROPS = (
    " Show only the character and the clothing worn on their body — do not "
    "include any backpacks, bags, straps, held objects, tools, props, or "
    "accessories that appear in the reference image."
)
_LOCAL_NO_PROPS = ", without any bags or carried accessories"


def apply_prop_exclusion(shot: Shot) -> Shot:
    """Return a copy of `shot` asking the generator to omit reference props.

    Applied at generation time (like `apply_wardrobe`) rather than baked into the
    plan, so the column stays honest and hand-edited prompt cells still get the
    clause. Idempotent.
    """
    cloud = shot.cloud_prompt
    if _CLOUD_NO_PROPS.strip() not in cloud:
        cloud = f"{cloud}{_CLOUD_NO_PROPS}"
    local = shot.local_prompt
    if shot.kind != "angle" and _LOCAL_NO_PROPS not in local:
        local = f"{local}{_LOCAL_NO_PROPS}"
    return shot.model_copy(update={"local_prompt": local, "cloud_prompt": cloud})


def default_plan(subject: str = "the character",
                 style: ShotStyle | None = None) -> list[Shot]:
    """Return the curated 24-shot Character plan."""
    shots: list[Shot] = []
    for suffix, kind, grammar_or_pose, description, chain, emotion, setting in _SHOTS:
        shots.append(
            Shot(
                id=f"{kind}-{suffix}",
                kind=kind,
                local_prompt=_build_local_prompt(kind, grammar_or_pose, setting, emotion,
                                                 subject=subject, style=style),
                cloud_prompt=_build_cloud_prompt(subject, kind, description, setting,
                                                 emotion, style=style),
                chain_from=chain,
                emotion=emotion,
                setting=setting,
                outfit="",
            )
        )
    return shots


def plan_subject(name: str, dataset_type: str = "character") -> str:
    """The noun phrase woven into the ② prompts for a dataset type.

    Shared by the UI and the CLI so the two can't drift: a character is named
    ("character Sy Snootles") because the prompts talk about a person, while a
    concept is just its own noun ("brass compass").
    """
    name = (name or "").strip()
    if dataset_type == "concept":
        return name or "the object"
    return f"character {name}" if name else "the character"


def plan_for_type(dataset_type: str, name: str = "",
                  shot_style_key: str = shot_style.MATCH,
                  shot_style_text: str = "") -> list[Shot]:
    """The shot plan for a dataset type — the single plan-selection seam.

    Style returns an empty plan: an aesthetic can't be synthesized from a
    reference, so there is nothing honest to put in the table. Callers that can
    act on a plan (the CLI) refuse Style explicitly before getting here; the UI
    disables ②'s buttons and shows the guidance note instead.

    `shot_style_key`/`shot_style_text` pick the visual style baked into the
    prompts (default: match the reference image's own medium).
    """
    subject = plan_subject(name, dataset_type)
    style = shot_style.resolve(shot_style_key, shot_style_text)
    if dataset_type == "style":
        return []
    if dataset_type == "concept":
        return concept_plan(subject=subject, style=style)
    return default_plan(subject=subject, style=style)


def concept_plan(subject: str = "the object",
                 style: ShotStyle | None = None) -> list[Shot]:
    """Return the curated 18-shot Concept plan (objects / actions / ideas).

    Same `Shot` model as the Character plan, so the ② table, YAML plans and the
    generation pipeline need no special case — emotion and outfit just stay
    empty (an object has neither).
    """
    shots: list[Shot] = []
    for suffix, kind, grammar_or_pose, description, chain, setting in _CONCEPT_SHOTS:
        shots.append(
            Shot(
                id=f"{kind}-{suffix}",
                kind=kind,
                local_prompt=_build_local_prompt(kind, grammar_or_pose, setting, "",
                                                 subject=subject, style=style),
                cloud_prompt=_build_cloud_prompt(subject, kind, description, setting,
                                                 "", style=style),
                chain_from=chain,
                emotion="",
                setting=setting,
                outfit="",
            )
        )
    return shots
