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
    # Only "angle" is special downstream (the "isolate angle shots" option).
    kind: str
    local_prompt: str  # Qwen-Image 2.1 instruction (same shape as cloud_prompt)
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
    # "full" | "waist-up" | "close-up" | "tight-face"; "" = unknown (concept
    # plans, older saved plans). Requested, not verified — the generator decides.
    framing: str = ""


# Framing per kind, with the rows that break the pattern. Tight face crops carry
# the most likeness signal; waist-up fills the gap between full body and close-up.
_KIND_FRAMING = {"angle": "full", "pose": "full", "emotion": "close-up"}
_FRAMING = {"pose-arms-raised": "waist-up", "pose-looking-back": "waist-up",
            "emotion-smiling": "tight-face", "emotion-confident": "tight-face"}


# Each tuple is: (id_suffix, kind, local phrase for Qwen-Image 2.1, plain-English
# description for Gemini, chain_from, emotion, setting).
#
# Design goals:
# - 9 angles: the core turnaround, each with a different setting/lighting so no
#   two are the same generic standing shot.
# - 8 poses: each pose is paired with a setting and emotion; lighting is part
#   of the setting rather than a separate repeated pose.
# - 7 emotions: close-up expression shots with varied angles and settings.
# - Framing tiers: two poses are waist-up and two close-ups are tight face crops
#   (see _FRAMING), so the set isn't only full body plus one close-up distance.
# - Total: 24 shots (down from 28) while improving per-shot diversity.
#
# Settings are written as "natural" lighting/environment phrases so both the
# local and cloud prompts read like plain English.
_SHOTS = [
    # ---------- angles ----------
    (
        "front",
        "angle",
        "seen directly from the front at eye level, full body visible",
        "seen directly from the front at eye level, full body visible",
        "",
        "neutral",
        "against a plain neutral gray studio background with soft even lighting",
    ),
    (
        "front-right",
        "angle",
        "with the camera moved 45 degrees around the character to its right, so the character is seen in a three-quarter view showing one side of the face and body, full body visible",
        "seen from a front-right three-quarter angle at eye level, full body visible",
        "",
        "neutral",
        "outdoors in daylight in an open field",
    ),
    (
        "right",
        "angle",
        "seen directly from the right side in full profile, full body visible",
        "seen directly from the right side in full profile, full body visible",
        "",
        "neutral",
        "in a warmly lit interior room",
    ),
    (
        "back-right",
        "angle",
        "seen from a back-right three-quarter angle, full body visible",
        "seen from a back-right three-quarter angle, full body visible",
        "angle-right",
        "neutral",
        "on a city street at dusk",
    ),
    (
        "back",
        "angle",
        "seen directly from behind, full body visible",
        "seen directly from behind, full body visible",
        "angle-right",
        "neutral",
        "against a plain neutral gray studio background with soft even lighting",
    ),
    (
        "back-left",
        "angle",
        "seen from a back-left three-quarter angle, full body visible",
        "seen from a back-left three-quarter angle, full body visible",
        "angle-left",
        "neutral",
        "standing in a forest with dappled sunlight",
    ),
    (
        "left",
        "angle",
        "seen directly from the left side in full profile, full body visible",
        "seen directly from the left side in full profile, full body visible",
        "",
        "neutral",
        "outdoors at golden hour with warm backlighting",
    ),
    (
        "front-left",
        "angle",
        "in a three-quarter view, with the character's body and face turned toward the left edge of the image, full body visible",
        "seen from a front-left three-quarter angle at eye level, full body visible",
        "",
        "neutral",
        "lit by dramatic hard side lighting against a dark background",
    ),
    (
        "low",
        "angle",
        "seen from a low camera angle looking up",
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
        "framed from the waist up, with both arms raised overhead",
        "framed from the waist up, with both arms raised overhead",
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
        "framed from the waist up, looking back over one shoulder",
        "framed from the waist up, looking back over one shoulder",
        "",
        "playful",
        "outdoors at golden hour with warm backlighting",
    ),
    # ---------- emotions (close-ups) ----------
    (
        "smiling",
        "emotion",
        "a tight close-up of the face filling the frame, smiling warmly",
        "a tight close-up of the face filling the frame, smiling warmly",
        "",
        "smiling",
        "against a soft neutral studio background",
    ),
    (
        "serious",
        "emotion",
        "a close-up of the face and upper shoulders, with a stern, serious expression and a furrowed brow",
        "a close-up of the face and upper shoulders, serious expression",
        "",
        "serious",
        "lit by dramatic hard side lighting against a dark background",
    ),
    (
        "surprised",
        "emotion",
        "a close-up of the face and upper shoulders, with wide eyes and raised eyebrows, mouth open in surprise",
        "a close-up of the face and upper shoulders, surprised expression",
        "",
        "surprised",
        "in a warmly lit interior room",
    ),
    (
        "laughing",
        "emotion",
        "a close-up of the face and upper shoulders, laughing heartily with the eyes squeezed half shut",
        "a close-up of the face and upper shoulders, laughing openly",
        "",
        "laughing",
        "outdoors in daylight in an open field",
    ),
    (
        "contemplative",
        "emotion",
        "a close-up of the face and upper shoulders, with a thoughtful, distant gaze",
        "a close-up of the face and upper shoulders, contemplative gaze",
        "",
        "contemplative",
        "outdoors at night under cool moonlight",
    ),
    (
        "confident",
        "emotion",
        "a tight close-up of the face filling the frame, with a confident look and a slight smirk",
        "a tight close-up of the face filling the frame, confident expression",
        "",
        "confident",
        "outdoors at golden hour with warm backlighting",
    ),
    (
        "sad",
        "emotion",
        "a close-up of the face and upper shoulders, with a sad, downcast expression and glistening eyes",
        "a close-up of the face and upper shoulders, sad expression",
        "",
        "sad",
        "in a warmly lit interior room",
    ),
]


# Concept plan. Each tuple is: (id_suffix, kind, local phrase, plain-English
# description, chain_from, setting).
#
# Design goals (mirroring the character plan's, minus identity):
# - 10 angles: the turnaround. Three-quarter fronts use camera-orbit wording
#   ("moved 45 degrees around it") because "front-right quarter view" came back
#   as a plain front view on Qwen-Image 2.1; side/back/low/high plain views work.
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
        "seen directly from the front at eye level, the whole subject in frame",
        "seen directly from the front at eye level, the whole subject in frame",
        "",
        "on a plain neutral gray studio backdrop with soft even lighting",
    ),
    (
        "front-right",
        "angle",
        "with the camera moved 45 degrees around it to its right, so it is seen in a three-quarter view showing its front and one side",
        "seen from a front-right three-quarter angle at eye level",
        "",
        "on a wooden tabletop in a warmly lit room",
    ),
    (
        "right",
        "angle",
        "seen directly from the right side in full profile",
        "seen directly from the right side in full profile",
        "",
        "outdoors in daylight on flat open ground",
    ),
    (
        "back-right",
        "angle",
        "seen from a back-right three-quarter angle",
        "seen from a back-right three-quarter angle",
        "angle-right",
        "on a concrete surface under overcast daylight",
    ),
    (
        "back",
        "angle",
        "seen directly from behind",
        "seen directly from behind",
        "angle-right",
        "on a plain neutral gray studio backdrop with soft even lighting",
    ),
    (
        "back-left",
        "angle",
        "seen from a back-left three-quarter angle",
        "seen from a back-left three-quarter angle",
        "angle-left",
        "against a dark background with dramatic hard side lighting",
    ),
    (
        "left",
        "angle",
        "seen directly from the left side in full profile",
        "seen directly from the left side in full profile",
        "",
        "outdoors at golden hour with warm backlighting",
    ),
    (
        "front-left",
        "angle",
        "in a three-quarter view, with it turned toward the left edge of the image, showing its front and one side",
        "seen from a front-left three-quarter angle at eye level",
        "",
        "on a wooden tabletop in a warmly lit room",
    ),
    (
        "low",
        "angle",
        "seen from a low camera angle looking up at it",
        "seen from a low camera angle looking up at it",
        "",
        "outdoors in daylight on flat open ground",
    ),
    (
        "high",
        "angle",
        "seen from a high camera angle looking down on it",
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


def _instruction(subject: str, phrase: str, setting: str, emotion: str,
                 outfit: str, style: ShotStyle, with_mood: bool) -> str:
    """The one-sentence edit instruction both engines are given.

    Qwen-Image 2.1 follows the same plain-English instruction Gemini does, and it
    keeps identity and medium far better with it than 2511 did with terse tags
    (measured). The medium is stated once, at the END, as its own sentence — never
    as an adjective on "image" (that used to read "Generate a photorealistic image
    of …", which turned every illustrated reference into a photograph).
    """
    parts = [
        f"Generate an image of exactly the same {_subject_phrase(subject)} "
        "from the reference image(s), identical in every physical detail",
        f", {phrase}",
    ]
    if setting:
        parts.append(f", {setting}")
    if with_mood and emotion and emotion != "neutral":
        parts.append(f", with {_indefinite_article(emotion)} {emotion} expression")
    if outfit:
        parts.append(f", wearing {outfit}")
    parts.append(f". {style.cloud}")
    return "".join(parts)


def _build_local_prompt(
    kind: str, phrase: str, setting: str, emotion: str,
    outfit: str = "", subject: str = "subject", style: ShotStyle | None = None
) -> str:
    """Build the ComfyUI / Qwen-Image 2.1 prompt.

    Same sentence as the cloud prompt, with one difference: close-up shots keep
    their setting. Without it every close-up came back on the reference's own
    backdrop, and the dataset ends up with seven identical backgrounds.

    `subject` is interpolated here, not left as a `{subject}` placeholder:
    nothing downstream formats the local prompt.
    """
    style = style or shot_style.SHOT_STYLES[shot_style.MATCH]
    return _instruction(subject, phrase, setting, emotion, outfit, style,
                        with_mood=kind != "emotion")


def _build_cloud_prompt(
    subject: str, kind: str, description: str, setting: str, emotion: str,
    outfit: str = "", style: ShotStyle | None = None
) -> str:
    """Build the plain-English Nano Banana instruction.

    Close-up expression shots carry their own framing AND their expression in
    the description, so they get neither a setting nor a mood clause here.
    """
    style = style or shot_style.SHOT_STYLES[shot_style.MATCH]
    is_close_up = kind == "emotion"
    return _instruction(subject, description, "" if is_close_up else setting, emotion,
                        outfit, style, with_mood=not is_close_up)


def _insert_outfit(prompt: str, phrase: str) -> str:
    """Put ", wearing …" at the end of the instruction clause, before the style
    sentence. Appends instead when the prompt has no sentence break (a cell the
    user rewrote by hand).

    ponytail: splits at the first ". " — a subject name containing one ("Dr. X")
    would take the outfit early; rename the subject or edit the prompt cell.
    """
    head, sep, tail = prompt.partition(". ")
    return f"{head}, {phrase}{sep}{tail}"


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
        local = _insert_outfit(local, phrase)
    if phrase.lower() not in cloud.lower():
        cloud = _insert_outfit(cloud, phrase)
    return shot.model_copy(update={"local_prompt": local, "cloud_prompt": cloud})


# Props carried in a reference image get copied into every generated shot, and a
# dataset where 20/24 images show the same backpack teaches the LoRA that the
# backpack IS the character. These clauses ask the generator to drop them.
#
# Deliberately NOT applied to local prompts: Qwen-Image 2.1 draws what a prompt
# names, even negated — "do not include any backpacks" put a backpack on the
# character in 2 of 8 test shots. Gemini follows the negation. Locally, isolating
# the source in ① removes props from the reference itself, which is the mechanism
# that actually works.
_CLOUD_NO_PROPS = (
    " Show only the character and the clothing worn on their body — do not "
    "include any backpacks, bags, straps, held objects, tools, props, or "
    "accessories that appear in the reference image."
)


def apply_prop_exclusion(shot: Shot) -> Shot:
    """Return a copy of `shot` whose CLOUD prompt asks to omit reference props.

    Applied at generation time (like `apply_wardrobe`) rather than baked into the
    plan, so the column stays honest and hand-edited prompt cells still get the
    clause. The local prompt is never touched (see above). Idempotent.
    """
    cloud = shot.cloud_prompt
    if _CLOUD_NO_PROPS.strip() not in cloud:
        cloud = f"{cloud}{_CLOUD_NO_PROPS}"
    return shot.model_copy(update={"cloud_prompt": cloud})


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
                framing=_FRAMING.get(f"{kind}-{suffix}", _KIND_FRAMING[kind]),
            )
        )
    return shots


FRAMING_TIERS = ("full", "waist-up", "close-up", "tight-face")
# Turnaround direction of an angle shot, by id suffix.
_VIEW = {"front": "front", "front-left": "front", "front-right": "front",
         "left": "side", "right": "side", "back": "back", "back-left": "back",
         "back-right": "back", "low": "low"}


def coverage(shots: list[Shot], threshold: float = 0.4) -> str:
    """Markdown summary of what a kept set covers — from the plan, not the pixels.

    Warns on an empty framing tier or turnaround direction, and on any view or
    expression above `threshold` of its group: the LoRA leans toward whatever
    dominates the set.
    """
    if not shots:
        return ""
    from collections import Counter

    framing = Counter(s.framing or "unknown" for s in shots)
    views = Counter(_VIEW.get(s.id.removeprefix("angle-"), "other")
                    for s in shots if s.kind == "angle")
    moods = Counter(s.emotion or "neutral" for s in shots)
    dressed = sum(1 for s in shots if s.outfit)

    def line(name: str, counts: Counter) -> str:
        top = counts.most_common(4)
        rest = sum(counts.values()) - sum(n for _, n in top)
        more = f", {rest} across {len(counts) - 4} more" if rest else ""
        return f"**{name}:** " + ", ".join(f"{k} {n}" for k, n in top) + more

    lines = [f"**Coverage of {len(shots)} kept shot(s)** — requested, not verified",
             line("Framing", framing), line("Turnaround", views) if views else "",
             line("Expression", moods), f"**Outfit:** {dressed} varied, "
             f"{len(shots) - dressed} as in the reference"]
    warn = [f"no {t} shots" for t in FRAMING_TIERS if any(s.framing for s in shots)
            and not framing[t]]
    if views:
        warn += [f"no {v} views" for v in ("front", "side", "back") if not views[v]]
    for name, counts in (("view", views), ("expression", moods)):
        total = sum(counts.values())
        warn += [f"{name} '{k}' {n}/{total} = {n / total:.0%}"
                 for k, n in counts.items() if total >= 5 and n / total > threshold]
    if warn:
        lines.append("⚠️ " + "; ".join(warn))
    return "  \n".join(x for x in lines if x)


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
