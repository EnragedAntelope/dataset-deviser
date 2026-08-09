"""Visual style for the ② shot prompts.

Why this exists: a user fed in an **illustrated** reference and got photographic
shots back. That was not a missing feature — both prompt builders hard-coded the
medium ("Generate a photorealistic image of …", "…, photorealistic, consistent
identity"), so an anime reference was explicitly *instructed* to come back as a
photo. The default is now `match`, which tells the model to preserve whatever
medium the reference is in.

Three rules the wording follows, each of which cost something to learn:

1. **`match` is an instruction, not silence.** Simply deleting the medium word
   leaves the model free to drift to its own prior (usually photographic). The
   clause has to actively say "keep the reference's medium".

2. **A style must be *introduced*, not appended.** Concatenating a style written
   as a noun phrase onto a subject description makes a prose-following model draw
   the style as scene *content* — ask for "a View-Master slide" and you get a
   View-Master in the picture. The connective ("Rendered as …") retags it as the
   medium.

3. **Never say "photorealistic" for photographic output.** They are different
   aesthetic targets: "photorealistic"/"hyperrealistic" trend toward an
   idealized, airbrushed, CG-adjacent render, while a photograph has the optics
   and imperfections of a real capture. The `photographic` preset is therefore
   built from camera vocabulary (lens, depth of field, sensor, texture).

`local` is kept short — it goes to Qwen-Image-Edit, which follows terse prompts
and is not a prose model. `cloud` is a full sentence for Gemini. `sample_lead`
is the noun phrase ⑤ uses to open a training sample prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

MATCH = "match"
CUSTOM = "custom"


@dataclass(frozen=True)
class ShotStyle:
    key: str
    label: str
    local: str        # terse clause for the ComfyUI / Qwen-Image-Edit prompt
    cloud: str        # full sentence for the Gemini prompt
    sample_lead: str  # how ⑤'s sample prompt names the medium

    @property
    def is_match(self) -> bool:
        return self.key == MATCH


_STYLES: tuple[ShotStyle, ...] = (
    ShotStyle(
        MATCH, "Match the reference image (default)",
        "in the same art style and medium as the reference",
        "Match the reference image's medium, art style and rendering exactly — if "
        "the reference is a drawing, painting, or 3D render, the result must be the "
        "same medium, not a photograph.",
        "a photo of",  # unknown medium: keep the long-standing default wording
    ),
    ShotStyle(
        "photographic", "Photographic (real camera capture)",
        "photographic, real camera capture, natural lens depth of field, "
        "true skin and material texture",
        "Rendered as a photograph: a real camera capture, with natural lens depth of "
        "field and focus falloff, true-to-life skin and material texture, and light "
        "behaving as it does on a sensor.",
        "a photo of",
    ),
    ShotStyle(
        "anime", "Anime / manga illustration",
        "anime illustration, clean linework, cel shading",
        "Rendered as an anime illustration: clean linework, cel shading and flat "
        "colour fills, in the drawing style of the reference.",
        "an anime illustration of",
    ),
    ShotStyle(
        "comic", "Comic / cartoon (Western)",
        "western comic illustration, bold ink outlines, flat colour",
        "Rendered as a Western comic illustration: bold ink outlines, flat colour "
        "fills and graphic shading.",
        "a comic illustration of",
    ),
    ShotStyle(
        "illustration", "Digital illustration / concept art",
        "digital illustration, painterly brushwork, concept art",
        "Rendered as a digital illustration: painterly brushwork and concept-art "
        "finish, with visible artistic rendering rather than a photographic capture.",
        "a digital illustration of",
    ),
    ShotStyle(
        "painting", "Traditional painting (oil / watercolour)",
        "traditional painting, visible brush strokes, canvas texture",
        "Rendered as a traditional painting: visible brush strokes, pigment and "
        "canvas or paper texture.",
        "a painting of",
    ),
    ShotStyle(
        "render3d", "3D render / CGI",
        "3D render, CGI, physically based shading",
        "Rendered as a 3D CGI render: physically based shading and materials, clean "
        "computer-generated geometry.",
        "a 3D render of",
    ),
    ShotStyle(
        "lineart", "Ink line art / sketch",
        "ink line art, black and white linework, minimal shading",
        "Rendered as ink line art: black-and-white linework with minimal shading and "
        "no photographic detail.",
        "a line art drawing of",
    ),
    ShotStyle(
        CUSTOM, "Custom — describe it below",
        "",  # filled in from the user's text by resolve()
        "",
        "",
    ),
)

SHOT_STYLES: dict[str, ShotStyle] = {s.key: s for s in _STYLES}
# (value, label) pairs for a Gradio dropdown / a CLI help string.
STYLE_CHOICES: list[tuple[str, str]] = [(s.label, s.key) for s in _STYLES]
STYLE_KEYS: tuple[str, ...] = tuple(SHOT_STYLES)


def _lower_first(text: str) -> str:
    """Lowercase the opening letter so the clause reads as a continuation —
    unless it starts on a proper noun / acronym, which we detect crudely by a
    second capital in the first word (e.g. "PS1", "McBess")."""
    first = text.split(" ", 1)[0]
    if sum(1 for c in first if c.isupper()) > 1:
        return text
    return text[:1].lower() + text[1:] if text else text


def resolve(key: str, custom_text: str = "") -> ShotStyle:
    """The style to build prompts with.

    An unknown key falls back to `match` rather than raising: a stale saved
    setting or a typo'd CLI value should degrade to the safe default, not stop a
    run. `custom` with no text does the same — an empty custom style is just
    `match` with extra steps.
    """
    style = SHOT_STYLES.get((key or "").strip().lower())
    if style is None:
        return SHOT_STYLES[MATCH]
    if style.key != CUSTOM:
        return style
    text = " ".join((custom_text or "").split()).rstrip(".")
    if not text:
        return SHOT_STYLES[MATCH]
    return ShotStyle(
        CUSTOM, style.label,
        _lower_first(text),
        # The connective is the whole point — see rule 2 in the module docstring.
        f"Rendered as {_lower_first(text)}.",
        f"{text}, ",
    )
