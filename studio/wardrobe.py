"""Random unisex outfits for the shot plan's `outfit` column.

Why this exists: a dataset built from source images that all show the same
clothes teaches the LoRA that the clothes are part of the character's identity.
Varying wardrobe across the shot set keeps the trigger token bound to the
character rather than to their jacket.

Everything here is deliberately gender-neutral so one pool serves any character
and the UI needs no masculine/feminine picker. Outfits are composed
(colour x garment) rather than hand-written in full, so ~40 vocabulary entries
yield thousands of combinations.
"""

from __future__ import annotations

import random

# Muted, photographic colours. Saturated primaries are avoided: they dominate a
# caption and pull the generator toward costume-like results.
COLORS = [
    "charcoal", "navy", "olive", "burgundy", "cream", "rust", "forest green",
    "slate blue", "mustard", "black", "off-white", "tan", "dusty rose", "plum",
    "teal", "stone grey", "camel", "brick red",
]

# Torso garments that read naturally on any build or gender.
TOPS = [
    "crew-neck sweater", "button-down shirt", "hooded sweatshirt",
    "utility jacket", "denim jacket", "turtleneck", "t-shirt",
    "flannel shirt", "bomber jacket", "cardigan", "long-sleeve henley",
    "polo shirt", "windbreaker", "peacoat", "trench coat", "linen shirt",
    "sweatshirt", "raincoat", "quilted vest", "work jacket",
]

BOTTOMS = [
    "jeans", "chinos", "cargo pants", "corduroy trousers", "track pants",
    "canvas work pants", "wool trousers", "straight-leg trousers",
    "denim shorts", "utility pants",
]


def _combos() -> int:
    return len(COLORS) * len(TOPS) * len(COLORS) * len(BOTTOMS)


def _article(word: str) -> str:
    """'a' or 'an' for the following word. These phrases land verbatim in
    generation prompts and captions, so 'a olive jacket' would be visible."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def random_outfit(rng: random.Random) -> str:
    """One outfit phrase, e.g. 'a charcoal crew-neck sweater and navy jeans'."""
    top_color, bottom_color = rng.sample(COLORS, 2)  # never head-to-toe one colour
    return (f"{_article(top_color)} {top_color} {rng.choice(TOPS)} and "
            f"{bottom_color} {rng.choice(BOTTOMS)}")


def random_outfits(n: int, seed: int | None = None) -> list[str]:
    """`n` outfit phrases, distinct where the vocabulary allows.

    Distinctness matters: the whole point is variety, so a set with repeats
    would partly defeat the feature. The combination space is large enough that
    the retry loop effectively never exhausts it, but it is bounded anyway so a
    caller asking for an absurd `n` degrades to repeats instead of hanging.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    attempts = 0
    budget = n * 20
    while len(out) < n and attempts < budget:
        attempts += 1
        outfit = random_outfit(rng)
        if outfit in seen:
            continue
        seen.add(outfit)
        out.append(outfit)
    while len(out) < n:  # vocabulary exhausted; repeats beat failing
        out.append(random_outfit(rng))
    return out


# Close-ups frame "the face and upper shoulders", where clothing is barely in
# frame; describing a full outfit there invites the model to widen the shot and
# defeat the close-up. Wardrobe variety is carried by the angle/pose shots.
OUTFIT_SHOT_KINDS = ("angle", "pose")


def dress(shots: list) -> list:
    """`shots` with distinct random outfits on the angle/pose rows."""
    targets = [s.id for s in shots if s.kind in OUTFIT_SHOT_KINDS]
    outfits = dict(zip(targets, random_outfits(len(targets)), strict=True))
    return [s.model_copy(update={"outfit": outfits[s.id]}) if s.id in outfits else s
            for s in shots]
