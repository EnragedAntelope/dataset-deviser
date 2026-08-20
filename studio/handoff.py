"""Hand an exported dataset to Idiot LoRa Builder with the triage already done.

[Idiot LoRa Builder](https://github.com/Fablestarexpanse/Idiot-Lora-Builder) is a
separate MIT desktop app (Tauri/Rust) that opens a folder of images and does the
curation work this project has no UI for — a virtualized grid, ratings, a
bucket-aware crop tool, batch rename/resize. Its input is exactly ④'s output: a
flat folder of images with sidecar `.txt` captions, which an export already is.

So the only thing missing at the boundary is our *judgement*. This module writes
the one sidecar that carries it: `.lora-studio/ratings.json`, in the schema it
reads (`RatingsData { ratings: map<relative path, rating> }`). Every exported
image lands as `good`; anything our advisory checks flagged lands as `needs_edit`,
so the grid opens with the questionable shots already picked out instead of blank.

Nothing is launched and nothing is executed — we write one JSON file into the
dataset folder and tell the user the path. Deciding what to run stays theirs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Idiot LoRa Builder's own layout and vocabulary. Its `ImageRating::from_str`
# maps anything it doesn't recognise to "none", which would silently drop the
# whole triage — so these strings must match its Rust enum exactly.
ILB_DIR = ".lora-studio"
RATINGS_FILE = "ratings.json"
GOOD = "good"
NEEDS_EDIT = "needs_edit"


def ratings_path(ds_dir: Path) -> Path:
    """Where the sidecar goes. Composed from `ds_dir` alone — nothing
    user-supplied enters the path, so there is no traversal surface."""
    return Path(ds_dir) / ILB_DIR / RATINGS_FILE


def triage(ds_dir: Path) -> dict[str, list[str]]:
    """Advisory reasons per exported image, keyed by ILB-relative path.

    Reuses the checks ④ already shows in its preview — sharpness, exposure and
    contrast (`studio.quality`) plus perceptual near-duplicates (`studio.dedupe`).
    Each check is isolated: an unreadable image or a missing optional dependency
    costs that check, never the hand-off. A dataset with no reasons is a dataset
    where everything is `good`, which is a fine thing to hand over.
    """
    from studio.config import list_images

    ds_dir = Path(ds_dir)
    images = list_images(ds_dir)
    reasons: dict[str, list[str]] = {}

    def add(img: Path, reason: str) -> None:
        reasons.setdefault(_rel_key(ds_dir, img), []).append(reason)

    try:
        from studio.quality import composition_flags, is_blurry
    except Exception:
        composition_flags = is_blurry = None  # type: ignore[assignment]
    if is_blurry is not None:
        for img in images:
            try:
                blurry, _score = is_blurry(img)
                if blurry:
                    add(img, "blurry")
                for flag in composition_flags(img):
                    add(img, flag)
            except Exception:
                continue  # advisory — one bad image never costs the rest

    try:
        from studio.dedupe import find_near_duplicate_groups

        for group in find_near_duplicate_groups(images):
            # Flag the copies, not the original: the first member of each group
            # is the one worth keeping, and marking it too would tell the user
            # to re-check a shot that has no problem.
            for dup in group[1:]:
                add(dup, "near-duplicate")
    except Exception:
        pass

    return reasons


def write_ilb_ratings(ds_dir: Path, flagged: dict[str, list[str]] | None = None) -> Path:
    """Write `.lora-studio/ratings.json` for `ds_dir` and return its path.

    Raises `FileExistsError` if the sidecar is already there. Idiot LoRa Builder
    treats a present-but-unparseable ratings file as an error precisely so a
    user's triage is never silently replaced — overwriting one from this side
    would defeat that. Export folders are always new, so this only guards a
    re-run against a dataset someone has since curated.
    """
    ds_dir = Path(ds_dir)
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"Not a dataset folder: {ds_dir}")
    target = ratings_path(ds_dir)
    if target.exists():
        raise FileExistsError(
            f"{target} already exists — leaving the existing ratings alone."
        )

    from studio.config import list_images

    flagged = flagged or {}
    ratings = {}
    for img in list_images(ds_dir):
        key = _rel_key(ds_dir, img)
        ratings[key] = NEEDS_EDIT if flagged.get(key) else GOOD

    target.parent.mkdir(parents=True, exist_ok=True)
    # Same-directory temp + os.replace, mirroring the atomic sidecar writes on
    # the other side: a half-written ratings.json is never observable.
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps({"ratings": ratings}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, target)
    return target


def prepare_handoff(ds_dir: Path) -> str:
    """Triage + write the sidecar. Returns one line for the user, never raises.

    Both the UI and the CLI go through this so they say the same thing, and so
    a hand-off problem can never fail an export that has already written its
    images — by then the dataset is on disk and usable with or without us.
    """
    ds_dir = Path(ds_dir)
    try:
        flagged = triage(ds_dir)
        write_ilb_ratings(ds_dir, flagged)
    except FileExistsError:
        return (f"ℹ️ Idiot LoRa Builder hand-off skipped — {ratings_path(ds_dir)} "
                f"already exists and was left untouched.")
    except OSError as e:
        return f"⚠️ Could not write the Idiot LoRa Builder hand-off sidecar: {e}"
    n = len(flagged)
    checked = (f"{n} image(s) marked 'needs edit' from the quality/duplicate checks"
               if n else "everything marked 'good'")
    return (f"🤝 Ready for Idiot LoRa Builder — open '{ds_dir}' in it ({checked}).")


def _rel_key(ds_dir: Path, img: Path) -> str:
    """The image's key in the ratings map: relative to the dataset root and
    forward-slashed, matching how the other side normalizes its own keys."""
    return img.relative_to(ds_dir).as_posix()
