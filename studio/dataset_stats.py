"""Inspect a finished dataset folder so training configs can be derived from it.

A flat 2000 steps is wrong for both an 8-image and a 60-image dataset, and a
single 1024 bucket wastes any non-square images. Reading the folder costs one
Pillow header parse per file and makes both choices explainable.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from studio.config import list_images

# Character LoRAs converge around ~75 steps per image on these trainers. Clamped
# because the ratio stops holding at the extremes: a 4-image set still needs a
# floor to learn anything, and a 100-image set does not need 7,500 steps.
STEPS_PER_IMAGE = 75
MIN_STEPS = 1000
MAX_STEPS = 4000

# PNG text chunk ① writes into every prepped image: the source photo's name,
# original size and capture time. ③ only writes sidecars and ④ copies bytes,
# so it reaches the dataset — where an upscaled source otherwise looks native.
PROVENANCE_KEY = "dd_source"


def provenance(im: Image.Image) -> dict:
    """The ① provenance of an opened image, or {} (generated shots carry none)."""
    try:
        data = json.loads(im.info.get(PROVENANCE_KEY) or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}

# ai-toolkit's established multi-resolution idiom: list the buckets and it sorts
# images into them. Only keys attested in ai-toolkit's own examples are emitted.
BUCKET_LADDER = [512, 768, 1024]


@dataclass
class DatasetStats:
    n_images: int
    n_captioned: int
    sizes: list[tuple[int, int]] = field(default_factory=list)
    aspect_counts: Counter = field(default_factory=Counter)
    # (source name, original long side) for images ① enlarged from a smaller source.
    upscaled_sources: list[tuple[str, int]] = field(default_factory=list)

    @property
    def min_long_side(self) -> int:
        return min((max(s) for s in self.sizes), default=0)

    @property
    def max_long_side(self) -> int:
        return max((max(s) for s in self.sizes), default=0)

    @property
    def target_steps(self) -> int:
        if not self.n_images:
            return MIN_STEPS
        return max(MIN_STEPS, min(MAX_STEPS, self.n_images * STEPS_PER_IMAGE))

    def suggested_repeats(self, epochs: int, batch_size: int = 1) -> int:
        """Repeats that bring `epochs` passes near `target_steps`.

        Epochs set how many checkpoints there are to compare; repeats set how
        long each one trains. 16 epochs of a 24-image set at repeats 1 is only
        384 steps — too few to learn a face.
        """
        if not self.n_images:
            return 1
        per_repeat = self.n_images * max(epochs, 1) / max(batch_size, 1)
        return max(1, round(self.target_steps / per_repeat))

    def buckets_for(self, resolution: int) -> list[int]:
        """Bucket ladder capped at `resolution` — never upscale past the
        training resolution, and never past what the images actually contain."""
        ceiling = min(resolution, self.max_long_side or resolution)
        ladder = [b for b in BUCKET_LADDER if b <= ceiling]
        if resolution <= ceiling and resolution not in ladder:
            ladder.append(resolution)
        return sorted(set(ladder)) or [resolution]

    def undersized(self, resolution: int) -> list[tuple[int, int]]:
        """Sizes whose long side is below `resolution` — the images a trainer
        would have to upscale (or bucket down) to reach the training size."""
        return [s for s in self.sizes if max(s) < resolution]

    def max_upscale(self, resolution: int) -> float:
        """Worst upscale factor the training resolution would demand, 1.0 when
        nothing is undersized. Sizes of 0 are ignored rather than dividing."""
        long_sides = [max(s) for s in self.sizes if max(s) > 0]
        smallest = min(long_sides, default=0)
        if not smallest or smallest >= resolution:
            return 1.0
        return resolution / smallest

    def upscale_note(self, resolution: int) -> str:
        """Advisory line when the dataset can't fill the training resolution.

        A LoRA trained on upscaled sources learns the upscaler's softness along
        with the subject, and nothing in the pipeline can tell you that from the
        config alone — the images have to be measured. Empty string when every
        image already clears the bar, so a clean dataset says nothing.
        """
        note = ""
        small = self.undersized(resolution)
        if small:
            factor = self.max_upscale(resolution)
            note = (f"\n⚠️ {len(small)} of {self.n_images} image(s) are below "
                    f"{resolution}px on the long side — up to {factor:.1f}× upscale to "
                    f"fill it. The kohya/musubi/Fizgig configs set `bucket_no_upscale`, "
                    f"so those land in a smaller bucket instead; for ai-toolkit, either "
                    f"lower the resolution or restore them at ① first.")
        grown = self.upscaled_sources
        if grown:
            names = ", ".join(f"{n} ({side}px)" for n, side in grown[:5])
            more = f" and {len(grown) - 5} more" if len(grown) > 5 else ""
            note += (f"\n⚠️ {len(grown)} of {self.n_images} image(s) were upscaled at ① "
                     f"from smaller sources: {names}{more}. They look full-size but hold "
                     f"only the source's detail — consider leaving them out at ④ when "
                     f"generated shots cover the same views.")
        return note

    def summary(self) -> str:
        if not self.n_images:
            return "No images found in that folder."
        shapes = ", ".join(f"{a} ×{n}" for a, n in self.aspect_counts.most_common(4))
        lines = [
            f"**{self.n_images} images** ({self.n_captioned} with captions)",
            f"Long side: {self.min_long_side}–{self.max_long_side}px",
            f"Aspect ratios: {shapes}",
            f"Target: **~{self.target_steps} steps** "
            f"({self.n_images} images × {STEPS_PER_IMAGE}, clamped to "
            f"{MIN_STEPS}–{MAX_STEPS}); Repeats = 0 sizes each epoch to reach it",
        ]
        if self.n_captioned < self.n_images:
            lines.append(f"⚠️ {self.n_images - self.n_captioned} image(s) have no "
                         f"`.txt` caption and will be trained with an empty caption.")
        return "  \n".join(lines)


def _aspect_label(w: int, h: int) -> str:
    if w == h:
        return "square"
    ratio = w / h
    known = {"3:2": 1.5, "4:3": 4 / 3, "16:9": 16 / 9, "2:3": 2 / 3,
             "3:4": 0.75, "9:16": 9 / 16}
    label, _ = min(known.items(), key=lambda kv: abs(kv[1] - ratio))
    return label


def inspect(dataset_dir: Path) -> DatasetStats:
    """Read image dimensions, caption coverage and ① provenance for `dataset_dir`.

    Pillow parses headers lazily, so this never decodes pixel data — and a PNG's
    text chunks sit before its image data, so provenance comes with the header.
    Unreadable files are skipped rather than raising — a stray non-image
    shouldn't break config generation.
    """
    images = list_images(dataset_dir)
    stats = DatasetStats(n_images=0, n_captioned=0)
    for path in images:
        try:
            with Image.open(path) as im:
                size = im.size
                source = provenance(im)
        except Exception:
            continue
        try:
            original = max(int(source.get("w") or 0), int(source.get("h") or 0))
        except (TypeError, ValueError):  # a malformed chunk is no provenance
            original = 0
        if 0 < original < max(size):
            stats.upscaled_sources.append((str(source.get("name", path.name)), original))
        stats.n_images += 1
        stats.sizes.append(size)
        stats.aspect_counts[_aspect_label(*size)] += 1
        if path.with_suffix(".txt").exists():
            stats.n_captioned += 1
    return stats
