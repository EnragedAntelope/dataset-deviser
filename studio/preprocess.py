"""Preprocess: analyze source images, optionally restore and isolate, resize.

Fully standalone — point it at any image(s). Restoration backends:
- "comfyui" — model-based DeJPG + photo upscale through ComfyUI (best quality)
- "basic"   — plain Lanczos resampling, no external dependency
- "auto"    — comfyui when reachable and restoration is warranted, else basic
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from studio.config import settings
from studio.isolate import isolate_subject

BLUR_THRESHOLD = 120.0  # Laplacian variance below this = soft/degraded image


@dataclass
class PreprocessReport:
    source: Path
    # None when this source failed — see `error`. Every consumer must check.
    output: Path | None
    original_size: tuple[int, int]
    final_size: tuple[int, int]
    restored: bool
    reason: str
    isolated: bool = False
    # Empty on success. When set, this source was skipped and nothing was
    # written for it; the batch carried on with the remaining images.
    error: str = ""


def failed_report(source: Path, error: str) -> PreprocessReport:
    """A report standing in for a source that could not be preprocessed.

    One unusable image must never cost the user the rest of the batch, so the
    failure is *data* the caller can show, not an exception that unwinds the run.
    """
    return PreprocessReport(source=source, output=None, original_size=(0, 0),
                            final_size=(0, 0), restored=False,
                            reason="skipped", error=error)


def _laplacian_variance(img: Image.Image) -> float:
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    lap = (
        -4 * gray
        + np.roll(gray, 1, 0)
        + np.roll(gray, -1, 0)
        + np.roll(gray, 1, 1)
        + np.roll(gray, -1, 1)
    )
    return float(lap.var())


def _needs_restoration(img: Image.Image, path: Path, target: int) -> str:
    """Return a human-readable reason, or '' if the image is fine as-is."""
    long_side = max(img.size)
    if long_side < target:
        return f"long side {long_side}px < target {target}px"
    if path.suffix.lower() in (".jpg", ".jpeg", ".webp"):
        return "lossy source format"
    if _laplacian_variance(img) < BLUR_THRESHOLD:
        return "low sharpness (blur/grain)"
    return ""


def _resize_to_target(img: Image.Image, target: int) -> Image.Image:
    long_side = max(img.size)
    if long_side == target:
        return img
    scale = target / long_side
    new_size = (round(img.width * scale), round(img.height * scale))
    return img.resize(new_size, Image.LANCZOS)


def _restore_comfyui(source: Path, out_path: Path) -> Path:
    from studio import comfy_api

    uploaded = comfy_api.upload_image(source)
    graph = comfy_api.load_template("restore_upscale")
    graph["1"]["inputs"]["image"] = uploaded
    refs = comfy_api.run_prompt(graph, timeout=420)
    return comfy_api.fetch_image(refs[0], out_path)


def preprocess(
    source: Path,
    work_dir: Path,
    target: int | None = None,
    force_restore: bool | None = None,
    isolate: bool = True,
    subject_prompt: str = "character",
    exclude_prompt: str = "",
    restore_backend: str = "",
    isolation_backend: str = "",
    tighten_crop: bool = False,
    alpha_cutout: bool = False,
    progress: Callable[[str], None] | None = None,
) -> PreprocessReport:
    """Copy + clean one source image into `work_dir` at target resolution.

    force_restore: True = always restore, False = never, None = auto-decide.
    isolate: cut out the subject so backgrounds and props don't leak into
    generations or the dataset.
    tighten_crop: after isolation, crop to the subject's bounding box (less white
    padding, more consistent framing). No effect unless `isolate` is on.
    alpha_cutout: export the isolated subject on a transparent background instead
    of white (builtin SAM3 backend only — see `isolate_subject`). No effect
    unless `isolate` is on. This is a terminal output for the caller's own
    compositing workflow, not a new reference format for ② Generate.
    """
    target = target or settings.target_long_side
    restore_backend = restore_backend or settings.restore_backend
    work_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(source).convert("RGB")
    original_size = img.size

    reason = _needs_restoration(img, source, target)
    restore = reason != "" if force_restore is None else force_restore
    if force_restore:
        reason = reason or "forced by user"

    # Never clobber a same-named output. `list_images` admits several extensions,
    # so two sources sharing a stem (e.g. cat.jpg + cat.png, in one folder or
    # across merged inputs) would otherwise both map to `cat_prepped.png` and the
    # second would silently overwrite the first — quietly dropping an image.
    out_path = work_dir / f"{source.stem}_prepped.png"
    n = 2
    while out_path.exists():
        out_path = work_dir / f"{source.stem}_prepped_{n}.png"
        n += 1
    # Every write below lands on `out_path`, and restoration writes it BEFORE
    # isolation runs. A failure after that point used to leave the restored (not
    # isolated, not resized) image behind, where `list_images` happily served it
    # to ②/③ as a finished source — a silent half-processed file in the dataset.
    # The stage is therefore atomic: complete output, or none at all.
    try:
        stage_path = source
        if restore:
            if restore_backend == "auto":
                from studio import comfy_api

                restore_backend = "comfyui" if comfy_api.is_up() else "basic"
            if restore_backend == "comfyui":
                _restore_comfyui(stage_path, out_path)
                stage_path = out_path
            else:
                # Basic path: Lanczos handles resolution; sharpness/compression
                # damage stays (note it so the user knows what they're getting).
                reason += " (basic Lanczos only — ComfyUI restore not used)"

        if isolate:
            isolate_subject(stage_path, out_path, subject_prompt, exclude_prompt,
                            backend=isolation_backend, progress=progress,
                            alpha_cutout=alpha_cutout, label=source.name)
            stage_path = out_path

        if isolate and alpha_cutout:
            img = Image.open(stage_path)  # keep RGBA — no forced flatten
        else:
            img = Image.open(stage_path).convert("RGB")
        if isolate and tighten_crop:
            # Crop the subject-on-white composite to its bounding box before resizing.
            from studio.isolate import crop_to_content

            img = crop_to_content(img)
        img = _resize_to_target(img, target)
        img.save(out_path, "PNG")
    except BaseException:
        out_path.unlink(missing_ok=True)
        raise
    return PreprocessReport(
        source=source,
        output=out_path,
        original_size=original_size,
        final_size=img.size,
        restored=restore,
        reason=reason or "clean source, resize only",
        isolated=isolate,
    )
