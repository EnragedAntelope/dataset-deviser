"""Preprocess: analyze source images, optionally restore and isolate, resize.

Fully standalone — point it at any image(s). Restoration backends:
- "comfyui" — model-based DeJPG + photo upscale through ComfyUI (best quality)
- "basic"   — plain Lanczos resampling, no external dependency
- "auto"    — comfyui when reachable and restoration is warranted, else basic
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from studio.config import settings
from studio.dataset_stats import PROVENANCE_KEY
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
    """Return a human-readable reason, or '' if the image is fine as-is.

    Judged at the size the dataset will hold. A lossy photo at 2x the target or
    more loses its block artefacts in the downscale, and sharpness measured on
    12 MP of sensor noise says nothing about the 1024 px result — flagging every
    phone JPEG sent it through a 4x upscale for nothing.
    """
    long_side = max(img.size)
    if long_side < target:
        return f"long side {long_side}px < target {target}px"
    if path.suffix.lower() in (".jpg", ".jpeg", ".webp") and long_side < 2 * target:
        return "lossy source format"
    if _laplacian_variance(_resize_to_target(img, target)) < BLUR_THRESHOLD:
        return "low sharpness (blur/grain)"
    return ""


def _resize_to_target(img: Image.Image, target: int) -> Image.Image:
    long_side = max(img.size)
    if long_side == target:
        return img
    scale = target / long_side
    new_size = (round(img.width * scale), round(img.height * scale))
    return img.resize(new_size, Image.LANCZOS)


def _restore_comfyui(source: Path, out_path: Path, upscale: bool = True,
                     front: bool = False) -> Path:
    """DeJPG, then the 4x photo upscale only when `upscale` (source below target).

    Running the 4x model on a full-size photo turned a 3000x4000 JPEG into a
    12000x16000 PNG that was saved, fetched and shrunk straight back to 1024.
    """
    from studio import comfy_api

    uploaded = comfy_api.upload_image(source)
    graph = comfy_api.load_template("restore_upscale")
    graph["1"]["inputs"]["image"] = uploaded
    if not upscale:
        graph["6"]["inputs"]["images"] = ["3", 0]  # save the DeJPG output
        del graph["4"], graph["5"]
    refs = comfy_api.run_prompt(graph, timeout=420, front=front)
    return comfy_api.fetch_image(refs[0], out_path)


def _stage_copy(oriented: Image.Image, changed: bool, max_side: int | None) -> Path | None:
    """A temp copy of the source for restore/isolation, or None to use the file.

    Needed when EXIF rotation changed the pixels (the file on disk is sideways)
    or the source exceeds `max_side` (restore/isolation cost scales with pixels).
    None = keep full size.
    """
    if not changed and (max_side is None or max(oriented.size) <= max_side):
        return None
    # convert(): a CMYK JPEG cannot be written as PNG.
    staged = oriented.convert("RGBA" if oriented.has_transparency_data else "RGB")
    if max_side:
        staged.thumbnail((max_side, max_side), Image.LANCZOS)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = Path(tmp.name)
    staged.save(path, "PNG")
    return path


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
    front: bool = False,
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
    with Image.open(source) as raw:
        exif = raw.getexif()
        # Phone photos store portrait shots landscape plus an EXIF rotate tag;
        # ignoring it put sideways people in the dataset.
        rotated = exif.get(0x0112, 1) != 1  # EXIF Orientation
        # DateTimeOriginal (Exif IFD), else DateTime: groups burst shots later.
        captured = str(exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132) or "")
        oriented = ImageOps.exif_transpose(raw)
    img = oriented.convert("RGB")
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
    # 2x the target is all a resize needs, but tighten-crop keeps only the
    # subject's box: shrinking first would upscale a small subject back up.
    staged = _stage_copy(oriented, rotated,
                         None if isolate and tighten_crop else 2 * target)
    try:
        stage_path = staged or source
        if restore:
            if restore_backend == "auto":
                from studio import comfy_api

                restore_backend = "comfyui" if comfy_api.is_up() else "basic"
            if restore_backend == "comfyui":
                _restore_comfyui(stage_path, out_path,
                                 upscale=max(original_size) < target, front=front)
                stage_path = out_path
            else:
                # Basic path: Lanczos handles resolution; sharpness/compression
                # damage stays (note it so the user knows what they're getting).
                reason += " (basic Lanczos only — ComfyUI restore not used)"

        if isolate:
            isolate_subject(stage_path, out_path, subject_prompt, exclude_prompt,
                            backend=isolation_backend, progress=progress,
                            alpha_cutout=alpha_cutout, label=source.name, front=front)
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
        info = PngInfo()
        info.add_text(PROVENANCE_KEY, json.dumps({
            "name": source.name, "w": original_size[0], "h": original_size[1],
            "captured": captured, "restored": restore, "isolated": isolate}))
        img.save(out_path, "PNG", pnginfo=info)
    except BaseException:
        out_path.unlink(missing_ok=True)
        raise
    finally:
        if staged:
            staged.unlink(missing_ok=True)
    return PreprocessReport(
        source=source,
        output=out_path,
        original_size=original_size,
        final_size=img.size,
        restored=restore,
        reason=reason or "clean source, resize only",
        isolated=isolate,
    )
