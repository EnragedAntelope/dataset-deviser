"""Subject isolation: cut the subject out onto a plain white background.

Two interchangeable backends:
- "builtin"  — SAM3 (facebook/sam3) in-process via transformers. No ComfyUI
  needed. The model is gated on Hugging Face: accept the license on the model
  page and authenticate (`hf auth login` or HF_TOKEN) before first use.
- "comfyui"  — the bundled SAM3 workflow templates, run on your ComfyUI.

Isolation matters twice: backgrounds/props can't leak into generations or the
final dataset, and edit models trained on clean renders (the Multiple-Angles
LoRA) behave far better on isolated subjects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

from studio import config
from studio.config import settings


class IsolationError(Exception):
    pass


# ---------- builtin backend: SAM3 via transformers ----------

_model = None
_processor = None


def _load_sam3():
    global _model, _processor
    if _model is None:
        try:
            import torch  # noqa: F401
            from transformers import Sam3Model, Sam3Processor
        except ImportError as e:
            raise IsolationError(f"transformers/torch not available: {e}")
        try:
            # SAM3 is small enough for any one card, so it is pinned to a single
            # device rather than handed to accelerate's "auto" — that sharded it
            # across cuda:0/cuda:1 on multi-GPU machines and the forward pass
            # died with a two-device tensor mismatch.
            _model = Sam3Model.from_pretrained(settings.sam3_hf_id).to(config.torch_device())
            _processor = Sam3Processor.from_pretrained(settings.sam3_hf_id)
        except Exception as e:
            raise IsolationError(
                f"Could not load {settings.sam3_hf_id} — it is a gated model: accept the "
                f"license at https://huggingface.co/{settings.sam3_hf_id} and authenticate "
                f"(`hf auth login` or set HF_TOKEN). Original error: {e}"
            )
    return _model, _processor


def unload() -> None:
    """Free the SAM3 weights (e.g. before loading a large captioner)."""
    global _model, _processor
    if _model is None:
        return
    import gc

    import torch

    _model = _processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _segment(image: Image.Image, prompt: str, threshold: float = 0.5) -> np.ndarray:
    """Union of all instance masks matching the text prompt (bool HxW array)."""
    import torch

    model, processor = _load_sam3()
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = results["masks"]
    if len(masks) == 0:
        return np.zeros((image.height, image.width), dtype=bool)
    union = np.zeros((image.height, image.width), dtype=bool)
    for m in masks:
        union |= np.asarray(m.cpu(), dtype=bool)
    return union


def split_terms(prompt: str) -> list[str]:
    """Split a comma-separated prompt into individual concepts.

    SAM3 scores a text prompt as ONE concept, so "backpack, walkie talkie" asks
    it for a single thing that is both — measured at 40,942 px on the reference
    image versus 60,084 px (1.47x) for the union of the terms segmented apart,
    and *less* than "backpack" alone. Each term therefore gets its own pass.
    """
    return [t.strip() for t in prompt.split(",") if t.strip()]


def _segment_terms(image: Image.Image, prompt: str) -> np.ndarray:
    """Union of one segmentation per comma-separated term in `prompt`."""
    union = np.zeros((image.height, image.width), dtype=bool)
    for term in split_terms(prompt):
        union |= _segment(image, term)
    return union


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow a boolean mask by ~px pixels.

    Only useful for a prop genuinely merged INTO the subject segment (see
    `isolate_builtin`); it is off by default because it otherwise just eats the
    subject. Kept configurable via `LDS_EXCLUDE_DILATE_PX`.
    """
    if px <= 0 or not mask.any():
        return mask
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    size = max(3, px * 2 + 1)
    if size % 2 == 0:
        size += 1
    return np.asarray(img.filter(ImageFilter.MaxFilter(size))) > 127


# Below this subject/prop overlap the exclusion is reported as a no-op rather
# than applied: SAM3 already kept the prop out of the subject segment, so
# subtracting can only remove genuine subject pixels.
_MERGED_PROP_OVERLAP = 0.10


def isolate_builtin(image_path: Path, out_path: Path, subject_prompt: str = "character",
                    exclude_prompt: str = "", progress: Callable[[str], None] | None = None,
                    alpha_cutout: bool = False) -> Path:
    image = Image.open(image_path).convert("RGB")
    subject = _segment(image, subject_prompt)
    if not subject.any():
        raise IsolationError(
            f"SAM3 found no '{subject_prompt}' in {image_path.name} — adjust the "
            f"subject prompt or turn isolation off for this image."
        )
    note = progress or (lambda _m: None)
    if exclude_prompt.strip():
        exclude = _segment_terms(image, exclude_prompt)
        overlap = (subject & exclude).sum()
        share = overlap / exclude.sum() if exclude.any() else 0.0
        if not exclude.any():
            note(f"  (exclude '{exclude_prompt}': not found — nothing removed)")
        elif share < _MERGED_PROP_OVERLAP:
            # Everything outside `subject` is whited out (or, with alpha_cutout, made
            # transparent) by the composite below, so a prop SAM3 already excluded is
            # gone whether or not we subtract it. Subtracting then only removes subject
            # pixels — measured at 2.75% of the character on the reference image for no
            # gain. Only act when the prop is actually inside the subject segment.
            note(f"  (exclude '{exclude_prompt}': already outside the subject mask "
                 f"— removed by isolation itself, no subtraction needed)")
        else:
            subject &= ~_dilate(exclude, px=settings.exclude_dilate_px)
            note(f"  (exclude '{exclude_prompt}': {share:.0%} of it was merged into "
                 f"the subject — subtracted)")

    arr = np.asarray(image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if alpha_cutout:
        # Straight (non-premultiplied) alpha: keep the source RGB everywhere and let
        # alpha carry the cutout, so the visible subject's colors are unaffected by
        # what's outside the mask (unlike the white composite, nothing to bleed from).
        rgba = np.dstack([arr, (subject * 255).astype(np.uint8)])
        Image.fromarray(rgba).save(out_path, "PNG")
    else:
        white = np.full_like(arr, 255)
        out = np.where(subject[..., None], arr, white)
        Image.fromarray(out).save(out_path, "PNG")
    return out_path


def crop_to_content(image: Image.Image, bg_tolerance: int = 8,
                    margin_frac: float = 0.02) -> Image.Image:
    """Crop an isolated (subject-on-white, or subject-with-alpha) image to the
    subject's bounding box.

    Both isolation backends composite the subject onto a plain white background
    (the default), so the subject's bounding box is simply the extent of the
    non-white pixels there. An `alpha_cutout` image has no white to key off —
    its bounding box is the extent of non-transparent (alpha > 0) pixels instead.
    A small margin (fraction of the long side) is left around the subject.
    If the image is effectively empty (nothing found), it is returned unchanged.
    """
    if image.mode == "RGBA":
        nonbg = np.asarray(image.getchannel("A")) > 0
    else:
        arr = np.asarray(image.convert("RGB")).astype(np.int16)
        nonbg = np.any(255 - arr > bg_tolerance, axis=-1)  # anything darker than white
    if not nonbg.any():
        return image
    rows = np.where(nonbg.any(axis=1))[0]
    cols = np.where(nonbg.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    margin = round(max(image.width, image.height) * max(0.0, margin_frac))
    y0, x0 = max(0, y0 - margin), max(0, x0 - margin)
    y1, x1 = min(image.height - 1, y1 + margin), min(image.width - 1, x1 + margin)
    return image.crop((x0, y0, x1 + 1, y1 + 1))


# ---------- comfyui backend ----------

def isolate_comfyui(image_path: Path, out_path: Path, subject_prompt: str = "character",
                    exclude_prompt: str = "", front: bool = False) -> Path:
    from studio import comfy_api

    uploaded = comfy_api.upload_image(image_path)
    if exclude_prompt.strip():
        graph = comfy_api.load_template("isolate_exclude")
        # SAM3 scores a prompt as one concept, so the ComfyUI graph gets the
        # terms joined with " or " — the closest single-encoder equivalent of the
        # builtin backend's per-term union (see split_terms).
        graph["122"]["inputs"]["text"] = " or ".join(split_terms(exclude_prompt))
    else:
        graph = comfy_api.load_template("isolate_subject")
    graph["1"]["inputs"]["image"] = uploaded
    graph["121"]["inputs"]["text"] = subject_prompt
    refs = comfy_api.run_prompt(graph, timeout=420, front=front)
    return comfy_api.fetch_image(refs[0], out_path)


# ---------- dispatch ----------

def isolate_subject(image_path: Path, out_path: Path, subject_prompt: str = "character",
                    exclude_prompt: str = "", backend: str = "",
                    progress: Callable[[str], None] | None = None,
                    front: bool = False, alpha_cutout: bool = False) -> Path:
    backend = backend or settings.isolation_backend
    if backend == "comfyui":
        if alpha_cutout:
            raise IsolationError(
                "Alpha cutout needs the builtin SAM3 backend — the bundled ComfyUI "
                "isolation workflows composite onto a solid background and haven't "
                "been extended for transparency. Switch 'Isolation backend' to "
                "builtin, or turn alpha cutout off."
            )
        return isolate_comfyui(image_path, out_path, subject_prompt, exclude_prompt,
                               front=front)
    return isolate_builtin(image_path, out_path, subject_prompt, exclude_prompt,
                           progress=progress, alpha_cutout=alpha_cutout)
