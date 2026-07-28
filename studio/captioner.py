"""Captioning: natural-language training captions behind one Captioner interface.

Backends (see CAPTIONERS in config.py):
- "transformers" — local VLM on your GPU (Qwen3-VL, JoyCaption, NSFW finetune)
- "gemini"       — Google Gemini via your API key (costs billed to you)
- "openai"       — any OpenAI-compatible server (Groq cloud, LM Studio, Ollama)

Each captioner spec carries its own prompt template, tuned to how that model
was trained (JoyCaption has a documented instruction convention; the NSFW
finetune wants explicitness requested plainly).

Fully standalone: `caption_folder()` tags any folder of images with .txt
sidecars — no other pipeline stage required.
"""

from __future__ import annotations

import base64
import re
import time
from pathlib import Path
from typing import Callable

import httpx
from PIL import Image

from studio import config
from studio.config import (
    CAPTION_IMAGE_PRICES,
    CAPTIONERS_BY_KEY,
    DEFAULT_CAPTION_PRICE,
    CaptionerSpec,
    settings,
)

_JOYCAPTION_SYSTEM = "You are a helpful image captioner."


class CaptionerConfigError(Exception):
    """A captioner is selected but not usable yet (e.g. custom endpoint unset).

    Plain exception rather than gr.Error so the CLI can use the same resolver;
    the UI translates it at the boundary.
    """


def resolve_captioner_config(captioner_key: str, gemini_model: str = "") -> tuple[str, dict | None]:
    """Return (model_override, spec_overrides) for the chosen captioner.

    Shared by the UI and CLI so the custom endpoint behaves identically in both.

    - Gemini captioner: the caller's model choice wins.
    - Custom endpoint: pull the saved base URL / model / key-env / spacing.
    - Everything else: the captioner's built-in config, unchanged.
    """
    spec = CAPTIONERS_BY_KEY[captioner_key]
    if captioner_key == "custom":
        from studio import user_config

        cfg = user_config.get_custom_captioner()
        if not cfg.get("base_url"):
            raise CaptionerConfigError(
                "The custom endpoint isn't configured yet. In the UI: open 'Custom "
                "endpoint settings' on the ③ Caption tab, enter a base URL (e.g. "
                "https://openrouter.ai/api/v1) and model, then click 💾 Save endpoint. "
                "It is saved on this machine and reused by the CLI."
            )
        return cfg.get("model", ""), {
            "base_url": cfg["base_url"],
            "api_key_env": cfg.get("api_key_env", ""),
            "min_interval_s": cfg.get("min_interval_s", 0.0),
        }
    if spec.backend == "gemini":
        return gemini_model.strip(), None
    return "", None


def merge_tagger_overrides(
    captioner_key: str,
    spec_overrides: dict | None = None,
    *,
    general_threshold: float | None = None,
    character_threshold: float | None = None,
    include_rating: bool | None = None,
    keep_underscores: bool | None = None,
) -> dict | None:
    """Fold the ③ Tag-options controls into spec_overrides — taggers only.

    Shared by the UI and CLI so a tagger behaves identically in both. For any
    non-tagger captioner the overrides are irrelevant, so `spec_overrides` is
    returned untouched. Only the values the caller actually set (not None) are
    merged, leaving the registry defaults for the rest.
    """
    if CAPTIONERS_BY_KEY[captioner_key].backend != "wd_tagger":
        return spec_overrides
    extra: dict = {}
    if general_threshold is not None:
        extra["general_threshold"] = float(general_threshold)
    if character_threshold is not None:
        extra["character_threshold"] = float(character_threshold)
    if include_rating is not None:
        extra["include_rating"] = bool(include_rating)
    if keep_underscores is not None:
        extra["keep_underscores"] = bool(keep_underscores)
    return {**(spec_overrides or {}), **extra}


def estimate_caption_cost(captioner_key: str, gemini_model: str, n_images: int) -> str:
    """Markdown cost line for captioning `n_images` with this captioner."""
    spec = CAPTIONERS_BY_KEY[captioner_key]
    if spec.backend != "gemini":
        return f"**Cost:** {spec.cost_note}"
    model = (gemini_model or spec.model).strip()
    price = CAPTION_IMAGE_PRICES.get(model, DEFAULT_CAPTION_PRICE)
    known = model in CAPTION_IMAGE_PRICES
    if not n_images:
        return (f"**Cost:** ~${price:.4f}/image on `{model}` — select images for a total "
                f"(billed by Google to your own API key)")
    approx = "" if known else " (unlisted model — priced as Flash)"
    return (f"**Cost:** ~${n_images * price:.2f} for {n_images} image(s) on `{model}`"
            f"{approx} — build-time estimate, billed by Google to your own API key. "
            f"[Check current pricing](https://ai.google.dev/pricing)")


class Captioner:
    def __init__(self, key: str, model_override: str = "",
                 spec_overrides: dict | None = None) -> None:
        self.spec: CaptionerSpec = CAPTIONERS_BY_KEY[key]
        # spec_overrides carries runtime config for the "custom" endpoint
        # (base_url / api_key_env / min_interval_s / model) so users can point
        # at any OpenAI-compatible server without editing config.py.
        if spec_overrides:
            self.spec = self.spec.model_copy(update=spec_overrides)
        # Let the UI pick a specific model (e.g. a live-refreshed Gemini model,
        # or a custom OpenAI-compatible endpoint's model) without editing config.
        self.model = (model_override or self.spec.model).strip()
        self._model = None
        self._processor = None
        self._tagger = None
        self._last_request = 0.0

    # ---------- shared ----------

    def caption(self, image_path: Path, subject: str = "the character",
                style: str = "prose", dataset_type: str = "character",
                sparse: bool = False) -> str:
        if self.spec.backend == "wd_tagger":
            # A dedicated tagger emits canonical tags directly from image features;
            # the prose/tags/e621 selector, the subject, and the dataset type (a
            # framing hint for VLMs) don't apply to it.
            return ", ".join(self._load_tagger().tag(image_path))
        instruction = self.spec.prompt_for(style, dataset_type, sparse).format(subject=subject)
        if self.spec.backend == "openai":
            return _clean(self._caption_openai(image_path, instruction))
        if self.spec.backend == "gemini":
            return _clean(self._caption_gemini(image_path, instruction))
        return _clean(self._caption_transformers(image_path, instruction))

    def load(self) -> None:
        if self.spec.backend == "transformers":
            self._load_transformers()
        elif self.spec.backend == "wd_tagger":
            self._load_tagger()

    def _load_tagger(self):
        if self._tagger is None:
            from studio.tagger import Tagger

            self._tagger = Tagger(self.spec.hf_id, self.spec.general_threshold,
                                  self.spec.character_threshold,
                                  self.spec.tags_file, self.spec.tag_scheme,
                                  include_rating=self.spec.include_rating,
                                  keep_underscores=self.spec.keep_underscores)
        self._tagger.load()
        return self._tagger

    def unload(self) -> None:
        if self._tagger is not None:
            self._tagger.unload()
            self._tagger = None
        if self._model is None:
            return
        import gc

        import torch

        del self._model, self._processor
        self._model = self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- local transformers backend ----------

    def _load_transformers(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        # One GPU: "auto" is what we want — it offloads whatever does not fit.
        # Two or more: "auto" shards the model across cards and the forward pass
        # dies with a cuda:0/cuda:1 tensor mismatch, so pin to a single device.
        # LDS_TORCH_DEVICE overrides either way.
        device_map = settings.torch_device or (
            "auto" if torch.cuda.device_count() <= 1 else config.torch_device()
        )
        self._processor = AutoProcessor.from_pretrained(self.spec.hf_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.spec.hf_id, dtype=dtype, device_map=device_map
        )
        self._model.eval()

    def _caption_transformers(self, image_path: Path, instruction: str) -> str:
        import torch

        self._load_transformers()
        image = Image.open(image_path).convert("RGB")
        messages = []
        if self.spec.prompt_style == "llava":  # JoyCaption expects its system prompt
            messages.append({"role": "system", "content": _JOYCAPTION_SYSTEM})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        )
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=300, do_sample=True, temperature=0.6, top_p=0.9
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True)

    # ---------- Gemini backend (costs billed to your Google API key) ----------

    def _caption_gemini(self, image_path: Path, instruction: str) -> str:
        key = settings.resolved_gemini_key()
        if not key:
            raise RuntimeError(
                f"{self.spec.label} needs GEMINI_API_KEY (or LDS_GEMINI_API_KEY in .env). "
                f"Get one at https://aistudio.google.com/apikey"
            )
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=self.model or "gemini-flash-latest",
            contents=[
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/png"),
                instruction,
            ],
        )
        if not resp.text:
            raise RuntimeError(f"{self.spec.label}: empty response (content refusal?)")
        return resp.text

    # ---------- OpenAI-compatible backend (Groq, LM Studio, Ollama) ----------

    def _caption_openai(self, image_path: Path, instruction: str) -> str:
        if not self.spec.base_url:
            raise RuntimeError(
                f"{self.spec.label} has no endpoint URL. Enter a base URL "
                f"(e.g. https://openrouter.ai/api/v1) and save it first."
            )
        if self.spec.min_interval_s:  # respect free-tier rate limits
            wait = self.spec.min_interval_s - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)

        headers = {"Content-Type": "application/json"}
        if self.spec.api_key_env:
            key = settings.resolved_key(self.spec.api_key_env)
            if not key:
                raise RuntimeError(
                    f"{self.spec.label} needs the {self.spec.api_key_env} env var (or "
                    f"LDS_{self.spec.api_key_env} in .env)."
                )
            headers["Authorization"] = f"Bearer {key}"

        model = self.model or self._first_served_model(headers)
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        payload = {
            "model": model,
            "max_tokens": self.spec.max_tokens,
            "temperature": 0.6,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": instruction},
                    ],
                }
            ],
            **self.spec.extra_params,  # e.g. {"reasoning_effort": "none"} for Groq Qwen
        }
        for attempt in range(4):
            r = httpx.post(f"{self.spec.base_url}/chat/completions", json=payload,
                           headers=headers, timeout=120)
            self._last_request = time.monotonic()
            if r.status_code == 429:  # rate limited: back off and retry
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"{self.spec.label}: still rate-limited after 4 attempts")

    def _first_served_model(self, headers: dict) -> str:
        try:
            r = httpx.get(f"{self.spec.base_url}/models", headers=headers, timeout=15)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Could not reach {self.spec.label} at {self.spec.base_url} — is the "
                f"server running with a vision model loaded? ({e})"
            )
        models = r.json().get("data", [])
        if not models:
            raise RuntimeError(f"No model loaded at {self.spec.base_url} — load one first.")
        return models[0]["id"]


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(here is|here's|caption:|tags:|sure[,!.]?)\s*", "", text, flags=re.I)
    return text.strip().strip('"')


def _normalize_tags(text: str) -> list[str]:
    """Split a model's tag output into a clean, deduped list.

    Tolerates newline- or comma-separated tags, lowercases them, trims stray
    punctuation, drops empties, and preserves first-seen order.
    """
    seen: list[str] = []
    for chunk in re.split(r"[,\n]", text):
        tag = re.sub(r"\s+", " ", chunk).strip().strip(".").strip().lower()
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def _finalize_tags(raw: str, trigger: str) -> str:
    """Trigger-first, comma-separated tag caption.

    Identity lives in the trigger, so (unlike the prose path) the character name
    is not injected as a tag. A trigger the model happened to emit is de-duped so
    it appears exactly once, first, in its intended casing.
    """
    tags = [t for t in _normalize_tags(raw) if not (trigger and t == trigger.lower())]
    if trigger:
        return ", ".join([trigger, *tags]) if tags else trigger
    return ", ".join(tags)


def apply_affixes(caption: str, prefix: str, suffix: str, style: str) -> str:
    """Wrap a finalized caption with fixed prefix/suffix text.

    Mainly for tag datasets — e.g. Pony's `score_9, score_8_up, …` quality prefix
    or a `masterpiece, best quality` Danbooru prefix — so those constant tags ride
    on every caption without re-tagging. Joined with a comma for tag styles (they
    ARE tags) and a space for prose; empty prefix/suffix are no-ops.
    """
    prefix, suffix = prefix.strip().strip(","), suffix.strip().strip(",")
    if not prefix and not suffix:
        return caption
    sep = ", " if style in ("tags", "e621") else " "
    return sep.join(p for p in (prefix, caption, suffix) if p)


def parse_blacklist(text: str) -> list[str]:
    """Normalize a user drop-list into comparable tag tokens.

    Accepts a comma- or newline-separated list. Each entry is lowercased,
    whitespace-collapsed and underscores→spaces, so it matches finalized tag
    output (`long hair`) whether the user typed `long_hair` or `Long Hair`.
    """
    out: list[str] = []
    for chunk in re.split(r"[,\n]", text or ""):
        tag = re.sub(r"\s+", " ", chunk).strip().strip(".").lower().replace("_", " ")
        if tag and tag not in out:
            out.append(tag)
    return out


def drop_blacklisted_tags(caption: str, blacklist: list[str], style: str) -> str:
    """Remove blacklisted tags from a comma-separated tag caption.

    A no-op for prose (`style` not a tag style) or an empty list. The first tag
    is the trigger and is always kept — the drop-list is for noisy descriptor
    tags a tagger loves (`simple background`, `signature`, `watermark`), not the
    identity token. Matching is on the normalized form, so casing/underscores
    don't matter.
    """
    if style not in ("tags", "e621") or not blacklist:
        return caption
    drop = set(blacklist)
    parts = [p.strip() for p in caption.split(",") if p.strip()]
    kept = [p for i, p in enumerate(parts)
            if i == 0 or p.lower().replace("_", " ") not in drop]
    return ", ".join(kept)


def _has_caption(image: Path) -> bool:
    """True if the image already has a non-empty .txt sidecar.

    Reads through `read_caption`, so `--skip-captioned` is not defeated by a
    sidecar this process did not write (a cp1252 one used to raise instead).
    """
    from studio.config import read_caption

    return image.with_suffix(".txt").exists() and bool(read_caption(image))


def finalize_caption(raw: str, trigger: str, character_name: str, aliases: list[str],
                     style: str = "prose", dataset_type: str = "character") -> str:
    """Apply the dataset caption template: trigger first, consistent naming.

    `style="tags"` (Danbooru) and `style="e621"` (furry/anthro) both produce a
    comma-separated tag caption — they differ only in the vocabulary the model
    was asked for, not the output shape — so they share the tag finalizer (already
    identity-free, so `dataset_type` doesn't change it). The default "prose" keeps
    the natural-language behaviour.

    `dataset_type="character"` (the default) is unchanged. Style/Concept prose is
    trigger-first with **no** alias→name replacement: there is no "the woman"→name
    mapping for a look or an object, so only the trigger is placed first.
    """
    if style in ("tags", "e621"):
        return _finalize_tags(raw, trigger)
    caption = raw
    if dataset_type == "character" and character_name:
        # Replace generic nouns the VLM used for the subject with the real name
        for alias in aliases:
            caption = re.sub(rf"\bthe {re.escape(alias)}\b", character_name, caption, flags=re.I)
    if trigger and not caption.lower().startswith(trigger.lower()):
        keep_name = (dataset_type == "character" and character_name
                     and caption.startswith(character_name.split()[0]))
        if caption and not keep_name:
            caption = f"{caption[0].lower()}{caption[1:]}"  # don't lowercase a proper name
        caption = f"{trigger}, {caption}" if caption else trigger
    return caption


SUBJECT_ALIASES = ["character", "creature", "figure", "subject", "person", "woman", "man"]


def caption_images(
    images: list[Path],
    captioner_key: str,
    character_name: str = "",
    trigger: str = "",
    progress: Callable[[str], None] = print,
    model_override: str = "",
    spec_overrides: dict | None = None,
    style: str = "prose",
    prefix: str = "",
    suffix: str = "",
    skip_existing: bool = False,
    blacklist: str = "",
    dataset_type: str = "character",
    sparse: bool = False,
) -> list[tuple[Path, str]]:
    """Caption a list of images. Standalone — no run/pipeline state needed.

    `style` is "prose" (natural language), "tags" (Danbooru comma list) or
    "e621" (furry/anthro comma list). `dataset_type` ("character"/"style"/
    "concept") frames what the caption describes; `sparse` is a Style-only
    minimal-caption variant. `prefix`/`suffix` wrap every caption (for
    quality/score tags); `blacklist` is a comma/newline drop-list of noisy tags
    to strip (tag styles only); `skip_existing` leaves images that already have a
    non-empty .txt untouched. Returns (image_path, finalized_caption) pairs; the
    captioner model is loaded once and freed afterwards.
    """
    subject = character_name or "the character"
    drop = parse_blacklist(blacklist)
    if skip_existing:
        kept = [p for p in images if not _has_caption(p)]
        if len(kept) != len(images):
            progress(f"Skipping {len(images) - len(kept)} image(s) that already "
                     f"have a caption.")
        images = kept
    if not images:
        progress("Nothing to caption.")
        return []
    cap = Captioner(captioner_key, model_override=model_override,
                    spec_overrides=spec_overrides)
    # A dedicated tagger always emits Danbooru tags, whatever `style` requested.
    style = "tags" if cap.spec.backend == "wd_tagger" else style
    if cap.spec.backend == "transformers":
        from studio import comfy_api

        if comfy_api.is_up():
            progress("Freeing ComfyUI VRAM for the captioner...")
            comfy_api.free_vram()
        from studio import isolate

        isolate.unload()  # SAM3 and a 17 GB captioner don't co-fit on most GPUs
        progress(f"Loading captioner {cap.spec.label} (first run downloads weights)...")
        cap.load()
    elif cap.spec.backend == "wd_tagger":
        progress(f"Loading {cap.spec.label} (first run downloads weights)...")
        cap.load()
    else:
        progress(f"Captioning via {cap.spec.label}...")

    items: list[tuple[Path, str]] = []
    try:
        for i, p in enumerate(images, 1):
            progress(f"Captioning {i}/{len(images)}: {p.name}")
            raw = cap.caption(p, subject=subject, style=style,
                              dataset_type=dataset_type, sparse=sparse)
            cap_text = finalize_caption(raw, trigger, character_name, SUBJECT_ALIASES,
                                        style=style, dataset_type=dataset_type)
            # Drop noisy tags before affixes so a fixed prefix/suffix survives.
            cap_text = drop_blacklisted_tags(cap_text, drop, style)
            items.append((p, apply_affixes(cap_text, prefix, suffix, style)))
    finally:
        cap.unload()
    return items


def caption_folder(
    folder: Path,
    captioner_key: str,
    character_name: str = "",
    trigger: str = "",
    only: list[Path] | None = None,
    progress: Callable[[str], None] = print,
    model_override: str = "",
    spec_overrides: dict | None = None,
    style: str = "prose",
    prefix: str = "",
    suffix: str = "",
    skip_existing: bool = False,
    blacklist: str = "",
    dataset_type: str = "character",
    sparse: bool = False,
) -> list[tuple[Path, str]]:
    """Caption images in `folder` (all, or the subset in `only`) and write
    .txt sidecars next to each image. The classic 'point at a folder and tag
    it' mode. `style` is "prose", "tags" (Danbooru) or "e621"; `dataset_type`
    frames the caption (character/style/concept) and `sparse` is a Style-only
    minimal-caption variant; `prefix`/`suffix` wrap every caption; `blacklist`
    drops noisy tags (tag styles only); `skip_existing` leaves already-captioned
    images alone."""
    from studio.config import list_images

    images = only if only else list_images(folder)
    if not images:
        raise RuntimeError(f"No images found in {folder}")
    items = caption_images(images, captioner_key, character_name, trigger, progress,
                           model_override=model_override, spec_overrides=spec_overrides,
                           style=style, prefix=prefix, suffix=suffix,
                           skip_existing=skip_existing, blacklist=blacklist,
                           dataset_type=dataset_type, sparse=sparse)
    for img, caption in items:
        img.with_suffix(".txt").write_text(caption, encoding="utf-8")
    progress(f"Wrote {len(items)} .txt sidecar(s) in {folder}")
    return items
