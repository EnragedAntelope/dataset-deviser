"""Cloud engine: Gemini image models (Nano Banana family) via google-genai.

Costs are billed by Google to YOUR API key. Prices shown in the app are
estimates captured at build time — always check current Google pricing.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from studio.config import (
    CLOUD_IMAGE_PRICES,
    DEPRECATED_IMAGE_MODELS,
    MODEL_CACHE_FILE,
    api_image,
    friendly_api_error,
    gemini_client,
    is_transient_api_error,
    load_caption_model_cache,
    load_cloud_model_cache,
    save_caption_model_cache,
    save_cloud_model_cache,
    settings,
)
from studio.engines.base import GenerationError
from studio.shotplan import Shot

MAX_REFERENCE_IMAGES = 14
# References are capped here before upload. ① already outputs 1024 px, but ②
# also takes files straight from the user (a pasted 12 MP photo), and every
# current Gemini image model works at 1-2K.
REFERENCE_MAX_SIDE = 2048

# What the "auto" model setting resolves to: the first of these the model list
# offers. Image models have no "-latest" alias, so this list is the one place a
# release moves the default. Order set by an identity A/B on real references.
AUTO_MODEL = "auto"
IMAGE_MODEL_PREFERENCE = [
    "gemini-3-pro-image",  # Nano Banana Pro: 5 character refs
    "gemini-nano-banana-2.1",
    "gemini-3.1-flash-image",  # Nano Banana 2
]
# 3 attempts at 2s/4s. Image generation is slow and billed per call, so this stays
# shorter than the captioner's ladder — enough to ride out a spike, not enough to
# quietly burn a user's budget retrying a model that is genuinely down.
GENERATE_RETRIES = 3
GENERATE_BACKOFF_S = 2.0

# Re-exported so tests can monkeypatch a single module-level path.
MODEL_CACHE_FILE = MODEL_CACHE_FILE


def _load_model_cache() -> list[dict] | None:
    """Thin wrapper around the config cache, kept here for testability."""
    return load_cloud_model_cache()


def _save_model_cache(models: list[dict]) -> None:
    """Thin wrapper around the config cache, kept here for testability."""
    save_cloud_model_cache(models)


# Shut-down ids still named in someone's .env (LDS_GEMINI_IMAGE_MODEL) or a saved
# CLI script, mapped to the GA release that replaced them.
_RETIRED_IMAGE_MODELS = {
    "gemini-3-pro-image-preview": "gemini-3-pro-image",
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image",
}


def _model_label(name: str) -> str:
    # Priced from the table at display time, never from the cache: a cached
    # price outlives the release that corrected it.
    price = CLOUD_IMAGE_PRICES.get(name)
    if price is None:
        return f"{name}  (price unknown)"
    note = ", deprecated" if name in DEPRECATED_IMAGE_MODELS else ""
    return f"{name}  (~${price:.3f}/img est.{note})"


def _labelled(ids: list[str]) -> list[tuple[str, str]]:
    return [(_model_label(m), m) for m in ids]


def _fallback_models() -> list[tuple[str, str]]:
    """Static fallback when live listing is impossible."""
    return _labelled(list(CLOUD_IMAGE_PRICES))


def _known_ids() -> list[str]:
    """Model ids from the fresh cache, else the price table. Never a network call."""
    cached = _load_model_cache()
    return [m["model_id"] for m in cached] if cached else list(CLOUD_IMAGE_PRICES)


def known_image_models() -> list[tuple[str, str]]:
    """`list_image_models` without the network: for building the UI at startup."""
    return _labelled(_known_ids())


def resolve_image_model(model: str = "", available: list[str] | None = None) -> str:
    """The concrete model id for `model` ("" or "auto" = best available).

    Resolves against `available`, else the cached/static list, so cost estimates
    can call it on every keystroke without touching the network.
    """
    model = model or settings.gemini_image_model
    if model != AUTO_MODEL:
        return _RETIRED_IMAGE_MODELS.get(model, model)
    ids = available if available is not None else _known_ids()
    return next((m for m in IMAGE_MODEL_PREFERENCE if m in ids), IMAGE_MODEL_PREFERENCE[0])


def image_price(model: str = "") -> tuple[str, float | None]:
    """(resolved model id, estimated USD per 1K image or None if unknown)."""
    resolved = resolve_image_model(model)
    return resolved, CLOUD_IMAGE_PRICES.get(resolved)


def image_model_choices(models: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Dropdown choices: "Auto" first, labelled with what it resolves to."""
    auto = resolve_image_model(AUTO_MODEL, [m for _, m in models])
    return [(f"Auto (recommended) → {_model_label(auto)}", AUTO_MODEL), *models]


def _is_image_model(name: str) -> bool:
    # imagen = text-to-image only, no reference editing. "nano-banana" ids carry
    # no "image" in the name (gemini-nano-banana-2.1).
    return ("image" in name or "nano-banana" in name) and "imagen" not in name


def list_image_models(force_refresh: bool = False) -> list[tuple[str, str]]:
    """Return image-capable Gemini models as [(display_label, model_id), ...].

    Uses a 24-hour local cache so the UI dropdown loads instantly. If the
    cache is stale/missing, live-pull from the API and persist. Falls back to
    the static price table if the API is unreachable or no key is configured.
    """
    if not force_refresh:
        cached = _load_model_cache()
        if cached:
            return _labelled([m["model_id"] for m in cached])

    key = settings.resolved_gemini_key()
    if not key:
        return _fallback_models()

    try:
        client = gemini_client(key)
        found: list[dict] = []
        for m in client.models.list():
            name = m.name.removeprefix("models/")
            if not _is_image_model(name):
                continue
            # Skip deprecated / shutdown models when the API lists them.
            if any(tag in name for tag in ("-shut-down", "deprecated", "-experimental")):
                continue
            found.append(
                {
                    "model_id": name,
                    "display_name": getattr(m, "display_name", name),
                    "cached_at": datetime.now(tz=timezone.utc).isoformat(),
                }
            )
        # A "-preview" id the API still lists next to its GA release is the
        # retired one (gemini-3-pro-image-preview, shut down 2026-06-25).
        ids = {f["model_id"] for f in found}
        found = [f for f in found
                 if not (f["model_id"].endswith("-preview")
                         and f["model_id"].removesuffix("-preview") in ids)]
        if found:
            _save_model_cache(found)
            return _labelled([f["model_id"] for f in found])
    except Exception:
        # Live pull failed; try stale cache as a last resort before falling
        # back to the static dict.
        stale = _load_model_cache()
        if stale:
            return _labelled([m["model_id"] for m in stale])

    return _fallback_models()


# Gemini caption models the app falls back to when the API can't be listed.
# All current Gemini text models accept image input; "-latest" aliases are
# safest as a default because they don't 404 when a pinned version is retired.
_CAPTION_FALLBACK = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    # Pinned fallbacks. Not 2.5: new projects no longer get access to it.
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]


def list_caption_models(force_refresh: bool = False) -> list[tuple[str, str]]:
    """Return vision-capable Gemini caption models as [(label, model_id), ...].

    Same 24-hour-cache + stale-fallback strategy as `list_image_models`, but
    filtered to text/vision models that support `generateContent` (excludes
    image-generation, embedding, TTS, and audio/live models).
    """
    def _plain(ids: list[str]) -> list[tuple[str, str]]:
        return [(m, m) for m in ids]

    if not force_refresh:
        cached = load_caption_model_cache()
        if cached:
            return _plain([m["model_id"] for m in cached])

    key = settings.resolved_gemini_key()
    if not key:
        return _plain(_CAPTION_FALLBACK)

    try:
        client = gemini_client(key)
        found: list[dict] = []
        for m in client.models.list():
            name = m.name.removeprefix("models/")
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" not in actions:
                continue
            # Keep chat/vision Gemini models; drop image-gen, embeddings, TTS,
            # and audio/live variants that can't caption a still image.
            if not name.startswith("gemini"):
                continue
            if any(t in name for t in ("image", "imagen", "embedding", "tts",
                                       "audio", "live", "-omni")):
                continue
            found.append({"model_id": name,
                          "display_name": getattr(m, "display_name", name)})
        if found:
            # Surface the rolling "-latest" aliases first so the safe default
            # is at the top of the dropdown.
            found.sort(key=lambda f: (0 if f["model_id"].endswith("latest") else 1,
                                      f["model_id"]))
            save_caption_model_cache(found)
            return _plain([f["model_id"] for f in found])
    except Exception:
        stale = load_caption_model_cache()
        if stale:
            return _plain([m["model_id"] for m in stale])

    return _plain(_CAPTION_FALLBACK)


class GeminiEngine:
    name = "gemini"

    def __init__(self, model: str = "") -> None:
        key = settings.resolved_gemini_key()
        if not key:
            raise GenerationError(
                "No Gemini API key found. Set GEMINI_API_KEY (or LDS_GEMINI_API_KEY in "
                ".env) to use the cloud engine, or switch to the local ComfyUI engine. "
                "Get a key at https://aistudio.google.com/apikey"
            )
        # gemini_client defers the google-genai import, so local-only installs never
        # need it configured.
        self._client = gemini_client(key)
        self._model = resolve_image_model(model)

    def generate(self, sources: list[Path], shot: Shot, out_path: Path, seed: int) -> Path:
        from google.genai import types

        parts: list = []
        for p in sources[:MAX_REFERENCE_IMAGES]:
            data, mime = api_image(p, REFERENCE_MAX_SIDE)
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        parts.append(shot.cloud_prompt)

        last_err: Exception | None = None
        for attempt in range(GENERATE_RETRIES):
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                    ),
                )
                for cand in resp.candidates or []:
                    for part in cand.content.parts or []:
                        if getattr(part, "inline_data", None) and part.inline_data.data:
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_path.write_bytes(part.inline_data.data)
                            return out_path
                # No image part usually means a content refusal
                text = "".join(
                    p.text or ""
                    for c in (resp.candidates or [])
                    for p in (c.content.parts or [])
                    if getattr(p, "text", None)
                )
                raise GenerationError(f"no image returned ({text[:200] or 'refused'})")
            except GenerationError as e:
                last_err = e
                break  # refusals don't get better with a retry
            except Exception as e:  # transient API errors
                last_err = e
                if attempt == GENERATE_RETRIES - 1 or not is_transient_api_error(e):
                    break
                # An immediate retry against a demand spike (503) just fails again.
                time.sleep(GENERATE_BACKOFF_S * (2 ** attempt))
        raise GenerationError(f"shot {shot.id}: {friendly_api_error(last_err)}"
                              if last_err else f"shot {shot.id}: failed")
