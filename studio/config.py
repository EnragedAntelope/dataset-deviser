"""Settings and captioner/engine registries.

Everything here is overridable via environment variables prefixed LDS_
(or a .env file in the repo root — see .env.example).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env into the process environment so unprefixed keys (GEMINI_API_KEY,
# GROQ_API_KEY, HF_TOKEN for the gated SAM3 download) work from there too.
load_dotenv(REPO_ROOT / ".env")

# Base captioning instruction, tuned for LoRA training captions: describe what
# VARIES between images (pose/angle/setting/lighting), not the subject's fixed
# identity — identity is what the trigger token learns.
_BASE_PROMPT = (
    "Describe this image in one flowing paragraph of natural language for use as an "
    'image-generation training caption. Refer to the main subject as "{subject}". '
    "Describe the subject's pose, the camera angle and framing, the setting, and the "
    "lighting. If the background is plain, just say so briefly. Do not describe the "
    "subject's fixed physical appearance in exhaustive detail (identity is learned "
    "separately); focus on what varies between images. Do not mention that this is "
    "AI-generated, a film still, or training data. Output only the caption paragraph, "
    "nothing else."
)

# JoyCaption was trained on a specific instruction format and supports a
# documented "refer to them as X" directive — use its convention, not ours.
_JOYCAPTION_PROMPT = (
    "Write a descriptive caption for this image in a formal tone, as one flowing "
    "paragraph. If there is a person or character in the image you must refer to "
    "them as {subject}. Describe the pose, the camera angle and framing, the "
    "setting, and the lighting, but do not describe the subject's fixed physical "
    "appearance in exhaustive detail. Do not mention that the image is AI-generated, "
    "a film still, or training data."
)

# The NSFW-Caption finetune responds best when explicitness is requested plainly.
# Kept as a shared constant so the Style/Concept composer can reuse the exact
# clause (detecting an NSFW spec by its presence) rather than string-sniffing.
_NSFW_PLAINNESS = (
    " If the image contains nudity or explicit content, describe it plainly and "
    "directly without euphemism."
)
_NSFW_PROMPT = _BASE_PROMPT + _NSFW_PLAINNESS

# --- Booru-tag caption style -------------------------------------------------
# SDXL/Pony/Illustrious LoRAs are trained on comma-separated Danbooru-style tags,
# not prose. These templates produce that format. Same rule as the prose ones:
# tag what VARIES between images (pose/angle/framing/setting/lighting), not the
# subject's fixed identity — identity is what the trigger token learns.
_BASE_TAGS_PROMPT = (
    "List concise Danbooru-style tags describing this image, for use as an "
    'image-generation training caption. Refer to the main subject as "{subject}". '
    "Output a single comma-separated list of lowercase tags (one attribute per tag) "
    "covering the subject's pose, the camera angle and framing (e.g. from side, "
    "from above, from behind, full body, upper body, close-up), the setting or "
    "background, and the lighting. Do not tag the subject's fixed physical identity "
    "in detail (it is learned separately); tag what varies between images. Do not "
    "write sentences or commentary, and do not mention that the image is "
    "AI-generated, a film still, or training data. Output only the comma-separated tags."
)

# JoyCaption has a documented booru-tag mode; keep its "refer to them as X"
# convention while asking for the tag-list format.
_JOYCAPTION_TAGS_PROMPT = (
    "Write a list of Booru-style tags for this image. If there is a person or "
    "character in the image you must refer to them as {subject}. Include tags for "
    "the pose, the camera angle and framing, the setting, and the lighting, but do "
    "not tag the subject's fixed physical appearance in detail. Output only a single "
    "comma-separated list of lowercase tags, no sentences."
)

_NSFW_TAGS_PROMPT = _BASE_TAGS_PROMPT + (
    " If the image contains nudity or explicit content, tag it plainly and "
    "directly without euphemism."
)

# e621 taxonomy (furry / anthro) — the vocabulary Pony Diffusion and most furry
# SDXL checkpoints are trained on. Same comma-list FORMAT as Danbooru but a
# DIFFERENT controlled vocabulary (species, anthro/feral, e621 anatomy + rating
# conventions), so it is a distinct option, not a synonym for the Danbooru one.
# Honesty note: a general VLM only *approximates* e621's vocabulary; a dedicated
# e621-trained tagger is the gold standard (logged in the backlog).
_BASE_E621_PROMPT = (
    "List concise e621-style tags describing this image, for use as an "
    'image-generation training caption. Refer to the main subject as "{subject}". '
    "Output a single comma-separated list of lowercase tags (one attribute per tag) "
    "using e621 tagging conventions: include the subject's species and form "
    "(e.g. anthro, feral, humanoid) when discernible, plus the pose, the camera "
    "angle and framing, the setting or background, and the lighting. Do not tag the "
    "subject's fixed physical identity in detail (it is learned separately); tag "
    "what varies between images. Do not write sentences or commentary, and do not "
    "mention that the image is AI-generated. Output only the comma-separated tags."
)

_JOYCAPTION_E621_PROMPT = (
    "Write a list of e621-style tags for this image. If there is a person or "
    "character in the image you must refer to them as {subject}. Use e621 tagging "
    "conventions including species and form (anthro, feral, humanoid) when "
    "discernible, plus the pose, camera angle and framing, setting, and lighting, "
    "but do not tag the subject's fixed physical appearance in detail. Output only a "
    "single comma-separated list of lowercase tags, no sentences."
)

_NSFW_E621_PROMPT = _BASE_E621_PROMPT + (
    " If the image contains nudity or explicit content, tag it plainly and "
    "directly without euphemism."
)

# --- Dataset types (Character / Style / Concept) -----------------------------
# A dataset is ONE type: the trigger learns an identity (character), an aesthetic
# (style), or an object/action/idea (concept). Character is the tuned default and
# keeps its per-model templates verbatim. Style/Concept captions instead COMPOSE
# their instruction from a per-type *framing* clause + the prose/tags/e621 *format*
# directive, so we don't store nine extra templates per captioner (anti-bloat).
DATASET_TYPES = ("character", "style", "concept")

# Framing: WHAT to describe (and what NOT to). The constant look/form is what the
# trigger learns, so the caption must describe everything *except* it.
_STYLE_FRAMING = (
    "Describe the CONTENT of this image for an image-generation training caption: "
    "the subjects, objects, setting and composition. Do NOT describe the artistic "
    "style, medium, colour palette, line work, or rendering technique — that constant "
    "look is learned from the trigger and must not be spelled out."
)
_STYLE_SPARSE_FRAMING = (
    "Give a very short caption for this image: name only the main subject or content "
    "in at most a few words. Do NOT describe the artistic style, medium, colour, or "
    "rendering technique."
)
_CONCEPT_FRAMING = (
    "Describe the CONTEXT of this image for an image-generation training caption: the "
    "surrounding setting and background, and how the main subject is placed, posed or "
    "used. Do NOT describe the fixed form of the concept itself — that is learned from "
    "the trigger and must not be spelled out."
)
# Format: HOW to shape the output. Shared by both non-character types.
_FORMAT_DIRECTIVE = {
    "prose": "Write one flowing paragraph of natural language. Output only the caption.",
    "tags": "Output a single comma-separated list of concise lowercase Danbooru-style "
            "tags (one attribute per tag), no sentences. Output only the tags.",
    "e621": "Output a single comma-separated list of concise lowercase e621-style tags "
            "(one attribute per tag), no sentences. Output only the tags.",
}
_PLAINNESS_DIRECTIVE = {
    "prose": " If the content is explicit, describe it plainly without euphemism.",
    "tags": " If the content is explicit, tag it plainly without euphemism.",
    "e621": " If the content is explicit, tag it plainly without euphemism.",
}


class CaptionerSpec(BaseModel):
    key: str
    label: str
    # "transformers" (local GPU VLM) | "openai" (Groq / LM Studio / Ollama) |
    # "gemini" (Google) | "wd_tagger" (local ONNX Danbooru tagger)
    backend: str = "transformers"
    hf_id: str = ""  # transformers + wd_tagger backends (HF repo id)
    model: str = ""  # openai/gemini backend model id ("" = server default / first listed)
    base_url: str = ""  # openai backend endpoint
    api_key_env: str = ""  # env var holding the key ("" = no key needed)
    min_interval_s: float = 0.0  # request spacing (free-tier rate limits)
    max_tokens: int = 400  # completion budget (raise for reasoning models)
    # Extra JSON body fields merged into the openai request — e.g. Groq's
    # {"reasoning_effort": "none"} to switch off a thinking model's scratchpad
    # so we get only the caption (and don't burn the token budget on reasoning).
    extra_params: dict = {}
    prompt_template: str = _BASE_PROMPT  # prose style; must contain {subject}
    # Danbooru-tag style (comma list for SDXL / Illustrious / NoobAI trainers).
    tags_template: str = _BASE_TAGS_PROMPT  # must contain {subject}
    # e621-tag style (comma list for Pony / furry checkpoints).
    e621_template: str = _BASE_E621_PROMPT  # must contain {subject}
    # "qwen_vl" and "llava" share one transformers code path but need
    # different chat-template quirks (JoyCaption wants its system prompt).
    prompt_style: str = "qwen_vl"
    # wd_tagger backend: probability cut-offs for general vs character tags,
    # plus the tag-CSV filename and vocabulary scheme ("danbooru" | "e621").
    general_threshold: float = 0.35
    character_threshold: float = 0.85
    tags_file: str = "selected_tags.csv"
    tag_scheme: str = "danbooru"
    # Opt-in tagger extras (set at runtime from the ③ Tag options): append the
    # top rating tag (Danbooru only), and keep raw booru underscores.
    include_rating: bool = False
    keep_underscores: bool = False
    vram_note: str = ""
    nsfw_capable: bool = True
    cost_note: str = "free"

    def prompt_for(self, style: str, dataset_type: str = "character",
                   sparse: bool = False) -> str:
        """The instruction template for the requested caption style + dataset type.

        `style="tags"` selects the Danbooru-tag template, `style="e621"` the
        e621 (furry/anthro) one; anything else (the default "prose") selects the
        natural-language template. Unknown styles fall back to prose so a bad
        value never crashes captioning.

        `dataset_type="character"` (the default) returns the model's tuned
        per-style templates **verbatim** — no behaviour change. `style`/`concept`
        instead COMPOSE the instruction from a per-type framing clause + the same
        format directive, so a Style/Concept dataset is captioned about its content/
        context rather than its (trigger-learned) look. `sparse` is a Style-only
        minimal-caption variant; it is ignored for the other types.
        """
        if dataset_type in ("style", "concept"):
            return self._compose_prompt(style, dataset_type, sparse)
        if style == "tags":
            return self.tags_template
        if style == "e621":
            return self.e621_template
        return self.prompt_template

    def _compose_prompt(self, style: str, dataset_type: str, sparse: bool) -> str:
        """Framing (per type) + format directive (per style), lightly per-model.

        Kept free of extra stored templates: one framing string per type times one
        format directive per style covers all combinations. JoyCaption keeps its
        "refer to them as {subject}" convention (via `prompt_style`); an NSFW spec
        (detected by its shared plainness clause) keeps a plainness directive.
        """
        fmt = style if style in _FORMAT_DIRECTIVE else "prose"
        if dataset_type == "style":
            framing = _STYLE_SPARSE_FRAMING if sparse else _STYLE_FRAMING
        else:  # concept
            framing = _CONCEPT_FRAMING
        parts = [framing]
        if self.prompt_style == "llava":  # JoyCaption's documented convention
            parts.append("If a person or character is present, refer to them as "
                         '"{subject}".')
        parts.append(_FORMAT_DIRECTIVE[fmt])
        if _NSFW_PLAINNESS in self.prompt_template:  # NSFW-specialist spec
            parts.append(_PLAINNESS_DIRECTIVE[fmt].strip())
        return " ".join(parts)


CAPTIONERS: list[CaptionerSpec] = [
    CaptionerSpec(
        key="qwen3vl",
        label="Local: Qwen3-VL-8B Instruct (heretic)",
        hf_id="heretic-org/Qwen-3-VL-8B-Instruct-heretic",
        vram_note="~17 GB bf16",
    ),
    CaptionerSpec(
        key="joycaption",
        label="Local: JoyCaption Beta One",
        hf_id="fancyfeast/llama-joycaption-beta-one-hf-llava",
        prompt_style="llava",
        prompt_template=_JOYCAPTION_PROMPT,
        tags_template=_JOYCAPTION_TAGS_PROMPT,
        e621_template=_JOYCAPTION_E621_PROMPT,
        vram_note="~17 GB bf16",
    ),
    CaptionerSpec(
        key="qwen3vl-nsfw",
        label="Local: Qwen3-VL-8B NSFW-Caption V4.5",
        hf_id="Disty0/Qwen3-VL-8B-NSFW-Caption-V4.5",
        prompt_template=_NSFW_PROMPT,
        tags_template=_NSFW_TAGS_PROMPT,
        e621_template=_NSFW_E621_PROMPT,
        vram_note="~17 GB bf16",
    ),
    # Dedicated taggers: emit canonical Danbooru tags directly (no VLM prose).
    # They ignore the prose/tags/e621 style selector — a tagger always produces
    # Danbooru tags. Needs `onnxruntime` (optional dep); weights download once.
    CaptionerSpec(
        key="wd-eva02",
        label="Local tagger: WD EVA02-Large v3 (canonical Danbooru tags)",
        backend="wd_tagger",
        hf_id="SmilingWolf/wd-eva02-large-tagger-v3",
        vram_note="~1.4 GB ONNX (needs onnxruntime)",
        cost_note="free (local tagger)",
    ),
    CaptionerSpec(
        key="wd-vit",
        label="Local tagger: WD ViT v3 (Danbooru tags, lighter/faster)",
        backend="wd_tagger",
        hf_id="SmilingWolf/wd-vit-tagger-v3",
        vram_note="~0.4 GB ONNX (needs onnxruntime)",
        cost_note="free (local tagger)",
    ),
    CaptionerSpec(
        key="z3d-e621",
        label="Local tagger: Z3D e621 ConvNeXt (canonical e621/furry tags)",
        backend="wd_tagger",
        hf_id="toynya/Z3D-E621-Convnext",
        tags_file="tags-selected.csv",
        tag_scheme="e621",
        vram_note="~0.4 GB ONNX (needs onnxruntime)",
        cost_note="free (local tagger)",
    ),
    CaptionerSpec(
        key="gemini-flash",
        label="Cloud: Gemini Flash (Google API key, SFW)",
        backend="gemini",
        # Rolling alias so this never 404s when a pinned version is retired.
        # The Caption tab can live-refresh and pick a specific model.
        model="gemini-flash-latest",
        api_key_env="GEMINI_API_KEY",
        nsfw_capable=False,
        cost_note="billed by Google to your key (~$0.001/img est. — check current pricing)",
    ),
    CaptionerSpec(
        key="groq-qwen3.6",
        label="Cloud: Groq Qwen3.6 27B (free tier, SFW)",
        backend="openai",
        base_url="https://api.groq.com/openai/v1",
        model="qwen/qwen3.6-27b",
        api_key_env="GROQ_API_KEY",
        min_interval_s=3.0,  # respect the free-tier 8K TPM limit
        nsfw_capable=False,
        cost_note="free tier (rate-limited; 8K TPM)",
        # Qwen3.6 is a reasoning model; "none" disables its <think> scratchpad
        # so the response is just the caption. Keep a slightly higher token
        # budget as insurance if a future model ignores the flag.
        max_tokens=800,
        extra_params={"reasoning_effort": "none"},
    ),
    CaptionerSpec(
        key="lmstudio",
        label="Advanced: LM Studio (whatever vision model is loaded)",
        backend="openai",
        base_url="http://127.0.0.1:1234/v1",
    ),
    CaptionerSpec(
        key="ollama",
        label="Advanced: Ollama (first vision model served)",
        backend="openai",
        base_url="http://127.0.0.1:11434/v1",
    ),
    CaptionerSpec(
        key="custom",
        label="Advanced: Custom OpenAI-compatible endpoint (you configure it)",
        backend="openai",
        # base_url / model / api_key_env / min_interval_s are supplied at
        # runtime from the Caption tab (persisted, minus the key, in user_config).
        cost_note="billed to you by whoever runs the endpoint",
    ),
]

CAPTIONERS_BY_KEY = {c.key: c for c in CAPTIONERS}

ENGINES = {
    "gemini": "Cloud - Gemini image model (best identity fidelity, SFW only)",
    "comfyui": "Local - ComfyUI Qwen Image Edit 2511 (free, private, uncensored)",
}

# Known Gemini image models -> USD per standard-resolution (1K) image.
# ESTIMATES captured at build time — actual costs are billed by Google to the
# user's own API key; the UI can live-pull the current model list.
CLOUD_IMAGE_PRICES = {
    "gemini-3-pro-image-preview": 0.134,  # Nano Banana Pro (1K-2K)
    "gemini-3.1-flash-image-preview": 0.067,  # Nano Banana 2 (1K)
    "gemini-2.5-flash-image": 0.039,  # Nano Banana (1K)
}

# Known Gemini caption models -> USD per captioned image. ESTIMATES captured at
# build time, derived from published token pricing: ~1290 tokens for a 1K image
# in, ~120 tokens of caption out. Actual costs are billed by Google to the
# user's own key — always check current pricing.
CAPTION_IMAGE_PRICES = {
    "gemini-flash-latest": 0.0007,
    "gemini-2.5-flash": 0.0007,
    "gemini-flash-lite-latest": 0.0002,
    "gemini-2.5-flash-lite": 0.0002,
}
# Applied to a Gemini caption model we have no specific price for, so the
# estimate degrades to "about a Flash" rather than silently reporting $0.
DEFAULT_CAPTION_PRICE = 0.0007

# Local cache for the live Gemini model list so the dropdown can be populated
# without an API call on every UI load. 24-hour TTL; falls back to stale cache,
# then to CLOUD_IMAGE_PRICES if the API is unreachable.
CACHE_DIR = REPO_ROOT / ".cache"
MODEL_CACHE_FILE = CACHE_DIR / "gemini_image_models.json"
CAPTION_MODEL_CACHE_FILE = CACHE_DIR / "gemini_caption_models.json"
MODEL_CACHE_TTL_HOURS = 24


def _load_model_cache_file(path: Path) -> list[dict] | None:
    """Return cached model entries from `path` if it exists and is fresh."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(data["cached_at"])
        age = datetime.now(tz=timezone.utc) - cached_at
        if age > timedelta(hours=MODEL_CACHE_TTL_HOURS):
            return None
        return data.get("models", [])
    except Exception:
        return None


def _save_model_cache_file(path: Path, models: list[dict]) -> None:
    """Persist a live model list to `path` with a timestamp."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"cached_at": datetime.now(tz=timezone.utc).isoformat(), "models": models},
            indent=2,
        ),
        encoding="utf-8",
    )


def load_cloud_model_cache() -> list[dict] | None:
    """Return cached image-model entries if the cache exists and is fresh."""
    return _load_model_cache_file(MODEL_CACHE_FILE)


def save_cloud_model_cache(models: list[dict]) -> None:
    """Persist the live image-model list with a timestamp."""
    _save_model_cache_file(MODEL_CACHE_FILE, models)


def load_caption_model_cache() -> list[dict] | None:
    """Return cached Gemini caption-model entries if fresh."""
    return _load_model_cache_file(CAPTION_MODEL_CACHE_FILE)


def save_caption_model_cache(models: list[dict]) -> None:
    """Persist the live Gemini caption-model list with a timestamp."""
    _save_model_cache_file(CAPTION_MODEL_CACHE_FILE, models)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LDS_", env_file=REPO_ROOT / ".env",
                                      extra="ignore")

    comfy_url: str = "http://127.0.0.1:8188"
    gemini_api_key: str = ""  # also read from GEMINI_API_KEY if unset
    groq_api_key: str = ""  # also read from GROQ_API_KEY if unset
    # Declared so LDS_HF_TOKEN works like the other two keys. Without a field here,
    # `LDS_HF_TOKEN=…` in .env was read by nothing and silently ignored, while the
    # Gemini/Groq aliases worked — an inconsistency users had no way to see.
    hf_token: str = ""  # also read from HF_TOKEN if unset
    gemini_image_model: str = "gemini-3-pro-image-preview"

    # Checks GitHub releases for a newer version at UI launch (cached 24h,
    # best-effort, never blocks). Set false to disable entirely (no network call).
    update_check_enabled: bool = True

    target_long_side: int = 1024
    default_engine: str = "gemini"
    default_captioner: str = "qwen3vl"
    # "builtin" = SAM3 via transformers in-process; "comfyui" = SAM3 workflow
    isolation_backend: str = "builtin"
    sam3_hf_id: str = "facebook/sam3"  # gated: accept license + `hf auth login`
    # Pixels to grow the exclusion mask by before subtracting it from the
    # subject. 0 on purpose: anything outside the subject mask is already
    # whited out, so growing the exclusion can only eat the subject (measured:
    # 7px destroyed 2.75% of the reference character for no gain). Raise only
    # for a prop genuinely merged into the subject segment.
    exclude_dilate_px: int = 0
    # Device for in-process torch models. "" = first CUDA GPU, else CPU.
    # Set e.g. "cuda:1" to pin the second card, or "cpu" to force CPU.
    # Never left to accelerate's "auto": on a multi-GPU box that shards one
    # model across cards and the forward pass dies with "Expected all tensors
    # to be on the same device, but found at least two devices, cuda:0 and
    # cuda:1". Sharding is only worth it for weights too big for one card.
    torch_device: str = ""
    # "auto" = model restore through ComfyUI when reachable, else basic Lanczos;
    # "comfyui" = require ComfyUI; "basic" = never call ComfyUI
    restore_backend: str = "auto"

    output_root: Path = REPO_ROOT / "datasets"
    runs_dir: Path = REPO_ROOT / "runs"
    shot_plans_dir: Path = REPO_ROOT / "shot_plans"

    # Images whose variance-of-Laplacian sharpness score is below this are
    # flagged (advisory only) in the curate/export views. Tune per source set.
    sharpness_blur_threshold: float = 100.0
    # Advisory exposure/contrast flags (curate view). Mean luminance (0-255)
    # at/under dark or at/over bright is flagged; luminance std at/under
    # low_contrast is "low contrast". All advisory — never block.
    dark_luma_threshold: float = 40.0
    bright_luma_threshold: float = 220.0
    low_contrast_threshold: float = 25.0

    # ComfyUI model filenames used by the optional workflow templates
    # (relative to your ComfyUI models folders — see docs/comfyui-setup.md)
    qwen_edit_model: str = "qwen_image_edit_2511_int8_convrot.safetensors"
    angles_lora: str = "qwen/Qwen-Image-Edit-2511-Multiple-Angles-LoRA.safetensors"
    upscale_model: str = "4xNomosWebPhoto_RealPLKSR.safetensors"
    dejpg_model: str = "1xDeJPG_OmniSR.pth"
    sam3_checkpoint: str = "sam3.1_multiplex_fp16.safetensors"

    def resolved_key(self, name: str) -> str:
        import os

        attr = getattr(self, name.lower(), "")
        return attr or os.environ.get(name.upper(), "")

    def resolved_gemini_key(self) -> str:
        return self.resolved_key("gemini_api_key") or self.resolved_key("GEMINI_API_KEY")

    def resolved_groq_key(self) -> str:
        return self.resolved_key("groq_api_key") or self.resolved_key("GROQ_API_KEY")


settings = Settings()


def torch_device() -> str:
    """The single device local torch models load onto.

    `LDS_TORCH_DEVICE` wins; otherwise the first CUDA GPU, else CPU. One device,
    never a shard map — see the `torch_device` setting for why.
    """
    if settings.torch_device:
        return settings.torch_device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def list_images(folder: Path) -> list[Path]:
    """All images directly inside `folder`, sorted by name."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# HTTP statuses a cloud provider returns for "try again", mapped to plain English.
# 503 is by far the common one — a Gemini demand spike, not anything the user did.
_TRANSIENT_STATUS = {
    429: "the provider is rate-limiting your key (429)",
    500: "the provider hit an internal error (500)",
    502: "the provider's gateway failed (502)",
    503: "the model is temporarily overloaded (503)",
    504: "the provider timed out (504)",
}
_RETRY_ADVICE = ("Wait a minute and try again, or switch captioner (Groq / a local "
                 "captioner) in the dropdown.")


def api_status_code(exc: BaseException) -> int | None:
    """The HTTP status behind a cloud SDK exception, if it exposes one.

    `google-genai` puts it on `.code`, `httpx.HTTPStatusError` on `.response.status_code`.
    Anything else returns None and gets the generic message.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def friendly_api_error(exc: BaseException) -> str:
    """One readable sentence for a cloud API failure.

    The raw `google-genai` exception stringifies to its whole JSON error body, which
    is what a user saw in the toast under a full traceback. Transient statuses get a
    named cause and the two things actually worth doing about them; everything else
    degrades to a trimmed message so no failure is ever swallowed.
    """
    reason = _TRANSIENT_STATUS.get(api_status_code(exc) or 0, "")
    if reason:
        return f"{reason}. {_RETRY_ADVICE}"
    text = " ".join(str(exc).split())
    return text[:300] + ("…" if len(text) > 300 else "") or exc.__class__.__name__


def is_transient_api_error(exc: BaseException) -> bool:
    """True when retrying the same call could plausibly succeed.

    Deliberately status-driven: a 400/401/403/404 (bad request, bad key, no access)
    and a content refusal never get better with a retry, so retrying them just makes
    the user wait longer for the same failure.
    """
    return (api_status_code(exc) or 0) in _TRANSIENT_STATUS


def gemini_client(api_key: str):
    """A `google-genai` client that uses only the key we hand it.

    The SDK logs "Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using
    GOOGLE_API_KEY." whenever both env vars happen to exist. With an explicit
    `api_key=` neither env var is consulted, so that line is not just noise — it
    tells the user the wrong key is in play. Filtered at the source rather than
    explained in the docs; every other genai log record still comes through.
    """
    import logging

    from google import genai

    logger = logging.getLogger("google_genai._api_client")
    if not any(getattr(f, "_dd_dual_key", False) for f in logger.filters):
        def _drop_dual_key_warning(record: logging.LogRecord) -> bool:
            return "Both GOOGLE_API_KEY and GEMINI_API_KEY" not in record.getMessage()

        _drop_dual_key_warning._dd_dual_key = True  # type: ignore[attr-defined]
        logger.addFilter(_drop_dual_key_warning)
    return genai.Client(api_key=api_key)


def read_caption(path: Path) -> str:
    """The stripped caption sidecar for `path` ("" if absent), read forgivingly.

    Accepts either an image path or the `.txt` itself. Caption sidecars are *user*
    files — hand-written, or produced by another tool — so a strict UTF-8 read is
    wrong twice over: a cp1252 file raised UnicodeDecodeError and killed the whole
    stage, and a UTF-8 BOM silently became part of the trigger word. `utf-8-sig`
    drops the BOM, and `errors="replace"` degrades one bad byte to one replacement
    character instead of losing the dataset. Never raises.
    """
    txt = path if path.suffix.lower() == ".txt" else path.with_suffix(".txt")
    try:
        return txt.read_text(encoding="utf-8-sig", errors="replace").strip()
    except OSError:
        return ""
