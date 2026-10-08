"""Generate LoRA-trainer config files for a finished dataset folder.

Four trainers are supported:

- **ostris ai-toolkit** — one self-contained `config.yaml`, launched with a
  single `python run.py config.yaml`. Genuinely one-command: the model is a HF
  id and every hyperparameter lives in the file.
- **kohya-ss musubi-tuner**, **kohya-ss sd-scripts** and **Fizgig** — a
  `dataset.toml` plus a command template. Their training invocations need the
  user's local DiT / VAE / text-encoder paths, which this tool cannot know, so
  the commands carry clearly-marked `<<FILL: ...>>` placeholders. They are
  deliberately NOT presented as one-click.

Every trainer also gets a validation pack in `validation/`: fixed prompts the
trainer samples after each epoch, plus a guide and a score sheet for picking
the best checkpoint.

Configs are hand-templated (not serialized) so inline comments, sample prompts,
and placeholders are preserved. No secrets are ever written — only the model id
/ paths the user supplies.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import yaml
from pydantic import BaseModel

# One seed for every validation sample, so epochs differ only by the LoRA.
VALIDATION_SEED = 42

# AdamW8bit is every trainer's own default and validated recipe. Prodigy finds
# its own step size (lr 1.0); these args are its README's advice for diffusion.
OPTIMIZERS = {"adamw8bit": "AdamW8bit (recommended)",
              "prodigy": "Prodigy (sets its own learning rate)"}
PRODIGY_ARGS = {"weight_decay": 0.01, "use_bias_correction": True, "safeguard_warmup": True}


def optimizer_choices(trainer: str) -> list[tuple[str, str]]:
    """Fizgig removed Prodigy: it fights Fizgig's adaptive LR (its docs/CLI.md)."""
    keys = ["adamw8bit"] if trainer == "fizgig" else list(OPTIMIZERS)
    return [(OPTIMIZERS[k], k) for k in keys]


def _yaml_str(value: str) -> str:
    """Render `value` as a safe single-line YAML scalar for interpolation.

    User-supplied strings reach the ai-toolkit config — a LoRA `name`, a model
    `name_or_path` (which may be a Windows checkpoint path like
    ``C:\\models\\x.safetensors``), and the sample prompt (which embeds the
    trigger/name). Dropping such text into a hand-written double-quoted YAML
    scalar breaks the file: a backslash is an escape char there and a stray ``"``
    ends the string early. Emitting through PyYAML picks correct quoting instead.
    Whitespace is collapsed first (a name/path is single-line by nature) so the
    result is always one physical line safe to splice into the template.
    """
    flat = " ".join(str(value).split())
    return yaml.safe_dump(flat, default_flow_style=True, allow_unicode=True).splitlines()[0].strip()

TRAINERS = {
    "ai-toolkit": "ostris ai-toolkit (one-command: python run.py config.yaml)",
    "musubi": "kohya-ss musubi-tuner (dataset.toml + command template)",
    "kohya": "kohya-ss sd-scripts (SDXL LoRA — dataset.toml + command)",
    "fizgig": "Fizgig (Krea 2 / Qwen Image 2.1 / Klein — dataset.toml + command)",
}


class ModelPreset(BaseModel):
    key: str
    label: str
    # ai-toolkit: HF id or local path -> model.name_or_path. May be a <<FILL>>
    # placeholder for models whose weights are user-local / not a plain HF id.
    name_or_path: str = ""
    # ai-toolkit newer configs select architecture via `arch:`; is_flux is the
    # legacy flag kept for FLUX where it is well-established.
    arch: str = ""
    is_flux: bool = False
    quantize: bool = True
    # ai-toolkit: extra `model:` keys (quantize_te, low_vram, qtype…) and the
    # train.timestep_type, copied from the arch's defaults in ai-toolkit's
    # extensions_built_in/diffusion_models/ui.tsx.
    model_extras: dict[str, bool | str] = {}
    timestep_type: str = ""
    # ai-toolkit train/sample knobs that vary by architecture family. Defaults
    # match the flow-matching models (Flux / Qwen-Image / Z-Image); SDXL, which
    # is not flow-matching, overrides them (ddpm scheduler, higher CFG).
    noise_scheduler: str = "flowmatch"
    sample_guidance: float = 4.0
    sample_steps: int = 20
    # musubi / sd-scripts read sample settings per prompt line (`--l 7 --s 25`);
    # appended after the size and seed. Empty = the trainer's per-arch defaults.
    sample_line_args: str = ""
    # Caption style this base model expects: tag-trained checkpoints (SDXL /
    # Illustrious / NoobAI / Pony) learn from comma tags, prose models (Flux /
    # Qwen-Image / Z-Image / Krea) from natural language. Drives the ④→⑤ advisory
    # that warns when the dataset's captions don't match — nothing else.
    expects_tags: bool = False
    # musubi training script (…_train_network.py). Its prefix also names the two
    # cache scripts. Everything below is from the arch's page in musubi's docs/.
    musubi_script: str = "<<FILL: see musubi docs for this arch>>"
    # Every musubi arch has its own LoRA module; plain networks.lora is wrong
    # for all of them.
    network_module: str = "networks.lora"
    # Extra train-command flags (musubi and Fizgig): timestep sampling, Fizgig's
    # recipe, sample-only model paths.
    musubi_args: str = ""
    musubi_version: str = ""  # --model_version, passed to all three commands
    # The train command always gets these: sampling each epoch needs them, even
    # where training itself reads cached text-encoder outputs (Krea 2).
    musubi_text_encoders: list[str] = ["text_encoder"]
    # kohya-ss sd-scripts training script (SDXL uses sdxl_train_network.py).
    kohya_script: str = "<<FILL: see kohya sd-scripts docs for this arch>>"
    # per-model defaults (UI pre-fills these; the user can override)
    resolution: int = 1024
    rank: int = 16
    alpha: int = 16
    # One checkpoint and one sample set per epoch. Repeats are derived from the
    # dataset size (⑤) so the total lands near the target step count.
    epochs: int = 16
    lr: float = 1e-4
    batch_size: int = 1


_SHIFT = "--timestep_sampling shift --weighting_scheme none --discrete_flow_shift {}"

# Curated, extensible — not exhaustive. Where a model's canonical HF id or
# musubi script is not something we can guarantee, it is a <<FILL>> placeholder
# so the emitted config is honest rather than silently wrong.
TRAINER_MODELS: dict[str, list[ModelPreset]] = {
    "ai-toolkit": [
        ModelPreset(key="flux-dev", label="FLUX.1-dev",
                    name_or_path="black-forest-labs/FLUX.1-dev", arch="flux",
                    is_flux=True),
        ModelPreset(key="flux2", label="FLUX.2",
                    name_or_path="black-forest-labs/FLUX.2-dev", arch="flux2"),
        ModelPreset(key="qwen-image", label="Qwen-Image",
                    name_or_path="Qwen/Qwen-Image", arch="qwen_image"),
        # SDXL is not flow-matching: it wants the ddpm scheduler and higher CFG,
        # and is small enough to train unquantized. Pairs with Danbooru/e621 tag
        # captions (③), which is what these checkpoints are trained on.
        ModelPreset(key="sdxl", label="SDXL 1.0 (base)",
                    name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
                    arch="sdxl", quantize=False, noise_scheduler="ddpm",
                    sample_guidance=7.0, sample_steps=25, expects_tags=True),
        ModelPreset(key="sdxl-custom",
                    label="SDXL-family checkpoint — Pony / Illustrious / NoobAI (set path)",
                    name_or_path="<<FILL: your SDXL-family checkpoint HF id or local path>>",
                    arch="sdxl", quantize=False, noise_scheduler="ddpm",
                    sample_guidance=7.0, sample_steps=25, expects_tags=True),
        ModelPreset(key="qwen-image-2.1", label="Qwen-Image 2.1",
                    name_or_path="Comfy-Org/Qwen-Image-2.1", arch="qwen_image_2",
                    timestep_type="shift", sample_guidance=3.0,
                    # The Comfy-Org weights are pre-quantized int8 convrot.
                    model_extras={"quantize_te": True, "low_vram": True,
                                  "qtype": "convrot8", "qtype_te": "convrot8"}),
        ModelPreset(key="zimage", label="Z-Image",
                    name_or_path="Tongyi-MAI/Z-Image", arch="zimage",
                    timestep_type="weighted", sample_steps=30,
                    model_extras={"quantize_te": True, "low_vram": True,
                                  "qtype": "qfloat8"}),
        # Key stays "krea" so saved ⑤ settings still find it. Train on Raw:
        # Turbo is distilled, and samples use Raw's own 28 steps / CFG 5.5.
        ModelPreset(key="krea", label="Krea 2 (Raw)",
                    name_or_path="krea/Krea-2-Raw", arch="krea2",
                    timestep_type="linear", sample_guidance=5.5, sample_steps=28,
                    model_extras={"quantize_te": True, "low_vram": True}),
        ModelPreset(key="custom", label="Custom (edit name_or_path below)",
                    name_or_path="<<FILL: your model name_or_path>>", arch="flux"),
    ],
    "musubi": [
        ModelPreset(key="qwen-image", label="Qwen-Image", arch="qwen_image",
                    musubi_script="qwen_image_train_network.py",
                    network_module="networks.lora_qwen_image", musubi_version="original",
                    musubi_args=_SHIFT.format(2.2)),
        ModelPreset(key="flux-kontext", label="FLUX.1 Kontext", arch="flux_kontext",
                    musubi_script="flux_kontext_train_network.py",
                    network_module="networks.lora_flux",
                    musubi_args="--timestep_sampling flux_shift --weighting_scheme none",
                    musubi_text_encoders=["text_encoder1", "text_encoder2"]),
        ModelPreset(key="flux2", label="FLUX.2", arch="flux2",
                    musubi_script="flux_2_train_network.py",
                    network_module="networks.lora_flux_2", musubi_version="dev",
                    musubi_args="--timestep_sampling flux2_shift --weighting_scheme none"),
        ModelPreset(key="zimage", label="Z-Image", arch="zimage",
                    musubi_script="zimage_train_network.py",
                    network_module="networks.lora_zimage", musubi_args=_SHIFT.format(2.0)),
        # docs/krea2.md: --dit is the Raw checkpoint. Samples render on Turbo
        # (how the LoRA is used): CFG off, 8 steps.
        ModelPreset(key="krea2", label="Krea 2 (Raw)", arch="krea2",
                    musubi_script="krea2_train_network.py",
                    network_module="networks.lora_krea2",
                    musubi_args=_SHIFT.format(2.5)
                    + " --turbo_dit <<FILL: Krea 2 Turbo DiT path (samples only)>>",
                    sample_line_args="--l 1 --s 8", sample_guidance=1.0, sample_steps=8,
                    rank=32, alpha=32),
    ],
    # kohya-ss sd-scripts is the standard SDXL LoRA trainer — the natural home for
    # the Danbooru/e621 tag captions (③). SDXL base is a runnable HF id; a family
    # checkpoint (Pony/Illustrious/NoobAI) is a user-local <<FILL>>.
    "kohya": [
        ModelPreset(key="sdxl", label="SDXL 1.0 (base)",
                    name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
                    arch="sdxl", kohya_script="sdxl_train_network.py", expects_tags=True,
                    sample_guidance=7.0, sample_steps=25, sample_line_args="--l 7 --s 25"),
        ModelPreset(key="sdxl-custom",
                    label="SDXL-family checkpoint — Pony / Illustrious / NoobAI (set path)",
                    name_or_path="<<FILL: your SDXL checkpoint (.safetensors path or HF id)>>",
                    arch="sdxl", kohya_script="sdxl_train_network.py", expects_tags=True,
                    sample_guidance=7.0, sample_steps=25, sample_line_args="--l 7 --s 25"),
    ],
    # Fizgig (shootthesound/Fizgig, docs/CLI.md): `arch` is its --family. Each
    # preset is the GUI's default recipe — adaptive LR (which ignores
    # --learning_rate), EMA, and the speed LoRA / adapter it renders and trains
    # with. Fizgig wants repeats 1 and more epochs.
    "fizgig": [
        ModelPreset(key="krea2", label="Krea 2 (Raw)", arch="krea2",
                    musubi_args="--adaptive_lr --adaptive_lr_min 2e-4 --adaptive_lr_max 4e-4 "
                                "--ema_decay 0.98 --speed_lora <<FILL: Krea 2 Turbo LoRA path>> "
                                "--sample_steps 8",
                    sample_guidance=1.0, sample_steps=8, rank=8, alpha=8, epochs=30),
        # 0.5 MP: Fizgig's Qwen presets train at [704, 704].
        ModelPreset(key="qwen_image21", label="Qwen Image 2.1", arch="qwen_image21",
                    musubi_args="--adaptive_lr --adaptive_lr_min 2e-4 --adaptive_lr_max 4e-4 "
                                "--ema_decay 0.98 --training_adapter <<FILL: Fizgig Qwen 2.1 "
                                "training adapter path>> --speed_lora <<FILL: Qwen 2.1 turbo "
                                "LoRA path>>",
                    # Previews render on the turbo LoRA at 6 steps (docs/CLI.md).
                    sample_guidance=1.0, sample_steps=6, resolution=704, rank=8, alpha=8,
                    epochs=30),
        ModelPreset(key="klein", label="FLUX.2 Klein 9B (Base)", arch="klein",
                    musubi_args="--adaptive_lr --adaptive_lr_min 5e-5 --adaptive_lr_max 4e-4 "
                                "--preview_checkpoint <<FILL: Klein distilled DiT path>>",
                    # Previews render on the distilled DiT at 4 steps (docs/CLI.md).
                    sample_guidance=1.0, sample_steps=4, epochs=55),
    ],
}


class TrainConfig(BaseModel):
    trainer: str
    model: ModelPreset
    dataset_dir: Path
    trigger: str = ""
    name: str = "lora"
    # "character" | "style" | "concept" — only tunes the sample prompt below.
    dataset_type: str = "character"
    resolution: int = 1024
    rank: int = 16
    alpha: int = 16
    epochs: int = 16
    num_repeats: int = 1
    # Images in the dataset; 0 = count them when writing (`write_configs`).
    n_images: int = 0
    lr: float = 1e-4
    optimizer: str = "adamw8bit"  # a key of OPTIMIZERS; anything else renders as AdamW8bit
    batch_size: int = 1
    # Multi-resolution buckets. Empty = single-bucket at `resolution` (the old
    # behaviour); populated from the dataset's real dimensions by the caller.
    buckets: list[int] = []
    # ② shot style, so the sample prompt names the right medium. Defaults keep
    # every pre-0.15.1 config byte-identical.
    shot_style: str = "match"
    shot_style_text: str = ""


def validation_prompts(cfg: TrainConfig) -> list[str]:
    """Eight fixed prompts the trainer samples every epoch, same seed each time.

    They probe what a LoRA gets wrong: a tight face (likeness), full body and a
    back view (physique, the views a few references rarely show), an outfit and
    a setting the dataset never had (does the trigger carry the subject or the
    training images?), another medium (flexibility), and action. House rules
    as ②: no negation, never "photorealistic".

    The medium follows the ② shot style: "a photo of <trigger>" is wrong for a
    dataset of anime shots. `match` keeps "a photo of" — with an unknown source
    medium there is nothing better to say.
    """
    from studio.shot_style import resolve

    who = cfg.trigger or cfg.name or "the subject"
    if cfg.dataset_type == "style":
        # The trigger is an aesthetic; the prompts name content it renders.
        return [f"{who}, {content}" for content in (
            "a mountain landscape at sunset", "a portrait of an old fisherman",
            "a bowl of fruit on a wooden table", "a busy city street at night",
            "a cat asleep on a windowsill", "a castle on a hill under storm clouds",
            "a woman reading in a cafe", "a spaceship over a desert")]
    lead = resolve(cfg.shot_style, cfg.shot_style_text).sample_lead
    subject = f"{lead} {who}" if lead.endswith(" of") else f"{lead}{who}"
    if cfg.dataset_type == "concept":
        return [subject, f"{subject}, close-up", f"{subject} on a wooden table",
                f"{subject} outdoors in daylight", f"{subject} at night under neon light",
                f"{subject} in a snowy forest", f"a watercolor painting of {who}",
                f"{subject}, seen from above"]
    return [f"{subject}, standing outdoors in daylight",
            f"{subject}, a tight close-up of the face, soft window light",
            f"{subject}, full body, standing facing the camera, plain studio backdrop",
            f"{subject}, seen from behind over the shoulder, walking down a city "
            f"street at dusk",
            f"{subject}, sitting at a cafe table wearing a yellow raincoat",
            f"{subject}, standing on the deck of a sailing ship in a storm",
            f"a charcoal sketch of {who}, waist-up portrait",
            f"{subject}, jumping mid-air on a beach, arms raised"]


def exposure(n_images: int, num_repeats: int, batch_size: int, epochs: int) -> tuple[int, int]:
    """(steps per epoch, total steps). An empty folder counts as one image."""
    per_epoch = max(1, math.ceil(max(n_images, 1) * max(num_repeats, 1) / max(batch_size, 1)))
    return per_epoch, per_epoch * max(epochs, 1)


def _exposure(cfg: TrainConfig) -> tuple[int, int]:
    return exposure(cfg.n_images, cfg.num_repeats, cfg.batch_size, cfg.epochs)


def caption_mismatch_warning(preset: ModelPreset, caption_kind: str) -> str:
    """Advisory line when a dataset's caption style doesn't fit the base model.

    `caption_kind` is 'tags', 'prose', or '' (unknown/uncaptioned → no warning).
    Never blocks — it only surfaces the likely mismatch so the user can re-caption
    (③) in the right style before a long training run.
    """
    if caption_kind == "prose" and preset.expects_tags:
        return ("⚠️ Caption/model mismatch: this base model is tag-trained "
                "(Danbooru/e621), but the dataset's captions look like PROSE. "
                "Tag-trained checkpoints learn poorly from sentences — consider "
                "re-captioning in ③ with a tag style or a tagger.")
    if caption_kind == "tags" and not preset.expects_tags:
        return ("⚠️ Caption/model mismatch: this base model expects natural-language "
                "captions, but the dataset's captions look like comma-separated TAGS. "
                "Consider re-captioning in ③ with the prose style.")
    return ""


def _resolution_list(cfg: TrainConfig) -> str:
    """ai-toolkit buckets by listing resolutions; a single entry means one
    bucket and wastes non-square images."""
    values = cfg.buckets or [cfg.resolution]
    return "[" + ", ".join(str(v) for v in values) + "]"


def render_aitoolkit_yaml(cfg: TrainConfig) -> str:
    """A complete, runnable ai-toolkit config.yaml."""
    m = cfg.model
    arch_line = "        is_flux: true\n" if m.is_flux else f'        arch: "{m.arch}"\n'
    model_extras = "".join(
        f"        {k}: {str(v).lower() if isinstance(v, bool) else _yaml_str(v)}\n"
        for k, v in m.model_extras.items())
    timestep = f"        timestep_type: {m.timestep_type}\n" if m.timestep_type else ""
    raw_note = ("        # Train on Raw. Turbo is distilled: a LoRA trained on it fights the\n"
                "        # distillation. Use the LoRA with Turbo at inference instead.\n"
                if m.arch.startswith("krea2") else "")
    # ai-toolkit counts steps, so an "epoch" is the steps one pass takes.
    per_epoch, total = _exposure(cfg)
    if cfg.optimizer == "prodigy":
        optimizer = ("        optimizer: prodigy\n        optimizer_params:\n"
                     + "".join(f"          {k}: {str(v).lower()}\n"
                               for k, v in PRODIGY_ARGS.items())
                     + "        lr: 1.0\n")
    else:
        optimizer = f"        optimizer: adamw8bit\n        lr: {cfg.lr}\n"
    prompts ="".join(f"          - {_yaml_str(p)}\n" for p in validation_prompts(cfg))
    return (
        "# ai-toolkit LoRA config generated by Dataset Deviser.\n"
        "# Run from your ai-toolkit install:  python run.py <this file>\n"
        "# Verify model-specific keys against ai-toolkit's config/examples if needed.\n"
        "---\n"
        "job: extension\n"
        "config:\n"
        f"  name: {_yaml_str(cfg.name)}\n"
        "  process:\n"
        "    - type: sd_trainer\n"
        f"      training_folder: \"output\"\n"
        "      device: cuda:0\n"
        "      network:\n"
        "        type: lora\n"
        f"        linear: {cfg.rank}\n"
        f"        linear_alpha: {cfg.alpha}\n"
        "      save:\n"
        "        dtype: float16\n"
        f"        save_every: {per_epoch}\n"
        # Keep every save: the best checkpoint is often an early one, and
        # pruning to the last few deleted it before anyone could compare.
        f"        max_step_saves_to_keep: {cfg.epochs}\n"
        "      datasets:\n"
        f"        - folder_path: \"{cfg.dataset_dir.as_posix()}\"\n"
        "          caption_ext: \"txt\"\n"
        "          caption_dropout_rate: 0.05\n"
        "          shuffle_tokens: false\n"
        "          cache_latents_to_disk: true\n"
        f"          resolution: {_resolution_list(cfg)}\n"
        "      train:\n"
        f"        batch_size: {cfg.batch_size}\n"
        f"        steps: {total}  # {cfg.epochs} epochs x {per_epoch} steps\n"
        "        gradient_accumulation_steps: 1\n"
        "        train_unet: true\n"
        "        train_text_encoder: false\n"
        "        gradient_checkpointing: true\n"
        f"        noise_scheduler: {m.noise_scheduler}\n"
        f"{timestep}"
        f"{optimizer}"
        "        dtype: bf16\n"
        "      model:\n"
        f"{raw_note}"
        f"        name_or_path: {_yaml_str(m.name_or_path)}\n"
        f"{arch_line}"
        f"        quantize: {str(m.quantize).lower()}\n"
        f"{model_extras}"
        "      sample:\n"
        f"        sample_every: {per_epoch}\n"
        f"        width: {cfg.resolution}\n"
        f"        height: {cfg.resolution}\n"
        f"        seed: {VALIDATION_SEED}\n"
        "        walk_seed: false\n"
        "        prompts:\n"
        f"{prompts}"
        f"        guidance_scale: {m.sample_guidance:g}\n"
        f"        sample_steps: {m.sample_steps}\n"
        "meta:\n"
        "  name: \"[name]\"\n"
        "  version: \"1.0\"\n"
    )


def render_musubi_toml(cfg: TrainConfig) -> str:
    """The standard musubi-tuner image dataset.toml (Fizgig reads the same shape)."""
    cache = (cfg.dataset_dir / "cache").as_posix()
    return (
        f"# {cfg.trainer} dataset config generated by Dataset Deviser.\n"
        "# Pass to training with:  --dataset_config <this file>\n"
        "# (the training command also needs your DiT/VAE/text-encoder paths.)\n"
        "\n"
        "[general]\n"
        f"resolution = [{cfg.resolution}, {cfg.resolution}]\n"
        'caption_extension = ".txt"\n'
        f"batch_size = {cfg.batch_size}\n"
        "enable_bucket = true\n"
        # Never invent detail by upscaling a small source into a big bucket.
        "bucket_no_upscale = true\n"
        "\n"
        "[[datasets]]\n"
        f'image_directory = "{cfg.dataset_dir.as_posix()}"\n'
        f'cache_directory = "{cache}"\n'
        f"num_repeats = {cfg.num_repeats}\n"
    )


def aitoolkit_command(install_path: str, config_path: Path) -> str:
    base = install_path.strip() or "<<FILL: path to your ai-toolkit install>>"
    return f'cd "{base}" && python run.py "{config_path.as_posix()}"'


def _optimizer_cli(cfg: TrainConfig, prodigy_type: str, adamw_type: str) -> str:
    """Optimizer + learning-rate flags for musubi and sd-scripts."""
    if cfg.optimizer == "prodigy":
        args = " ".join(f'"{k}={v}"' for k, v in PRODIGY_ARGS.items())
        return f"--optimizer_type {prodigy_type} --learning_rate 1.0 --optimizer_args {args}"
    return f"--optimizer_type {adamw_type} --learning_rate {cfg.lr}"


def _sampling(prompts_path: Path) -> str:
    """Per-epoch sampling flags, shared by musubi, sd-scripts and Fizgig."""
    return (f'  --sample_prompts "{prompts_path.as_posix()}" '
            "--sample_every_n_epochs 1 --sample_at_first \\\n")


def musubi_command(install_path: str, toml_path: Path, cfg: TrainConfig,
                   prompts_path: Path) -> str:
    """Build the musubi run command from `cfg`.

    Takes the whole TrainConfig, not just the preset: rank/alpha/epochs/lr are
    user-tunable in the UI and must actually reach the command line.
    """
    m = cfg.model
    base = install_path.strip() or "<<FILL: path to your musubi-tuner install>>"
    prefix = m.musubi_script.removesuffix("_train_network.py")
    toml = toml_path.as_posix()
    version = f" --model_version {m.musubi_version}" if m.musubi_version else ""
    te = " ".join(f"--{f} <<FILL: {f.replace('_', ' ')} path>>"
                  for f in m.musubi_text_encoders)
    extra = f"  {m.musubi_args} \\\n" if m.musubi_args else ""
    return (
        "# 1-2: cache latents and text-encoder outputs (re-run after changing the\n"
        "# images or captions). 3: train, saving a LoRA and samples every epoch.\n"
        f'cd "{base}"\n'
        f'python src/musubi_tuner/{prefix}_cache_latents.py --dataset_config "{toml}" '
        f"--vae <<FILL: VAE path>>{version}\n"
        f'python src/musubi_tuner/{prefix}_cache_text_encoder_outputs.py '
        f'--dataset_config "{toml}" {te} --batch_size 1{version}\n'
        "accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 "
        f"src/musubi_tuner/{m.musubi_script} \\\n"
        "  --dit <<FILL: DiT/model weights path>> \\\n"
        "  --vae <<FILL: VAE path>> \\\n"
        f"  {te} \\\n"
        f'  --dataset_config "{toml}"{version} \\\n'
        "  --sdpa --mixed_precision bf16 --gradient_checkpointing \\\n"
        f"{extra}"
        # musubi takes any optimizer as module.Class; Prodigy needs prodigyopt installed.
        f"  {_optimizer_cli(cfg, 'prodigyopt.Prodigy', 'adamw8bit')} \\\n"
        "  --max_data_loader_n_workers 2 --persistent_data_loader_workers \\\n"
        f"  --network_module {m.network_module} \\\n"
        f"  --network_dim {cfg.rank} \\\n"
        f"  --network_alpha {cfg.alpha} \\\n"
        f"  --max_train_epochs {cfg.epochs} --save_every_n_epochs 1 "
        f"--seed {VALIDATION_SEED} \\\n"
        f"{_sampling(prompts_path)}"
        f'  --output_dir output --output_name "{cfg.name}"\n'
        "# The text encoder in step 3 only renders the samples. "
        + ("--turbo_dit can't be combined\n# with --blocks_to_swap: drop it to sample "
           "on Raw (edit the prompt lines to --l 5.5 --s 28).\n"
           if "--turbo_dit" in m.musubi_args else "\n")
        + f"# Flags follow musubi-tuner's docs/{prefix}.md example; check it if a "
        "run complains."
    )


def render_kohya_toml(cfg: TrainConfig) -> str:
    """A kohya-ss sd-scripts image `dataset.toml` (subsets layout)."""
    return (
        "# kohya-ss sd-scripts dataset config generated by Dataset Deviser.\n"
        "# Pass to training with:  --dataset_config <this file>\n"
        "\n"
        "[general]\n"
        'caption_extension = ".txt"\n'
        "shuffle_caption = false\n"
        "\n"
        "[[datasets]]\n"
        f"resolution = [{cfg.resolution}, {cfg.resolution}]\n"
        f"batch_size = {cfg.batch_size}\n"
        "enable_bucket = true\n"
        # Never invent detail by upscaling a small source into a big bucket.
        "bucket_no_upscale = true\n"
        "\n"
        "  [[datasets.subsets]]\n"
        f'  image_dir = "{cfg.dataset_dir.as_posix()}"\n'
        f"  num_repeats = {cfg.num_repeats}\n"
    )


def kohya_command(install_path: str, toml_path: Path, cfg: TrainConfig,
                  prompts_path: Path) -> str:
    """Build the kohya sd-scripts run command from `cfg`.

    The pretrained model is the preset's `name_or_path` (SDXL base is a runnable
    HF id; a family checkpoint stays a `<<FILL>>` the user supplies). Every
    hyperparameter comes from the TrainConfig, so the ⑤-tab sliders reach the CLI.
    """
    base = install_path.strip() or "<<FILL: path to your kohya sd-scripts install>>"
    model = cfg.model.name_or_path or "<<FILL: your base checkpoint path or HF id>>"
    return (
        f'cd "{base}" && accelerate launch {cfg.model.kohya_script} \\\n'
        f'  --pretrained_model_name_or_path "{model}" \\\n'
        f'  --dataset_config "{toml_path.as_posix()}" \\\n'
        f'  --output_dir output --output_name "{cfg.name}" \\\n'
        "  --network_module networks.lora \\\n"
        f"  --network_dim {cfg.rank} --network_alpha {cfg.alpha} \\\n"
        f"  {_optimizer_cli(cfg, 'Prodigy', 'AdamW8bit')} \\\n"
        f"  --max_train_epochs {cfg.epochs} --mixed_precision bf16 --sdpa \\\n"
        f"{_sampling(prompts_path)}"
        "  --gradient_checkpointing --save_model_as safetensors --save_every_n_epochs 1\n"
        "# SDXL uses sdxl_train_network.py; verify flags against the kohya-ss/sd-scripts docs."
    )


def fizgig_command(install_path: str, toml_path: Path, cfg: TrainConfig,
                   prompts_path: Path) -> str:
    """Fizgig's three steps, per its docs/CLI.md: cache latents, cache text, train.

    `arch` is the --family. `--precision auto --blocks_to_swap -1` is the GUI's
    behaviour (the CLI defaults to bf16). Samples render at 1024: Fizgig warns
    that smaller previews undersell the checkpoint.
    """
    m = cfg.model
    base = install_path.strip() or "<<FILL: path to your Fizgig install>>"
    toml = toml_path.as_posix()
    fam = f"--family {m.arch}"
    cache = f'python src/fizgig/families/cache.py {fam} --dataset_config "{toml}"'
    return (
        "# 1-2: cache latents and text-encoder outputs (re-run after changing the\n"
        "# images or captions). 3: train, saving a LoRA and samples every epoch.\n"
        "# On Windows use venv\\Scripts\\python.exe and put each command on one line.\n"
        f'cd "{base}"\n'
        f"{cache} --stage latents --model <<FILL: VAE path>>\n"
        f"{cache} --stage text --model <<FILL: text encoder path>>\n"
        f"python src/fizgig/families/train.py {fam} \\\n"
        f'  --dataset_config "{toml}" \\\n'
        "  --dit <<FILL: DiT path (the base, never Turbo/distilled)>> \\\n"
        "  --vae <<FILL: VAE path>> --text_encoder <<FILL: text encoder path>> \\\n"
        f'  --output_dir output --output_name "{cfg.name}" \\\n'
        f"  --network_dim {cfg.rank} --network_alpha {cfg.alpha} "
        f"--learning_rate {cfg.lr} \\\n"
        f"  --max_train_epochs {cfg.epochs} --save_every_n_epochs 1 "
        f"--seed {VALIDATION_SEED} \\\n"
        "  --optimizer_type adamw8bit --precision auto --blocks_to_swap -1 \\\n"
        f"  {m.musubi_args} \\\n"
        f"{_sampling(prompts_path)}"
        f"  --sample_width 1024 --sample_height 1024 --sample_seed {VALIDATION_SEED}\n"
        "# --adaptive_lr sets the learning rate itself (--learning_rate is ignored).\n"
        "# Flags follow Fizgig's docs/CLI.md; run train.py --help if one is refused."
    )


def _nonclobber(path: Path) -> Path:
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}.{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _write(path: Path, text: str) -> Path:
    path = _nonclobber(path)
    path.write_text(text, encoding="utf-8")
    return path


def _prompt_file(cfg: TrainConfig, prompts: list[str]) -> str:
    """The trainer's --sample_prompts file. musubi and sd-scripts take size, seed
    and sampler settings per line; Fizgig and ai-toolkit take plain prompts."""
    suffix = ""
    if cfg.trainer in ("musubi", "kohya"):
        suffix = (f" --w {cfg.resolution} --h {cfg.resolution} --d {VALIDATION_SEED} "
                  f"{cfg.model.sample_line_args}").rstrip()
    return ("# Validation prompts generated by Dataset Deviser, sampled every epoch "
            "at one seed.\n# One prompt per line; lines starting with # are comments.\n"
            + "".join(f"{p}{suffix}\n" for p in prompts))


def _heldout(dataset_dir: Path) -> dict:
    """The held-out photos ④ recorded in metadata.json, or {}."""
    try:
        meta = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    held = meta.get("heldout") if isinstance(meta, dict) else None
    return held if isinstance(held, dict) else {}


def _guide(cfg: TrainConfig, prompts: list[str]) -> str:
    m = cfg.model
    if m.arch.startswith("krea2"):
        settings = ("Use the LoRA on **Krea 2 Turbo**: 8 steps, CFG 1. On Raw: 28 steps, "
                    "CFG 5.5.")
    else:
        settings = (f"{m.label}: {m.sample_steps} steps, CFG {m.sample_guidance:g} — the "
                    f"settings the samples use.")
    held = _heldout(cfg.dataset_dir)
    held_block = ""
    if held.get("files"):
        held_block = (
            "\n## Held-out photos\n\n"
            f"These real photos were kept out of training: `{held.get('dir', '')}`\n"
            + "".join(f"- {f}\n" for f in held["files"])
            + "\nThe LoRA never saw them, so they are the fair likeness test: compare "
              "each epoch's close-up and full-body samples against them.\n")
    numbered = "".join(f"{i}. {p}\n" for i, p in enumerate(prompts, 1))
    return (
        f"# Validating {cfg.name}\n\n"
        f"{TRAINERS[cfg.trainer].split(' (')[0]} · {m.label} · {cfg.epochs} epochs. "
        f"Every epoch saves a checkpoint and renders the prompts below at seed "
        f"{VALIDATION_SEED}, so between epochs only the LoRA changes.\n\n"
        "## Pick the checkpoint\n\n"
        "1. Open the sample folder inside the trainer's output folder.\n"
        "2. Score epochs in `validation_scores.csv`, 1–5: **likeness** (the face), "
        "**physique** (build, height, proportions), **adherence** (did it wear, do and "
        "go where the prompt said), **unwanted** (something from the training images "
        "that appears unasked: a prop, a pose, a background, a colour cast).\n"
        "3. Keep the earliest epoch where likeness stops improving. When adherence "
        "drops or unwanted traits climb, later epochs are overfitting.\n"
        f"{held_block}\n"
        f"## Inference settings\n\n{settings}\n\n"
        f"## Prompts (seed {VALIDATION_SEED})\n\n{numbered}")


def _score_sheet(cfg: TrainConfig, n_prompts: int) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["epoch", "prompt", "likeness", "physique", "adherence", "unwanted",
                     "notes"])
    for epoch in range(1, cfg.epochs + 1):
        for prompt in range(1, n_prompts + 1):
            writer.writerow([epoch, prompt, "", "", "", "", ""])
    return out.getvalue()


def write_configs(cfg: TrainConfig, install_path: str = "") -> tuple[list[Path], str]:
    """Write the trainer's config file(s) and the validation pack.

    Config files go into the dataset folder; the validation pack into its
    `validation/` subfolder, where no trainer reads a `.txt` as a caption.
    Returns (written_paths, run_command), the config first. Never clobbers
    existing files — collisions get a `.N` suffix. `install_path` is only used
    to compose the displayed run command; it is never written into any file.
    """
    if cfg.trainer not in TRAINERS:
        raise ValueError(f"Unknown trainer: {cfg.trainer}")
    ds = cfg.dataset_dir
    ds.mkdir(parents=True, exist_ok=True)
    if not cfg.n_images:
        from studio.config import list_images

        cfg = cfg.model_copy(update={"n_images": len(list_images(ds))})
    prompts = validation_prompts(cfg)
    val = ds / "validation"
    val.mkdir(exist_ok=True)
    prompts_path = _write(val / "validation_prompts.txt", _prompt_file(cfg, prompts))
    if cfg.trainer == "ai-toolkit":
        path = _write(ds / "ai-toolkit.yaml", render_aitoolkit_yaml(cfg))
        command = aitoolkit_command(install_path, path)
    elif cfg.trainer == "kohya":
        path = _write(ds / "kohya-dataset.toml", render_kohya_toml(cfg))
        command = kohya_command(install_path, path, cfg, prompts_path)
    elif cfg.trainer == "musubi":
        path = _write(ds / "dataset.toml", render_musubi_toml(cfg))
        command = musubi_command(install_path, path, cfg, prompts_path)
    else:  # fizgig reads musubi's dataset.toml shape
        path = _write(ds / "fizgig-dataset.toml", render_musubi_toml(cfg))
        command = fizgig_command(install_path, path, cfg, prompts_path)
    return [path, prompts_path, _write(val / "validation.md", _guide(cfg, prompts)),
            _write(val / "validation_scores.csv", _score_sheet(cfg, len(prompts)))], command
