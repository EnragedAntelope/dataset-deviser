"""Generate LoRA-trainer config files for a finished dataset folder.

Two trainers are supported:

- **ostris ai-toolkit** — one self-contained `config.yaml`, launched with a
  single `python run.py config.yaml`. Genuinely one-command: the model is a HF
  id and every hyperparameter lives in the file.
- **kohya-ss musubi-tuner** — the standard image `dataset.toml`. musubi's
  *training* invocation additionally needs the user's local DiT / VAE / text-
  encoder paths, which this tool cannot know, so `musubi_command()` returns a
  template with clearly-marked `<<FILL: ...>>` placeholders. It is deliberately
  NOT presented as one-click.

Configs are hand-templated (not serialized) so inline comments, sample prompts,
and placeholders are preserved. No secrets are ever written — only the model id
/ paths the user supplies.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


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
    # ai-toolkit train/sample knobs that vary by architecture family. Defaults
    # match the flow-matching models (Flux / Qwen-Image / Z-Image); SDXL, which
    # is not flow-matching, overrides them (ddpm scheduler, higher CFG).
    noise_scheduler: str = "flowmatch"
    sample_guidance: float = 4.0
    sample_steps: int = 20
    # Caption style this base model expects: tag-trained checkpoints (SDXL /
    # Illustrious / NoobAI / Pony) learn from comma tags, prose models (Flux /
    # Qwen-Image / Z-Image / Krea) from natural language. Drives the ④→⑤ advisory
    # that warns when the dataset's captions don't match — nothing else.
    expects_tags: bool = False
    # musubi training script (…_train_network.py). Placeholder when unverified.
    musubi_script: str = "<<FILL: see musubi docs for this arch>>"
    # kohya-ss sd-scripts training script (SDXL uses sdxl_train_network.py).
    kohya_script: str = "<<FILL: see kohya sd-scripts docs for this arch>>"
    # per-model defaults (UI pre-fills these; the user can override)
    resolution: int = 1024
    rank: int = 16
    alpha: int = 16
    steps: int = 2000
    lr: float = 1e-4
    batch_size: int = 1


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
        ModelPreset(key="zimage", label="Z-Image",
                    name_or_path="<<FILL: Z-Image model path or HF id>>",
                    arch="zimage"),
        ModelPreset(key="krea", label="Krea 2",
                    name_or_path="<<FILL: Krea 2 diffusers model path>>",
                    arch="krea"),
        ModelPreset(key="custom", label="Custom (edit name_or_path below)",
                    name_or_path="<<FILL: your model name_or_path>>", arch="flux"),
    ],
    "musubi": [
        ModelPreset(key="qwen-image", label="Qwen-Image", arch="qwen_image",
                    musubi_script="qwen_image_train_network.py"),
        ModelPreset(key="flux-kontext", label="FLUX.1 Kontext", arch="flux_kontext",
                    musubi_script="<<FILL: flux train script, see musubi docs>>"),
        ModelPreset(key="flux2", label="FLUX.2", arch="flux2",
                    musubi_script="<<FILL: flux2 train script, see musubi docs>>"),
        ModelPreset(key="zimage", label="Z-Image", arch="zimage",
                    musubi_script="<<FILL: z-image train script, see musubi docs>>"),
    ],
    # kohya-ss sd-scripts is the standard SDXL LoRA trainer — the natural home for
    # the Danbooru/e621 tag captions (③). SDXL base is a runnable HF id; a family
    # checkpoint (Pony/Illustrious/NoobAI) is a user-local <<FILL>>.
    "kohya": [
        ModelPreset(key="sdxl", label="SDXL 1.0 (base)",
                    name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
                    arch="sdxl", kohya_script="sdxl_train_network.py", expects_tags=True),
        ModelPreset(key="sdxl-custom",
                    label="SDXL-family checkpoint — Pony / Illustrious / NoobAI (set path)",
                    name_or_path="<<FILL: your SDXL checkpoint (.safetensors path or HF id)>>",
                    arch="sdxl", kohya_script="sdxl_train_network.py", expects_tags=True),
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
    steps: int = 2000
    lr: float = 1e-4
    batch_size: int = 1
    # Multi-resolution buckets. Empty = single-bucket at `resolution` (the old
    # behaviour); populated from the dataset's real dimensions by the caller.
    buckets: list[int] = []


def _sample_prompt(cfg: TrainConfig) -> str:
    who = cfg.trigger or cfg.name or "the subject"
    if cfg.dataset_type == "style":
        # The trigger is an aesthetic; the prompt names content it renders.
        return f"{who}, a mountain landscape at sunset"
    if cfg.dataset_type == "concept":
        return f"a photo of {who}"
    return f"a photo of {who}, standing outdoors in daylight"


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
        "        save_every: 250\n"
        "        max_step_saves_to_keep: 4\n"
        "      datasets:\n"
        f"        - folder_path: \"{cfg.dataset_dir.as_posix()}\"\n"
        "          caption_ext: \"txt\"\n"
        "          caption_dropout_rate: 0.05\n"
        "          shuffle_tokens: false\n"
        "          cache_latents_to_disk: true\n"
        f"          resolution: {_resolution_list(cfg)}\n"
        "      train:\n"
        f"        batch_size: {cfg.batch_size}\n"
        f"        steps: {cfg.steps}\n"
        "        gradient_accumulation_steps: 1\n"
        "        train_unet: true\n"
        "        train_text_encoder: false\n"
        "        gradient_checkpointing: true\n"
        f"        noise_scheduler: {m.noise_scheduler}\n"
        f"        optimizer: adamw8bit\n"
        f"        lr: {cfg.lr}\n"
        "        dtype: bf16\n"
        "      model:\n"
        f"        name_or_path: {_yaml_str(m.name_or_path)}\n"
        f"{arch_line}"
        f"        quantize: {str(m.quantize).lower()}\n"
        "      sample:\n"
        "        sample_every: 250\n"
        f"        width: {cfg.resolution}\n"
        f"        height: {cfg.resolution}\n"
        "        prompts:\n"
        f"          - {_yaml_str(_sample_prompt(cfg))}\n"
        f"        guidance_scale: {m.sample_guidance:g}\n"
        f"        sample_steps: {m.sample_steps}\n"
        "meta:\n"
        "  name: \"[name]\"\n"
        "  version: \"1.0\"\n"
    )


def render_musubi_toml(cfg: TrainConfig, num_repeats: int = 1) -> str:
    """The standard musubi-tuner image dataset.toml."""
    cache = (cfg.dataset_dir / "cache").as_posix()
    return (
        "# musubi-tuner dataset config generated by Dataset Deviser.\n"
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
        f"num_repeats = {num_repeats}\n"
    )


def aitoolkit_command(install_path: str, config_path: Path) -> str:
    base = install_path.strip() or "<<FILL: path to your ai-toolkit install>>"
    return f'cd "{base}" && python run.py "{config_path.as_posix()}"'


def musubi_command(install_path: str, toml_path: Path, cfg: TrainConfig) -> str:
    """Build the musubi run command from `cfg`.

    Takes the whole TrainConfig, not just the preset: rank/alpha/steps/lr are
    user-tunable in the UI and must actually reach the command line.
    """
    base = install_path.strip() or "<<FILL: path to your musubi-tuner install>>"
    return (
        f'cd "{base}" && accelerate launch src/musubi_tuner/{cfg.model.musubi_script} \\\n'
        f'  --dataset_config "{toml_path.as_posix()}" \\\n'
        "  --dit <<FILL: DiT/model weights path>> \\\n"
        "  --vae <<FILL: VAE path>> \\\n"
        "  --text_encoder <<FILL: text encoder path>> \\\n"
        "  --network_module networks.lora \\\n"
        f"  --network_dim {cfg.rank} \\\n"
        f"  --network_alpha {cfg.alpha} \\\n"
        f"  --learning_rate {cfg.lr} \\\n"
        f'  --output_dir output --output_name "{cfg.name}" \\\n'
        f"  --max_train_steps {cfg.steps} --mixed_precision bf16\n"
        "# Consult the musubi-tuner arch-specific doc for the exact required flags."
    )


def render_kohya_toml(cfg: TrainConfig, num_repeats: int = 1) -> str:
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
        f"  num_repeats = {num_repeats}\n"
    )


def kohya_command(install_path: str, toml_path: Path, cfg: TrainConfig) -> str:
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
        f"  --learning_rate {cfg.lr} --max_train_steps {cfg.steps} \\\n"
        "  --optimizer_type AdamW8bit --mixed_precision bf16 --sdpa \\\n"
        "  --gradient_checkpointing --save_model_as safetensors --save_every_n_steps 250\n"
        "# SDXL uses sdxl_train_network.py; verify flags against the kohya-ss/sd-scripts docs."
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


def write_configs(cfg: TrainConfig, install_path: str = "",
                  num_repeats: int = 1) -> tuple[list[Path], str]:
    """Write the trainer's config file(s) into the dataset folder.

    Returns (written_paths, run_command). Never clobbers existing files —
    collisions get a `.N` suffix. `install_path` is only used to compose the
    displayed run command; it is never written into any file.
    """
    cfg.dataset_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if cfg.trainer == "ai-toolkit":
        path = _nonclobber(cfg.dataset_dir / "ai-toolkit.yaml")
        path.write_text(render_aitoolkit_yaml(cfg), encoding="utf-8")
        written.append(path)
        command = aitoolkit_command(install_path, path)
    elif cfg.trainer == "musubi":
        path = _nonclobber(cfg.dataset_dir / "dataset.toml")
        path.write_text(render_musubi_toml(cfg, num_repeats=num_repeats),
                        encoding="utf-8")
        written.append(path)
        command = musubi_command(install_path, path, cfg)
    elif cfg.trainer == "kohya":
        path = _nonclobber(cfg.dataset_dir / "kohya-dataset.toml")
        path.write_text(render_kohya_toml(cfg, num_repeats=num_repeats),
                        encoding="utf-8")
        written.append(path)
        command = kohya_command(install_path, path, cfg)
    else:
        raise ValueError(f"Unknown trainer: {cfg.trainer}")
    return written, command
