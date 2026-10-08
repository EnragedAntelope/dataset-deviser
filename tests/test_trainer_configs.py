"""Tests for trainer config generation."""

from __future__ import annotations

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

import pytest
import yaml

from studio.trainer_configs import (
    TRAINER_MODELS,
    ModelPreset,
    TrainConfig,
    render_aitoolkit_yaml,
    render_musubi_toml,
    write_configs,
)


def _cfg(trainer: str, tmp_path: Path) -> TrainConfig:
    return TrainConfig(trainer=trainer, model=TRAINER_MODELS[trainer][0],
                       dataset_dir=tmp_path, trigger="sysnootles", name="sy-lora",
                       resolution=1024, rank=16, alpha=16, steps=1500, lr=1e-4)


def test_aitoolkit_yaml_parses_and_has_keys(tmp_path: Path) -> None:
    doc = yaml.safe_load(render_aitoolkit_yaml(_cfg("ai-toolkit", tmp_path)))
    proc = doc["config"]["process"][0]
    assert proc["type"] == "sd_trainer"
    assert proc["network"]["linear"] == 16
    assert proc["train"]["steps"] == 1500
    assert proc["datasets"][0]["folder_path"] == tmp_path.as_posix()
    assert proc["model"]["name_or_path"]  # non-empty


def test_musubi_toml_parses_and_has_dataset(tmp_path: Path) -> None:
    doc = tomllib.loads(render_musubi_toml(_cfg("musubi", tmp_path)))
    assert doc["general"]["caption_extension"] == ".txt"
    assert doc["datasets"][0]["image_directory"] == tmp_path.as_posix()


def test_write_configs_writes_file_and_command(tmp_path: Path) -> None:
    written, command = write_configs(_cfg("ai-toolkit", tmp_path), install_path=r"C:\ai-toolkit")
    assert written[0].exists()
    assert written[0].name == "ai-toolkit.yaml"
    assert "ai-toolkit" in command and "run.py" in command


def test_write_configs_never_clobbers(tmp_path: Path) -> None:
    write_configs(_cfg("musubi", tmp_path))
    written2, _ = write_configs(_cfg("musubi", tmp_path))
    assert written2[0].name == "dataset.2.toml"


def test_musubi_command_flags_model_paths(tmp_path: Path) -> None:
    _, command = write_configs(_cfg("musubi", tmp_path), install_path="/opt/musubi")
    assert "<<FILL:" in command  # honest about needing model paths
    assert "/opt/musubi" in command


def test_aitoolkit_yaml_escapes_windows_path_and_quotes(tmp_path: Path) -> None:
    """A user-supplied Windows checkpoint path (backslashes) or a name/prompt with
    a double-quote must not break the emitted YAML — regression for the naive
    double-quoted interpolation that produced an unparseable config."""
    m = ModelPreset(key="sdxl-custom", label="x",
                    name_or_path=r'C:\models\my "best" ckpt.safetensors',
                    arch="sdxl", quantize=False, noise_scheduler="ddpm")
    cfg = TrainConfig(trainer="ai-toolkit", model=m, dataset_dir=tmp_path,
                      name='weird: name "v2"', trigger='trg, x')
    doc = yaml.safe_load(render_aitoolkit_yaml(cfg))  # must not raise
    proc = doc["config"]["process"][0]
    assert proc["model"]["name_or_path"] == r'C:\models\my "best" ckpt.safetensors'
    assert doc["config"]["name"] == 'weird: name "v2"'
    assert "trg, x" in proc["sample"]["prompts"][0]


def test_aitoolkit_sample_dims_follow_resolution(tmp_path: Path) -> None:
    """Sample width/height track the training resolution instead of a hardcoded 1024."""
    cfg = _cfg("ai-toolkit", tmp_path)
    cfg.resolution = 768
    sample = yaml.safe_load(render_aitoolkit_yaml(cfg))["config"]["process"][0]["sample"]
    assert sample["width"] == 768
    assert sample["height"] == 768


def test_unknown_trainer_raises(tmp_path: Path) -> None:
    cfg = _cfg("ai-toolkit", tmp_path)
    cfg.trainer = "nope"
    with pytest.raises(ValueError):
        write_configs(cfg)


# ---------- 0.17.3: arch strings pinned to what the trainers actually accept ----------

def _preset(trainer: str, key: str) -> ModelPreset:
    return next(p for p in TRAINER_MODELS[trainer] if p.key == key)


# ai-toolkit: extensions_built_in/diffusion_models/ui.tsx. "krea" was never an arch.
AITOOLKIT_ARCHS = {
    "flux2": ("flux2", "black-forest-labs/FLUX.2-dev"),
    "qwen-image": ("qwen_image", "Qwen/Qwen-Image"),
    "qwen-image-2.1": ("qwen_image_2", "Comfy-Org/Qwen-Image-2.1"),
    "zimage": ("zimage", "Tongyi-MAI/Z-Image"),
    "krea": ("krea2", "krea/Krea-2-Raw"),
}

# musubi-tuner: src/musubi_tuner/*_train_network.py and networks/lora_*.py.
MUSUBI_SCRIPTS = {
    "qwen-image": ("qwen_image_train_network.py", "networks.lora_qwen_image"),
    "flux-kontext": ("flux_kontext_train_network.py", "networks.lora_flux"),
    "flux2": ("flux_2_train_network.py", "networks.lora_flux_2"),
    "zimage": ("zimage_train_network.py", "networks.lora_zimage"),
    "krea2": ("krea2_train_network.py", "networks.lora_krea2"),
}


@pytest.mark.parametrize("key", AITOOLKIT_ARCHS)
def test_aitoolkit_arch_and_model_are_pinned(key: str, tmp_path: Path) -> None:
    preset = _preset("ai-toolkit", key)
    assert (preset.arch, preset.name_or_path) == AITOOLKIT_ARCHS[key]
    cfg = _cfg("ai-toolkit", tmp_path).model_copy(update={"model": preset})
    model = yaml.safe_load(render_aitoolkit_yaml(cfg))["config"]["process"][0]["model"]
    assert model["arch"] == preset.arch
    for k, v in preset.model_extras.items():
        assert model[k] == v


def test_no_preset_trains_on_a_turbo_checkpoint() -> None:
    # Turbo checkpoints are distilled; a LoRA must be trained on the base/Raw one.
    for presets in TRAINER_MODELS.values():
        for p in presets:
            assert "turbo" not in p.name_or_path.lower(), p.key


@pytest.mark.parametrize("key", MUSUBI_SCRIPTS)
def test_musubi_script_module_and_cache_steps(key: str, tmp_path: Path) -> None:
    script, module = MUSUBI_SCRIPTS[key]
    preset = _preset("musubi", key)
    assert (preset.musubi_script, preset.network_module) == (script, module)
    cfg = _cfg("musubi", tmp_path).model_copy(update={"model": preset})
    _, command = write_configs(cfg, install_path="/opt/musubi")
    prefix = script.removesuffix("_train_network.py")
    assert f"{prefix}_cache_latents.py" in command
    assert f"{prefix}_cache_text_encoder_outputs.py" in command
    assert f"--network_module {module}" in command
    assert "--network_module networks.lora " not in command
    assert "--save_every_n_epochs 1" in command
    assert "<<FILL: see musubi" not in command


def test_musubi_krea2_matches_its_reference_recipe(tmp_path: Path) -> None:
    preset = _preset("musubi", "krea2")
    assert (preset.rank, preset.alpha) == (32, 32)
    cfg = _cfg("musubi", tmp_path).model_copy(update={"model": preset})
    _, command = write_configs(cfg)
    assert "--discrete_flow_shift 2.5" in command
    train = command.split("accelerate launch", 1)[1]
    # Krea 2 trains from cached text-encoder outputs alone.
    assert "--text_encoder" not in train


def test_aitoolkit_keeps_every_checkpoint(tmp_path: Path) -> None:
    cfg = _cfg("ai-toolkit", tmp_path).model_copy(update={"steps": 3000})
    save = yaml.safe_load(render_aitoolkit_yaml(cfg))["config"]["process"][0]["save"]
    assert save["max_step_saves_to_keep"] * save["save_every"] >= 3000
