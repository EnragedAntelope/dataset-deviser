"""Tests for the kohya-ss sd-scripts trainer target."""

from __future__ import annotations

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

from studio.trainer_configs import (
    TRAINER_MODELS,
    TrainConfig,
    kohya_command,
    render_kohya_toml,
    write_configs,
)


def _cfg(tmp_path: Path, key: str = "sdxl") -> TrainConfig:
    preset = next(p for p in TRAINER_MODELS["kohya"] if p.key == key)
    return TrainConfig(trainer="kohya", model=preset, dataset_dir=tmp_path,
                       name="sy-lora", trigger="trig", resolution=1024, rank=16,
                       alpha=16, epochs=10, lr=1e-4, batch_size=2)


def test_kohya_toml_parses_with_subsets(tmp_path: Path) -> None:
    doc = tomllib.loads(render_kohya_toml(_cfg(tmp_path).model_copy(update={"num_repeats": 5})))
    assert doc["general"]["caption_extension"] == ".txt"
    ds = doc["datasets"][0]
    assert ds["resolution"] == [1024, 1024]
    assert ds["batch_size"] == 2
    sub = ds["subsets"][0]
    assert sub["image_dir"] == tmp_path.as_posix()
    assert sub["num_repeats"] == 5


def test_kohya_command_threads_hparams_and_script(tmp_path: Path) -> None:
    cmd = kohya_command("C:/sd-scripts", tmp_path / "kohya-dataset.toml", _cfg(tmp_path),
                        tmp_path / "p.txt")
    assert "sdxl_train_network.py" in cmd
    assert "--network_dim 16" in cmd
    assert "--network_alpha 16" in cmd
    assert "--max_train_epochs 10" in cmd
    assert "stabilityai/stable-diffusion-xl-base-1.0" in cmd
    assert "C:/sd-scripts" in cmd


def test_kohya_custom_preset_keeps_fill(tmp_path: Path) -> None:
    cmd = kohya_command("", tmp_path / "d.toml", _cfg(tmp_path, "sdxl-custom"), tmp_path / "p.txt")
    assert "<<FILL" in cmd  # honest about the user-local checkpoint


def test_write_configs_kohya(tmp_path: Path) -> None:
    written, command = write_configs(_cfg(tmp_path), "C:/sd-scripts")
    assert written[0].name == "kohya-dataset.toml"
    assert written[0].exists()
    assert "accelerate launch sdxl_train_network.py" in command
