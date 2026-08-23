"""Regression tests for the 0.4.0 trainer-config fixes."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from studio.trainer_configs import (
    TRAINER_MODELS,
    TrainConfig,
    musubi_command,
    render_aitoolkit_yaml,
    render_musubi_toml,
    write_configs,
)


def _cfg(trainer: str, tmp_path: Path, **kw) -> TrainConfig:
    preset = TRAINER_MODELS[trainer][0]
    base = {"trainer": trainer, "model": preset, "dataset_dir": tmp_path,
            "name": "sy-lora", "trigger": "sysnootles", "resolution": 1024,
            "rank": 32, "alpha": 64, "steps": 1800, "lr": 8e-5, "batch_size": 2}
    base.update(kw)
    return TrainConfig(**base)


def test_musubi_command_honors_every_hyperparameter(tmp_path: Path) -> None:
    """The 0.3.1 bug: musubi_command() took only the ModelPreset, so rank/steps
    were hardcoded and every ⑤-tab slider was silently discarded."""
    cfg = _cfg("musubi", tmp_path)
    cmd = musubi_command("C:/musubi", tmp_path / "dataset.toml", cfg)

    assert "--network_dim 32" in cmd
    assert "--network_alpha 64" in cmd
    assert "--max_train_steps 1800" in cmd
    assert "--learning_rate 8e-05" in cmd
    assert "sy-lora" in cmd
    # The old code hardcoded these; make sure they can't creep back.
    assert "--network_dim 16" not in cmd
    assert "--max_train_steps 2000" not in cmd


def test_musubi_command_has_no_unrendered_fstring_braces(tmp_path: Path) -> None:
    """The old code emitted the literal `{16}` from a broken f-string."""
    cmd = musubi_command("C:/musubi", tmp_path / "dataset.toml", _cfg("musubi", tmp_path))
    # <<FILL: ...>> placeholders are intentional; bare braces are not.
    assert not re.search(r"\{\d", cmd)
    assert "{16}" not in cmd


def test_write_configs_threads_config_into_musubi_command(tmp_path: Path) -> None:
    _written, command = write_configs(_cfg("musubi", tmp_path), "C:/musubi")
    assert "--network_dim 32" in command
    assert "--max_train_steps 1800" in command


def test_musubi_toml_scales_repeats(tmp_path: Path) -> None:
    toml = render_musubi_toml(_cfg("musubi", tmp_path), num_repeats=17)
    assert "num_repeats = 17" in toml


def test_musubi_toml_does_not_upscale_into_buckets(tmp_path: Path) -> None:
    toml = render_musubi_toml(_cfg("musubi", tmp_path))
    assert "enable_bucket = true" in toml
    assert "bucket_no_upscale = true" in toml


def test_aitoolkit_emits_multi_resolution_buckets(tmp_path: Path) -> None:
    cfg = _cfg("ai-toolkit", tmp_path, buckets=[512, 768, 1024])
    yaml = render_aitoolkit_yaml(cfg)
    assert "resolution: [512, 768, 1024]" in yaml


def test_aitoolkit_falls_back_to_single_bucket(tmp_path: Path) -> None:
    yaml = render_aitoolkit_yaml(_cfg("ai-toolkit", tmp_path, buckets=[]))
    assert "resolution: [1024]" in yaml


def test_aitoolkit_respects_sliders(tmp_path: Path) -> None:
    yaml = render_aitoolkit_yaml(_cfg("ai-toolkit", tmp_path))
    assert "linear: 32" in yaml
    assert "linear_alpha: 64" in yaml
    assert "steps: 1800" in yaml
    assert "batch_size: 2" in yaml


def test_no_secrets_are_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-SHOULD-NOT-APPEAR")
    written, command = write_configs(_cfg("ai-toolkit", tmp_path), "C:/ai-toolkit")
    body = written[0].read_text(encoding="utf-8")
    assert "SECRET-SHOULD-NOT-APPEAR" not in body
    assert "SECRET-SHOULD-NOT-APPEAR" not in command
