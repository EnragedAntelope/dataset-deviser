"""Tests for the single-device pin that replaced device_map="auto" for SAM3 /
multi-GPU captioners (PR #1) — see the multi-GPU gotcha in ARCHITECTURE.md.
"""

from __future__ import annotations

import pytest
import torch

from studio import config


def test_explicit_setting_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.settings, "torch_device", "cuda:1")

    assert config.torch_device() == "cuda:1"


def test_defaults_to_first_cuda_device_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.settings, "torch_device", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert config.torch_device() == "cuda:0"


def test_falls_back_to_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.settings, "torch_device", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert config.torch_device() == "cpu"
