"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_run_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite from writing into the developer's own working folders.

    `cli.build` / `cli.generate` call the real `pipeline.new_run_dir()` even when
    the stage functions themselves are stubbed, so every run of the suite left a
    fresh `runs/<stamp>/` behind in the repo — 59 stray folders had accumulated,
    some holding a full 24-shot set. Redirecting the three output roots per test
    makes the suite hermetic and keeps `runs/` meaning "work the user did".
    """
    from studio import user_config
    from studio.config import settings

    for attr in ("runs_dir", "output_root", "shot_plans_dir"):
        monkeypatch.setattr(settings, attr, tmp_path / attr, raising=False)
    # UI handlers persist choices (dataset type, ⑤ settings); without this the suite
    # overwrote the developer's own .cache/user_settings.json.
    monkeypatch.setattr(user_config, "USER_SETTINGS_FILE", tmp_path / "user_settings.json")


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    cache = tmp_path / ".cache"
    cache.mkdir()
    return cache
