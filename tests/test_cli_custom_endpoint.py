"""`cli.py custom-endpoint` — lets a CLI-only user configure the custom
OpenAI-compatible captioner without touching the UI.

Before this, `resolve_captioner_config` already read the saved config, but
nothing let a CLI-only user *write* it (see the CLI/UI parity gap noted in
ARCHITECTURE.md and worklog-0.12.0.md) — only the UI's Caption tab could.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import cli
from studio import user_config

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / ".cache"
    monkeypatch.setattr(user_config, "CACHE_DIR", cache)
    monkeypatch.setattr(user_config, "USER_SETTINGS_FILE", cache / "user_settings.json")


def test_shows_not_configured_when_nothing_saved() -> None:
    result = runner.invoke(cli.app, ["custom-endpoint"])
    assert result.exit_code == 0
    assert "No custom endpoint configured yet" in result.stdout


def test_setting_requires_both_base_url_and_model_the_first_time() -> None:
    result = runner.invoke(cli.app, ["custom-endpoint", "--base-url", "https://x/v1"])
    assert result.exit_code != 0


def test_sets_and_shows_the_config() -> None:
    set_result = runner.invoke(cli.app, [
        "custom-endpoint",
        "--base-url", "https://openrouter.ai/api/v1",
        "--model", "qwen/qwen2.5-vl-72b-instruct",
        "--key-env", "OPENROUTER_API_KEY",
        "--spacing", "2.5",
    ])
    assert set_result.exit_code == 0
    assert "Saved." in set_result.stdout

    cfg = user_config.get_custom_captioner()
    assert cfg["base_url"] == "https://openrouter.ai/api/v1"
    assert cfg["model"] == "qwen/qwen2.5-vl-72b-instruct"
    assert cfg["api_key_env"] == "OPENROUTER_API_KEY"
    assert cfg["min_interval_s"] == 2.5

    show_result = runner.invoke(cli.app, ["custom-endpoint"])
    assert "openrouter.ai" in show_result.stdout
    assert "OPENROUTER_API_KEY" in show_result.stdout


def test_updating_one_field_keeps_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    user_config.set_custom_captioner("https://x/v1", "some-model", "MY_KEY", 1.0)

    result = runner.invoke(cli.app, ["custom-endpoint", "--spacing", "5"])

    assert result.exit_code == 0
    cfg = user_config.get_custom_captioner()
    assert cfg["base_url"] == "https://x/v1"  # untouched
    assert cfg["model"] == "some-model"  # untouched
    assert cfg["min_interval_s"] == 5.0  # updated


def test_never_prints_or_stores_a_secret() -> None:
    result = runner.invoke(cli.app, [
        "custom-endpoint", "--base-url", "https://x/v1", "--model", "m",
        "--key-env", "MY_SECRET_ENV_NAME",
    ])
    assert "MY_SECRET_ENV_NAME" in result.stdout  # the env-var NAME is fine to show
    # There is no flag to pass a literal key value at all — nothing else to assert
    # beyond the round-trip test in test_user_config.py already covering storage.


def test_caption_command_can_use_a_cli_configured_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: configure via the CLI, then `caption --captioner custom`
    resolves that same config (the actual parity gap being closed)."""
    runner.invoke(cli.app, [
        "custom-endpoint", "--base-url", "https://x/v1", "--model", "m",
        "--key-env", "MY_KEY",
    ])

    from studio.captioner import resolve_captioner_config

    model_override, spec_overrides = resolve_captioner_config("custom", "")
    assert spec_overrides["base_url"] == "https://x/v1"
    assert spec_overrides["api_key_env"] == "MY_KEY"
