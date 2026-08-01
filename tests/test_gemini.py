"""Tests for Gemini model cache and listing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from studio.config import CLOUD_IMAGE_PRICES


def _write_cache(cache_dir: Path, models: list[dict], cached_at: datetime | None = None) -> Path:
    cached_at = cached_at or datetime.now(tz=timezone.utc)
    path = cache_dir / "gemini_image_models.json"
    import json  # noqa: PLC0415

    path.write_text(
        json.dumps({"cached_at": cached_at.isoformat(), "models": models}, indent=2),
        encoding="utf-8",
    )
    return path


def test_load_model_cache_returns_fresh_cache(temp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from studio import config as config_mod
    from studio.engines import gemini

    cache_file = temp_cache_dir / "gemini_image_models.json"
    monkeypatch.setattr(config_mod, "MODEL_CACHE_FILE", cache_file)
    monkeypatch.setattr(gemini, "MODEL_CACHE_FILE", cache_file)

    models = [
        {"model_id": "gemini-test", "display_name": "Test", "price": 0.1},
    ]
    _write_cache(temp_cache_dir, models)

    cached = gemini._load_model_cache()
    assert cached is not None
    assert len(cached) == 1
    assert cached[0]["model_id"] == "gemini-test"


def test_load_model_cache_returns_none_when_stale(temp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from studio import config as config_mod
    from studio.engines import gemini

    cache_file = temp_cache_dir / "gemini_image_models.json"
    monkeypatch.setattr(config_mod, "MODEL_CACHE_FILE", cache_file)
    monkeypatch.setattr(gemini, "MODEL_CACHE_FILE", cache_file)

    stale_at = datetime.now(tz=timezone.utc) - timedelta(hours=48)
    _write_cache(temp_cache_dir, [{"model_id": "stale", "price": 0.1}], cached_at=stale_at)

    assert gemini._load_model_cache() is None


def test_load_model_cache_returns_none_when_missing(temp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from studio import config as config_mod
    from studio.engines import gemini

    missing_file = temp_cache_dir / "missing.json"
    monkeypatch.setattr(config_mod, "MODEL_CACHE_FILE", missing_file)
    monkeypatch.setattr(gemini, "MODEL_CACHE_FILE", missing_file)
    assert gemini._load_model_cache() is None


def test_save_model_cache_roundtrip(temp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from studio import config as config_mod
    from studio.engines import gemini

    cache_file = temp_cache_dir / "gemini_image_models.json"
    monkeypatch.setattr(config_mod, "MODEL_CACHE_FILE", cache_file)
    monkeypatch.setattr(gemini, "MODEL_CACHE_FILE", cache_file)

    models = [{"model_id": "roundtrip", "display_name": "Round", "price": 0.2}]
    gemini._save_model_cache(models)
    cached = gemini._load_model_cache()
    assert cached is not None
    assert cached[0]["model_id"] == "roundtrip"


def test_list_image_models_fallback_without_key(
    temp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio import config as config_mod
    from studio.engines import gemini

    # Point at an empty cache like the tests above. Without this the assertion is
    # decided by whether whoever runs the suite has used the app recently: a warm
    # .cache/gemini_image_models.json short-circuits the lookup and returns live
    # model ids instead of the static fallback table.
    cache_file = temp_cache_dir / "gemini_image_models.json"
    monkeypatch.setattr(config_mod, "MODEL_CACHE_FILE", cache_file)
    monkeypatch.setattr(gemini, "MODEL_CACHE_FILE", cache_file)
    monkeypatch.setattr(config_mod.settings, "gemini_api_key", "")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LDS_GEMINI_API_KEY", raising=False)

    models = gemini.list_image_models()
    ids = {m[1] for m in models}
    assert ids == set(CLOUD_IMAGE_PRICES)
