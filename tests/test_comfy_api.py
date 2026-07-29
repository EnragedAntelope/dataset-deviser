"""Tests for the ComfyUI model-filename matching/caching seam (PR #1 fixes).

No network: httpx.get is monkeypatched to a fake /object_info response.
"""

from __future__ import annotations

import pytest

from studio import comfy_api


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _combo_payload(class_type: str, input_key: str, options: list[str]) -> dict:
    return {class_type: {"input": {"required": {input_key: [options]}}}}


@pytest.fixture(autouse=True)
def _clear_combo_cache() -> None:
    comfy_api._combo_cache.clear()
    yield
    comfy_api._combo_cache.clear()


def test_server_filename_matches_ignoring_separator(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _combo_payload("CheckpointLoaderSimple", "ckpt_name", ["qwen\\model.safetensors"])
    monkeypatch.setattr(comfy_api.httpx, "get", lambda *a, **k: _FakeResponse(payload))

    result = comfy_api._server_filename(
        "CheckpointLoaderSimple", "ckpt_name", "qwen/model.safetensors", "sam3_checkpoint"
    )

    assert result == "qwen\\model.safetensors"


def test_server_filename_case_insensitive_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _combo_payload("CheckpointLoaderSimple", "ckpt_name", ["Qwen/Model.safetensors"])
    monkeypatch.setattr(comfy_api.httpx, "get", lambda *a, **k: _FakeResponse(payload))

    result = comfy_api._server_filename(
        "CheckpointLoaderSimple", "ckpt_name", "qwen/model.safetensors", "sam3_checkpoint"
    )

    assert result == "Qwen/Model.safetensors"


def test_server_filename_raises_with_hint_on_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _combo_payload(
        "CheckpointLoaderSimple", "ckpt_name", ["qwen/close-match.safetensors"]
    )
    monkeypatch.setattr(comfy_api.httpx, "get", lambda *a, **k: _FakeResponse(payload))

    with pytest.raises(comfy_api.ComfyError) as exc_info:
        comfy_api._server_filename(
            "CheckpointLoaderSimple", "ckpt_name", "qwen/close-match2.safetensors",
            "sam3_checkpoint",
        )

    message = str(exc_info.value)
    assert "LDS_SAM3_CHECKPOINT" in message
    assert "qwen/close-match.safetensors" in message


def test_server_filename_passes_through_when_server_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*a: object, **k: object) -> None:
        raise ConnectionError("no server")

    monkeypatch.setattr(comfy_api.httpx, "get", _boom)

    result = comfy_api._server_filename(
        "CheckpointLoaderSimple", "ckpt_name", "whatever.safetensors", "sam3_checkpoint"
    )

    assert result == "whatever.safetensors"


def test_combo_options_caches_only_successful_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _flaky_get(*a: object, **k: object) -> _FakeResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("timeout")
        return _FakeResponse(_combo_payload("Node", "field", ["a.safetensors"]))

    monkeypatch.setattr(comfy_api.httpx, "get", _flaky_get)

    first = comfy_api._combo_options("Node", "field")
    second = comfy_api._combo_options("Node", "field")

    assert first == []  # failed lookup: not cached, returns empty
    assert second == ["a.safetensors"]  # retried and cached
    assert calls["n"] == 2


def test_combo_options_reuses_cache_without_refetching(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _get(*a: object, **k: object) -> _FakeResponse:
        calls["n"] += 1
        return _FakeResponse(_combo_payload("Node", "field", ["a.safetensors"]))

    monkeypatch.setattr(comfy_api.httpx, "get", _get)

    comfy_api._combo_options("Node", "field")
    comfy_api._combo_options("Node", "field")

    assert calls["n"] == 1
