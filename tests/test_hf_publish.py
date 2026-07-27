"""Tests for HF dataset publishing validation/guards (Item 5).

No network: the real upload path is exercised only on a live publish. These
cover the pure validation and the pre-upload guards that must fail fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from studio.hf_publish import (
    HFPublishError,
    normalize_repo_id,
    publish_dataset,
    resolve_token,
)


def test_normalize_repo_id_trims_and_accepts_valid() -> None:
    assert normalize_repo_id("  my-character-lora ") == "my-character-lora"
    assert normalize_repo_id("user/name") == "user/name"
    assert normalize_repo_id("/user/name/") == "user/name"


def test_normalize_repo_id_rejects_empty() -> None:
    with pytest.raises(HFPublishError):
        normalize_repo_id("   ")


@pytest.mark.parametrize("bad", ["a/b/c", "has space", "bad$char", "-leading"])
def test_normalize_repo_id_rejects_malformed(bad: str) -> None:
    with pytest.raises(HFPublishError):
        normalize_repo_id(bad)


def test_resolve_token_prefers_explicit(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert resolve_token("explicit-token") == "explicit-token"


def test_resolve_token_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert resolve_token() == "env-token"


def test_publish_requires_a_token(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ds = tmp_path / "d"
    ds.mkdir()
    with pytest.raises(HFPublishError, match="token"):
        publish_dataset(ds, "my-lora")


def test_publish_requires_existing_folder(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    with pytest.raises(HFPublishError, match="not found"):
        publish_dataset(tmp_path / "nope", "my-lora")


# --- visibility of an already-existing repo --------------------------------
# `create_repo(exist_ok=True)` does not change an existing repo's visibility, so
# publishing "privately" into an already-public repo left it public while the UI
# reported private. That is a privacy claim we must not make silently.

class _FakeInfo:
    def __init__(self, private: bool) -> None:
        self.private = private


class _FakeApi:
    def __init__(self, info: object | Exception) -> None:
        self._info = info

    def dataset_info(self, repo_id: str):
        if isinstance(self._info, Exception):
            raise self._info
        return self._info


def test_existing_public_repo_is_detected() -> None:
    from studio.hf_publish import _is_existing_public_repo

    assert _is_existing_public_repo(_FakeApi(_FakeInfo(private=False)), "u/n")


def test_existing_private_repo_is_not_flagged() -> None:
    from studio.hf_publish import _is_existing_public_repo

    assert not _is_existing_public_repo(_FakeApi(_FakeInfo(private=True)), "u/n")


def test_a_repo_that_does_not_exist_is_not_flagged() -> None:
    """The common case: a brand-new repo must not produce a scary warning."""
    from studio.hf_publish import _is_existing_public_repo

    assert not _is_existing_public_repo(_FakeApi(RuntimeError("404")), "u/n")


def test_lookup_failure_stays_quiet_rather_than_warning_wrongly() -> None:
    from studio.hf_publish import _is_existing_public_repo

    assert not _is_existing_public_repo(_FakeApi(OSError("network down")), "u/n")
