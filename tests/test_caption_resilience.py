"""Tests for surviving a mid-batch cloud failure.

Before 0.14.0 `caption_images` collected every caption in memory and the caller
wrote sidecars only after the loop returned — so one 503 on the last image threw
away every caption already generated and billed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from studio import captioner as C
from studio import config


class _Boom(Exception):
    def __init__(self, code: int = 503) -> None:
        super().__init__(f"{code} UNAVAILABLE")
        self.code = code


def _images(tmp_path: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        p = tmp_path / f"{i:02d}.png"
        p.write_bytes(b"")
        paths.append(p)
    return paths


@pytest.fixture
def fake_captioner(monkeypatch: pytest.MonkeyPatch):
    """Replace Captioner with a scripted stub; `fail_on` raises for that filename."""
    calls: list[Path] = []

    class _Stub:
        spec = C.CAPTIONERS_BY_KEY["gemini-flash"]

        def __init__(self, *a, **kw) -> None:
            self.unloaded = False

        def caption(self, path: Path, **kw) -> str:
            calls.append(path)
            if path.name == _Stub.fail_on:
                raise _Boom()
            return f"a photo of {path.stem}"

        def load(self) -> None:
            pass

        def unload(self) -> None:
            _Stub.unloaded = True

    _Stub.fail_on = ""
    _Stub.unloaded = False
    monkeypatch.setattr(C, "Captioner", _Stub)
    return _Stub, calls


def test_on_item_fires_once_per_image(tmp_path: Path, fake_captioner) -> None:
    images = _images(tmp_path, 3)
    seen: list[tuple[Path, str]] = []
    items = C.caption_images(images, "gemini-flash", on_item=lambda p, c: seen.append((p, c)))
    assert [p for p, _ in seen] == images
    assert seen == items


def test_on_item_receives_the_finalized_caption(tmp_path: Path, fake_captioner) -> None:
    # Trigger/prefix must already be applied — the callback writes the sidecar verbatim.
    images = _images(tmp_path, 1)
    seen: list[str] = []
    C.caption_images(images, "gemini-flash", trigger="spkc0rn",
                     on_item=lambda p, c: seen.append(c))
    assert seen[0].startswith("spkc0rn")


def test_a_mid_batch_failure_keeps_the_captions_already_written(
    tmp_path: Path, fake_captioner
) -> None:
    stub, _ = fake_captioner
    stub.fail_on = "03.png"
    images = _images(tmp_path, 5)
    written: list[Path] = []

    def persist(img: Path, caption: str) -> None:
        img.with_suffix(".txt").write_text(caption, encoding="utf-8")
        written.append(img)

    with pytest.raises(_Boom):
        C.caption_images(images, "gemini-flash", on_item=persist)

    assert written == images[:3]
    assert all(p.with_suffix(".txt").exists() for p in images[:3])
    assert not any(p.with_suffix(".txt").exists() for p in images[3:])


def test_the_model_is_still_freed_when_a_batch_fails(tmp_path: Path, fake_captioner) -> None:
    stub, _ = fake_captioner
    stub.fail_on = "00.png"
    with pytest.raises(_Boom):
        C.caption_images(_images(tmp_path, 2), "gemini-flash", on_item=lambda p, c: None)
    assert stub.unloaded


def test_caption_images_still_works_without_a_callback(tmp_path: Path, fake_captioner) -> None:
    # cli.py and caption_folder() call it positionally with no on_item.
    items = C.caption_images(_images(tmp_path, 2), "gemini-flash")
    assert len(items) == 2


# ---------- Gemini retry ladder ----------

def _gemini_captioner(monkeypatch: pytest.MonkeyPatch, responses: list):
    """A Captioner whose genai client yields `responses` in order (exceptions raise)."""
    monkeypatch.setattr(config.settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(C, "GEMINI_BACKOFF_S", 0)  # no real sleeping in tests
    attempts: list[int] = []

    class _Models:
        def generate_content(self, **kw):
            attempts.append(1)
            nxt = responses.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

    monkeypatch.setattr(config, "gemini_client", lambda key: type("C", (), {"models": _Models()})())
    return C.Captioner("gemini-flash"), attempts


def test_gemini_retries_a_503_then_succeeds(tmp_path: Path, monkeypatch) -> None:
    img = tmp_path / "a.png"
    img.write_bytes(b"")
    ok = type("R", (), {"text": "a caption"})()
    cap, attempts = _gemini_captioner(monkeypatch, [_Boom(503), _Boom(503), ok])
    assert cap._caption_gemini(img, "describe") == "a caption"
    assert len(attempts) == 3


def test_gemini_gives_up_after_the_last_attempt(tmp_path: Path, monkeypatch) -> None:
    img = tmp_path / "a.png"
    img.write_bytes(b"")
    cap, attempts = _gemini_captioner(monkeypatch, [_Boom(503)] * C.GEMINI_RETRIES)
    with pytest.raises(_Boom):
        cap._caption_gemini(img, "describe")
    assert len(attempts) == C.GEMINI_RETRIES


def test_gemini_does_not_retry_a_bad_request(tmp_path: Path, monkeypatch) -> None:
    img = tmp_path / "a.png"
    img.write_bytes(b"")
    cap, attempts = _gemini_captioner(monkeypatch, [_Boom(400)])
    with pytest.raises(_Boom):
        cap._caption_gemini(img, "describe")
    assert len(attempts) == 1
