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


# ---------- model resolution (0.17.3: the -preview defaults were shut down) ----------

def _point_cache_at(temp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from studio import config as config_mod
    from studio.engines import gemini

    cache_file = temp_cache_dir / "gemini_image_models.json"
    monkeypatch.setattr(config_mod, "MODEL_CACHE_FILE", cache_file)
    monkeypatch.setattr(gemini, "MODEL_CACHE_FILE", cache_file)
    return cache_file


def test_auto_picks_first_preferred_model_on_offer() -> None:
    from studio.engines import gemini

    pref = gemini.IMAGE_MODEL_PREFERENCE
    assert gemini.resolve_image_model("auto", [pref[1], "other"]) == pref[1]
    assert gemini.resolve_image_model("auto", ["other"]) == pref[0]
    assert gemini.resolve_image_model("gemini-x", []) == "gemini-x"


def test_retired_preview_ids_map_to_their_ga_release() -> None:
    from studio.engines import gemini

    assert gemini.resolve_image_model("gemini-3-pro-image-preview") == "gemini-3-pro-image"


def test_every_preferred_model_has_a_price() -> None:
    from studio.engines import gemini

    assert set(gemini.IMAGE_MODEL_PREFERENCE) <= set(CLOUD_IMAGE_PRICES)


def test_default_model_is_auto_and_choices_lead_with_it(
    temp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.config import Settings
    from studio.engines import gemini

    _point_cache_at(temp_cache_dir, monkeypatch)
    assert Settings.model_fields["gemini_image_model"].default == gemini.AUTO_MODEL
    choices = gemini.image_model_choices(gemini.known_image_models())
    assert choices[0][1] == gemini.AUTO_MODEL
    assert gemini.IMAGE_MODEL_PREFERENCE[0] in choices[0][0]


def test_live_list_keeps_nano_banana_and_drops_retired_previews(
    temp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio import config as config_mod
    from studio.engines import gemini

    _point_cache_at(temp_cache_dir, monkeypatch)
    monkeypatch.setattr(config_mod.settings, "gemini_api_key", "k")
    names = ["gemini-nano-banana-2.1", "gemini-3-pro-image", "gemini-3-pro-image-preview",
             "imagen-4.0-generate", "gemini-flash-latest"]
    listed = [type("M", (), {"name": f"models/{n}"})() for n in names]
    client = type("C", (), {"models": type("Ms", (), {"list": lambda self: listed})()})()
    monkeypatch.setattr(gemini, "gemini_client", lambda key: client)

    ids = [m for _, m in gemini.list_image_models(force_refresh=True)]
    assert ids == ["gemini-nano-banana-2.1", "gemini-3-pro-image"]


def test_refresh_keeps_the_users_pick(monkeypatch: pytest.MonkeyPatch) -> None:
    import app
    from studio.engines import gemini

    live = [("a", "gemini-3-pro-image"), ("b", "gemini-nano-banana-2.1")]
    monkeypatch.setattr(gemini, "list_image_models", lambda force_refresh=False: live)
    assert app.refresh_cloud_models("gemini-nano-banana-2.1").value == "gemini-nano-banana-2.1"
    # A pick the API no longer offers falls back to Auto, never to whatever is first.
    assert app.refresh_cloud_models("gone-model").value == gemini.AUTO_MODEL


# ---------- api_image: what every cloud/VLM call sends ----------

def test_api_image_passes_small_files_through_with_their_real_mime(tmp_path: Path) -> None:
    from PIL import Image

    from studio.config import api_image

    jpg = tmp_path / "a.png"  # wrong extension on purpose: the format decides
    Image.new("RGB", (64, 32)).save(jpg, "JPEG")
    data, mime = api_image(jpg, 1024)
    assert mime == "image/jpeg"
    assert data == jpg.read_bytes()


def test_api_image_downscales_and_keeps_alpha(tmp_path: Path) -> None:
    import io

    from PIL import Image

    from studio.config import api_image

    big = tmp_path / "big.png"
    Image.new("RGBA", (3000, 4000)).save(big)
    data, mime = api_image(big, 2048)
    assert mime == "image/png"
    with Image.open(io.BytesIO(data)) as im:
        assert max(im.size) == 2048 and im.mode == "RGBA"

    photo = tmp_path / "photo.tif"
    Image.new("RGB", (3000, 4000)).save(photo)
    data, mime = api_image(photo, 1536)
    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(data)) as im:
        assert im.size == (1152, 1536)


def test_a_pinned_retired_id_opens_on_its_ga_release_not_auto() -> None:
    # Falling back to Auto (Pro) silently doubled a Flash pin's cost.
    import app

    choices = [("a", "auto"), ("b", "gemini-3-pro-image"), ("c", "gemini-3.1-flash-image")]
    assert app._cloud_model_default("gemini-3.1-flash-image-preview", choices) == \
        "gemini-3.1-flash-image"
    assert app._cloud_model_default("auto", choices) == "auto"
    assert app._cloud_model_default("some-unlisted-id", choices) == "auto"


def test_api_image_keeps_16_bit_grey_and_palette_transparency(tmp_path: Path) -> None:
    import io

    import numpy as np
    from PIL import Image

    from studio.config import api_image

    grey = tmp_path / "grey16.png"
    Image.fromarray(np.full((3000, 3000), 32768, dtype=np.uint16)).save(grey)
    data, _ = api_image(grey, 1536)
    with Image.open(io.BytesIO(data)) as im:
        assert 120 <= im.convert("L").getpixel((10, 10)) <= 136  # mid-grey, not clipped white

    pal = tmp_path / "pal.png"
    p = Image.new("P", (3000, 3000), 1)
    p.putpalette([255, 0, 0, 0, 0, 255] + [0] * 762)
    p.info["transparency"] = 0
    p.paste(0, (0, 0, 100, 100))  # transparent corner
    p.save(pal, transparency=0)
    data, mime = api_image(pal, 1536)
    assert mime == "image/png"
    with Image.open(io.BytesIO(data)) as im:
        assert im.mode == "RGBA" and im.getpixel((2, 2))[3] == 0


def _engine_capturing(monkeypatch: pytest.MonkeyPatch, model: str) -> tuple[object, dict]:
    """A GeminiEngine whose client records the last generate_content call."""
    from studio import config as config_mod
    from studio.engines import gemini

    seen: dict = {}

    class Models:
        def generate_content(self, **kw: object) -> object:
            seen.update(kw)
            part = type("P", (), {"inline_data": type("D", (), {"data": b"png"})()})()
            content = type("Ct", (), {"parts": [part]})()
            return type("R", (), {"candidates": [type("Cd", (), {"content": content})()]})()

    monkeypatch.setattr(config_mod.settings, "gemini_api_key", "k")
    monkeypatch.setattr(gemini, "gemini_client", lambda key: type("C", (), {"models": Models()})())
    return gemini.GeminiEngine(model=model), seen


@pytest.mark.parametrize(("model", "cap"), [("gemini-3-pro-image", 5), ("gemini-nano-banana-2.1", 4)])
def test_references_are_capped_at_the_models_character_limit(
    model: str, cap: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image

    from studio.shotplan import default_plan

    engine, seen = _engine_capturing(monkeypatch, model)
    refs = []
    for i in range(7):
        refs.append(tmp_path / f"r{i}.png")
        Image.new("RGB", (8, 8)).save(refs[-1])

    engine.generate(refs + [refs[0]], default_plan()[0], tmp_path / "o.png", seed=1)

    assert len(seen["contents"]) == cap + 1  # references + the prompt


def test_aspect_follows_the_shots_framing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from PIL import Image

    from studio.engines.gemini import shot_aspect
    from studio.shotplan import default_plan

    by = {(s.kind, s.framing): shot_aspect(s) for s in default_plan()}
    assert by[("angle", "full")] == "2:3"
    assert by[("emotion", "close-up")] == by[("emotion", "tight-face")] == "1:1"
    # A pose keeps the reference's aspect: a crouch squeezed into 2:3 loses the body.
    assert by[("pose", "full")] is None and by[("pose", "waist-up")] is None

    engine, seen = _engine_capturing(monkeypatch, "gemini-nano-banana-2.1")
    ref = tmp_path / "r.png"
    Image.new("RGB", (8, 8)).save(ref)
    angle = default_plan()[0]
    engine.generate([ref], angle, tmp_path / "o.png", seed=1)
    assert seen["config"].image_config.aspect_ratio == "2:3"
    assert seen["config"].image_config.image_size is None  # 1K stays the model default
