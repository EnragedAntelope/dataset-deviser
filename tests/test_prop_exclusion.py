"""Tests for prop exclusion, caption cost estimates, and the export-list merge.

Nothing here touches a network: the cost estimator is pure arithmetic over the
build-time price table, per the project's no-charges testing rule.
"""

from __future__ import annotations

import pytest

from studio.captioner import CaptionerConfigError, estimate_caption_cost, resolve_captioner_config
from studio.shotplan import apply_prop_exclusion, apply_wardrobe, default_plan


def _shot(kind: str):
    return next(s for s in default_plan() if s.kind == kind)


# ---------- prop exclusion ----------

def test_cloud_prompt_gets_the_exclusion_clause() -> None:
    shot = apply_prop_exclusion(_shot("angle"))
    assert "do not include any backpacks" in shot.cloud_prompt


def test_local_prompt_is_never_touched() -> None:
    """Qwen-Image 2.1 draws what a prompt names, even negated — the clause put a
    backpack on the character in 2 of 8 test shots. Gemini follows the negation."""
    for kind in ("angle", "pose", "emotion"):
        original = _shot(kind)
        assert apply_prop_exclusion(original).local_prompt == original.local_prompt


def test_is_idempotent() -> None:
    once = apply_prop_exclusion(_shot("pose"))
    twice = apply_prop_exclusion(once)
    assert once.cloud_prompt == twice.cloud_prompt
    assert once.local_prompt == twice.local_prompt


def test_composes_with_wardrobe() -> None:
    shot = _shot("pose").model_copy(update={"outfit": "a navy peacoat and tan chinos"})
    result = apply_prop_exclusion(apply_wardrobe(shot))
    assert "wearing a navy peacoat and tan chinos" in result.cloud_prompt
    assert "do not include any backpacks" in result.cloud_prompt


def test_default_plan_is_not_mutated() -> None:
    shot = _shot("pose")
    before = shot.cloud_prompt
    apply_prop_exclusion(shot)
    assert shot.cloud_prompt == before


# ---------- caption cost ----------

def test_local_captioner_reports_free() -> None:
    assert "free" in estimate_caption_cost("qwen3vl", "", 24).lower()


def test_gemini_cost_scales_with_image_count() -> None:
    one = estimate_caption_cost("gemini-flash", "gemini-flash-latest", 1)
    many = estimate_caption_cost("gemini-flash", "gemini-flash-latest", 100)
    assert "1 image(s)" in one
    assert "100 image(s)" in many
    assert "$0.16" in many  # 100 * 0.0016


def test_gemini_cost_differs_by_model() -> None:
    flash = estimate_caption_cost("gemini-flash", "gemini-flash-latest", 1000)
    lite = estimate_caption_cost("gemini-flash", "gemini-flash-lite-latest", 1000)
    assert flash != lite


def test_unknown_gemini_model_is_flagged_not_silently_free() -> None:
    out = estimate_caption_cost("gemini-flash", "gemini-99-ultra", 10)
    assert "unlisted model" in out
    assert "$0.00" not in out


def test_zero_selection_shows_per_image_price() -> None:
    out = estimate_caption_cost("gemini-flash", "gemini-flash-latest", 0)
    assert "/image" in out


# ---------- captioner resolution (shared by UI + CLI) ----------

def test_gemini_model_override_passes_through() -> None:
    model, overrides = resolve_captioner_config("gemini-flash", "gemini-2.5-flash")
    assert model == "gemini-2.5-flash"
    assert overrides is None


def test_local_captioner_needs_no_overrides() -> None:
    assert resolve_captioner_config("qwen3vl", "") == ("", None)


def test_unconfigured_custom_endpoint_raises_plain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain exception, not gr.Error — the CLI shares this resolver."""
    from studio import user_config

    monkeypatch.setattr(user_config, "get_custom_captioner", lambda: {})
    with pytest.raises(CaptionerConfigError, match="isn't configured yet"):
        resolve_captioner_config("custom", "")


def test_configured_custom_endpoint_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    from studio import user_config

    monkeypatch.setattr(user_config, "get_custom_captioner", lambda: {
        "base_url": "https://openrouter.ai/api/v1", "model": "qwen/qwen2.5-vl-72b-instruct",
        "api_key_env": "OPENROUTER_API_KEY", "min_interval_s": 1.5})
    model, overrides = resolve_captioner_config("custom", "")
    assert model == "qwen/qwen2.5-vl-72b-instruct"
    assert overrides["base_url"] == "https://openrouter.ai/api/v1"
    assert overrides["api_key_env"] == "OPENROUTER_API_KEY"
    # The key NAME travels; the secret itself never does.
    assert "OPENROUTER_API_KEY" not in str(overrides.get("api_key", ""))


# ---------- export folder merge ----------

def test_export_folders_accumulate() -> None:
    """The 0.3.1 bug: captioning a second folder replaced the first in ④'s list,
    silently dropping it from the export."""
    from app import _merge_export_folders

    merged = _merge_export_folders("C:/runs/prepped", "C:/runs/generated")
    assert merged.splitlines() == ["C:/runs/prepped", "C:/runs/generated"]


def test_export_folders_dedupe() -> None:
    from app import _merge_export_folders

    merged = _merge_export_folders("C:/runs/prepped", "C:/runs/prepped")
    assert merged.splitlines() == ["C:/runs/prepped"]


def test_export_folders_from_empty() -> None:
    from app import _merge_export_folders

    assert _merge_export_folders("", "C:/runs/prepped") == "C:/runs/prepped"
