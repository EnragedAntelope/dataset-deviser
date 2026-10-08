"""Tests for captioner registry configuration."""

from __future__ import annotations

from studio.captioner import Captioner
from studio.config import CAPTIONERS_BY_KEY


def test_groq_qwen_spec_exists() -> None:
    spec = CAPTIONERS_BY_KEY["groq-qwen"]
    # qwen3.6-27b was shut down 2026-09-14.
    assert spec.model == "qwen/qwen3.8-27b"
    assert spec.backend == "openai"
    assert spec.base_url == "https://api.groq.com/openai/v1"
    assert spec.api_key_env == "GROQ_API_KEY"


def test_retired_groq_key_still_resolves() -> None:
    # An .env naming the old key must keep working, and the ③ dropdown must get
    # the current key (the alias is not one of its choices).
    from studio.config import Settings

    assert CAPTIONERS_BY_KEY["groq-qwen3.6"].key == "groq-qwen"
    assert Settings(default_captioner="groq-qwen3.6").default_captioner == "groq-qwen"


def test_groq_qwen_rate_interval_fits_8k_tpm() -> None:
    # ~2.6K tokens a call against 8K tokens/minute: anything under ~20 s 429s.
    spec = CAPTIONERS_BY_KEY["groq-qwen"]
    assert spec.min_interval_s >= 20.0


def test_groq_llama4_scout_removed() -> None:
    # Decommissioned by Groq — must not linger in the registry.
    assert "groq-llama4-scout" not in CAPTIONERS_BY_KEY


def test_groq_qwen_disables_reasoning() -> None:
    # A thinking model; its scratchpad is switched off so the response is just
    # the caption, and the budget stays small because Groq bills it per minute.
    spec = CAPTIONERS_BY_KEY["groq-qwen"]
    assert spec.extra_params.get("reasoning_effort") == "none"
    assert 300 <= spec.max_tokens <= 400


def test_gemini_caption_default_is_rolling_alias() -> None:
    spec = CAPTIONERS_BY_KEY["gemini-flash"]
    assert spec.model == "gemini-flash-latest"
    assert spec.backend == "gemini"


def test_custom_captioner_spec_overrides_apply() -> None:
    cap = Captioner("custom", spec_overrides={
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_interval_s": 3.0,
    })
    assert cap.spec.base_url == "https://openrouter.ai/api/v1"
    assert cap.spec.api_key_env == "OPENROUTER_API_KEY"
    assert cap.spec.min_interval_s == 3.0
    # Registry spec must stay pristine (model_copy, not in-place mutation).
    assert CAPTIONERS_BY_KEY["custom"].base_url == ""


def test_model_override_wins_over_spec_model() -> None:
    cap = Captioner("gemini-flash", model_override="gemini-3.5-flash")
    assert cap.model == "gemini-3.5-flash"
