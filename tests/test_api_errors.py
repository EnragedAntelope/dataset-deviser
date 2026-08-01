"""Tests for cloud-API error humanising, retry classification and the genai client.

A Gemini 503 used to reach the user as a full traceback plus the raw JSON error
body, and abort a batch that had already paid for 15 captions.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from studio import config


class _GenaiError(Exception):
    """Shaped like google.genai.errors.APIError — status lives on `.code`."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{code} UNAVAILABLE. {{'error': {{'code': {code}}}}}")
        self.code = code
        self.message = message


# ---------- status extraction ----------

def test_api_status_code_reads_genai_code() -> None:
    assert config.api_status_code(_GenaiError(503, "high demand")) == 503


def test_api_status_code_reads_httpx_response() -> None:
    request = httpx.Request("POST", "https://example.invalid/v1/chat")
    response = httpx.Response(429, request=request)
    err = httpx.HTTPStatusError("rate limited", request=request, response=response)
    assert config.api_status_code(err) == 429


def test_api_status_code_is_none_for_a_plain_exception() -> None:
    assert config.api_status_code(RuntimeError("boom")) is None


# ---------- retry classification ----------

@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_transient_statuses_are_retried(code: int) -> None:
    assert config.is_transient_api_error(_GenaiError(code, ""))


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_are_not_retried(code: int) -> None:
    # Retrying a bad key or a bad request just makes the user wait for the same
    # failure — and for a billed API, pay for it more than once.
    assert not config.is_transient_api_error(_GenaiError(code, ""))


def test_unknown_exceptions_are_not_retried() -> None:
    assert not config.is_transient_api_error(ValueError("nope"))


# ---------- friendly text ----------

def test_friendly_error_names_the_503_cause_and_the_way_out() -> None:
    text = config.friendly_api_error(_GenaiError(503, "high demand"))
    assert "overloaded (503)" in text
    assert "Groq" in text  # the actionable alternative, not just "try again"
    assert "{" not in text  # never the raw JSON body


def test_friendly_error_trims_an_unrecognised_failure() -> None:
    text = config.friendly_api_error(RuntimeError("x" * 500))
    assert len(text) <= 301
    assert text.endswith("…")


def test_friendly_error_collapses_whitespace() -> None:
    assert config.friendly_api_error(RuntimeError("a\n  b\tc")) == "a b c"


def test_friendly_error_never_returns_empty() -> None:
    # An exception with no message must still say something nameable.
    assert config.friendly_api_error(TimeoutError()) == "TimeoutError"


# ---------- genai client ----------

def test_gemini_client_silences_the_misleading_dual_key_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """We always pass api_key= explicitly, so "Using GOOGLE_API_KEY" is untrue here."""
    import google.genai as genai

    monkeypatch.setattr(genai, "Client", lambda **kwargs: kwargs)
    logger = logging.getLogger("google_genai._api_client")
    for existing in list(logger.filters):
        logger.removeFilter(existing)

    assert config.gemini_client("k") == {"api_key": "k"}
    record = logger.makeRecord(
        logger.name, logging.WARNING, __file__, 0,
        "Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.",
        (), None)
    kept = logger.makeRecord(logger.name, logging.WARNING, __file__, 0,
                             "something else worth seeing", (), None)
    assert not logger.filter(record)  # dropped
    assert logger.filter(kept)  # every other genai warning still comes through


def test_gemini_client_installs_its_filter_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import google.genai as genai

    monkeypatch.setattr(genai, "Client", lambda **kwargs: kwargs)
    logger = logging.getLogger("google_genai._api_client")
    for existing in list(logger.filters):
        logger.removeFilter(existing)

    for _ in range(3):
        config.gemini_client("k")
    assert len(logger.filters) == 1
