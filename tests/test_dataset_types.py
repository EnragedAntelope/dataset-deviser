"""Tests for Style / Concept dataset types (Phase 1).

The Character path must stay byte-identical (it is the tuned default); Style and
Concept compose their instruction from a per-type framing clause + the per-style
format directive, and finalize trigger-first with no name injection.
"""

from __future__ import annotations

from studio.captioner import SUBJECT_ALIASES, finalize_caption
from studio.config import CAPTIONERS_BY_KEY, DATASET_TYPES

# ---------- constant ----------

def test_dataset_types_constant() -> None:
    assert DATASET_TYPES == ("character", "style", "concept")


# ---------- character path is unchanged ----------

def test_character_prompt_for_is_verbatim_templates() -> None:
    spec = CAPTIONERS_BY_KEY["qwen3vl"]
    # Default dataset_type must return the exact tuned templates (no regression).
    assert spec.prompt_for("prose") == spec.prompt_template
    assert spec.prompt_for("tags") == spec.tags_template
    assert spec.prompt_for("e621") == spec.e621_template
    # Explicit character type is identical to the default.
    assert spec.prompt_for("prose", "character") == spec.prompt_template
    assert spec.prompt_for("tags", "character") == spec.tags_template


def test_character_finalize_unchanged_by_default() -> None:
    # A regression guard: the documented character prose behaviour.
    out = finalize_caption("The woman stands in a park.", "sysnootles",
                           "Sy Snootles", SUBJECT_ALIASES)
    assert out.startswith("sysnootles, ")
    assert "Sy Snootles" in out
    # Same call with dataset_type="character" is identical.
    assert finalize_caption("The woman stands in a park.", "sysnootles",
                            "Sy Snootles", SUBJECT_ALIASES,
                            dataset_type="character") == out


# ---------- style / concept compose (not stored) ----------

def test_style_and_concept_compose_per_style_and_type() -> None:
    spec = CAPTIONERS_BY_KEY["qwen3vl"]
    for dt in ("style", "concept"):
        for style, marker in (("prose", "paragraph"), ("tags", "Danbooru"),
                              ("e621", "e621")):
            prompt = spec.prompt_for(style, dt)
            assert marker.lower() in prompt.lower()  # format directive present
    # Style vs concept framing differs (content vs context).
    assert spec.prompt_for("prose", "style") != spec.prompt_for("prose", "concept")
    assert "content" in spec.prompt_for("prose", "style").lower()
    assert "context" in spec.prompt_for("prose", "concept").lower()


def test_style_prompt_says_not_to_describe_style() -> None:
    spec = CAPTIONERS_BY_KEY["qwen3vl"]
    p = spec.prompt_for("prose", "style").lower()
    assert "do not describe" in p and "style" in p


def test_sparse_is_style_only_and_minimal() -> None:
    spec = CAPTIONERS_BY_KEY["qwen3vl"]
    full = spec.prompt_for("prose", "style", sparse=False)
    sparse = spec.prompt_for("prose", "style", sparse=True)
    assert sparse != full
    assert "few words" in sparse.lower()
    # Concept ignores sparse (no minimal variant).
    assert spec.prompt_for("prose", "concept", sparse=True) == \
        spec.prompt_for("prose", "concept", sparse=False)


def test_composed_prompt_formats_without_error() -> None:
    # .format(subject=...) is called on every returned prompt; composed prompts
    # must not carry stray braces that would raise.
    for key in ("qwen3vl", "joycaption", "qwen3vl-nsfw"):
        spec = CAPTIONERS_BY_KEY[key]
        for dt in ("style", "concept"):
            for style in ("prose", "tags", "e621"):
                spec.prompt_for(style, dt).format(subject="Sy")


def test_joycaption_keeps_refer_convention_in_composed_prompt() -> None:
    joy = CAPTIONERS_BY_KEY["joycaption"]  # prompt_style == "llava"
    generic = CAPTIONERS_BY_KEY["qwen3vl"]
    assert "{subject}" in joy.prompt_for("prose", "style")  # refer-to-them slot
    assert "{subject}" not in generic.prompt_for("prose", "style")


def test_nsfw_spec_keeps_plainness_in_composed_prompt() -> None:
    nsfw = CAPTIONERS_BY_KEY["qwen3vl-nsfw"]
    generic = CAPTIONERS_BY_KEY["qwen3vl"]
    assert "explicit" in nsfw.prompt_for("prose", "style").lower()
    assert "explicit" not in generic.prompt_for("prose", "style").lower()


# ---------- finalize for style / concept ----------

def test_style_concept_prose_is_trigger_first_no_name_injection() -> None:
    for dt in ("style", "concept"):
        out = finalize_caption("The woman stands in a park.", "mytoken",
                               "Sy Snootles", SUBJECT_ALIASES, dataset_type=dt)
        assert out.startswith("mytoken, ")
        # No alias→name mapping for a look/object: the name is not injected.
        assert "Sy Snootles" not in out
        assert "the woman" in out.lower()  # alias left as-is


# ---------- CLI validation ----------

def test_cli_rejects_bad_dataset_type() -> None:
    import pytest
    import typer

    from cli import _check_dataset_type

    assert _check_dataset_type("Style") == "style"  # case-insensitive
    with pytest.raises(typer.BadParameter):
        _check_dataset_type("portrait")


def test_style_concept_tag_finalize_matches_character() -> None:
    # Tag/e621 finalization is identity-free, so dataset_type doesn't change it.
    base = finalize_caption("From Side, forest", "trig", "", SUBJECT_ALIASES, style="tags")
    for dt in ("character", "style", "concept"):
        assert finalize_caption("From Side, forest", "trig", "", SUBJECT_ALIASES,
                                style="tags", dataset_type=dt) == base
