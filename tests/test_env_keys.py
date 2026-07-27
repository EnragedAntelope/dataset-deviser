"""`.env` key management: the logic that used to live (buggily) in setup.bat/setup.sh.

The shell versions could not be tested; that is why this moved into Python. These
tests pin the behaviours the shell scripts got wrong: appending to a file with no
trailing newline, updating a key instead of duplicating or skipping it, and never
printing a whole secret.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from studio import env_keys


# --- reading ---------------------------------------------------------------

def test_read_env_parses_pairs_and_ignores_noise(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "GEMINI_API_KEY=abc123\n"
        "export GROQ_API_KEY=gsk_xyz\n"
        "  HF_TOKEN = hf_spaced  \n"
        "NOT_A_PAIR\n",
        encoding="utf-8",
    )
    assert env_keys.read_env(env) == {
        "GEMINI_API_KEY": "abc123",
        "GROQ_API_KEY": "gsk_xyz",
        "HF_TOKEN": "hf_spaced",
    }


def test_read_env_missing_file_is_empty(tmp_path: Path) -> None:
    assert env_keys.read_env(tmp_path / "nope.env") == {}


def test_read_env_keeps_values_containing_equals(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("TOKEN=a=b=c\n", encoding="utf-8")
    assert env_keys.read_env(env)["TOKEN"] == "a=b=c"


def test_read_env_strips_surrounding_quotes(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text('GEMINI_API_KEY="quoted"\nGROQ_API_KEY=\'single\'\n', encoding="utf-8")
    parsed = env_keys.read_env(env)
    assert parsed == {"GEMINI_API_KEY": "quoted", "GROQ_API_KEY": "single"}


# --- writing ---------------------------------------------------------------

def test_set_key_creates_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env_keys.set_key(env, "GEMINI_API_KEY", "abc")
    assert env_keys.read_env(env)["GEMINI_API_KEY"] == "abc"


def test_set_key_appends_without_gluing_onto_a_file_lacking_a_final_newline(
    tmp_path: Path,
) -> None:
    """`echo K=v>> .env` in setup.bat glued the new key onto the last line."""
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=abc", encoding="utf-8")  # no trailing newline
    env_keys.set_key(env, "GROQ_API_KEY", "xyz")
    assert env_keys.read_env(env) == {"GEMINI_API_KEY": "abc", "GROQ_API_KEY": "xyz"}
    assert "abcGROQ_API_KEY" not in env.read_text(encoding="utf-8")


def test_set_key_updates_in_place_without_duplicating(tmp_path: Path) -> None:
    """setup.bat skipped an existing key (typo unfixable); setup.sh duplicated it."""
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=old\nGROQ_API_KEY=keep\n", encoding="utf-8")
    env_keys.set_key(env, "GEMINI_API_KEY", "new")
    body = env.read_text(encoding="utf-8")
    assert body.count("GEMINI_API_KEY=") == 1
    assert env_keys.read_env(env) == {"GEMINI_API_KEY": "new", "GROQ_API_KEY": "keep"}


def test_set_key_preserves_comments_and_other_lines(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("# keep me\nGEMINI_API_KEY=old\n# and me\n", encoding="utf-8")
    env_keys.set_key(env, "GEMINI_API_KEY", "new")
    body = env.read_text(encoding="utf-8")
    assert "# keep me" in body and "# and me" in body


def test_set_key_updates_an_exported_line_in_place(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("export GEMINI_API_KEY=old\n", encoding="utf-8")
    env_keys.set_key(env, "GEMINI_API_KEY", "new")
    assert env.read_text(encoding="utf-8").count("GEMINI_API_KEY") == 1
    assert env_keys.read_env(env)["GEMINI_API_KEY"] == "new"


def test_set_key_rejects_a_value_with_a_newline(tmp_path: Path) -> None:
    """A pasted multi-line value must not be able to inject a second setting."""
    env = tmp_path / ".env"
    with pytest.raises(ValueError):
        env_keys.set_key(env, "GEMINI_API_KEY", "abc\nGROQ_API_KEY=evil")


def test_set_key_rejects_an_unknown_name_shape(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    with pytest.raises(ValueError):
        env_keys.set_key(env, "not a name", "abc")


def test_set_key_trims_surrounding_whitespace(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env_keys.set_key(env, "GEMINI_API_KEY", "  abc  ")
    assert env_keys.read_env(env)["GEMINI_API_KEY"] == "abc"


def test_unset_key_removes_the_line(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=abc\nGROQ_API_KEY=xyz\n", encoding="utf-8")
    env_keys.unset_key(env, "GEMINI_API_KEY")
    assert env_keys.read_env(env) == {"GROQ_API_KEY": "xyz"}


# --- masking ---------------------------------------------------------------

def test_mask_never_reveals_the_middle_of_a_secret() -> None:
    masked = env_keys.mask("AIzaSyEXAMPLEsecretVALUE7Xq2")
    assert "EXAMPLEsecret" not in masked
    assert masked.startswith("AIza") and masked.endswith("7Xq2")


def test_mask_hides_a_short_value_entirely() -> None:
    """Too short to reveal any of it without giving away most of the secret."""
    assert "abc" not in env_keys.mask("abcd")


def test_mask_of_empty_is_a_placeholder() -> None:
    assert env_keys.mask("") == "(not set)"


# --- key specs -------------------------------------------------------------

def test_key_specs_cover_the_documented_keys() -> None:
    assert [s.name for s in env_keys.KEY_SPECS] == [
        "GEMINI_API_KEY", "GROQ_API_KEY", "HF_TOKEN",
    ]


def test_every_key_spec_explains_what_it_unlocks_and_where_to_get_it() -> None:
    for spec in env_keys.KEY_SPECS:
        assert spec.unlocks and spec.without and spec.url.startswith("https://")


def test_key_status_reports_set_and_missing(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=AIzaSyEXAMPLEsecretVALUE7Xq2\n", encoding="utf-8")
    status = {s.name: s for s in env_keys.key_status(env, environ={})}
    assert status["GEMINI_API_KEY"].is_set
    assert not status["GROQ_API_KEY"].is_set
    assert "EXAMPLEsecret" not in status["GEMINI_API_KEY"].masked


def test_key_status_sees_a_key_from_the_environment(tmp_path: Path) -> None:
    """Keys may come from the real environment, not just .env."""
    status = {s.name: s for s in
              env_keys.key_status(tmp_path / ".env", environ={"HF_TOKEN": "hf_abc"})}
    assert status["HF_TOKEN"].is_set
    assert status["HF_TOKEN"].source == "environment"


def test_key_status_prefers_dotenv_as_the_reported_source(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=hf_from_file\n", encoding="utf-8")
    status = {s.name: s for s in
              env_keys.key_status(env, environ={"HF_TOKEN": "hf_from_env"})}
    assert status["HF_TOKEN"].source == ".env"


# --- LDS_-prefixed aliases -------------------------------------------------
# `.env.example` documents that every setting can be given with the LDS_ prefix,
# and pydantic-settings reads those *in preference to* the bare name. A status
# check that only looked at the bare name reported a working key as missing, and a
# writer that only wrote the bare name would be silently shadowed by a stale alias.

def test_every_key_has_an_lds_alias() -> None:
    for spec in env_keys.KEY_SPECS:
        assert spec.alias == f"LDS_{spec.name}"


def test_key_status_sees_a_key_stored_under_its_lds_alias(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("LDS_GEMINI_API_KEY=AIzaSyLongEnoughValue\n", encoding="utf-8")
    status = {s.name: s for s in env_keys.key_status(env, environ={})}
    assert status["GEMINI_API_KEY"].is_set
    assert "LDS_GEMINI_API_KEY" in status["GEMINI_API_KEY"].source


def test_alias_in_dotenv_wins_over_bare_name_matching_pydantic_precedence(
    tmp_path: Path,
) -> None:
    """pydantic-settings populates the field from LDS_*, so that is the live value."""
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=bare\nLDS_GEMINI_API_KEY=aliased\n", encoding="utf-8")
    status = {s.name: s for s in env_keys.key_status(env, environ={})}
    assert status["GEMINI_API_KEY"].masked == env_keys.mask("aliased")


def test_set_key_updates_the_alias_when_that_is_the_spelling_in_use(
    tmp_path: Path,
) -> None:
    """Writing the bare name here would leave the stale alias winning at runtime."""
    env = tmp_path / ".env"
    env.write_text("LDS_GEMINI_API_KEY=old\n", encoding="utf-8")
    env_keys.set_managed_key(env, "GEMINI_API_KEY", "new")
    parsed = env_keys.read_env(env)
    assert parsed["LDS_GEMINI_API_KEY"] == "new"
    assert "GEMINI_API_KEY" not in parsed  # no shadowing pair created


def test_set_managed_key_writes_the_bare_name_when_neither_exists(
    tmp_path: Path,
) -> None:
    env = tmp_path / ".env"
    env_keys.set_managed_key(env, "GEMINI_API_KEY", "fresh")
    assert env_keys.read_env(env) == {"GEMINI_API_KEY": "fresh"}


def test_set_managed_key_prefers_the_bare_name_when_both_exist(tmp_path: Path) -> None:
    """Both spellings present is already broken; update the documented one and
    drop the alias so the result is unambiguous."""
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=bare\nLDS_GEMINI_API_KEY=aliased\n", encoding="utf-8")
    env_keys.set_managed_key(env, "GEMINI_API_KEY", "new")
    parsed = env_keys.read_env(env)
    assert parsed == {"GEMINI_API_KEY": "new"}


def test_unset_managed_key_clears_both_spellings(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=a\nLDS_GEMINI_API_KEY=b\nHF_TOKEN=keep\n",
                   encoding="utf-8")
    env_keys.unset_managed_key(env, "GEMINI_API_KEY")
    assert env_keys.read_env(env) == {"HF_TOKEN": "keep"}


def test_settings_reads_an_lds_prefixed_hf_token(tmp_path: Path) -> None:
    """HF_TOKEN had no Settings field, so LDS_HF_TOKEN in .env was silently
    ignored while the other two keys honoured their alias."""
    from studio.config import Settings

    s = Settings(_env_file=None, hf_token="hf_aliased")
    assert s.resolved_key("HF_TOKEN") == "hf_aliased"
