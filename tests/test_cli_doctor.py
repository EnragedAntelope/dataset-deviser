"""`cli.py doctor` / `cli.py keys` — the returning-user diagnostic.

These exist because a new user hit a setup crash whose only symptom was a window
closing. `doctor` has to state plainly what is configured, what is missing, and what
each missing piece blocks — without ever printing a whole secret.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from studio import doctor as D
from studio import env_keys


# --- python version check --------------------------------------------------

def test_python_check_accepts_a_supported_version() -> None:
    check = D.check_python((3, 12, 1))
    assert check.ok and not check.warn


def test_python_check_rejects_too_old() -> None:
    check = D.check_python((3, 9, 18))
    assert not check.ok
    assert "3.10" in check.detail


def test_python_check_warns_above_the_tested_range() -> None:
    """Newer Python is allowed but flagged — wheels may not exist yet."""
    check = D.check_python((3, 99, 0))
    assert check.ok and check.warn


# --- dependency check ------------------------------------------------------

def test_dependency_check_reports_missing_modules() -> None:
    check = D.check_dependencies(importer=lambda name: name != "gradio")
    assert not check.ok
    assert "gradio" in check.detail
    assert "setup" in check.detail.lower()  # tells the user how to fix it


def test_dependency_check_passes_when_all_present() -> None:
    assert D.check_dependencies(importer=lambda name: True).ok


def test_dependency_check_covers_the_imports_app_needs_at_startup() -> None:
    """A dep added to requirements.txt but not here would reintroduce the
    "start.bat did nothing" failure after a git pull."""
    assert {"gradio", "dotenv", "pydantic_settings", "pandas", "typer"} <= set(
        D.REQUIRED_IMPORTS)


# --- optional tagger backend ----------------------------------------------

def test_onnxruntime_is_optional_not_a_failure() -> None:
    check = D.check_onnxruntime(importer=lambda name: False)
    assert check.ok and check.warn  # advisory: only the taggers need it


# --- key reporting ---------------------------------------------------------

def test_key_report_names_what_a_missing_key_blocks(tmp_path: Path) -> None:
    lines = D.key_report(tmp_path / ".env", environ={})
    body = "\n".join(lines)
    for spec in env_keys.KEY_SPECS:
        assert spec.name in body
    assert "unavailable" in body


def test_key_report_never_prints_a_whole_secret(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    secret = "AIzaSyEXAMPLEsecretVALUE7Xq2"
    env.write_text(f"GEMINI_API_KEY={secret}\n", encoding="utf-8")
    body = "\n".join(D.key_report(env, environ={}))
    assert secret not in body
    assert "EXAMPLEsecret" not in body


def test_key_report_marks_a_configured_key_as_set(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=gsk_averylongvaluehere\n", encoding="utf-8")
    body = "\n".join(D.key_report(env, environ={}))
    assert "GROQ_API_KEY" in body and "set" in body


# --- overall report --------------------------------------------------------

def test_report_is_not_ok_when_a_dependency_is_missing(tmp_path: Path) -> None:
    report = D.build_report(env_path=tmp_path / ".env", environ={},
                            importer=lambda name: name != "gradio",
                            version=(3, 12, 0), check_comfy=False)
    assert not report.ok


def test_report_is_ok_with_no_keys_at_all(tmp_path: Path) -> None:
    """Missing keys are not failures — the whole app runs fully local."""
    report = D.build_report(env_path=tmp_path / ".env", environ={},
                            importer=lambda name: True,
                            version=(3, 12, 0), check_comfy=False)
    assert report.ok


def test_report_renders_without_raising(tmp_path: Path) -> None:
    report = D.build_report(env_path=tmp_path / ".env", environ={},
                            importer=lambda name: True,
                            version=(3, 12, 0), check_comfy=False)
    text = D.render(report)
    assert "LoRA Dataset Studio" in text
    assert "cli.py keys" in text  # points at the fix


@pytest.mark.parametrize("codepage", ["cp1252", "cp437", "ascii"])
def test_the_whole_report_is_encodable_on_a_legacy_windows_codepage(
    tmp_path: Path, codepage: str
) -> None:
    """setup.bat prints this in a plain cmd window, and users redirect it to a file.
    Where the ANSI code page is cp1252/cp437, a "✓" would raise UnicodeEncodeError —
    a diagnostic that crashes while diagnosing is worse than a plain one."""
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=AIzaSyEXAMPLEsecretVALUE7Xq2\n", encoding="utf-8")
    for importer in (lambda name: True, lambda name: False):
        report = D.build_report(env_path=env, environ={}, importer=importer,
                                version=(3, 12, 0), check_comfy=False)
        D.render(report).encode(codepage)  # raises if any glyph is unencodable


def test_prompt_secret_hides_input_when_a_terminal_is_present(monkeypatch) -> None:
    import studio.doctor as mod

    monkeypatch.setattr(mod.sys, "stdin", type("S", (), {"isatty": lambda self: True})())
    monkeypatch.setattr("getpass.getpass", lambda label: "typed-secret")
    assert mod.prompt_secret("k: ") == "typed-secret"


def test_prompt_secret_falls_back_instead_of_hanging_without_a_terminal(
    monkeypatch, capsys
) -> None:
    """getpass on Windows reads the console device, not stdin — with stdin
    redirected it blocks forever on a keypress nobody can send."""
    import studio.doctor as mod

    monkeypatch.setattr(mod.sys, "stdin", type("S", (), {"isatty": lambda self: False})())
    monkeypatch.setattr("builtins.input", lambda: "piped-secret")
    monkeypatch.setattr("getpass.getpass",
                        lambda label: pytest.fail("getpass would hang here"))
    assert mod.prompt_secret("k: ") == "piped-secret"
    assert "VISIBLE" in capsys.readouterr().out  # says so rather than pretending


def test_prompt_secret_survives_a_stdin_that_cannot_be_queried(monkeypatch) -> None:
    import studio.doctor as mod

    class Broken:
        def isatty(self):
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(mod.sys, "stdin", Broken())
    monkeypatch.setattr("builtins.input", lambda: "fallback")
    assert mod.prompt_secret("k: ") == "fallback"


def test_check_symbols_are_ascii() -> None:
    for check in (D.Check("x", ok=True), D.Check("x", ok=True, warn=True),
                  D.Check("x", ok=False)):
        check.symbol.encode("ascii")


# --- keys command plumbing ------------------------------------------------

def test_prompt_for_key_keeps_the_existing_value_on_empty_input(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=original\n", encoding="utf-8")
    changed = D.apply_key_input(env, "GEMINI_API_KEY", "")
    assert not changed
    assert env_keys.read_env(env)["GEMINI_API_KEY"] == "original"


def test_prompt_for_key_replaces_on_new_input(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=original\n", encoding="utf-8")
    changed = D.apply_key_input(env, "GEMINI_API_KEY", "replacement")
    assert changed
    assert env_keys.read_env(env)["GEMINI_API_KEY"] == "replacement"


def test_apply_key_input_can_clear_a_key(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=original\n", encoding="utf-8")
    assert D.apply_key_input(env, "GEMINI_API_KEY", "-")
    assert "GEMINI_API_KEY" not in env_keys.read_env(env)


def test_apply_key_input_rejects_a_pasted_placeholder(tmp_path: Path) -> None:
    """Users paste the example text from a docs page; catch the obvious ones."""
    env = tmp_path / ".env"
    with pytest.raises(ValueError):
        D.apply_key_input(env, "GEMINI_API_KEY", "YOUR_API_KEY_HERE")


def test_apply_key_input_strips_wrapping_quotes_users_paste(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    D.apply_key_input(env, "GEMINI_API_KEY", '"AIzaSyLongEnoughValue"')
    assert env_keys.read_env(env)["GEMINI_API_KEY"] == "AIzaSyLongEnoughValue"
