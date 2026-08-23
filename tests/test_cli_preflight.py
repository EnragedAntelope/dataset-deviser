"""`cli.py build` must decide it can't generate BEFORE it preprocesses.

`build` runs preprocess → generate, and the ComfyUI engine only tests
reachability inside its own constructor — which `build` doesn't reach until
preprocess has finished. On a GPU-bound restore/isolate pass over a folder of
sources, that is a long wait paid for output the run then can't use.

These assert the ordering, not just the message: `preprocess_sources` is stubbed
to record that it was called, and the point of every abort case is that it wasn't.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import cli
from studio import comfy_api, doctor

runner = CliRunner()


@pytest.fixture
def src(tmp_path: Path) -> Path:
    p = tmp_path / "ref.png"
    p.write_bytes(b"not-a-real-png")
    return p


@pytest.fixture
def touched(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Record whether the expensive stage ran at all."""
    seen = {"preprocess": 0}

    def _fake(*a: object, **k: object):
        seen["preprocess"] += 1
        raise AssertionError("preprocess ran despite a failed preflight")

    monkeypatch.setattr(cli.pipeline, "preprocess_sources", _fake)
    return seen


def _down(*_a: object, **_k: object) -> tuple[bool, str]:
    return False, "nothing is listening at http://127.0.0.1:8188"


def _up(*_a: object, **_k: object) -> tuple[bool, str]:
    return True, "ok"


def test_unreachable_comfyui_aborts_before_preprocess(
    src: Path, touched: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(comfy_api, "server_status", _down)

    result = runner.invoke(cli.app, ["build", str(src), "--engine", "comfyui",
                                     "--output-root", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert touched["preprocess"] == 0
    assert "not reachable" in result.output
    assert "Nothing was preprocessed" in result.output
    # The message has to name a way out, not just the failure.
    assert "LDS_COMFY_URL" in result.output
    assert "--engine gemini" in result.output


def test_the_cloud_engine_never_touches_comfyui(
    src: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A preflight that phones a server the run doesn't need is a new hang."""
    def _boom(*_a: object, **_k: object):
        raise AssertionError("server_status was called for the cloud engine")

    monkeypatch.setattr(comfy_api, "server_status", _boom)
    cli._preflight_comfyui("gemini")  # no exception = pass


def test_a_style_build_skips_the_preflight(
    src: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Style never generates, so an unreachable ComfyUI must not stop it."""
    calls = {"n": 0}

    def _counted(*a: object, **k: object) -> tuple[bool, str]:
        calls["n"] += 1
        return False, "down"

    monkeypatch.setattr(comfy_api, "server_status", _counted)
    monkeypatch.setattr(cli.pipeline, "preprocess_sources",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("far enough")))

    result = runner.invoke(cli.app, ["build", str(src), "--engine", "comfyui",
                                     "--dataset-type", "style",
                                     "--output-root", str(tmp_path / "out")])

    assert calls["n"] == 0, "a style build asked about a server it never uses"
    assert result.exit_code != 0  # it got past the preflight and hit our stub


def test_a_model_name_mismatch_warns_but_does_not_abort(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """`check_comfyui_models` covers every bundled template, including ones this
    run may never submit — so it must not block a run that would have worked."""
    monkeypatch.setattr(comfy_api, "server_status", _up)
    monkeypatch.setattr(doctor, "check_comfyui_models",
                        lambda: doctor.Check("ComfyUI models", True,
                                             "qwen_edit: no such vae_name", warn=True))

    cli._preflight_comfyui("comfyui")  # must not raise

    err = capsys.readouterr().err
    assert "no such vae_name" in err
    assert "docs/comfyui-setup.md" in err


def test_a_clean_model_check_says_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(comfy_api, "server_status", _up)
    monkeypatch.setattr(doctor, "check_comfyui_models",
                        lambda: doctor.Check("ComfyUI models", True, "all found"))

    cli._preflight_comfyui("comfyui")

    assert capsys.readouterr().err == ""
