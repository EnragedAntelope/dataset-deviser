"""`cli.py build` must run the right stages for each dataset type.

The Style case is the one that costs money if it regresses: before Phase 2,
`build --dataset-type style` ran the full 24-shot *character* generation, which
on the cloud engine bills the user's own API key for images they can't use.
Every stage here is a stub — no engine, no captioner, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import cli
from studio import captioner as captioner_mod
from studio import package as package_mod
from studio.preprocess import PreprocessReport

runner = CliRunner()


@pytest.fixture
def stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace every stage with a recorder, so only the orchestration is tested."""
    calls: dict = {"generated": 0, "shots": [], "captioned": [], "metadata": {}}
    src = tmp_path / "ref.png"
    src.write_bytes(b"not-a-real-png")
    calls["src"] = src

    def fake_preprocess(sources, out_dir, **kw):
        calls["isolate"] = kw.get("isolate")
        out_dir.mkdir(parents=True, exist_ok=True)
        prepped = out_dir / "ref_prepped.png"
        prepped.write_bytes(b"x")
        return [PreprocessReport(source=sources[0], output=prepped,
                                 original_size=(1, 1), final_size=(1, 1),
                                 restored=False, reason="ok")]

    def fake_generate(sources, shots, engine, out_dir, **kw):
        calls["generated"] += 1
        calls["shots"] = shots
        calls["exclude_props"] = kw.get("exclude_props")
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for shot in shots:
            path = out_dir / f"{shot.id}.png"
            path.write_bytes(b"x")
            results.append(cli.pipeline.GenResult(shot, path, seed=1))
        return results

    def fake_caption_images(images, *a, **kw):
        calls["captioned"] = list(images)
        return [(p, "caption") for p in images]

    def fake_package(items, output_root, name, trigger, metadata):
        calls["metadata"] = metadata
        ds = Path(output_root) / "ds"
        ds.mkdir(parents=True, exist_ok=True)
        return ds

    monkeypatch.setattr(cli.pipeline, "preprocess_sources", fake_preprocess)
    monkeypatch.setattr(cli.pipeline, "generate_shots", fake_generate)
    monkeypatch.setattr(captioner_mod, "caption_images", fake_caption_images)
    monkeypatch.setattr(package_mod, "package_dataset", fake_package)
    return calls


def _build(stubs: dict, tmp_path: Path, *args: str):
    result = runner.invoke(cli.app, [
        "build", str(stubs["src"]), "--output-root", str(tmp_path / "out"), *args])
    assert result.exit_code == 0, result.output
    return result


def test_style_build_skips_generation_entirely(stubs: dict, tmp_path: Path) -> None:
    result = _build(stubs, tmp_path, "--dataset-type", "style")
    assert stubs["generated"] == 0
    assert "skipping ② generation" in result.output
    # Only the preprocessed sources are captioned and exported.
    assert len(stubs["captioned"]) == 1
    assert stubs["metadata"]["dataset_type"] == "style"
    # No generation happened, so no engine/shot record is invented.
    assert "shots" not in stubs["metadata"]
    assert "engine" not in stubs["metadata"]


def test_style_build_leaves_isolation_off(stubs: dict, tmp_path: Path) -> None:
    _build(stubs, tmp_path, "--dataset-type", "style")
    assert stubs["isolate"] is False


def test_concept_build_uses_the_concept_plan(stubs: dict, tmp_path: Path) -> None:
    _build(stubs, tmp_path, "--dataset-type", "concept")
    assert stubs["generated"] == 1
    assert len(stubs["shots"]) == 18
    assert {s.kind for s in stubs["shots"]} == {"angle", "framing", "context"}
    # The prop-exclusion clause is character-worded: off unless asked for.
    assert stubs["exclude_props"] is False
    assert stubs["isolate"] is True


def test_character_build_is_unchanged(stubs: dict, tmp_path: Path) -> None:
    _build(stubs, tmp_path)
    assert stubs["generated"] == 1
    assert len(stubs["shots"]) == 24
    assert stubs["exclude_props"] is True
    assert stubs["isolate"] is True
    assert stubs["metadata"]["dataset_type"] == "character"
    assert len(stubs["metadata"]["shots"]) == 24


def test_explicit_flags_beat_the_type_defaults(stubs: dict, tmp_path: Path) -> None:
    _build(stubs, tmp_path, "--dataset-type", "concept",
           "--exclude-props", "--no-isolate")
    assert stubs["exclude_props"] is True
    assert stubs["isolate"] is False


def test_generate_command_refuses_style(stubs: dict, tmp_path: Path) -> None:
    result = runner.invoke(cli.app, [
        "generate", str(stubs["src"]), "--dataset-type", "style"])
    assert result.exit_code != 0
    assert "no synthetic generation" in result.output
    assert stubs["generated"] == 0


def test_generate_command_builds_a_concept_set(stubs: dict, tmp_path: Path) -> None:
    result = runner.invoke(cli.app, [
        "generate", str(stubs["src"]), "--dataset-type", "concept",
        "--out", str(tmp_path / "gen")])
    assert result.exit_code == 0, result.output
    assert len(stubs["shots"]) == 18
