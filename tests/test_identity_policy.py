"""0.19.0: the character identity policy — what the trigger owns.

"identity" (default) varies outfits and captions the clothing, so the trigger
learns the person; "costume" keeps the reference's outfit and leaves it out of
captions, so the trigger carries it.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

import app as A
import cli
from studio.config import CAPTIONERS_BY_KEY

runner = CliRunner()


def test_caption_instruction_follows_the_policy() -> None:
    spec = CAPTIONERS_BY_KEY["qwen3vl"]
    assert spec.prompt_for("prose", identity="") == spec.prompt_template
    assert spec.prompt_for("prose", identity="identity").endswith(
        "Describe the clothing and the facial expression.")
    assert "Do not tag the clothing" in spec.prompt_for("e621", identity="costume")
    # Style/Concept captions never mention a costume policy.
    assert "clothing" not in spec.prompt_for("prose", "concept", identity="identity")


def test_the_plan_is_dressed_unless_the_outfit_is_the_character() -> None:
    dressed = A._plan_table("character")
    worn = {k: [bool(o) for o in g] for k, g in dressed.groupby("kind")["outfit"]}
    assert all(worn["angle"] + worn["pose"]) and not any(worn["emotion"])
    assert not any(A._plan_table("character", identity="costume")["outfit"])
    assert not any(A._plan_table("concept")["outfit"])
    plain, _note = A.on_identity_change(dressed, "costume", "character")
    assert not any(plain["outfit"])


def test_identity_policy_reaches_export_metadata(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (64, 64)).save(src / "a.png")
    (src / "a.txt").write_text("a caption", encoding="utf-8")
    _r, ds_dir, _hf = A.do_export([str(src / "a.png")], "Sy", "trg", str(tmp_path / "out"),
                                  identity="costume")
    meta = json.loads((Path(ds_dir) / "metadata.json").read_text(encoding="utf-8"))
    assert meta["identity"] == "costume"


def test_cli_generate_dresses_by_default(monkeypatch, tmp_path: Path) -> None:
    seen: dict = {}

    def fake_generate(sources, shots, engine, out_dir, **kw):
        seen["outfits"] = [s.outfit for s in shots if s.kind == "angle"]
        return []

    monkeypatch.setattr(cli.pipeline, "generate_shots", fake_generate)
    ref = tmp_path / "r.png"
    Image.new("RGB", (8, 8)).save(ref)
    runner.invoke(cli.app, ["generate", str(ref), "--out", str(tmp_path / "g")])
    assert all(seen["outfits"])
    runner.invoke(cli.app, ["generate", str(ref), "--out", str(tmp_path / "g"),
                            "--identity", "costume"])
    assert not any(seen["outfits"])
    bad = runner.invoke(cli.app, ["generate", str(ref), "--identity", "nope"])
    assert bad.exit_code != 0
