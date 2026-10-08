"""The local engine: what it puts in the Qwen-Image 2.1 graph, and how it fails.

No ComfyUI needed — every `comfy_api` call that would touch the network is stubbed.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from studio import comfy_api, pipeline
from studio.engines.base import GenerationError
from studio.engines.comfyui import ComfyUIEngine
from studio.shotplan import default_plan


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 'reachable' server that has no model lists — and never a real one: a
    developer's own ComfyUI on :8188 would otherwise answer `_combo_options`."""
    monkeypatch.setattr(comfy_api, "server_status", lambda timeout=3.0: (True, "stub"))
    monkeypatch.setattr(comfy_api, "_combo_options", lambda *_: [])


def _src(tmp_path: Path) -> list[Path]:
    src = tmp_path / "ref.png"
    src.write_bytes(b"x")
    return [src]


def test_the_graph_carries_the_shots_prompt_seed_and_reference(
    offline: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict = {}
    monkeypatch.setattr(comfy_api, "upload_image", lambda path: "uploaded.png")

    def run_prompt(graph: dict, timeout: float = 0, **_: object) -> list[dict]:
        seen.update(graph)
        return [{"filename": "o.png"}]

    monkeypatch.setattr(comfy_api, "run_prompt", run_prompt)
    monkeypatch.setattr(comfy_api, "fetch_image", lambda ref, out: out)
    shot = default_plan()[0]

    ComfyUIEngine().generate(_src(tmp_path), shot, tmp_path / "o.png", seed=1234)

    assert seen["1"]["inputs"]["image"] == "uploaded.png"
    assert seen["5"]["inputs"]["prompt"] == shot.local_prompt
    assert seen["6"]["inputs"]["seed"] == 1234
    # Model names come from settings, not the template's hard-coded defaults.
    assert seen["2"]["inputs"]["unet_name"].endswith(".safetensors")


def test_multi_reference_wires_one_loader_per_reference_in_order(
    offline: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from studio.config import settings

    seen: dict = {}
    monkeypatch.setattr(comfy_api, "upload_image", lambda path: f"up-{path.name}")

    def run_prompt(graph: dict, timeout: float = 0, front: bool = False) -> list[dict]:
        seen.update(graph=graph, front=front)
        return [{"filename": "o.png"}]

    monkeypatch.setattr(comfy_api, "run_prompt", run_prompt)
    monkeypatch.setattr(comfy_api, "fetch_image", lambda ref, out: out)
    monkeypatch.setattr(settings, "qwen21_max_refs", 3)
    monkeypatch.setattr(settings, "qwen21_resolution", 1024)
    refs = []
    for name in ("chain", "primary", "extra", "spare"):
        refs.append(tmp_path / f"{name}.png")
        refs[-1].write_bytes(b"x")

    ComfyUIEngine(front=True).generate(refs + [refs[1]], default_plan()[0],
                                       tmp_path / "o.png", seed=1)

    g, enc = seen["graph"], seen["graph"]["5"]["inputs"]
    loaded = [g[enc[f"images.image_{i}"][0]]["inputs"]["image"] for i in (1, 2, 3)]
    assert loaded == ["up-chain.png", "up-primary.png", "up-extra.png"]
    assert "images.image_4" not in enc and enc["resolution"] == 1024
    assert seen["front"] is True


@pytest.mark.parametrize("stage", ["upload_image", "run_prompt", "fetch_image"])
def test_a_dead_comfyui_fails_the_shot_not_the_run(
    stage: str, offline: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ComfyUI dying mid-batch (an OOM restart) raises a raw httpx error from any
    of three calls. It must become a per-shot failure: anything else reaches the
    UI's gr.Error, which throws away the shots that already finished."""
    def boom(*_: object, **__: object) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(comfy_api, "upload_image", lambda path: "uploaded.png")
    monkeypatch.setattr(comfy_api, "run_prompt", lambda *a, **k: [{"filename": "o.png"}])
    monkeypatch.setattr(comfy_api, "fetch_image", lambda ref, out: out)
    monkeypatch.setattr(comfy_api, stage, boom)

    with pytest.raises(GenerationError, match="connection refused"):
        ComfyUIEngine().generate(_src(tmp_path), default_plan()[0], tmp_path / "o.png", seed=1)


def test_generate_shots_keeps_going_when_comfyui_dies(
    offline: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(path: Path) -> str:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(comfy_api, "upload_image", boom)

    results = pipeline.generate_shots(_src(tmp_path), default_plan()[:2], "comfyui",
                                      tmp_path / "out", progress=lambda _: None)

    assert [r.path for r in results] == [None, None]
    assert all("connection refused" in r.error for r in results)
