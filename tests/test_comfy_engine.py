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
    monkeypatch.setattr(comfy_api, "device_type", lambda: "cuda")


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
    monkeypatch.setattr(settings, "local_max_refs", 3)
    monkeypatch.setattr(settings, "qwen21_resolution", 1024)
    refs = []
    for name in ("chain", "primary", "extra", "spare"):
        refs.append(tmp_path / f"{name}.png")
        refs[-1].write_bytes(b"x")

    ComfyUIEngine(front=True, model="comfyui-qwen21").generate(
        refs + [refs[1]], default_plan()[0], tmp_path / "o.png", seed=1)

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


def test_a_timeout_is_not_retried(
    offline: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A timed-out job may still be running in ComfyUI; retrying queues it twice."""
    calls = []

    def slow(*_: object, **__: object) -> None:
        calls.append(1)
        raise comfy_api.ComfyTimeout("ComfyUI did not finish within 600s")

    monkeypatch.setattr(comfy_api, "upload_image", lambda path: "uploaded.png")
    monkeypatch.setattr(comfy_api, "run_prompt", slow)

    with pytest.raises(GenerationError, match="did not finish"):
        ComfyUIEngine().generate(_src(tmp_path), default_plan()[0], tmp_path / "o.png", seed=1)
    assert len(calls) == 1


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


def test_fibo2_is_the_default_local_model_and_gets_its_own_settings(
    offline: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from studio.config import settings

    seen: dict = {}
    monkeypatch.setattr(comfy_api, "upload_image", lambda path: "uploaded.png")
    monkeypatch.setattr(comfy_api, "run_prompt",
                        lambda graph, **_: seen.update(graph) or [{"filename": "o.png"}])
    monkeypatch.setattr(comfy_api, "fetch_image", lambda ref, out: out)
    monkeypatch.setattr(settings, "fibo2_megapixels", 9.0)  # over the cap
    shot = default_plan()[0]

    ComfyUIEngine().generate(_src(tmp_path), shot, tmp_path / "o.png", seed=7)

    assert seen["5"]["class_type"] == "TextEncodeFibo2"
    assert seen["5"]["inputs"]["prompt"] == shot.local_prompt
    assert "resolution" not in seen["5"]["inputs"]
    assert seen["2"]["inputs"]["unet_name"] == settings.fibo2_model
    assert seen["3"]["inputs"]["clip_name"] == settings.fibo2_text_encoder
    assert seen["10"]["inputs"]["value"] == 4.0
    assert seen["11"]["inputs"]["attention"] == "comfy kitchen attention"


def test_fibo2_uses_pytorch_attention_off_cuda(
    offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(comfy_api, "device_type", lambda: "mps")
    assert ComfyUIEngine()._graph["11"]["inputs"]["attention"] == "pytorch attention"


def test_a_missing_model_fails_before_any_shot_and_names_the_other_model(
    offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(name: str) -> dict:
        raise comfy_api.ComfyError("ComfyUI has no unet_name 'x'. Set LDS_FIBO2_MODEL.")

    monkeypatch.setattr(comfy_api, "load_template", missing)
    with pytest.raises(GenerationError, match="LDS_FIBO2_MODEL.*comfyui-qwen21"):
        ComfyUIEngine()


def test_unknown_engine_keys_are_refused(offline: None) -> None:
    with pytest.raises(GenerationError, match="comfyui-qwen21"):
        pipeline.make_engine("comfyui-flux")
    with pytest.raises(GenerationError, match="Unknown engine"):
        pipeline.make_engine("gemeni")


def test_the_old_max_refs_name_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.config import Settings

    monkeypatch.setenv("LDS_QWEN21_MAX_REFS", "3")
    assert Settings().local_max_refs == 3
