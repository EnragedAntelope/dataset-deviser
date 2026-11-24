"""Fully local engine via ComfyUI: Fibo 2.0 Edit Turbo (default) or Qwen-Image 2.1."""

from __future__ import annotations

import copy
from pathlib import Path

import httpx

from studio import comfy_api
from studio.config import settings
from studio.engines.base import GenerationError
from studio.shotplan import Shot

# TextEncodeQwenImage21 / TextEncodeFibo2's `images` socket grows to this many slots.
MAX_REFS = 16
# Engine key -> (bundled template, name the user sees).
LOCAL_MODELS = {
    "comfyui": ("fibo2_edit", "Fibo 2.0 Edit Turbo"),
    "comfyui-qwen21": ("qwen21_edit", "Qwen-Image 2.1"),
}
# Fibo above 4 MP is untested; under 0.5 MP the faces are mush.
FIBO2_MP_RANGE = (0.5, 4.0)


class ComfyUIEngine:
    name = "comfyui"

    def __init__(self, front: bool = False, model: str = "comfyui") -> None:
        if model not in LOCAL_MODELS:
            raise GenerationError(f"Unknown local model '{model}' - use one of "
                                  f"{', '.join(LOCAL_MODELS)}.")
        self.name = model
        self.template, label = LOCAL_MODELS[model]
        up, reason = comfy_api.server_status()
        if not up:
            raise GenerationError(
                f"ComfyUI is not reachable — {reason}. The local engine needs it "
                f"(see docs/comfyui-setup.md), or switch the engine to Cloud (Gemini)."
            )
        try:
            # Once, before any shot: a missing model file fails here with its setting
            # named, not once per shot.
            self._graph = comfy_api.load_template(self.template)
        except comfy_api.ComfyError as e:
            other = next(k for k in LOCAL_MODELS if k != model)
            raise GenerationError(
                f"{label}: {e} Or pick the other local model ({LOCAL_MODELS[other][1]}, "
                f"engine '{other}') if that is the one you have installed.") from e
        if self.template == "fibo2_edit":
            lo, hi = FIBO2_MP_RANGE
            self._graph["10"]["inputs"]["value"] = min(max(settings.fibo2_megapixels, lo), hi)
            self._graph["11"]["inputs"]["attention"] = settings.fibo2_attention or (
                "comfy kitchen attention" if comfy_api.device_type() == "cuda"
                else "pytorch attention")
        else:
            self._graph["5"]["inputs"]["resolution"] = settings.qwen21_resolution
        self.front = front
        self._uploaded: dict[Path, str] = {}

    def _source_name(self, source: Path) -> str:
        if source not in self._uploaded:
            self._uploaded[source] = comfy_api.upload_image(source)
        return self._uploaded[source]

    def generate(self, sources: list[Path], shot: Shot, out_path: Path, seed: int) -> Path:
        # The first reference sets the output's aspect, so callers order them:
        # chained view, primary, extras. LDS_LOCAL_MAX_REFS caps how many go.
        refs = list(dict.fromkeys(sources))[:max(1, min(settings.local_max_refs, MAX_REFS))]
        graph = copy.deepcopy(self._graph)
        graph["5"]["inputs"]["prompt"] = shot.local_prompt
        graph["6"]["inputs"]["seed"] = seed
        for i in range(2, len(refs) + 1):
            graph[str(100 + i)] = {"class_type": "LoadImage", "inputs": {}}
            graph["5"]["inputs"][f"images.image_{i}"] = [str(100 + i), 0]
        loaders = ["1"] + [str(100 + i) for i in range(2, len(refs) + 1)]

        last_err: Exception | None = None
        for _ in range(2):
            try:
                # Upload inside the try: ComfyUI dying mid-batch (OOM restart)
                # surfaces as a raw httpx error from here, from /history polling
                # or from /view — all of which must fail ONE shot, not the run.
                for node, ref in zip(loaders, refs, strict=True):
                    graph[node]["inputs"]["image"] = self._source_name(ref)
                images = comfy_api.run_prompt(graph, timeout=600, front=self.front)
                return comfy_api.fetch_image(images[0], out_path)
            except (comfy_api.ComfyError, httpx.HTTPError) as e:
                last_err = e
                graph["6"]["inputs"]["seed"] = seed + 1
        raise GenerationError(f"shot {shot.id}: {last_err}")
