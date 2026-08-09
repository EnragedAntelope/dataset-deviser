"""Fully local engine: Qwen Image Edit 2511 (+ Multiple-Angles LoRA) via ComfyUI."""

from __future__ import annotations

from pathlib import Path

from studio import comfy_api
from studio.engines.base import GenerationError
from studio.shotplan import Shot


class ComfyUIEngine:
    name = "comfyui"

    def __init__(self) -> None:
        up, reason = comfy_api.server_status()
        if not up:
            raise GenerationError(
                f"ComfyUI is not reachable — {reason}. The local engine needs it "
                f"(see docs/comfyui-setup.md), or switch the engine to Cloud (Gemini)."
            )
        self._uploaded: dict[Path, str] = {}

    def _source_name(self, source: Path) -> str:
        if source not in self._uploaded:
            self._uploaded[source] = comfy_api.upload_image(source)
        return self._uploaded[source]

    def generate(self, sources: list[Path], shot: Shot, out_path: Path, seed: int) -> Path:
        # Qwen Edit works from one reference; use the primary (first) source.
        graph = comfy_api.load_template("qwen_edit")
        graph["1"]["inputs"]["image"] = self._source_name(sources[0])
        graph["9"]["inputs"]["prompt"] = shot.local_prompt
        # The Multiple-Angles LoRA is trigger-based (<sks>); zero it out for
        # plain pose/scene edits so it cannot bias them. 0.9 per fal's tips.
        graph["8"]["inputs"]["strength_model"] = 0.9 if shot.kind == "angle" else 0.0
        graph["14"]["inputs"]["seed"] = seed

        last_err: Exception | None = None
        for _ in range(2):
            try:
                refs = comfy_api.run_prompt(graph, timeout=600)
                return comfy_api.fetch_image(refs[0], out_path)
            except comfy_api.ComfyError as e:
                last_err = e
                graph["14"]["inputs"]["seed"] = seed + 1
        raise GenerationError(f"shot {shot.id}: {last_err}")
