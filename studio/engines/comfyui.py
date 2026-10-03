"""Fully local engine: Qwen-Image 2.1 (unified generate + edit) via ComfyUI."""

from __future__ import annotations

from pathlib import Path

import httpx

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
        # One reference per shot; use the primary (first) source. Qwen-Image 2.1
        # accepts up to 10, but that is untested here (see ARCHITECTURE.md).
        graph = comfy_api.load_template("qwen21_edit")
        graph["5"]["inputs"]["prompt"] = shot.local_prompt
        graph["6"]["inputs"]["seed"] = seed

        last_err: Exception | None = None
        for _ in range(2):
            try:
                # Upload inside the try: ComfyUI dying mid-batch (OOM restart)
                # surfaces as a raw httpx error from here, from /history polling
                # or from /view — all of which must fail ONE shot, not the run.
                graph["1"]["inputs"]["image"] = self._source_name(sources[0])
                refs = comfy_api.run_prompt(graph, timeout=600)
                return comfy_api.fetch_image(refs[0], out_path)
            except (comfy_api.ComfyError, httpx.HTTPError) as e:
                last_err = e
                graph["6"]["inputs"]["seed"] = seed + 1
        raise GenerationError(f"shot {shot.id}: {last_err}")
