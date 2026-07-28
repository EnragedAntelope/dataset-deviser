"""Thin client for ComfyUI's HTTP API (upload, queue, poll, fetch, free)."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx

from studio.config import settings

WORKFLOWS_DIR = Path(__file__).resolve().parent / "comfy_workflows"


class ComfyError(Exception):
    pass


# Model filenames inside the bundled templates, remapped to whatever the user
# configured in .env (LDS_QWEN_EDIT_MODEL etc.) so renamed files just work.
_MODEL_INPUTS = {
    "unet_name": "qwen_edit_model",
    "lora_name": "angles_lora",
    "ckpt_name": "sam3_checkpoint",
}
_UPSCALE_SETTINGS = ("dejpg_model", "upscale_model")  # in template node order


_combo_cache: dict[tuple[str, str], list[str]] = {}


def _combo_options(class_type: str, input_key: str) -> list[str]:
    """The server's accepted values for one combo input, cached per process.

    Only successful lookups are cached: a failed one used to be stored as an
    empty list, so a single timed-out call (ComfyUI stalls this endpoint while
    it is busy) permanently disabled filename matching for the rest of the run.
    """
    key = (class_type, input_key)
    if _combo_cache.get(key):
        return _combo_cache[key]
    try:
        info = httpx.get(f"{settings.comfy_url}/object_info/{class_type}",
                         timeout=30).json()
        options = info[class_type]["input"]["required"][input_key][0]
    except Exception:
        return []
    if isinstance(options, list):
        _combo_cache[key] = options
        return options
    return []


def _server_filename(class_type: str, input_key: str, value: str,
                     setting_attr: str) -> str:
    """Match `value` against the server's file list ignoring path separators —
    ComfyUI enumerates subfolder files with the server OS's separator, and a
    graph value must match that list exactly.

    Raises if the server has a list and `value` is not in it: ComfyUI's own
    rejection ("Value not in list: ... not in (list of length 10574)") never
    says which setting was wrong or what is installed.
    """
    options = _combo_options(class_type, input_key)
    if not options:  # server unreachable / unknown node — send it as configured
        return value
    norm = value.replace("\\", "/")
    for opt in options:
        if opt.replace("\\", "/") == norm:
            return opt
    for opt in options:  # Windows paths are case-insensitive, ComfyUI's list is not
        if opt.replace("\\", "/").casefold() == norm.casefold():
            return opt
    import difflib

    close = difflib.get_close_matches(norm, [o.replace("\\", "/") for o in options],
                                      n=3, cutoff=0.5)
    hint = f" Closest installed: {', '.join(close)}." if close else ""
    raise ComfyError(
        f"ComfyUI has no {input_key} '{value}' for {class_type} "
        f"({len(options)} installed).{hint} Set LDS_{setting_attr.upper()} in .env to "
        f"a filename ComfyUI lists, or install the model into the matching folder."
    )


def load_template(name: str) -> dict:
    graph = json.loads((WORKFLOWS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    upscale_iter = iter(_UPSCALE_SETTINGS)
    for node in graph.values():
        for input_key, setting_attr in _MODEL_INPUTS.items():
            if input_key in node.get("inputs", {}):
                node["inputs"][input_key] = _server_filename(
                    node["class_type"], input_key, getattr(settings, setting_attr),
                    setting_attr)
        if node.get("class_type") == "UpscaleModelLoader":
            setting_attr = next(upscale_iter)
            node["inputs"]["model_name"] = _server_filename(
                "UpscaleModelLoader", "model_name", getattr(settings, setting_attr),
                setting_attr)
    return graph


def is_up(timeout: float = 3.0) -> bool:
    try:
        httpx.get(f"{settings.comfy_url}/system_stats", timeout=timeout).raise_for_status()
        return True
    except Exception:
        return False


def upload_image(path: Path) -> str:
    """Upload into ComfyUI's input folder; returns the stored filename."""
    name = f"lds_{uuid.uuid4().hex[:10]}{path.suffix.lower()}"
    with path.open("rb") as f:
        r = httpx.post(
            f"{settings.comfy_url}/upload/image",
            files={"image": (name, f, "image/png")},
            data={"overwrite": "true"},
            timeout=60,
        )
    r.raise_for_status()
    return r.json()["name"]


def queue_backlog() -> int:
    """Number of pending prompts other work has queued in ComfyUI."""
    try:
        q = httpx.get(f"{settings.comfy_url}/queue", timeout=10).json()
        return len(q.get("queue_pending", []))
    except Exception:
        return 0


def run_prompt(graph: dict, timeout: float = 600.0, front: bool = False) -> list[dict]:
    """Queue an API-format graph, wait for completion, return output image refs.

    `front=True` asks ComfyUI to put this job at the head of the pending queue
    (its /prompt endpoint negates the job's priority number). It does NOT
    interrupt whatever is already running, so the caller still waits out the
    in-flight job. Polling is by our own prompt_id either way, so other people's
    jobs are never mistaken for ours.
    """
    backlog = queue_backlog()
    if backlog > 10 and not front:
        raise ComfyError(
            f"ComfyUI queue is busy: {backlog} jobs already pending. This app's jobs "
            f"would wait behind them. Either enable 'Prioritize this app's ComfyUI jobs' "
            f"to jump the pending queue, or open ComfyUI ({settings.comfy_url}) and clear "
            f"it (Queue panel → Clear, or the Manager's 'Clear Queue'), then retry. To "
            f"generate without ComfyUI, switch the engine to the Cloud (Gemini) option, "
            f"or set the isolation/restore backend to Built-in/Basic."
        )
    r = httpx.post(f"{settings.comfy_url}/prompt",
                   json={"prompt": graph, "front": front}, timeout=30)
    if r.status_code != 200:
        raise ComfyError(f"queue rejected: {r.text[:500]}")
    prompt_id = r.json()["prompt_id"]

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        h = httpx.get(f"{settings.comfy_url}/history/{prompt_id}", timeout=30).json()
        if prompt_id in h:
            entry = h[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [
                    m[1].get("exception_message", "")
                    for m in status.get("messages", [])
                    if m[0] == "execution_error"
                ]
                raise ComfyError(f"execution error: {'; '.join(msgs) or 'unknown'}")
            images = []
            for node_output in entry.get("outputs", {}).values():
                images.extend(node_output.get("images", []))
            if images:
                return images
            if status.get("completed"):
                raise ComfyError("run completed but produced no images")
        time.sleep(1.5)
    raise ComfyError(f"timed out after {timeout}s waiting for prompt {prompt_id}")


def fetch_image(ref: dict, out_path: Path) -> Path:
    r = httpx.get(
        f"{settings.comfy_url}/view",
        params={
            "filename": ref["filename"],
            "subfolder": ref.get("subfolder", ""),
            "type": ref.get("type", "output"),
        },
        timeout=120,
    )
    r.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)
    return out_path


def free_vram() -> None:
    """Ask ComfyUI to unload models + free memory (before loading the captioner)."""
    try:
        httpx.post(
            f"{settings.comfy_url}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=30,
        )
    except Exception:
        pass  # best effort; captioner load will fail loudly if VRAM is truly short
