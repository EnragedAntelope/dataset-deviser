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
    return server_status(timeout)[0]


def server_status(timeout: float = 3.0) -> tuple[bool, str]:
    """(reachable, reason). The reason is a plain sentence for the user.

    "ComfyUI is not reachable" on its own sends people to restart a server that
    is already running — the real cause is usually a different port, a machine
    that isn't this one, or a proxy. Naming *which* failure it was, and which
    URL was tried, turns a guess into one obvious fix.
    """
    url = settings.comfy_url
    try:
        r = httpx.get(f"{url}/system_stats", timeout=timeout)
    except httpx.ConnectError:
        return False, (f"nothing is listening at {url} — start ComfyUI, or set "
                       f"LDS_COMFY_URL in .env if it runs on another port/machine")
    except httpx.TimeoutException:
        return False, (f"{url} did not answer within {timeout:g}s — ComfyUI may be "
                       f"starting up, or busy loading a model")
    except Exception as e:
        return False, f"could not reach {url}: {type(e).__name__}: {e}"
    if r.status_code != 200:
        return False, (f"{url} answered HTTP {r.status_code} — that address is "
                       f"reachable but is not a ComfyUI server (a proxy or another "
                       f"app may hold the port)")
    return True, f"reachable at {url}"


# ComfyUI reports a graph it cannot run in three shapes; all of them arrive as a
# 400 from /prompt. Verified against a live ComfyUI (v0.29) rather than guessed:
#   missing_node_type            -> error.extra_info.class_type is the node
#   prompt_outputs_failed_validation -> node_errors{id: {class_type, errors[]}}
#   anything else (prompt_no_outputs, invalid_prompt, ...) -> error.message
# The old code printed `r.text[:500]`, i.e. raw truncated JSON.
def describe_rejection(body: str) -> str:
    """Translate ComfyUI's /prompt rejection body into one readable message."""
    try:
        data = json.loads(body)
    except Exception:
        # Not JSON at all (an HTML error page from a proxy, say) — show a slice
        # rather than pretend we understood it.
        return f"ComfyUI rejected the job: {body[:300]}"
    if not isinstance(data, dict):
        return f"ComfyUI rejected the job: {body[:300]}"
    err = data.get("error") or {}
    etype = err.get("type", "")
    if etype == "missing_node_type":
        info = err.get("extra_info") or {}
        node = info.get("class_type") or "?"
        return (
            f"ComfyUI has no node called '{node}', so this workflow cannot run. "
            f"The bundled workflows use ONLY core ComfyUI nodes, so this almost "
            f"always means your ComfyUI is older than the workflow — update it "
            f"(Manager → Update ComfyUI, or `git pull` in your ComfyUI folder) and "
            f"restart it. If you edited the bundled workflow JSON yourself, install "
            f"the node pack that provides '{node}'."
        )
    parts: list[str] = []
    for node_id, node_err in (data.get("node_errors") or {}).items():
        cls = (node_err or {}).get("class_type", "?")
        for e in (node_err or {}).get("errors") or []:
            detail = e.get("details") or e.get("message") or ""
            field = (e.get("extra_info") or {}).get("input_name") or ""
            where = f"{cls} (node {node_id})"
            if field:
                where += f" input '{field}'"
            parts.append(f"{where}: {detail}")
    headline = err.get("message") or "ComfyUI rejected the job"
    if parts:
        return f"{headline} — " + "; ".join(parts)
    details = err.get("details") or ""
    return f"{headline} — {details}" if details else headline


def missing_node_types(graph: dict) -> list[str]:
    """Node classes in `graph` that this ComfyUI does not have installed.

    A preflight check: without it the first sign of an out-of-date ComfyUI is a
    rejection *after* the reference images have been uploaded. Returns [] when
    the server can't be asked, so an unreachable/odd server never blocks a run
    that might otherwise have worked.
    """
    try:
        installed = httpx.get(f"{settings.comfy_url}/object_info", timeout=30).json()
    except Exception:
        return []
    if not isinstance(installed, dict) or not installed:
        return []
    wanted = {node.get("class_type") for node in graph.values()
              if isinstance(node, dict) and node.get("class_type")}
    return sorted(c for c in wanted if c not in installed)


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
    missing = missing_node_types(graph)
    if missing:
        # Fail before uploading/queueing anything: the message is the same one
        # ComfyUI would eventually give, minus the wasted round trip.
        raise ComfyError(
            f"ComfyUI does not have these node(s) installed: {', '.join(missing)}. "
            f"The bundled workflows use ONLY core ComfyUI nodes, so this almost "
            f"always means your ComfyUI is out of date — update it (Manager → "
            f"Update ComfyUI, or `git pull` in your ComfyUI folder) and restart it."
        )
    r = httpx.post(f"{settings.comfy_url}/prompt",
                   json={"prompt": graph, "front": front}, timeout=30)
    if r.status_code != 200:
        raise ComfyError(describe_rejection(r.text))
    prompt_id = r.json()["prompt_id"]

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        h = httpx.get(f"{settings.comfy_url}/history/{prompt_id}", timeout=30).json()
        if prompt_id in h:
            entry = h[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                # Name the node that blew up. "execution error: 'NoneType' object
                # is not subscriptable" is unactionable on its own; the node type
                # tells the user which part of the graph to look at.
                msgs = []
                for m in status.get("messages", []):
                    if m[0] != "execution_error":
                        continue
                    info = m[1] if len(m) > 1 and isinstance(m[1], dict) else {}
                    node = info.get("node_type") or info.get("node_id") or "?"
                    msgs.append(f"{node}: {info.get('exception_message', '')}")
                raise ComfyError(
                    f"ComfyUI failed while running the graph — {'; '.join(msgs) or 'unknown error'}"
                    f". Check the ComfyUI console for the full traceback.")
            images = []
            for node_output in entry.get("outputs", {}).values():
                images.extend(node_output.get("images", []))
            if images:
                return images
            if status.get("completed"):
                raise ComfyError("run completed but produced no images")
        time.sleep(1.5)
    raise ComfyError(
        f"ComfyUI did not finish within {timeout:g}s (prompt {prompt_id}). It may "
        f"still be running — check the ComfyUI window. A first run that downloads "
        f"or loads a large model can exceed this; retry once the model is cached, "
        f"or free VRAM and try again.")


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
