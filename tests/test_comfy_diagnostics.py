"""ComfyUI failure translation.

Every payload here was captured from a live ComfyUI (v0.29) by POSTing a broken
graph — not invented — so the parsing matches what the server actually sends.
"""

from __future__ import annotations

import httpx
import pytest

from studio import comfy_api


# ---------- /prompt rejection bodies ----------

MISSING_NODE = """
{"error": {"type": "missing_node_type",
           "message": "Node 'TotallyNotARealNode' not found. The custom node may not be installed.",
           "details": "Node ID '#1'",
           "extra_info": {"node_id": "1", "class_type": "TotallyNotARealNode",
                          "node_title": "TotallyNotARealNode"}},
 "node_errors": {}}
"""

BAD_INPUT = """
{"error": {"type": "prompt_outputs_failed_validation",
           "message": "Prompt outputs failed validation", "details": "", "extra_info": {}},
 "node_errors": {"1": {"errors": [{"type": "custom_validation_failed",
                                   "message": "Custom validation failed for node",
                                   "details": "image - Invalid image file: no_such.png",
                                   "extra_info": {"input_name": "image"}}],
                       "dependent_outputs": ["2"], "class_type": "LoadImage"}}}
"""

NO_OUTPUTS = """
{"error": {"type": "prompt_no_outputs", "message": "Prompt has no outputs",
           "details": "", "extra_info": {}}, "node_errors": {}}
"""


def test_missing_node_names_the_node_and_the_real_fix() -> None:
    msg = comfy_api.describe_rejection(MISSING_NODE)
    assert "TotallyNotARealNode" in msg
    # The bundled workflows are core-only, so an unknown node means a stale
    # ComfyUI far more often than a missing third-party pack.
    assert "core" in msg.lower()
    assert "update" in msg.lower()


def test_validation_failure_names_node_class_and_input() -> None:
    msg = comfy_api.describe_rejection(BAD_INPUT)
    assert "LoadImage" in msg
    assert "node 1" in msg
    assert "image" in msg
    assert "Invalid image file: no_such.png" in msg


def test_other_error_types_fall_back_to_the_servers_own_message() -> None:
    assert "Prompt has no outputs" in comfy_api.describe_rejection(NO_OUTPUTS)


def test_a_non_json_body_is_shown_not_swallowed() -> None:
    """A proxy in front of ComfyUI can answer with an HTML error page."""
    msg = comfy_api.describe_rejection("<html><body>502 Bad Gateway</body></html>")
    assert "502 Bad Gateway" in msg


def test_rejection_never_raises_on_odd_json() -> None:
    for body in ("[]", "null", '"a string"', "{}"):
        assert comfy_api.describe_rejection(body)


# ---------- server status ----------

def test_status_distinguishes_refused_from_timeout(monkeypatch) -> None:
    """'Not reachable' alone sends people to restart a server that is running."""
    def refused(*a, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", refused)
    up, why = comfy_api.server_status()
    assert not up
    assert "nothing is listening" in why
    assert "LDS_COMFY_URL" in why

    def slow(*a, **kw):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(httpx, "get", slow)
    up, why = comfy_api.server_status()
    assert not up
    assert "did not answer" in why


def test_status_flags_a_wrong_app_on_the_port(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get",
                        lambda *a, **kw: httpx.Response(404, request=httpx.Request("GET", "http://x")))
    up, why = comfy_api.server_status()
    assert not up
    assert "404" in why
    assert "not a ComfyUI server" in why


def test_status_ok(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get",
                        lambda *a, **kw: httpx.Response(200, request=httpx.Request("GET", "http://x")))
    up, why = comfy_api.server_status()
    assert up and "reachable" in why
    assert comfy_api.is_up() is True


# ---------- preflight node check ----------

def test_missing_node_types_lists_only_what_is_absent(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(
        200, json={"LoadImage": {}, "PreviewImage": {}},
        request=httpx.Request("GET", "http://x")))
    graph = {"1": {"class_type": "LoadImage"}, "2": {"class_type": "SAM3_Detect"},
             "3": {"class_type": "PreviewImage"}}
    assert comfy_api.missing_node_types(graph) == ["SAM3_Detect"]


def test_missing_node_types_is_silent_when_the_server_cannot_be_asked(monkeypatch) -> None:
    """An unreachable or odd server must never block a run that might work."""
    def boom(*a, **kw):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", boom)
    assert comfy_api.missing_node_types({"1": {"class_type": "Whatever"}}) == []

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(
        200, json={}, request=httpx.Request("GET", "http://x")))
    assert comfy_api.missing_node_types({"1": {"class_type": "Whatever"}}) == []


def test_run_prompt_preflights_before_queueing(monkeypatch) -> None:
    """The missing-node error must arrive without uploading or queueing anything."""
    monkeypatch.setattr(comfy_api, "queue_backlog", lambda: 0)
    monkeypatch.setattr(comfy_api, "missing_node_types", lambda g: ["Krea2EditModelPatch"])

    def must_not_post(*a, **kw):
        raise AssertionError("run_prompt queued a graph it knew would be rejected")

    monkeypatch.setattr(httpx, "post", must_not_post)
    with pytest.raises(comfy_api.ComfyError) as e:
        comfy_api.run_prompt({"1": {"class_type": "Krea2EditModelPatch"}})
    assert "Krea2EditModelPatch" in str(e.value)
    assert "out of date" in str(e.value)
