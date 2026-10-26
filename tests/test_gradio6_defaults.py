"""Gradio 6 defaults a Textbox to None; the app's handlers expect strings."""
from __future__ import annotations

import gradio as gr

import app as A


def test_no_textbox_starts_as_none() -> None:
    boxes = [c for c in A.demo.blocks.values() if isinstance(c, gr.Textbox)]
    assert boxes and all(c.value is not None for c in boxes)
    sent = {c["id"]: c["props"].get("value") for c in A.demo.get_config_file()["components"]}
    assert all(sent[c._id] is not None for c in boxes)
