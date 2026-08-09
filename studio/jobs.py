"""Cooperative cancellation for the long-running stages.

A 24-shot ComfyUI generation or a 40-image cloud caption run can take many
minutes, and until now there was no way to call it off short of killing the
process — which loses the work already done and, for a paid API, the money
already spent.

The contract is deliberately small:

- Cancellation is **cooperative and between items**. A stage checks the token
  before starting each image/shot and returns what it has finished; it never
  interrupts work in flight. Tearing down a half-written PNG or a half-billed
  API call to save a few seconds is not a good trade, and the ComfyUI job we
  would have to interrupt may belong to something else the user queued.
- The token is a plain callable (`should_stop()`), so `pipeline`/`captioner`
  stay free of any UI dependency and the tests drive them with a lambda.
- **Stopping is not failing.** A stopped stage returns normally with partial
  results; the caller reports what finished and how to resume.
"""

from __future__ import annotations

import threading
from typing import Callable

ShouldStop = Callable[[], bool]


class JobControl:
    """A one-shot stop flag shared between a running stage and the UI.

    `threading.Event` because Gradio runs event handlers in worker threads: the
    Stop button is handled on a different thread from the stage it interrupts.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def start(self) -> None:
        """Arm a fresh run. Every stage MUST call this before its loop, or a
        stop left over from the previous run cancels it instantly."""
        self._event.clear()

    def request_stop(self) -> None:
        self._event.set()

    @property
    def stopped(self) -> bool:
        return self._event.is_set()

    def __call__(self) -> bool:
        """So the control can be passed directly as `should_stop=`."""
        return self._event.is_set()


def should_stop_now(should_stop: ShouldStop | None) -> bool:
    """True when a caller-supplied stop token is set. `None` never stops."""
    return bool(should_stop is not None and should_stop())
