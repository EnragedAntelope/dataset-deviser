"""_pick_port: LDS_PORT pins, an occupied preferred port falls back to a free one."""
import socket

from app import _pick_port


def test_lds_port_pins(monkeypatch):
    monkeypatch.setenv("LDS_PORT", "9123")
    assert _pick_port() == 9123


def test_occupied_preferred_falls_back(monkeypatch):
    monkeypatch.delenv("LDS_PORT", raising=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        taken = s.getsockname()[1]
        picked = _pick_port(taken)
    assert picked not in (0, taken)
