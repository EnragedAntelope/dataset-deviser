"""upload_image sends an oversized source downscaled, and cleans up its temp file."""
import numpy as np
from PIL import Image

from studio import comfy_api


class _Resp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"name": "ok.png"}


def test_oversized_source_is_shrunk_and_temp_removed(tmp_path, monkeypatch):
    src = tmp_path / "big.png"
    # Incompressible noise so the PNG is genuinely large relative to the tiny limit.
    noise = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    Image.fromarray(noise).save(src)
    monkeypatch.setattr(comfy_api, "MAX_UPLOAD_BYTES", 100_000)
    assert src.stat().st_size > 100_000

    sent = {}

    def fake_post(url, files, data, timeout):
        sent["size"] = len(files["image"][1].read())
        sent["path"] = files["image"][1].name
        return _Resp()

    monkeypatch.setattr(comfy_api.httpx, "post", fake_post)
    assert comfy_api.upload_image(src) == "ok.png"
    assert sent["size"] <= 100_000
    assert not __import__("pathlib").Path(sent["path"]).exists()
    assert src.exists()


def test_small_source_sent_as_is(tmp_path, monkeypatch):
    src = tmp_path / "small.png"
    Image.new("RGB", (8, 8)).save(src)
    sent = {}

    def fake_post(url, files, data, timeout):
        sent["path"] = files["image"][1].name
        return _Resp()

    monkeypatch.setattr(comfy_api.httpx, "post", fake_post)
    comfy_api.upload_image(src)
    assert sent["path"] == str(src)
