"""Dataset Deviser: turn a character, style, or concept into a ready-to-train LoRA dataset."""

__version__ = "0.21.0"

# Decode only the formats config.IMAGE_EXTS accepts. Pillow content-sniffs, so a
# ".png" that is really EPS/PDF/JPEG2000/... would otherwise reach decoders with
# open CVEs (EPS even shells out to Ghostscript).
# shortcut: Pillow 13 fixes most of these but gradio 6.30 caps pillow<13; keep this after.
from PIL import Image as _Image

_Image.init()
_DECODERS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}
_Image.ID[:] = [f for f in _Image.ID if f in _DECODERS]
for _f in set(_Image.OPEN) - _DECODERS:
    del _Image.OPEN[_f]
