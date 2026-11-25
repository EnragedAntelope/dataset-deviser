"""Dataset Deviser: turn a character, style, or concept into a ready-to-train LoRA dataset."""

__version__ = "0.22.1"

# Decode only the formats config.IMAGE_EXTS accepts. Pillow content-sniffs, so a
# ".png" that is really EPS/PDF/JPEG2000/... would otherwise reach decoders with
# open CVEs (EPS even shells out to Ghostscript). Pillow 13 (Python 3.11+) fixes
# the known ones; 3.10 stays on 12.3, and the allowlist narrows the surface either way.
from PIL import Image as _Image

_Image.init()
_DECODERS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}
_Image.ID[:] = [f for f in _Image.ID if f in _DECODERS]
for _f in set(_Image.OPEN) - _DECODERS:
    del _Image.OPEN[_f]
