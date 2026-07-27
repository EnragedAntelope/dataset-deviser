"""Advisory caption analysis: health lint + tag-frequency report.

Two advisories, in the same spirit as `quality.py` (sharpness/exposure) and
`dedupe.py` (near-duplicates): they surface problems for a human to act on and
**never** block, delete, or rewrite anything.

- **Health lint** — flags captions that are empty, suspiciously short, missing
  the trigger, or byte-identical across images (the tell-tale sign of a captioner
  that silently returned junk). Applies to prose *and* tag captions.
- **Tag-frequency report** — for tag-style datasets, counts how many images each
  tag appears in and surfaces the near-ubiquitous ones. A tag present in almost
  every image can't be learned as a *variable* (identity is the trigger), so it
  is either intentional (`1girl`, `solo`) or a prime drop-list candidate — the
  report just shows the counts so the user decides.

Pure Python (no numpy/model/network); the only I/O is reading `.txt` sidecars in
the folder helper. Everything else is unit-tested string logic.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


def _words(caption: str) -> list[str]:
    return caption.split()


# CLIP text encoders (SDXL / Illustrious / NoobAI / Pony) hard-truncate at 77
# tokens including the BOS/EOS markers, silently dropping everything past it.
CLIP_TOKEN_LIMIT = 77


def estimate_clip_tokens(caption: str) -> int:
    """Rough CLIP-token count WITHOUT loading a tokenizer (best-effort estimate).

    CLIP uses BPE, so a word is often more than one token; English averages a bit
    over one token per word, and each comma in a tag list is its own token. We use
    ~1.3 tokens/word + one per comma + two for the BOS/EOS markers, and round up —
    deliberately erring high so the 77-token truncation warning fires a little
    early rather than missing a caption that will actually be cut. Exactness would
    need the real CLIP tokenizer (kept as a possible future dependency).
    """
    words = len(_words(caption))
    commas = caption.count(",")
    return math.ceil(words * 1.3) + commas + 2


def _norm(text: str) -> str:
    """Collapse whitespace + lowercase — the comparison form for tags/captions."""
    return " ".join(text.split()).lower()


@dataclass
class LintReport:
    """Advisory findings for a set of (image-name, caption) pairs."""
    total: int
    empty: list[str] = field(default_factory=list)
    short: list[str] = field(default_factory=list)
    missing_trigger: list[str] = field(default_factory=list)
    # Captions estimated to exceed CLIP's 77-token limit (tag datasets only).
    too_long: list[str] = field(default_factory=list)
    # (caption preview, [image names]) for captions repeated verbatim.
    duplicates: list[tuple[str, list[str]]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.empty or self.short or self.missing_trigger
                    or self.too_long or self.duplicates)


def lint_captions(
    pairs: Iterable[tuple[str, str]],
    trigger: str = "",
    min_words: int = 2,
    clip_limit: int = 0,
) -> LintReport:
    """Flag empty / too-short / trigger-missing / duplicate captions.

    `pairs` is (image name, caption). `min_words` is deliberately conservative
    (a caption with fewer words is flagged "short") so the advisory only fires on
    near-junk, never on a legitimately concise caption. Duplicate detection is on
    the normalized (whitespace-collapsed, lowercased) caption and only reports
    non-empty captions shared by 2+ images. `clip_limit` (>0) additionally flags
    captions whose estimated CLIP-token count exceeds it — used for tag datasets,
    where SDXL-family encoders truncate at 77. Order-preserving.
    """
    pairs = list(pairs)
    trig = _norm(trigger)
    report = LintReport(total=len(pairs))
    seen: dict[str, list[str]] = {}
    for name, caption in pairs:
        text = caption.strip()
        if not text:
            report.empty.append(name)
            continue
        if len(_words(text)) < min_words:
            report.short.append(name)
        if trig and not _norm(text).startswith(trig):
            report.missing_trigger.append(name)
        if clip_limit and estimate_clip_tokens(text) > clip_limit:
            report.too_long.append(name)
        seen.setdefault(_norm(text), []).append(name)
    for norm_text, names in seen.items():
        if len(names) > 1:
            preview = norm_text if len(norm_text) <= 60 else norm_text[:57] + "…"
            report.duplicates.append((preview, names))
    return report


def looks_like_tags(captions: Iterable[str]) -> bool:
    """Heuristic: are these comma-separated tag captions rather than prose?

    Tag captions are short comma segments (1-3 words each); prose has long
    clauses. Averaging words-per-comma-segment separates the two without needing
    to know the caption style — so the tag-frequency report only fires when it is
    meaningful.
    """
    seg_lens = [len(seg.split()) for cap in captions for seg in cap.split(",")
                if seg.split()]
    if not seg_lens:
        return False
    return sum(seg_lens) / len(seg_lens) <= 3.0


def tag_frequency(captions: Iterable[str], trigger: str = "") -> list[tuple[str, int]]:
    """Count how many captions each tag appears in (comma-split, trigger excluded).

    Presence is per-caption (a tag repeated inside one caption counts once), so a
    count is 'number of images carrying this tag'. Returned most-common first.
    """
    trig = _norm(trigger)
    counts: Counter[str] = Counter()
    for cap in captions:
        seen = {t for seg in cap.split(",") if (t := _norm(seg)) and t != trig}
        counts.update(seen)
    return counts.most_common()


def ubiquitous_tags(
    captions: Iterable[str],
    trigger: str = "",
    min_fraction: float = 0.9,
    min_images: int = 4,
) -> list[tuple[str, int]]:
    """Tags present in at least `min_fraction` of the captions.

    Skipped on tiny sets (`< min_images`), where 'ubiquitous' is meaningless.
    These are drop-list candidates: a tag on nearly every image can't be learned
    as a variable. Advisory only — some (1girl/solo) are intentional.
    """
    captions = list(captions)
    if len(captions) < min_images:
        return []
    threshold = max(2, math.ceil(len(captions) * min_fraction))
    return [(tag, n) for tag, n in tag_frequency(captions, trigger) if n >= threshold]


def analyze_pairs(
    pairs: Iterable[tuple[str, str]],
    trigger: str = "",
) -> tuple[LintReport, list[tuple[str, int]]]:
    """(health report, ubiquitous tags) for (name, caption) pairs.

    The tag report is only computed for tag-like captions (empty otherwise), so
    prose datasets get the health lint without a meaningless frequency list. The
    CLIP 77-token truncation check is likewise gated to tag-like captions (that is
    where the SDXL-family encoders that truncate live; prose Flux/T5 encoders do
    not have the 77-token cap), so prose datasets are never nagged about length.
    """
    pairs = list(pairs)
    captions = [c for _, c in pairs if c.strip()]
    is_tags = looks_like_tags(captions)
    report = lint_captions(pairs, trigger=trigger,
                           clip_limit=CLIP_TOKEN_LIMIT if is_tags else 0)
    ubiquitous = ubiquitous_tags(captions, trigger) if is_tags else []
    return report, ubiquitous


def folder_caption_kind(folder: Path) -> str:
    """Heuristic caption kind for a folder: 'tags', 'prose', or '' if uncaptioned.

    Used by the ④→⑤ sanity check to warn when a dataset's captions don't match the
    trainer preset (tag captions for a prose model or vice-versa). Reuses the same
    `looks_like_tags` heuristic the tag-frequency report is gated on.
    """
    from studio.config import list_images

    captions = [c for img in list_images(folder) if (c := _read_caption(img))]
    if not captions:
        return ""
    return "tags" if looks_like_tags(captions) else "prose"


def analyze_folder(
    folder: Path,
    trigger: str = "",
) -> tuple[LintReport, list[tuple[str, int]]]:
    """Read a folder's `.txt` sidecars and analyze them (missing sidecar = empty)."""
    from studio.config import list_images

    pairs = [(img.name, _read_caption(img)) for img in list_images(folder)]
    return analyze_pairs(pairs, trigger=trigger)


def _read_caption(image: Path) -> str:
    """Shared forgiving sidecar read — see `studio.config.read_caption`."""
    from studio.config import read_caption

    return read_caption(image)


def markdown_summary(
    report: LintReport,
    ubiquitous: list[tuple[str, int]],
    max_list: int = 8,
) -> str:
    """Render the two advisories as a compact Markdown block for UI/CLI reuse."""
    def _names(names: list[str]) -> str:
        shown = ", ".join(names[:max_list])
        return shown + (f" (+{len(names) - max_list} more)" if len(names) > max_list else "")

    lines: list[str] = []
    if report.clean:
        lines.append(f"✅ **Caption health:** {report.total} caption(s) look OK.")
    else:
        lines.append(f"**Caption health** ({report.total} caption(s)):")
        if report.empty:
            lines.append(f"- ⚠ **{len(report.empty)} empty:** {_names(report.empty)}")
        if report.missing_trigger:
            lines.append(f"- ⚠ **{len(report.missing_trigger)} missing the trigger:** "
                         f"{_names(report.missing_trigger)}")
        if report.short:
            lines.append(f"- ⚠ **{len(report.short)} very short:** {_names(report.short)}")
        if report.too_long:
            lines.append(f"- ✂️ **{len(report.too_long)} likely over CLIP's 77-token "
                         f"limit** (SDXL/Illustrious/Pony will truncate; token estimate): "
                         f"{_names(report.too_long)}")
        if report.duplicates:
            groups = "; ".join(f"{_names(names)} = “{preview}”"
                               for preview, names in report.duplicates[:5])
            more = (f" (+{len(report.duplicates) - 5} more)"
                    if len(report.duplicates) > 5 else "")
            lines.append(f"- 🔁 **{len(report.duplicates)} identical-caption group(s)** "
                         f"(often a captioner that returned the same text): {groups}{more}")
    if ubiquitous:
        tags = ", ".join(f"`{tag}` ×{n}" for tag, n in ubiquitous[:max_list])
        more = f" (+{len(ubiquitous) - max_list} more)" if len(ubiquitous) > max_list else ""
        lines.append(f"\n🏷️ **Tags on nearly every image:** {tags}{more}  \n"
                     "Intentional (like `1girl`, `solo`) or drop-list candidates — a tag on "
                     "every image can't be learned as a variable. Use the **drop-list** in "
                     "*Tag options* to remove any you don't want.")
    return "\n".join(lines)
