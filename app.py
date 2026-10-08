"""Dataset Deviser — Gradio UI.

Every tab is standalone: point it at any folder (or upload files) and run just
that stage. When you do run stages in order, each one auto-fills the next
tab's input folder — chaining is a convenience, never a requirement.

Run:  python app.py   then open http://127.0.0.1:7861 (or another free port if 7861 is
taken or reserved; the URL is printed at launch, and LDS_PORT in .env pins one)
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import string
import warnings
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd

from studio import pipeline, shot_style
from studio import user_config as _uc_boot
from studio.captioner import (
    SUBJECT_ALIASES,
    Captioner,
    CaptionerConfigError,
    apply_affixes,
    caption_images,
    drop_blacklisted_tags,
    estimate_caption_cost,
    finalize_caption,
    merge_tagger_overrides,
    parse_blacklist,
    resolve_captioner_config,
)
from studio.config import (
    CAPTIONERS,
    CAPTIONERS_BY_KEY,
    friendly_api_error,
    list_images,
    load_caption_model_cache,
    read_caption,
    settings,
)
from studio.engines.gemini import (
    AUTO_MODEL,
    image_model_choices,
    image_price,
    known_image_models,
    resolve_image_model,
)
from studio.jobs import JobControl
from studio.shotplan import (
    Shot,
    apply_prop_exclusion,
    apply_wardrobe,
    coverage,
    plan_for_type,
)
from studio.trainer_configs import TRAINER_MODELS, TRAINERS, optimizer_choices

TRAINER_CHOICES = [(label, key) for key, label in TRAINERS.items()]

ENGINE_CHOICES = [
    ("Cloud — Gemini image model (best identity fidelity, SFW only)", "gemini"),
    ("Local — ComfyUI Qwen Image 2.1 (free, private, uncensored)", "comfyui"),
]
CLOUD_MODEL_CHOICES = image_model_choices(known_image_models())


def _cloud_model_default(pinned: str, choices: list[tuple[str, str]]) -> str:
    """The ② dropdown's start value: the .env pin (retired ids mapped) if listed, else Auto."""
    if pinned != AUTO_MODEL:
        pinned = resolve_image_model(pinned)
    return pinned if pinned in {m for _, m in choices} else AUTO_MODEL


CLOUD_MODEL_DEFAULT = _cloud_model_default(settings.gemini_image_model, CLOUD_MODEL_CHOICES)
CAPTIONER_CHOICES = [(c.label, c.key) for c in CAPTIONERS]

# Gemini caption-model dropdown seed: use the local cache if present, else a
# safe rolling-alias default. Live refresh happens on demand via the button
# (kept off the startup path so the UI loads instantly and offline).
_DEFAULT_CAPTION_MODEL = "gemini-flash-latest"
_cached_caption_models = load_caption_model_cache() or []
CAPTION_MODEL_CHOICES = [(m["model_id"], m["model_id"]) for m in _cached_caption_models] \
    or [(_DEFAULT_CAPTION_MODEL, _DEFAULT_CAPTION_MODEL)]
ISOLATION_CHOICES = [
    ("Built-in SAM3 (no ComfyUI needed; gated HF model)", "builtin"),
    ("ComfyUI SAM3 workflow", "comfyui"),
]
RESTORE_BACKEND_CHOICES = [
    ("Auto (ComfyUI models if reachable, else basic)", "auto"),
    ("ComfyUI (DeJPG + photo upscale models)", "comfyui"),
    ("Basic (Lanczos only, no ComfyUI)", "basic"),
]

# Global dataset-type selector — a deliberate, documented exception to the
# "no global mode" design rule (a dataset IS one type; per-tab type controls
# invite mismatch). It only tunes prompts/defaults; stages still run standalone.
DATASET_TYPE_CHOICES = [
    ("Character — a person/creature identity (default)", "character"),
    ("Style — an art style / aesthetic", "style"),
    ("Concept — an object, action, or idea", "concept"),
]
_YT_TOOL = "https://github.com/EnragedAntelope/youtube-screenshot-extractor"
_TYPE_GUIDANCE = {
    "character": "",
    "style": (f"**Style dataset — ② does not apply.** A style can't be synthesized from a "
              f"reference the way an identity or an object can, so generation is disabled "
              f"here. Collect your own images that share the look (a [YouTube Screenshot "
              f"Extractor]({_YT_TOOL}) can pull high-quality frames from video), then go "
              "straight to **③ Caption → ④ Export → ⑤ Train**. Caption the *content*, not the "
              "style — the trigger learns the look. Isolation defaults **off** (a style is "
              "whole-image)."),
    "concept": ("**Concept dataset** — the plan below is an 18-shot *object* set: a "
                "turnaround (angles), framing/scale variation, and context shots. It works "
                "best for a **solid object** you have a clean reference of. For an action or "
                "an abstract idea, bring your own images instead (a [YouTube Screenshot "
                f"Extractor]({_YT_TOOL}) helps) and start at **③ Caption** — every row here "
                "is editable, so you can also prune or rewrite shots. Isolation defaults "
                "**on** with subject `object`; change it to name your thing (e.g. `radio`, "
                "`sword`) or turn it off for scenes."),
}
_TRIGGER_INFO = {
    "character": "Unique token the LoRA learns as the subject. Placed first in every caption.",
    "style": "Unique token the LoRA learns as the STYLE/aesthetic. Placed first in every caption.",
    "concept": "Unique token the LoRA learns as the CONCEPT. Placed first in every caption.",
}
# Per-type wording for the "who/what is this dataset about" fields (②/③/④).
_NAME_LABEL = {"character": "Character name", "style": "Style name",
               "concept": "Concept name"}
_NAME_INFO = {
    "character": "Used in prose captions; taggers ignore it.",
    "style": "Names the dataset only — Style captions are trigger-first and never "
             "mention a name.",
    "concept": "Names the dataset only — Concept captions are trigger-first and never "
               "mention a name.",
}
# What SAM3 should keep when isolating, per type.
_ISOLATE_SUBJECT = {"character": "character", "style": "character", "concept": "object"}


def on_dataset_type_change(dataset_type: str, name: str = "",
                           style_key: str = shot_style.MATCH, style_text: str = "",
                           identity: str = "identity"):
    """Retune every type-dependent control across the tabs, and remember the
    choice for the next launch.

    Character keeps exactly the UI it had before dataset types existed. Style
    disables ② (a style cannot be generated); Concept swaps in the object shot
    plan and drops the character-only controls (wardrobe, prop exclusion).
    """
    from studio import user_config

    user_config.set_dataset_type(dataset_type)
    is_concept = dataset_type == "concept"
    is_style = dataset_type == "style"
    label = _NAME_LABEL.get(dataset_type, _NAME_LABEL["character"])
    subject_kw = _ISOLATE_SUBJECT.get(dataset_type, "character")
    return (
        # ① preprocess
        gr.Checkbox(value=not is_style),                       # isolate default
        gr.Textbox(value=subject_kw),                          # SAM3 subject
        # ② generate & curate
        gr.Markdown(value=_TYPE_GUIDANCE.get(dataset_type, ""),
                    visible=bool(_TYPE_GUIDANCE.get(dataset_type, ""))),
        gr.Button(value=f"Rebuild default plan with {label.lower()}",
                  interactive=not is_style),                   # refresh plan
        _plan_table(dataset_type, name, style_key, style_text, identity),  # shot plan
        gr.Button(interactive=not is_style),                   # generate
        gr.Button(interactive=not is_style),                   # regenerate
        gr.Button(visible=not (is_style or is_concept)),       # randomize outfits
        gr.Button(visible=not (is_style or is_concept)),       # clear outfits
        gr.Markdown(visible=not (is_style or is_concept)),     # wardrobe blurb
        gr.Checkbox(value=not is_concept),                     # exclude props
        gr.Textbox(value=subject_kw),                          # ② isolation subject
        # ③ caption
        gr.Checkbox(visible=is_style),                         # sparse (style only)
        # header identity row — ②/③/④/⑤ read these two directly, so this is the
        # only place the per-type wording has to land
        gr.Textbox(label=label,                                # project_name
                   info=_NAME_INFO.get(dataset_type, _NAME_INFO["character"])),
        gr.Textbox(                                            # project_trigger
            info=_TRIGGER_INFO.get(dataset_type, _TRIGGER_INFO["character"])),
    )


# ---------- helpers ----------

# One shared stop flag. The app is single-user on localhost and Gradio runs one
# queued event at a time, so exactly one heavy stage can be in flight; a token
# per stage would suggest a concurrency this app doesn't have. Every stage arms
# it with JOB.start() before its loop, so a stop left over from a previous run
# can never cancel the next one.
JOB = JobControl()


def request_stop() -> str:
    """Wired with queue=False so the click is served while a stage is running —
    a queued Stop button would wait for the job it is meant to interrupt."""
    JOB.request_stop()
    return ("⏹ Stop requested — finishing the current image/shot, then returning "
            "everything completed so far.")


def _stopped_note(kind: str, done: int, total: int, resume: str) -> str:
    """Consistent 'you stopped this' line: what finished, and how to resume."""
    return (f"\n\n⏹ **Stopped after {done} of {total} {kind}.** The {done} already "
            f"finished are saved. {resume}")


def _stamped(kind: str) -> Path:
    d = settings.runs_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{kind}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _inputs(files: list[str] | None, folder: str) -> list[Path]:
    """Uploaded files win; otherwise list the folder."""
    if files:
        return [Path(f) for f in files]
    if folder.strip():
        images = list_images(Path(folder.strip()))
        if images:
            return images
        raise gr.Error(f"No images found in folder: {folder}")
    raise gr.Error("Upload image(s) or enter an input folder first.")


# Characters Windows forbids in a path (excluding the drive-letter colon).
_WIN_INVALID_PATH = re.compile(r'[<>"|?*]')


def _validate_out_dir(path_str: str) -> Path:
    """Validate a user-entered output folder, raising a friendly gr.Error for
    paths the OS can't create — instead of a raw OSError traceback."""
    raw = path_str.strip().strip('"')
    if not raw:
        raise gr.Error("Enter an output folder.")
    # Ignore a leading drive-letter colon (C:\...) when scanning the rest for
    # the colon / other characters Windows forbids inside a path.
    tail = raw[2:] if len(raw) >= 2 and raw[1] == ":" else raw
    if _WIN_INVALID_PATH.search(tail) or ":" in tail or any(ord(c) < 32 for c in raw):
        raise gr.Error(
            f"'{path_str}' isn't a valid folder path — it contains characters the OS "
            f'forbids (< > : " | ? * or line breaks). Use a path like D:\\my-folder.'
        )
    return Path(raw)


def _allowed_media_paths() -> list[str]:
    """Folders Gradio is allowed to serve images from. The app writes generated
    images to run folders AND to arbitrary user-chosen output folders on any
    drive, so we allow the configured roots plus every present drive root.
    Acceptable only because the server binds to localhost with no auth (see the
    note on demo.launch)."""
    paths = {str(settings.runs_dir), str(settings.output_root),
             str(settings.shot_plans_dir)}
    if os.name == "nt":
        paths |= {f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()}
    else:
        paths.add("/")
    return sorted(paths)


# Human-editable columns lead; the long prompt cells trail. Column ORDER and
# WIDTHS must be set explicitly: pydantic field order otherwise puts the two
# ~200-char prompts in the middle, squeezing `outfit` to an unreadable sliver.
PLAN_COLUMNS = ["id", "kind", "framing", "emotion", "setting", "outfit",
                "local_prompt", "cloud_prompt", "chain_from"]
PLAN_COLUMN_WIDTHS = ["110px", "70px", "90px", "110px", "200px", "220px",
                      "260px", "260px", "100px"]


def _plan_table(dataset_type: str, name: str = "", style_key: str = shot_style.MATCH,
                style_text: str = "", identity: str = "identity") -> pd.DataFrame:
    """The ② table for a dataset type (empty for Style, which never generates).

    A character under the "identity" policy starts dressed in varied outfits, so
    the trigger learns the person rather than their clothes.
    """
    shots = plan_for_type(dataset_type, name, style_key, style_text)
    if dataset_type == "character" and identity == "identity":
        from studio.wardrobe import dress

        shots = dress(shots)
    return _shots_to_df(shots)


def _identity_visible(dataset_type: str):
    return gr.update(visible=dataset_type == "character")


def on_identity_change(df: pd.DataFrame, identity: str, dataset_type: str):
    """Dress or undress the current plan to match the identity policy."""
    if dataset_type != "character":
        return gr.skip(), gr.skip()
    return randomize_outfits(df) if identity == "identity" else clear_outfits(df)


def rebuild_plan_for_style(dataset_type: str, name: str, style_key: str,
                           style_text: str, identity: str = "identity"):
    """Rebuild the ② table when the shot style changes, and remember the choice.

    The style is baked into the prompt cells at build time (so the table shows
    what will actually be sent), which means changing it has to regenerate them
    — and that discards hand edits. Say so rather than letting the user discover
    it: silence here looks like the edits vanished for no reason.
    """
    from studio import user_config

    user_config.set_shot_style(style_key, style_text)
    style = shot_style.resolve(style_key, style_text)
    if dataset_type == "style":
        # Style never generates; there is no table to rebuild.
        return gr.skip(), gr.skip()
    note = (f"Shot plan rebuilt for **{style.label}** — any hand-edited prompt "
            f"cells were replaced.")
    if style_key == shot_style.CUSTOM and not (style_text or "").strip():
        note = ("⚠️ Custom style selected but no description typed — falling back to "
                "**matching the reference image**. Type a style below and the plan "
                "rebuilds.")
    return _plan_table(dataset_type, name, style_key, style_text, identity), note


def _toggle_style_text(style_key: str):
    return gr.Textbox(visible=style_key == shot_style.CUSTOM)


def preview_final_prompt(plan_df: pd.DataFrame, engine: str, exclude_props: bool):
    """Show the prompt the FIRST plan row will actually send.

    Wardrobe, prop exclusion and (via the rebuild) the shot style are folded in
    at generation time, so the table's own cell is not the final text. Without
    this there is no way to check what the style control actually did short of
    spending a generation.
    """
    shots = _df_to_shots(plan_df)
    if not shots:
        raise gr.Error("The plan is empty — nothing to preview.")
    shot = apply_wardrobe(shots[0])
    if exclude_props:
        shot = apply_prop_exclusion(shot)
    field = "local_prompt" if engine == "comfyui" else "cloud_prompt"
    which = "Local (ComfyUI / Qwen-Image 2.1)" if engine == "comfyui" else "Cloud (Gemini)"
    return (f"**{which} prompt for `{shot.id}`** — exactly what the engine receives, "
            f"after outfit and prop-exclusion are folded in:\n\n```\n"
            f"{getattr(shot, field)}\n```")


def _shots_to_df(shots: list[Shot]) -> pd.DataFrame:
    """Single place that builds the plan table, so column order can't drift
    between the default plan and a loaded one."""
    return pd.DataFrame([s.model_dump() for s in shots], columns=PLAN_COLUMNS)


def randomize_outfits(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Fill the outfit column with distinct random unisex outfits.

    Close-ups are skipped: they frame the face and upper shoulders, so a full
    outfit description there tends to widen the shot instead of dressing it.
    """
    from studio.wardrobe import OUTFIT_SHOT_KINDS, random_outfits

    df = df.copy()
    targets = [i for i, row in df.iterrows()
               if str(row.get("kind", "")) in OUTFIT_SHOT_KINDS]
    if not targets:
        raise gr.Error("No angle/pose rows to dress — outfits are skipped for "
                       "close-ups, where clothing is barely in frame.")
    outfits = random_outfits(len(targets))
    for i, outfit in zip(targets, outfits, strict=True):
        df.at[i, "outfit"] = outfit
    return df, (f"🎲 Dressed {len(targets)} angle/pose shots in distinct outfits "
                f"({len(df) - len(targets)} close-ups left blank). Click again to "
                f"reroll, or clear the column to go back to the reference's clothing.")


def clear_outfits(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    df["outfit"] = ""
    return df, "Outfit column cleared — every shot keeps the reference's clothing."


# ---------- click-to-pick galleries (②/③/④) ----------
#
# A Gradio Gallery is output-only, so each picker is a Gallery + CheckboxGroup pair
# driven by one list of rows: (image path, checkbox value, base label). The gallery
# caption carries the ✅/⬜ mark as its FIRST characters so the state is readable even
# when a long label is truncated. Wiring is deliberately one-directional to avoid an
# event loop: gallery.select -> checkbox value, checkbox.change -> gallery labels.

_PICK_ON = "✅"
_PICK_OFF = "⬜"


def _goto_tab(tab_id: str):
    """Select a tab by id — a hand-off button that leaves you staring at the tab you
    were already on reads as 'nothing happened'."""
    return gr.Tabs(selected=tab_id)


def _picker_gallery(rows: list[tuple[str, str, str]], selected) -> list[tuple[str, str]]:
    """Render (path, label) gallery items, marking each row's selection state."""
    chosen = set(selected or [])
    return [(path, f"{_PICK_ON if value in chosen else _PICK_OFF} {label}")
            for path, value, label in rows]


def _picker_order(rows: list[tuple[str, str, str]], selected) -> list[str]:
    """Selected values in row order — a CheckboxGroup value must follow its choices."""
    chosen = set(selected or [])
    return [value for _, value, _ in rows if value in chosen]


def _picker_mark(rows: list[tuple[str, str, str]], selected):
    """Re-render the gallery marks after the CheckboxGroup changed (either source)."""
    return _picker_gallery(rows, selected)


# Element ids the picker script pairs up: (gallery, checkbox group, zoom checkbox).
PICKER_IDS = [
    ("dd-gallery-gen", "dd-picks-gen", "dd-zoom-gen"),
    ("dd-gallery-cap", "dd-picks-cap", "dd-zoom-cap"),
    ("dd-gallery-exp", "dd-picks-exp", "dd-zoom-exp"),
    ("dd-gallery-quick", "dd-picks-quick", "dd-zoom-quick"),
]

# Clicking a thumbnail must toggle it, and Gradio's own `Gallery.select` event cannot
# do that: it only fires when the clicked index DIFFERS from the one the component
# already holds, so a second click on the same image is swallowed — an image could be
# unpicked and never re-picked. The internal index is not resettable from the server
# either (see the Gotcha). So the click is forwarded to the CheckboxGroup entry at the
# same position instead; the group is the single source of truth and its `.change`
# already re-renders the gallery marks. If this script never runs, the checkbox list
# below every gallery still works exactly as before.
_PICKER_SCRIPT = """
<script>
(() => {
  const PICKERS = %s;
  const ON = %s, OFF = %s;
  // Last thumbnail index clicked in each gallery, for shift-click range select.
  const lastIndex = new Map();

  // Flip the mark straight away. The round-trip that re-renders the gallery takes
  // ~1.5s, and a picker whose tick lands a second and a half after the click reads
  // as unresponsive — you end up clicking twice. The server value overwrites this
  // moments later, so a wrong guess self-corrects.
  function flipMark(thumb) {
    const label = thumb.querySelector(".caption-label");
    if (!label) return;
    const text = label.textContent;
    if (text.startsWith(ON)) label.textContent = OFF + text.slice(ON.length);
    else if (text.startsWith(OFF)) label.textContent = ON + text.slice(OFF.length);
  }

  document.addEventListener("click", (event) => {
    const thumb = event.target.closest && event.target.closest(".thumbnail-item");
    if (!thumb) return;
    for (const [galleryId, picksId, zoomId] of PICKERS) {
      const gallery = document.getElementById(galleryId);
      if (!gallery || !gallery.contains(thumb)) continue;
      const zoom = document.getElementById(zoomId);
      const zoomOn = zoom && zoom.querySelector('input[type="checkbox"]');
      if (zoomOn && zoomOn.checked) return;  // the click belongs to the lightbox
      const picks = document.getElementById(picksId);
      if (!picks) return;
      const thumbs = Array.from(gallery.querySelectorAll(".thumbnail-item"));
      const boxes = Array.from(picks.querySelectorAll('input[type="checkbox"]'));
      const index = thumbs.indexOf(thumb);
      if (index < 0 || index >= boxes.length) return;
      boxes[index].click();
      flipMark(thumb);
      // Shift-click extends the pick to match this click's new state across the
      // range since the last click IN THIS GALLERY — the common file-manager
      // convention. A stale last index (list reloaded shorter since) is clamped
      // rather than trusted.
      const prev = lastIndex.get(galleryId);
      if (event.shiftKey && prev !== undefined) {
        const from = Math.min(prev, boxes.length - 1);
        const target = boxes[index].checked;
        const [lo, hi] = from < index ? [from, index] : [index, from];
        for (let i = lo; i <= hi; i++) {
          if (boxes[i].checked !== target) {
            boxes[i].click();
            flipMark(thumbs[i]);
          }
        }
      }
      lastIndex.set(galleryId, index);
      return;
    }
  }, true);
})();
</script>
""" % (json.dumps(PICKER_IDS), json.dumps(_PICK_ON), json.dumps(_PICK_OFF))  # noqa: UP031 - the JS body is full of literal { }, which .format/f-strings would need escaped


def _set_zoom(on: bool):
    """Flip a picker gallery between toggle-on-click and zoom-on-click.

    Gradio's Gallery only offers the enlarge-on-click lightbox, and its fullscreen
    icon is part of that same preview UI — with `allow_preview=False` there is no
    way at all to see an image bigger. Picking is the common action so it stays the
    default, but ④'s final review genuinely needs a closer look, hence the mode.
    """
    return gr.update(allow_preview=bool(on))


def _pick_all(rows: list[tuple[str, str, str]]):
    return gr.CheckboxGroup(value=[value for _, value, _ in rows])


def _pick_none(rows: list[tuple[str, str, str]]):
    return gr.CheckboxGroup(value=[])


def _pick_captioned(rows: list[tuple[str, str, str]]):
    """Keep only rows whose image has a non-empty .txt sidecar."""
    return gr.CheckboxGroup(
        value=[value for path, value, _ in rows if read_caption(Path(path))])


def _df_to_shots(df: pd.DataFrame) -> list[Shot]:
    def val(row, k):
        v = row[k] if k in row else ""
        return "" if pd.isna(v) else str(v)

    cols = (
        "id", "kind", "local_prompt", "cloud_prompt",
        "chain_from", "emotion", "setting", "outfit", "framing",
    )
    return [Shot(**{k: val(row, k) for k in cols})
            for _, row in df.iterrows() if val(row, "id").strip()]

def _flagged(path: Path, label: str) -> str:
    """`label` plus this image's advisory quality flags (blurry, dark, …)."""
    try:
        from studio.quality import composition_flags, is_blurry

        flags: list[str] = []
        blurry, score = is_blurry(path)
        if blurry:
            flags.append(f"blurry ({score:.0f})")
        flags += composition_flags(path)
    except Exception:
        return label  # quality checks are advisory — never block the gallery on them
    return f"{label}  ⚠ {', '.join(flags)}" if flags else label


def _gen_gallery(results: list[pipeline.GenResult], selected=None):
    """Picker rows + gallery + CheckboxGroup for ②'s kept-shot list.

    `selected=None` keeps everything (a fresh generation); pass a value to carry an
    existing pick across a re-sync instead of silently re-checking rejected shots.
    """
    ok = [r for r in results if r.path and r.path.exists()]
    rows = [(str(r.path), r.shot.id, _flagged(r.path, r.shot.id)) for r in ok]
    ids = [r.shot.id for r in ok]
    keep = ids if selected is None else _picker_order(rows, selected)
    return rows, _picker_gallery(rows, keep), gr.CheckboxGroup(choices=ids, value=keep)


# ---------- ① preprocess ----------

def _failure_hint(error: str) -> str:
    """One actionable sentence for a per-image preprocess failure.

    A skipped image is only useful information if it says what to *do*. The
    mapping is on the error text because the exception types cross three
    backends (SAM3, ComfyUI, Pillow) and the message is what the user sees.
    """
    low = error.lower()
    if "found no" in low:
        return ("adjust **Subject to keep** (try `person`, `object`, or the thing's own "
                "noun), or untick **Isolate subject** and re-run just this image.")
    if "gated" in low or "authenticate" in low:
        return ("accept the SAM3 licence on its Hugging Face model page and set "
                "`HF_TOKEN` in `.env`, or switch **Isolation backend** to ComfyUI.")
    if "comfyui" in low:
        return ("start ComfyUI (or fix the setting the message names), or switch "
                "**Restoration backend** to Basic / **Isolation backend** to Built-in.")
    if "cannot identify" in low or "truncated" in low or "decoder" in low:
        return "the file looks corrupt or isn't a real image — re-export or drop it."
    return "see the Log below for the full message."


def _plain(markdown: str) -> str:
    """Strip the light markdown a note uses, for contexts that render plain text.

    `gr.Warning`/`gr.Error` toasts are NOT markdown — emphasis and backticks show
    up as literal `**` and `` ` `` characters in the popup.
    """
    return markdown.replace("**", "").replace("`", "")


def _preprocess_note(reports, out_dir: Path, alpha_cutout: bool) -> str:
    """Result markdown for ①, naming every skipped image and what to do about it."""
    ok = [r for r in reports if r.output]
    failed = [r for r in reports if r.error]
    if not ok:
        head = (f"❌ **No image could be preprocessed** — nothing was written to "
                f"{out_dir}.")
    else:
        head = f"✅ {len(ok)} image(s) preprocessed into {out_dir}"
        refs = next((r.reference.parent for r in ok if r.reference), None)
        if refs and alpha_cutout:
            head += (f" — transparent cutouts in {refs}, not auto-filled into ②, "
                     f"which expects a white-background reference.")
        elif refs:
            head += f" — isolated references for ② in {refs}."
    cut = [r.source.name for r in ok if r.cut_off]
    if cut and len(cut) == len(ok):
        head += (f"\n\n⚠️ **Every reference is cut off at the bottom** "
                 f"({', '.join(cut)}) — full-body shots will invent the legs. Add a "
                 f"full-body photo as another reference if you have one.")
    elif cut:
        head += (f"\n\n<sub>Cut off at the bottom: {', '.join(cut)} — the other "
                 f"reference(s) show more of the body.</sub>")
    if not failed:
        return head
    lines = "\n".join(f"- `{r.source.name}` — {r.error}  \n  → {_failure_hint(r.error)}"
                      for r in failed)
    if not ok:
        return f"{head}\n\nAll {len(reports)} failed:\n\n{lines}"
    return (f"{head}\n\n⚠️ **Skipped {len(failed)} of {len(reports)}** — the "
            f"{len(ok)} image(s) above were still written:\n\n{lines}")


def do_preprocess(files: list[str], folder: str, target: int, restore_mode: str,
                  restore_backend: str, isolate: bool, isolation_backend: str,
                  subject_prompt: str, exclude_prompt: str, tighten: bool = False,
                  alpha_cutout: bool = False, front: bool = False,
                  progress=gr.Progress()):
    sources = _inputs(files, folder)
    out_dir = _stamped("prepped")
    force = {"Auto (only if needed)": None, "Always": True, "Never": False}[restore_mode]
    log: list[str] = []
    JOB.start()

    def report(msg: str):
        log.append(msg)
        progress((len(log), len(sources) * 2 + 1), desc=msg)

    try:
        reports = pipeline.preprocess_sources(
            sources, out_dir, target=target, force_restore=force, isolate=isolate,
            subject_prompt=subject_prompt or "character",
            exclude_prompt=exclude_prompt or "", restore_backend=restore_backend,
            isolation_backend=isolation_backend, tighten_crop=tighten,
            alpha_cutout=alpha_cutout, front=front, should_stop=JOB, progress=report)
    except OSError as e:
        raise gr.Error(f"Couldn't write to '{out_dir}': {e}. Check the output folder "
                       f"(valid drive, writable, enough space).") from e
    except Exception as e:  # whole-batch failure only — per-image ones are reported
        raise gr.Error(f"Preprocess failed: {e}") from e
    ok = [r for r in reports if r.output]
    failed = [r for r in reports if r.error]
    gallery = [(str(r.output), f"{r.source.name}: {r.reason}") for r in ok]
    note = _preprocess_note(reports, out_dir, alpha_cutout)
    if JOB.stopped:
        note += _stopped_note("image(s)", len(reports), len(sources),
                              "Re-run ① on the remaining sources to finish them.")
        gr.Warning(f"Stopped after {len(reports)} of {len(sources)} image(s).")
    if not ok:
        # Even a total failure returns normally. gr.Error would discard the
        # outputs — including the Log, which holds the per-image reasons that
        # are the whole point here — and paint every output component with an
        # "Error" placeholder, which reads as a broken app rather than a bad
        # input. The note says what happened; the toast points at it.
        with contextlib.suppress(OSError):
            out_dir.rmdir()  # don't litter an empty run folder
        gr.Warning(_plain(f"No image could be preprocessed — {len(failed)} failed. "
                          f"See the result note."))
        # No auto-fill: pointing ②/③ at an empty folder is worse than leaving
        # whatever the user already had in those fields.
        return gallery, note, "\n".join(log), gr.update(), gr.update()
    if failed:
        # A toast, not gr.Error: the finished images, the gallery and the
        # auto-filled folders all have to survive a partial failure.
        gr.Warning(f"Preprocessed {len(ok)} of {len(reports)} — "
                   f"{len(failed)} skipped, see the result note.")
    # Auto-fill downstream tabs (they can still be pointed anywhere else). ③ gets
    # the training copies; ② gets the isolated references when there are any.
    # Alpha cutouts aren't a drop-in reference for ② (see the checkbox's info).
    refs = [r.reference for r in ok if r.reference]
    if alpha_cutout and refs:
        gen_src = gr.update()
    else:
        gen_src = str(refs[0].parent if refs else out_dir)
    return gallery, note, "\n".join(log), gen_src, str(out_dir)


# ---------- ② generate & curate ----------

def do_generate(files: list[str], folder: str, plan_df: pd.DataFrame, engine: str,
                cloud_model: str, exclude_props: bool, isolate_angles: bool,
                isolation_backend: str, subject_prompt: str, exclude_prompt: str,
                front: bool, anchor: bool, gen_dir_prev: str, results_state,
                progress=gr.Progress()):
    sources = _inputs(files, folder)
    out_dir = _validate_out_dir(gen_dir_prev) if gen_dir_prev.strip() else _stamped("generated")
    shots = _df_to_shots(plan_df)
    log: list[str] = []
    JOB.start()

    def report(msg: str):
        log.append(msg)
        progress((len(log), len(shots) + 2), desc=msg)

    try:
        results = pipeline.generate_shots(
            sources, shots, engine, out_dir, cloud_model=cloud_model,
            isolate_angles=isolate_angles, subject_prompt=subject_prompt or "character",
            exclude_prompt=exclude_prompt, isolation_backend=isolation_backend,
            exclude_props=exclude_props, front=front, anchor=anchor, should_stop=JOB, progress=report)
    except OSError as e:
        raise gr.Error(f"Couldn't write to '{out_dir}': {e}. Check the output folder "
                       f"path (valid drive, no forbidden characters, writable).") from e
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}") from e
    rows, gallery, keep = _gen_gallery(results)
    if JOB.stopped:
        gr.Warning(f"Stopped after {len(results)} of {len(shots)} shot(s) — click "
                   f"'Generate/regenerate UNCHECKED shots' to finish the rest.")
    return results, rows, gallery, keep, "\n".join(log), str(out_dir), str(out_dir)


def do_regenerate(files: list[str], folder: str, plan_df: pd.DataFrame, engine: str,
                  cloud_model: str, exclude_props: bool, isolate_angles: bool,
                  isolation_backend: str, subject_prompt: str, exclude_prompt: str,
                  front: bool, anchor: bool, gen_dir: str, results_state, keep_ids: list[str],
                  progress=gr.Progress()):
    if not results_state:
        raise gr.Error("Nothing generated yet.")
    # The redo set comes from the PLAN, not from the previous results, so a shot
    # that was never generated (the run was stopped, or it failed) is included.
    # That makes this button the resume path after ⏹ Stop, not just a re-roll.
    plan_ids = [s.id for s in _df_to_shots(plan_df)]
    redo = {sid for sid in plan_ids if sid not in set(keep_ids or [])}
    if not redo:
        raise gr.Error("Every shot in the plan is checked — uncheck the ones you "
                       "want regenerated (or that never finished), then click again.")
    sources = _inputs(files, folder)
    log: list[str] = []
    JOB.start()
    try:
        results = pipeline.generate_shots(
            sources, _df_to_shots(plan_df), engine, Path(gen_dir), cloud_model=cloud_model,
            isolate_angles=isolate_angles, subject_prompt=subject_prompt or "character",
            exclude_prompt=exclude_prompt, isolation_backend=isolation_backend,
            exclude_props=exclude_props, front=front, anchor=anchor, existing=results_state,
            only_ids=redo, should_stop=JOB, progress=log.append)
    except OSError as e:
        raise gr.Error(f"Couldn't write to '{gen_dir}': {e}. Check the output folder "
                       f"path (valid drive, no forbidden characters, writable).") from e
    except Exception as e:
        raise gr.Error(f"Regeneration failed: {e}") from e
    # Regenerated shots are brand new, so a fresh full keep is the honest default.
    rows, gallery, keep = _gen_gallery(results)
    return results, rows, gallery, keep, "\n".join(log)


def do_refresh_disk(results_state, gen_dir: str, keep_ids: list[str]):
    """Re-sync with the output folder — files you deleted externally drop out.

    Re-syncing must not undo curation: shots you already rejected stay rejected.
    """
    if not results_state:
        raise gr.Error("No generation results in this session.")
    before = len(results_state)
    results = [r for r in results_state if r.path is None or r.path.exists()]
    rows, gallery, keep = _gen_gallery(results, selected=keep_ids)
    note = f"Re-synced with {gen_dir}: {before - len(results)} externally deleted shot(s) dropped."
    return results, rows, gallery, keep, note


def coverage_note(results_state, keep_ids: list[str], percent: float) -> str:
    """② coverage line for the kept shots (see `shotplan.coverage`)."""
    kept = set(keep_ids or [])
    shots = [r.shot for r in results_state or [] if r.path and r.shot.id in kept]
    return coverage(shots, (percent or 40) / 100)


def primary_reference(files: list[str], folder: str) -> str | None:
    """The reference ② leads with (it sets the local engine's output shape)."""
    try:
        return str(_inputs(files, folder)[0])
    except gr.Error:
        return None


def find_gen_duplicates(results_state, keep_ids: list[str], files: list[str],
                        folder: str, distance: float) -> str:
    """Near-duplicate kept shots, and shots that just copied a reference.

    Uses ④'s sensitivity slider so both scans agree on what "near" means.
    """
    from studio.dedupe import find_near_duplicate_groups

    kept = {r.path for r in results_state or []
            if r.path and r.path.exists() and r.shot.id in set(keep_ids or [])}
    if not kept:
        raise gr.Error("No kept shots to compare — generate first.")
    refs = [] if primary_reference(files, folder) is None else _inputs(files, folder)
    groups = find_near_duplicate_groups(refs + sorted(kept), max_distance=int(distance))
    copies = [g for g in groups if any(p in refs for p in g)]
    twins = [g for g in groups if g not in copies]
    if not groups:
        return f"🔁 No near-duplicates among {len(kept)} kept shot(s) (sensitivity {distance:g})."
    note = ""
    if copies:
        note += (f"🔁 **{len(copies)} shot group(s) look like a copy of a reference** — "
                 f"the generator ignored the prompt; regenerate them: {_groups_text(copies)}")
    if twins:
        note += ("\n\n" if note else "") + (
            f"🔁 **{len(twins)} near-duplicate group(s)** among kept shots — keep one of "
            f"each: {_groups_text(twins)}")
    return note


def send_kept_to_caption(results_state, keep_ids: list[str], gen_dir: str):
    """Load ②'s output folder into ③ with only the kept shots preselected.

    Also switches to ③ and echoes the count on ② — the note this returns lands on
    the ③ tab, which the user cannot see from ② (that silence was the whole bug).
    """
    if not gen_dir.strip():
        raise gr.Error("Nothing generated yet.")
    kept = {r.path.name for r in (results_state or [])
            if r.path and r.path.exists() and r.shot.id in set(keep_ids or [])}
    if not kept:
        raise gr.Error("No kept shots selected — check at least one shot first.")
    images = list_images(Path(gen_dir.strip()))
    names = [p.name for p in images]
    rows = _caption_rows(images)
    preselected = [n for n in names if n in kept]
    note = (f"{len(names)} image(s) loaded from ② — **{len(preselected)} kept shot(s) "
            f"preselected** for captioning. Click a thumbnail to toggle it.")
    sent = (f"➡ Sent **{len(preselected)} kept shot(s)** to ③ Caption "
            f"(from {gen_dir.strip()}).")
    return (_goto_tab("caption"), gen_dir, rows, _picker_gallery(rows, preselected),
            gr.CheckboxGroup(choices=names, value=preselected), note, sent)


# ---------- ③ caption ----------

def _caption_rows(images: list[Path]) -> list[tuple[str, str, str]]:
    """Picker rows for ③ — the checkbox value is the bare filename (③ is single-folder)."""
    return [(str(p), p.name,
             f"{p.name}{' ✓ captioned' if p.with_suffix('.txt').exists() else ''}")
            for p in images]


def load_caption_folder(folder: str, selected=None):
    """Load a folder into ③'s picker.

    `selected=None` selects everything (a fresh "Load folder"); pass a value to keep
    the user's existing pick — re-checking the whole batch after a run destroys the
    subset they deliberately chose.
    """
    images = list_images(Path(folder.strip())) if folder.strip() else []
    if not images:
        raise gr.Error(f"No images found in folder: {folder or '(empty)'}")
    captioned = {p.name for p in images if p.with_suffix(".txt").exists()}
    rows = _caption_rows(images)
    names = [p.name for p in images]
    picked = names if selected is None else _picker_order(rows, selected)
    note = (f"{len(images)} image(s) loaded, {len(captioned)} already have .txt "
            f"sidecars (re-captioning overwrites them). **{len(picked)} selected** — "
            f"click a thumbnail to toggle it.")
    return (rows, _picker_gallery(rows, picked),
            gr.CheckboxGroup(choices=names, value=picked), note)


def _resolve_captioner_config(captioner_key: str, gemini_model: str):
    """UI wrapper over the shared resolver — translates its error into gr.Error
    so the same logic serves the CLI without importing gradio."""
    try:
        return resolve_captioner_config(captioner_key, gemini_model)
    except CaptionerConfigError as e:
        raise gr.Error(str(e)) from e


def save_custom_captioner(base_url: str, model: str, api_key_env: str,
                          min_interval_s) -> str:
    from studio import user_config

    if not base_url.strip():
        raise gr.Error("Enter the endpoint base URL (e.g. https://openrouter.ai/api/v1).")
    user_config.set_custom_captioner(base_url, model, api_key_env, min_interval_s or 0)
    key_note = (f" Reads the API key from the `{api_key_env.strip()}` env var (set it in "
                f".env)." if api_key_env.strip() else " No API key configured.")
    return (f"✅ Saved custom endpoint: {base_url.strip().rstrip('/')} "
            f"(model: {model.strip() or 'server default'}).{key_note} "
            f"Select the 'Custom OpenAI-compatible endpoint' captioner to use it.")


def _tagger_overrides(captioner_key: str, spec_overrides, gen_thr, char_thr,
                      rating: bool = False, underscores: bool = False):
    """Merge the ③ Tag-options controls into spec_overrides (taggers only)."""
    return merge_tagger_overrides(
        captioner_key, spec_overrides, general_threshold=gen_thr,
        character_threshold=char_thr, include_rating=rating, keep_underscores=underscores)


def do_test_caption(folder: str, selected: list[str], captioner_key: str,
                    name: str, trigger: str, gemini_model: str, style: str,
                    gen_thr: float, char_thr: float, prefix: str, suffix: str,
                    blacklist: str, rating: bool, underscores: bool,
                    dataset_type: str = "character", sparse: bool = False,
                    identity: str = "identity"):
    if not folder.strip() or not selected:
        raise gr.Error("Load a folder and select at least one image first.")
    path = Path(folder.strip()) / selected[0]
    # A dedicated tagger always emits Danbooru/e621 tags, whatever the radio says.
    if CAPTIONERS_BY_KEY[captioner_key].backend == "wd_tagger":
        style = "tags"
    model_override, spec_overrides = _resolve_captioner_config(captioner_key, gemini_model)
    spec_overrides = _tagger_overrides(captioner_key, spec_overrides, gen_thr, char_thr,
                                       rating, underscores)
    cap = Captioner(captioner_key, model_override=model_override, spec_overrides=spec_overrides)
    try:
        raw = cap.caption(path, subject=name or "the character", style=style,
                          dataset_type=dataset_type, sparse=sparse, identity=identity)
    except Exception as e:
        raise gr.Error(str(e)) from e
    finally:
        cap.unload()
    caption = finalize_caption(raw, trigger, name, SUBJECT_ALIASES, style=style,
                               dataset_type=dataset_type)
    caption = drop_blacklisted_tags(caption, parse_blacklist(blacklist), style)
    return apply_affixes(caption, prefix, suffix, style)


def load_one_caption(folder: str, filename: str) -> str:
    """Read the .txt sidecar for a single image into the inline editor.

    Empty selection returns empty rather than raising: this also runs on the
    dropdown's `.change`, which fires with `None` whenever the folder is
    repopulated — an error toast there would be noise, not information.
    """
    if not folder.strip() or not filename:
        return ""
    # Forgiving read: the editor must be able to open (and so fix) a sidecar written
    # by another tool in cp1252, not raise at the user.
    return read_caption(Path(folder.strip()) / filename)


def save_one_caption(folder: str, filename: str, text: str) -> str:
    """Write the inline editor's text back to the image's .txt sidecar."""
    if not folder.strip() or not filename:
        raise gr.Error("Load a folder and pick an image first.")
    txt = (Path(folder.strip()) / filename).with_suffix(".txt")
    try:
        txt.write_text(text.strip(), encoding="utf-8")
    except OSError as e:
        raise gr.Error(f"Couldn't write {txt}: {e}. Check the file isn't read-only "
                       f"and the folder is still available.") from e
    return f"✅ Saved caption for {filename}"


def _editor_labels(folder: Path, names: list[str]) -> list[tuple[str, str]]:
    """(label, value) pairs — a trailing ✓ marks a file that already has a
    caption, so review progress is visible in the picker itself.

    The mark TRAILS on purpose: Gradio renders its own selection tick as the
    first child of each option (hidden on unselected rows), so a leading ✓ made
    the selected-and-captioned row read "✓ ✓ img01.png". The value stays the
    bare filename, so everything downstream is unchanged.
    """
    return [(f"{n} ✓" if read_caption(folder / n) else n, n) for n in names]


def _editor_choices(folder: str):
    """Repopulate the inline editor's picker. Returns the dropdown update and the
    ordered filename list, which Prev/Next use to resolve an index without
    re-scanning the folder on every click."""
    base = Path(folder.strip()) if folder.strip() else None
    names = [p.name for p in list_images(base)] if base else []
    value = names[0] if names else None
    return gr.Dropdown(choices=_editor_labels(base, names) if base else [],
                       value=value), names


def _editor_context(folder: str, filename: str, names: list[str]):
    """Preview image + 'n / N' position for the file being edited.

    Editing a caption with no sight of the image was the real gap — the dropdown
    told you a filename and nothing else.
    """
    if not folder.strip() or not filename or filename not in (names or []):
        return None, ""
    idx = names.index(filename)
    return (str(Path(folder.strip()) / filename),
            f"**{idx + 1} / {len(names)}** · `{filename}`")


def _editor_step(folder: str, filename: str, names: list[str], delta: int):
    """Move to the previous/next image. Clamps at the ends rather than wrapping —
    silently looping back to image 1 during a review pass loses your place."""
    if not names:
        raise gr.Error("Load a folder first (📂 Load folder).")
    # gr.update(), NOT gr.Dropdown(value=...): the constructor form merges over
    # the component's ORIGINAL constructor args, where `choices` was [] — so the
    # new value validates against an empty list, warns, and can be dropped,
    # leaving Prev/Next silently doing nothing.
    if filename not in names:
        return gr.update(value=names[0])
    idx = names.index(filename) + delta
    if idx < 0:
        gr.Info("Already at the first image.")
        return gr.skip()
    if idx >= len(names):
        gr.Info("That was the last image.")
        return gr.skip()
    return gr.update(value=names[idx])


def _editor_relabel(folder: str, filename: str, names: list[str], move: int = 0):
    """Refresh the picker's ✓ marks (a just-saved caption should show as done),
    optionally moving to another image at the same time.

    Choices and value are set together: doing them as two updates made the
    dropdown briefly hold a value that was not in its choices.
    """
    if not folder.strip() or not names:
        return gr.skip()
    base = Path(folder.strip())
    idx = names.index(filename) if filename in names else 0
    target = idx + move
    if not 0 <= target < len(names):
        if move:
            gr.Info("That was the last image." if move > 0
                    else "Already at the first image.")
        target = idx
    return gr.Dropdown(choices=_editor_labels(base, names), value=names[target])


def editor_prev(folder: str, filename: str, names: list[str]):
    return _editor_step(folder, filename, names, -1)


def editor_next(folder: str, filename: str, names: list[str]):
    return _editor_step(folder, filename, names, 1)


def save_and_next(folder: str, filename: str, text: str, names: list[str]):
    """The review-pass action: write this caption, then advance."""
    note = save_one_caption(folder, filename, text)
    return note, _editor_relabel(folder, filename, names, move=1)


def _merge_export_folders(existing: str, new_folder: str) -> str:
    """Add `new_folder` to ④ Export's folder list, keeping what's already there.

    Captioning the prepped sources and then the generated shots is the documented
    workflow, so this must accumulate — overwriting silently dropped the first
    folder from the export.
    """
    folders = [ln.strip() for ln in (existing or "").splitlines() if ln.strip()]
    if new_folder not in folders:
        folders.append(new_folder)
    return "\n".join(folders)


def _merge_carry(existing, new_paths: list[str]) -> list[str]:
    """Accumulate the set of images ③ has captioned, in first-seen order.

    ④ preselects this set. It accumulates for the same reason the folder list does:
    captioning the prepped sources and then the generated shots is the documented
    workflow, and both halves belong in the export.
    """
    carried = list(existing or [])
    seen = set(carried)
    for p in new_paths:
        if p not in seen:
            carried.append(p)
            seen.add(p)
    return carried


def do_caption(folder: str, selected: list[str], captioner_key: str,
               name: str, trigger: str, gemini_model: str, style: str,
               gen_thr: float, char_thr: float, prefix: str, suffix: str,
               blacklist: str, rating: bool, underscores: bool,
               skip_existing: bool, dataset_type: str, sparse: bool,
               exp_folders_prev: str, carry_prev, identity: str = "identity",
               progress=gr.Progress()):
    if not folder.strip() or not selected:
        raise gr.Error("Load a folder and select the images to caption first.")
    base = Path(folder.strip())
    images = [base / s for s in selected]
    model_override, spec_overrides = _resolve_captioner_config(captioner_key, gemini_model)
    spec_overrides = _tagger_overrides(captioner_key, spec_overrides, gen_thr, char_thr,
                                       rating, underscores)
    log: list[str] = []

    def report(msg: str):
        log.append(msg)
        progress((len(log), len(images) + 2), desc=msg)

    # Write each sidecar as it arrives instead of batching them to the end: a
    # mid-batch cloud failure used to discard every caption already paid for.
    written: list[Path] = []

    def persist(img: Path, caption: str) -> None:
        img.with_suffix(".txt").write_text(caption, encoding="utf-8")
        written.append(img)

    failure = ""
    JOB.start()
    try:
        caption_images(images, captioner_key, name, trigger, progress=report,
                       model_override=model_override, spec_overrides=spec_overrides,
                       style=style, prefix=prefix, suffix=suffix,
                       skip_existing=skip_existing, blacklist=blacklist,
                       dataset_type=dataset_type, sparse=sparse, on_item=persist,
                       should_stop=JOB, identity=identity)
    except Exception as e:
        if not written:
            raise gr.Error(f"Captioning failed: {friendly_api_error(e)}") from e
        # Partial success: keep the finished sidecars, keep the user's selection, and
        # say exactly how to resume. gr.Warning toasts without discarding the outputs
        # below, which gr.Error would.
        remaining = len(images) - len(written)
        failure = (f"\n\n⚠️ **Stopped after {len(written)} of {len(images)}** — "
                   f"{friendly_api_error(e)}\n\nThe {len(written)} caption(s) already "
                   f"written are saved and your selection is unchanged. Tick **Skip "
                   f"images that already have a caption** and click ③ again to do the "
                   f"remaining {remaining}.")
        gr.Warning(f"Captioned {len(written)} of {len(images)} before failing — "
                   f"finished captions were saved.")
    # Reload the folder but KEEP the user's pick; re-checking the whole batch after a
    # run silently discarded the subset they chose.
    rows, gallery, boxes, _ = load_caption_folder(folder, selected=selected)
    if JOB.stopped and not failure:
        failure = _stopped_note(
            "image(s)", len(written), len(images),
            "Tick **Skip images that already have a caption** and click ③ again "
            "to finish the rest.")
        gr.Warning(f"Stopped after {len(written)} of {len(images)} caption(s) — "
                   f"those are saved.")
    result = f"✅ Wrote {len(written)} caption sidecar(s) in {base}{failure}"
    # Auto-fill ④ Export: ADD this folder to its list (captioning several folders
    # in turn must accumulate). Name and trigger are NOT carried — ④ reads the
    # same two header boxes this stage did, so there is nothing to copy and
    # nothing that can go stale.
    folders = _merge_export_folders(exp_folders_prev, str(base))
    analysis = _caption_analysis(str(base), trigger)
    # Carry the captioned images forward so ④ preselects them instead of the folder.
    carry = _merge_carry(carry_prev, [str(p) for p in written])
    return (rows, gallery, boxes, result, "\n".join(log), folders, analysis, carry)


# ---------- ④ export ----------

def _caption_analysis(folder: str, trigger: str) -> str:
    """Markdown health-lint + tag-frequency report for a captioned folder."""
    from studio.caption_lint import analyze_folder, markdown_summary

    if not folder.strip():
        return ""
    try:
        report, ubiquitous = analyze_folder(Path(folder.strip()), trigger.strip())
    except Exception:
        return ""  # advisory — never break the caption flow
    return markdown_summary(report, ubiquitous)


def do_analyze_captions(folder: str, trigger: str) -> str:
    if not folder.strip():
        raise gr.Error("Load a folder first.")
    return _caption_analysis(folder, trigger) or "No captions found to analyze yet."


def _export_flag(img: Path) -> str:
    if not img.with_suffix(".txt").exists():
        return "⚠ no caption"
    return "✓" if read_caption(img) else "⚠ empty"


def _export_label(img: Path, flag: str) -> tuple[str, str]:
    """(checkbox label, value). Value is the full path so same-named files in
    different folders never collide; label is folder/name + caption flag."""
    return f"{img.parent.name}/{img.name} — {flag}", str(img)


def _export_preselect(images: list[Path], carry) -> tuple[list[str], str]:
    """Which images to check, and one sentence saying why.

    ③'s captioned set wins when it covers any of the listed images — exporting
    everything in the folder was silently re-adding shots the user had rejected two
    tabs earlier. Falls back to all so ④ still works standalone on a folder that
    never went through ③ (Design rule 1).
    """
    carried = set(carry or [])
    picked = [str(img) for img in images if str(img) in carried]
    if picked:
        return picked, (f"**{len(picked)} of {len(images)} preselected** — the images "
                        f"③ Caption just captioned")
    return [str(img) for img in images], "all checked"


def _export_scan(folders_text: str) -> tuple[list[Path], dict[Path, str], list[tuple[str, str, str]],
                                              list[tuple[str, str]]]:
    """Shared folder scan behind both the initial preview and a flags-only refresh.

    One read per image: the flag feeds the gallery caption, the checkbox label and
    the ready/empty/none counters, and recomputing it separately for each meant
    several disk reads per image.
    """
    folders = [Path(line.strip()) for line in folders_text.splitlines() if line.strip()]
    images = [img for folder in folders for img in list_images(folder)]
    flags = {img: _export_flag(img) for img in images}
    rows = [(str(img), str(img), f"{img.parent.name}/{img.name} — {flags[img]}")
            for img in images]
    choices = [_export_label(img, flags[img]) for img in images]
    return images, flags, rows, choices


def _export_counts(flags: dict[Path, str]) -> tuple[int, int, int]:
    ready = sum(1 for f in flags.values() if f == "✓")
    empty = sum(1 for f in flags.values() if f == "⚠ empty")
    none_ = sum(1 for f in flags.values() if f == "⚠ no caption")
    return ready, empty, none_


def refresh_export_preview(folders_text: str, current_rows, current_selected):
    """Re-flag an already-loaded ④ preview after an inline caption edit in ③.

    A caption edited in ③'s inline editor left ④ showing the stale ⚠/✓ flag until
    "Load & preview" was clicked again. This keeps EXACTLY the images the user had
    checked (unlike load_export_preview's carry-based preselect, which is only for
    the first load) and is a no-op — via gr.skip() on every output — if ④ hasn't
    been loaded yet (current_rows empty) or the edited folder isn't one it's showing.
    """
    if not current_rows:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    images, flags, rows, choices = _export_scan(folders_text)
    if not images:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    kept = set(current_selected or [])
    values = [v for _, v in choices if v in kept]
    ready, empty, none_ = _export_counts(flags)
    note = (f"**{len(images)} image(s)** — {ready} ready · {empty} empty caption · "
            f"{none_} no caption. Refreshed after a ③ caption edit — your selection "
            "is unchanged.")
    return rows, _picker_gallery(rows, values), gr.CheckboxGroup(choices=choices, value=values), note


def _groups_text(groups: list[list[Path]], limit: int = 5) -> str:
    shown = "; ".join("=".join(f"{p.parent.name}/{p.name}" for p in g) for g in groups[:limit])
    return shown + (f" (+{len(groups) - limit} more)" if len(groups) > limit else "")


def load_export_preview(folders_text: str, dup_distance: float = 5, carry=None):
    if not (folders_text or "").strip():
        raise gr.Error("Enter at least one folder of captioned images (one per line).")
    images, flags, rows, choices = _export_scan(folders_text)
    if not images:
        raise gr.Error("No images found in the listed folder(s).")
    values, why = _export_preselect(images, carry)
    ready, empty, none_ = _export_counts(flags)
    note = (f"**{len(images)} image(s)** — {ready} ready · {empty} empty caption · "
            f"{none_} no caption. Below: {why} — **click a thumbnail to toggle it**. "
            "Only checked images are exported; a checked image without a usable "
            "caption is skipped and called out in the result.")
    try:  # advisory near-duplicate scan — never blocks the preview
        from studio.dedupe import find_bursts, find_near_duplicate_groups

        groups = find_near_duplicate_groups(images, max_distance=int(dup_distance))
        if groups:
            note += (f"\n\n🔁 **{len(groups)} near-duplicate group(s)** — consider "
                     f"unchecking extras so one shot isn't over-weighted: "
                     f"{_groups_text(groups)}")
        bursts = find_bursts(images)
        if bursts:
            note += (f"\n\n📸 **{len(bursts)} burst(s)** — photos taken seconds apart "
                     f"are one moment; keep the best of each: {_groups_text(bursts)}")
    except Exception:
        pass
    try:  # advisory caption health + tag frequency — never blocks the preview
        from studio.caption_lint import analyze_pairs, markdown_summary

        cap_pairs = []
        for img in images:
            if c := read_caption(img):
                cap_pairs.append((f"{img.parent.name}/{img.name}", c))
        # trigger unknown at preview -> skip the missing-trigger check; empties are
        # already summarized above, so only short/duplicate/ubiquitous add value.
        report, ubiquitous = analyze_pairs(cap_pairs, trigger="")
        if not report.clean or ubiquitous:
            note += "\n\n" + markdown_summary(report, ubiquitous)
    except Exception:
        pass
    return (rows, _picker_gallery(rows, values),
            gr.CheckboxGroup(choices=choices, value=values), note)


def do_export(selected: list[str], name: str, trigger: str, output_root: str,
              make_zip: bool = False, dataset_type: str = "character",
              style_key: str = shot_style.MATCH, style_text: str = "",
              ilb_handoff: bool = False, holdout=0, identity: str = "identity"):
    if not selected:
        raise gr.Error("Click '📂 Load & preview', then keep at least one image checked.")
    from studio.package import package_dataset, resolve_export_items

    paths = [Path(s) for s in selected]
    res = resolve_export_items(paths)
    if not res.items:
        raise gr.Error("None of the checked images have a usable caption — "
                       "run ③ Caption first (each export needs a non-empty .txt).")
    source_folders = sorted({str(p.parent) for p in paths})
    style = shot_style.resolve(style_key, style_text)
    metadata = {"character_name": name, "trigger": trigger,
                "dataset_type": dataset_type,
                # Recorded so ⑤ can name the right medium in its sample prompt
                # even when pointed at this folder in a later session.
                "shot_style": style.key,
                "shot_style_text": style_text if style.key == shot_style.CUSTOM else "",
                "source_folders": source_folders,
                "skipped_uncaptioned": res.missing,
                "skipped_empty_caption": res.empties}
    if dataset_type == "character":
        metadata["identity"] = identity
    out_root = _validate_out_dir(output_root)
    try:
        ds = package_dataset(res.items, out_root, name, trigger, metadata,
                             holdout=int(holdout or 0))
    except OSError as e:
        raise gr.Error(f"Couldn't write the dataset to '{out_root}': {e}. Check the "
                       f"output folder path (valid drive, no forbidden characters, writable).") from e
    # Show the first numbered caption (README.txt is excluded).
    caption_files = sorted(p for p in ds.glob("*.txt") if p.name != "README.txt")
    samples = [(p.name, read_caption(p)) for p in caption_files]
    first = next(((n, t) for n, t in samples if t), None)
    sample_block = f"\n\nSample caption ({first[0]}):\n{first[1]}" if first else ""
    # Say "of the N you checked" explicitly. The bare "Skipped (no caption): x.png"
    # read as a complete skip list, so a user who had unchecked 8 images wondered why
    # only one was named — unchecked images were never candidates and never listed.
    checked = len(paths)
    skipped = (f"\n⚠️ Skipped {len(res.missing)} of the {checked} image(s) you checked "
               f"— no caption sidecar: {', '.join(res.missing)}"
               if res.missing else "")
    empty_note = (f"\n⚠️ Skipped {len(res.empties)} of the {checked} image(s) you checked "
                  f"— caption file is empty: {', '.join(res.empties)}"
                  if res.empties else "")
    zip_note = ""
    if make_zip:
        from studio.package import zip_dataset

        try:
            zip_note = f"\n🗜️ Zipped: {zip_dataset(ds)}"
        except OSError as e:
            zip_note = f"\n⚠️ Could not write the .zip: {e}"
    ilb_note = ""
    if ilb_handoff:
        from studio.handoff import prepare_handoff

        # prepare_handoff never raises: the dataset is already written by now,
        # so a sidecar problem is a note, not a failed export.
        ilb_note = f"\n{prepare_handoff(ds)}"
    # State the identity this dataset was stamped with. A user exported a whole
    # dataset under a name and trigger left over from several runs earlier and
    # only found out at the end — the fields are now kept in step, and this is
    # the receipt that makes a wrong one visible before training on it.
    identity = (f"\n🏷️ Name: {name.strip() or '(none)'}  ·  Trigger: "
                f"{trigger.strip() or '(none)'}")
    if not trigger.strip():
        identity += ("\n⚠️ No trigger word — captions have nothing to teach the LoRA "
                     "to respond to. Set one at the top and re-caption if that wasn't "
                     "deliberate.")
    held_dir = ds.parent / f"{ds.name}-heldout"
    held = list_images(held_dir)
    held_note = (f"\n🔒 Held out {len(held)} photo(s) as a likeness test, never trained "
                 f"on: {held_dir}" if held else "")
    if int(holdout or 0) and not held:
        held_note = ("\n⚠️ Nothing held out — only ①'s photos qualify, and none of the "
                     "checked images came from ①.")
    result = (f"✅ Dataset ready: {ds}  ({len(res.items) - len(held)} image/caption pairs "
              f"from the {checked} image(s) you checked)"
              f"{identity}{held_note}{skipped}{empty_note}{zip_note}{ilb_note}{sample_block}")
    # ds path auto-fills the ⑤ Train tab AND the HF-publish box below.
    return result, str(ds), str(ds)


def send_captioned_to_export(folders_text: str, dup_distance: float, carry):
    """③ → ④ hand-off: load the export preview, preselect what ③ captioned, switch tab."""
    if not (folders_text or "").strip():
        raise gr.Error("Caption a folder first (③) — nothing has been sent to ④ yet.")
    rows, gallery, boxes, note = load_export_preview(folders_text, dup_distance, carry)
    return _goto_tab("export"), rows, gallery, boxes, note


def do_publish_hf(ds_dir: str, repo_id: str, private: bool, progress=gr.Progress()):
    """Publish an exported dataset folder to the Hugging Face Hub (opt-in)."""
    from studio.hf_publish import HFPublishError, publish_dataset

    if not (ds_dir or "").strip():
        raise gr.Error("Export a dataset first (④) — then publish the folder it created.")
    log: list[str] = []

    def report(msg: str):
        log.append(msg)
        progress((len(log), 3), desc=msg)

    try:
        url = publish_dataset(ds_dir.strip(), repo_id, private=bool(private), progress=report)
    except HFPublishError as e:
        raise gr.Error(str(e)) from e
    except Exception as e:  # network/auth/etc. — surface, never crash the UI
        raise gr.Error(f"Publishing failed: {e}") from e
    vis = "private" if private else "PUBLIC"
    return f"✅ Published ({vis}): [{url}]({url})"


# ---------- misc ----------


def refresh_plan(name: str, dataset_type: str = "character",
                 style_key: str = shot_style.MATCH,
                 style_text: str = "", identity: str = "identity") -> pd.DataFrame:
    return _plan_table(dataset_type, name, style_key, style_text, identity)


def do_save_plan(plan_df: pd.DataFrame, plan_name: str) -> str:
    from studio.plan_io import save_plan

    shots = _df_to_shots(plan_df)
    if not shots:
        raise gr.Error("The plan is empty — nothing to save.")
    path = settings.shot_plans_dir / (plan_name.strip() or "my-plan")
    try:
        saved = save_plan(shots, path)
    except OSError as e:
        raise gr.Error(f"Couldn't save the plan to {path}: {e}") from e
    return f"✅ Saved {len(shots)} shots to {saved}"


def do_load_plan(plan_name: str):
    from studio.plan_io import load_plan

    name = plan_name.strip()
    if not name:
        raise gr.Error("Enter the name of a saved plan to load.")
    path = settings.shot_plans_dir / name
    if not path.suffix:
        path = path.with_suffix(".yaml")
    if not path.exists():
        raise gr.Error(f"No plan file at {path}")
    try:
        shots = load_plan(path)
    except Exception as e:
        # A hand-edited YAML plan is a user file: a bad key or bad indentation
        # must name the file, not dump a pydantic/yaml traceback into the UI.
        raise gr.Error(f"Couldn't read the plan at {path}: {e}. Check the YAML — "
                       f"each shot needs at least an `id`, `kind`, `local_prompt` "
                       f"and `cloud_prompt`.") from e
    note = f"✅ Loaded {len(shots)} shots from {path}"
    # Plans saved before 0.17.0 carry the old engine's `<sks>` LoRA grammar, which
    # Qwen-Image 2.1 reads as literal text and renders badly.
    stale = sum("<sks>" in s.local_prompt for s in shots)
    if stale:
        note += (f" ⚠️ {stale} shot(s) still use the old Qwen-Image-Edit `<sks>` prompts, "
                 f"which the local Qwen-Image 2.1 engine doesn't understand — rebuild the "
                 f"plan from the dataset type / shot style, or rewrite those local prompts.")
    return _shots_to_df(shots), note


def estimate_cost(engine: str, cloud_model: str, df: pd.DataFrame) -> str:
    n = len(df)
    if engine == "gemini":
        model, price = image_price(cloud_model)
        if price is None:
            return f"**Cost:** {n} images on `{model}` (price unknown — billed to your API key)"
        return (f"**Cost:** ~${n * price:.2f} for {n} images on `{model}` "
                f"(estimate at build time — billed to your own Google API key)")
    return f"**Cost:** {n} images, $0 (local generation)"

# ---------- ⑤ train (configs) ----------

def _preset(trainer: str, model_key: str):
    for p in TRAINER_MODELS[trainer]:
        if p.key == model_key:
            return p
    return TRAINER_MODELS[trainer][0]


def _model_dropdown(trainer: str):
    presets = TRAINER_MODELS[trainer]
    return gr.Dropdown(choices=[(p.label, p.key) for p in presets], value=presets[0].key)


def on_trainer_change(trainer: str):
    from studio import user_config

    p = TRAINER_MODELS[trainer][0]
    return (_model_dropdown(trainer), user_config.get_trainer_path(trainer),
            p.resolution, p.rank, p.alpha, p.epochs, p.lr, p.batch_size,
            gr.update(choices=optimizer_choices(trainer), value="adamw8bit"))


def on_model_change(trainer: str, model_key: str):
    p = _preset(trainer, model_key)
    return p.resolution, p.rank, p.alpha, p.epochs, p.lr, p.batch_size


def _repeats(trainer: str, stats, epochs, batch_size, repeats) -> int:
    """The typed repeats, or (0 = auto) the count that reaches the target steps."""
    if int(repeats or 0) > 0:
        return int(repeats)
    if trainer == "fizgig":  # Fizgig's own guidance: repeats 1, more epochs
        return 1
    return stats.suggested_repeats(int(epochs or 1), int(batch_size or 1))


def exposure_line(trainer: str, dataset_dir: str, epochs, repeats, batch_size) -> str:
    """Live "how much training is this" line under ⑤'s epochs/repeats fields."""
    from studio.dataset_stats import inspect
    from studio.trainer_configs import exposure

    ds = Path(dataset_dir.strip()) if dataset_dir.strip() else None
    if not ds or not ds.is_dir():
        return ""
    stats = inspect(ds)
    if not stats.n_images:
        return ""
    reps = _repeats(trainer, stats, epochs, batch_size, repeats)
    batch, n_epochs = max(int(batch_size or 1), 1), max(int(epochs or 1), 1)
    per_epoch, total = exposure(stats.n_images, reps, batch, n_epochs)
    auto = " (auto)" if not int(repeats or 0) else ""
    line = (f"**Exposure:** {stats.n_images} images × {reps} repeats{auto} ÷ batch {batch} "
            f"= {per_epoch} steps/epoch × {n_epochs} epochs = **{total} steps**, saving "
            f"{n_epochs} checkpoints with samples.")
    # Fizgig's adaptive LR is tuned for small sets at repeats 1, so few steps is its norm.
    if total < 400 and trainer != "fizgig":
        line += " ⚠️ Few steps — likely undertrained; raise epochs or repeats."
    elif total > 6000:
        line += " ⚠️ Many steps — slow, and late epochs will likely overfit."
    return line


def save_trainer_path(trainer: str, path: str) -> str:
    from studio import user_config

    user_config.set_trainer_path(trainer, path.strip())
    return f"✅ Saved {trainer} install path: {path.strip() or '(cleared)'}"


def dataset_type_note(ds: Path, selected_type: str) -> str:
    """Advisory line reconciling a dataset's recorded type with the header pick.

    ④ writes `dataset_type` into `metadata.json`; the header selector resets to
    the remembered type, which may not be the type of the folder you point ⑤ at.
    The type drives the sample prompt, so a silent mismatch is worth surfacing.
    Never blocks, and stays quiet for datasets with no/older metadata.
    """
    import json

    meta_file = ds / "metadata.json"
    if not meta_file.is_file():
        return ""
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(meta, dict):
        return ""
    recorded = str(meta.get("dataset_type") or "")
    style = str(meta.get("caption_style") or "")
    if not recorded:
        return ""
    line = f"\n\nRecorded in `metadata.json`: **{recorded}** dataset"
    line += f", **{style}** captions." if style else "."
    if recorded != selected_type:
        line += (f" ⚠️ The header **Dataset type** is set to **{selected_type}** — the "
                 f"sample prompt in the generated config follows the header. Switch it to "
                 f"**{recorded}** if this is that dataset.")
    return line


def inspect_dataset(dataset_dir: str, dataset_type: str = "character") -> str:
    """Summarize the dataset: count, sizes, captions and the target step count."""
    from studio.dataset_stats import inspect

    if not dataset_dir.strip():
        return ""
    ds = Path(dataset_dir.strip())
    if not ds.is_dir():
        return f"⚠️ Folder not found: {ds}"
    try:
        stats = inspect(ds)
    except Exception as e:  # unreadable/corrupt image headers — report, never crash
        return f"⚠️ Couldn't inspect {ds}: {e}"
    if not stats.n_images:
        return f"⚠️ No images in {ds}"
    return stats.summary() + dataset_type_note(ds, dataset_type)


def do_generate_train_config(trainer: str, model_key: str, dataset_dir: str,
                             install_path: str, name: str, trigger: str,
                             resolution, rank, alpha, epochs, lr, batch_size,
                             multi_res: bool, dataset_type: str = "character",
                             style_key: str = shot_style.MATCH,
                             style_text: str = "", project_name: str = "",
                             repeats=0, optimizer: str = "adamw8bit") -> str:
    # ⑤'s "LoRA name" is the trained file's name, not the subject's — the one
    # identity-ish field that is legitimately its own (people want "-v2"). Blank
    # means "follow the header name", which is why it is not auto-filled: an
    # auto-filled box is exactly what went stale before. The fallback is
    # slugified because this becomes a filename ("Sy Snootles" → "sy-snootles");
    # a name typed here is used verbatim, because it was chosen.
    if not name.strip():
        from studio.package import slugify

        name = slugify(project_name) if project_name.strip() else ""
    if not dataset_dir.strip():
        raise gr.Error("Enter the dataset folder to write the config into "
                       "(④ Export produces one and auto-fills this).")
    ds = Path(dataset_dir.strip())
    if not ds.is_dir():
        raise gr.Error(f"Dataset folder not found: {ds}")
    from studio import user_config
    from studio.dataset_stats import inspect
    from studio.trainer_configs import TrainConfig, write_configs

    try:
        stats = inspect(ds)
    except Exception as e:
        raise gr.Error(f"Couldn't read the dataset at {ds}: {e}") from e
    if not stats.n_images:
        raise gr.Error(f"No images found in {ds} — export a dataset first (④).")
    preset = _preset(trainer, model_key)
    buckets = stats.buckets_for(int(resolution)) if multi_res else []
    num_repeats = _repeats(trainer, stats, epochs, batch_size, repeats)
    if optimizer not in {k for _, k in optimizer_choices(trainer)}:
        optimizer = "adamw8bit"
    cfg = TrainConfig(
        trainer=trainer, model=preset, dataset_dir=ds,
        trigger=trigger.strip(), name=(name.strip() or "lora"),
        dataset_type=dataset_type,
        resolution=int(resolution), rank=int(rank), alpha=int(alpha),
        epochs=max(1, int(epochs)), num_repeats=num_repeats, n_images=stats.n_images,
        lr=float(lr), optimizer=optimizer, batch_size=int(batch_size),
        buckets=buckets, shot_style=style_key, shot_style_text=style_text)
    try:
        written, command = write_configs(cfg, install_path.strip())
    except OSError as e:
        raise gr.Error(f"Couldn't write the config into {ds}: {e}. Check the dataset "
                       f"folder is writable.") from e
    user_config.set_last_train_settings({
        "trainer": trainer, "model": model_key, "resolution": int(resolution),
        "rank": int(rank), "alpha": int(alpha), "epochs": cfg.epochs,
        "repeats": int(repeats or 0), "lr": float(lr), "optimizer": optimizer,
        "batch_size": int(batch_size)})
    files = "\n".join(str(p) for p in written)
    bucket_note = (f"\nBuckets: {buckets} (from the dataset's actual sizes)"
                   if buckets else f"\nSingle bucket at {int(resolution)}px")
    bucket_note += stats.upscale_note(int(resolution))
    exposure = exposure_line(trainer, str(ds), cfg.epochs, num_repeats, batch_size)
    caveat = ("\n\n📋 validation/validation.md says how to pick the best epoch from "
              "the per-epoch samples.")
    if optimizer == "prodigy":
        caveat += ("\n\nProdigy finds its own step size, so the config uses learning rate "
                   "1.0 and the Learning rate field is ignored.")
        if trainer in ("musubi", "kohya"):
            caveat += (f" {trainer} doesn't install Prodigy: run `pip install prodigyopt` in "
                       "its environment first.")
    if trainer in ("musubi", "fizgig"):
        caveat += (f"\n\n⚠️ {trainer} needs your local model paths — fill the "
                   "<<FILL: …>> placeholders in the command before running.")
    elif trainer == "kohya":
        caveat += ("\n\n⚠️ kohya sd-scripts: SDXL base runs from the HF id shown; for a "
                  "Pony / Illustrious / NoobAI checkpoint, replace the <<FILL>> pretrained "
                  "path. Verify flags against the sd-scripts docs before a long run.")
    # Advisory ④→⑤ sanity check: do the dataset's captions fit this base model?
    from studio.caption_lint import folder_caption_kind
    from studio.trainer_configs import caption_mismatch_warning

    mismatch = caption_mismatch_warning(preset, folder_caption_kind(ds))
    if mismatch:
        caveat += f"\n\n{mismatch}"
    return (f"✅ Wrote:\n{files}\n\nDataset: {stats.n_images} images, "
            f"{stats.min_long_side}-{stats.max_long_side}px long side{bucket_note}\n"
            f"{exposure.replace('**', '')}\n\n"
            f"Run it with:\n{command}{caveat}\n\n"
            f"⚠️ Configs are generated, not test-trained — verify keys against your "
            f"trainer's own docs before a long run.")


def refresh_cloud_models(current: str = "", force: bool = False):
    """New choices from the API; keeps the user's pick when it is still offered."""
    from studio.engines.gemini import list_image_models

    try:
        choices = image_model_choices(list_image_models(force_refresh=force))
    except Exception as e:
        raise gr.Error(f"Could not list models: {e}") from e
    keep = current if current in {m for _, m in choices} else AUTO_MODEL
    return gr.Dropdown(choices=choices, value=keep)


def refresh_caption_models():
    """Live-pull the current Gemini caption model list (Caption tab)."""
    from studio.engines.gemini import list_caption_models

    try:
        models = list_caption_models(force_refresh=True)
    except Exception as e:
        raise gr.Error(f"Could not list caption models: {e}") from e
    value = _DEFAULT_CAPTION_MODEL
    ids = [m[1] for m in models]
    if value not in ids and ids:
        value = ids[0]
    return gr.Dropdown(choices=models, value=value)


def run_doctor() -> str:
    """The `cli.py doctor` self-check, surfaced in the UI.

    Renders the CLI report verbatim inside a code fence rather than re-formatting
    it: one implementation, so the UI can never drift from what `doctor` says, and
    its deliberately ASCII output needs no escaping. `render()` masks every key.
    """
    from studio.doctor import build_report, render

    try:
        report = build_report()
    except Exception as e:  # a self-check that crashes is worse than none
        return f"⚠️ The setup check itself failed: {e}"
    verdict = ("✅ **Ready.**" if report.ok else
               "❌ **Something needs fixing** — see the FAIL line below.")
    return f"{verdict}\n\n```\n{render(report)}\n```"


def _check_for_update():
    """Best-effort GitHub release check on UI load; silently shows nothing on
    any failure (offline, rate-limited, disabled) so it can never block launch."""
    from studio.update_check import update_banner_markdown

    try:
        text = update_banner_markdown()
    except Exception:
        text = ""
    return gr.Markdown(value=text, visible=bool(text))

# ---------- ⚡ Quick build ----------

QUICK_ENGINES = [("Cloud — Gemini, billed to your Google key", "gemini"),
                 ("Local — ComfyUI, free", "comfyui")]


def quick_trigger(name: str, trigger: str) -> str:
    """The header trigger, or one made from the name ("Sy Snootles" → "sysnootles")."""
    from studio.package import slugify

    return trigger.strip() or slugify(name).replace("-", "")


def quick_engine_note(engine: str, dataset_type: str) -> str:
    """What Build will cost (cloud) or whether it can run at all (local)."""
    if dataset_type == "style":
        return "Style datasets generate nothing: Build prepares and Finish captions your own images."
    n = len(plan_for_type(dataset_type, ""))
    if engine == "gemini":
        if not settings.resolved_gemini_key():
            return ("⚠️ No Gemini key yet: run `python cli.py keys --set GEMINI_API_KEY` "
                    "and restart, or pick Local.")
        model, price = image_price(AUTO_MODEL)
        cost = f"~${n * price:.2f}" if price else "price unknown"
        return f"Auto → `{model}`: {cost} for {n} shots (estimate, billed to your Google key)."
    from studio.doctor import check_comfyui, check_comfyui_models

    up = check_comfyui()
    if up.warn:
        return f"⚠️ ComfyUI: {up.detail}"
    models = check_comfyui_models()
    if models and not models.ok:
        return f"⚠️ {models.detail}"
    return f"✅ ComfyUI is ready: {n} shots, $0."


def _quick_rows(start: pipeline.BuildStart) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Picker rows (your photos first, then the shots) and what starts ticked:
    everything except a photo ① enlarged a lot — a LoRA learns upscaling blur as
    the subject's texture. That photo still served as a reference."""
    rows: list[tuple[str, str, str]] = []
    keep: list[str] = []
    for r in start.reports:
        if not r.output:
            continue
        label = f"📷 {r.source.name}"
        factor = pipeline.upscale_factor(r)
        if factor > pipeline.UPSCALE_UNTICK:
            label += f"  ⚠ upscaled ×{factor:.1f} from {max(r.original_size)}px"
        else:
            keep.append(str(r.output))
        rows.append((str(r.output), str(r.output), _flagged(r.output, label)))
    for g in start.results:
        if g.path and g.path.exists():
            rows.append((str(g.path), str(g.path), _flagged(g.path, g.shot.id)))
            keep.append(str(g.path))
    return rows, keep


def _quick_note(start: pipeline.BuildStart) -> str:
    note = _preprocess_note(start.reports, start.run_dir / "prepped", alpha_cutout=False)
    failed = [g for g in start.results if not g.path]
    if failed:
        note += ("\n\n❌ **Failed shots** (the rest carried on; the ② tab can regenerate "
                 "them):\n" + "\n".join(f"- `{g.shot.id}`: {g.error}" for g in failed))
    big = [r.source.name for r in start.reports
           if r.output and pipeline.upscale_factor(r) > pipeline.UPSCALE_UNTICK]
    if big:
        note += (f"\n\n<sub>Unticked: {', '.join(big)}. ① enlarged them, and a LoRA "
                 f"learns the upscaling blur as texture. They were still used as "
                 f"references; tick them to train on them anyway.</sub>")
    if start.results:
        note += "\n\nUntick anything that looks wrong, then press **Finish**."
    return note


def quick_review(start, keep: list[str]) -> str:
    """Coverage of the kept shots and any near-duplicates among everything kept."""
    if not start:
        return ""
    kept = set(keep or [])
    shots = [g.shot for g in start.results if g.path and str(g.path) in kept]
    note = coverage(shots) if shots else ""
    try:
        from studio.dedupe import find_near_duplicate_groups

        groups = find_near_duplicate_groups(sorted(Path(k) for k in kept), max_distance=5)
    except Exception:
        groups = []  # advisory
    if groups:
        note += ("\n\n" if note else "") + (
            f"🔁 **{len(groups)} near-duplicate group(s)**, keep one of each: "
            f"{_groups_text(groups)}")
    return note


def do_quick_build(files: list[str], name: str, trigger: str, dataset_type: str,
                   identity: str, engine: str, held: str, front: bool,
                   progress=gr.Progress()):
    if not files:
        raise gr.Error("Drop one or more images of your subject first.")
    if not name.strip():
        raise gr.Error("Type a name at the top first: it names the dataset and goes into "
                       "every prompt.")
    trigger = quick_trigger(name, trigger)
    log: list[str] = []
    total = len(files) + (0 if dataset_type == "style" else
                          len(plan_for_type(dataset_type, ""))) + 2

    def report(msg: str):
        log.append(msg)
        progress((min(len(log), total), total), desc=msg)

    JOB.start()
    try:
        start = pipeline.build_start(
            [Path(f) for f in files], name.strip(), engine, dataset_type=dataset_type,
            exclude_prompt=held.strip(), identity=identity, front=front, should_stop=JOB,
            progress=report)
    except Exception as e:
        raise gr.Error(f"Build failed: {e}") from e
    rows, keep = _quick_rows(start)
    boxes = gr.CheckboxGroup(choices=[(Path(v).name, v) for _, v, _ in rows], value=keep)
    return (start, rows, _picker_gallery(rows, keep), boxes, _quick_note(start),
            "\n".join(log), trigger)


def do_quick_finish(start, keep: list[str], name: str, trigger: str, dataset_type: str,
                    identity: str, engine: str, captioner: str, trainer: str,
                    model_key: str, progress=gr.Progress()):
    if not start:
        raise gr.Error("Press Build first.")
    images = [Path(k) for k in keep or [] if Path(k).exists()]
    if not images:
        raise gr.Error("Tick at least one image to keep.")
    trigger = quick_trigger(name, trigger)
    log: list[str] = []

    def report(msg: str):
        log.append(msg)
        progress((min(len(log), len(images) + 2), len(images) + 2), desc=msg)

    # A tag-trained base model (SDXL family) wants tag captions.
    style = "tags" if _preset(trainer, model_key).expects_tags else "prose"
    try:
        ds, command = pipeline.build_finish(
            start, images, name=name.strip(), trigger=trigger, captioner=captioner,
            output_root=settings.output_root, dataset_type=dataset_type, identity=identity,
            engine_key=engine, caption_style=style, trainer=trainer, model_key=model_key,
            progress=report)
    except Exception as e:
        raise gr.Error(f"Finish failed: {e}") from e
    note = (f"✅ **Dataset ready:** `{ds}` ({len(images)} images, trigger `{trigger}`).\n\n"
            f"Train with:\n```\n{command}\n```\n`validation/validation.md` in the dataset "
            f"says how to pick the best epoch.\n\n<sub>Configs are generated, not "
            f"test-trained. Fill any `<<FILL>>` placeholders first; ⑤ has every setting.</sub>")
    return note, "\n".join(log), str(ds)


# ---------- layout ----------

# Gradio 5 warns that `head=` moves to `launch()` in Gradio 6 — but `launch()` does not
# accept it in 5.x, and `requirements.txt` pins `gradio<6` on purpose. Suppressed
# narrowly so a start-up console that should be empty stays empty; the move is recorded
# against lifting the gradio<6 cap in docs/ARCHITECTURE.md.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*'head' parameter in the Blocks.*",
                            category=DeprecationWarning)
    _blocks = gr.Blocks(title="Dataset Deviser", head=_PICKER_SCRIPT)

with _blocks as demo:
    gr.Markdown(
        "# Dataset Deviser\n"
        "Character, style, or concept → ready-to-train LoRA dataset. Every tab works standalone "
        "on any folder — or run them in order and each step auto-fills the next: "
        "**① Preprocess → ② Generate & curate → ③ Caption → ④ Export → ⑤ Train config**. "
        "Pick the **Dataset type** below (Character and Concept generate a shot set in ②; "
        "Style brings its own images and starts at ③)."
    )
    gr.Markdown(
        "> ⚠️ **Cloud options cost money and you are responsible for what you make.** "
        "Gemini image generation and Gemini captioning are **billed by Google to your own "
        "API key**; any custom endpoint you add is billed to you by that provider. You are "
        "solely responsible for the images you upload and the content you generate, caption, "
        "or send to third-party services — make sure you have the rights to your sources and "
        "comply with each provider's policies and the law. See **Costs & your responsibility** below."
    )
    update_notice = gr.Markdown(visible=False)
    with gr.Accordion("🩺 Check my setup (Python, ComfyUI, models, API keys)",
                      open=False):
        gr.Markdown(
            "Runs the same check as `python cli.py doctor`: Python version, "
            "required packages, torch/onnxruntime, whether ComfyUI is reachable "
            "(and whether every configured model filename actually exists on it), "
            "and which API keys are set — **masked**, never shown in full — with "
            "what each missing one blocks. Nothing is sent anywhere; the only "
            "network call is to your own ComfyUI.")
        btn_doctor = gr.Button("🩺 Run setup check", size="sm")
        doctor_out = gr.Markdown()
    with gr.Accordion("💲 Costs & your responsibility (read me)", open=False):
        gr.Markdown(
            "**Costs**\n"
            "- **Local options are free** (your GPU/CPU): ComfyUI generation, built-in SAM3 "
            "isolation, local `transformers` captioners, LM Studio/Ollama.\n"
            "- **Gemini image generation** (② Generate, Cloud engine) and **Gemini captioning** "
            "(③ Caption, Gemini captioner) are **billed by Google to the API key you provide**. "
            "In-app prices are build-time estimates — always check current Google pricing.\n"
            "- **Groq** captioning uses its free tier (rate-limited). **Custom OpenAI-compatible "
            "endpoints** you add are billed to you by whoever runs them (OpenRouter, etc.).\n"
            "- This tool never bills you and takes no cut — all charges are between you and the "
            "provider whose key you supply.\n\n"
            "**Your responsibility**\n"
            "- You are **solely responsible** for the source images you supply and for everything "
            "you generate, caption, export, or transmit with this tool.\n"
            "- Only use images you have the rights to. Respect each model/provider's acceptable-use "
            "policy and all applicable laws when generating or sending content.\n"
            "- This software is provided under the MIT License **with no warranty**; the authors are "
            "not liable for your use of it, for provider charges, or for content you create with it."
        )
    # Seeded from the last session (a user usually builds several datasets of the
    # same type); demo.load re-applies the dependent defaults below on launch.
    dataset_type = gr.Radio(
        DATASET_TYPE_CHOICES, value=_uc_boot.get_dataset_type(), label="Dataset type",
        info="What the LoRA learns, remembered between launches. Character and "
             "Concept generate a shot set in ②; Style brings its own images and starts "
             "at ③ Caption. Tunes caption framing, the ② shot plan, the ① isolation "
             "default, and the ⑤ sample prompt.")
    identity_policy = gr.Radio(
        [("Identity only — outfits vary", "identity"),
         ("Signature costume — the outfit is part of the character", "costume")],
        value="identity", label="Identity policy",
        info="Identity only dresses ②'s angle/pose shots in varied outfits and has ③ "
             "describe the clothing, so the trigger learns the person. Signature "
             "costume keeps the reference's outfit and leaves it out of captions, so "
             "the trigger carries it.")
    # The ONE place that owns "who is this dataset about". ②, ③, ④ and ⑤ all
    # read these two boxes directly instead of each keeping its own copy — see
    # the note above `type_outputs` for why copies were removed rather than kept
    # in sync.
    with gr.Row():
        project_name = gr.Textbox(
            label="Character name", placeholder="Sy Snootles",
            elem_id="dd-name-project",
            info="Used by ② (woven into each shot prompt), ③ (prose captions; "
                 "taggers ignore it), ④ (names the dataset folder) and ⑤. Leave "
                 "blank for a generic subject.")
        project_trigger = gr.Textbox(
            label="Trigger word", placeholder="sysnootles",
            elem_id="dd-trigger-project",
            info="Unique token the LoRA learns, placed first in every caption. "
                 "Used by ③, ④'s metadata and ⑤'s sample prompt — they have to "
                 "agree, so there is one box.")
    with gr.Row():
        btn_stop = gr.Button("⏹ Stop the running job", size="sm", scale=0,
                             variant="stop")
        stop_note = gr.Markdown()
    gr.Markdown(
        "<sub>Stop applies to ① Preprocess, ② Generate and ③ Caption. It finishes "
        "the image or shot in flight, then keeps everything already completed — "
        "the result note tells you how to resume.</sub>")
    results_state = gr.State([])
    # Picker rows behind each click-to-toggle gallery: (image path, checkbox value,
    # base label). Kept in State so a click can resolve an index to a value.
    gen_rows = gr.State([])
    cap_rows = gr.State([])
    exp_rows = gr.State([])
    # Absolute paths ③ has captioned, carried into ④'s preselection.
    cap_carry = gr.State([])
    # Ordered filenames behind the ③ inline editor, so ◀/▶ can resolve an index
    # without re-reading the folder on every click.
    cap_edit_names = gr.State([])

    with gr.Tabs() as tabs:
        with gr.Tab("⚡ Quick build", id="quick"):
            gr.Markdown(
                "Drop a few images, type a name at the top, press **Build**. Untick "
                "anything that looks wrong, then **Finish** captions the dataset and "
                "writes the trainer config. The numbered tabs run the same steps one "
                "at a time, with every option.")
            with gr.Row():
                with gr.Column(scale=1):
                    qb_files = gr.File(label="Images of your subject", file_count="multiple",
                                       file_types=["image"], height=160)
                    qb_held = gr.Textbox(
                        label="Anything held to remove?", placeholder="cup, plate",
                        info="Name objects the subject is holding, so they are cut out "
                             "and not redrawn in every shot.")
                    qb_engine = gr.Radio(QUICK_ENGINES, value=settings.default_engine,
                                         label="Generate with")
                    qb_engine_note = gr.Markdown()
                    with gr.Row():
                        qb_trainer = gr.Dropdown(TRAINER_CHOICES, value="ai-toolkit",
                                                 label="Trainer")
                        qb_model = gr.Dropdown(
                            [(m.label, m.key) for m in TRAINER_MODELS["ai-toolkit"]],
                            value=TRAINER_MODELS["ai-toolkit"][0].key, label="Base model")
                    qb_captioner = gr.Dropdown(CAPTIONER_CHOICES,
                                               value=settings.default_captioner,
                                               label="Captioner")
                    qb_front = gr.Checkbox(value=False,
                                           label="Prioritize this app's ComfyUI jobs")
                    btn_qb_build = gr.Button("⚡ Build", variant="primary")
                with gr.Column(scale=2):
                    qb_note = gr.Markdown()
                    qb_gallery = gr.Gallery(
                        label="Click a thumbnail to untick/tick it", columns=6, height=420,
                        allow_preview=False, elem_id="dd-gallery-quick")
                    qb_zoom = gr.Checkbox(value=False, label="🔍 Zoom on click",
                                          elem_id="dd-zoom-quick")
                    qb_keep = gr.CheckboxGroup(label="✅ Kept: UNCHECK to leave out",
                                               choices=[], elem_id="dd-picks-quick")
                    qb_review = gr.Markdown()
                    btn_qb_finish = gr.Button("✅ Finish: caption, export, write configs",
                                              variant="primary")
                    qb_result = gr.Markdown()
            qb_state = gr.State(None)
            qb_rows = gr.State([])

        with gr.Tab("① Preprocess (optional)", id="preprocess"):
            gr.Markdown("Restore / upscale / isolate source images. Skip this tab entirely "
                        "if your images are already clean.")
            with gr.Row():
                with gr.Column(scale=1):
                    pre_files = gr.File(label="Source image(s)", file_count="multiple",
                                        file_types=["image"],
                                        height=160)
                    pre_folder = gr.Textbox(
                        label="…or input folder",
                        placeholder="path/to/images (used if no upload)",
                        info="Used only when nothing is uploaded above — an upload "
                             "always wins. Any folder on any drive.")
                    target = gr.Slider(512, 2048, value=settings.target_long_side, step=64,
                                       label="Dataset resolution (long side, px)",
                                       info="1024 suits Flux/Krea/SDXL. Match your base model.")
                    restore_mode = gr.Radio(["Auto (only if needed)", "Always", "Never"],
                                            value="Auto (only if needed)", label="Restoration",
                                            info="Deblur/upscale degraded sources. Auto only "
                                                 "acts when an image looks low-quality.")
                    restore_backend = gr.Dropdown(RESTORE_BACKEND_CHOICES,
                                                  value=settings.restore_backend,
                                                  label="Restoration backend",
                                                  info="Auto uses ComfyUI models if reachable, "
                                                       "else basic Lanczos resize.")
                    isolate = gr.Checkbox(value=True,
                                          label="Isolate subject for ② generation (training "
                                                "copies keep their background)",
                                          info="Also writes the subject cut out onto white to "
                                               "refs/, as ②'s reference, so the old background "
                                               "and props don't leak into generated shots.")
                    isolation_backend = gr.Dropdown(ISOLATION_CHOICES,
                                                    value=settings.isolation_backend,
                                                    label="Isolation backend",
                                                    info="Built-in SAM3 needs no ComfyUI (gated "
                                                         "HF model + HF_TOKEN).")
                    subject_prompt = gr.Textbox(label="Subject to keep (SAM3 prompt)",
                                                value="character",
                                                info="What SAM3 keeps — e.g. 'character', "
                                                     "'person', 'robot'. Name the thing itself "
                                                     "for a Concept dataset ('radio', 'sword').")
                    exclude_prompt = gr.Textbox(
                        label="Objects to remove (props the subject holds/touches)",
                        placeholder="cup, plate, microphone",
                        info="Name anything the subject holds. SAM3 keeps a held object as "
                             "part of the subject, and ② then redraws it in every shot. "
                             "Background clutter needs no entry.")
                    pre_tighten = gr.Checkbox(
                        value=False, label="Tighten crop to subject (refs/ copy)",
                        info="Crop out the white padding around the isolated reference so the "
                             "subject fills more of what ② sees. Needs isolation on.")
                    pre_alpha_cutout = gr.Checkbox(
                        value=False, label="Transparent cutout (alpha) instead of white",
                        info="Writes the refs/ copy on a transparent background for your "
                             "own compositing workflows. Builtin SAM3 backend only. Leave off "
                             "(default) if you're continuing to ② Generate — it expects a white "
                             "background reference. Needs isolation on.")
                    pre_front = gr.Checkbox(
                        value=False, label="Prioritize this app's ComfyUI jobs",
                        info="Puts ComfyUI restore/isolation jobs at the head of its "
                             "pending queue. Does not interrupt a job already running.")
                    btn_pre = gr.Button("① Preprocess", variant="primary")
                with gr.Column(scale=2):
                    pre_note = gr.Markdown()
                    prep_gallery = gr.Gallery(label="Preprocessed output", columns=4, height=340)

        with gr.Tab("② Generate & Curate", id="generate"):
            gr.Markdown("Turn reference image(s) into a full shot set — 24 shots for a "
                        "**Character**, 18 for a **Concept** (turnaround + framing + "
                        "context). Each plan row becomes one generated image; `chain_from` "
                        "makes rear views build on a generated side view. **Style** datasets "
                        "don't generate — see ③.")
            # Type-specific guidance (Style/Concept collect their own images); hidden
            # for Character. Updated by the header dataset-type selector.
            gen_type_note = gr.Markdown(visible=False)
            with gr.Row():
                with gr.Column(scale=1):
                    gen_files = gr.File(label="Reference image(s)", file_count="multiple",
                                        file_types=["image"], height=160)
                    gen_src_folder = gr.Textbox(
                        label="…or reference folder (auto-filled by ①)",
                        info="Used only when nothing is uploaded above. A clean, "
                             "isolated reference gives much better shots.")
                    _style_key0, _style_text0 = _uc_boot.get_shot_style()
                    gen_style = gr.Dropdown(
                        shot_style.STYLE_CHOICES, value=_style_key0,
                        label="Shot style (medium the shots are drawn in)",
                        info="Default keeps your reference's own medium — pick this if "
                             "your source is a drawing, painting or render and you want "
                             "the shots to stay that way. Choose another to convert the "
                             "whole set to that medium. Changing this rebuilds the shot "
                             "plan below, replacing hand-edited prompt cells.")
                    gen_style_text = gr.Textbox(
                        label="Describe the custom style", value=_style_text0,
                        visible=_style_key0 == shot_style.CUSTOM,
                        placeholder="a 1970s screen-printed poster, halftone dots, "
                                    "limited ink palette",
                        info="A noun phrase describing the medium — it is introduced as "
                             "'Rendered as …' so the model treats it as the style, not "
                             "as something to put in the picture. Press Enter to apply.")
                    refresh = gr.Button("Rebuild default plan with character name")
                    engine = gr.Radio(ENGINE_CHOICES, value=settings.default_engine,
                                      label="Generation engine",
                                      info="Cloud Gemini needs no GPU (best identity, SFW); "
                                           "local ComfyUI is free, private, uncensored.")
                    # A pinned id the list lacks (a retired preview in .env) would
                    # make Gradio reject every event that reads this dropdown.
                    cloud_model = gr.Dropdown(CLOUD_MODEL_CHOICES,
                                              value=CLOUD_MODEL_DEFAULT,
                                              label="Cloud image model",
                                              info="Only used by the Cloud engine. Prices are "
                                                   "build-time estimates.")
                    refresh_models = gr.Button("🔄 Refresh model list from API")
                    force_refresh_models = gr.Button("🔄 Force refresh model list now")
                    cost = gr.Markdown()
                    gen_exclude_props = gr.Checkbox(
                        value=True,
                        label="Exclude props/accessories from the reference",
                        info="Cloud engine only: asks Gemini to drop bags, held objects "
                             "and accessories carried in your reference, so they don't "
                             "get baked into every dataset image. The local engine "
                             "ignores it (naming a prop, even to forbid it, makes "
                             "Qwen draw it) — isolate the source in ① instead, the more "
                             "reliable fix either way. Character-oriented wording — off "
                             "by default for Concept datasets.")
                    gen_anchor = gr.Checkbox(
                        value=False, label="Anchor shot: build every shot from the front view",
                        info="Generates the front full-body view first, then leads every "
                             "other shot's references with it. Helps when your sources "
                             "are partial or poor (face in shadow, body cut off). It "
                             "copies whatever the front view gets wrong (a held object, "
                             "an outfit) into every shot, so check that view first.")
                    gen_isolate = gr.Checkbox(value=False,
                                              label="Isolate generated angle shots (white background)",
                                              info="Cut generated angle shots onto white too. "
                                                   "Off by default: a white void in many "
                                                   "training images trains into the LoRA. "
                                                   "Remove held props with ①'s exclude "
                                                   "prompt instead.")
                    gen_iso_backend = gr.Dropdown(ISOLATION_CHOICES,
                                                  value=settings.isolation_backend,
                                                  label="Isolation backend",
                                                  info="Built-in SAM3 needs no ComfyUI.")
                    gen_subject = gr.Textbox(label="Subject prompt for isolation", value="character",
                                             info="What to keep when isolating generated shots "
                                                  "— e.g. 'character', 'robot', or the object "
                                                  "itself for a Concept dataset.")
                    gen_exclude = gr.Textbox(
                        label="Objects to remove when isolating (auto-filled by ①)",
                        placeholder="backpack, walkie talkie",
                        info="One concept per comma — each is segmented separately.")
                    gen_front = gr.Checkbox(
                        value=False, label="Prioritize this app's ComfyUI jobs",
                        info="Puts our jobs at the head of ComfyUI's pending queue. "
                             "Does not interrupt a job already running.")
                with gr.Column(scale=2):
                    # wrap=False on purpose: wrapping the two ~200-char prompt
                    # cells inflates every row to ~250px, so only two of the 24
                    # shots are on screen at once. Unwrapped, the plan is
                    # scannable and the short columns (outfit/emotion) are fully
                    # readable; click any cell to see or edit its full text.
                    plan = gr.Dataframe(value=_plan_table("character"), label="Shot plan",
                                        interactive=True, wrap=False,
                                        column_widths=PLAN_COLUMN_WIDTHS, max_height=520)
                    gr.Markdown(
                        "<sub>One row = one generated image. Every cell is editable — "
                        "delete rows you don't want, or rewrite a prompt. `chain_from` "
                        "builds a shot from an earlier generated one (rear views). "
                        "Outfit, prop-exclusion and the shot style are folded in when "
                        "you generate, so use **👁 Preview final prompt** to see the "
                        "exact text a row will send.</sub>")
                    wardrobe_note = gr.Markdown(
                        "The **outfit** column varies wardrobe without breaking identity — "
                        "filled for you under **Identity only** (header), blank under "
                        "**Signature costume** to keep the reference's clothing. Varied "
                        "outfits stop the LoRA learning the clothes as part of the "
                        "character. Save/load plans as reusable "
                        "prompt libraries under `shot_plans/`.")
                    with gr.Row():
                        btn_outfits = gr.Button("🎲 Randomize outfits", scale=1)
                        btn_outfits_clear = gr.Button("Clear outfits", scale=1)
                    with gr.Row():
                        plan_name = gr.Textbox(label="Plan name", placeholder="my-plan",
                                               scale=2,
                                               info="Saved as YAML under shot_plans/ — "
                                                    "a reusable prompt library.")
                        btn_save_plan = gr.Button("💾 Save plan", scale=1)
                        btn_load_plan = gr.Button("📂 Load plan", scale=1)
                        btn_preview_prompt = gr.Button("👁 Preview final prompt", scale=1)
                    plan_note = gr.Markdown()
            with gr.Row():
                btn_gen = gr.Button("② Generate all shots", variant="primary")
                btn_regen = gr.Button("♻️ Regenerate UNCHECKED shots (new seeds)")
                btn_disk = gr.Button("🔃 Re-sync with output folder")
                btn_send = gr.Button("➡ Send kept shots to ③ Caption")
            gen_out_dir = gr.Textbox(
                label="Output folder (blank = new run folder)", value="",
                info="Filled in after a run. Keep it to write more shots into the "
                     "same folder — that is how ♻️ finishes a stopped or partly "
                     "failed set.")
            gen_send_note = gr.Markdown()
            # allow_preview=False so a click TOGGLES the shot instead of opening a
            # lightbox; the Zoom checkbox flips it back when you want a closer look.
            with gr.Row():
                gen_gallery = gr.Gallery(
                    label="Generated shots — click a thumbnail to keep/reject it "
                          "(shift-click for a range)",
                    columns=6, height=420, allow_preview=False, elem_id="dd-gallery-gen",
                    scale=4)
                gen_primary = gr.Image(label="Primary reference", height=420,
                                       interactive=False, scale=1)
            with gr.Row():
                btn_gen_all = gr.Button("Select all", size="sm")
                btn_gen_none = gr.Button("Select none", size="sm")
                btn_gen_dupes = gr.Button("🔁 Find near-duplicates", size="sm")
                gen_zoom = gr.Checkbox(value=False, label="🔍 Zoom on click",
                                       elem_id="dd-zoom-gen",
                                       info="Clicks enlarge instead of selecting.")
            keep = gr.CheckboxGroup(label="✅ Kept shots — UNCHECK to reject", choices=[],
                                    elem_id="dd-picks-gen")
            with gr.Row():
                gen_coverage = gr.Markdown()
                gen_dominance = gr.Number(
                    value=40, minimum=10, maximum=100, step=5, scale=0,
                    label="Dominance warning (%)",
                    info="Warn when one view or expression is more than this share "
                         "of the kept shots.")

        with gr.Tab("③ Caption", id="caption"):
            gr.Markdown("Tag any folder of images with caption `.txt` sidecars — the folder "
                        "does **not** need to come from ① or ②. Pick **prose**, **Danbooru "
                        "tags** or **e621 tags** to match your target base model. Each "
                        "captioner uses a prompt tuned to that model.")
            with gr.Row():
                with gr.Column(scale=1):
                    cap_folder = gr.Textbox(
                        label="Image folder (auto-filled by ①/②)",
                        info="Any folder of images — it does not have to come from "
                             "① or ②. Captions are written as .txt sidecars beside "
                             "each image.")
                    btn_load = gr.Button("📂 Load folder")
                    captioner = gr.Dropdown(CAPTIONER_CHOICES, value=settings.default_captioner,
                                            label="Captioner",
                                            info="Local VLMs need a GPU; taggers run on CPU too; "
                                                 "Gemini/Groq are cloud. See the cost line below.")
                    cap_style = gr.Radio(
                        [("Prose — natural language (Flux, Qwen, SDXL 3, …)", "prose"),
                         ("Danbooru tags (SDXL, Illustrious, NoobAI, …)", "tags"),
                         ("e621 tags — furry/anthro vocab (Pony, furry checkpoints)", "e621")],
                        value="prose", label="Caption style",
                        info="Match your target base model: tag-trained checkpoints want "
                             "comma-separated tags, not prose. Danbooru and e621 are different "
                             "vocabularies — pick the one your base model was trained on. The "
                             "trigger stays first either way. (The 'Local tagger' captioners "
                             "ignore this and always emit canonical tags.)")
                    cap_sparse = gr.Checkbox(
                        value=False, label="Sparse captions (Style datasets only)",
                        visible=False,
                        info="Caption only the trigger plus a few words of content. Stronger "
                             "style transfer, but the trigger may absorb some content. "
                             "Ignored for Character/Concept.")
                    with gr.Accordion("Tag options (taggers & tag styles)", open=False):
                        gr.Markdown(
                            "Fixed **prefix/suffix** ride on every caption — e.g. Pony's "
                            "`score_9, score_8_up, score_7_up` quality tags. The **drop-list** "
                            "strips noisy tags across the whole folder. **Thresholds** tune "
                            "how many tags the *taggers* emit (lower general = more tags).")
                        cap_prefix = gr.Textbox(
                            label="Fixed prefix (added before the trigger)",
                            placeholder="score_9, score_8_up, score_7_up",
                            info="Constant tags added to every caption, before the trigger. "
                                 "Tag styles only.")
                        cap_suffix = gr.Textbox(
                            label="Fixed suffix (added at the end)",
                            info="Constant tags added at the end of every caption.")
                        cap_blacklist = gr.Textbox(
                            label="Drop-list (tags to remove)",
                            placeholder="simple background, signature, watermark",
                            info="Comma-separated tags stripped from every tag caption "
                                 "(taggers & tag styles). Casing/underscores don't matter; "
                                 "the trigger is always kept.")
                        with gr.Row():
                            cap_rating = gr.Checkbox(
                                value=False, label="Append rating tag",
                                info="Adds the tagger's top rating "
                                     "(general/sensitive/questionable/explicit). "
                                     "WD/Danbooru taggers only.")
                            cap_underscores = gr.Checkbox(
                                value=False, label="Keep underscores",
                                info="Emit raw booru tags (long_hair) instead of "
                                     "'long hair'. Taggers only.")
                        with gr.Row():
                            cap_gen_thr = gr.Slider(
                                0.05, 0.95, value=0.35, step=0.05,
                                label="Tagger: general threshold",
                                info="Lower = more descriptor tags.")
                            cap_char_thr = gr.Slider(
                                0.05, 0.95, value=0.85, step=0.05,
                                label="Tagger: character threshold",
                                info="Higher avoids mislabelling as a known character.")
                        cap_skip = gr.Checkbox(
                            value=False,
                            label="Skip images that already have a caption",
                            info="Leave existing .txt sidecars untouched — caption only the rest.")
                    cap_cost = gr.Markdown()
                    cap_gemini_model = gr.Dropdown(
                        CAPTION_MODEL_CHOICES, value=_DEFAULT_CAPTION_MODEL,
                        label="Gemini caption model (only used by the Gemini captioner)",
                        info="Ignored unless the Gemini captioner is selected.")
                    btn_refresh_cap_models = gr.Button("🔄 Refresh Gemini model list from API")
                    _custom_cfg = _uc_boot.get_custom_captioner()
                    with gr.Accordion("Custom endpoint settings (for the 'Custom …' captioner)",
                                      open=False):
                        gr.Markdown(
                            "Point at any **OpenAI-compatible** chat/vision endpoint "
                            "(OpenRouter, vLLM, a local proxy, …). **You pay that provider** "
                            "and are responsible for what you send. 429s are retried with "
                            "backoff; set spacing below if you hit limits.")
                        cap_custom_url = gr.Textbox(
                            label="Base URL", value=_custom_cfg.get("base_url", ""),
                            placeholder="https://openrouter.ai/api/v1")
                        cap_custom_model = gr.Textbox(
                            label="Model (blank = first model the server lists)",
                            value=_custom_cfg.get("model", ""),
                            placeholder="qwen/qwen2.5-vl-72b-instruct")
                        cap_custom_keyenv = gr.Textbox(
                            label="API key env var name (blank if none; set the key itself in .env)",
                            value=_custom_cfg.get("api_key_env", ""),
                            placeholder="OPENROUTER_API_KEY",
                            info="The NAME of the variable, not the key. Only the name "
                                 "is saved; the secret stays in .env and is never "
                                 "written to this app's settings file.")
                        cap_custom_interval = gr.Number(
                            label="Min seconds between requests (0 = no spacing)",
                            value=_custom_cfg.get("min_interval_s", 0.0), precision=1)
                        btn_save_custom = gr.Button("💾 Save endpoint")
                        cap_custom_note = gr.Markdown()
                    btn_test = gr.Button("🧪 Test caption on first selected image")
                    btn_caption = gr.Button("③ Caption selected images", variant="primary")
                    btn_send_export = gr.Button("➡ Send captioned images to ④ Export")
                with gr.Column(scale=2):
                    cap_note = gr.Markdown()
                    cap_gallery = gr.Gallery(
                        label="Folder contents — click a thumbnail to include/exclude it "
                              "(shift-click for a range)",
                        columns=6, height=340, allow_preview=False,
                        elem_id="dd-gallery-cap")
                    with gr.Row():
                        btn_cap_all = gr.Button("Select all", size="sm")
                        btn_cap_none = gr.Button("Select none", size="sm")
                        btn_cap_captioned = gr.Button("Only already-captioned", size="sm")
                        cap_zoom = gr.Checkbox(value=False, label="🔍 Zoom on click",
                                               elem_id="dd-zoom-cap",
                                               info="Clicks enlarge instead of selecting.")
                    cap_select = gr.CheckboxGroup(label="Images to caption", choices=[],
                                                  elem_id="dd-picks-cap")
            test_caption = gr.Textbox(label="Test caption output", lines=4,
                                      info="Result of 🧪 Test — a dry run on one image "
                                           "that writes no sidecar.")
            gr.Markdown("**Inline editor** — review captions one by one and save them back "
                        "to their `.txt` sidecars (independent of the model). "
                        "**◀ / ▶** step through the folder; a trailing **✓** in the "
                        "picker marks an image that already has a caption.")
            with gr.Row():
                btn_edit_prev = gr.Button("◀ Prev", scale=0, min_width=90)
                cap_edit_file = gr.Dropdown(label="Image", choices=[], scale=3,
                                            info="A trailing ✓ means that image "
                                                 "already has a caption.")
                btn_edit_next = gr.Button("Next ▶", scale=0, min_width=90)
            with gr.Row():
                with gr.Column(scale=1):
                    # Editing a caption you cannot see is guesswork — this is the
                    # image the text below describes.
                    # No show_download_button= — it is deprecated in Gradio 5 and
                    # prints a warning on every start; the default is fine here.
                    cap_edit_image = gr.Image(label="Image being captioned", height=260,
                                              interactive=False)
                    cap_edit_pos = gr.Markdown()
                with gr.Column(scale=2):
                    cap_edit_text = gr.Textbox(
                        label="Caption editor", lines=8,
                        info="Saved verbatim to the image's .txt sidecar.")
                    with gr.Row():
                        btn_edit_save = gr.Button("💾 Save caption", variant="primary")
                        btn_edit_save_next = gr.Button("💾 Save & next ▶", variant="primary")
                        btn_edit_load = gr.Button("↺ Reload from disk")
            cap_result = gr.Markdown()
            btn_lint = gr.Button("🔎 Analyze captions (health & tag frequency)")
            gr.Markdown(
                "Advisory only — flags empty / too-short / trigger-missing captions, "
                "identical captions (a captioner that returned junk), and, for tag "
                "datasets, tags that appear on nearly every image (drop-list candidates). "
                "Runs automatically after captioning; click to re-check any loaded folder.")
            cap_analysis = gr.Markdown()

        with gr.Tab("④ Export", id="export"):
            gr.Markdown("Package captioned images into a flat `NN.png` + `NN.txt` dataset "
                        "folder (ai-toolkit / OneTrainer ready), with `metadata.json` and "
                        "`README.txt`. List one or more folders (one per line) — e.g. the "
                        "preprocessed sources **and** the generated shots — then **Load & "
                        "preview** to make your final pick before exporting.")
            exp_folders = gr.Textbox(
                label="Folders of captioned images (one per line)", lines=3,
                info="All listed folders are merged into ONE dataset — e.g. your "
                     "preprocessed sources plus the generated shots. ③ adds to this "
                     "list as you caption.")
            with gr.Row():
                btn_load_preview = gr.Button("📂 Load & preview", scale=2)
                exp_dup_dist = gr.Slider(
                    1, 12, value=5, step=1, scale=1,
                    label="Near-duplicate sensitivity",
                    info="Higher flags more images as near-duplicates (dHash bit distance).")
            exp_preview_note = gr.Markdown()
            exp_gallery = gr.Gallery(
                label="Final review — click a thumbnail to include/exclude it "
                      "(shift-click for a range)",
                columns=6, height=420, allow_preview=False, elem_id="dd-gallery-exp")
            with gr.Row():
                btn_exp_all = gr.Button("Select all", size="sm")
                btn_exp_none = gr.Button("Select none", size="sm")
                btn_exp_captioned = gr.Button("Only images with a caption", size="sm")
                exp_zoom = gr.Checkbox(value=False, label="🔍 Zoom on click",
                                       elem_id="dd-zoom-exp",
                                       info="Clicks enlarge instead of selecting.")
            exp_select = gr.CheckboxGroup(
                label="✅ Images to export — UNCHECK to drop", choices=[],
                elem_id="dd-picks-exp")
            output_root = gr.Textbox(label="Output folder", value=str(settings.output_root),
                                     info="Where the NN.png/NN.txt dataset folder is written.")
            exp_zip = gr.Checkbox(
                value=False, label="Also save a .zip of the dataset",
                info="A single archive next to the folder — handy for uploading to a cloud trainer.")
            exp_ilb = gr.Checkbox(
                value=False, label="Prepare for Idiot LoRa Builder",
                info="Writes a ratings sidecar into the dataset folder so Idiot LoRa "
                     "Builder's grid opens pre-triaged — blurry, over/under-exposed and "
                     "near-duplicate shots marked 'needs edit'. Nothing is launched.")
            exp_holdout = gr.Number(
                value=0, precision=0, minimum=0, label="Hold out N reference photos",
                info="Keeps your N largest ① photos OUT of training, in a "
                     "'-heldout' folder beside the dataset — a fair test of whether "
                     "the LoRA learned the face. 0 trains on everything.")
            btn_export = gr.Button("④ Export dataset", variant="primary")
            exp_result = gr.Textbox(label="Result", lines=8)
            with gr.Accordion("Publish to Hugging Face (optional)", open=False):
                gr.Markdown(
                    "Upload the exported dataset to the **Hugging Face Hub**. Created "
                    "**private by default** — uncheck only if you deliberately want it public. "
                    "**You are responsible** for holding the rights to every image and for "
                    "following [Hugging Face's terms](https://huggingface.co/terms-of-service). "
                    "Needs a **write** token in `.env` as `HF_TOKEN` "
                    "([create one](https://huggingface.co/settings/tokens)). Nothing is uploaded "
                    "until you click the button.")
                exp_ds_dir = gr.Textbox(label="Dataset folder to publish (auto-filled by ④ Export)")
                with gr.Row():
                    exp_hf_repo = gr.Textbox(label="Dataset name (or owner/name)",
                                             placeholder="my-character-lora")
                    exp_hf_private = gr.Checkbox(
                        value=True, label="Private (recommended)",
                        info="Unchecking publishes every image PUBLICLY. Note an "
                             "existing repo keeps its current visibility — this cannot "
                             "make an already-public dataset private.")
                btn_publish_hf = gr.Button("⬆ Publish to Hugging Face")
                exp_hf_note = gr.Markdown()

        with gr.Tab("⑤ Train (configs, optional)", id="train"):
            gr.Markdown(
                "Generate a ready-to-edit LoRA training config for your dataset. "
                "**ai-toolkit** produces a one-command `config.yaml` (`python run.py …`); "
                "**musubi-tuner**, **kohya** and **Fizgig** produce a `dataset.toml` plus a "
                "command template where you fill in your local model paths. Every trainer "
                "saves a checkpoint and renders fixed validation prompts each epoch; "
                "`validation/validation.md` explains how to pick the best one. Nothing is "
                "launched here — the files are written into the dataset folder and the run "
                "command is shown.")
            from studio import user_config as _uc

            _ai_presets = TRAINER_MODELS["ai-toolkit"]
            with gr.Row():
                with gr.Column(scale=1):
                    tr_trainer = gr.Radio(TRAINER_CHOICES, value="ai-toolkit", label="Trainer",
                                          info="ai-toolkit is one-command; musubi/kohya/Fizgig "
                                               "emit a config plus a run-command template.")
                    tr_path = gr.Textbox(label="Trainer install path (saved on this machine)",
                                         value=_uc.get_trainer_path("ai-toolkit"),
                                         placeholder=r"C:\ai-toolkit",
                                         info="Only used to compose the displayed run command.")
                    tr_save_path = gr.Button("💾 Save install path")
                    tr_path_note = gr.Markdown()
                    tr_model = gr.Dropdown([(p.label, p.key) for p in _ai_presets],
                                           value=_ai_presets[0].key, label="Model",
                                           info="Pick your target base model. Presets whose "
                                                "label mentions setting/editing a path need you "
                                                "to supply your own model path or HF id.")
                    tr_name = gr.Textbox(
                        label="LoRA name (optional)", placeholder="follows the name above",
                        elem_id="dd-name-train",
                        info="Output filename for the trained LoRA — the one field that "
                             "is NOT the character name (you may want '-v2'). Leave it "
                             "blank and it follows the name at the top of the page.")
                    with gr.Row():
                        tr_res = gr.Number(value=_ai_presets[0].resolution, precision=0,
                                           label="Resolution",
                                           info="Train resolution; match your dataset.")
                        tr_batch = gr.Number(value=_ai_presets[0].batch_size, precision=0,
                                             label="Batch size",
                                             info="Raise only if VRAM allows.")
                    with gr.Row():
                        tr_rank = gr.Number(value=_ai_presets[0].rank, precision=0, label="Rank",
                                            info="LoRA capacity. 16 is a safe default.")
                        tr_alpha = gr.Number(value=_ai_presets[0].alpha, precision=0,
                                             label="Alpha", info="Usually equal to rank.")
                    with gr.Row():
                        tr_epochs = gr.Number(value=_ai_presets[0].epochs, precision=0,
                                              label="Epochs",
                                              info="One checkpoint + one sample set each.")
                        tr_repeats = gr.Number(value=0, precision=0, minimum=0,
                                               label="Repeats (0 = auto)",
                                               info="Auto sizes epochs to the target steps.")
                    with gr.Row():
                        tr_lr = gr.Number(value=_ai_presets[0].lr, label="Learning rate",
                                          info="1e-4 is a common starting point.")
                        tr_optimizer = gr.Dropdown(
                            optimizer_choices("ai-toolkit"), value="adamw8bit",
                            label="Optimizer",
                            info="AdamW8bit is every trainer's tested default. Prodigy "
                                 "picks its own step size and ignores the learning rate.")
                    tr_exposure = gr.Markdown()
                    tr_multi_res = gr.Checkbox(
                        value=True, label="Multi-resolution buckets",
                        info="Bucket by the dataset's real aspect ratios instead of "
                             "forcing one square resolution.")
                with gr.Column(scale=2):
                    tr_dataset = gr.Textbox(
                        label="Dataset folder (auto-filled by ④ Export)",
                        info="The config and a validation/ pack are written INTO this "
                             "folder; repeats and buckets are derived from its images. "
                             "Works on any dataset folder, not just ④'s.")
                    btn_inspect = gr.Button("🔍 Inspect dataset")
                    tr_stats = gr.Markdown()
                    tr_gen = gr.Button("⑤ Generate training config", variant="primary")
                    tr_result = gr.Textbox(label="Result / run command", lines=14)

    log_box = gr.Textbox(
        label="Log", lines=8,
        info="Per-item progress from the last ①/②/③ run, including anything that "
             "was skipped and why. Shared by every tab.")

    # ---------- wiring ----------

    btn_pre.click(
        do_preprocess,
        [pre_files, pre_folder, target, restore_mode, restore_backend, isolate,
         isolation_backend, subject_prompt, exclude_prompt, pre_tighten,
         pre_alpha_cutout, pre_front],
        [prep_gallery, pre_note, log_box, gen_src_folder, cap_folder]) \
           .then(lambda s, e: (s, e), [subject_prompt, exclude_prompt],
                 [gen_subject, gen_exclude])

    # queue=False is load-bearing: with the default queued dispatch this click
    # would sit BEHIND the very job it is meant to interrupt and only run once
    # that job had finished on its own.
    btn_stop.click(request_stop, [], [stop_note], queue=False)
    btn_doctor.click(run_doctor, [], [doctor_out])

    # Header dataset-type selector retunes type-dependent controls across tabs
    # (and persists the choice). demo.load applies the same handler on launch so
    # a remembered Style/Concept type arrives with its defaults already set.
    # There is exactly ONE name box and ONE trigger box, in the header above the
    # tabs. Each tab used to keep its own, carried forward by "copy into the next
    # tab IF it is still blank" — so the first dataset of a session seeded ④/⑤ and
    # every later one silently kept the old value. A user exported a dataset
    # stamped with a name and trigger from several runs earlier and only noticed
    # at the end. Mirroring the boxes live was tried first and is worse: Gradio
    # fires `.input` per keystroke, and with unqueued handlers the responses land
    # out of order, so the copies settle on a PREFIX of what was typed (verified
    # in a browser). One box cannot disagree with itself.
    type_outputs = [isolate, subject_prompt,
                    gen_type_note, refresh, plan, btn_gen, btn_regen,
                    btn_outfits, btn_outfits_clear, wardrobe_note,
                    gen_exclude_props, gen_subject,
                    cap_sparse, project_name, project_trigger]
    # The style controls are INPUTS only — adding them to type_outputs would
    # change the handler's return arity, which a test pins on purpose.
    type_inputs = [dataset_type, project_name, gen_style, gen_style_text, identity_policy]
    dataset_type.change(on_dataset_type_change, type_inputs, type_outputs)
    demo.load(on_dataset_type_change, type_inputs, type_outputs)
    # Wardrobe is a character-only idea.
    for _event in (dataset_type.change, demo.load):
        _event(_identity_visible, [dataset_type], [identity_policy])
    identity_policy.change(on_identity_change, [plan, identity_policy, dataset_type],
                           [plan, plan_note])

    refresh.click(refresh_plan,
                  [project_name, dataset_type, gen_style, gen_style_text, identity_policy],
                  [plan])
    # Rebuild on pick. The custom textbox applies on Enter/blur rather than per
    # keystroke — rebuilding 24 prompts on every character typed is pure churn.
    _style_inputs = [dataset_type, project_name, gen_style, gen_style_text, identity_policy]
    gen_style.change(_toggle_style_text, [gen_style], [gen_style_text])
    gen_style.change(rebuild_plan_for_style, _style_inputs, [plan, plan_note])
    gen_style_text.submit(rebuild_plan_for_style, _style_inputs, [plan, plan_note])
    gen_style_text.blur(rebuild_plan_for_style, _style_inputs, [plan, plan_note])
    btn_preview_prompt.click(preview_final_prompt,
                             [plan, engine, gen_exclude_props], [plan_note])
    btn_outfits.click(randomize_outfits, [plan], [plan, plan_note])
    btn_outfits_clear.click(clear_outfits, [plan], [plan, plan_note])
    btn_save_plan.click(do_save_plan, [plan, plan_name], [plan_note])
    btn_load_plan.click(do_load_plan, [plan_name], [plan, plan_note])
    refresh_models.click(refresh_cloud_models, [cloud_model], [cloud_model])

    def _force_refresh(current: str):
        return refresh_cloud_models(current, force=True)

    force_refresh_models.click(_force_refresh, [cloud_model], [cloud_model])
    engine.change(estimate_cost, [engine, cloud_model, plan], [cost])
    cloud_model.change(estimate_cost, [engine, cloud_model, plan], [cost])
    plan.change(estimate_cost, [engine, cloud_model, plan], [cost])

    gen_inputs = [gen_files, gen_src_folder, plan, engine, cloud_model,
                  gen_exclude_props, gen_isolate, gen_iso_backend, gen_subject,
                  gen_exclude, gen_front, gen_anchor]
    btn_gen.click(do_generate, gen_inputs + [gen_out_dir, results_state],
                  [results_state, gen_rows, gen_gallery, keep, log_box, gen_out_dir,
                   cap_folder])
    btn_regen.click(do_regenerate, gen_inputs + [gen_out_dir, results_state, keep],
                    [results_state, gen_rows, gen_gallery, keep, log_box])
    btn_disk.click(do_refresh_disk, [results_state, gen_out_dir, keep],
                   [results_state, gen_rows, gen_gallery, keep, log_box])
    btn_send.click(send_kept_to_caption, [results_state, keep, gen_out_dir],
                   [tabs, cap_folder, cap_rows, cap_gallery, cap_select, cap_note,
                    gen_send_note])

    # Click-to-toggle: the browser-side script forwards a thumbnail click to this
    # CheckboxGroup (see _PICKER_SCRIPT for why Gallery.select can't do it), and any
    # change to the group — from a thumbnail, the boxes themselves, a quick-select
    # button or a reload — re-marks the gallery labels.
    for _gallery, _rows, _boxes, _zoom in ((gen_gallery, gen_rows, keep, gen_zoom),
                                           (cap_gallery, cap_rows, cap_select, cap_zoom),
                                           (exp_gallery, exp_rows, exp_select, exp_zoom),
                                           (qb_gallery, qb_rows, qb_keep, qb_zoom)):
        _boxes.change(_picker_mark, [_rows, _boxes], [_gallery])
        _zoom.change(_set_zoom, [_zoom], [_gallery])
    btn_qb_build.click(
        do_quick_build,
        [qb_files, project_name, project_trigger, dataset_type, identity_policy, qb_engine,
         qb_held, qb_front],
        [qb_state, qb_rows, qb_gallery, qb_keep, qb_note, log_box, project_trigger])
    qb_keep.change(quick_review, [qb_state, qb_keep], [qb_review])
    btn_qb_finish.click(
        do_quick_finish,
        [qb_state, qb_keep, project_name, project_trigger, dataset_type, identity_policy,
         qb_engine, qb_captioner, qb_trainer, qb_model],
        [qb_result, log_box, tr_dataset])
    qb_trainer.change(_model_dropdown, [qb_trainer], [qb_model])
    for _event in (qb_engine.change, dataset_type.change):
        _event(quick_engine_note, [qb_engine, dataset_type], [qb_engine_note])
    demo.load(quick_engine_note, [qb_engine, dataset_type], [qb_engine_note])
    for _event in (keep.change, gen_dominance.change):
        _event(coverage_note, [results_state, keep, gen_dominance], [gen_coverage])
    for _event in (gen_files.change, gen_src_folder.change):
        _event(primary_reference, [gen_files, gen_src_folder], [gen_primary])
    btn_gen_dupes.click(find_gen_duplicates,
                        [results_state, keep, gen_files, gen_src_folder, exp_dup_dist],
                        [gen_send_note])
    btn_gen_all.click(_pick_all, [gen_rows], [keep])
    btn_gen_none.click(_pick_none, [gen_rows], [keep])
    btn_cap_all.click(_pick_all, [cap_rows], [cap_select])
    btn_cap_none.click(_pick_none, [cap_rows], [cap_select])
    btn_cap_captioned.click(_pick_captioned, [cap_rows], [cap_select])
    btn_exp_all.click(_pick_all, [exp_rows], [exp_select])
    btn_exp_none.click(_pick_none, [exp_rows], [exp_select])
    btn_exp_captioned.click(_pick_captioned, [exp_rows], [exp_select])

    btn_load.click(load_caption_folder, [cap_folder],
                   [cap_rows, cap_gallery, cap_select, cap_note]) \
            .then(_editor_choices, [cap_folder], [cap_edit_file, cap_edit_names])
    # Picking an image (by dropdown, ◀/▶, or Save & next) loads its caption and
    # shows the image + position. One path, so every route stays in sync.
    cap_edit_file.change(load_one_caption, [cap_folder, cap_edit_file], [cap_edit_text]) \
                 .then(_editor_context, [cap_folder, cap_edit_file, cap_edit_names],
                       [cap_edit_image, cap_edit_pos])
    # Named handlers, not lambdas: a wiring test can then assert *which* function
    # each button reaches, which a lambda makes impossible to check.
    btn_edit_prev.click(editor_prev, [cap_folder, cap_edit_file, cap_edit_names],
                        [cap_edit_file])
    btn_edit_next.click(editor_next, [cap_folder, cap_edit_file, cap_edit_names],
                        [cap_edit_file])
    btn_edit_load.click(load_one_caption, [cap_folder, cap_edit_file], [cap_edit_text])

    # Saving re-marks the picker (✓) and re-flags ④ if its preview is open.
    _after_save = [exp_rows, exp_gallery, exp_select, exp_preview_note]
    btn_edit_save.click(save_one_caption, [cap_folder, cap_edit_file, cap_edit_text],
                        [cap_result]) \
                 .then(_editor_relabel, [cap_folder, cap_edit_file, cap_edit_names],
                       [cap_edit_file]) \
                 .then(refresh_export_preview, [exp_folders, exp_rows, exp_select],
                       _after_save)
    btn_edit_save_next.click(save_and_next,
                             [cap_folder, cap_edit_file, cap_edit_text, cap_edit_names],
                             [cap_result, cap_edit_file]) \
                      .then(refresh_export_preview, [exp_folders, exp_rows, exp_select],
                            _after_save)
    def _cap_cost(key: str, model: str, selected: list[str]) -> str:
        line = estimate_caption_cost(key, model, len(selected or []))
        vram = CAPTIONERS_BY_KEY[key].vram_note
        return f"{line}  \nVRAM: {vram}" if vram else line

    cap_cost_inputs = [captioner, cap_gemini_model, cap_select]
    captioner.change(_cap_cost, cap_cost_inputs, [cap_cost])
    cap_gemini_model.change(_cap_cost, cap_cost_inputs, [cap_cost])
    cap_select.change(_cap_cost, cap_cost_inputs, [cap_cost])
    # Populate on load too: these only fired on .change, so the cost/VRAM line
    # was blank until the user touched something.
    demo.load(_cap_cost, cap_cost_inputs, [cap_cost])
    demo.load(estimate_cost, [engine, cloud_model, plan], [cost])
    btn_refresh_cap_models.click(refresh_caption_models, [], [cap_gemini_model])
    btn_save_custom.click(
        save_custom_captioner,
        [cap_custom_url, cap_custom_model, cap_custom_keyenv, cap_custom_interval],
        [cap_custom_note])
    btn_test.click(do_test_caption,
                   [cap_folder, cap_select, captioner, project_name, project_trigger,
                    cap_gemini_model,
                    cap_style, cap_gen_thr, cap_char_thr, cap_prefix, cap_suffix,
                    cap_blacklist, cap_rating, cap_underscores, dataset_type, cap_sparse,
                    identity_policy],
                   [test_caption])
    btn_caption.click(
        do_caption,
        [cap_folder, cap_select, captioner, project_name, project_trigger,
         cap_gemini_model, cap_style,
         cap_gen_thr, cap_char_thr, cap_prefix, cap_suffix,
         cap_blacklist, cap_rating, cap_underscores, cap_skip, dataset_type, cap_sparse,
         exp_folders, cap_carry, identity_policy],
        [cap_rows, cap_gallery, cap_select, cap_result, log_box, exp_folders,
         cap_analysis, cap_carry]) \
               .then(_editor_choices, [cap_folder], [cap_edit_file, cap_edit_names])
    btn_lint.click(do_analyze_captions, [cap_folder, project_trigger], [cap_analysis])
    btn_send_export.click(send_captioned_to_export,
                          [exp_folders, exp_dup_dist, cap_carry],
                          [tabs, exp_rows, exp_gallery, exp_select, exp_preview_note])

    btn_load_preview.click(load_export_preview, [exp_folders, exp_dup_dist, cap_carry],
                           [exp_rows, exp_gallery, exp_select, exp_preview_note])
    btn_export.click(do_export,
                     [exp_select, project_name, project_trigger, output_root, exp_zip,
                      dataset_type, gen_style, gen_style_text, exp_ilb, exp_holdout,
                      identity_policy],
                     [exp_result, tr_dataset, exp_ds_dir]) \
              .then(inspect_dataset, [tr_dataset, dataset_type], [tr_stats])
    btn_publish_hf.click(do_publish_hf, [exp_ds_dir, exp_hf_repo, exp_hf_private],
                         [exp_hf_note])

    tr_hparams = [tr_res, tr_rank, tr_alpha, tr_epochs, tr_lr, tr_batch]
    tr_trainer.change(on_trainer_change, [tr_trainer],
                      [tr_model, tr_path] + tr_hparams + [tr_optimizer])
    tr_model.change(on_model_change, [tr_trainer, tr_model], tr_hparams)
    tr_save_path.click(save_trainer_path, [tr_trainer, tr_path], [tr_path_note])
    btn_inspect.click(inspect_dataset, [tr_dataset, dataset_type], [tr_stats])
    # Epochs change with the trainer/model presets, so this follows them too.
    for field in (tr_trainer, tr_dataset, tr_epochs, tr_repeats, tr_batch):
        field.change(exposure_line, [tr_trainer, tr_dataset, tr_epochs, tr_repeats, tr_batch],
                     [tr_exposure])
    tr_gen.click(do_generate_train_config,
                 [tr_trainer, tr_model, tr_dataset, tr_path, tr_name, project_trigger]
                 + tr_hparams + [tr_multi_res, dataset_type, gen_style, gen_style_text,
                                 project_name, tr_repeats, tr_optimizer],
                 [tr_result])

    demo.load(_check_for_update, None, update_notice)

def _pick_port(preferred: int = 7861) -> int:
    """LDS_PORT if set, else ``preferred`` if bindable, else any free port.

    Windows (Hyper-V/WSL/Docker) reserves ~500 contiguous ports around 7861, so a
    "next port up" scan (Gradio's included) can land wholly inside the reserved
    block. Port 0 lets the OS pick one that is genuinely bindable.
    """
    import socket
    pinned = os.environ.get("LDS_PORT", "").strip()
    if pinned:
        return int(pinned)
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise OSError("No free local port; set LDS_PORT in .env.")


if __name__ == "__main__":
    # Bound to localhost on purpose: no auth layer, and .env keys are reachable
    # through the process. Do not expose publicly / use share=True.
    # allowed_paths lets the galleries display images in user-chosen input/output
    # folders on any drive (Gradio otherwise refuses paths outside the CWD/temp
    # dir). It is fixed at launch, so it can't be narrowed per-request; the
    # consequence is that the local file endpoint can serve any file the process
    # can read. Safe ONLY because of the localhost-only, no-auth bind above — see
    # the Security posture note in docs/ARCHITECTURE.md.
    demo.launch(server_name="127.0.0.1", server_port=_pick_port(), inbrowser=True,
                allowed_paths=_allowed_media_paths())
