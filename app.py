"""LoRA Dataset Studio — Gradio UI.

Every tab is standalone: point it at any folder (or upload files) and run just
that stage. When you do run stages in order, each one auto-fills the next
tab's input folder — chaining is a convenience, never a requirement.

Run:  python app.py   then open http://127.0.0.1:7861
"""

from __future__ import annotations

import os
import re
import string
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd

from studio import pipeline
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
    CLOUD_IMAGE_PRICES,
    list_images,
    load_caption_model_cache,
    settings,
)
from studio.shotplan import Shot, plan_for_type
from studio.trainer_configs import TRAINER_MODELS, TRAINERS

TRAINER_CHOICES = [(label, key) for key, label in TRAINERS.items()]

ENGINE_CHOICES = [
    ("Cloud — Gemini image model (best identity fidelity, SFW only)", "gemini"),
    ("Local — ComfyUI Qwen Image Edit 2511 (free, private, uncensored)", "comfyui"),
]
CLOUD_MODEL_CHOICES = [(f"{m}  (~${p:.3f}/img est.)", m) for m, p in CLOUD_IMAGE_PRICES.items()]
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


def on_dataset_type_change(dataset_type: str, gen_name: str = ""):
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
        gr.Textbox(label=f"{label} (used in prompts)"),        # gen_name
        gr.Button(value=f"Rebuild default plan with {label.lower()}",
                  interactive=not is_style),                   # refresh plan
        _plan_table(dataset_type, gen_name),                   # shot plan table
        gr.Button(interactive=not is_style),                   # generate
        gr.Button(interactive=not is_style),                   # regenerate
        gr.Button(visible=not (is_style or is_concept)),       # randomize outfits
        gr.Button(visible=not (is_style or is_concept)),       # clear outfits
        gr.Markdown(visible=not (is_style or is_concept)),     # wardrobe blurb
        gr.Checkbox(value=not is_concept),                     # exclude props
        gr.Textbox(value=subject_kw),                          # ② isolation subject
        # ③ caption
        gr.Textbox(label=f"{label} (optional)",
                   info=_NAME_INFO.get(dataset_type, _NAME_INFO["character"])),
        gr.Textbox(info=_TRIGGER_INFO.get(dataset_type, _TRIGGER_INFO["character"])),
        gr.Checkbox(visible=is_style),                         # sparse (style only)
        # ④ export
        gr.Textbox(label=label),                               # exp_name
    )


# ---------- helpers ----------

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
PLAN_COLUMNS = ["id", "kind", "emotion", "setting", "outfit",
                "local_prompt", "cloud_prompt", "chain_from"]
PLAN_COLUMN_WIDTHS = ["110px", "70px", "110px", "200px", "220px",
                      "260px", "260px", "100px"]


def _plan_table(dataset_type: str, name: str = "") -> pd.DataFrame:
    """The ② table for a dataset type (empty for Style, which never generates)."""
    return _shots_to_df(plan_for_type(dataset_type, name))


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
    for i, outfit in zip(targets, outfits):
        df.at[i, "outfit"] = outfit
    return df, (f"🎲 Dressed {len(targets)} angle/pose shots in distinct outfits "
                f"({len(df) - len(targets)} close-ups left blank). Click again to "
                f"reroll, or clear the column to go back to the reference's clothing.")


def clear_outfits(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    df["outfit"] = ""
    return df, "Outfit column cleared — every shot keeps the reference's clothing."


def _df_to_shots(df: pd.DataFrame) -> list[Shot]:
    def val(row, k):
        v = row[k] if k in row else ""
        return "" if pd.isna(v) else str(v)

    cols = (
        "id", "kind", "local_prompt", "cloud_prompt",
        "chain_from", "emotion", "setting", "outfit",
    )
    return [Shot(**{k: val(row, k) for k in cols})
            for _, row in df.iterrows() if val(row, "id").strip()]

def _gen_gallery(results: list[pipeline.GenResult]):
    ok = [r for r in results if r.path and r.path.exists()]
    gallery = []
    for r in ok:
        label = r.shot.id
        try:
            from studio.quality import composition_flags, is_blurry

            flags: list[str] = []
            blurry, score = is_blurry(r.path)
            if blurry:
                flags.append(f"blurry ({score:.0f})")
            flags += composition_flags(r.path)
            if flags:
                label = f"{r.shot.id}  ⚠ {', '.join(flags)}"
        except Exception:
            pass  # quality checks are advisory — never block the gallery on them
        gallery.append((str(r.path), label))
    ids = [r.shot.id for r in ok]
    return gallery, gr.CheckboxGroup(choices=ids, value=ids)


# ---------- ① preprocess ----------

def do_preprocess(files: list[str], folder: str, target: int, restore_mode: str,
                  restore_backend: str, isolate: bool, isolation_backend: str,
                  subject_prompt: str, exclude_prompt: str, tighten: bool = False):
    sources = _inputs(files, folder)
    out_dir = _stamped("prepped")
    force = {"Auto (only if needed)": None, "Always": True, "Never": False}[restore_mode]
    log: list[str] = []
    try:
        reports = pipeline.preprocess_sources(
            sources, out_dir, target=target, force_restore=force, isolate=isolate,
            subject_prompt=subject_prompt or "character",
            exclude_prompt=exclude_prompt or "", restore_backend=restore_backend,
            isolation_backend=isolation_backend, tighten_crop=tighten, progress=log.append)
    except Exception as e:
        raise gr.Error(f"Preprocess failed: {e}")
    gallery = [(str(r.output), f"{r.source.name}: {r.reason}") for r in reports]
    note = f"✅ {len(reports)} image(s) preprocessed into {out_dir}"
    # Auto-fill downstream tabs (they can still be pointed anywhere else)
    return gallery, note, "\n".join(log), str(out_dir), str(out_dir)


# ---------- ② generate & curate ----------

def do_generate(files: list[str], folder: str, plan_df: pd.DataFrame, engine: str,
                cloud_model: str, exclude_props: bool, isolate_angles: bool,
                isolation_backend: str, subject_prompt: str, exclude_prompt: str,
                front: bool, gen_dir_prev: str, results_state,
                progress=gr.Progress()):
    sources = _inputs(files, folder)
    out_dir = _validate_out_dir(gen_dir_prev) if gen_dir_prev.strip() else _stamped("generated")
    shots = _df_to_shots(plan_df)
    log: list[str] = []

    def report(msg: str):
        log.append(msg)
        progress((len(log), len(shots) + 2), desc=msg)

    try:
        results = pipeline.generate_shots(
            sources, shots, engine, out_dir, cloud_model=cloud_model,
            isolate_angles=isolate_angles, subject_prompt=subject_prompt or "character",
            exclude_prompt=exclude_prompt, isolation_backend=isolation_backend,
            exclude_props=exclude_props, front=front, progress=report)
    except OSError as e:
        raise gr.Error(f"Couldn't write to '{out_dir}': {e}. Check the output folder "
                       f"path (valid drive, no forbidden characters, writable).")
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}")
    gallery, keep = _gen_gallery(results)
    return results, gallery, keep, "\n".join(log), str(out_dir), str(out_dir)


def do_regenerate(files: list[str], folder: str, plan_df: pd.DataFrame, engine: str,
                  cloud_model: str, exclude_props: bool, isolate_angles: bool,
                  isolation_backend: str, subject_prompt: str, exclude_prompt: str,
                  front: bool, gen_dir: str, results_state, keep_ids: list[str],
                  progress=gr.Progress()):
    if not results_state:
        raise gr.Error("Nothing generated yet.")
    all_ids = {r.shot.id for r in results_state}
    redo = all_ids - set(keep_ids or [])
    if not redo:
        raise gr.Error("Uncheck the shots you want regenerated, then click again.")
    sources = _inputs(files, folder)
    log: list[str] = []
    results = pipeline.generate_shots(
        sources, _df_to_shots(plan_df), engine, Path(gen_dir), cloud_model=cloud_model,
        isolate_angles=isolate_angles, subject_prompt=subject_prompt or "character",
        exclude_prompt=exclude_prompt, isolation_backend=isolation_backend,
        exclude_props=exclude_props, front=front, existing=results_state, only_ids=redo,
        progress=log.append)
    gallery, keep = _gen_gallery(results)
    return results, gallery, keep, "\n".join(log)


def do_refresh_disk(results_state, gen_dir: str):
    """Re-sync with the output folder — files you deleted externally drop out."""
    if not results_state:
        raise gr.Error("No generation results in this session.")
    before = len(results_state)
    results = [r for r in results_state if r.path is None or r.path.exists()]
    gallery, keep = _gen_gallery(results)
    note = f"Re-synced with {gen_dir}: {before - len(results)} externally deleted shot(s) dropped."
    return results, gallery, keep, note


def send_kept_to_caption(results_state, keep_ids: list[str], gen_dir: str):
    if not gen_dir.strip():
        raise gr.Error("Nothing generated yet.")
    kept = {r.path.name for r in (results_state or [])
            if r.path and r.path.exists() and r.shot.id in set(keep_ids or [])}
    if not kept:
        raise gr.Error("No kept shots selected.")
    images = list_images(Path(gen_dir.strip()))
    names = [p.name for p in images]
    gallery = [(str(p), p.name) for p in images]
    preselected = [n for n in names if n in kept]
    note = (f"{len(names)} image(s) loaded from ② — {len(preselected)} kept shot(s) "
            f"preselected for captioning.")
    return gen_dir, gallery, gr.CheckboxGroup(choices=names, value=preselected), note


# ---------- ③ caption ----------

def load_caption_folder(folder: str):
    images = list_images(Path(folder.strip())) if folder.strip() else []
    if not images:
        raise gr.Error(f"No images found in folder: {folder or '(empty)'}")
    captioned = {p.name for p in images if p.with_suffix(".txt").exists()}
    gallery = [(str(p), f"{p.name}{' ✓ captioned' if p.name in captioned else ''}")
               for p in images]
    names = [p.name for p in images]
    note = (f"{len(images)} image(s) loaded, {len(captioned)} already have .txt "
            f"sidecars (re-captioning overwrites them).")
    return gallery, gr.CheckboxGroup(choices=names, value=names), note


def _resolve_captioner_config(captioner_key: str, gemini_model: str):
    """UI wrapper over the shared resolver — translates its error into gr.Error
    so the same logic serves the CLI without importing gradio."""
    try:
        return resolve_captioner_config(captioner_key, gemini_model)
    except CaptionerConfigError as e:
        raise gr.Error(str(e))


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
                    dataset_type: str = "character", sparse: bool = False):
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
                          dataset_type=dataset_type, sparse=sparse)
    except Exception as e:
        raise gr.Error(str(e))
    finally:
        cap.unload()
    caption = finalize_caption(raw, trigger, name, SUBJECT_ALIASES, style=style,
                               dataset_type=dataset_type)
    caption = drop_blacklisted_tags(caption, parse_blacklist(blacklist), style)
    return apply_affixes(caption, prefix, suffix, style)


def load_one_caption(folder: str, filename: str) -> str:
    """Read the .txt sidecar for a single image into the inline editor."""
    if not folder.strip() or not filename:
        raise gr.Error("Load a folder and pick an image first.")
    txt = (Path(folder.strip()) / filename).with_suffix(".txt")
    return txt.read_text(encoding="utf-8") if txt.exists() else ""


def save_one_caption(folder: str, filename: str, text: str) -> str:
    """Write the inline editor's text back to the image's .txt sidecar."""
    if not folder.strip() or not filename:
        raise gr.Error("Load a folder and pick an image first.")
    txt = (Path(folder.strip()) / filename).with_suffix(".txt")
    txt.write_text(text.strip(), encoding="utf-8")
    return f"✅ Saved caption for {filename}"


def _editor_choices(folder: str):
    names = [p.name for p in list_images(Path(folder.strip()))] if folder.strip() else []
    return gr.Dropdown(choices=names, value=names[0] if names else None)


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


def do_caption(folder: str, selected: list[str], captioner_key: str,
               name: str, trigger: str, gemini_model: str, style: str,
               gen_thr: float, char_thr: float, prefix: str, suffix: str,
               blacklist: str, rating: bool, underscores: bool,
               skip_existing: bool, dataset_type: str, sparse: bool,
               exp_folders_prev: str, exp_name_prev: str,
               exp_trigger_prev: str, progress=gr.Progress()):
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

    try:
        items = caption_images(images, captioner_key, name, trigger, progress=report,
                               model_override=model_override, spec_overrides=spec_overrides,
                               style=style, prefix=prefix, suffix=suffix,
                               skip_existing=skip_existing, blacklist=blacklist,
                               dataset_type=dataset_type, sparse=sparse)
    except Exception as e:
        raise gr.Error(f"Captioning failed: {e}")
    for img, caption in items:
        img.with_suffix(".txt").write_text(caption, encoding="utf-8")
    gallery, boxes, note = load_caption_folder(folder)
    result = f"✅ Wrote {len(items)} caption sidecar(s) in {base}"
    # Auto-fill ④ Export: ADD this folder to its list (captioning several folders
    # in turn must accumulate), and carry name/trigger without clobbering values
    # the user already typed there.
    folders = _merge_export_folders(exp_folders_prev, str(base))
    analysis = _caption_analysis(str(base), trigger)
    return (gallery, boxes, result, "\n".join(log), folders,
            exp_name_prev or name, exp_trigger_prev or trigger, analysis)


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
    txt = img.with_suffix(".txt")
    if not txt.exists():
        return "⚠ no caption"
    return "✓" if txt.read_text(encoding="utf-8").strip() else "⚠ empty"


def _export_label(img: Path) -> tuple[str, str]:
    """(checkbox label, value). Value is the full path so same-named files in
    different folders never collide; label is folder/name + caption flag."""
    return f"{img.parent.name}/{img.name} — {_export_flag(img)}", str(img)


def load_export_preview(folders_text: str, dup_distance: float = 5):
    folders = [Path(line.strip()) for line in folders_text.splitlines() if line.strip()]
    if not folders:
        raise gr.Error("Enter at least one folder of captioned images (one per line).")
    images = [img for folder in folders for img in list_images(folder)]
    if not images:
        raise gr.Error("No images found in the listed folder(s).")
    gallery = [(str(img), f"{img.parent.name}/{img.name} — {_export_flag(img)}")
               for img in images]
    choices = [_export_label(img) for img in images]
    values = [v for _, v in choices]  # all checked by default (uncheck to drop)
    ready = sum(1 for img in images if _export_flag(img) == "✓")
    empty = sum(1 for img in images if _export_flag(img) == "⚠ empty")
    none_ = sum(1 for img in images if _export_flag(img) == "⚠ no caption")
    note = (f"**{len(images)} image(s)** — {ready} ready · {empty} empty caption · "
            f"{none_} no caption. All checked below; **uncheck to drop**. "
            "Images without a usable caption are skipped even if left checked.")
    try:  # advisory near-duplicate scan — never blocks the preview
        from studio.dedupe import find_near_duplicate_groups

        groups = find_near_duplicate_groups(images, max_distance=int(dup_distance))
        if groups:
            shown = "; ".join("=".join(f"{p.parent.name}/{p.name}" for p in g)
                              for g in groups[:5])
            more = f" (+{len(groups) - 5} more)" if len(groups) > 5 else ""
            note += (f"\n\n🔁 **{len(groups)} near-duplicate group(s)** — consider "
                     f"unchecking extras so one shot isn't over-weighted: {shown}{more}")
    except Exception:
        pass
    try:  # advisory caption health + tag frequency — never blocks the preview
        from studio.caption_lint import analyze_pairs, markdown_summary

        cap_pairs = []
        for img in images:
            txt = img.with_suffix(".txt")
            if txt.exists() and (c := txt.read_text(encoding="utf-8").strip()):
                cap_pairs.append((f"{img.parent.name}/{img.name}", c))
        # trigger unknown at preview -> skip the missing-trigger check; empties are
        # already summarized above, so only short/duplicate/ubiquitous add value.
        report, ubiquitous = analyze_pairs(cap_pairs, trigger="")
        if not report.clean or ubiquitous:
            note += "\n\n" + markdown_summary(report, ubiquitous)
    except Exception:
        pass
    return gallery, gr.CheckboxGroup(choices=choices, value=values), note


def do_export(selected: list[str], name: str, trigger: str, output_root: str,
              make_zip: bool = False, dataset_type: str = "character"):
    if not selected:
        raise gr.Error("Click '📂 Load & preview', then keep at least one image checked.")
    from studio.package import package_dataset, resolve_export_items

    paths = [Path(s) for s in selected]
    res = resolve_export_items(paths)
    if not res.items:
        raise gr.Error("None of the checked images have a usable caption — "
                       "run ③ Caption first (each export needs a non-empty .txt).")
    source_folders = sorted({str(p.parent) for p in paths})
    metadata = {"character_name": name, "trigger": trigger,
                "dataset_type": dataset_type,
                "source_folders": source_folders,
                "skipped_uncaptioned": res.missing,
                "skipped_empty_caption": res.empties}
    out_root = _validate_out_dir(output_root)
    try:
        ds = package_dataset(res.items, out_root, name, trigger, metadata)
    except OSError as e:
        raise gr.Error(f"Couldn't write the dataset to '{out_root}': {e}. Check the "
                       f"output folder path (valid drive, no forbidden characters, writable).")
    # Show the first numbered caption (README.txt is excluded).
    caption_files = sorted(p for p in ds.glob("*.txt") if p.name != "README.txt")
    samples = [(p.name, p.read_text(encoding="utf-8").strip()) for p in caption_files]
    first = next(((n, t) for n, t in samples if t), None)
    sample_block = f"\n\nSample caption ({first[0]}):\n{first[1]}" if first else ""
    skipped = (f"\n⚠️ Skipped (no caption): {', '.join(res.missing)}"
               if res.missing else "")
    empty_note = (f"\n⚠️ Skipped (empty caption): {', '.join(res.empties)}"
                  if res.empties else "")
    zip_note = ""
    if make_zip:
        from studio.package import zip_dataset

        try:
            zip_note = f"\n🗜️ Zipped: {zip_dataset(ds)}"
        except OSError as e:
            zip_note = f"\n⚠️ Could not write the .zip: {e}"
    result = (f"✅ Dataset ready: {ds}  ({len(res.items)} image/caption pairs)"
              f"{skipped}{empty_note}{zip_note}{sample_block}")
    # ds path auto-fills the ⑤ Train tab AND the HF-publish box below.
    return result, str(ds), str(ds)


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
        raise gr.Error(str(e))
    except Exception as e:  # network/auth/etc. — surface, never crash the UI
        raise gr.Error(f"Publishing failed: {e}")
    vis = "private" if private else "PUBLIC"
    return f"✅ Published ({vis}): [{url}]({url})"


# ---------- misc ----------

def _fill_if_empty(current: str, incoming: str) -> str:
    """Carry a value into the next tab without clobbering a hand-typed one."""
    return current.strip() or incoming


def refresh_plan(name: str, dataset_type: str = "character") -> pd.DataFrame:
    return _plan_table(dataset_type, name)


def do_save_plan(plan_df: pd.DataFrame, plan_name: str) -> str:
    from studio.plan_io import save_plan

    shots = _df_to_shots(plan_df)
    if not shots:
        raise gr.Error("The plan is empty — nothing to save.")
    path = settings.shot_plans_dir / (plan_name.strip() or "my-plan")
    saved = save_plan(shots, path)
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
    shots = load_plan(path)
    return _shots_to_df(shots), f"✅ Loaded {len(shots)} shots from {path}"


def estimate_cost(engine: str, cloud_model: str, df: pd.DataFrame) -> str:
    n = len(df)
    if engine == "gemini":
        from studio.config import CLOUD_IMAGE_PRICES, load_cloud_model_cache

        price = CLOUD_IMAGE_PRICES.get(cloud_model)
        cached = load_cloud_model_cache() or []
        for m in cached:
            if m.get("model_id") == cloud_model and m.get("price") is not None:
                price = m["price"]
                break
        if price is None:
            return f"**Cost:** {n} images on `{cloud_model}` (price unknown — billed to your API key)"
        return (f"**Cost:** ~${n * price:.2f} for {n} images on `{cloud_model}` "
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
            p.resolution, p.rank, p.alpha, p.steps, p.lr, p.batch_size)


def on_model_change(trainer: str, model_key: str):
    p = _preset(trainer, model_key)
    return p.resolution, p.rank, p.alpha, p.steps, p.lr, p.batch_size


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


def inspect_dataset(dataset_dir: str, dataset_type: str = "character") -> tuple[str, gr.Number]:
    """Read the dataset and suggest a step count derived from its image count."""
    from studio.dataset_stats import inspect

    if not dataset_dir.strip():
        return "", gr.Number()
    ds = Path(dataset_dir.strip())
    if not ds.is_dir():
        return f"⚠️ Folder not found: {ds}", gr.Number()
    stats = inspect(ds)
    if not stats.n_images:
        return f"⚠️ No images in {ds}", gr.Number()
    return stats.summary() + dataset_type_note(ds, dataset_type), \
        gr.Number(value=stats.suggested_steps)


def do_generate_train_config(trainer: str, model_key: str, dataset_dir: str,
                             install_path: str, name: str, trigger: str,
                             resolution, rank, alpha, steps, lr, batch_size,
                             multi_res: bool, dataset_type: str = "character") -> str:
    if not dataset_dir.strip():
        raise gr.Error("Enter the dataset folder to write the config into "
                       "(④ Export produces one and auto-fills this).")
    ds = Path(dataset_dir.strip())
    if not ds.is_dir():
        raise gr.Error(f"Dataset folder not found: {ds}")
    from studio import user_config
    from studio.dataset_stats import inspect
    from studio.trainer_configs import TrainConfig, write_configs

    stats = inspect(ds)
    if not stats.n_images:
        raise gr.Error(f"No images found in {ds} — export a dataset first (④).")
    preset = _preset(trainer, model_key)
    buckets = stats.buckets_for(int(resolution)) if multi_res else []
    cfg = TrainConfig(
        trainer=trainer, model=preset, dataset_dir=ds,
        trigger=trigger.strip(), name=(name.strip() or "lora"),
        dataset_type=dataset_type,
        resolution=int(resolution), rank=int(rank), alpha=int(alpha),
        steps=int(steps), lr=float(lr), batch_size=int(batch_size),
        buckets=buckets)
    written, command = write_configs(cfg, install_path.strip(),
                                     num_repeats=max(1, round(400 / stats.n_images)))
    user_config.set_last_train_settings({
        "trainer": trainer, "model": model_key, "resolution": int(resolution),
        "rank": int(rank), "alpha": int(alpha), "steps": int(steps),
        "lr": float(lr), "batch_size": int(batch_size)})
    files = "\n".join(str(p) for p in written)
    bucket_note = (f"\nBuckets: {buckets} (from the dataset's actual sizes)"
                   if buckets else f"\nSingle bucket at {int(resolution)}px")
    caveat = ""
    if trainer == "musubi":
        caveat = ("\n\n⚠️ musubi needs your local DiT / VAE / text-encoder paths — "
                  "fill the <<FILL: …>> placeholders in the command before running.")
    elif trainer == "kohya":
        caveat = ("\n\n⚠️ kohya sd-scripts: SDXL base runs from the HF id shown; for a "
                  "Pony / Illustrious / NoobAI checkpoint, replace the <<FILL>> pretrained "
                  "path. Verify flags against the sd-scripts docs before a long run.")
    # Advisory ④→⑤ sanity check: do the dataset's captions fit this base model?
    from studio.caption_lint import folder_caption_kind
    from studio.trainer_configs import caption_mismatch_warning

    mismatch = caption_mismatch_warning(preset, folder_caption_kind(ds))
    if mismatch:
        caveat += f"\n\n{mismatch}"
    return (f"✅ Wrote:\n{files}\n\nDataset: {stats.n_images} images, "
            f"{stats.min_long_side}-{stats.max_long_side}px long side{bucket_note}\n\n"
            f"Run it with:\n{command}{caveat}\n\n"
            f"⚠️ Configs are generated, not test-trained — verify keys against your "
            f"trainer's own docs before a long run.")


def refresh_cloud_models(force: bool = False):
    from studio.engines.gemini import list_image_models

    try:
        models = list_image_models(force_refresh=force)
    except Exception as e:
        raise gr.Error(f"Could not list models: {e}")
    return gr.Dropdown(choices=models,
                       value=models[0][1] if models else settings.gemini_image_model)


def refresh_caption_models():
    """Live-pull the current Gemini caption model list (Caption tab)."""
    from studio.engines.gemini import list_caption_models

    try:
        models = list_caption_models(force_refresh=True)
    except Exception as e:
        raise gr.Error(f"Could not list caption models: {e}")
    value = _DEFAULT_CAPTION_MODEL
    ids = [m[1] for m in models]
    if value not in ids and ids:
        value = ids[0]
    return gr.Dropdown(choices=models, value=value)


def _check_for_update():
    """Best-effort GitHub release check on UI load; silently shows nothing on
    any failure (offline, rate-limited, disabled) so it can never block launch."""
    from studio.update_check import update_banner_markdown

    try:
        text = update_banner_markdown()
    except Exception:
        text = ""
    return gr.Markdown(value=text, visible=bool(text))

# ---------- layout ----------

with gr.Blocks(title="LoRA Dataset Studio") as demo:
    gr.Markdown(
        "# LoRA Dataset Studio\n"
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
    results_state = gr.State([])

    with gr.Tabs():
        with gr.Tab("① Preprocess (optional)"):
            gr.Markdown("Restore / upscale / isolate source images. Skip this tab entirely "
                        "if your images are already clean.")
            with gr.Row():
                with gr.Column(scale=1):
                    pre_files = gr.File(label="Source image(s)", file_count="multiple",
                                        file_types=["image"])
                    pre_folder = gr.Textbox(label="…or input folder",
                                            placeholder="path/to/images (used if no upload)")
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
                                          label="Isolate subject (cutout onto white background)",
                                          info="Cuts the subject out onto white so background "
                                               "and props aren't baked into the LoRA.")
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
                        placeholder="microphone, microphone stand",
                        info="Usually leave blank — SAM3 already excludes most props. Use only "
                             "for a prop fused into the subject.")
                    pre_tighten = gr.Checkbox(
                        value=False, label="Tighten crop to subject (after isolation)",
                        info="Crop out the white padding around the isolated subject so framing "
                             "is consistent and less empty background is trained. Needs isolation on.")
                    btn_pre = gr.Button("① Preprocess", variant="primary")
                with gr.Column(scale=2):
                    pre_note = gr.Markdown()
                    prep_gallery = gr.Gallery(label="Preprocessed output", columns=4, height=340)

        with gr.Tab("② Generate & Curate"):
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
                                        file_types=["image"])
                    gen_src_folder = gr.Textbox(label="…or reference folder (auto-filled by ①)")
                    gen_name = gr.Textbox(label="Character name (used in prompts)",
                                          placeholder="Sy Snootles",
                                          info="Woven into each shot prompt. Leave blank for "
                                               "a generic subject.")
                    refresh = gr.Button("Rebuild default plan with character name")
                    engine = gr.Radio(ENGINE_CHOICES, value=settings.default_engine,
                                      label="Generation engine",
                                      info="Cloud Gemini needs no GPU (best identity, SFW); "
                                           "local ComfyUI is free, private, uncensored.")
                    cloud_model = gr.Dropdown(CLOUD_MODEL_CHOICES,
                                              value=settings.gemini_image_model,
                                              label="Cloud image model",
                                              info="Only used by the Cloud engine. Prices are "
                                                   "build-time estimates.")
                    refresh_models = gr.Button("🔄 Refresh model list from API")
                    force_refresh_models = gr.Button("🔄 Force refresh model list now")
                    cost = gr.Markdown()
                    gen_exclude_props = gr.Checkbox(
                        value=True,
                        label="Exclude props/accessories from the reference",
                        info="Asks the generator to drop bags, held objects and "
                             "accessories carried in your reference, so they don't get "
                             "baked into every dataset image. Isolating the source in ① "
                             "is the more reliable fix. Character-oriented wording — off "
                             "by default for Concept datasets.")
                    gen_isolate = gr.Checkbox(value=False,
                                              label="Isolate generated angle shots (white background)",
                                              info="Cut generated angle shots onto white too "
                                                   "(helps the angles LoRA on back views).")
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
                    wardrobe_note = gr.Markdown(
                        "The **outfit** column varies wardrobe without breaking identity — "
                        "leave blank to keep the reference's clothing. If your source images "
                        "all show the same clothes, randomizing here stops the LoRA learning "
                        "the outfit as part of the character. Save/load plans as reusable "
                        "prompt libraries under `shot_plans/`.")
                    with gr.Row():
                        btn_outfits = gr.Button("🎲 Randomize outfits", scale=1)
                        btn_outfits_clear = gr.Button("Clear outfits", scale=1)
                    with gr.Row():
                        plan_name = gr.Textbox(label="Plan name", placeholder="my-plan",
                                               scale=2)
                        btn_save_plan = gr.Button("💾 Save plan", scale=1)
                        btn_load_plan = gr.Button("📂 Load plan", scale=1)
                    plan_note = gr.Markdown()
            with gr.Row():
                btn_gen = gr.Button("② Generate all shots", variant="primary")
                btn_regen = gr.Button("♻️ Regenerate UNCHECKED shots (new seeds)")
                btn_disk = gr.Button("🔃 Re-sync with output folder")
                btn_send = gr.Button("➡ Send kept shots to ③ Caption")
            gen_out_dir = gr.Textbox(label="Output folder (blank = new run folder)", value="")
            gen_gallery = gr.Gallery(label="Generated shots", columns=6, height=420)
            keep = gr.CheckboxGroup(label="✅ Kept shots — UNCHECK to reject", choices=[])

        with gr.Tab("③ Caption"):
            gr.Markdown("Tag any folder of images with caption `.txt` sidecars — the folder "
                        "does **not** need to come from ① or ②. Pick **prose**, **Danbooru "
                        "tags** or **e621 tags** to match your target base model. Each "
                        "captioner uses a prompt tuned to that model.")
            with gr.Row():
                with gr.Column(scale=1):
                    cap_folder = gr.Textbox(label="Image folder (auto-filled by ①/②)")
                    btn_load = gr.Button("📂 Load folder")
                    cap_name = gr.Textbox(label="Character name (optional)",
                                          placeholder="Sy Snootles",
                                          info="Used in prose captions; taggers ignore it.")
                    cap_trigger = gr.Textbox(label="Trigger word (optional, placed first)",
                                             placeholder="sysnootles",
                                             info="Unique token the LoRA learns as the subject. "
                                                  "Placed first in every caption.")
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
                            placeholder="OPENROUTER_API_KEY")
                        cap_custom_interval = gr.Number(
                            label="Min seconds between requests (0 = no spacing)",
                            value=_custom_cfg.get("min_interval_s", 0.0), precision=1)
                        btn_save_custom = gr.Button("💾 Save endpoint")
                        cap_custom_note = gr.Markdown()
                    btn_test = gr.Button("🧪 Test caption on first selected image")
                    btn_caption = gr.Button("③ Caption selected images", variant="primary")
                with gr.Column(scale=2):
                    cap_note = gr.Markdown()
                    cap_gallery = gr.Gallery(label="Folder contents", columns=6, height=340)
                    cap_select = gr.CheckboxGroup(label="Images to caption", choices=[])
            test_caption = gr.Textbox(label="Test caption output", lines=4)
            gr.Markdown("**Inline editor** — tweak any caption by hand and save it back to "
                        "its `.txt` sidecar (independent of the model).")
            with gr.Row():
                cap_edit_file = gr.Dropdown(label="Image", choices=[], scale=2)
                btn_edit_load = gr.Button("Load its caption", scale=1)
                btn_edit_save = gr.Button("💾 Save caption", variant="primary", scale=1)
            cap_edit_text = gr.Textbox(label="Caption editor", lines=4)
            cap_result = gr.Markdown()
            btn_lint = gr.Button("🔎 Analyze captions (health & tag frequency)")
            gr.Markdown(
                "Advisory only — flags empty / too-short / trigger-missing captions, "
                "identical captions (a captioner that returned junk), and, for tag "
                "datasets, tags that appear on nearly every image (drop-list candidates). "
                "Runs automatically after captioning; click to re-check any loaded folder.")
            cap_analysis = gr.Markdown()

        with gr.Tab("④ Export"):
            gr.Markdown("Package captioned images into a flat `NN.png` + `NN.txt` dataset "
                        "folder (ai-toolkit / OneTrainer ready), with `metadata.json` and "
                        "`README.txt`. List one or more folders (one per line) — e.g. the "
                        "preprocessed sources **and** the generated shots — then **Load & "
                        "preview** to make your final pick before exporting.")
            exp_folders = gr.Textbox(label="Folders of captioned images (one per line)", lines=3)
            with gr.Row():
                btn_load_preview = gr.Button("📂 Load & preview", scale=2)
                exp_dup_dist = gr.Slider(
                    1, 12, value=5, step=1, scale=1,
                    label="Near-duplicate sensitivity",
                    info="Higher flags more images as near-duplicates (dHash bit distance).")
            exp_preview_note = gr.Markdown()
            exp_gallery = gr.Gallery(label="Final review — click a thumbnail to zoom",
                                     columns=6, height=420, allow_preview=True)
            exp_select = gr.CheckboxGroup(
                label="✅ Images to export — UNCHECK to drop", choices=[])
            with gr.Row():
                exp_name = gr.Textbox(label="Character name", placeholder="Sy Snootles",
                                      info="Names the dataset folder and metadata.")
                exp_trigger = gr.Textbox(label="Trigger word", placeholder="sysnootles",
                                         info="Recorded in the dataset metadata/README.")
            output_root = gr.Textbox(label="Output folder", value=str(settings.output_root),
                                     info="Where the NN.png/NN.txt dataset folder is written.")
            exp_zip = gr.Checkbox(
                value=False, label="Also save a .zip of the dataset",
                info="A single archive next to the folder — handy for uploading to a cloud trainer.")
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
                    exp_hf_private = gr.Checkbox(value=True, label="Private (recommended)")
                btn_publish_hf = gr.Button("⬆ Publish to Hugging Face")
                exp_hf_note = gr.Markdown()

        with gr.Tab("⑤ Train (configs, optional)"):
            gr.Markdown(
                "Generate a ready-to-edit LoRA training config for your dataset. "
                "**ai-toolkit** produces a one-command `config.yaml` (`python run.py …`); "
                "**musubi-tuner** produces a `dataset.toml` plus a command template where "
                "you fill in your local model paths. Nothing is launched or executed here — "
                "the config is written into the dataset folder and the run command is shown.")
            from studio import user_config as _uc

            _ai_presets = TRAINER_MODELS["ai-toolkit"]
            with gr.Row():
                with gr.Column(scale=1):
                    tr_trainer = gr.Radio(TRAINER_CHOICES, value="ai-toolkit", label="Trainer",
                                          info="ai-toolkit is one-command; musubi/kohya emit a "
                                               "config plus a run-command template.")
                    tr_path = gr.Textbox(label="Trainer install path (saved on this machine)",
                                         value=_uc.get_trainer_path("ai-toolkit"),
                                         placeholder=r"C:\ai-toolkit",
                                         info="Only used to compose the displayed run command.")
                    tr_save_path = gr.Button("💾 Save install path")
                    tr_path_note = gr.Markdown()
                    tr_model = gr.Dropdown([(p.label, p.key) for p in _ai_presets],
                                           value=_ai_presets[0].key, label="Model",
                                           info="Pick your target base model. <<FILL>> presets "
                                                "need you to supply a model path.")
                    tr_name = gr.Textbox(label="LoRA name", placeholder="sysnootles-lora",
                                         info="Output name for the trained LoRA file.")
                    tr_trigger = gr.Textbox(label="Trigger word (used in the sample prompt)",
                                            placeholder="sysnootles",
                                            info="Should match the trigger you captioned with.")
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
                        tr_steps = gr.Number(value=_ai_presets[0].steps, precision=0,
                                             label="Steps",
                                             info="Auto-suggested from image count on Inspect.")
                        tr_lr = gr.Number(value=_ai_presets[0].lr, label="Learning rate",
                                          info="1e-4 is a common starting point.")
                    tr_multi_res = gr.Checkbox(
                        value=True, label="Multi-resolution buckets",
                        info="Bucket by the dataset's real aspect ratios instead of "
                             "forcing one square resolution.")
                with gr.Column(scale=2):
                    tr_dataset = gr.Textbox(label="Dataset folder (auto-filled by ④ Export)")
                    btn_inspect = gr.Button("🔍 Inspect dataset & suggest steps")
                    tr_stats = gr.Markdown()
                    tr_gen = gr.Button("⑤ Generate training config", variant="primary")
                    tr_result = gr.Textbox(label="Result / run command", lines=14)

    log_box = gr.Textbox(label="Log", lines=8)

    # ---------- wiring ----------

    btn_pre.click(
        do_preprocess,
        [pre_files, pre_folder, target, restore_mode, restore_backend, isolate,
         isolation_backend, subject_prompt, exclude_prompt, pre_tighten],
        [prep_gallery, pre_note, log_box, gen_src_folder, cap_folder]) \
           .then(lambda s, e: (s, e), [subject_prompt, exclude_prompt],
                 [gen_subject, gen_exclude])

    # Header dataset-type selector retunes type-dependent controls across tabs
    # (and persists the choice). demo.load applies the same handler on launch so
    # a remembered Style/Concept type arrives with its defaults already set.
    type_outputs = [isolate, subject_prompt,
                    gen_type_note, gen_name, refresh, plan, btn_gen, btn_regen,
                    btn_outfits, btn_outfits_clear, wardrobe_note,
                    gen_exclude_props, gen_subject,
                    cap_name, cap_trigger, cap_sparse, exp_name]
    dataset_type.change(on_dataset_type_change, [dataset_type, gen_name], type_outputs)
    demo.load(on_dataset_type_change, [dataset_type, gen_name], type_outputs)

    refresh.click(refresh_plan, [gen_name, dataset_type], [plan])
    btn_outfits.click(randomize_outfits, [plan], [plan, plan_note])
    btn_outfits_clear.click(clear_outfits, [plan], [plan, plan_note])
    btn_save_plan.click(do_save_plan, [plan, plan_name], [plan_note])
    btn_load_plan.click(do_load_plan, [plan_name], [plan, plan_note])
    refresh_models.click(refresh_cloud_models, [], [cloud_model])

    def _force_refresh():
        return refresh_cloud_models(force=True)

    force_refresh_models.click(_force_refresh, [], [cloud_model])
    engine.change(estimate_cost, [engine, cloud_model, plan], [cost])
    cloud_model.change(estimate_cost, [engine, cloud_model, plan], [cost])
    plan.change(estimate_cost, [engine, cloud_model, plan], [cost])

    gen_inputs = [gen_files, gen_src_folder, plan, engine, cloud_model,
                  gen_exclude_props, gen_isolate, gen_iso_backend, gen_subject,
                  gen_exclude, gen_front]
    btn_gen.click(do_generate, gen_inputs + [gen_out_dir, results_state],
                  [results_state, gen_gallery, keep, log_box, gen_out_dir, cap_folder]) \
           .then(_fill_if_empty, [cap_name, gen_name], [cap_name])
    btn_regen.click(do_regenerate, gen_inputs + [gen_out_dir, results_state, keep],
                    [results_state, gen_gallery, keep, log_box])
    btn_disk.click(do_refresh_disk, [results_state, gen_out_dir],
                   [results_state, gen_gallery, keep, log_box])
    btn_send.click(send_kept_to_caption, [results_state, keep, gen_out_dir],
                   [cap_folder, cap_gallery, cap_select, cap_note])

    btn_load.click(load_caption_folder, [cap_folder], [cap_gallery, cap_select, cap_note]) \
            .then(_editor_choices, [cap_folder], [cap_edit_file])
    cap_edit_file.change(load_one_caption, [cap_folder, cap_edit_file], [cap_edit_text])
    btn_edit_load.click(load_one_caption, [cap_folder, cap_edit_file], [cap_edit_text])
    btn_edit_save.click(save_one_caption, [cap_folder, cap_edit_file, cap_edit_text],
                        [cap_result])
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
                   [cap_folder, cap_select, captioner, cap_name, cap_trigger, cap_gemini_model,
                    cap_style, cap_gen_thr, cap_char_thr, cap_prefix, cap_suffix,
                    cap_blacklist, cap_rating, cap_underscores, dataset_type, cap_sparse],
                   [test_caption])
    btn_caption.click(
        do_caption,
        [cap_folder, cap_select, captioner, cap_name, cap_trigger, cap_gemini_model, cap_style,
         cap_gen_thr, cap_char_thr, cap_prefix, cap_suffix,
         cap_blacklist, cap_rating, cap_underscores, cap_skip, dataset_type, cap_sparse,
         exp_folders, exp_name, exp_trigger],
        [cap_gallery, cap_select, cap_result, log_box, exp_folders, exp_name, exp_trigger,
         cap_analysis]) \
               .then(_editor_choices, [cap_folder], [cap_edit_file])
    btn_lint.click(do_analyze_captions, [cap_folder, cap_trigger], [cap_analysis])

    btn_load_preview.click(load_export_preview, [exp_folders, exp_dup_dist],
                           [exp_gallery, exp_select, exp_preview_note])
    btn_export.click(do_export,
                     [exp_select, exp_name, exp_trigger, output_root, exp_zip, dataset_type],
                     [exp_result, tr_dataset, exp_ds_dir]) \
              .then(inspect_dataset, [tr_dataset, dataset_type], [tr_stats, tr_steps]) \
              .then(_fill_if_empty, [tr_name, exp_name], [tr_name]) \
              .then(_fill_if_empty, [tr_trigger, exp_trigger], [tr_trigger])
    btn_publish_hf.click(do_publish_hf, [exp_ds_dir, exp_hf_repo, exp_hf_private],
                         [exp_hf_note])

    tr_hparams = [tr_res, tr_rank, tr_alpha, tr_steps, tr_lr, tr_batch]
    tr_trainer.change(on_trainer_change, [tr_trainer],
                      [tr_model, tr_path] + tr_hparams)
    tr_model.change(on_model_change, [tr_trainer, tr_model], tr_hparams)
    tr_save_path.click(save_trainer_path, [tr_trainer, tr_path], [tr_path_note])
    btn_inspect.click(inspect_dataset, [tr_dataset, dataset_type], [tr_stats, tr_steps])
    tr_gen.click(do_generate_train_config,
                 [tr_trainer, tr_model, tr_dataset, tr_path, tr_name, tr_trigger]
                 + tr_hparams + [tr_multi_res, dataset_type],
                 [tr_result])

    demo.load(_check_for_update, None, update_notice)

if __name__ == "__main__":
    # Bound to localhost on purpose: no auth layer, and .env keys are reachable
    # through the process. Do not expose publicly / use share=True.
    # allowed_paths lets the galleries display images in user-chosen input/output
    # folders on any drive (Gradio otherwise refuses paths outside the CWD/temp
    # dir). It is fixed at launch, so it can't be narrowed per-request; the
    # consequence is that the local file endpoint can serve any file the process
    # can read. Safe ONLY because of the localhost-only, no-auth bind above — see
    # the Security posture note in docs/ARCHITECTURE.md.
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True,
                allowed_paths=_allowed_media_paths())
