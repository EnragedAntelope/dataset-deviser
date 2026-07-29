"""Headless CLI. Every stage is its own subcommand and runs standalone:

  python cli.py preprocess ./sources --out ./prepped
  python cli.py generate ./prepped --name "Sy Snootles" --engine comfyui
  python cli.py caption ./any/folder --trigger sysnootles      # .txt sidecars
  python cli.py export ./prepped ./generated --name "Sy Snootles"
  python cli.py build img.png --name "Sy Snootles" --trigger sysnootles  # all four
"""

from __future__ import annotations

from pathlib import Path

import typer

from studio import env_keys, pipeline
from studio.config import CAPTIONERS_BY_KEY, list_images, settings
from studio.shotplan import plan_for_type

app = typer.Typer(add_completion=False, help=__doc__)


def _expand(paths: list[Path]) -> list[Path]:
    """Accept image files and/or folders of images."""
    out: list[Path] = []
    for p in paths:
        out.extend(list_images(p) if p.is_dir() else [p])
    if not out:
        raise typer.BadParameter("No images found in the given paths.")
    return out


def _echo_cloud_estimate(engine: str, cloud_model: str, n_shots: int) -> None:
    """Warn what a cloud run will cost before it starts billing the user."""
    if engine != "gemini":
        return
    from studio.config import CLOUD_IMAGE_PRICES, load_cloud_model_cache

    model_id = cloud_model or settings.gemini_image_model
    price = CLOUD_IMAGE_PRICES.get(model_id)
    for m in load_cloud_model_cache() or []:
        if m.get("model_id") == model_id and m.get("price") is not None:
            price = m["price"]
            break
    if price:
        typer.echo(f"Cloud engine: ~${n_shots * price:.2f} estimated for {n_shots} "
                   f"images (build-time estimate, billed to your Google API key).")


def _check_caption_style(style: str) -> str:
    """Validate the --caption-style value so a typo fails fast, not silently."""
    style = style.strip().lower()
    if style not in ("prose", "tags", "e621"):
        raise typer.BadParameter("--caption-style must be 'prose', 'tags' or 'e621'.")
    return style


def _check_dataset_type(dataset_type: str) -> str:
    """Validate --dataset-type so a typo fails fast, not silently."""
    from studio.config import DATASET_TYPES

    dt = dataset_type.strip().lower()
    if dt not in DATASET_TYPES:
        raise typer.BadParameter(
            f"--dataset-type must be one of {', '.join(DATASET_TYPES)}.")
    return dt


def _plan_for(dataset_type: str, name: str) -> list:
    """The shot plan for this dataset type — the same choice the ② tab makes.

    Style never generates: an aesthetic can't be synthesized from a reference,
    so we say so and exit instead of silently producing a character turnaround
    (which, on the cloud engine, would also bill the user for it).
    """
    if dataset_type == "style":
        raise typer.BadParameter(
            "Style datasets have no synthetic generation — collect your own images "
            "that share the look, then run `caption` and `export` on that folder.")
    return plan_for_type(dataset_type, name)


def _props_default(exclude_props: bool | None, dataset_type: str) -> bool:
    """Resolve --exclude-props/--keep-props when neither was passed.

    The exclusion clause names bags, held objects and clothing, so it only makes
    sense for a character; a concept dataset defaults to keeping props.
    """
    if exclude_props is None:
        return dataset_type == "character"
    return exclude_props


def _dress(shots: list, dataset_type: str = "character") -> list:
    """Fill angle/pose shots with random unisex outfits (close-ups stay blank).

    Wardrobe is a character-only idea — an object has no clothing — so it is a
    no-op (with a note) for any other dataset type.
    """
    from studio.wardrobe import OUTFIT_SHOT_KINDS, random_outfits

    if dataset_type != "character":
        typer.echo(f"Skipping outfit randomization: not applicable to a "
                   f"{dataset_type} dataset.")
        return shots
    targets = [s for s in shots if s.kind in OUTFIT_SHOT_KINDS]
    outfits = random_outfits(len(targets))
    dressed = dict(zip((s.id for s in targets), outfits))
    return [s.model_copy(update={"outfit": dressed[s.id]}) if s.id in dressed else s
            for s in shots]


@app.command()
def preprocess(
    inputs: list[Path] = typer.Argument(..., exists=True, help="Image files and/or folders"),
    out: Path = typer.Option(None, help="Output folder (default: new runs/ subfolder)"),
    target: int = typer.Option(settings.target_long_side, help="Long-side resolution"),
    restore: bool = typer.Option(None, "--restore/--no-restore",
                                 help="Force restoration on/off (default: auto)"),
    restore_backend: str = typer.Option(settings.restore_backend, help="auto | comfyui | basic"),
    isolate: bool = typer.Option(True, "--isolate/--no-isolate",
                                 help="Cut out subject, drop background/props"),
    isolation_backend: str = typer.Option(settings.isolation_backend, help="builtin | comfyui"),
    subject_prompt: str = typer.Option("character", help="SAM3 prompt for what to keep"),
    exclude_prompt: str = typer.Option("", help="SAM3 prompt for held props to remove"),
    tighten: bool = typer.Option(
        False, "--tighten/--no-tighten",
        help="Crop to the subject's bounding box after isolation (less white padding)"),
):
    """Restore/upscale/isolate images (standalone)."""
    out = out or pipeline.new_run_dir("prepped")
    pipeline.preprocess_sources(
        _expand(inputs), out, target=target, force_restore=restore, isolate=isolate,
        subject_prompt=subject_prompt, exclude_prompt=exclude_prompt,
        restore_backend=restore_backend, isolation_backend=isolation_backend,
        tighten_crop=tighten, progress=typer.echo)
    typer.echo(f"Done: {out}")


@app.command()
def generate(
    references: list[Path] = typer.Argument(..., exists=True,
                                            help="Reference image files and/or folders"),
    out: Path = typer.Option(None, help="Output folder (default: new runs/ subfolder)"),
    name: str = typer.Option("", help="Character/concept name used in prompts"),
    dataset_type: str = typer.Option(
        "character", "--dataset-type",
        help="character (24-shot set) | concept (18-shot object set); style does not generate"),
    engine: str = typer.Option(settings.default_engine, help="gemini (cloud) or comfyui (local)"),
    cloud_model: str = typer.Option("", help=f"Cloud image model (default {settings.gemini_image_model})"),
    max_shots: int = typer.Option(0, help="Limit number of shots (0 = full plan)"),
    isolate_angles: bool = typer.Option(False, help="Isolate generated angle shots onto white"),
    subject_prompt: str = typer.Option(
        "character", help="SAM3 prompt if isolating (name the object for a concept)"),
    exclude_prompt: str = typer.Option("", help="Props to remove when isolating"),
    exclude_props: bool = typer.Option(
        None, "--exclude-props/--keep-props",
        help="Ask the generator to omit bags/held objects from the reference "
             "(default: on for character, off for concept — the clause is character-worded)"),
    randomize_outfits: bool = typer.Option(
        False, help="Dress angle/pose shots in random unisex outfits (character only)"),
    front: bool = typer.Option(False, help="Jump ComfyUI's pending queue"),
):
    """Generate the shot set from reference image(s) (standalone)."""
    dtype = _check_dataset_type(dataset_type)
    out = out or pipeline.new_run_dir("generated")
    shots = _plan_for(dtype, name)
    if max_shots:
        shots = shots[:max_shots]
    if randomize_outfits:
        shots = _dress(shots, dtype)
    _echo_cloud_estimate(engine, cloud_model, len(shots))
    results = pipeline.generate_shots(
        _expand(references), shots, engine, out, cloud_model=cloud_model,
        isolate_angles=isolate_angles, subject_prompt=subject_prompt,
        exclude_prompt=exclude_prompt, exclude_props=_props_default(exclude_props, dtype),
        front=front, progress=typer.echo)
    if not any(r.path for r in results):
        raise typer.Exit(1)
    typer.echo(f"Done: {out}")


@app.command()
def caption(
    folder: Path = typer.Argument(..., exists=True, file_okay=False,
                                  help="Folder of images to tag"),
    captioner: str = typer.Option(settings.default_captioner,
                                  help=f"one of {list(CAPTIONERS_BY_KEY)}"),
    name: str = typer.Option("", help="Character name used in captions"),
    trigger: str = typer.Option("", help="Trigger word placed first in every caption"),
    model: str = typer.Option("", help="Model id (gemini captioner; blank = default)"),
    caption_style: str = typer.Option(
        "prose", "--caption-style",
        help="prose (natural language), tags (Danbooru: SDXL/Illustrious) or e621 (furry/Pony)"),
    dataset_type: str = typer.Option(
        "character", "--dataset-type",
        help="character | style | concept — frames what the caption describes"),
    sparse: bool = typer.Option(
        False, "--sparse",
        help="Style datasets only: trigger + a few words of content (stronger style)"),
    prefix: str = typer.Option(
        "", help="Fixed text added before every caption (e.g. Pony 'score_9, score_8_up')"),
    suffix: str = typer.Option("", help="Fixed text added after every caption"),
    drop_tags: str = typer.Option(
        "", "--drop-tags",
        help="Comma-separated tags to strip from tag captions (e.g. 'watermark, signature')"),
    rating_tags: bool = typer.Option(
        False, "--rating-tags/--no-rating-tags",
        help="Append the tagger's top rating tag (WD/Danbooru taggers only)"),
    keep_underscores: bool = typer.Option(
        False, "--keep-underscores",
        help="Keep raw booru underscores in tagger output (long_hair, not 'long hair')"),
    skip_captioned: bool = typer.Option(
        False, "--skip-captioned",
        help="Leave images that already have a non-empty .txt caption untouched"),
):
    """Write .txt caption sidecars for every image in a folder (standalone).

    The `custom` captioner reuses the endpoint saved by the UI's Caption tab, so
    the CLI and UI behave identically.
    """
    from studio.captioner import (
        CaptionerConfigError,
        caption_folder,
        merge_tagger_overrides,
        resolve_captioner_config,
    )

    style = _check_caption_style(caption_style)
    dtype = _check_dataset_type(dataset_type)
    try:
        model_override, spec_overrides = resolve_captioner_config(captioner, model)
    except CaptionerConfigError as e:
        typer.echo(str(e))
        raise typer.Exit(1)
    spec_overrides = merge_tagger_overrides(
        captioner, spec_overrides, include_rating=rating_tags,
        keep_underscores=keep_underscores)
    caption_folder(folder, captioner, name, trigger, progress=typer.echo,
                   model_override=model_override, spec_overrides=spec_overrides, style=style,
                   prefix=prefix, suffix=suffix, skip_existing=skip_captioned,
                   blacklist=drop_tags, dataset_type=dtype, sparse=sparse)


@app.command()
def lint(
    folder: Path = typer.Argument(..., exists=True, file_okay=False,
                                  help="Folder of captioned images to analyze"),
    trigger: str = typer.Option("", help="Trigger expected first in every caption"),
):
    """Advisory caption health + tag-frequency report for a folder (standalone).

    Flags empty / short / trigger-missing / identical captions and, for tag
    datasets, tags present on nearly every image. Never modifies anything.
    """
    from studio.caption_lint import analyze_folder, markdown_summary

    report, ubiquitous = analyze_folder(folder, trigger)
    # Strip Markdown emphasis for a clean terminal read.
    text = markdown_summary(report, ubiquitous).replace("**", "").replace("`", "")
    typer.echo(text)


@app.command()
def export(
    folders: list[Path] = typer.Argument(..., exists=True, file_okay=False,
                                         help="Folder(s) of captioned images"),
    name: str = typer.Option("", help="Character name"),
    trigger: str = typer.Option("", help="Trigger word"),
    dataset_type: str = typer.Option(
        "character", "--dataset-type", help="character | style | concept (recorded in metadata)"),
    output_root: Path = typer.Option(settings.output_root),
    zip_: bool = typer.Option(False, "--zip", help="Also write a .zip of the dataset"),
    publish_hf: str = typer.Option(
        "", "--publish-hf",
        help="Publish to the HF Hub as this dataset id (needs a write HF_TOKEN)"),
    hf_private: bool = typer.Option(
        True, "--hf-private/--hf-public", help="Visibility of the published HF dataset"),
):
    """Package captioned folders into a flat NN.png/NN.txt dataset (standalone)."""
    from studio.package import package_dataset, resolve_export_items

    dtype = _check_dataset_type(dataset_type)
    candidates = [img for folder in folders for img in list_images(folder)]
    res = resolve_export_items(candidates)
    if not res.items:
        typer.echo("No captioned images found — run `caption` first.")
        raise typer.Exit(1)
    skipped = res.missing + res.empties
    if skipped:
        typer.echo(f"Skipping {len(skipped)} image(s) without a usable caption: "
                   f"{', '.join(skipped)}")
    metadata = {"character_name": name, "trigger": trigger, "dataset_type": dtype,
                "source_folders": [str(f) for f in folders],
                "skipped_uncaptioned": res.missing, "skipped_empty_caption": res.empties}
    ds = package_dataset(res.items, output_root, name, trigger, metadata)
    typer.echo(f"Dataset written to {ds}")
    if zip_:
        from studio.package import zip_dataset

        typer.echo(f"Zipped: {zip_dataset(ds)}")
    if publish_hf.strip():
        from studio.hf_publish import HFPublishError, publish_dataset

        try:
            url = publish_dataset(ds, publish_hf, private=hf_private, progress=typer.echo)
        except HFPublishError as e:
            typer.echo(str(e))
            raise typer.Exit(1)
        typer.echo(f"Published: {url}")


@app.command()
def build(
    images: list[Path] = typer.Argument(..., exists=True, readable=True),
    name: str = typer.Option("", help="Character/concept name used in prompts + captions"),
    trigger: str = typer.Option("", help="LoRA trigger word placed first in every caption"),
    engine: str = typer.Option(settings.default_engine, help="gemini (cloud) or comfyui (local)"),
    captioner: str = typer.Option(settings.default_captioner,
                                  help=f"one of {list(CAPTIONERS_BY_KEY)}"),
    caption_style: str = typer.Option(
        "prose", "--caption-style",
        help="prose (natural language), tags (Danbooru: SDXL/Illustrious) or e621 (furry/Pony)"),
    dataset_type: str = typer.Option(
        "character", "--dataset-type",
        help="character | style | concept — picks the shot plan (style skips generation), "
             "frames captions, and is recorded in metadata"),
    sparse: bool = typer.Option(
        False, "--sparse", help="Style datasets only: trigger + a few words of content"),
    target: int = typer.Option(settings.target_long_side, help="Long-side resolution"),
    output_root: Path = typer.Option(settings.output_root),
    restore: bool = typer.Option(None, "--restore/--no-restore",
                                 help="Force restoration on/off (default: auto)"),
    max_shots: int = typer.Option(0, help="Limit number of shots (0 = full plan)"),
    isolate: bool = typer.Option(
        None, "--isolate/--no-isolate",
        help="Cut out subject, drop background/props (default: on, except style — "
             "a style is whole-image)"),
    tighten: bool = typer.Option(
        False, "--tighten/--no-tighten",
        help="Crop to the subject's bounding box after isolation (less white padding)"),
    subject_prompt: str = typer.Option("character", help="SAM3 prompt for what to keep"),
    exclude_prompt: str = typer.Option("", help="SAM3 prompt for held props to remove"),
    cloud_model: str = typer.Option("", help=f"Cloud image model (default {settings.gemini_image_model})"),
    exclude_props: bool = typer.Option(
        None, "--exclude-props/--keep-props",
        help="Ask the generator to omit bags/held objects from the reference "
             "(default: on for character, off for concept)"),
    randomize_outfits: bool = typer.Option(
        False, help="Dress angle/pose shots in random unisex outfits (character only)"),
    zip_: bool = typer.Option(False, "--zip", help="Also write a .zip of the dataset"),
    prefix: str = typer.Option(
        "", help="Fixed text added before every caption (e.g. Pony 'score_9, score_8_up')"),
    suffix: str = typer.Option("", help="Fixed text added after every caption"),
    drop_tags: str = typer.Option(
        "", "--drop-tags",
        help="Comma-separated tags to strip from tag captions (e.g. 'watermark, signature')"),
):
    """Full pipeline: preprocess -> generate -> caption -> export."""
    from studio.captioner import caption_images
    from studio.package import package_dataset

    style = _check_caption_style(caption_style)
    dtype = _check_dataset_type(dataset_type)
    run_dir = pipeline.new_run_dir(name or trigger)
    typer.echo(f"Run dir: {run_dir}")

    # Isolation defaults follow the dataset type (a style is whole-image), the
    # same rule the ① tab applies when the header type changes.
    do_isolate = (dtype != "style") if isolate is None else isolate
    reports = pipeline.preprocess_sources(
        _expand(images), run_dir / "prepped", target=target, force_restore=restore,
        isolate=do_isolate, subject_prompt=subject_prompt, exclude_prompt=exclude_prompt,
        tighten_crop=tighten, progress=typer.echo)

    # Style has no synthetic generation — go straight from the preprocessed
    # sources to captioning instead of running (and billing) a character
    # turnaround the user can't use.
    results = []
    kept: list[Path] = []
    if dtype == "style":
        typer.echo("Style dataset: skipping ② generation — captioning your own images.")
    else:
        shots = _plan_for(dtype, name)
        if max_shots:
            shots = shots[:max_shots]
        if randomize_outfits:
            shots = _dress(shots, dtype)
        _echo_cloud_estimate(engine, cloud_model, len(shots))

        results = pipeline.generate_shots(
            [r.output for r in reports], shots, engine, run_dir / "generated",
            cloud_model=cloud_model, isolate_angles=do_isolate, subject_prompt=subject_prompt,
            exclude_prompt=exclude_prompt,
            exclude_props=_props_default(exclude_props, dtype), progress=typer.echo)
        kept = [r.path for r in results if r.path]
        if not kept:
            typer.echo("No shots succeeded; aborting before captioning.")
            raise typer.Exit(1)

    all_images = [r.output for r in reports] + kept
    items = caption_images(all_images, captioner, name, trigger, progress=typer.echo,
                           style=style, prefix=prefix, suffix=suffix, blacklist=drop_tags,
                           dataset_type=dtype, sparse=sparse)
    metadata = {
        "character_name": name,
        "trigger": trigger,
        "dataset_type": dtype,
        "captioner": captioner,
        "caption_style": style,
        "sources": [str(s) for s in images],
    }
    if results:  # generation ran (character/concept)
        metadata["engine"] = engine
        metadata["shots"] = [{"id": r.shot.id, "seed": r.seed, "error": r.error}
                             for r in results]
    ds = package_dataset(items, output_root, name, trigger, metadata)
    if zip_:
        from studio.package import zip_dataset

        typer.echo(f"Zipped: {zip_dataset(ds)}")
    typer.echo(f"\nDone: {ds}")


# --- setup / diagnostics ---------------------------------------------------
# These two exist because a broken install used to announce itself only by a
# console window closing. The setup scripts call `keys --setup` instead of
# hand-rolling .env editing in batch/bash (see studio/env_keys.py).

@app.command()
def doctor() -> None:
    """Check this install: Python, dependencies, API keys, ComfyUI.

    Run this first whenever something doesn't start or a cloud option says it has
    no key. Exits non-zero if anything actually needs fixing.
    """
    from studio.doctor import build_report, render

    report = build_report()
    typer.echo(render(report))
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def keys(
    set_: str = typer.Option(
        "", "--set", metavar="NAME",
        help=f"Set or change one key ({', '.join(env_keys.KEY_NAMES)}). Prompts without echo."),
    setup: bool = typer.Option(
        False, "--setup",
        help="Walk through every key (Enter keeps the current value). Used by setup.bat/setup.sh."),
) -> None:
    """Show, set or change the API keys in .env.

    With no options, lists which keys are configured (masked, never the whole
    value). `--set NAME` changes one. `--setup` walks all of them, so re-running
    setup can *fix* a mistyped key instead of skipping it.

    Output here is deliberately ASCII: setup.bat runs it in a plain cmd window and
    users redirect it, where a non-cp1252 glyph raises UnicodeEncodeError.
    """
    from studio.doctor import apply_key_input, key_report, prompt_secret

    env_path = env_keys.default_env_path()

    if set_ and setup:
        raise typer.BadParameter("Use either --set or --setup, not both.")

    if not set_ and not setup:
        typer.echo(f"API keys in {env_path}:\n")
        typer.echo("\n".join(key_report(env_path)))
        typer.echo("\nChange one:  python cli.py keys --set GEMINI_API_KEY")
        return

    names = [set_.strip().upper()] if set_ else list(env_keys.KEY_NAMES)
    for name in names:
        if name not in env_keys.KEY_SPECS_BY_NAME:
            raise typer.BadParameter(
                f"Unknown key '{name}'. Known keys: {', '.join(env_keys.KEY_NAMES)}")

    if setup:
        typer.echo("\n--- API keys (all optional - press Enter to skip or keep) ---")
        typer.echo(f"Stored only in {env_path} (gitignored, never uploaded).\n")

    current = env_keys.read_env(env_path)
    for name in names:
        spec = env_keys.KEY_SPECS_BY_NAME[name]
        existing = current.get(name, "") or ""
        typer.echo(f"{name}: {spec.unlocks}")
        typer.echo(f"  Get one at {spec.url}")
        if existing:
            typer.echo(f"  Currently set ({env_keys.mask(existing)}). "
                       "Enter a new value to replace it, '-' to remove it, "
                       "or press Enter to keep it.")
        else:
            typer.echo(f"  Not set - without it, {spec.without}.")
        try:
            # Hidden where there is a terminal, so the key stays out of the console
            # scrollback; the old batch prompt echoed it in the clear.
            raw = prompt_secret(f"  {name} (hidden, Enter to skip): ")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\n  Skipped.")
            typer.echo("")
            continue
        try:
            changed = apply_key_input(env_path, name, raw)
        except ValueError as e:
            typer.echo(f"  [skipped] {e}")
            typer.echo("")
            continue
        if changed:
            typer.echo(f"  Saved to {env_path.name}."
                       if raw.strip() != "-" else f"  Removed from {env_path.name}.")
        elif existing:
            typer.echo("  Kept the existing value.")
        else:
            typer.echo("  Skipped.")
        typer.echo("")

    if setup:
        typer.echo("Change any of these later with:  python cli.py keys")


@app.command("custom-endpoint")
def custom_endpoint(
    base_url: str = typer.Option(
        "", "--base-url", help="Base URL, e.g. https://openrouter.ai/api/v1"),
    model: str = typer.Option("", "--model", help="Model id/name at that endpoint"),
    key_env: str = typer.Option(
        "", "--key-env",
        help="Name of the environment variable holding the API key (not the key itself)"),
    spacing: float = typer.Option(
        0.0, "--spacing", help="Minimum seconds between requests (rate-limit spacing)"),
) -> None:
    """Show or set the custom OpenAI-compatible captioner (--captioner custom).

    With no options, shows the currently saved config. To save one, pass at
    least --base-url and --model; any option left blank keeps its current saved
    value, so `python cli.py custom-endpoint --spacing 3` alone just adjusts
    spacing. Saved to the same .cache/user_settings.json the UI's Caption tab
    writes to, so the CLI and UI share one config either way sets it.

    The API key itself is never stored here — --key-env only records which
    environment variable (in .env or the shell) to read it from at caption time.
    """
    from studio import user_config

    cfg = user_config.get_custom_captioner()

    if not base_url and not model and not key_env and not spacing:
        if not cfg.get("base_url"):
            typer.echo("No custom endpoint configured yet.")
            typer.echo(
                "Set one:  python cli.py custom-endpoint --base-url URL --model NAME "
                "[--key-env VAR_NAME] [--spacing SECONDS]")
            return
        typer.echo(f"Base URL:  {cfg.get('base_url', '')}")
        typer.echo(f"Model:     {cfg.get('model', '')}")
        typer.echo(f"Key env:   {cfg.get('api_key_env') or '(none set)'}")
        typer.echo(f"Spacing:   {cfg.get('min_interval_s', 0.0)}s")
        return

    final_base_url = base_url.strip() or cfg.get("base_url", "")
    final_model = model.strip() or cfg.get("model", "")
    if not final_base_url or not final_model:
        raise typer.BadParameter(
            "Both --base-url and --model are needed the first time you set this "
            "(nothing saved yet to fall back on).")

    user_config.set_custom_captioner(
        final_base_url, final_model,
        key_env.strip() or cfg.get("api_key_env", ""),
        spacing or cfg.get("min_interval_s", 0.0),
    )
    typer.echo("Saved.")
    typer.echo(
        "Use it with:  python cli.py caption ./folder --captioner custom --trigger ...")
    key_env_final = key_env.strip() or cfg.get("api_key_env", "")
    if key_env_final:
        typer.echo(f"Make sure {key_env_final} is set (in .env or your shell) before captioning.")
    else:
        typer.echo("No --key-env set — fine only if this endpoint needs no API key.")


if __name__ == "__main__":
    app()
