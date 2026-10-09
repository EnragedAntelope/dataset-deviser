"""Independent pipeline stages, shared by the CLI and UI.

Every stage is standalone: it takes explicit input paths and writes to an
explicit output folder. Chaining them (preprocess -> generate -> caption ->
export) is a convenience the UI/CLI provide, never a requirement.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from studio.config import settings
from studio.engines.base import GenerationError
from studio.jobs import ShouldStop, should_stop_now
from studio.package import slugify
from studio.preprocess import REFS_DIR, PreprocessReport, failed_report, preprocess
from studio.shotplan import Shot, apply_prop_exclusion, apply_wardrobe

ProgressFn = Callable[[str], None]
ANCHOR_SHOT = "angle-front"


@dataclass
class GenResult:
    shot: Shot
    path: Path | None
    seed: int
    error: str = ""


def new_run_dir(name: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = settings.runs_dir / f"{slugify(name or 'run')}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_engine(engine_key: str, cloud_model: str = "", front: bool = False):
    if engine_key == "comfyui":
        from studio.engines.comfyui import ComfyUIEngine

        return ComfyUIEngine(front=front)
    from studio.engines.gemini import GeminiEngine

    return GeminiEngine(model=cloud_model)


def preprocess_sources(
    sources: list[Path],
    out_dir: Path,
    target: int | None = None,
    force_restore: bool | None = None,
    isolate: bool = True,
    subject_prompt: str = "character",
    exclude_prompt: str = "",
    restore_backend: str = "",
    isolation_backend: str = "",
    tighten_crop: bool = False,
    alpha_cutout: bool = False,
    front: bool = False,
    should_stop: ShouldStop | None = None,
    progress: ProgressFn = print,
) -> list[PreprocessReport]:
    reports: list[PreprocessReport] = []
    for src in sources:
        # Cooperative stop, checked between images: the one in flight finishes
        # (a half-written PNG helps nobody) and the rest are simply not started.
        if should_stop_now(should_stop):
            progress(f"⏹ Stopped by request — {len(reports)} of {len(sources)} done, "
                     f"{len(sources) - len(reports)} not started.")
            break
        progress(f"Preprocessing {src.name}...")
        # One unusable source (SAM3 finds no subject, a truncated file, a
        # ComfyUI hiccup) must not discard the images already done. The failure
        # is recorded in the report list and the batch continues; the caller
        # decides how loudly to surface it. Before this, a single IsolationError
        # unwound the whole run and the UI's gr.Error threw the outputs away.
        try:
            rep = preprocess(src, out_dir, target=target, force_restore=force_restore,
                             isolate=isolate, subject_prompt=subject_prompt,
                             exclude_prompt=exclude_prompt, restore_backend=restore_backend,
                             isolation_backend=isolation_backend, tighten_crop=tighten_crop,
                             alpha_cutout=alpha_cutout, front=front, progress=progress)
        except Exception as e:
            progress(f"  SKIPPED {src.name}: {e}")
            reports.append(failed_report(src, str(e)))
            continue
        extra = f", isolated reference in {REFS_DIR}/" if rep.isolated else ""
        progress(
            f"  {src.name}: {rep.original_size[0]}x{rep.original_size[1]} -> "
            f"{rep.final_size[0]}x{rep.final_size[1]} ({rep.reason}{extra})"
        )
        reports.append(rep)
    failed = [r for r in reports if r.error]
    if failed:
        progress(f"Preprocess done: {len(reports) - len(failed)} succeeded, "
                 f"{len(failed)} skipped.")
    return reports


def generate_shots(
    sources: list[Path],
    shots: list[Shot],
    engine_key: str,
    out_dir: Path,
    cloud_model: str = "",
    isolate_angles: bool = False,
    subject_prompt: str = "character",
    exclude_prompt: str = "",
    isolation_backend: str = "",
    existing: list[GenResult] | None = None,
    only_ids: set[str] | None = None,
    exclude_props: bool = True,
    front: bool = False,
    anchor: bool = False,
    should_stop: ShouldStop | None = None,
    progress: ProgressFn = print,
) -> list[GenResult]:
    """Generate one image per shot from `sources` (identity references).

    `existing` + `only_ids` support regeneration: previous results for shots
    in only_ids are dropped and redone; everything else is kept.

    `exclude_props` asks the generator to omit bags/held objects carried in the
    reference, so they don't end up baked into every dataset image.

    `anchor` generates the front full-body view first and leads every other
    shot's references with it: partial or poor sources (a face in shadow, a
    body cut at the hips) then give way to one clean, complete view to copy.
    """
    if not sources:
        raise GenerationError("No reference images given — nothing to generate from.")
    engine = make_engine(engine_key, cloud_model, front=front)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [r for r in (existing or []) if only_ids is None or r.shot.id not in only_ids]
    todo = [s for s in shots if only_ids is None or s.id in only_ids]
    # A plan without the front view (a concept, or the row deleted) has no anchor.
    anchor_id = ANCHOR_SHOT if anchor else None
    # The anchor runs first; chained shots (back views built from a generated
    # side view) run last.
    todo.sort(key=lambda s: (s.id != anchor_id, bool(s.chain_from)))

    done: dict[str, Path] = {r.shot.id: r.path for r in results if r.path}
    stopped_at = 0
    for i, shot in enumerate(todo, 1):
        # Between-shot stop. Shots not started are left OUT of the results
        # entirely rather than recorded as failures — they were never attempted,
        # and "regenerate unchecked" then picks them up as the resume path.
        if should_stop_now(should_stop):
            stopped_at = i - 1
            progress(f"⏹ Stopped by request — {stopped_at} of {len(todo)} shot(s) "
                     f"generated, {len(todo) - stopped_at} not started. Use "
                     f"'Generate/regenerate UNCHECKED shots' to finish the rest.")
            break
        shot = apply_wardrobe(shot)  # fold the outfit column into the prompts
        if exclude_props:
            shot = apply_prop_exclusion(shot)
        seed = random.randint(0, 2**48)
        out = out_dir / f"{shot.id}.png"
        shot_sources = sources
        if anchor_id in done and shot.id != anchor_id:
            shot_sources = [done[anchor_id], *sources]
        if shot.chain_from and shot.chain_from in done:
            # Lead with the chained view so single-reference engines rotate
            # stepwise instead of hallucinating the far side of the character.
            shot_sources = [done[shot.chain_from], *shot_sources]
        progress(f"[{i}/{len(todo)}] {shot.id} ({engine_key})...")
        try:
            engine.generate(shot_sources, shot, out, seed)
            if isolate_angles and shot.kind == "angle":
                # Angle shots are turnaround views; strip whatever background
                # the model invented so props can't leak into the dataset.
                try:
                    from studio.isolate import isolate_subject

                    isolate_subject(out, out, subject_prompt, exclude_prompt,
                                    backend=isolation_backend, progress=progress,
                                    front=front)
                except Exception as e:
                    progress(f"  (isolation skipped: {e})")
            results.append(GenResult(shot, out, seed))
            done[shot.id] = out
        except GenerationError as e:
            progress(f"  FAILED: {e}")
            results.append(GenResult(shot, None, seed, error=str(e)))
    ok = sum(1 for r in results if r.path)
    progress(f"Generation done: {ok} succeeded, {len(results) - ok} failed.")
    return results


# ---------- full build: ①→② then ③→④(→⑤), shared by `cli build` and Quick build ----------

# A source ① enlarged more than this starts unticked in Quick build: upscaling
# invents detail, and the LoRA learns that as the subject's texture. It still
# serves as a reference.
UPSCALE_UNTICK = 1.33


def upscale_factor(report: PreprocessReport) -> float:
    """How much ① enlarged this source (1.0 = not at all, or shrunk)."""
    return max(report.final_size) / max(1, *report.original_size)


@dataclass
class BuildStart:
    """What ①→② made: the run folder, ①'s reports, ②'s references and results."""
    run_dir: Path
    reports: list[PreprocessReport]
    refs: list[Path]
    results: list[GenResult]

    @property
    def prepped(self) -> list[Path]:
        """The training copies ① wrote (a skipped source has none)."""
        return [r.output for r in self.reports if r.output]


def build_start(
    images: list[Path],
    name: str,
    engine_key: str,
    *,
    dataset_type: str = "character",
    run_dir: Path | None = None,
    cloud_model: str = "",
    target: int | None = None,
    restore: bool | None = None,
    isolate: bool | None = None,
    subject_prompt: str = "character",
    exclude_prompt: str = "",
    tighten: bool = False,
    isolate_angles: bool = False,
    shot_style: str = "match",
    shot_style_text: str = "",
    max_shots: int = 0,
    identity: str = "identity",
    exclude_props: bool | None = None,
    anchor: bool = False,
    front: bool = False,
    should_stop: ShouldStop | None = None,
    progress: ProgressFn = print,
) -> BuildStart:
    """① preprocess, then ② generate from the isolated references.

    The highest-resolution source leads the references: the local engine takes
    its output aspect from the first one, and at one reference it is the only one.
    Style datasets stop after ① (an aesthetic can't be generated from a reference).
    """
    from studio.shotplan import plan_for_type
    from studio.wardrobe import dress

    run_dir = run_dir or new_run_dir(name)
    reports = preprocess_sources(
        images, run_dir / "prepped", target=target, force_restore=restore,
        isolate=(dataset_type != "style") if isolate is None else isolate,
        subject_prompt=subject_prompt, exclude_prompt=exclude_prompt, tighten_crop=tighten,
        front=front, should_stop=should_stop, progress=progress)
    ok = sorted((r for r in reports if r.output),
                key=lambda r: r.original_size[0] * r.original_size[1], reverse=True)
    start = BuildStart(run_dir, reports, [r.reference or r.output for r in ok], [])
    if dataset_type == "style":
        progress("Style dataset: skipping ② generation — captioning your own images.")
        return start
    if not start.refs or should_stop_now(should_stop):
        return start
    shots = plan_for_type(dataset_type, name, shot_style, shot_style_text)
    if max_shots:
        shots = shots[:max_shots]
    if identity == "identity" and dataset_type == "character":
        shots = dress(shots)
    start.results = generate_shots(
        start.refs, shots, engine_key, run_dir / "generated", cloud_model=cloud_model,
        isolate_angles=isolate_angles, subject_prompt=subject_prompt,
        exclude_prompt=exclude_prompt,
        exclude_props=(dataset_type == "character") if exclude_props is None else exclude_props,
        front=front, anchor=anchor, should_stop=should_stop, progress=progress)
    return start


def build_finish(
    start: BuildStart,
    images: list[Path],
    *,
    name: str,
    trigger: str,
    captioner: str,
    output_root: Path,
    dataset_type: str = "character",
    identity: str = "identity",
    engine_key: str = "",
    shot_style: str = "match",
    shot_style_text: str = "",
    caption_style: str = "prose",
    prefix: str = "",
    suffix: str = "",
    drop_tags: str = "",
    sparse: bool = False,
    holdout: int = 0,
    trainer: str = "",
    model_key: str = "",
    progress: ProgressFn = print,
) -> tuple[Path, str]:
    """③ caption `images`, ④ export them, and (with `trainer`) ⑤ write its configs.

    Returns the dataset folder and the train command ("" without a trainer).
    """
    from studio.captioner import caption_images
    from studio.package import package_dataset

    items = caption_images(images, captioner, name, trigger, progress=progress,
                           style=caption_style, prefix=prefix, suffix=suffix,
                           blacklist=drop_tags, dataset_type=dataset_type, sparse=sparse,
                           identity=identity)
    metadata: dict = {
        "character_name": name,
        "trigger": trigger,
        "dataset_type": dataset_type,
        "shot_style": shot_style,
        "shot_style_text": shot_style_text,
        "captioner": captioner,
        "caption_style": caption_style,
        "sources": [str(r.source) for r in start.reports],
    }
    if dataset_type == "character":
        metadata["identity"] = identity
    if start.results:  # generation ran (character/concept)
        metadata["engine"] = engine_key
        metadata["shots"] = [{"id": r.shot.id, "seed": r.seed, "error": r.error}
                             for r in start.results]
    ds = package_dataset(items, output_root, name, trigger, metadata, holdout=holdout)
    command = ""
    if trainer:
        from studio.trainer_configs import default_config

        _, command = default_config(ds, trainer, model_key, name=name, trigger=trigger,
                                    dataset_type=dataset_type, shot_style=shot_style,
                                    shot_style_text=shot_style_text)
    return ds, command
