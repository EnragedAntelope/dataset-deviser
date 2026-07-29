"""Independent pipeline stages, shared by the CLI and UI.

Every stage is standalone: it takes explicit input paths and writes to an
explicit output folder. Chaining them (preprocess -> generate -> caption ->
export) is a convenience the UI/CLI provide, never a requirement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from studio.config import settings
from studio.engines.base import GenerationError
from studio.package import slugify
from studio.preprocess import PreprocessReport, preprocess
from studio.shotplan import Shot, apply_prop_exclusion, apply_wardrobe

ProgressFn = Callable[[str], None]


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


def make_engine(engine_key: str, cloud_model: str = ""):
    if engine_key == "comfyui":
        from studio.engines.comfyui import ComfyUIEngine

        return ComfyUIEngine()
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
    progress: ProgressFn = print,
) -> list[PreprocessReport]:
    reports: list[PreprocessReport] = []
    for src in sources:
        progress(f"Preprocessing {src.name}...")
        rep = preprocess(src, out_dir, target=target, force_restore=force_restore,
                         isolate=isolate, subject_prompt=subject_prompt,
                         exclude_prompt=exclude_prompt, restore_backend=restore_backend,
                         isolation_backend=isolation_backend, tighten_crop=tighten_crop,
                         alpha_cutout=alpha_cutout, progress=progress)
        extra = ", subject isolated" if rep.isolated else ""
        progress(
            f"  {src.name}: {rep.original_size[0]}x{rep.original_size[1]} -> "
            f"{rep.final_size[0]}x{rep.final_size[1]} ({rep.reason}{extra})"
        )
        reports.append(rep)
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
    progress: ProgressFn = print,
) -> list[GenResult]:
    """Generate one image per shot from `sources` (identity references).

    `existing` + `only_ids` support regeneration: previous results for shots
    in only_ids are dropped and redone; everything else is kept.

    `exclude_props` asks the generator to omit bags/held objects carried in the
    reference, so they don't end up baked into every dataset image.
    """
    if not sources:
        raise GenerationError("No reference images given — nothing to generate from.")
    engine = make_engine(engine_key, cloud_model)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [r for r in (existing or []) if only_ids is None or r.shot.id not in only_ids]
    todo = [s for s in shots if only_ids is None or s.id in only_ids]
    # Chained shots (e.g. back views built from a generated side view) run last
    todo.sort(key=lambda s: bool(s.chain_from))

    done: dict[str, Path] = {r.shot.id: r.path for r in results if r.path}
    for i, shot in enumerate(todo, 1):
        shot = apply_wardrobe(shot)  # fold the outfit column into the prompts
        if exclude_props:
            shot = apply_prop_exclusion(shot)
        seed = random.randint(0, 2**48)
        out = out_dir / f"{shot.id}.png"
        shot_sources = sources
        if shot.chain_from and shot.chain_from in done:
            # Lead with the chained view so single-reference engines rotate
            # stepwise instead of hallucinating the far side of the character.
            shot_sources = [done[shot.chain_from], *sources]
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
