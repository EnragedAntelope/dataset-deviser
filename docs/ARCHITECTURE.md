# Architecture

Version: 0.13.2

```
app.py                  Gradio UI — thin wiring over the stage functions (5 tabs)
cli.py                  Typer CLI — one subcommand per stage + `build` for all four,
                        plus `doctor` (install self-check), `keys` (view/set the .env
                        API keys — what setup.bat/setup.sh call), and
                        `custom-endpoint` (view/set the custom OpenAI-compatible
                        captioner without the UI — see the CLI/UI parity gotcha)
setup.bat / setup.sh    One-time install: Python version gate, venv, GPU-or-CPU torch,
                        requirements, ONNX Runtime, then `cli.py keys --setup` and
                        `cli.py doctor`. Deliberately thin — see the shell gotchas
start.bat / start.sh    Launch the UI, and explain a non-zero exit instead of vanishing
studio/
  env_keys.py           The ONE implementation of reading/editing the .env API keys:
                        KEY_SPECS (what each key unlocks / where to get it),
                        read_env(), set_managed_key() (in-place update, newline-safe,
                        lands on the LDS_-alias spelling if that is the one in use),
                        mask(), key_status(). Replaced the per-shell versions in
                        setup.bat/setup.sh — see the shell gotchas
  doctor.py             Install self-check behind `cli.py doctor`: Python version,
                        required imports, torch/onnxruntime, ComfyUI reachability,
                        configured model filenames against ComfyUI's own list (via
                        `comfy_api.load_template`, reachable-only), and masked key
                        status naming what each missing key blocks. Every check
                        takes its inputs as parameters, so it is unit-tested
  config.py             Settings (env/.env overridable), captioner registry with
                        per-model prompt templates, engine/price tables. Also the two
                        dataset-folder-convention primitives: list_images() and
                        read_caption() (forgiving sidecar read — utf-8-sig +
                        errors=replace; see the caption-encoding gotcha). Owns the
                        DATASET_TYPES = (character|style|concept) constant and
                        CaptionerSpec.prompt_for(style, dataset_type, sparse):
                        character returns the tuned per-model templates verbatim;
                        style/concept COMPOSE their instruction from a per-type
                        framing clause + the prose/tags/e621 format directive (no
                        stored per-type templates — see the compose-not-store gotcha)
  pipeline.py           Stage orchestration: preprocess_sources(), generate_shots()
  preprocess.py         Restore (comfyui|basic|auto) + isolate + optional tighten-crop
                        + resize
  isolate.py            Subject isolation: builtin SAM3 (transformers) | comfyui.
                        crop_to_content() tightens the isolated (subject-on-white, or
                        subject-with-alpha) image to the subject's bounding box (opt-in,
                        both backends). alpha_cutout composites onto transparency
                        instead of white — builtin backend only, see the Gotcha
  tagger.py             ONNX booru-tagger backend: canonical tags straight from
                        image features — WD (Danbooru) and Z3D (e621) via one code
                        path, differing only in tag file + category SCHEMES. Pure
                        select/format logic is I/O-free (unit-tested); onnxruntime +
                        hub download are lazy. The "wd_tagger" captioner backend.
                        Opt-in extras: select_rating() appends the top rating tag
                        (Danbooru cat 9), format_tag(keep_underscores=) keeps raw
                        booru tokens
  dedupe.py             Advisory near-duplicate detection (perceptual dHash,
                        numpy-only) surfaced in the ④ export preview; the Hamming
                        distance (default 5) is a ④ slider
  quality.py            Advisory sharpness (blur) + exposure/contrast flags
                        (dark/bright/low-contrast) for the curate/export views
  caption_lint.py       Advisory caption analysis (pure string logic): health lint
                        (empty/short/trigger-missing/identical/over-CLIP-77-tokens) +
                        tag-frequency / near-ubiquitous-tag report for tag datasets.
                        estimate_clip_tokens() is a tokenizer-free best estimate;
                        folder_caption_kind() (tags|prose) drives the ④→⑤ sanity
                        check. Surfaced after ③ and in the ④ preview; CLI `lint`
  hf_publish.py         Optional, opt-in publish of a dataset folder to the HF Hub
                        (private by default; write HF_TOKEN; huggingface_hub lazy)
  captioner.py          Captioner backends (transformers | gemini | openai-compat |
                        wd_tagger), caption finalization, standalone caption_folder().
                        caption()/caption_images()/caption_folder() thread
                        dataset_type + sparse (Style-only) into prompt selection and
                        finalize_caption(style=, dataset_type=). Also: model_override
                        (Gemini picker), spec_overrides (custom endpoint), apply_affixes
                        (prefix/suffix), drop_blacklisted_tags (drop-list),
                        merge_tagger_overrides (③ tag controls), skip_existing
  package.py            Dataset export (NN.<ext>/NN.txt + metadata.json + metadata.jsonl
                        + README.txt); records dataset_type + a detected caption_style
                        in metadata.json; zip_dataset() bundles a folder into a .zip;
                        resolve_export_items() classifies candidates by caption sidecar
                        state (ready/empty/missing) — shared by the UI gate and the CLI
  shotplan.py           Shot plans + Shot model. default_plan() = Character (curated 24:
                        angles, poses, emotions, settings + outfit); concept_plan() =
                        Concept (18: 10-angle turnaround, framing/scale, context — no
                        emotion, no outfit). Same Shot model for both, so the ② table,
                        YAML plans and the pipeline need no special case.
                        plan_for_type()/plan_subject() are the single selection seam
                        shared by UI + CLI (Style → empty plan). Plus apply_wardrobe()
                        outfit injection + apply_prop_exclusion()
  wardrobe.py           Compositional unisex outfit pool for the outfit column
                        (colour x garment; one pool, no gender picker)
  plan_io.py            Save/load shot plans as YAML (user-editable prompt libraries)
  dataset_stats.py      Inspect a dataset folder (count/sizes/aspects) -> suggested
                        steps + bucket ladder, so ⑤'s numbers are derived not guessed
  trainer_configs.py    Emit LoRA-trainer configs (ai-toolkit config.yaml / kohya
                        sd-scripts kohya-dataset.toml / musubi dataset.toml) +
                        run-command builders + model registry. ModelPreset carries
                        per-arch train/sample knobs (noise_scheduler/sample_guidance/
                        sample_steps) and expects_tags (drives the caption/model
                        sanity check). _sample_prompt varies by TrainConfig.dataset_type;
                        caption_mismatch_warning() is the ④→⑤ advisory
  user_config.py        Persist trainer install paths, last training settings, the header
                        dataset type, and the custom captioner endpoint (URL/model/
                        key-env-NAME/spacing) to .cache/user_settings.json (no secrets;
                        gitignored)
  update_check.py       Best-effort GitHub-release check (cached 24h) -> dismissible
                        UI banner when a newer version is published
  comfy_api.py          Thin ComfyUI HTTP client (upload, queue, poll, fetch, free).
                        _combo_options()/_server_filename() validate configured model
                        filenames against the server's own combo list (separator/case
                        -insensitive match, raises a named-setting+closest-match error
                        on a genuine miss) — see the ComfyUI model-lookup gotcha
  engines/
    base.py             Engine protocol + GenerationError
    gemini.py           Cloud engine (Gemini image models via google-genai) with
                        cached, force-refreshable image- AND caption-model lists
                        (list_image_models / list_caption_models)
    comfyui.py          Local engine (Qwen Image Edit 2511 + Multiple-Angles LoRA)
  comfy_workflows/*.json  API-format ComfyUI graphs (restore, isolate ×2, qwen edit)
```

## Testing & CI

`pytest` (config in `pytest.ini`) covers the pure logic — the tagger `select_*`/`format_*` seams,
caption finalization/lint, trainer-config rendering, package/zip resolution, shot planning,
`comfy_api`'s filename-matching/combo-cache seam, the single-device pin, etc. — without
downloading a model or touching the network (heavy backends are lazy-imported and mocked;
`comfy_api`'s tests monkeypatch `httpx.get`). Run the suite with `pytest -q`; lint with
`ruff check .` (rules in `pyproject.toml`).

`.github/workflows/ci.yml` runs both on every push to `main` and every PR, across Python
3.10–3.12. It installs `requirements.txt` only (**no `torch`**): the suite is CPU-only and never
needs the real ML backends, so CI stays fast and deterministic. The workflow declares
`permissions: contents: read` — it only ever reads the repo, so the default write-capable token
is more authority than it needs.

The setup/launch scripts are the one thing `pytest` cannot reach. They are validated by running
them: `setup.bat` against a stubbed copy that exercises every branch (happy GPU/CPU, fresh venv,
old Python, new Python, pip failure, doctor failure) asserting no cmd parse error and that every
path pauses, plus one real fresh-venv install; `bash -n` and failure-path runs for the `.sh`
pair. The fragile logic was moved into `studio/env_keys.py` precisely so it could be tested
properly — see the shell gotchas.

## Design rules

1. **Every stage is standalone.** Stage functions take explicit input paths and an
   explicit output folder; none of them require state from another stage. The UI's
   "auto-fill the next tab's folder" is convenience wiring in `app.py` only.
2. **Local/cloud is a per-stage choice**, selected at call time (engine key, captioner
   key, isolation/restore backend), never a global mode. **Documented exception:** the
   header **Dataset type** (Character/Style/Concept) *is* global — a dataset is one type,
   and per-tab type controls invite mismatch. It only tunes prompts/defaults; stages
   still run standalone (see the dataset-type gotcha).
3. **Heavy imports are lazy.** `torch`/`transformers`/`google-genai` are imported inside
   the functions that need them, so cloud-only or export-only use never loads them.
4. **Captions are `.txt` sidecars** next to the images. That makes stage ③ composable
   with anything (hand-written captions, other tools) and lets export (④) work on any
   folder that has pairs.

## Maintainer principles (standing demands)

These are the project owner's repeated, non-negotiable expectations for every change. They
sit alongside the architectural Design rules above and apply to all future work — treat a
change that violates one as incomplete.

1. **No bloat, no unwarranted complexity.** Add code, dependencies, UI, and abstractions only
   when they earn their place. Prefer reusing an existing seam over inventing a new one; prefer
   a small pure function over a framework. If a feature can't be justified as *genuinely useful*,
   it does not go in — never add things for the sake of adding them.
2. **Efficiency.** Keep startup fast (heavy imports stay lazy), keep the hot paths cheap, and
   don't do work the user didn't ask for. Advisories are numpy/string-only unless a real model
   is unavoidable.
3. **Help/tooltips stay useful and current.** Every non-obvious control carries a concise `info=`
   tooltip; when behaviour changes, update the tooltip in the same change. Guidance must be
   accurate — a stale hint is worse than none.
4. **Sensible defaults, optimized for modern models.** Defaults target current-model dataset
   building (Flux / Krea 2 / Qwen-Image / SDXL): prose captions by default, 1024 long-side, a
   modern flow-matching trainer preset, cloud engine for no-GPU users. New controls default to
   the safe/no-op choice. Re-audit defaults whenever a stage changes.
5. **README stays human-readable and useful without being long.** Features/why-use-it up top,
   marketable but accurate, scannable. Trim before adding; deep detail lives in this file.
6. **Keep every MD file updated in the same change.** `README.md` and `docs/ARCHITECTURE.md`
   (module map, gotchas, roadmap, version line) must reflect the code as it lands, not later.
7. **Log genuinely-useful future ideas here, never build them unprompted.** While working, watch
   for enhancements that fit the project shape and record them under "Future ideas & enhancements"
   as candidate to-dos — *only if legitimately useful*. Noting is free; building without a green
   light is not.
8. **Best practices throughout.** UI/UX and ease of use, security (localhost-only, no secrets on
   disk, opt-in/outbound-flagged network), coding (typed, tested, `ruff`-clean, honest output —
   never fake capability), and GitHub hygiene (clear commits, no PRs unless asked, releases when
   a version bumps). Every version bump updates `studio.__version__` **and** ships a GitHub
   Release, or the in-app update check never fires.

## Data flow (full pipeline)

```
sources ─① preprocess→ runs/<stamp>-prepped/*_prepped.png
        └─ restore (comfyui models | basic lanczos) → isolate (SAM3) → [tighten crop] → resize
prepped ─② generate→ runs/<stamp>-generated/<shot-id>.png   (one per plan row;
                     Character 24-shot plan | Concept 18-shot plan; Style never generates)
any folder ─③ caption→ *.txt sidecars (trigger-first; framing per dataset type; per-model template)
folders(s) ─④ export→ datasets/<name>-dataset/NN.png + NN.txt + metadata.json (+ dataset_type,
                     caption_style) + metadata.jsonl
dataset ──⑤ train  → writes ai-toolkit config.yaml OR kohya/musubi dataset.toml into the
                     dataset folder + prints the run command (optional; nothing launched)
```

Dataset types: **Character** (default; identity, 24-shot set in ②), **Style** (an aesthetic —
caption the *content*, not the look; bring your own images, start at ③), **Concept** (an
object/action/idea — caption the *context*, not the fixed form; 18-shot object set in ②).
The type is chosen once in the header, remembered between launches (`user_config`), and tunes
caption framing, the ② shot plan and its character-only controls, the ① isolation default and
SAM3 subject, the ⑤ sample prompt, and the ④ metadata. **Style never generates** — an aesthetic
can't be synthesized from a reference the way an identity or an object can — so ②'s buttons are
disabled for it in the UI and `cli.py build --dataset-type style` skips the stage outright.

## Gotchas (hard-won)

- **Never put an unescaped `(` or `)` inside a cmd `if … ( … )` block.** cmd ends the block
  at the first bare `)`, so `echo free-tier captioning (SFW, rate-limited)` inside an `if`
  left a stray `.` behind and the script died with `. was unexpected at this time`. This
  shipped in `setup.bat` 0.11.0: it killed setup immediately after the Gemini key prompt,
  so users got a Gemini key written, no Groq/HF prompt, no "Setup complete" — and, because
  nothing paused, an instantly-closing window with no message. `setup.bat` now uses
  `if not X goto :label` instead of parenthesized blocks, and every failure path goes
  through `:die`, which pauses. **A `.bat` must never exit without pausing** — double-clicked,
  `exit /b 1` closes the window before anyone can read the error.
- **Batch/bash are the wrong place for `.env` editing.** The two setup scripts each
  hand-rolled it and each broke differently: `setup.bat` *skipped* a key already present
  (so a typo'd key was unfixable through the installer) and `echo K=v>> .env` glued onto the
  last line of a file with no trailing newline; `setup.sh` appended unconditionally, so every
  re-run duplicated all three keys. Both echoed the key in the clear. It now lives in
  `studio/env_keys.py` behind `cli.py keys`, is unit-tested, prompts via `getpass`, and
  behaves identically on both platforms. The shells only run pip.
- **`.bat` files must be CRLF, `.sh` files LF** — enforced by `.gitattributes`
  (`*.bat text eol=crlf`). An LF-only batch file breaks `goto`/`:label` resolution in cmd,
  which `setup.bat` depends on throughout. A fresh clone gets this right; a working tree
  whose files were written by LF-emitting tools does not, so test from a real clone.
- **`LDS_`-prefixed keys outrank the bare names.** `Settings` uses `env_prefix="LDS_"`, so
  pydantic-settings fills `gemini_api_key` from `LDS_GEMINI_API_KEY` in preference to
  `GEMINI_API_KEY`. Anything that reports or writes a key must handle both spellings:
  reporting only the bare name called a working key "not set", and *writing* only the bare
  name would leave a stale `LDS_` line still winning at runtime, so the user's "changed" key
  would silently have no effect. `env_keys.set_managed_key` lands on whichever spelling the
  file already uses. `HF_TOKEN` needed a `Settings.hf_token` field added for the same
  reason — without it `LDS_HF_TOKEN` was read by nothing at all.
- **Export copies source bytes verbatim, so the extension must be the source's.** It used to
  hard-code `NN.png`, which wrote JPEG bytes into a `.png` and declared `.png` in
  `metadata.jsonl` — a real mislabel, since stage ④ is documented as working on *any*
  folder. `package.py` now preserves the (normalised, lowercased, `.jpeg`→`.jpg`) suffix and
  `metadata.jsonl`'s `file_name` is the name actually written, which HF `imagefolder`
  resolves literally. Captions still pair by stem, so a mixed-format folder is fine.
- **Caption sidecars are user files; read them forgivingly.** Every read used strict
  `encoding="utf-8"`, so a cp1252 `.txt` from another tool raised `UnicodeDecodeError` and
  killed the whole stage, and a UTF-8 BOM silently prefixed `﻿` to the trigger word in
  every caption. `config.read_caption()` (`utf-8-sig`, `errors="replace"`) is now the single
  reader for all of them.
- **`create_repo(exist_ok=True)` does not change visibility.** Publishing "privately" into a
  repo that already exists and is public left it public while the UI said private.
  `hf_publish` now checks and warns plainly rather than implying a privacy guarantee.
- **`facebook/sam3` is gated.** Users must accept Meta's license on the model page and
  authenticate (`HF_TOKEN` in `.env` or `hf auth login`) or `isolate.py` raises a clear
  error with instructions.
- **Rear views hallucinate** when generated straight from a front reference. The shot
  plan chains them (`chain_from`) off a generated side view; chained shots always run
  last, and the chained image is passed *first* so single-reference engines rotate
  stepwise.
- **SAM3 usually does NOT merge props into the subject segment** — and the exclusion
  step must not assume it does. Measured on a reference image (character with a backpack
  and a belt radio): only **3% of prop pixels** fell inside the `character` mask. Because
  `isolate_builtin` composites with `np.where(subject, image, white)`, everything outside
  the subject mask is *already* white, so a prop SAM3 excluded is gone whether or not it
  is subtracted. Subtracting it can therefore only remove genuine subject pixels.
  `isolate_builtin` measures the subject/prop overlap and, below
  `_MERGED_PROP_OVERLAP`, reports the exclusion as a no-op instead of carving. The
  feature still earns its place for genuinely merged props (a microphone gripped in a
  hand), which is the case it was built for.
- **Never dilate the exclusion mask by default.** Growing it cannot remove out-of-subject
  halos (they are already white) — it can only eat the subject. At the old
  `max(2, long_side // 200)` (7px on a 1480px image) it destroyed **2.75% of the
  character**, versus 0.39% for a raw subtract and 0% for no exclusion at all.
  `LDS_EXCLUDE_DILATE_PX` defaults to 0 and exists only for the merged-prop case.
- **Tighten-crop reads the composite, not the mask.** `crop_to_content` crops the isolated
  image to the bounding box of its non-white pixels — both isolation backends composite onto
  white, so this is backend-agnostic and needs no mask handle. It runs only when isolation is
  on and the toggle is set (default off → the Character path is byte-identical), before the
  resize. An effectively all-white image (nothing found) is returned unchanged.
- **Preprocess output names must not clobber.** `list_images` admits several extensions, so
  two sources sharing a stem (`cat.jpg` + `cat.png`, in one folder or across merged input
  folders) both map to `cat_prepped.png`. `preprocess()` makes the output non-clobbering
  (`_prepped`, `_prepped_2`, …) so the second source never silently overwrites the first —
  same "never clobber" rule `package.py`/`zip_dataset` already follow.
- **SAM3 scores a text prompt as ONE concept.** `"backpack, walkie talkie"` asks for a
  single object that is both, and found *less* than `"backpack"` alone (40,942 px vs
  42,712 px); segmenting each term and unioning found 60,084 px (1.47x). `split_terms()`
  splits on commas and `_segment_terms()` unions per-term masks. The ComfyUI graph has one
  text encoder, so it joins terms with `" or "` as the closest equivalent.
- **Occlusion holes cannot be masked away.** Where a prop physically covers the body, the
  subject mask has a real hole and the composite renders it white. Only inpainting fixes
  that; no amount of threshold/dilation tuning will.
- **The Multiple-Angles LoRA is trained on clean splat renders** — isolating the subject
  onto white dramatically improves its output, especially direct back views. The LoRA is
  trigger-based (`<sks>` grammar), so its strength is 0.9 for `kind="angle"` shots and
  zeroed for pose/emotion shots.
- **VRAM choreography:** before loading a ~17 GB local captioner, the pipeline asks
  ComfyUI to `/free` its models (best effort) and unloads the in-process SAM3.
- **ComfyUI queue guard:** if ComfyUI already has >10 pending jobs, the client fails
  fast instead of silently queueing behind them — unless `front=True`, which asks
  ComfyUI to put the job at the head of the pending queue (`/prompt` accepts `front`
  and negates the job's priority number). `front` does **not** interrupt the job already
  running, so the UI must not promise "immediate". Polling is always by our own
  `prompt_id`, so another client's jobs are never mistaken for ours.
- **The bundled workflows use ONLY core nodes.** `isolate_*.json` previously needed
  `MaskPreview+` from the third-party `ComfyUI_essentials` pack — used purely as a hack to
  render a white backdrop — which silently broke the workflows for anyone who didn't
  happen to have that pack. Replaced with core `EmptyImage` + `GetImageSize`.
  `EmptyImage` **must** be sized from the source via `GetImageSize`: hardcoding 1024²
  breaks `ImageCompositeMasked` (`resize_source: false` requires matching dimensions).
  `SAM3_Detect` *is* core (`comfy_extras/nodes_sam3.py`). Keep it this way — a
  third-party node in a bundled workflow is a silent portability failure.
- **Gemini refusals return no image part** — that's detected and reported as a refusal;
  retries don't help, so only transient errors are retried.
- **The Concept plan reuses the character machinery, not a parallel one.** `concept_plan()` emits
  the same `Shot` model with `emotion`/`outfit` left empty, so the ② dataframe, `plan_io` YAML
  round-trip, `apply_wardrobe` (a no-op on an empty outfit) and `generate_shots` need no branch.
  Its angle shots reuse the **attested** `<sks>` grammar from the character plan (view + camera
  height + shot size) — the one addition, `front view high-angle`, mirrors the existing low-angle
  shot. Do not invent grammar the Multiple-Angles LoRA was never trained on (a "top view" would
  silently render a plain front view); a test pins the vocabulary. The non-angle kinds are
  `framing` (scale: detail → wide) and `context` (where it sits / how it's used, including a hand
  for scale), which keep LoRA strength at 0 like character pose shots.
- **Character-only ② controls are hidden, not just ignored, for Concept.** Wardrobe
  (`OUTFIT_SHOT_KINDS` includes `angle`, so the randomizer *would* dress an object's turnaround)
  and prop exclusion (its clause literally says "show only the character and the clothing worn on
  their body") are character ideas. The UI hides the outfit buttons + blurb and defaults
  prop-exclusion off; the CLI's `_dress()` is a no-op with a note for non-character types and
  `--exclude-props/--keep-props` defaults **per type** (a tri-state `None` option, since a typer
  default can't depend on another flag). `--isolate/--no-isolate` is tri-state for the same reason.
- **Local prompts interpolate the subject — nothing downstream formats them.** `_build_local_prompt`
  used to emit a literal `{subject}` for pose/emotion shots; no caller ever called `.format()` on
  it, so ComfyUI received the braces verbatim in 15 of the 24 character shots. It now takes the
  subject, and `_subject_phrase()` strips a leading article so "the same the object" can't happen
  (the cloud builder had that wart too). Settings are complete phrases ("in a warmly lit interior
  room", "outdoors at golden hour"), so neither builder prefixes its own "in" — that produced
  "in in a warmly lit interior room" / "in against a plain … background".
- **Shot plan is a curated list, not a cartesian product.** Each of the 24 default
  shots combines a unique angle/pose/emotion/setting. Scene/lighting is folded into
  other shot kinds so the dataset is not skewed by many images of the same standing
  pose with different backgrounds.
- **Gemini model lists are cached locally.** `studio/engines/gemini.py` persists the
  live image-model list to `.cache/gemini_image_models.json` and the caption-model list
  to `.cache/gemini_caption_models.json`, both with a 24-hour TTL. The UI seeds each
  dropdown from cache (never a network call on startup); a refresh button force-pulls.
  Both fall back to a static list if the API is unreachable or no key is set.
- **Update check is best-effort by design.** `update_check.py` hits the GitHub releases
  API on `demo.load` (not at process start, so it can't slow down launch), caches the
  result 24h in `.cache/update_check.json`, and falls back to a stale cache on a failed
  refetch. Any exception is caught and simply shows no banner; it must never raise into
  the UI. `LDS_UPDATE_CHECK_ENABLED=false` skips the network call entirely.
- **The update check reads GitHub Releases, not commits or tags.** It compares
  `studio.__version__` against `GET /repos/.../releases/latest`, which 404s (silently, no
  banner) until a Release object exists — a `git tag` alone is not enough. So **every
  version bump needs both**: (1) update `__version__` in `studio/__init__.py` to match
  the `Version:` line at the top of this file, and (2) publish a Release
  (`gh release create vX.Y.Z --generate-notes`, or the GitHub API) targeting the commit
  that lands the bump. Skipping step 2 means users on the old version never see a notice.
- **User strings in the ai-toolkit config go through PyYAML, never a raw quoted template.**
  The generated `config.yaml` embeds a user-supplied `name`, `model.name_or_path` (which may be
  a Windows checkpoint path like `C:\models\x.safetensors`), and the sample prompt (which carries
  the trigger/name). Splicing those into a hand-written double-quoted YAML scalar breaks the file
  — a backslash is an escape char there and a stray `"` closes the string early. `_yaml_str()`
  emits each through `yaml.safe_dump` (whitespace collapsed to one line first) so quoting is
  always correct. The `.toml` renderers are unaffected: they emit only `.as_posix()` paths and
  numbers, no free text.
- **`python-dotenv` is a declared direct dependency.** `studio/config.py` imports
  `dotenv.load_dotenv` and `Settings` sets `env_file=.env`; both need `python-dotenv` at runtime.
  It is **not** a hard dependency of `pydantic-settings>=2.2`, so it is listed explicitly in
  `requirements.txt` — a fresh install without it fails at import of `studio.config`, which every
  entry point pulls in.
- **Gemini caption default is a rolling alias.** The Gemini captioner defaults to
  `gemini-flash-latest` so it doesn't 404 when a pinned version is decommissioned. The
  Caption tab can refresh and pick a specific model; `Captioner(model_override=...)`
  applies it only when the backend is Gemini.
- **Groq Qwen3.6 is a reasoning model.** Its spec sets `extra_params={"reasoning_effort":
  "none"}` to switch off the `<think>` scratchpad (otherwise the token budget is spent
  on reasoning and the caption comes back empty/truncated), plus a higher `max_tokens`
  as insurance and `_clean()` strips any residual `<think>` tags. 429s are retried with
  backoff; `min_interval_s` spaces requests for the free tier.
- **Custom OpenAI-compatible captioner.** The `custom` captioner carries no endpoint in
  config; either the Caption tab or `cli.py custom-endpoint` (0.12.2 — previously UI-only,
  a CLI-only user had no way to set this at all) collects base URL / model / key-env-NAME /
  request spacing and persists them (minus the secret) via `user_config.set_custom_captioner`.
  At caption time `resolve_captioner_config` (shared by UI and CLI) loads them into
  `spec_overrides`, applied with `CaptionerSpec.model_copy(update=...)` so the registry spec
  is never mutated. The API key is read from the named env var. `cli.py custom-endpoint`
  with no options shows the saved config; passing only some fields keeps the others at
  their last-saved value rather than requiring the whole config restated every time.
- **Caption style is prose / Danbooru tags / e621 tags, per call — not a global mode.**
  `CaptionerSpec` carries three templates (`prompt_template` prose, `tags_template`
  Danbooru, `e621_template` furry/anthro) and `prompt_for(style)` selects one; every backend
  just sends the chosen instruction. Tags exist because Danbooru-trained (SDXL / Illustrious /
  NoobAI) and e621-trained (Pony / furry) checkpoints learn poorly from prose — and Danbooru
  vs e621 are **different controlled vocabularies**, hence two separate tag templates. Both tag
  styles share `finalize_caption(style in {"tags","e621"})`, which lowercases, dedupes,
  comma-joins, keeps the trigger first, and does **not** inject the character name (identity is
  the trigger). An unknown style value falls back to prose rather than raising. Caveat: the
  backend *instructs* a general VLM to emit tags, so output approximates the vocabulary; a
  dedicated tagger is needed for a canonical set.
- **Dataset type composes prompts; it never stores per-type templates.** Character returns the
  tuned per-model templates verbatim (byte-identical — the default must not regress). Style and
  Concept instead COMPOSE `prompt_for(style, dataset_type, sparse)` from one *framing* clause
  per type (Style → describe the content, not the look; Concept → describe the context, not the
  fixed form; sparse Style → trigger + a few words) plus the same prose/tags/e621 *format*
  directive. That is one framing × one directive, not nine new templates per captioner — the
  anti-bloat choice. JoyCaption keeps its `refer to them as "{subject}"` convention via the
  existing `prompt_style == "llava"` flag; an NSFW spec (detected by the shared `_NSFW_PLAINNESS`
  clause in its template, not by a new field) keeps a plainness directive. `finalize_caption(...,
  dataset_type=)` makes Style/Concept prose trigger-first with **no** alias→name replacement (no
  "the woman"→name mapping for a look or object); tag/e621 finalization is already identity-free
  so it is unchanged. Taggers (`wd_tagger`) ignore dataset type entirely — they read tags from
  image features, not from a framed instruction. This is the one place the global-mode exception
  to Design-rule #2 lives.
- **One handler owns every type-dependent control, and it runs on load too.**
  `on_dataset_type_change` returns updates for all 17 type-sensitive controls across ①–④ (isolation
  default + SAM3 subject, ② guidance/labels/plan/buttons/wardrobe/props, ③ name+trigger tooltips and
  the Style-only sparse toggle, ④ name label) and persists the choice. It is wired to both
  `dataset_type.change` **and** `demo.load`, because the header seeds its value from
  `user_config.get_dataset_type()` — without the load pass a remembered Style/Concept type would
  arrive with Character defaults still applied. A test asserts the handler's return arity equals the
  wired outputs list; Gradio would otherwise mismatch them silently. Style's ② table is
  **empty**, not a character plan — 24 shots you can't generate is a lie the guidance note
  would have to talk you out of.
- **⑤ reconciles the dataset's recorded type with the header.** The header is a remembered global,
  but ⑤ can be pointed at any folder, and the type drives the sample prompt. `dataset_type_note()`
  reads `metadata.json` on Inspect and reports the recorded `dataset_type`/`caption_style`, warning
  when it differs from the header. Advisory only, silent for datasets without metadata, and never
  raises on a corrupt file.
- **The tagger is a captioner backend, not a style.** `backend="wd_tagger"` runs an ONNX
  tagger that emits canonical tags directly from image features — so it *ignores* the
  prose/tags/e621 selector and the dataset-type framing, and both `caption_images` and the UI's
  test path force `style="tags"` for it. Its pure selection/formatting logic lives in `tagger.py`
  free of model/file I/O and is unit-tested; `onnxruntime` and the huggingface download are lazy.
  Preprocessing is exact and non-obvious (white-pad to square, resize, **RGB→BGR**, float32
  **0–255, no normalization**, NHWC) — do not "tidy" it. Outputs are already sigmoid probabilities.
- **Danbooru and e621 taggers share one code path; only the tag file + categories differ.**
  WD (`selected_tags.csv`) and Z3D-E621 (`tags-selected.csv`) both expose `name`/`category`
  columns. The *meaning* of the category ids differs, captured in `SCHEMES`: Danbooru
  {general:0, character:4}; e621 {general:0+5(species), character:1(artist)/3(copyright)/4/
  8(lore)} with meta/invalid dropped. e621 **species** rides the general threshold; artist/
  character/copyright ride the higher character threshold. The e621 model
  (`toynya/Z3D-E621-Convnext`) is a community weight — its output should be spot-checked.
- **Tagger thresholds flow through `spec_overrides`, not a new arg.** The ③ sliders set
  `general_threshold`/`character_threshold` via `CaptionerSpec.model_copy(update=…)`. The rating
  and keep-underscores toggles ride the same seam (`include_rating`, `keep_underscores`).
  `merge_tagger_overrides()` is the single place UI *and* CLI fold those four controls in, and it
  is a no-op for any non-tagger captioner.
- **Rating tags are opt-in and Danbooru-only.** WD taggers expose a rating category (cat 9);
  `include_rating` appends only the single highest-confidence one. The e621/Z3D scheme carries
  **no** rating category, so `SCHEMES["e621"]["rating"]` is empty and the toggle is a documented
  no-op for Z3D.
- **The tag drop-list runs before affixes, never on prose, never on the trigger.**
  `drop_blacklisted_tags` filters a comma tag list against a normalized blacklist (lowercase,
  underscores→spaces). It is applied after `finalize_caption` but before `apply_affixes`, so a
  fixed prefix/suffix survives the drop; it is a strict no-op for prose or an empty list; and the
  first token (the identity trigger) is always kept. `parse_blacklist` is the shared normalizer.
- **Caption prefix/suffix is a post-finalize wrap, applied once.** `apply_affixes` runs after
  `finalize_caption` and joins with a comma for tag styles, a space for prose. It exists for
  constant tags the model shouldn't be asked to invent — Pony `score_9, …`, Danbooru
  `masterpiece, best quality`. Empty prefix/suffix is a strict no-op.
- **Advisories are honest heuristics, never a rewrite or a block.** `quality.composition_flags`
  labels exposure/contrast outliers (dark/bright/low-contrast) from luminance mean/std — it does
  **not** claim semantic framing (face/bust/body/back), which would need a detector. `caption_
  lint.py` flags empty / too-short / trigger-missing / byte-identical captions and, for tag
  datasets only, near-ubiquitous tags. `dedupe.find_near_duplicate_groups` groups images within a
  Hamming distance (default 5, a ④ slider). All never edit or block; all call sites wrap them in
  try/except so an unreadable folder degrades to no report. Two lint choices keep it trustworthy:
  `min_words` defaults to 2 (only near-junk is "short"), and the tag reports are gated on
  `looks_like_tags()` (avg ≤3 words/segment) so prose never produces a meaningless frequency list.
  Note a dHash quirk: flat/solid images all hash alike, so near-blank frames may group — acceptable
  for an advisory.
- **The CLIP 77-token warning is a tokenizer-free estimate, tag-datasets only.**
  `estimate_clip_tokens` approximates CLIP BPE as ~1.3 tokens/word + one per comma + 2 for
  BOS/EOS and rounds up, erring high so it fires a little early rather than missing a truncation.
  `analyze_pairs` only applies it (limit 77) when the captions look like tags — SDXL/Illustrious/
  Pony are the CLIP encoders that truncate; prose targets T5/Flux (512 tokens) and is never
  nagged. Exactness would need the real CLIP tokenizer (a possible future dependency, not worth
  it for an advisory).
- **The ④→⑤ caption/model sanity check is advisory, driven by `ModelPreset.expects_tags`.**
  SDXL-family presets (SDXL base + the Pony/Illustrious/NoobAI `<<FILL>>`, ai-toolkit and kohya)
  set `expects_tags=True`; the flow-matching prose presets leave it False.
  `caption_lint.folder_caption_kind()` classifies the exported dataset's captions (tags|prose) and
  `trainer_configs.caption_mismatch_warning()` warns at ⑤ when they don't fit the chosen base
  model (prose→tag-model, or tags→prose-model). It never blocks; it points the user back to ③.
  Detection is done on the *actual* captions, so it holds even if `metadata.json` is absent.
- **`metadata.jsonl` is written for every export.** `package_dataset` emits the HuggingFace
  `imagefolder` sidecar (`{"file_name","text"}` per image) alongside `metadata.json`, so the
  dataset also loads via `datasets.load_dataset("imagefolder", …)`. `metadata.json` also records
  `dataset_type` and a detected `caption_style` (unless the caller passed one). `.txt`-based
  trainers ignore the jsonl; it carries captions only, no secrets.
- **kohya sd-scripts uses a different dataset layout from musubi.** Even though musubi is a
  kohya fork, sd-scripts wants `[[datasets.subsets]]` with `image_dir` (not musubi's
  `[[datasets]]` + `image_directory`), so `render_kohya_toml` is its own renderer. SDXL base
  is a runnable HF id; a family checkpoint (Pony/Illustrious/NoobAI) is a user-local `<<FILL>>`.
  `kohya_command` threads every hyperparameter from the `TrainConfig` and uses
  `sdxl_train_network.py`. Verified by unit tests on the rendered text, never a live run.
- **SDXL is not flow-matching.** The ai-toolkit renderer used to hardcode `noise_scheduler:
  flowmatch` and `guidance_scale: 4`, correct for Flux/Qwen/Z-Image but wrong for SDXL (wants
  `ddpm` and CFG ~7). Those three knobs live on `ModelPreset` (`noise_scheduler`/
  `sample_guidance`/`sample_steps`) with the old values as defaults, so every existing preset
  renders byte-identically and only SDXL overrides them. `guidance` is formatted `:g` so `4.0`→`4`.
- **HF publishing is opt-in and private by default.** `hf_publish.publish_dataset` never runs on
  its own — it needs an explicit UI button click or the CLI `--publish-hf` flag.
  `create_repo(private=True)` unless the user deliberately unchecks it; the write token comes only
  from `HF_TOKEN` (env/.env) and is never written to disk. `normalize_repo_id` validates the id
  (I/O-free, unit-tested) and the missing-token / missing-folder guards fail fast *before*
  `huggingface_hub` is imported. Only files inside the dataset folder are uploaded.
- **`zip_dataset` arcnames are derived from the folder, never absolute.** Entries are stored
  under `<dataset-name>/…` and the archive is non-clobbering (`-2.zip`, …) — no path-traversal
  surface because we only ever add files found *inside* the packaged dataset dir.
- **ComfyUI caches model combo lists**; a freshly downloaded model file may need a ComfyUI
  restart before the bundled workflows validate.
- **Model filenames are configuration.** The bundled workflow JSONs are patched at load time from
  settings (`LDS_QWEN_EDIT_MODEL` etc.), so users don't edit JSON to match their filenames.
- **Trainer configs are derived from the dataset, not from constants.** `dataset_stats.inspect()`
  reads image count/dimensions (Pillow header parse only) so steps scale with the set
  (`STEPS_PER_IMAGE`, clamped to 1000–4000) and buckets come from the images that actually exist.
  Only config keys attested in each trainer's own examples are emitted — inventing plausible keys
  produces configs that fail hours into a run.
- **`musubi_command()` takes the whole `TrainConfig`, not just the `ModelPreset`.** It previously
  took the preset alone and hardcoded `--network_dim 16` / `--max_train_steps 2000`, so every
  ⑤-tab slider was silently discarded. Regression-tested in `tests/test_trainer_configs_0_4_0.py`.
- **Emitted configs are verified by unit tests on the rendered text, never by a live training
  run.** The ⑤ tab says so. Don't imply otherwise.
- **ai-toolkit configs are one-command; musubi configs are not.** ai-toolkit emits a fully
  runnable `config.yaml` (model is a HF id → `python run.py config.yaml`). musubi's `dataset.toml`
  is complete, but its *training* invocation needs the user's local DiT/VAE/text-encoder paths, so
  the musubi run command is a `<<FILL: ...>>` template. The UI says so; it is not faked as one-click.
- **Trainer model registry is curated, not exhaustive.** Where a model's canonical HF id or musubi
  script isn't something we can guarantee, the preset carries a `<<FILL>>` placeholder so the
  emitted config is honest rather than silently wrong.
- **Sharpness is advisory.** `quality.py` labels blurry shots (`⚠ blurry (score)`) in the curate
  gallery and lists them at export; it never deletes or blocks. Threshold is tunable via
  `LDS_SHARPNESS_BLUR_THRESHOLD`.
- **The outfit column is applied at generation time.** `apply_wardrobe()` folds a shot's outfit
  into both prompts just before the engine runs (idempotent), so the column stays functional even
  if prompt cells were hand-edited. Default shots leave outfit empty to avoid identity drift.
- **`user_settings.json` holds no secrets.** Trainer install paths, last hyperparameters, and the
  custom endpoint (URL/model/key-env-NAME/spacing) live in `.cache/user_settings.json`
  (gitignored). API keys never go there — only the *name* of the env var that holds the key.
- **Gradio only serves whitelisted paths.** The galleries display images the app writes to
  user-chosen output folders, which can be on any drive. Gradio refuses paths outside the CWD/temp
  by default, so `demo.launch(allowed_paths=...)` is fed the configured run/dataset roots plus
  every present drive root. This is acceptable only because the server binds to localhost with no
  auth (see Security posture).
- **User-entered output paths are validated.** `_validate_out_dir` in `app.py` rejects paths with
  characters the OS forbids (`< > : " | ? *`, control chars) with a friendly message, and the
  generate/export handlers wrap writes in `try/except OSError` — a bad path surfaces as a GUI hint,
  not a raw traceback.
- **Export sample caption is deterministic and empty-aware.** `do_export` sorts the numbered `.txt`
  files (excluding `README.txt`), shows the first non-empty one, and explicitly flags when captions
  are empty — so a blank-caption bug is visible instead of looking like a UI glitch.
- **Export is selection-gated, path-keyed.** ④ requires "Load & preview" before export; the
  `CheckboxGroup` value is the full image path (not a `gr.State`), so identically-named files in
  different folders never collide and the stage stays stateless. `resolve_export_items()` is the
  single scan/classify seam shared by the UI and `cli.py export`; a checked image with a missing/
  empty caption is skipped and reported, never silently dropped or fatal.
- **`device_map="auto"` shards a model across every visible GPU — never pin SAM3 or a
  captioner to it on a multi-GPU box.** A 2-GPU machine crashed preprocess with "Expected all
  tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1" because
  accelerate split SAM3 across both cards and the forward pass mixed them. SAM3 fits any single
  card, so `isolate.py` now loads it via `from_pretrained(...).to(config.torch_device())` instead
  of `device_map="auto"`. Local captioners *can* legitimately need more than one card, so
  `captioner.py` keeps `"auto"` when `torch.cuda.device_count() <= 1` (offloads whatever doesn't
  fit) and pins to a single device only when there are two or more — trading automatic sharding
  for crash-safety on multi-GPU boxes. `LDS_TORCH_DEVICE` (`config.torch_device()`) overrides
  either path (`cuda:1`, `cpu`, …) for a user who wants to pick the card themselves.
- **A failed ComfyUI combo-list lookup must never be cached.** `comfy_api._combo_options` used to
  cache a timed-out `/object_info` call as an empty list — `/object_info` stalls while ComfyUI is
  busy — which permanently disabled filename matching for the rest of the process; every later
  graph then sent the raw (unmatched) value and ComfyUI rejected it with an opaque "Value not in
  list: ... not in (list of length 10574)". Only successful lookups are cached now (empty results
  re-fetch on the next call), the timeout is 30s, matching falls back to a case-insensitive
  compare (Windows paths vs. ComfyUI's exact-case list), and — new behavior — **an unmatched
  filename now raises** `ComfyError` naming the `LDS_` setting to fix and the closest installed
  filenames (via `difflib`), instead of silently sending a value ComfyUI will reject anyway.
  Applies to LoRA/checkpoint/UNet inputs and `UpscaleModelLoader` alike.
- **Gradio is capped `<6`.** On Gradio 6.20.0, components fed by `demo.load` in ②/③ (shot plan,
  cost line, character-name box, isolation subject, wardrobe note) get stuck under a loading
  overlay that never clears — the value arrives but the spinner keeps counting up. Reproduced by
  A/B on the same code/`.env` (6 stuck components on 6.20.0, 0 on 5.50.0). Root cause inside
  Gradio 6 is still unknown — logged under Future ideas so an eventual upgrade attempt starts from
  here instead of re-discovering it. **Caveat for anyone changing the pin by hand:** installing
  Gradio 5 can pull `pydantic` down to a version older than `google-genai`'s floor; reinstall
  `pydantic>=2.12.5` afterward if `import google.genai` breaks.
- **Alpha cutout is builtin-backend-only and doesn't feed ② Generate.** `isolate_builtin`'s
  `alpha_cutout` flag composites the subject onto transparency (straight alpha = the mask)
  instead of white; `isolate_subject` raises `IsolationError` if asked for alpha cutout on
  the `comfyui` backend rather than silently returning white or ignoring the flag — the
  bundled ComfyUI isolation workflows composite via core nodes onto a solid `EmptyImage`
  and haven't been extended for transparency. `crop_to_content` dispatches on `image.mode`
  (alpha>0 bounding box for RGBA, non-white heuristic otherwise) so tighten-crop still works
  on a cutout. This is a terminal output for the user's own compositing workflow — ②
  Generate is unchanged and still expects (and gets, by default) a white-background
  reference; turning alpha cutout on and continuing to ② is not supported.

## Feature history (consolidated)

Rather than a per-release changelog (the git history has that), here is what the tool does today,
grouped by stage. Milestone versions are noted only where they explain a design choice.

- **① Preprocess** — restore (ComfyUI models / basic Lanczos / auto), SAM3 subject isolation
  (built-in transformers or ComfyUI, with measured over-cut fixes), optional tighten-to-subject
  crop, resize to target long-side. Optional alpha (RGBA) cutout output (builtin backend
  only) for external compositing workflows.
- **② Generate & curate** — curated 24-shot Character plan (angles/poses/emotions/settings) or
  18-shot Concept plan (turnaround/framing/context), Qwen-Image-Edit 2511 + Multiple-Angles LoRA
  (local) or Gemini (cloud), chained rear views, prop exclusion, wardrobe randomizer, per-shot
  outfit column, save/load YAML plans, sharpness + exposure/contrast advisories, per-model cost
  estimate. Character + Concept; Style never generates (buttons disabled, with guidance).
- **③ Caption** — prose / Danbooru-tag / e621-tag styles per call; local VLMs (Qwen3-VL, JoyCaption,
  NSFW), dedicated WD + Z3D ONNX taggers, Gemini/Groq/any-OpenAI-endpoint; dataset-type framing
  (character/style/concept + Style sparse mode); tagger thresholds, rating tag, keep-underscores;
  fixed prefix/suffix, tag drop-list, skip-already-captioned; inline caption editor; caption-health
  + tag-frequency + CLIP-77-token advisories.
- **④ Export** — flat NN.png/NN.txt + metadata.json (records dataset_type + detected caption_style)
  + metadata.jsonl (HF imagefolder) + README.txt; per-image selection gate with near-duplicate and
  caption-health advisories; optional .zip; opt-in private-by-default Hugging Face publish.
- **⑤ Train config** — ai-toolkit (one-command, incl. correct SDXL knobs), kohya sd-scripts SDXL,
  musubi-tuner; steps + multi-resolution buckets derived from the dataset; caption/model sanity
  check; type-aware sample prompt. Nothing is ever launched — configs + the run command are shown.

The **CLI** mirrors every stage (`preprocess`/`generate`/`caption`/`lint`/`export`/`build`) with
the same options and the same shared seams (`resolve_captioner_config`, `merge_tagger_overrides`,
`resolve_export_items`), so UI and CLI never drift. It also carries the two setup-facing commands
the installer itself uses: **`doctor`** (install self-check) and **`keys`** (view/set the `.env`
API keys).

- **Onboarding (0.12.0)** — the install path was rebuilt after a new user's `setup.bat` died at the
  Gemini key prompt and closed its own window. Setup now gates the Python version, pauses on every
  exit, and delegates key entry to `cli.py keys --setup`; `start.bat`/`start.sh` explain a non-zero
  exit (usually a dependency added by a `git pull`) instead of vanishing; `cli.py doctor` reports
  what is configured and what each missing key blocks. See the shell gotchas for why none of this
  lives in the scripts any more.
- **Multi-GPU + ComfyUI model-lookup + Gradio pin (0.12.1)** — community PR (marduk191) from a
  real 2-GPU Windows box: SAM3/captioners no longer crash under `device_map="auto"`
  (`LDS_TORCH_DEVICE` to override), a timed-out ComfyUI `/object_info` call can no longer poison
  filename matching for the rest of the run (and an unmatched filename now raises a named,
  closest-match error instead of ComfyUI's opaque one), and `gradio` is capped `<6` after a
  reproduced stuck-loading-overlay regression on 6.20.0. See the three matching gotchas.
- **Repo rename + usability follow-ups (0.12.2)** — `lora-dataset-studio` renamed to
  `lora-distillery` (see the rename record below) with a matching product-name rebrand
  everywhere it appeared. Plus two small parity/usability fixes surfaced while reviewing
  0.12.1: `cli.py doctor` now proactively validates every configured ComfyUI model filename
  against the live server (reuses `load_template()`'s own check, so a bad filename shows up
  at `doctor` time instead of mid-generate); and `cli.py custom-endpoint` lets a CLI-only
  user configure the custom OpenAI-compatible captioner, closing the UI-only gap noted since
  0.7.0.
- **Second rename + alpha cutout (0.13.0), final-review fixes (0.13.1)** — `lora-distillery`
  renamed to `dataset-deviser` (see the rename record below); the "LoRA" was also dropped
  from the product name since the tool covers Style/Concept, not just character LoRAs.
  New opt-in alpha (RGBA) cutout toggle at ① Preprocess, builtin-SAM3-backend only (see the
  matching Gotcha). A same-day final-review pass (0.13.1) fixed four issues the per-commit
  review missed: ① no longer auto-fills ②/③'s folder fields when alpha cutout is on (that
  output's un-masked pixels are the original background, not white — safe only as a
  terminal artifact, not as a ②/③ input); dropped `Image.fromarray`'s deprecated `mode` arg
  before Pillow 13 removes it; `.env.example` still said "LoRA Distillery"; and a deleted
  measured-rationale comment (the 2.75% figure backing `_MERGED_PROP_OVERLAP`) was restored.
- **Usability fixes (0.13.2)** — two small issues surfaced by a manual usability pass, fixed
  on request: the ⑤ Train tab's Model dropdown tooltip contained the literal string
  `<<FILL>>`, which Gradio's Markdown/HTML rendering of `info=` text silently truncated at
  the first `<` — reworded to avoid the placeholder syntax entirely in user-facing copy
  (the dropdown's own preset labels already say "(set path)"/"(edit … below)" for presets
  that need one). And `shotplan._build_cloud_prompt` emitted "with a alert expression" /
  "with a intense expression" for vowel-starting emotions — new `_indefinite_article()`
  helper picks "a"/"an" from the emotion's first letter (a plain first-letter check is
  enough for this small curated emotion vocabulary; no phonetic library needed).

## Future ideas & enhancements

Candidate to-dos that fit the project shape (standalone stages, per-stage local/cloud choice, lazy
heavy imports, honest output). Recorded when noticed; **not** built without a green light. Roughly
ordered by benefit-to-cost.

- **Concept plan tuning from real runs (②).** The 18-shot concept plan is verified by unit tests on
  the rendered prompts, not by a generation run — the shot mix (10 angles / 4 framing / 4 context)
  and the `front view high-angle` addition are reasoned, not measured. If object turnarounds come
  back weak, retune the mix or drop the shots the LoRA can't do, and record the finding here.
- **Face-similarity identity guard (② curate).** Flag generated shots that drift from the
  reference's identity. Genuinely useful for character LoRAs but needs a face-recognition
  dependency (InsightFace + onnxruntime); could be an optional extra like the gated SAM3 download.
  Revisit if identity drift bites users.
- **Resolution normalization at export (④).** Optional pad-to-square / center-crop to a target size
  for trainers that want uniform square inputs. The bucket ladder covers most cases and a real
  transform mutates pixels, so it wants its own design pass rather than being rushed in.
- **Exact CLIP token count (③/④).** The 77-token warning is a tokenizer-free estimate; loading the
  real CLIP tokenizer would make it exact. Only worth it if a user needs high accuracy — the
  estimate errs safe and is fine for an advisory.
- **ComfyUI-backend alpha cutout.** The `alpha_cutout` toggle at ① (0.13.0) is builtin-SAM3-only —
  the bundled ComfyUI isolation workflows would need a graph change (a core alpha-join node) to
  support it. Revisit if a ComfyUI-only user asks for it.
- **Root-cause the Gradio 6 stuck-loading-overlay bug (0.12.1).** `requirements.txt` caps
  `gradio<6` because 6.20.0 leaves `demo.load`-fed components stuck under a spinner that never
  clears (see the Gradio gotcha). Nobody has dug into *why* Gradio 6 does this — worth a real
  investigation before ever attempting to lift the cap, so the fix doesn't repeat.

## Repo rename: lora-dataset-studio → lora-distillery (0.12.2)

Recorded 2026-07-27 as an open decision, resolved 2026-07-28.

`lora-dataset-studio` was a crowded name: `perfectgf/lora-dataset-studio` was created 2026-07-05 —
six days before this repo — with ~99 stars, and four other repos shared the name. That project is
a whole-lifecycle workbench (in-app/cloud training, an Image Bank, scraping, a Test Studio with
checkpoint ranking), which is **precisely** the scope the Deferred section below rejects on
purpose; the differentiation was real, but the name collision made it invisible in search.

Renaming was cheap at the time: 0 forks, 0 issues, 3 stars, nothing published to PyPI
(`pyproject.toml` has no `[project]` table). The project (and its product display name, "LoRA
Dataset Studio" → "**LoRA Distillery**") were renamed together across the repo, README, in-app UI
title/header, CLI/doctor output, and generated-file provenance strings (dataset `README.txt`,
trainer-config comments, HF commit messages) so nothing keeps saying the old name.

**Existing clones keep working.** GitHub permanently redirects the old URL for the web UI,
`git clone`, `git fetch` and `git push`, so anyone who already cloned can `git pull` as normal —
verified after the rename. Three things this depended on getting right:

- **Never create a new repo under the old name** in the same account — that is the one action
  that would have broken the redirect. (Nothing has been created there.)
- `update_check.py`'s `GITHUB_REPO` constant was updated to `EnragedAntelope/lora-distillery` in
  the same commit as the rename — `httpx` does not follow redirects by default, so anyone still on
  an older version won't see an update banner until they pull the commit with this change. That is
  expected, one-time friction, not a bug.
- **Kept the `LDS_` env prefix** unchanged. Renaming it would have silently invalidated every
  existing `.env` for no user benefit.

## Repo rename: lora-distillery → dataset-deviser (0.13.0)

Recorded 2026-07-29. A colleague pointed out that "distill" has a specific, different
meaning in AI (knowledge/model distillation — training a smaller model to mimic a larger
one), which is not what this tool does: it builds a LoRA training dataset from one
reference image, it does not distill a model. The 0.12.2 name was well-intentioned (it
resolved a real collision) but wrong on the merits.

Landed on **Dataset Deviser** (repo slug `dataset-deviser`) after a fresh brainstorm,
verified clean via GitHub repo search against several alliterative candidates
(`dataset-drafter`, `reference-refinery`, `angle-architect` were also clean; Deviser was
the maintainer's pick — originally suggested by the colleague who flagged the "distill"
problem). This rename also **drops "lora" from the name**: the tool already covers Style
and Concept datasets, not just character LoRAs, so a LoRA-specific name was already
slightly too narrow.

Same cheap-to-rename profile as before (minimal stars, no forks, nothing on PyPI). This is
now a **two-hop rename** (`lora-dataset-studio` → `lora-distillery` → `dataset-deviser`);
the redirect chain was verified to still resolve end-to-end after this hop (`git ls-remote`
against the original `lora-dataset-studio` URL succeeded). The same three things that
mattered for the first rename still apply:

- **Never create a new repo under any of the three names** (`lora-dataset-studio`,
  `lora-distillery`, `dataset-deviser` once it's the current one — trivially true — but
  also never re-create either of the two old names) in this account.
- `update_check.py`'s `GITHUB_REPO` constant was updated in the same commit as the actual
  `gh repo rename` call.
- **Kept the `LDS_` env prefix unchanged** — same reasoning as last time: renaming it would
  silently invalidate every existing `.env` for no user benefit.

## Deferred (with rationale)

Considered and deliberately **not** pursued, with the reason each stays out.

- **Synthetic generation for Style datasets (②).** Permanently out, not deferred. Generation copies a
  *subject* from a reference; a style is the property you'd have to already have in order to copy it,
  so the honest workflow is to collect images that share the look and start at ③. ② is disabled for
  Style rather than quietly producing character shots.

- **In-app / cloud training launch, Test Studio / checkpoint ranking, Merge Lab.** These cross the
  "never launch training" line this tool holds on purpose — launching is fragile across trainer
  venvs/CUDA setups. Configs + saved paths + the shown run command cover the workflow. The command
  is displayed, never executed.
- **Web scraping of training images.** Carries rights/ToS liability we don't want to put one click
  away. (For sourcing, the README points to the author's separate video-frame extractor tool.)
- **Inpainting occluded regions.** Where a prop physically covers the body, isolation leaves a real
  hole; the honest fix needs a generative edit pass per source. The app has the machinery
  (Qwen-Image-Edit) — revisit only if the white notch bites users.
- **Live cloud pricing.** Google publishes no pricing API; only scraping the pricing page could beat
  the build-time table, and it would break silently. Estimates are labelled as such.
- **Automated aesthetic scoring beyond sharpness.** Needs a heavy scoring model; the cheap Laplacian
  check covers the common "is it blurry" case without a new dependency.
- **Regularization-image generation.** Niche for single-character LoRAs on these trainers.
- **`platformdirs` cache relocation.** The repo-local `.cache/` is consistent with how the app
  resolves `.env`/models and is self-contained; not worth the churn.

## GitHub About / topics

Kept in sync with the code so the repo blurb is accurate. Current copy (refreshed for the
Style/Concept release):

- *Description:* "Turn a character, style, or concept into a ready-to-train LoRA dataset —
  consistent multi-angle shots, smart captions (prose / Danbooru / e621 / dedicated taggers), and
  trainer-ready configs for Flux, SDXL, Krea, Pony & more. Local or cloud, per stage. No training
  launched, no telemetry."
- *Topics:* lora, lora-training, dataset-generation, stable-diffusion, flux, sdxl, pony-diffusion,
  image-captioning, wd-tagger, sam3, comfyui, qwen-image, ai-toolkit, kohya, gradio, huggingface,
  generative-ai.

## Security posture

- UI binds to `127.0.0.1`; no auth layer, so it must not be exposed (`share=True` is deliberately
  not used). `_allowed_media_paths()` widens Gradio's `allowed_paths` to the drive roots (`/` on
  POSIX, each present drive on Windows) so galleries can preview images in **any** user-entered
  input/output folder — the app is designed to point any tab at any folder, and `allowed_paths` is
  fixed at launch, so it cannot be narrowed per-request as the user types new paths. The tradeoff,
  stated plainly: **while the app runs, the localhost file endpoint can serve any file the process
  can read** (`http://127.0.0.1:7861/file=…`). That is acceptable *only* because the bind is
  localhost-with-no-auth — the same trust boundary the whole app already assumes. Mitigation is
  operational, not code: keep the bind on `127.0.0.1`, never add `share=True`, and don't forward or
  expose the port. (Considered and rejected: narrowing to the configured roots or to `$HOME` — both
  break the standalone "preview images from any folder" workflow for datasets stored elsewhere, for
  a read surface that the localhost boundary already contains.)
- **Costs and content are the user's responsibility**, stated in-app (header banner + "Costs & your
  responsibility" accordion) and in the README. Cloud calls bill the user's own key; custom
  endpoints bill whatever provider the user points at. No charges flow through this project.
- API keys live in `.env` (gitignored; `setup.sh` chmods it 600) or the environment, and are only
  sent to their own vendor endpoints.
- No telemetry; the only network calls are the ones the selected backends require (plus the
  best-effort, disable-able update check).
- **Publishing to Hugging Face is opt-in, private by default, and outbound.** It runs only on an
  explicit button/`--publish-hf`, creates the dataset private unless the user chooses public, and
  reads the write token only from `HF_TOKEN` (never persisted). It uploads every image in the
  dataset folder to a remote host — the in-app notice states the user is responsible for the rights
  to that content and for HF's terms. Once uploaded, content may be cached/indexed even if later
  deleted.
- Local model weights come from Hugging Face as safetensors; the WD/Z3D taggers download an ONNX
  model + tag CSV from a public repo (no gating). `onnxruntime` is an optional, lazy dep.
- The ⑤ Train tab **generates and displays** config files and a run command — it never executes a
  subprocess or shell, so there is no command-injection surface. Generated configs reference the
  user's own model ids/paths; no credentials are embedded.
- `.cache/user_settings.json` (gitignored) stores filesystem paths + hyperparameters only.
</content>
