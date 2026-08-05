# AGENTS.md — dataset-deviser

Turn a character, style, or concept into a ready-to-train LoRA dataset. One reference image becomes ~24 consistent shots (character) or 18 shots (concept) across angles, poses, and settings. Output is a flat folder (`NN.png` + `NN.txt`) that drops straight into any trainer (ai-toolkit / kohya / OneTrainer / …), plus a generated training config. Gradio UI + Typer CLI, Python 3.10+, local or cloud backends per stage.

**Deep reference: `docs/ARCHITECTURE.md` (read it before pipeline/stage/backend changes)**

## Architecture in 60 seconds

- **Five-stage pipeline:** ① Preprocess → ② Generate & curate → ③ Caption → ④ Export → ⑤ Train config. Each stage is standalone — run them in order or jump to any one.
- **Three dataset types:** Character (identity trigger, 24 generated shots), Style (aesthetic trigger, bring-your-own images), Concept (object/idea trigger, 18 generated shots). Each retunes the shot plan, caption framing, and defaults.
- **Local OR cloud per stage.** Every step has a free/private local path AND a no-GPU cloud path. Mix and match: generate on cloud, caption on GPU, or reverse.
- **Multiple backends:** ComfyUI (generation, preprocessing), built-in SAM3 (isolation), ONNX booru tagger (WD + e621), Qwen3-VL-8B / JoyCaption / NSFW finetune (local captioning), Gemini Flash / Groq (cloud captioning), LM Studio / Ollama / any OpenAI-compatible endpoint.
- **Curate by clicking.** Toggle shots from the gallery with ✅/⬜ marks. Selection carries through stages: what you keep in ② is what ③ captions, and what ③ captions is what ④ preselects.
- **Trainer configs:** Generated for ai-toolkit (incl. SDXL), kohya sd-scripts, musubi-tuner. Training is NEVER launched for you — configs are shown, you run them.
- **Honest by design.** No telemetry, no upsell. Cloud calls bill YOUR key; this tool takes no cut.

## Layout

| File / Directory | Purpose |
|------------------|---------|
| `app.py` | Gradio UI — thin wiring over stage functions (5 tabs) |
| `cli.py` | Typer CLI — one subcommand per stage + `build`, `doctor`, `keys`, `custom-endpoint` |
| `setup.bat` / `setup.sh` | One-time install: Python gate, venv, GPU/CPU torch, requirements, ONNX, keys, doctor |
| `start.bat` / `start.sh` | Launch the UI |
| `studio/` | Core modules: env_keys, doctor, config, pipeline, preprocess, isolate, tagger, dedupe, quality, caption_lint, hf_publish, captioner, etc. |
| `tests/` | pytest test suite |
| `docs/` | Architecture (deep reference), images, worklogs |

## Build / test / run

```bash
# One-time setup (installs venv, torch, requirements, ONNX, configures keys)
setup.bat        # Windows
./setup.sh       # Linux/macOS

# Launch the UI
start.bat        # Windows
./start.sh       # Linux/macOS

# Run tests
pytest

# Lint
ruff check .

# CLI usage
python cli.py --help
python cli.py doctor
python cli.py keys --setup
```

## Conventions & gotchas

- Python ≥3.10. Gradio ≥5.0, <6 (Gradio 6 has a loading-overlay bug).
- torch + torchvision installed separately by setup scripts (CUDA/CPU/macOS specific).
- `studio/env_keys.py` is the ONE implementation of reading/editing .env API keys. Per-shell versions were replaced.
- Caption encoding: `read_caption()` uses utf-8-sig + errors=replace (forgiving sidecar read).
- ONNX tagger: WD (Danbooru) and Z3D (e621) via one code path, differing only in tag file + category schemes.
- Dedupe is advisory (perceptual dHash, numpy-only) — surfaced in export preview, Hamming distance is a slider.
- Quality flags (blur, exposure, contrast) are advisory — surfaced in curate/export views.
- Caption lint is pure string logic (no tokenizer) — estimate_clip_tokens() is a best estimate.
- HF publish is opt-in, private by default, requires HF_TOKEN.

## Security

This file is **public-safe by default**. Never add local paths, credentials, API keys, personal data, infrastructure details, or subscription info.

Before pushing: `pwsh scripts/check-agents-md.ps1 AGENTS.md CLAUDE.md` — must exit 0.

**API keys are managed via `cli.py keys` or the .env file.** Never commit .env or hardcode keys.

Deep architecture, stage details, backend specifics, and gotchas: `docs/ARCHITECTURE.md`.

## Maintenance

**Update rule:** When you change the architecture, build/test commands, or conventions, update this AGENTS.md in the same commit. Keep under 200 lines. Link to `docs/ARCHITECTURE.md` for detail.

**CLAUDE.md:** One-line shim: `@AGENTS.md`.

**New-repo rule:** Create AGENTS.md in the first session a new repo is worked on.

**No-overlap rule:** Explanatory prose lives in one file. AGENTS.md = agent-facing summary; `docs/ARCHITECTURE.md` = deep reference. Identical build/test/run commands may be restated verbatim. Explanatory prose must not be duplicated — link instead.
