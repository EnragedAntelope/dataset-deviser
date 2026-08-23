# ComfyUI setup (optional — fully-local generation & restoration)

ComfyUI is **not required**: cloud generation, built-in SAM3 isolation, local
captioning, and export all work without it. Install it only if you want:

- **Fully-local image generation** (free, private, uncensored) — Qwen Image Edit 2511
  with the fal Multiple-Angles LoRA, or
- **Model-based photo restoration** (DeJPG + photo upscaler) for degraded sources, or
- To run **SAM3 isolation inside ComfyUI** instead of in-process.

## 1. Install ComfyUI

Follow <https://github.com/comfyanonymous/ComfyUI> (or use the desktop installer /
ComfyUI portable). A recent build is required — the SAM3 nodes (`SAM3_Detect`) are
part of ComfyUI core in current releases. Start it on the default port; if yours
differs, set `LDS_COMFY_URL` in `.env`.

**No custom nodes are needed.** Every bundled workflow uses only core ComfyUI nodes.
If a workflow reports a missing node, you're on a build too old for core SAM3 — update
ComfyUI rather than installing a node pack.

## 2. Download models

Place these in your ComfyUI `models/` tree (filenames are configurable in `.env`
if yours differ — see `.env.example`):

| Purpose | File (default name) | Goes in | Source |
|---|---|---|---|
| Edit model | `qwen_image_edit_2511_int8_convrot.safetensors` | `models/unet/` (a.k.a. `diffusion_models/`) | any Qwen-Image-Edit-2511 checkpoint packaged for ComfyUI — e.g. the [Comfy-Org repackages](https://huggingface.co/Comfy-Org) or a quantized variant that fits your VRAM; set `LDS_QWEN_EDIT_MODEL` to its filename |
| Qwen text encoder | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `models/text_encoders/` | same source as the edit model (follow its README); set `LDS_QWEN_TEXT_ENCODER` if yours is named differently |
| Qwen VAE | `qwen_image_vae.safetensors` | `models/vae/` | same source as the edit model; set `LDS_QWEN_VAE` if yours is named differently |
| Multi-angle LoRA | `qwen/Qwen-Image-Edit-2511-Multiple-Angles-LoRA.safetensors` | `models/loras/qwen/` | [fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) (Apache-2.0) |
| SAM3 (ComfyUI backend only) | `sam3.1_multiplex_fp16.safetensors` | `models/checkpoints/` | [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) |
| Restoration: JPEG cleanup | `1xDeJPG_OmniSR.pth` | `models/upscale_models/` | [OpenModelDB](https://openmodeldb.info/models/1x-DeJPG-OmniSR) |
| Restoration: photo upscale | `4xNomosWebPhoto_RealPLKSR.safetensors` | `models/upscale_models/` | [OpenModelDB](https://openmodeldb.info/models/4x-NomosWebPhoto-RealPLKSR) |

> The angle LoRA uses fal's `<sks>` camera grammar (`<sks> [azimuth] [elevation]
> [distance]`, 96 poses); the default shot plan already speaks it.

## 3. How the app talks to ComfyUI

`studio/comfy_workflows/*.json` are API-format graphs submitted over ComfyUI's HTTP
API — you don't need to load anything manually. At load time the app patches the model
filenames in each graph from your settings, so a renamed file only needs an `.env` line,
never a JSON edit.

Workflows used:

- `qwen_edit.json` — generation (angles via the LoRA at 0.9 strength, pose/scene edits
  with the LoRA zeroed)
- `restore_upscale.json` — DeJPG → 4× photo upscale
- `isolate_subject.json` / `isolate_exclude.json` — SAM3 cutout onto white, optionally
  removing held props via a second segmentation

Things to know:

- If ComfyUI's queue already has more than 10 pending jobs, the app fails fast rather
  than queueing behind them. Tick **"Prioritize this app's ComfyUI jobs"** on ② to jump
  the pending queue instead. This does **not** interrupt a job already running, so you
  still wait out the one in flight.
- Before local captioning, the app asks ComfyUI to free VRAM (`/free`) so the ~17 GB
  captioner fits.
- ComfyUI caches its model file lists — restart it after adding new model files if a
  workflow reports a missing model that is definitely on disk.
