# ComfyUI setup (optional — fully-local generation & restoration)

ComfyUI is **not required**: cloud generation, built-in SAM3 isolation, local
captioning, and export all work without it. Install it only if you want:

- **Fully-local image generation** (free, private, uncensored) — Qwen-Image 2.1, or
- **Model-based photo restoration** (DeJPG + photo upscaler) for degraded sources, or
- To run **SAM3 isolation inside ComfyUI** instead of in-process.

## 1. Install ComfyUI

Follow <https://github.com/comfyanonymous/ComfyUI> (or use the desktop installer /
ComfyUI portable). A recent build is required — the SAM3 nodes (`SAM3_Detect`) and the
Qwen-Image 2.1 nodes (`TextEncodeQwenImage21`) are part of ComfyUI core in current releases. Start it on the default port; if yours
differs, set `LDS_COMFY_URL` in `.env`.

**No custom nodes are needed.** Every bundled workflow uses only core ComfyUI nodes.
If a workflow reports a missing node, your build is too old — update ComfyUI rather
than installing a node pack.

## 2. Download models

Place these in your ComfyUI `models/` tree (filenames are configurable in `.env`
if yours differ — see `.env.example`):

| Purpose | File (default name) | Goes in | Source |
|---|---|---|---|
| Qwen-Image 2.1 model | `qwen_image_2.1_int8_convrot.safetensors` | `models/diffusion_models/` (a.k.a. `unet/`) | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) (`diffusion_models/`) — a bf16 file also exists there; set `LDS_QWEN21_MODEL` to whichever filename you have. **Qwen Research License (non-commercial).** |
| Qwen3-VL text encoder | `qwen3vl_8b_int8_convrot.safetensors` | `models/text_encoders/` | same repo (`text_encoders/`); set `LDS_QWEN21_TEXT_ENCODER` if yours is named differently |
| Qwen-Image 2.1 VAE | `qwen_image_2.1_vae_bf16.safetensors` | `models/vae/` | same repo (`vae/`); set `LDS_QWEN21_VAE` if yours is named differently |
| SAM3 (ComfyUI backend only) | `sam3.1_multiplex_fp16.safetensors` | `models/checkpoints/` | [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) |
| Restoration: JPEG cleanup | `1xDeJPG_OmniSR.pth` | `models/upscale_models/` | [OpenModelDB](https://openmodeldb.info/models/1x-DeJPG-OmniSR) |
| Restoration: photo upscale | `4xNomosWebPhoto_RealPLKSR.safetensors` | `models/upscale_models/` | [OpenModelDB](https://openmodeldb.info/models/4x-NomosWebPhoto-RealPLKSR) |

## 3. How the app talks to ComfyUI

`studio/comfy_workflows/*.json` are API-format graphs submitted over ComfyUI's HTTP
API — you don't need to load anything manually. At load time the app patches the model
filenames in each graph from your settings, so a renamed file only needs an `.env` line,
never a JSON edit.

Workflows used:

- `qwen21_edit.json` — generation: one reference image plus one plain-English instruction
  per shot (~26 s a shot on an RTX 5090). The old Qwen-Image-Edit-2511 + Multiple-Angles
  graph is kept, unwired, in `studio/comfy_workflows/legacy/` (see its README)
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
