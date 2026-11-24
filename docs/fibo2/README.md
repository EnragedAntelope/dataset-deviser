# Dataset Deviser: switch the local engine to Fibo 2.0 Edit Turbo

Spec only. No deviser code was changed. The template `fibo2_edit.json` sits next to this file; it lives under `docs/` on purpose, so `doctor` and the template tests do not pick it up until the engine supports it.

## Recommendation

Use **Fibo 2.0 Edit Turbo** (8 steps, CFG 1, euler/simple) at **4 MP with ScaleROPE** for
photographic turnaround shots. It replaces Qwen-Image 2.1 in the local ComfyUI engine.

Evidence (local A/B, 2026-11-24, RTX 5090, 3 stock subjects, deviser's own photographic
shot prompts, seed 1). Likeness = ArcFace cosine similarity of the generated face to the
reference face (same photo flipped = 0.93-0.96, different people < 0.09).

| Model / setting | Mean likeness | Angle and scene following | Look |
|---|---|---|---|
| Qwen-Image 2.1 (current) | 0.739 | Weak: often keeps the reference pose and background | Photo |
| **Fibo 2.0 Edit Turbo, 4 MP** | **0.80** | Strong: real 3/4, profile, back and seated full-body views | Photographic, detailed skin |
| Fibo 2.0 Edit Turbo, 1 MP (default size) | 0.785 | 3/4 heads on "full profile" | Soft faces |
| Fibo 2.0 Dev, 24 steps CFG 5 | 0.81 | Turns the face to camera on back views | Softer, painterly; ~4x slower |
| MiniMax H3-Image (4-24 MP, face ref, recon VAE) | 0.57-0.63 | Strong | Most filmic, but identity drifts |
| SenseNova U1.6 (8-step LoRA / base) | 0.40 / 0.22 | Strong | HDR, cartoonish: rejected |

Samplers and schedulers (euler 6/8, euler_ancestral, er_sde, res_2s 4/8, beta, bong_tangent)
all scored 0.77-0.80 on Fibo2 Turbo (seed 1). A seed-2 recheck of the top three gave euler 8 0.842,
euler 6 0.839, res_2s 4 0.848, with no visible difference. Within noise; euler/simple 8 stays
(res_2s is a custom-pack sampler, not core).
Citation tokens (`<image_1>`) changed nothing, so deviser's current prompt text works as is.

## Template contract (`fibo2_edit.json`, API format, core nodes only)

It keeps the node ids `ComfyUIEngine.generate` already writes, so the engine code does not
need to change to run it:

| Node | Class | Engine writes |
|---|---|---|
| `1` | LoadImage | first reference (sets the output aspect) |
| `5` | TextEncodeFibo2 | `prompt`; extra references as `images.image_N` from LoadImage `10N` (N >= 2) |
| `6` | KSampler | `seed` |
| `10` | PrimitiveFloat | output megapixels (default 4.0) - new knob, see below |

The engine also writes `5.resolution` (a Qwen-only input). ComfyUI ignores unknown inputs, so
this is harmless; verified by a run.

Inside the template: `GetImageSize` + `ComfyMathExpression` compute width/height for the
reference's aspect at the chosen megapixels (multiple of 16), and a third expression feeds
`ScaleROPE` scale_x/scale_y = sqrt(1 / MP). Without ScaleROPE, Fibo 2.0 repeats people
above ~1 MP. `ModelAttentionBackend` is set to Comfy Kitchen attention (INT8, Nvidia/AMD
only; use `pytorch attention` elsewhere).

## Zero-code trial (works today)

1. Copy `docs/fibo2/fibo2_edit.json` over `studio/comfy_workflows/qwen21_edit.json` in a local
   checkout (do not commit).
2. In `.env`:
   ```
   LDS_QWEN21_MODEL=fibo/fibo2_edit_turbo_bf16.safetensors
   LDS_QWEN21_TEXT_ENCODER=qwen3vl_4b_bf16.safetensors
   LDS_QWEN21_VAE=flux2-vae.safetensors
   ```
   `load_template` remaps `unet_name` / `clip_name` / `vae_name` from these. The CLIPLoader
   `type: fibo2` is in the template and is not remapped. `LDS_QWEN21_RESOLUTION` is ignored.
   `LDS_QWEN21_MAX_REFS` works unchanged.

## Proper changes (for the implementing session)

1. **Model choice setting.** `LDS_LOCAL_MODEL = qwen21 | fibo2` (default fibo2 once landed).
   Bundle `fibo2_edit.json` next to `qwen21_edit.json`; `ComfyUIEngine.generate` loads the
   template named by the setting.
2. **Settings per model.** `fibo2_model`, `fibo2_text_encoder`, `fibo2_vae`,
   `fibo2_megapixels` (default 4.0, written to node `10`). `_MODEL_INPUTS` in
   `comfy_api.py` is keyed by input name and maps to `qwen21_*` settings; make it
   per-template (dict keyed by template name) so `doctor` checks the files the chosen
   model needs.
3. **Resolution.** Only write `5.resolution` for qwen21. For fibo2 write `10.value`.
   Tested: 1 MP (soft) and 4 MP (good). Above 4 MP is untested; keep 4 as the default and
   cap the setting at 4 until someone tests higher.
4. **Docs.** `docs/comfyui-setup.md` model table: the three Fibo files below, folder names,
   and the license line. README engine row: "Fibo 2.0 (CC BY-NC 4.0)".
5. **Tests.** `tests/test_cli_preflight.py` names `qwen21_edit`; add the fibo2 template to
   whatever preflight/doctor tests enumerate templates and model settings.
6. **Timeout.** 600 s per shot is ample (Turbo 4 MP is ~45 s on a 5090 when the GPU is free).

## Open items (not decided by the tests)

- **Face close-up as a second reference** (tested, recommended per shot). A square head crop
  of the reference (about 5x the eye distance, from the same photo) passed as `images.image_2`
  raised likeness from 0.813 to 0.849 over the 15 shots where the face should show, and made
  seated full-body shots more complete. Cost: the frontal crop pulls the head toward the
  camera, so full profiles relax to near 3/4 and one back view turned its face to camera.
  Rule for the engine: add the face crop for front, three-quarter, low/high angle, seated and
  emotion shots; leave it out for profile (`angle-right`, `angle-left`) and any back view.
  Deviser does not produce a face crop today; it needs a face detector step (a SCRFD/ArcFace
  ONNX pair, or a box from the existing SAM3 run) and a crop saved next to the source.
- **Per-shot aspect.** Output aspect follows the first reference. Full-body shots from a
  landscape photo come out as small figures. A per-shot portrait canvas would help, but
  TextEncodeFibo2 *center-crops a single reference to the output size*, which could cut the
  body out of the reference. Test before adopting (with the reference also padded, or with a
  second uncropped reference).
- **Low-angle shot** follows the night/moonlight scene but reads as an eye-level portrait.
  A stronger phrase (for example "worm's-eye view from near the ground, looking up at the full
  figure") is untested.
- **Back views** hide the face by design, so likeness scores there are meaningless; Turbo
  produced correct back views, Dev turned the face to camera.
- **INT8 model files** (`fibo2_edit_turbo_int8_convrot`, `qwen3vl_4b_int8_convrot`, about half
  the size) exist for smaller GPUs; their quality was not tested here.

## Model files

From https://huggingface.co/Comfy-Org/Fibo-2.0 :

| File | Folder | Size |
|---|---|---|
| `diffusion_models/fibo2_edit_turbo_bf16.safetensors` | `models/diffusion_models/` | 16.5 GB |
| `text_encoders/qwen3vl_4b_bf16.safetensors` | `models/text_encoders/` | 8.9 GB |
| `vae/flux2-vae.safetensors` | `models/vae/` | 0.3 GB |

**License: CC BY-NC 4.0 (BRIA AI), non-commercial.** Commercial use needs a BRIA
agreement. Qwen-Image 2.1 (Qwen Research License) is also non-commercial, so this is not a
regression; name it the same way the README names Qwen's.
