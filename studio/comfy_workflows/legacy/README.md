# Legacy workflows

Kept for anyone who prefers them. **Nothing in the app loads this folder**, and the
template tests and `doctor` only scan `studio/comfy_workflows/*.json`.

## `qwen_edit_2511.json` — Qwen-Image-Edit-2511 + Multiple-Angles LoRA

The local ② generator up to v0.16.0 (Apache-2.0 models; the current engine,
Qwen-Image 2.1, is under the Qwen Research License). API-format graph with two
placeholders and hard-coded model filenames:

- `__SOURCE__` — the uploaded reference's filename, in node 1 (`LoadImage`)
- `__PROMPT__` — the prompt, in node 9 (`TextEncodeQwenImageEditPlus`)

To use it, edit the model filenames (UNET, text encoder, VAE, LoRA) to match your
ComfyUI, fill in the placeholders and POST `{"prompt": <graph>}` to ComfyUI's
`/prompt` — or rebuild it in the ComfyUI editor from the node list.

Prompts for it were written in the LoRA's `<sks>` grammar, e.g.
`<sks> front-right quarter view eye-level shot medium shot` (LoRA strength 0.9 for
angle shots, 0 for pose/emotion), and plain English for everything else. Those
prompt builders live in the `v0.16.0` git tag (`studio/shotplan.py`,
`studio/engines/comfyui.py`); the LoRA is
[fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA).
