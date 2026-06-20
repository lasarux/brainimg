# brainimg

A semantic image compression experiment. Instead of storing pixels, brainimg
stores the **meaning** of an image (a text caption) plus a tiny **structural
blueprint** (a 64x64 depth map + 64x64 Canny edge map) and a seed. On decode,
Stable Diffusion + two ControlNets regenerate a visually faithful image at any
resolution.

> This is a research toy, not a replacement for JPEG. See `PLAN.md` for the
> motivation and the "Semantic-Relational Field" paradigm.

## Architecture (hybrid)

| Stage | Framework | Model |
|---|---|---|
| Captioning (encoder) | **MLX** (Apple Silicon) / **transformers** (CPU/CUDA) | `mlx-community/Qwen2-VL-2B-Instruct-4bit` / `Qwen/Qwen2.5-VL-7B-Instruct` |
| Depth map (encoder) | PyTorch + MPS | `depth-anything/Depth-Anything-V2-Base-hf` |
| Canny edges (encoder) | OpenCV | — |
| Segmentation (encoder) | PyTorch (OneFormer ADE20K) | `shi-labs/oneformer_ade20k_swin_tiny` |
| Image generation (decoder) | PyTorch + MPS | `stable-diffusion-v1-5/stable-diffusion-v1-5` + `lllyasviel/control_v11f1p_sd15_depth` + `lllyasviel/control_v11p_sd15_canny` (+ `lllyasviel/control_v11p_sd15_seg` when the blueprint has a seg map) |
| Image generation (decoder, `--model zimage`) | PyTorch + bf16 | `Tongyi-MAI/Z-Image-Turbo` (6B DiT) + `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` (8-step distill) |
| Image generation (decoder, `--model flux-depth` / `--model flux-canny`) | PyTorch + bf16 (+ optional FP8) | `black-forest-labs/FLUX.1-Depth-dev` / `FLUX.1-Canny-dev` (12B MMDiT + T5-XXL; one conditioning image, channel-concat) |

Captioning uses the MLX 4-bit model on Apple Silicon (fast, low memory) and
falls back to the HuggingFace transformers Qwen2.5-VL-7B model on any other
platform (x86/x64 CPUs, CUDA). Both produce an equivalent caption; the 7B is
noticeably more detailed.

The segmentation map is an **optional** field (added after the initial v0.1
release), so older `.brainimg` files without it still decode exactly as before
— the decoder just uses the two ControlNets (depth + Canny). Newer files carry
an ADE20K colorized seg map and the decoder adds a third ControlNet for it.

### `--model flux-depth` / `--model flux-canny`: FLUX backend

Black Forest Labs' **FLUX.1 guidance-distilled** variants via diffusers'
`FluxControlPipeline`. Same architectural pattern as Z-Image's Union net:
**one conditioning image per call**, concatenated into the transformer
channels (NOT a separate ControlNet model). The conditioning *type* is
baked into which FLUX.1-*-dev checkpoint you load.

- **`flux-depth`**: `FLUX.1-Depth-dev` + the blueprint's `depth_map_b64`.
  Strongest structural grip; the natural counterpart to Z-Image's default.
- **`flux-canny`**: `FLUX.1-Canny-dev` + `canny_map_b64`. Edge-faithful;
  best for line-art / architectural subjects where edges dominate.

The blueprint's other map (and any `segmentation_map_b64`) are silently
ignored on this path — no schema change. bf16 throughout (FLUX is bf16-native
and bf16 sidesteps the MPS fp16 NaN bug); `max_sequence_length=512`; ~30
steps; per-model guidance scale (depth: 10.0, canny: 30.0).

Memory cost is the headline tradeoff: T5-XXL (~9.5 GB bf16) + transformer
(~12 GB bf16) + CLIP-L (~0.5 GB) + VAE (~0.2 GB) ≈ **22 GB resident**.
`--quantize` FP8-quantizes the transformer + T5 via `optimum.quanto`
(`qfloat8` weights-only — activation quant is brittle on FLUX), dropping
resident to ~12 GB at a small quality cost. 8 GB Apple Silicon is **not
supported** — use `--model sd15` there.

```bash
# CUDA: fast + full bf16 (needs ~22 GB VRAM)
python decoder.py out.brainimg -o recon.png --model flux-depth --device cuda

# CUDA + FP8: fits in ~12 GB VRAM, small quality cost
python decoder.py out.brainimg -o recon.png --model flux-depth --device cuda --quantize

# CPU-only with FP8: ~12 GB RAM resident, slow but works (recommended for CPU)
python decoder.py out.brainimg -o recon.png --model flux-depth --device cpu --quantize

# CPU without --quantize: ~22 GB RAM resident (won't fit most dev boxes)
python decoder.py out.brainimg -o recon.png --model flux-depth --device cpu
```

### `--model zimage`: Z-Image-Turbo backend

An alternative decoder backed by Tongyi-MAI's **Z-Image-Turbo** (a 6B
single-stream DiT) with Alibaba-PAI's **Union ControlNet**. Key differences
from the SD 1.5/SDXL path:

- **Depth-only conditioning.** The Union ControlNet takes a *single*
  conditioning image per call (it supports canny/depth/pose/mlsd/hed). We feed
  the depth map. The blueprint's Canny and segmentation maps are **ignored**
  on this path — no schema change, they're just unused. This trades a little
  edge fidelity for Z-Image's photorealism.
- **bf16 throughout.** Z-Image's native dtype is bfloat16, which sidesteps the
  MPS fp16 NaN bug entirely. No int8 quantization is used on this path.
- **8 steps.** The ControlNet is the 8-step-distilled `2.1-8steps` variant
  (the *lite* 2601/2602 files don't load cleanly under diffusers 0.38 — their
  widened `control_all_x_embedder` triggers a shape mismatch), so Turbo's
  sub-second latency is preserved (vs ~20-40 steps for the undistilled 2.1).
- **~18 GB VRAM minimum on GPU.** The 6B DiT (~12 GB bf16) + 6.4 GB
  8-step-distilled Union ControlNet don't fit in 8 GB. On `mps` the pipeline
  uses `enable_model_cpu_offload()` (layers stream between host and device) —
  slow. On a **CPU-only** box the whole bf16 pipeline is kept resident in host
  RAM (~18 GB); there is no layer offload trick on CPU (diffusers'
  `enable_model_cpu_offload` requires an accelerator to move *to*). For the
  8 GB Apple Silicon target or low-RAM CPU boxes, stay on `--model sd15`.
- **guidance_scale 0.0.** Turbo is distilled for zero CFG; the `color_style`
  prefix is prepended unconditionally (Z-Image's Qwen text encoder has a
  512-token limit vs CLIP's 77).

```bash
# CUDA: fast and high fidelity (needs ~16 GB VRAM)
python decoder.py out.brainimg -o recon.png --model zimage --device cuda

# CPU-only: works but slow, needs ~14 GB RAM resident (no offload on CPU)
python decoder.py out.brainimg -o recon.png --model zimage --device cpu
```

Encoder and decoder are separate processes, so models are never resident at the
same time (important on an 8 GB Apple Silicon Mac).

### Decode quality enhancements

The decoder applies several fixes for SD 1.5's known weaknesses (no schema
change; uses data already in the file):

- **`sd-vae-ft-mse` VAE**: the stock SD 1.5 VAE is swapped for the fine-tuned
  MSE VAE — cleaner decode, better skin tones and colors, fewer washed-out
  highlights. Tiny (~335 MB), no runtime cost.
- **Brightness/saturation matching**: SD 1.5 tends to over-brighten and
  over-saturate. The blueprint stores the original image's
  `target_brightness` / `target_saturation`, and the decoder post-processes
  the generation (uniform RGB gain for brightness, HSV-S scaling for
  saturation) to match. A no-op for older files with no stored stats.
- **Color style prefix**: the encoder's mood descriptor
  (`"dark, low-key lighting, red dominant tones"`, ...) is stored in `extra`
  and prepended to the caption **only when the combined length fits the CLIP
  77-token limit** — so it biases the mood without ever truncating the caption.
- **30 default steps** (was 20), and tunable ControlNet scales / CFG via CLI.

### Why int8 quantization?

On Apple Silicon (MPS), SD 1.5 in fp16 produces NaNs (a black output frame)
because MPS fp16 matmuls are numerically unstable for this model. The decoder
therefore quantizes the UNet and both ControlNets to **int8 weights + int8
activations** via [`optimum-quanto`](https://github.com/huggingface/optimum-quanto),
which avoids fp16 matmuls entirely and roughly halves memory. The VAE runs in
fp32 for a clean final decode. This is the Apple Silicon equivalent of the
"fp8 on H100" trick.

## Install

Requires Python 3.12 and [`uv`](https://github.com/astral-sh/uv). Runs on
Apple Silicon (MLX for captioning) or any x86/x64 CPU / NVIDIA CUDA machine
(transformers Qwen2.5-VL-7B fallback for captioning).

```bash
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

> On a non-Apple platform, uninstall the non-functional `mlx`/`mlx-vlm` stub
> wheels (they ship without `libmlx.so`): `pip uninstall -y mlx mlx-vlm mlx-lm`.

The first run downloads the models to `~/.cache/huggingface` (captioner ~15 GB
for the 7B on non-Apple, ~2 GB for the MLX 4-bit on Apple; depth + seg ~1 GB;
decoder ~3.5 GB). Close memory-hungry apps (browsers, etc.) before decoding on
an 8 GB machine.

## Usage

```bash
# Encode: photo -> tiny .brainimg blueprint
python encoder.py samples/real.jpg -o out.brainimg

# Decode: blueprint -> regenerated image
python decoder.py out.brainimg -o recon.png

# CPU mode: full fp32, best fidelity (needs ~10 GB RAM, slow)
python decoder.py out.brainimg -o recon.png --device cpu

# CPU + int8 weights: fits in ~5 GB RAM, small quality cost
python decoder.py out.brainimg -o recon.png --device cpu --quantize

# CUDA: fp16, fast and high fidelity (NVIDIA GPUs)
python decoder.py out.brainimg -o recon.png --device cuda

# Larger output on a high-RAM machine
python decoder.py out.brainimg -o recon.png --device cpu --size 512x512

# Tune ControlNet scales / guidance (defaults: depth 1.5, canny 1.2, seg 0.9, cfg 7.5)
python decoder.py out.brainimg -o recon.png --device cpu \
    --depth-scale 1.8 --canny-scale 1.0 --seg-scale 1.1 --cfg 8.5
```

Encoder prints the compression ratio. Decoder prints the device, seed, and
generation time. Re-running the decoder with the same seed reproduces the
same image **exactly** (verified: 0 pixel difference between runs).

### Device modes

| `--device` | Precision | RAM needed | Speed | Fidelity |
|---|---|---|---|---|
| `auto` (default) | detects best available | varies | varies | varies |
| `cpu` | fp32 (no quantization) | ~10 GB | slow (min/image) | **best** (sd15/sdxl) |
| `cpu --quantize` | int8 weights, fp32 activations | ~5 GB | slow | good |
| `mps` | int8 weights + activations | ~5 GB | medium (8 GB Mac) | fair |
| `cuda` | fp16 | ~5 GB | **fast** | good |
| `--model zimage --device cuda` | bf16 | ~18 GB | **fast** (8 steps) | good (depth-only) |
| `--model zimage --device mps` | bf16 + cpu-offload | varies | slow | good (depth-only) |
| `--model zimage --device cpu` | bf16 (resident) | ~18 GB RAM | very slow | good (depth-only) |
| `--model flux-depth --device cuda` | bf16 | ~22 GB | medium (30 steps) | good (depth-only) |
| `--model flux-depth --device cuda --quantize` | bf16 + FP8 weights | ~12 GB | medium | good (depth-only) |
| `--model flux-depth --device cpu --quantize` | bf16 + FP8 (RAM) | ~12 GB | very slow | good (depth-only) |
| `--model flux-depth --device cpu` | bf16 (resident in RAM) | ~22 GB RAM | very slow | good (depth-only) |

Use `--device cpu` on a high-RAM machine for the best reconstruction quality.

## `.brainimg` file

A small JSON document, typically 3-10 KB regardless of source resolution:

```json
{
  "format_version": "0.1",
  "original_width": 1024, "original_height": 768,
  "prompt": "a red apple on a wooden table next to a window",
  "negative_prompt": "blurry, low quality, deformed",
  "depth_map_b64": "...", "canny_map_b64": "...",
  "segmentation_map_b64": "...",
  "seed": 42, "steps": 20
}
```

`segmentation_map_b64` is optional; older files omit it.

## Tests

```bash
uv pip install pytest
pytest                       # format tests, no models needed
```

## Limitations

- **Lossy by design**: reconstruction is semantically faithful (same scene,
  layout, lighting) but not pixel-identical. Do not use for medical, legal, or
  forensic images.
- **Quality depends on device**: `--device cpu` (full fp32) gives the best
  reconstruction. On 8 GB Apple Silicon (`--device mps`), int8 quantization is
  required to avoid the MPS fp16 NaN bug, which degrades structural fidelity.
- **256x256 max on 8 GB Apple Silicon**: 512x512 OOMs on MPS. On a high-RAM
  machine with `--device cpu`, 512x512 works fine.
- **Slow on CPU**: minutes per image. `--device cuda` is much faster.
- **Deterministic given the seed**: re-running with the same seed reproduces
  the same image exactly.
- **Z-Image path is depth-only**: `--model zimage` ignores the blueprint's
  Canny and segmentation maps (the Union ControlNet takes one image and has no
  seg mode). Edge detail is slightly lower than the SD 1.5/SDXL three-net
  stack; photorealism is higher. Needs ~18 GB VRAM on GPU or ~18 GB RAM on CPU
  (no offload on a CPU-only box); not for 8 GB Apple Silicon.
- **SDXL hue drift at small sizes**: SDXL @ 512 can land in a different hue
  *distribution* than the source (e.g. orange/yellow when the source is
  pink/magenta). The decoder's brightness/saturation matching cannot correct
  this — it is a content/palette drift, not a stat drift. SDXL @ 1024 is
  much closer to the source palette. Workaround: prefer `--model sdxl
  --size 1024x1024`, or use `--model sd15` when the source's color palette
  matters most. See TODO.md "SDXL hue distribution drift" for details.
- **FLUX is heavy**: FLUX is T5-XXL + a 12B transformer (~22 GB bf16
  resident). On CPU you almost always need `--quantize` (FP8 weights,
  drops to ~12 GB). CUDA GPUs need ~22 GB VRAM (or ~12 GB with
  `--quantize`). 8 GB Apple Silicon is **not supported** — fall back
  to `--model sd15`. Like SDXL, FLUX is trained at 1024 and may show
  the same hue-distribution drift at smaller sizes; prefer
  `--size 1024x1024` for fidelity. FLUX.1-Depth-dev /
  FLUX.1-Canny-dev carry the FLUX.1 non-commercial license.


## Verified results (M1, 8 GB)

| Image | Original | brainimg | Ratio | Decode |
|---|---|---|---|---|
| `samples/real.jpg` (puppy) | 13.4 KB | 2.7 KB | 5.0x | 59 s @ 256x256 |

The captioner correctly described the scene ("a black puppy sitting on a
wooden surface"); the decoder produced a visually faithful reconstruction. See
`comparison.jpg` for a side-by-side.

## Project layout

```
brainimg/          # package: format, device, extract, generate
encoder.py         # CLI entry: image -> .brainimg
decoder.py         # CLI entry: .brainimg -> image
scripts/           # sample-image generator
tests/             # format round-trip + schema tests
samples/           # bundled test images
```
