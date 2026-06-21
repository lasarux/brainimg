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
| Image generation (decoder, `--model sd15-turbo` / `sdxl-turbo`) | PyTorch | same SD 1.5 / SDXL base + ControlNets + ByteDance **Hyper-SD** 8-step distilled LoRA (`ByteDance/Hyper-SD`) |
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

### `--model sd15-turbo` / `--model sdxl-turbo`: Hyper-SD distilled backends

ByteDance's **Hyper-SD** (arXiv 2404.13686) trajectory-segmented consistency
distillation ships small (~70-150 MB) LoRAs that fold the SD 1.5 / SDXL base
models down to **8 inference steps** while keeping the stock ControlNets in
play. Same base + VAE + depth/canny/seg ControlNets as `sd15` / `sdxl`; the
LoRA is loaded + `fuse_lora(0.125)` + the scheduler is swapped to
`DDIMScheduler(timestep_spacing="trailing")` per the model card.

- **8 steps** (vs 20-30 for the non-turbo paths) — the main win on a CPU-only
  box, where each step costs the same wall time. Measured on the AMD CPU
  target with `samples/lenna.tiff` (512x512, same seed + blueprint, after the
  ControlNet scale tuning below): SD 1.5 turbo **50.1 s / 9.65 dB PSNR**
  vs ~3 min / 8.70 dB for the 30-step path with the old defaults
  (~3.5x faster and **+0.95 dB** — the distilled schedule + tuned scales both
  help); SDXL turbo **84.2 s** vs ~17 min for the 30-step path at 512²
  (~12x faster, at a small −0.23 dB cost).
- **guidance_scale 7.0/7.5** (CFG-preserved LoRA; supports 5-8 if you tune
  `--cfg`). The 1/2/4-step LoRAs on the same repo want `--cfg 0`; not wired
  up by default.
- **Same fidelity maps** as `sd15` / `sdxl` — depth + canny + optional seg.
  No schema change; the blueprint is identical.
- Small quality cost vs the 30-step non-turbo path on most images
  (distillation trades a little detail for the speedup); use `sd15` /
  `sdxl` when fidelity matters most and wall time is not the bottleneck.
  On some images (Lenna included) the distilled schedule actually wins.

```bash
# CPU-only with lots of RAM (the brainimg target): 8-step SDXL @ 1024
python decoder.py out.brainimg -o recon.png --model sdxl-turbo --device cpu

# CPU-only SD 1.5 turbo @ 512
python decoder.py out.brainimg -o recon.png --model sd15-turbo --device cpu
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

Requires Python 3.12 and [`uv`](https://github.com/astral-sh/uv). The current
target is an **x86/x64 CPU box with abundant RAM** (the dev machine is an AMD
CPU-only system with 188 GB RAM); the decoder runs full fp32 SD 1.5 / SDXL /
Z-Image / FLUX without quantization. Apple Silicon (MLX captioning + MPS
int8 decoder) and NVIDIA CUDA (fp16) still work but are no longer the focus.

```bash
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

> On a non-Apple platform, uninstall the non-functional `mlx`/`mlx-vlm` stub
> wheels (they ship without `libmlx.so`): `pip uninstall -y mlx mlx-vlm mlx-lm`.
> Captioning then falls back to the transformers Qwen2.5-VL-7B model.

The first run downloads the models to `~/.cache/huggingface` (captioner ~15 GB
for the 7B on CPU; depth + seg ~1 GB; decoder ~3.5 GB SD 1.5 / ~7 GB SDXL /
~18 GB Z-Image / ~22 GB FLUX).

## Usage

```bash
# Encode: photo -> tiny .brainimg blueprint
python encoder.py samples/real.jpg -o out.brainimg

# Decode: blueprint -> regenerated image
python decoder.py out.brainimg -o recon.png

# CPU mode: full fp32, best fidelity (the AMD CPU target; needs ~10 GB RAM, slow)
python decoder.py out.brainimg -o recon.png --device cpu

# CPU + Hyper-SD 8-step turbo: ~4x faster, small quality cost
python decoder.py out.brainimg -o recon.png --device cpu --model sd15-turbo
python decoder.py out.brainimg -o recon.png --device cpu --model sdxl-turbo --size 1024x1024

# CPU + int8 weights: fits in ~5 GB RAM, small quality cost
python decoder.py out.brainimg -o recon.png --device cpu --quantize

# CUDA: fp16, fast and high fidelity (NVIDIA GPUs)
python decoder.py out.brainimg -o recon.png --device cuda

# Larger output on a high-RAM machine
python decoder.py out.brainimg -o recon.png --device cpu --size 512x512

# Tune ControlNet scales / guidance (defaults: depth 0.8, canny 1.0, seg 1.0, cfg 7.5)
python decoder.py out.brainimg -o recon.png --device cpu \
    --depth-scale 1.8 --canny-scale 1.0 --seg-scale 1.1 --cfg 8.5
```

Encoder prints the compression ratio. Decoder prints the device, seed, and
generation time. Re-running the decoder with the same seed reproduces the
same image **exactly** (verified: 0 pixel difference between runs).

### Device modes

The dev target is an AMD x86_64 CPU-only box with 188 GB RAM, so the CPU fp32
columns are the primary path. Apple Silicon (MPS int8) and NVIDIA (CUDA fp16)
remain supported but are secondary.

| `--device` + `--model` | Precision | RAM needed | Speed | Fidelity |
|---|---|---|---|---|
| `cpu` (default target) | fp32 (no quantization) | ~10 GB SD 1.5 / ~17 GB SDXL | slow (min/image, SDXL @ 1024) | **best** (sd15/sdxl) |
| `cpu --quantize` | int8 weights, fp32 activations | ~5 GB / ~9 GB | slow | good |
| `cpu --model sd15-turbo` | fp32 + Hyper-SD 8-step LoRA | ~10 GB | **~52 s @ 512²** | good |
| `cpu --model sdxl-turbo` | fp32 + Hyper-SD 8-step LoRA | ~17 GB | **~84 s @ 512²** | good |
| `cpu --model zimage` | bf16 (resident in RAM) | ~18 GB | very slow (8 steps, but big DiT) | good (depth-only) |
| `cpu --model flux-depth --quantize` | bf16 + FP8 (RAM) | ~12 GB | very slow (30 steps) | good (depth-only) |
| `cpu --model flux-depth` | bf16 (resident in RAM) | ~22 GB | very slow (30 steps) | good (depth-only) |
| `mps` | int8 weights + activations | ~5 GB | medium (8 GB Mac) | fair |
| `cuda` | fp16 | ~5 GB | **fast** | good |
| `--model zimage --device cuda` | bf16 | ~18 GB | **fast** (8 steps) | good (depth-only) |
| `--model flux-depth --device cuda` | bf16 | ~22 GB | medium (30 steps) | good (depth-only) |
| `--model flux-depth --device cuda --quantize` | bf16 + FP8 weights | ~12 GB | medium | good (depth-only) |

Use `--device cpu` on a high-RAM machine for the best reconstruction quality.
Add `--model sd15-turbo` / `sdxl-turbo` when wall time matters more than the
last few percent of fidelity.

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
  On the AMD CPU target (188 GB RAM), all backends run full fp32 / bf16 with
  no quantization.
- **Slow on CPU**: minutes per image at 20-30 steps. `--model sd15-turbo` /
  `sdxl-turbo` (Hyper-SD 8-step LoRA) cuts wall time ~4x at a small quality
  cost. `--device cuda` is much faster when a GPU is available.
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

## Verified results (AMD CPU, 188 GB RAM)

Lenna round-trip (`samples/lenna.tiff`, same blueprint + seed 916570520,
512x512 output). MSE / PSNR / MAE computed against the original at 512x512.

| Backend | Steps | Wall time | MSE | PSNR (dB) | MAE |
|---|---|---|---|---|---|
| `sd15` (30-step, old scales 1.5/1.2/0.9) | 30 | ~3 min | 8762.95 | 8.70 | 77.54 |
| `sd15` (30-step, tuned scales 0.8/1.0/1.0) | 30 | 156 s | 7560.33 | 9.35 | 70.80 |
| `sd15-turbo` (8-step, tuned scales) | 8 | **50.1 s** | 7055.30 | 9.65 | 68.10 |
| `sdxl` @ 512 (30-step) | 30 | 220 s | 5774.05 | 10.52 | 58.79 |
| `sdxl-turbo` @ 512 (8-step) | 8 | **69.3 s** | 6085.01 | 10.29 | 61.10 |
| `zimage` (depth-only) | 8 | 237 s | 7651.40 | 9.29 | 70.31 |
| `flux-depth` (FP8) | 30 | 654 s | 3202.12 | 13.08 | 43.63 |
| **`flux-depth-turbo`** (FP8) | 8 | **166 s** | **2314.24** | **14.49** | **37.05** |

Notes: All decodes at 512x512 on the AMD CPU target (188 GB RAM), same
blueprint + seed 916570520. The scale tuning (depth 1.5 -> 0.8, canny 1.2 ->
1.0, seg 0.9 -> 1.0) lifted SD 1.5 from 8.70 to 9.35 dB (+0.65 dB) on the
30-step path and to 9.65 dB (+0.95 dB vs the old baseline) on the 8-step
turbo path. SDXL turbo at 512² is within ~0.23 dB of the 30-step SDXL at
the same size, at ~3x less wall time. **FLUX depth turbo is the best
result across all backends**: 14.49 dB at 166 s — the 8-step distilled
schedule actually beats the 30-step FLUX (13.08 dB) by +1.41 dB at ~4x
less wall time. The Hyper-SD FLUX LoRA was trained on base FLUX.1-dev, not
the Control variants; the decoder strips the `x_embedder` /
`context_embedder` LoRA deltas (shape-incompatible with the Control
transformer's extra input channels) and keeps the attention/FFN deltas.
Z-Image is depth-only so it ignores the canny/seg maps; competitive with
SD 1.5 turbo on MSE despite using only one conditioning map. See
`lenna_grid.jpg` for a combined side-by-side grid of all backends.

## Project layout

```
brainimg/          # package: format, device, extract, generate
encoder.py         # CLI entry: image -> .brainimg
decoder.py         # CLI entry: .brainimg -> image
scripts/           # sample-image generator
tests/             # format round-trip + schema tests
samples/           # bundled test images
```
