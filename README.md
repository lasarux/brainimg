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

Captioning uses the MLX 4-bit model on Apple Silicon (fast, low memory) and
falls back to the HuggingFace transformers Qwen2.5-VL-7B model on any other
platform (x86/x64 CPUs, CUDA). Both produce an equivalent caption; the 7B is
noticeably more detailed.

The segmentation map is an **optional** field (added after the initial v0.1
release), so older `.brainimg` files without it still decode exactly as before
— the decoder just uses the two ControlNets (depth + Canny). Newer files carry
an ADE20K colorized seg map and the decoder adds a third ControlNet for it.

Encoder and decoder are separate processes, so models are never resident at the
same time (important on an 8 GB Apple Silicon Mac).

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
```

Encoder prints the compression ratio. Decoder prints the device, seed, and
generation time. Re-running the decoder with the same seed reproduces the
same image **exactly** (verified: 0 pixel difference between runs).

### Device modes

| `--device` | Precision | RAM needed | Speed | Fidelity |
|---|---|---|---|---|
| `auto` (default) | detects best available | varies | varies | varies |
| `cpu` | fp32 (no quantization) | ~10 GB | slow (min/image) | **best** |
| `cpu --quantize` | int8 weights, fp32 activations | ~5 GB | slow | good |
| `mps` | int8 weights + activations | ~5 GB | medium (8 GB Mac) | fair |
| `cuda` | fp16 | ~5 GB | **fast** | good |

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
