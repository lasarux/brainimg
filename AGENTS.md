# AGENTS.md

High-signal notes for OpenCode agents working in this repo. See `README.md`
for the full project description and `TODO.md` for planned decode-quality work.

## Setup

- Python **3.12** (pyproject pins `>=3.12,<3.13`) via [`uv`](https://github.com/astral-sh/uv):
  `uv venv -p 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt`
- The current dev target is an **AMD x86_64 CPU-only box with 188 GB RAM**
  (no CUDA/MPS). The decoder runs full fp32 SD 1.5 / SDXL / Z-Image / FLUX
  without quantization. Captioning uses the transformers Qwen2.5-VL-7B
  fallback (MLX is Apple-Silicon-only).
- **On non-Apple platforms**, uninstall the non-functional MLX stub wheels (they
  ship without `libmlx.so`): `pip uninstall -y mlx mlx-vlm mlx-lm`. Captioning then
  falls back to the transformers Qwen2.5-VL-7B model.
- First encode/decode downloads several GB of models into `~/.cache/huggingface`.

## Commands

- **Tests** (no models needed, runs in seconds): `pytest`  (or `.venv/bin/python -m pytest`)
  - Single test: `pytest tests/test_color.py::test_match_brightness_moves_toward_target`
  - Only `tests/test_format.py` (schema/round-trip) and `tests/test_color.py` (post-processing)
    run without ML deps; both import nothing heavier than numpy/Pillow. `tests/test_flux_config.py`
    is also ML-free and asserts the `_model_config` contract for all `--model` choices.
- **Lint**: `ruff check .`  (line-length 100, rules E/F/W/I).
- **Encode**: `python encoder.py samples/real.jpg -o out.brainimg [--seed 42]`
- **Decode**: `python decoder.py out.brainimg -o recon.png [--device cpu|mps|cuda] [--quantize] [--model ...]`
  - **AMD CPU target** (the dev box): `--device cpu` fp32, no quantization needed
    (188 GB RAM fits SDXL/Z-Image/FLUX). Add `--model sd15-turbo` / `sdxl-turbo`
    for Hyper-SD 8-step distilled LoRA — measured on `samples/mandril_color.tif`
    (512², same seed, after the ControlNet scale tuning): SD 1.5 turbo
    50.1 s / 9.65 dB PSNR vs ~3 min / 8.70 dB for the 30-step path with the
    old defaults (+0.95 dB — distilled schedule + tuned scales both help),
    SDXL turbo 84.2 s at 512² vs ~17 min for the 30-step path at 512²
    (~12x faster at −0.23 dB). `peft` is required for LoRA loading.
  - Best fidelity: `--device cpu` (fp32). Low-RAM: add `--quantize`.
  - Apple Silicon: `--device mps` uses int8 (fp16 NaNs on MPS). 512x512 OOMs on 8 GB; use 256.
- Helpers: `python scripts/make_sample.py` (synthetic test image), `scripts/make_comparison.py`
  (side-by-side original vs recon).

## Architecture

- `encoder.py` / `decoder.py` are the CLI entrypoints; the `brainimg/` package holds
  `format.py` (schema, ML-free), `device.py` (torch/mlx device + memory helpers),
  `extract.py` (encoder stages), `generate.py` (SD + ControlNet decoder).
- `brainimg/generate.py` has five decoder backends gated by `--model`:
  `sd15` (default, depth+canny+seg ControlNets), `sdxl` (same three at 1024),
  `sd15-turbo` / `sdxl-turbo` (same base + ControlNets + ByteDance Hyper-SD
  8-step distilled LoRA, DDIM trailing schedule), `zimage` (Z-Image-Turbo 6B
  DiT + single Union ControlNet, depth-only), `qwen-image` (Qwen-Image DiT +
  Union ControlNet, depth-only), `hunyuan` / `hunyuan-full` (HunyuanDiT v1.2
  Distilled/full, depth+canny ControlNets), `sana` (NVIDIA SANA 600M linear
  DiT, HED ControlNet fed the canny map — only edge ControlNet available),
  `flux2-klein` (FLUX.2-klein-4B img2img, depth map as starting image — no
  ControlNet, experimental pseudo-ControlNet), `flux-depth` / `flux-canny`
  (FLUX.1 Control variants, channel-concat, one conditioning image), and
  `flux-depth-turbo` / `flux-canny-turbo` (FLUX Control + Hyper-SD 8-step
  FLUX LoRA, strips x_embedder/context_embedder deltas that are
  shape-incompatible with the Control transformer).
  The zimage path lives in `_generate_zimage` / `_build_zimage_pipeline`,
  the qwen-image path in `_generate_qwen_image` / `_build_qwen_image_pipeline`,
  the hunyuan path in `_generate_hunyuan` / `_build_hunyuan_pipeline`,
  the sana path in `_generate_sana` / `_build_sana_pipeline`,
  the flux2-klein path in `_generate_flux2` / `_build_flux2_pipeline`,
  the FLUX path in `_generate_flux` / `_build_flux_pipeline`, and the
  SD path (which also serves `*-turbo` via a `cfg["turbo"]` flag) in
  `_generate_sd` / `_build_pipeline`. Schema is unchanged — zimage/FLUX/
  qwen-image/hunyuan/sana/flux2-klein simply ignore the maps they don't use.
- **Encoder and decoder must stay separate processes** — models are never
  resident together (historical 8 GB Apple Silicon constraint; on the AMD CPU
  target with 188 GB this is less critical but still a clean separation).
  Memory is released between heavy stages via `free_torch()` / `free_mlx()`.
- `brainimg/format.py` is **deliberately free of ML imports** so the format tests run
  without downloading models. torch/mlx/cv2 are imported **lazily inside functions**
  in the other modules — keep that pattern when editing.
- `.brainimg` is a small JSON doc; `format_version` is `"0.1"`. `segmentation_map_b64`
  is optional so older files still decode (the seg ControlNet is added only when present).
- `MAP_SIZE` (currently 128) in `format.py` controls conditioning-map resolution;
  changing it requires re-encoding existing samples. See TODO.md tier 2.
- Deterministic given the seed: re-decoding with the same seed reproduces the image exactly.

## Gotchas

- **MPS fp16 NaN bug** (Apple Silicon only; not relevant on the AMD CPU target):
  SD 1.5 fp16 matmuls produce NaNs (black frame) on Apple Silicon. The decoder
  int8-quantizes UNet + ControlNets via `optimum.quanto`; the VAE runs fp32.
  Don't "fix" this by switching MPS to fp16. **Z-Image / FLUX are exempt**: they
  run bf16 everywhere (their native dtype) and are not int8-quantized; the NaN
  bug is an SD 1.5 fp16-specific issue, not a general bf16 issue on MPS.
- The decoder swaps the stock SD 1.5 VAE for `sd-vae-ft-mse` and post-processes
  brightness/saturation to match stored targets — these are intentional quality fixes,
  not noise. The color-style prefix is prepended to the caption **only** if it fits the
  CLIP 77-token limit (SD 1.5/SDXL/turbo); for Z-Image/FLUX the Qwen / T5 encoder's
  512-token limit is large enough that the prefix is prepended unconditionally.
- Captioner can misidentify scenes (e.g. dark hair read as a hat); conditioning maps
  (depth/canny/seg) drive fidelity, so a wrong caption is lower-impact than it looks.
- **Z-Image is depth-only**: `--model zimage` feeds only the depth map to the Union
  ControlNet. The blueprint's canny and seg maps are ignored (no schema change).
  `optimum.quanto` is not used on this path; on `mps` `enable_model_cpu_offload()`
  handles memory, but on a **CPU-only** box the whole bf16 pipeline is kept
  resident in host RAM (~18 GB) -- `enable_model_cpu_offload` raises
  `RuntimeError` without an accelerator to offload *to*, so the zimage CPU path
  calls `pipe.to("cpu")` directly. 8 GB Apple Silicon is not supported for
  Z-Image -- use `sd15`.
- **Hyper-SD turbo LoRA loading**: `sd15-turbo` / `sdxl-turbo` load the
  `ByteDance/Hyper-SD` LoRA via `pipe.load_lora_weights` + `pipe.fuse_lora(0.125)`
  inside `_build_pipeline`, then swap the scheduler to `DDIMScheduler(timestep_spacing="trailing")`.
  The LoRA must be fused *after* device placement. Turbo paths ignore the file's
  stored step count (tuned for 20-30 step SD) and use 8 steps unless `--steps` is
  passed. The `--cfg` default stays at 7.0/7.5 (CFG-preserved LoRAs support 5-8).
- **Hyper-SD FLUX turbo LoRA loading**: `flux-depth-turbo` / `flux-canny-turbo`
  load the `Hyper-FLUX.1-dev-8steps-lora.safetensors` LoRA inside
  `_build_flux_pipeline`. The LoRA was trained on base `FLUX.1-dev`, not the
  Control variants -- the Control transformer's `x_embedder` (extra input
  channels for the control image, 128 vs 64) and `context_embedder` (doesn't
  exist on base dev) are shape-incompatible, so those LoRA deltas are stripped
  before loading. The `transformer.` prefix is also stripped (diffusers adds
  it internally when loading from a state dict). No scheduler swap -- FLUX
  uses `FlowMatchEulerDiscreteScheduler` natively and the 8-step LoRA just
  works with fewer steps. Guidance 3.5 (the FLUX dev default, not the 10.0/
  30.0 the non-turbo control variants use).
- **Z-Image ControlNet file choice**: the *lite* 2.1-2601/2602-8steps
  safetensors look attractive (~2 GB) but their widened `control_all_x_embedder`
  (input dim 132 vs diffusers' expected 64) is rejected by
  `ZImageControlNetModel.from_single_file` on diffusers 0.38 with a
  shape-mismatch ValueError. The full `2.1-8steps` file (~6.4 GB, 5 control
  types) loads cleanly and is what the code pins. Re-evaluate if a future
  diffusers release adds the lite configs.
- **HunyuanDiT size handling + CPU bf16 black frame**: by default `--model
  hunyuan` / `hunyuan-full` honors `--size` exactly (diffusers'
  `use_resolution_binning=False`). HunyuanDiT was trained at 1024²; running
  it at 512² is off-distribution (PSNR drops ~1-2 dB) but ~4× faster, which
  is what CPU iteration needs. Pass `--bin-resolution` to let diffusers
  remap to the nearest trained shape (e.g. 512² → 1024²). The non-distilled
  `hunyuan-full` variant can emit a black/NaN frame under bf16 on CPU
  (the distilled path has not been observed to); `_generate_hunyuan`
  detects this and retries once at fp32 (~2× RAM/runtime, numerically
  safe) — the same recovery philosophy as the SD 1.5 MPS fp16-NaN handling,
  applied to CPU bf16.
- **SANA HED/canny mismatch**: `--model sana` uses NVIDIA's SANA 600M (MIT,
  linear DiT) with an HED (soft-edge) ControlNet — the only ControlNet type
  available for SANA. The blueprint's canny map is fed to the HED ControlNet
  (both are edge maps, but HED produces soft probability edges while canny
  produces hard binary edges). This type mismatch creates a PSNR-vs-color
  trade-off: scale=0.5 gives the best PSNR (10.20 dB) but collapses the
  blue/purple band (20% vs source 53%); scale=1.0 preserves color (54% blue)
  but gives the worst PSNR (8.69 dB). The default 0.4 is the visually best
  compromise (9.91 dB, 16% blue). SANA is the fastest 1024-native backend
  (52 s at 1024², 20 steps, ~5 GB RAM) but the lowest-PSNR backend due to
  the mismatch.
  Depth and seg maps are ignored (no depth/seg ControlNet exists for SANA).
- **FLUX.2-klein img2img pseudo-ControlNet**: `--model flux2-klein` uses
  FLUX.2-klein-4B (Apache 2.0, ungated, 4B, 4-step distilled) as an
  image-to-image model, feeding the blueprint's depth map as the starting
  image. No ControlNet exists for FLUX.2-klein -- this is an experimental
  pseudo-ControlNet approach. The model "edits" the depth map into a
  photorealistic image matching the caption, rather than being structurally
  constrained by a ControlNet. The img2img approach gives the #2 PSNR
  overall (13.76 dB at 512², after FLUX depth turbo's 14.49 dB) but
  collapses the color palette (15% blue vs source 53%) -- the model
  converts the depth map's grayscale into warm tones regardless of the
  caption. 240 s at 512², 4 steps, ~13 GB RAM. Canny and seg maps are
  ignored (only one image input).

## Conventions

- Ruff: line-length 100, target py312, rules E/F/W/I (import sorting enforced).
- CLI CLIs use argparse with `main(argv=None) -> int` + `raise SystemExit(main())`.
- Docstrings document the "why"; code is comment-light.