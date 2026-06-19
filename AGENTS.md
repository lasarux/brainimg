# AGENTS.md

High-signal notes for OpenCode agents working in this repo. See `README.md`
for the full project description and `TODO.md` for planned decode-quality work.

## Setup

- Python **3.12** (pyproject pins `>=3.12,<3.13`) via [`uv`](https://github.com/astral-sh/uv):
  `uv venv -p 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt`
- **On non-Apple platforms**, uninstall the non-functional MLX stub wheels (they
  ship without `libmlx.so`): `pip uninstall -y mlx mlx-vlm mlx-lm`. Captioning then
  falls back to the transformers Qwen2.5-VL-7B model.
- First encode/decode downloads several GB of models into `~/.cache/huggingface`.

## Commands

- **Tests** (no models needed, runs in seconds): `pytest`  (or `.venv/bin/python -m pytest`)
  - Single test: `pytest tests/test_color.py::test_match_brightness_moves_toward_target`
  - Only `tests/test_format.py` (schema/round-trip) and `tests/test_color.py` (post-processing)
    run without ML deps; both import nothing heavier than numpy/Pillow.
- **Lint**: `ruff check .`  (line-length 100, rules E/F/W/I). Note: pre-existing errors
  exist on committed macOS AppleDouble files `brainimg/._*.py` — see "Gotchas" below.
- **Encode**: `python encoder.py samples/real.jpg -o out.brainimg [--seed 42]`
- **Decode**: `python decoder.py out.brainimg -o recon.png [--device cpu|mps|cuda] [--quantize]`
  - Best fidelity: `--device cpu` (fp32, ~10 GB RAM, slow). Low-RAM: add `--quantize`.
  - Apple Silicon: `--device mps` uses int8 (fp16 NaNs on MPS). 512x512 OOMs on 8 GB; use 256.
- Helpers: `python scripts/make_sample.py` (synthetic test image), `scripts/make_comparison.py`
  (side-by-side original vs recon).

## Architecture

- `encoder.py` / `decoder.py` are the CLI entrypoints; the `brainimg/` package holds
  `format.py` (schema, ML-free), `device.py` (torch/mlx device + memory helpers),
  `extract.py` (encoder stages), `generate.py` (SD + ControlNet decoder).
- `brainimg/generate.py` has three decoder backends gated by `--model`:
  `sd15` (default, depth+canny+seg ControlNets), `sdxl` (same three at 1024),
  and `zimage` (Z-Image-Turbo 6B DiT + single Union ControlNet, depth-only).
  The zimage path lives in `_generate_zimage` / `_build_zimage_pipeline`; the
  SD path in `_generate_sd` / `_build_pipeline`. Schema is unchanged — zimage
  simply ignores the canny/seg maps in the blueprint.
- **Encoder and decoder must stay separate processes** — models are never resident
  together (8 GB Apple Silicon constraint). Memory is released between heavy stages
  via `free_torch()` / `free_mlx()`.
- `brainimg/format.py` is **deliberately free of ML imports** so the format tests run
  without downloading models. torch/mlx/cv2 are imported **lazily inside functions**
  in the other modules — keep that pattern when editing.
- `.brainimg` is a small JSON doc; `format_version` is `"0.1"`. `segmentation_map_b64`
  is optional so older files still decode (the seg ControlNet is added only when present).
- `MAP_SIZE` (currently 128) in `format.py` controls conditioning-map resolution;
  changing it requires re-encoding existing samples. See TODO.md tier 2.
- Deterministic given the seed: re-decoding with the same seed reproduces the image exactly.

## Gotchas

- **Committed AppleDouble files**: `brainimg/._*.py`, `brainimg/.___init__.py`,
  `.___pycache__` are macOS resource-fork junk that were committed by mistake. They are
  not real source and cause `ruff` errors. Do not edit them; ideally remove + gitignore.
- **MPS fp16 NaN bug**: SD 1.5 fp16 matmuls produce NaNs (black frame) on Apple Silicon.
  The decoder int8-quantizes UNet + ControlNets via `optimum.quanto`; the VAE runs fp32.
  Don't "fix" this by switching MPS to fp16. **Z-Image is exempt**: it runs bf16
  everywhere (its native dtype) and is not int8-quantized; the NaN bug is an SD 1.5
  fp16-specific issue, not a general bf16 issue on MPS.
- The decoder swaps the stock SD 1.5 VAE for `sd-vae-ft-mse` and post-processes
  brightness/saturation to match stored targets — these are intentional quality fixes,
  not noise. The color-style prefix is prepended to the caption **only** if it fits the
  CLIP 77-token limit (SD 1.5/SDXL); for Z-Image the Qwen encoder's 512-token limit is
  large enough that the prefix is prepended unconditionally.
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
- **Z-Image ControlNet file choice**: the *lite* 2.1-2601/2602-8steps
  safetensors look attractive (~2 GB) but their widened `control_all_x_embedder`
  (input dim 132 vs diffusers' expected 64) is rejected by
  `ZImageControlNetModel.from_single_file` on diffusers 0.38 with a
  shape-mismatch ValueError. The full `2.1-8steps` file (~6.4 GB, 5 control
  types) loads cleanly and is what the code pins. Re-evaluate if a future
  diffusers release adds the lite configs.

## Conventions

- Ruff: line-length 100, target py312, rules E/F/W/I (import sorting enforced).
- CLI CLIs use argparse with `main(argv=None) -> int` + `raise SystemExit(main())`.
- Docstrings document the "why"; code is comment-light.