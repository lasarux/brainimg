"""The decoder: regenerate an image from a .brainimg blueprint.

Loads Stable Diffusion 1.5 (default), SDXL, or Z-Image-Turbo and re-paints the
scene described by the blueprint.

SD 1.5 / SDXL stack (two or three ControlNets):
  * depth (always)        -- SD 1.5: lllyasviel/control_v11f1p_sd15_depth
                             SDXL:   diffusers/controlnet-depth-sdxl-1.0
  * canny (always)        -- SD 1.5: lllyasviel/control_v11p_sd15_canny
                             SDXL:   diffusers/controlnet-canny-sdxl-1.0
  * segmentation (opt.)   -- SD 1.5: lllyasviel/control_v11p_sd15_seg
                             SDXL:   abovzv/sdxl_segmentation_controlnet_ade20k
      used only when the blueprint carries an ADE20K segmentation map (newer
      v0.1 files). Older files without it decode exactly as before.

Z-Image-Turbo stack (single Union ControlNet, depth-only):
  * depth (only)          -- alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
                            (2.1-lite-2602-8steps safetensor, distilled for the
                            8-step Turbo schedule). The Union net accepts one
                            conditioning image per call (one of canny/depth/
                            pose/mlsd/hed/scribble/gray); we feed depth because
                            it carries the most structure. The blueprint's
                            canny and segmentation maps are *ignored* on this
                            path (no schema change -- ``segmentation_map_b64``
                            stays optional). bf16 throughout (Z-Image is a 6B
                            DiT; bf16 sidesteps the MPS fp16 NaN bug entirely).

Quality post-processing:
  * The stock SD 1.5 VAE is replaced with the fine-tuned ``sd-vae-ft-mse``
    (cleaner decode, better skin/color).
  * After generation, brightness/saturation are matched to the targets stored
    in the blueprint (SD 1.5 tends to over-brighten/over-saturate). A no-op
    for older files with no stored stats; harmless on Z-Image.
  * A stored color style prefix ("dark, low-key lighting, red dominant
    tones", ...) is prepended to the caption. For SD 1.5/SDXL it is prepended
    only when the combined length fits the CLIP 77-token limit; for Z-Image
    the Qwen text encoder's 512-token limit is large enough to prepend it
    unconditionally.

Device strategy (SD 1.5 / SDXL):
  * ``cpu``  : full fp32, no quantization. Best fidelity, slow (minutes/image).
    Needs ~10 GB RAM. Pass ``--quantize`` to int8-quantize the weights and
    fit in ~5 GB at a small quality cost.
  * ``mps``  (Apple Silicon default): SD 1.5 fp16 produces NaNs on MPS, so
    the UNet and ControlNets are int8-quantized (weights + activations) to
    avoid fp16 matmuls. Fits in 8 GB.
  * ``cuda`` (NVIDIA): fp16, works correctly. Fast and high fidelity.

Z-Image-Turbo device strategy:
  * bf16 everywhere (native dtype). No int8 quantization on this path.
  * ``cuda``: ``pipe.to("cuda")``. Needs ~16 GB VRAM (6B DiT + 2 GB lite
    ControlNet). Fast (~8 steps).
  * ``mps`` / ``cpu``: ``enable_model_cpu_offload()`` so layers move between
    host and device. Slower but avoids OOM. The 8 GB Apple Silicon target is
    not supported for Z-Image -- use ``--model sd15`` there.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .device import free_torch, get_dtype, get_torch_device
from .extract import b64_to_image, brightness_saturation_of
from .format import BrainimgData, load_brainimg

# --- model stacks ----------------------------------------------------------- #
# Two base models are supported, gated by the `model` arg ("sd15" default,
# "sdxl" opt-in). Each carries its own base + VAE + ControlNet ids and its own
# sensible defaults (SDXL trains at 1024 and uses different scale ranges).
SD15_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD15_VAE_ID = "stabilityai/sd-vae-ft-mse"
SD15_CONTROLNET_DEPTH_ID = "lllyasviel/control_v11f1p_sd15_depth"
SD15_CONTROLNET_CANNY_ID = "lllyasviel/control_v11p_sd15_canny"
SD15_CONTROLNET_SEG_ID = "lllyasviel/control_v11p_sd15_seg"

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
SDXL_CONTROLNET_DEPTH_ID = "diffusers/controlnet-depth-sdxl-1.0"
SDXL_CONTROLNET_CANNY_ID = "diffusers/controlnet-canny-sdxl-1.0"
# An SDXL ADE20K seg ControlNet now exists (the xinsir one returned 401 at the
# time TODO.md was written; abovzv's is ungated and safetensors-only).
SDXL_CONTROLNET_SEG_ID = "abovzv/sdxl_segmentation_controlnet_ade20k"

DEFAULT_MODEL = "sd15"

# Per-model generation defaults. ControlNet scales are high: structural
# fidelity to the original is the whole point. SDXL Conditioning scales
# tend to run a bit lower than SD 1.5 for comparable visual grip.
#
# SD 1.5 defaults below were tuned via a grid sweep on samples/lenna.tiff
# and samples/test512.jpg at 512x512 with sd15-turbo (8 steps, seed from
# the blueprint). The old defaults (1.5/1.2/0.9/7.5) were set for the
# Depth-Anything-Small + no-seg pipeline; with Depth-Anything-V2-Base +
# the ADE20K seg ControlNet, lower depth + lower canny + seg at parity
# beats the old stack on both samples -- the V2 depth map is sharper so
# it over-constrains at 1.5, and the seg map adds material cues that
# were missing before. See scripts/sweep_lenna.py for the sweep grid.
SD15_CONTROLNET_DEPTH_SCALE = 0.8
SD15_CONTROLNET_CANNY_SCALE = 1.0
SD15_CONTROLNET_SEG_SCALE = 1.0
SD15_GUIDANCE_SCALE = 7.5
# Default generation side length. 512 is reasonable on a machine with plenty
# of RAM; 256 is the safe ceiling on an 8 GB Apple Silicon Mac with int8.
SD15_MAX_DEFAULT_SIDE = 512

SDXL_CONTROLNET_DEPTH_SCALE = 1.0
SDXL_CONTROLNET_CANNY_SCALE = 0.8
SDXL_CONTROLNET_SEG_SCALE = 0.6
SDXL_GUIDANCE_SCALE = 7.0
# SDXL is trained at 1024; smaller sizes hurt quality.
SDXL_MAX_DEFAULT_SIDE = 1024

# --- Z-Image-Turbo stack ---------------------------------------------------- #
# A 6B single-stream DiT (Tongyi-MAI/Z-Image-Turbo) + the Alibaba-PAI Union
# ControlNet. The Union net is a single network that accepts one conditioning
# image per call; the conditioning *type* (canny/depth/pose/mlsd/hed) is
# whatever the preprocessor produced -- we feed depth because it carries the
# most structural information. canny + segmentation from the blueprint are
# *not* used on this path (the Union net has no seg mode and the diffusers
# ZImageControlNetPipeline takes a single ``control_image``).
# bf16 is Z-Image's native dtype and also sidesteps the MPS fp16 NaN bug.
#
# Variant note: the *lite* 2.1-2601/2602-8steps files (3 control blocks, ~2 GB)
# look attractive but their ``control_all_x_embedder`` ships a widened input
# dim (132 vs diffusers' expected 64) -- ``ZImageControlNetModel.from_single_file``
# rejects them with a shape-mismatch ValueError on diffusers 0.38. The full
# 2.1-8steps file (15 control blocks, ~6.4 GB) loads cleanly and is the
# 8-step-distilled variant that preserves Turbo's sub-second latency. v1.0
# (5 control blocks, ~3.1 GB) also loads but is not distilled -- it needs
# ~20-40 steps and is much slower.
ZIMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_CONTROLNET_REPO = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"
ZIMAGE_CONTROLNET_FILE = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors"
# Union ControlNet scale sits in the 0.65-1.0 band per the model card; depth is
# the dominant structural cue so we lean toward the upper end.
ZIMAGE_CONTROLNET_DEPTH_SCALE = 0.85
# Turbo is distilled for guidance_scale == 0.0 (no CFG). The user can override
# via --cfg but the result is untested and likely worse.
ZIMAGE_GUIDANCE_SCALE = 0.0
# 9 ``num_inference_steps`` -> 8 DiT forward passes (Turbo schedule).
ZIMAGE_DEFAULT_STEPS = 9
# Z-Image is trained at 1024. Generate at the original's aspect ratio capped to
# a 1024 long side, rounded to 8 (VAE requirement).
ZIMAGE_MAX_DEFAULT_SIDE = 1024
# Qwen text-encoder max prompt length. Much larger than CLIP's 77, so the
# color_style prefix is prepended unconditionally on this path.
ZIMAGE_MAX_TOKENS = 512

# --- FLUX stack ----------------------------------------------------------- #
# Black Forest Labs' FLUX.1 guidance-distilled variants, consumed via diffusers'
# ``FluxControlPipeline``. Unlike SD 1.5/SDXL's separate ControlNet models,
# FLUX's "control" variants bake the conditioning into the transformer via
# channel-wise concatenation of the control image -- one conditioning image
# per call, no seg mode. Same architectural pattern as Z-Image's Union net.
#
# Two variants are wired in:
#   * ``flux-depth``: ``FLUX.1-Depth-dev`` + the blueprint's ``depth_map_b64``.
#     Strongest structural grip; mirrors Z-Image's default.
#   * ``flux-canny``: ``FLUX.1-Canny-dev`` + ``canny_map_b64``. Edge-faithful;
#     best for line-art / architectural subjects where edges dominate.
#
# The non-control base ``FLUX.1-dev`` is *not* used (brainimg always
# conditions on structure). ``FLUX.1-schnell`` is text-only + 4-step and has
# no control variant.
#
# Memory: T5-XXL (~9.5 GB bf16) + transformer (~12 GB bf16) + CLIP-L (~0.5 GB)
# + VAE (~0.2 GB) ~= 22 GB resident. ``--quantize`` FP8-quantizes the
# transformer and T5 (the big two) via ``optimum.quanto`` to drop the resident
# set to ~12 GB at a small quality cost. 8 GB Apple Silicon is not supported
# (consistent with Z-Image).
FLUX_DEPTH_MODEL_ID = "black-forest-labs/FLUX.1-Depth-dev"
FLUX_CANNY_MODEL_ID = "black-forest-labs/FLUX.1-Canny-dev"
# Guidance-distilled defaults per the model cards. The canny variant wants a
# higher CFG than depth (FLUX dev notes / community recipes).
FLUX_DEPTH_GUIDANCE_SCALE = 10.0
FLUX_CANNY_GUIDANCE_SCALE = 30.0
# ~30 steps is the "good quality / reasonable time" point on guidance-distilled
# FLUX (the docs show 28-50; 30 is a fair default).
FLUX_DEFAULT_STEPS = 30
# Note: there is no per-call ``controlnet_conditioning_scale`` for FLUX.
# The FLUX.1-Depth-dev / -Canny-dev checkpoints bake the conditioning
# strength into the channel-concat weights -- diffusers' FluxControlPipeline
# takes a ``control_image`` but no scale kwarg. ``--depth-scale`` /
# ``--canny-scale`` are silently ignored on FLUX paths for that reason.
# Trained at 1024. ``compute_target_size`` will cap to this on the long side.
FLUX_MAX_DEFAULT_SIDE = 1024
# T5's max sequence length. The T5-XXL encoder takes 256 normally and 512 for
# the longer prompts; we let it use the full 512.
FLUX_MAX_TOKENS = 512

# --- Hyper-SD turbo distillation stack --------------------------------------- #
# ByteDance's Hyper-SD (arXiv 2404.13686) distilled the SD 1.5 and SDXL base
# models down to 1-8 steps via trajectory-segmented consistency distillation,
# shipping them as small (~70-150 MB) LoRAs that work with the stock ControlNets
# we already load. Two scheduler families are paired with the LoRAs:
#   * ``DDIMScheduler.from_config(..., timestep_spacing="trailing")`` for the
#     2/4/8-step CFG LoRAs (guidance_scale 0 -- CFG is folded in by distillation).
#   * ``TCDScheduler`` for the 1-step unified LoRA (guidance_scale 0, ``eta``
#     tunes detail vs. cleanliness; higher eta -> more detail, see model card).
#
# We pin the 8-step CFG-preserved LoRAs as the default -- the card recommends
# them as the standard config (good quality/speed trade-off, supports the same
# 5-8 guidance scales the user might want via --cfg). 1/2/4-step LoRAs exist on
# the same repo (``Hyper-SD15-1step-lora.safetensors`` etc.) and can be wired
# in by lowering ``*_TURBO_DEFAULT_STEPS`` + swapping the file name.
HYPER_SD_REPO = "ByteDance/Hyper-SD"
SD15_TURBO_LORA_FILE = "Hyper-SD15-8steps-CFG-lora.safetensors"
SDXL_TURBO_LORA_FILE = "Hyper-SDXL-8steps-CFG-lora.safetensors"
# LoRA scale per the model card (0.125 is the recommended fuse scale for both
# SD 1.5 and SDXL; larger scales over-apply the distillation and hurt quality).
HYPER_SD_LORA_SCALE = 0.125
# 8-step CFG-preserved LoRAs support guidance 5-8 per the card; default matches
# the non-turbo SDXL default of 7.0 (SD 1.5 turbo defaults to 7.5 like the base).
SD15_TURBO_GUIDANCE_SCALE = SD15_GUIDANCE_SCALE
SDXL_TURBO_GUIDANCE_SCALE = SDXL_GUIDANCE_SCALE
SD15_TURBO_DEFAULT_STEPS = 8
SDXL_TURBO_DEFAULT_STEPS = 8

# Backwards-compat single-name constants (callers below use the helpers).
SD_MODEL_ID = SD15_MODEL_ID
VAE_MODEL_ID = SD15_VAE_ID
CONTROLNET_DEPTH_ID = SD15_CONTROLNET_DEPTH_ID
CONTROLNET_CANNY_ID = SD15_CONTROLNET_CANNY_ID
CONTROLNET_SEG_ID = SD15_CONTROLNET_SEG_ID
CONTROLNET_DEPTH_SCALE = SD15_CONTROLNET_DEPTH_SCALE
CONTROLNET_CANNY_SCALE = SD15_CONTROLNET_CANNY_SCALE
CONTROLNET_SEG_SCALE = SD15_CONTROLNET_SEG_SCALE
GUIDANCE_SCALE = SD15_GUIDANCE_SCALE
MAX_DEFAULT_SIDE = SD15_MAX_DEFAULT_SIDE

# CLIP token limit (both SD 1.5 and SDXL use the same CLIP text encoders' 77
# token cap). The style prefix is only prepended to the prompt if the combined
# length fits, otherwise the caption alone is used so it is never truncated.
CLIP_MAX_TOKENS = 77


def _match_color_statistics(
    image: Image.Image,
    target_brightness: float,
    target_saturation: float,
) -> Image.Image:
    """Post-process *image* so its brightness/saturation match the targets.

    SD 1.5 systematically over-brightens and over-saturates; the blueprint
    stores the original image's stats so the decoder can correct this.

      * saturation: scaled in HSV space on the S channel by
        ``target/current``, iterated 1-2x to converge (HSV-S is not linear in
        our custom saturation metric). Ratios are clamped to [0.5, 2.0] to
        avoid clipping artefacts.
      * brightness: applied as a uniform RGB gain ``target/current`` *after*
        saturation. A uniform gain preserves color balance and leaves the
        saturation metric (a ratio of channels) invariant, so correcting
        brightness last does not undo the saturation work. (The reverse order
        would let the HSV-S scaling disturb the brightness.)

    Returns the original image unchanged if the targets are 0.0 (older v0.1
    files without stored stats) or the image stats are already within ~2%.
    """
    import numpy as np

    if target_brightness <= 0 and target_saturation <= 0:
        return image

    out = image.convert("RGB")
    cur_b, cur_s = brightness_saturation_of(out)
    if cur_b <= 0 and cur_s <= 0:
        return out

    # --- saturation first: HSV-S scaling, iterated to converge ---
    # This also perturbs brightness; the brightness step below corrects that.
    if target_saturation > 0 and cur_s > 0 and abs(cur_s - target_saturation) > 2.0:
        for _ in range(2):
            _, cur_s = brightness_saturation_of(out)
            if cur_s <= 0 or abs(cur_s - target_saturation) <= 2.0:
                break
            ratio = max(0.5, min(2.0, target_saturation / cur_s))
            hsv = np.array(out.convert("HSV"), dtype=np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * ratio, 0, 255)
            out = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")

    # --- brightness last: uniform gain (preserves color balance & saturation) ---
    if target_brightness > 0:
        cur_b, _ = brightness_saturation_of(out)
        if cur_b > 0 and abs(cur_b - target_brightness) > 2.0:
            gain = max(0.5, min(2.0, target_brightness / cur_b))
            arr = np.array(out, dtype=np.float32) * gain
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            out = Image.fromarray(arr, "RGB")

    return out


def _model_config(model: str) -> dict:
    """Return the per-model stack config (ids + defaults) for *model*.

    ``model`` is "sd15", "sdxl", "zimage", "flux-depth", or "flux-canny".
    Centralized so callers don't sprinkle conditionals; the SD 1.5 path
    keeps the historical defaults.

    Structural notes:
      * SD 1.5 / SDXL use separate ControlNet models (depth + canny + optional
        seg). Three nets on the SD stack.
      * Z-Image uses a single Union ControlNet fed one depth image (no canny,
        no seg) and a Qwen text encoder (512 tokens vs CLIP 77).
      * FLUX's ``flux-depth`` and ``flux-canny`` use diffusers'
        ``FluxControlPipeline`` which concatenates the conditioning image
        into the transformer channels (NOT a separate ControlNet model).
        One conditioning image per call -- depth OR canny, not both.
    """
    if model == "zimage":
        return {
            "base_id": ZIMAGE_MODEL_ID,
            "controlnet_repo": ZIMAGE_CONTROLNET_REPO,
            "controlnet_file": ZIMAGE_CONTROLNET_FILE,
            # No separate depth/canny/seg ids: the Union net is a single file.
            "depth_scale": ZIMAGE_CONTROLNET_DEPTH_SCALE,
            "canny_scale": None,
            "seg_scale": None,
            "guidance": ZIMAGE_GUIDANCE_SCALE,
            "max_side": ZIMAGE_MAX_DEFAULT_SIDE,
            "default_steps": ZIMAGE_DEFAULT_STEPS,
            "max_tokens": ZIMAGE_MAX_TOKENS,
            "turbo": False,
        }
    if model == "sdxl":
        return {
            "base_id": SDXL_MODEL_ID,
            "vae_id": SDXL_VAE_ID,
            "depth_id": SDXL_CONTROLNET_DEPTH_ID,
            "canny_id": SDXL_CONTROLNET_CANNY_ID,
            "seg_id": SDXL_CONTROLNET_SEG_ID,
            "depth_scale": SDXL_CONTROLNET_DEPTH_SCALE,
            "canny_scale": SDXL_CONTROLNET_CANNY_SCALE,
            "seg_scale": SDXL_CONTROLNET_SEG_SCALE,
            "guidance": SDXL_GUIDANCE_SCALE,
        "max_side": SDXL_MAX_DEFAULT_SIDE,
        "vae_fp16_variant": None,
        "seg_fp16_variant": None,
        # abovzv's seg net uses a non-standard weight filename; diffusers
        # auto-looks for diffusion_pytorch_model.safetensors, so name it.
        "seg_weight_name": "sdxl_segmentation_ade20k_controlnet.safetensors",
        "turbo": False,
    }
    if model == "sdxl-turbo":
        # Hyper-SD XL distilled stack: same SDXL base + VAE + ControlNets as
        # "sdxl", plus a tiny LoRA loaded in _build_pipeline that drops generation
        # to 8 steps. guidance 7.0 (CFG-preserved LoRA), DDIM trailing schedule.
        return {
            "base_id": SDXL_MODEL_ID,
            "vae_id": SDXL_VAE_ID,
            "depth_id": SDXL_CONTROLNET_DEPTH_ID,
            "canny_id": SDXL_CONTROLNET_CANNY_ID,
            "seg_id": SDXL_CONTROLNET_SEG_ID,
            "depth_scale": SDXL_CONTROLNET_DEPTH_SCALE,
            "canny_scale": SDXL_CONTROLNET_CANNY_SCALE,
            "seg_scale": SDXL_CONTROLNET_SEG_SCALE,
            "guidance": SDXL_TURBO_GUIDANCE_SCALE,
            "max_side": SDXL_MAX_DEFAULT_SIDE,
            "vae_fp16_variant": None,
            "seg_fp16_variant": None,
            "seg_weight_name": "sdxl_segmentation_ade20k_controlnet.safetensors",
            "turbo": True,
            "turbo_lora_repo": HYPER_SD_REPO,
            "turbo_lora_file": SDXL_TURBO_LORA_FILE,
            "turbo_lora_scale": HYPER_SD_LORA_SCALE,
            "default_steps": SDXL_TURBO_DEFAULT_STEPS,
        }
    if model == "sd15-turbo":
        # Hyper-SD 1.5 distilled stack: SD 1.5 base + sd-vae-ft-mse + the
        # existing depth/canny/seg ControlNets, plus the 8-step CFG-preserved
        # LoRA. Same defaults as "sd15" except for steps + the turbo flag.
        return {
            "base_id": SD15_MODEL_ID,
            "vae_id": SD15_VAE_ID,
            "depth_id": SD15_CONTROLNET_DEPTH_ID,
            "canny_id": SD15_CONTROLNET_CANNY_ID,
            "seg_id": SD15_CONTROLNET_SEG_ID,
            "depth_scale": SD15_CONTROLNET_DEPTH_SCALE,
            "canny_scale": SD15_CONTROLNET_CANNY_SCALE,
            "seg_scale": SD15_CONTROLNET_SEG_SCALE,
            "guidance": SD15_TURBO_GUIDANCE_SCALE,
            "max_side": SD15_MAX_DEFAULT_SIDE,
            "vae_fp16_variant": None,
            "seg_fp16_variant": "fp16",
            "seg_weight_name": None,
            "turbo": True,
            "turbo_lora_repo": HYPER_SD_REPO,
            "turbo_lora_file": SD15_TURBO_LORA_FILE,
            "turbo_lora_scale": HYPER_SD_LORA_SCALE,
            "default_steps": SD15_TURBO_DEFAULT_STEPS,
        }
    if model in ("flux-depth", "flux-canny"):
        # Single-control stack: pick which conditioning map to feed and which
        # base model (Depth vs Canny variant). The two FLUX.1-* Control
        # checkpoints differ in what conditioning they were trained against,
        # so the choice of base is the choice of conditioning type. There
        # is no per-call control scale -- it's baked into the checkpoint
        # (channel-concat conditioning); see FLUX_DEFAULT_STEPS comment.
        if model == "flux-depth":
            return {
                "base_id": FLUX_DEPTH_MODEL_ID,
                "kind": "flux",
                "control_source": "depth",   # which b64 field on BrainimgData
                "guidance": FLUX_DEPTH_GUIDANCE_SCALE,
                "max_side": FLUX_MAX_DEFAULT_SIDE,
                "default_steps": FLUX_DEFAULT_STEPS,
                "max_tokens": FLUX_MAX_TOKENS,
                "turbo": False,
            }
        return {
            "base_id": FLUX_CANNY_MODEL_ID,
            "kind": "flux",
            "control_source": "canny",
            "guidance": FLUX_CANNY_GUIDANCE_SCALE,
            "max_side": FLUX_MAX_DEFAULT_SIDE,
            "default_steps": FLUX_DEFAULT_STEPS,
            "max_tokens": FLUX_MAX_TOKENS,
            "turbo": False,
        }
    return {
        "base_id": SD15_MODEL_ID,
        "vae_id": SD15_VAE_ID,
        "depth_id": SD15_CONTROLNET_DEPTH_ID,
        "canny_id": SD15_CONTROLNET_CANNY_ID,
        "seg_id": SD15_CONTROLNET_SEG_ID,
        "depth_scale": SD15_CONTROLNET_DEPTH_SCALE,
        "canny_scale": SD15_CONTROLNET_CANNY_SCALE,
        "seg_scale": SD15_CONTROLNET_SEG_SCALE,
        "guidance": SD15_GUIDANCE_SCALE,
        "max_side": SD15_MAX_DEFAULT_SIDE,
        "vae_fp16_variant": None,
        "seg_fp16_variant": "fp16",
        "seg_weight_name": None,
        "turbo": False,
    }


def compute_target_size(
    orig_w: int, orig_h: int, override: str | None = None, max_side: int = MAX_DEFAULT_SIDE
) -> tuple[int, int]:
    """Pick a generation size, rounded to a multiple of 8 (SD requirement)."""
    if override:
        try:
            ws, hs = override.lower().split("x")
            w, h = int(ws), int(hs)
        except ValueError as exc:
            raise ValueError(f"--size expects WxH, got {override!r}") from exc
    else:
        scale = max_side / max(orig_w, orig_h)
        w, h = int(round(orig_w * scale)), int(round(orig_h * scale))

    w = max(8, (w // 8) * 8)
    h = max(8, (h // 8) * 8)
    return w, h


def _load_conditioning_maps(
    data: BrainimgData, target_w: int, target_h: int
) -> list[Image.Image]:
    """Decode the conditioning maps and upscale them to the target size.

    Always returns depth + Canny. If the blueprint carries a segmentation map
    (newer v0.1 files), it is appended as a third map for the seg ControlNet.
    """
    depth = b64_to_image(data.depth_map_b64).convert("RGB").resize(
        (target_w, target_h), Image.LANCZOS
    )
    canny = b64_to_image(data.canny_map_b64).convert("RGB").resize(
        (target_w, target_h), Image.NEAREST  # keep edges crisp
    )
    maps = [depth, canny]
    if getattr(data, "segmentation_map_b64", ""):
        seg = b64_to_image(data.segmentation_map_b64).convert("RGB").resize(
            (target_w, target_h), Image.NEAREST  # palette colors must stay crisp
        )
        maps.append(seg)
    return maps


def _build_pipeline(
    device: str,
    dtype,
    quantize: bool = False,
    with_seg: bool = False,
    model: str = DEFAULT_MODEL,
):
    """Construct the base + ControlNet pipeline for the chosen model.

    Args:
        device: "cuda", "mps", or "cpu".
        dtype: torch dtype for the device (fp16 on GPU, fp32 on CPU).
        quantize: if True, int8-quantize weights to save memory (useful on
            low-RAM machines). On MPS this is always done regardless.
        with_seg: if True, add the ADE20K segmentation ControlNet (3rd net)
            on top of depth + Canny.
        model: "sd15" (default) or "sdxl". SDXL is trained at 1024 and uses
            fp16-variant ControlNets from diffusers + a non-variant seg net.
    """
    import torch
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        StableDiffusionXLControlNetPipeline,
        UniPCMultistepScheduler,
    )

    cfg = _model_config(model)
    if model in ("sdxl", "sdxl-turbo"):
        pipe_cls = StableDiffusionXLControlNetPipeline
    else:
        # sd15 + sd15-turbo share the SD 1.5 pipeline class.
        pipe_cls = StableDiffusionControlNetPipeline

    # Always load from the fp16 checkpoint (smaller download) and upcast
    # to fp32 in memory when needed.
    load_dtype = torch.float16
    variant = "fp16"

    cn_load_kwargs = {"torch_dtype": load_dtype, "variant": variant}
    controlnets = [
        ControlNetModel.from_pretrained(cfg["depth_id"], **cn_load_kwargs),
        ControlNetModel.from_pretrained(cfg["canny_id"], **cn_load_kwargs),
    ]
    if with_seg:
        # The SD 1.5 seg net uses the standard diffusers layout with an fp16
        # variant. The SDXL seg net (abovzv) ships a single checkpoint-format
        # safetensors (not a diffusers repo), so it loads via from_single_file.
        if cfg["seg_weight_name"]:
            from huggingface_hub import hf_hub_download

            seg_file = hf_hub_download(cfg["seg_id"], cfg["seg_weight_name"])
            seg_config = hf_hub_download(cfg["seg_id"], "config.json")
            controlnets.append(
                ControlNetModel.from_single_file(
                    seg_file, config=seg_config, torch_dtype=load_dtype
                )
            )
        else:
            seg_kwargs = {"torch_dtype": load_dtype, "variant": cfg["seg_fp16_variant"]}
            controlnets.append(
                ControlNetModel.from_pretrained(cfg["seg_id"], **seg_kwargs)
            )

    pipe = pipe_cls.from_pretrained(
        cfg["base_id"],
        controlnet=controlnets,
        torch_dtype=load_dtype,
        variant=variant,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Swap the stock VAE for a fine-tuned one: cleaner decode, better skin
    # tones and colors, fewer washed-out highlights. SD 1.5 uses sd-vae-ft-mse
    # (fp16 variant); SDXL uses madebyollin's fp16-fix (no variant).
    vae_kwargs = {"torch_dtype": load_dtype}
    if cfg["vae_fp16_variant"]:
        vae_kwargs["variant"] = cfg["vae_fp16_variant"]
    pipe.vae = AutoencoderKL.from_pretrained(cfg["vae_id"], **vae_kwargs)

    # CPU: upcast to fp32 BEFORE moving to device (diffusers refuses fp16 on CPU).
    if device == "cpu":
        pipe = pipe.to(torch.float32)

    pipe = pipe.to(device)

    if cfg.get("turbo"):
        # Hyper-SD distilled LoRA: load + fuse before inference so the
        # UNet weights are permanently adjusted (no per-call LoRA overhead).
        # Per ByteDance's model card: fuse scale 0.125 for both SD 1.5 and
        # SDXL distilled LoRAs, then swap the scheduler to DDIM with trailing
        # timestep spacing -- the distilled schedule uses a different timestep
        # progression than the stock UniPC scheduler.
        from huggingface_hub import hf_hub_download

        lora_path = hf_hub_download(cfg["turbo_lora_repo"], cfg["turbo_lora_file"])
        # diffusers' load_lora_weights accepts a path to a safetensors file.
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=cfg["turbo_lora_scale"])
        # SD 1.5 + SDXL distilled stacks both use the DDIM trailing schedule.
        from diffusers import DDIMScheduler

        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    else:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    if device == "mps":
        # MPS fp16 matmuls produce NaNs -> int8 weights + activations.
        from optimum.quanto import freeze, qint8, quantize

        quantize(pipe.unet, weights=qint8, activations=qint8)
        freeze(pipe.unet)
        for cn in pipe.controlnet.nets:
            quantize(cn, weights=qint8, activations=qint8)
            freeze(cn)

        # VAE in fp32 for a clean decode.
        pipe.vae = pipe.vae.to(torch.float32)
        _orig_decode = pipe.vae.decode

        def _decode_fp32(z, *args, **kwargs):
            return _orig_decode(z.to(torch.float32), *args, **kwargs)

        pipe.vae.decode = _decode_fp32

    elif device == "cpu" and quantize:
        # Optional int8 weights (activations stay fp32) to fit low-RAM machines.
        from optimum.quanto import freeze, qint8, quantize

        quantize(pipe.unet, weights=qint8)
        freeze(pipe.unet)
        for cn in pipe.controlnet.nets:
            quantize(cn, weights=qint8)
            freeze(cn)

    # cuda + fp16: works correctly, no quantization needed.
    # cpu without --quantize: full fp32, best fidelity.

    return pipe, torch


def _build_zimage_pipeline(device: str):
    """Construct the Z-Image-Turbo + Union ControlNet pipeline.

    Distinct from the SD 1.5/SDXL path: a single Union ControlNet (one
    conditioning image, depth), bf16 throughout (no int8 quant -- bf16 is
    Z-Image's native dtype and sidesteps the MPS fp16 NaN bug), and the
    diffusers FlowMatch scheduler that ships with the model.

    Memory strategy:
      * cuda: ``pipe.to("cuda")``. ~18 GB VRAM floor (6B DiT ~12 GB + 6.4 GB
        8-step-distilled Union ControlNet).
      * mps: ``enable_model_cpu_offload()`` streams layers host<->device per
        step. Slower but avoids OOM. The 8 GB Apple Silicon target is *not*
        supported on this path -- use ``--model sd15`` there.
      * cpu: keep the whole pipeline resident in host RAM (``pipe.to("cpu")``).
        ``enable_model_cpu_offload`` is *not* an option here -- it raises
        ``RuntimeError("requires accelerator, but not found")`` on a CPU-only
        box (diffusers' offload hooks need an accelerator to move *to*). So
        Z-Image on CPU needs roughly the model size in RAM: ~12 GB for the bf16
        6B DiT + ~6.4 GB for the 8-step ControlNet. If the host has the RAM it
        runs (slowly); if not, the user must fall back to ``--model sd15``. The
        ``--quantize`` flag is intentionally not honored on this path (Z-Image
        has no tested int8 story here, unlike the SD 1.5 UNet).

    Returns ``(pipe, torch)`` to match the SD 1.5/SDXL builder signature.
    """
    import torch
    from diffusers import ZImageControlNetModel, ZImageControlNetPipeline
    from huggingface_hub import hf_hub_download

    cfg = _model_config("zimage")
    load_dtype = torch.bfloat16

    controlnet = ZImageControlNetModel.from_single_file(
        hf_hub_download(cfg["controlnet_repo"], cfg["controlnet_file"]),
        torch_dtype=load_dtype,
    )
    pipe = ZImageControlNetPipeline.from_pretrained(
        cfg["base_id"],
        controlnet=controlnet,
        torch_dtype=load_dtype,
    )

    if device == "cuda":
        pipe = pipe.to("cuda")
    elif device == "mps":
        # Stream layers host<->MPS per step. bf16 is safe on MPS (no fp16 NaN
        # issue for Z-Image), but VRAM is the limit on small Apple Silicon.
        pipe.enable_model_cpu_offload()
    else:
        # Pure CPU: enable_model_cpu_offload() raises RuntimeError without an
        # accelerator, so just keep everything resident in host RAM. Needs
        # roughly the bf16 model size in RAM (~14 GB for the lite stack).
        pipe = pipe.to("cpu")

    return pipe, torch


def _build_flux_pipeline(device: str, variant: str, quantize: bool = False):
    """Construct the FLUX Control pipeline (flux-depth or flux-canny).

    Distinct from the SD 1.5/SDXL and Z-Image paths:

      * bf16 throughout. FLUX is bf16-native (T5-XXL + a 12B MMDiT
        transformer); bf16 also sidesteps the MPS fp16 NaN bug, same as
        Z-Image. No fp16 anywhere, no int8 weight quant -- ``--quantize``
        instead uses ``optimum.quanto``'s ``qfloat8`` on the two big
        modules to halve memory at a small quality cost.

      * Channel-concat conditioning (NOT a ControlNet model). Diffusers'
        ``FluxControlPipeline`` takes a single ``control_image``; the
        conditioning *type* is baked into which FLUX.1-*-dev checkpoint
        you load (``FLUX.1-Depth-dev`` or ``FLUX.1-Canny-dev``). We pick
        the right base for ``variant`` ("flux-depth" or "flux-canny")
        and feed the matching map from the blueprint. The other map
        (and the seg map, if present) are silently ignored -- no schema
        change.

      * Per-device memory strategy (mirrors Z-Image):
          - cuda: ``pipe.to("cuda")``. ~22 GB VRAM (T5-XXL ~9.5 GB +
            transformer ~12 GB + CLIP-L ~0.5 GB + VAE ~0.2 GB).
          - mps:  ``enable_model_cpu_offload()`` streams layers
            host<->MPS. Slow but avoids OOM. 8 GB Apple Silicon is NOT
            supported -- use ``--model sd15`` there.
          - cpu:  ``pipe.to("cpu")`` keeps the full bf16 pipeline
            resident in host RAM (~22 GB). Same constraint as Z-Image:
            ``enable_model_cpu_offload`` raises ``RuntimeError`` on a
            CPU-only box.

      * ``--quantize`` (FLUX-only on this path): FP8-quantize the
        transformer and T5-XXL via ``optimum.quanto``. Drops resident
        set from ~22 GB to ~12 GB. Activations stay bf16 (FLUX
        quantization recipes in the diffusers docs use weights-only
        ``qfloat8`` -- we follow that). ``freeze()`` after each quantize
        so the calibration graph doesn't get re-run on each forward.

    Returns ``(pipe, torch)`` to match the SD/Z-Image builder signature.
    """
    import torch
    from diffusers import FluxControlPipeline

    cfg = _model_config(variant)
    load_dtype = torch.bfloat16

    pipe = FluxControlPipeline.from_pretrained(
        cfg["base_id"],
        torch_dtype=load_dtype,
    )

    if device == "cuda":
        pipe = pipe.to("cuda")
    elif device == "mps":
        pipe.enable_model_cpu_offload()
    else:
        # CPU-only: keep the full bf16 pipeline resident. No offload trick
        # on a box without an accelerator.
        pipe = pipe.to("cpu")

    if quantize:
        # FLUX is fp8-brittle on activation quant (per the diffusers blog
        # post on Quanto + Flux), so we quantize weights only. T5-XXL is
        # the dominant text encoder (FLUX uses T5-XXL + a small CLIP-L);
        # both are fp8-safe on weights.
        from optimum.quanto import freeze, qfloat8, quantize

        quantize(pipe.transformer, weights=qfloat8)
        freeze(pipe.transformer)
        # ``pipe.text_encoder_2`` is T5-XXL on FLUX; ``pipe.text_encoder``
        # is CLIP-L (~0.5 GB, not worth quantizing).
        if getattr(pipe, "text_encoder_2", None) is not None:
            quantize(pipe.text_encoder_2, weights=qfloat8)
            freeze(pipe.text_encoder_2)

    return pipe, torch


def _build_prompt(data: BrainimgData, tokenizer, max_tokens: int = CLIP_MAX_TOKENS) -> str:
    """Prepend the stored color style prefix to the caption if it fits.

    The style prefix ("dark, low-key lighting, red dominant tones", ...) is
    extracted by the encoder and stored in ``data.extra['color_style']``.
    Prepending it weights the mood first -- the front of the prompt has the
    strongest encoder weight. It is prepended when the combined length fits
    within the model's token limit so the caption itself is never truncated.

    SD 1.5 / SDXL use CLIP (77 tokens); Z-Image uses a Qwen text encoder with a
    512-token limit, where the prefix effectively always fits. Older files
    without a stored color_style use the caption verbatim.
    """
    style = (data.extra or {}).get("color_style", "")
    if not style:
        return data.prompt
    combined = f"{style}, {data.prompt}"
    try:
        n = len(tokenizer(combined, max_length=max_tokens)["input_ids"])
    except Exception:
        return data.prompt
    if n <= max_tokens:
        return combined
    return data.prompt


def generate_image(
    data: BrainimgData,
    size: str | None = None,
    steps: int | None = None,
    device_override: str | None = None,
    quantize: bool = False,
    guidance_scale: float | None = None,
    depth_scale: float | None = None,
    canny_scale: float | None = None,
    seg_scale: float | None = None,
    model: str = DEFAULT_MODEL,
) -> Image.Image:
    """Regenerate a single image from *data*.

    Args:
        device_override: "cpu", "mps", "cuda", or None (auto-detect).
        quantize: int8-quantize weights on SD 1.5/SDXL; FP8-quantize
            transformer + T5 on FLUX. Z-Image ignores it (bf16, no tested
            quant path).
        guidance_scale: override the classifier-free guidance scale.
            Defaults are per-model (SD 1.5: 7.5, SDXL: 7.0, Z-Image: 0.0,
            flux-depth: 10.0, flux-canny: 30.0).
        depth_scale / canny_scale / seg_scale: override the ControlNet
            conditioning scales. Defaults are per-model. For Z-Image only
            depth_scale applies (canny/seg are ignored on that path).
            For FLUX only one scale applies (depth or canny, matching the
            chosen variant).
        model: "sd15" (default), "sdxl", "zimage", "flux-depth", or
            "flux-canny". SDXL/FLUX both trained at 1024. FLUX is bf16 and
            takes a single depth OR canny conditioning image (not both).
    """
    if model == "zimage":
        return _generate_zimage(
            data,
            size=size,
            steps=steps,
            device_override=device_override,
            guidance_scale=guidance_scale,
            depth_scale=depth_scale,
        )
    if model in ("flux-depth", "flux-canny"):
        return _generate_flux(
            data,
            size=size,
            steps=steps,
            device_override=device_override,
            quantize=quantize,
            guidance_scale=guidance_scale,
            depth_scale=depth_scale,
            canny_scale=canny_scale,
            variant=model,
        )
    return _generate_sd(
        data,
        size=size,
        steps=steps,
        device_override=device_override,
        quantize=quantize,
        guidance_scale=guidance_scale,
        depth_scale=depth_scale,
        canny_scale=canny_scale,
        seg_scale=seg_scale,
        model=model,
    )


def _generate_sd(
    data: BrainimgData,
    size: str | None,
    steps: int | None,
    device_override: str | None,
    quantize: bool,
    guidance_scale: float | None,
    depth_scale: float | None,
    canny_scale: float | None,
    seg_scale: float | None,
    model: str,
) -> Image.Image:
    """SD 1.5 / SDXL path: depth + canny (+ optional seg) ControlNets."""
    device = device_override or get_torch_device()
    dtype = get_dtype(device)
    cfg = _model_config(model)

    target_w, target_h = compute_target_size(
        data.original_width, data.original_height, size, max_side=cfg["max_side"]
    )
    # Turbo models ignore the file's stored step count (tuned for the 20-30
    # step SD schedule); they use the distilled LoRA's step count (8) unless
    # the user passes --steps explicitly. Non-turbo SD 1.5/SDXL honor data.steps.
    if cfg.get("turbo"):
        n_steps = steps or cfg["default_steps"]
    else:
        n_steps = steps or data.steps

    has_seg = bool(getattr(data, "segmentation_map_b64", ""))
    conditioning = _load_conditioning_maps(data, target_w, target_h)
    scales = [
        depth_scale if depth_scale is not None else cfg["depth_scale"],
        canny_scale if canny_scale is not None else cfg["canny_scale"],
    ]
    if has_seg:
        scales.append(seg_scale if seg_scale is not None else cfg["seg_scale"])

    pipe, torch = _build_pipeline(
        device, dtype, quantize=quantize, with_seg=has_seg, model=model
    )

    prompt = _build_prompt(data, pipe.tokenizer, max_tokens=CLIP_MAX_TOKENS)

    gen = torch.Generator(device).manual_seed(data.seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=data.negative_prompt,
        image=conditioning,
        controlnet_conditioning_scale=scales,
        guidance_scale=guidance_scale if guidance_scale is not None else cfg["guidance"],
        num_inference_steps=n_steps,
        generator=gen,
    )
    image: Image.Image = result.images[0]

    image = _match_color_statistics(
        image, data.target_brightness, data.target_saturation
    )

    del pipe
    free_torch()
    return image


def _generate_zimage(
    data: BrainimgData,
    size: str | None,
    steps: int | None,
    device_override: str | None,
    guidance_scale: float | None,
    depth_scale: float | None,
) -> Image.Image:
    """Z-Image-Turbo path: single Union ControlNet fed the depth map.

    The blueprint's canny and segmentation maps are *ignored* here -- the
    Union net accepts one conditioning image and has no seg mode. No schema
    change: ``segmentation_map_b64`` stays optional and is simply unused.
    """
    device = device_override or get_torch_device()
    cfg = _model_config("zimage")

    target_w, target_h = compute_target_size(
        data.original_width, data.original_height, size, max_side=cfg["max_side"]
    )
    # Ignore the file's stored ``steps`` (tuned for SD 1.5's 20-30 step range) --
    # Z-Image-Turbo is distilled for 8-9 steps and over-stepping a distilled
    # model hurts quality. Only an explicit ``--steps`` overrides the zimage
    # default.
    n_steps = steps or cfg["default_steps"]

    # Only depth is fed to the Union ControlNet. canny/seg from the blueprint
    # are deliberately unused on this path.
    depth = b64_to_image(data.depth_map_b64).convert("RGB").resize(
        (target_w, target_h), Image.LANCZOS
    )
    scale = depth_scale if depth_scale is not None else cfg["depth_scale"]

    pipe, torch = _build_zimage_pipeline(device)

    prompt = _build_prompt(data, pipe.tokenizer, max_tokens=cfg["max_tokens"])

    # Z-Image Turbo is distilled for guidance_scale == 0.0. The generator
    # lives on the host when cpu-offloaded; pin it to cpu for mps/cpu paths.
    gen_device = "cpu" if device in ("mps", "cpu") else device
    gen = torch.Generator(gen_device).manual_seed(data.seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=data.negative_prompt,
        control_image=depth,
        controlnet_conditioning_scale=scale,
        guidance_scale=guidance_scale if guidance_scale is not None else cfg["guidance"],
        num_inference_steps=n_steps,
        height=target_h,
        width=target_w,
        generator=gen,
    )
    image: Image.Image = result.images[0]

    # Harmless on Z-Image (no-op when targets == 0.0); corrects any color drift.
    image = _match_color_statistics(
        image, data.target_brightness, data.target_saturation
    )

    del pipe
    free_torch()
    return image


def _generate_flux(
    data: BrainimgData,
    size: str | None,
    steps: int | None,
    device_override: str | None,
    quantize: bool,
    guidance_scale: float | None,
    depth_scale: float | None,
    canny_scale: float | None,
    variant: str,
) -> Image.Image:
    """FLUX Control path (flux-depth or flux-canny).

    Feeds the blueprint's depth_map_b64 OR canny_map_b64 to
    ``FluxControlPipeline`` as a single ``control_image``. The blueprint's
    other map and any segmentation_map_b64 are silently ignored -- the
    pipeline takes one image per call, same as Z-Image's Union net.

    Note: ``depth_scale`` / ``canny_scale`` are accepted for signature
    parity with the SD 1.5/SDXL paths but have no effect on FLUX -- the
    conditioning strength is baked into the channel-concat weights, with
    no per-call scale kwarg on diffusers' ``FluxControlPipeline.__call__``.
    """
    device = device_override or get_torch_device()
    cfg = _model_config(variant)

    target_w, target_h = compute_target_size(
        data.original_width, data.original_height, size, max_side=cfg["max_side"]
    )
    n_steps = steps or cfg["default_steps"]

    # Pick which conditioning image to feed based on the variant.
    if cfg["control_source"] == "depth":
        control_b64 = data.depth_map_b64
    elif cfg["control_source"] == "canny":
        control_b64 = data.canny_map_b64
    else:  # pragma: no cover -- defensive, _model_config only emits depth/canny
        raise ValueError(f"unknown FLUX control source: {cfg['control_source']!r}")
    control_image = b64_to_image(control_b64).convert("RGB").resize(
        (target_w, target_h), Image.LANCZOS
    )

    # FLUX's channel-concat control has no per-call scale (the conditioning
    # strength is baked into the trained checkpoint), so ``depth_scale`` /
    # ``canny_scale`` are silently ignored here. The scale constants in
    # FLUX_DEFAULT_STEPS's docstring are documentation-only.

    pipe, torch = _build_flux_pipeline(device, variant, quantize=quantize)

    # FLUX's tokenizer is CLIPTokenizer (text_encoder=CLIP-L) for ``tokenizer``
    # and T5TokenizerFast (text_encoder_2=T5-XXL) for ``tokenizer_2``. The
    # T5 path can take 512 tokens; combined prompt always fits.
    prompt = _build_prompt(data, pipe.tokenizer_2, max_tokens=cfg["max_tokens"])

    gen_device = "cpu" if device in ("mps", "cpu") else device
    gen = torch.Generator(gen_device).manual_seed(data.seed)
    result = pipe(
        prompt=prompt,
        control_image=control_image,
        guidance_scale=guidance_scale if guidance_scale is not None else cfg["guidance"],
        num_inference_steps=n_steps,
        height=target_h,
        width=target_w,
        max_sequence_length=cfg["max_tokens"],
        generator=gen,
    )
    image: Image.Image = result.images[0]

    image = _match_color_statistics(
        image, data.target_brightness, data.target_saturation
    )

    del pipe
    free_torch()
    return image


def decode_brainimg(
    path: str | Path,
    out_path: str | Path,
    size: str | None = None,
    steps: int | None = None,
    device_override: str | None = None,
    quantize: bool = False,
    guidance_scale: float | None = None,
    depth_scale: float | None = None,
    canny_scale: float | None = None,
    seg_scale: float | None = None,
    model: str = DEFAULT_MODEL,
) -> tuple[BrainimgData, Image.Image]:
    """Read *path*, regenerate the image, save it to *out_path*."""
    data = load_brainimg(path)
    image = generate_image(
        data,
        size=size,
        steps=steps,
        device_override=device_override,
        quantize=quantize,
        guidance_scale=guidance_scale,
        depth_scale=depth_scale,
        canny_scale=canny_scale,
        seg_scale=seg_scale,
        model=model,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "PNG" if str(out_path).lower().endswith(".png") else "JPEG"
    save_kwargs = {"quality": 95} if fmt == "JPEG" else {}
    image.save(out_path, format=fmt, **save_kwargs)
    return data, image
