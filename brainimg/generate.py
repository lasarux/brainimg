"""The decoder: regenerate an image from a .brainimg blueprint.

Loads Stable Diffusion 1.5 with two ControlNets (depth + Canny) and re-paints
the scene described by the blueprint.

Device strategy:
  * ``cpu``  : full fp32, no quantization. Best fidelity, slow (minutes/image).
    Needs ~10 GB RAM. Pass ``--quantize`` to int8-quantize the weights and
    fit in ~5 GB at a small quality cost.
  * ``mps``  (Apple Silicon default): SD 1.5 fp16 produces NaNs on MPS, so
    the UNet and ControlNets are int8-quantized (weights + activations) to
    avoid fp16 matmuls. Fits in 8 GB.
  * ``cuda`` (NVIDIA): fp16, works correctly. Fast and high fidelity.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .device import free_torch, get_dtype, get_torch_device
from .extract import b64_to_image
from .format import BrainimgData, load_brainimg

SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_DEPTH_ID = "lllyasviel/control_v11f1p_sd15_depth"
CONTROLNET_CANNY_ID = "lllyasviel/control_v11p_sd15_canny"

# High scales: structural fidelity to the original is the whole point.
CONTROLNET_DEPTH_SCALE = 1.5
CONTROLNET_CANNY_SCALE = 1.2

# Standard SD 1.5 guidance.
GUIDANCE_SCALE = 7.5

# Default generation side length. 512 is reasonable on a machine with plenty
# of RAM; 256 is the safe ceiling on an 8 GB Apple Silicon Mac with int8.
MAX_DEFAULT_SIDE = 512


def compute_target_size(
    orig_w: int, orig_h: int, override: str | None = None
) -> tuple[int, int]:
    """Pick a generation size, rounded to a multiple of 8 (SD requirement)."""
    if override:
        try:
            ws, hs = override.lower().split("x")
            w, h = int(ws), int(hs)
        except ValueError as exc:
            raise ValueError(f"--size expects WxH, got {override!r}") from exc
    else:
        scale = MAX_DEFAULT_SIDE / max(orig_w, orig_h)
        w, h = int(round(orig_w * scale)), int(round(orig_h * scale))

    w = max(8, (w // 8) * 8)
    h = max(8, (h // 8) * 8)
    return w, h


def _load_conditioning_maps(
    data: BrainimgData, target_w: int, target_h: int
) -> list[Image.Image]:
    """Decode the depth + Canny maps and upscale them to the target size."""
    depth = b64_to_image(data.depth_map_b64).convert("RGB").resize(
        (target_w, target_h), Image.LANCZOS
    )
    canny = b64_to_image(data.canny_map_b64).convert("RGB").resize(
        (target_w, target_h), Image.NEAREST  # keep edges crisp
    )
    return [depth, canny]


def _build_pipeline(device: str, dtype, quantize: bool = False):
    """Construct the SD1.5 + dual-ControlNet pipeline.

    Args:
        device: "cuda", "mps", or "cpu".
        dtype: torch dtype for the device (fp16 on GPU, fp32 on CPU).
        quantize: if True, int8-quantize weights to save memory (useful on
            low-RAM machines). On MPS this is always done regardless.
    """
    import torch
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    # Always load from the fp16 checkpoint (smaller download) and upcast
    # to fp32 in memory when needed.
    load_dtype = torch.float16
    variant = "fp16"

    controlnets = [
        ControlNetModel.from_pretrained(
            CONTROLNET_DEPTH_ID, torch_dtype=load_dtype, variant=variant
        ),
        ControlNetModel.from_pretrained(
            CONTROLNET_CANNY_ID, torch_dtype=load_dtype, variant=variant
        ),
    ]

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL_ID,
        controlnet=controlnets,
        torch_dtype=load_dtype,
        variant=variant,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # CPU: upcast to fp32 BEFORE moving to device (diffusers refuses fp16 on CPU).
    if device == "cpu":
        pipe = pipe.to(torch.float32)

    pipe = pipe.to(device)

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


def generate_image(
    data: BrainimgData,
    size: str | None = None,
    steps: int | None = None,
    device_override: str | None = None,
    quantize: bool = False,
) -> Image.Image:
    """Regenerate a single image from *data*.

    Args:
        device_override: "cpu", "mps", "cuda", or None (auto-detect).
        quantize: int8-quantize weights on CPU to save memory (small quality cost).
    """
    device = device_override or get_torch_device()
    dtype = get_dtype(device)

    target_w, target_h = compute_target_size(
        data.original_width, data.original_height, size
    )
    n_steps = steps or data.steps

    conditioning = _load_conditioning_maps(data, target_w, target_h)

    pipe, torch = _build_pipeline(device, dtype, quantize=quantize)

    gen = torch.Generator(device).manual_seed(data.seed)
    result = pipe(
        prompt=data.prompt,
        negative_prompt=data.negative_prompt,
        image=conditioning,
        controlnet_conditioning_scale=[CONTROLNET_DEPTH_SCALE, CONTROLNET_CANNY_SCALE],
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=n_steps,
        generator=gen,
    )
    image: Image.Image = result.images[0]

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
) -> tuple[BrainimgData, Image.Image]:
    """Read *path*, regenerate the image, save it to *out_path*."""
    data = load_brainimg(path)
    image = generate_image(
        data, size=size, steps=steps, device_override=device_override, quantize=quantize
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "PNG" if str(out_path).lower().endswith(".png") else "JPEG"
    save_kwargs = {"quality": 95} if fmt == "JPEG" else {}
    image.save(out_path, format=fmt, **save_kwargs)
    return data, image
