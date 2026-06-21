"""brainimg decoder CLI: .brainimg blueprint -> regenerated image.

Usage:
    python decoder.py out.brainimg -o recon.jpg [--steps 20] [--size 256x256]
    python decoder.py out.brainimg -o recon.png --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from brainimg.device import get_torch_device
from brainimg.format import load_brainimg
from brainimg.generate import decode_brainimg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decoder",
        description="Regenerate an image from a .brainimg semantic blueprint.",
    )
    parser.add_argument("brainimg", help="path to the .brainimg file")
    parser.add_argument(
        "-o", "--output", default="recon.jpg", help="output image path (jpg/png)"
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="inference steps (default: from file)"
    )
    parser.add_argument(
        "--size",
        default=None,
        help="output size as WxH (e.g. 256x256); default: scaled from original",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="compute device: 'cpu' = full fp32 (slow, best fidelity, needs "
        "~10 GB RAM), 'mps' = int8 quantized (Apple Silicon), 'cuda' = fp16 "
        "(NVIDIA), 'auto' = detect best (default: auto)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="int8-quantize weights on CPU to fit low-RAM machines (~5 GB "
        "instead of ~10 GB). Small quality cost. Ignored on MPS (always "
        "quantized) and CUDA (never needed).",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=None,
        help="classifier-free guidance scale (default: 7.5 sd15, 7.0 sdxl, 0.0 "
        "zimage). Higher = more prompt adherence, lower = more "
        "ControlNet/structural fidelity. Z-Image-Turbo is distilled for 0.0; "
        "overriding it is untested.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="depth ControlNet conditioning scale (default: 0.8 sd15, 1.0 sdxl, "
        "0.85 zimage)",
    )
    parser.add_argument(
        "--canny-scale",
        type=float,
        default=None,
        help="canny ControlNet conditioning scale (default: 1.0 sd15, 0.8 sdxl). "
        "Ignored under --model zimage (depth-only).",
    )
    parser.add_argument(
        "--seg-scale",
        type=float,
        default=None,
        help="segmentation ControlNet conditioning scale (default: 1.0 sd15, "
        "0.6 sdxl). Ignored when the file has no seg map, and under --model "
        "zimage (depth-only).",
    )
    parser.add_argument(
        "--model",
        choices=[
            "sd15",
            "sd15-turbo",
            "sdxl",
            "sdxl-turbo",
            "zimage",
            "flux-depth",
            "flux-canny",
        ],
        default="sd15",
        help="base diffusion model: 'sd15' (default, ~3.5 GB, 512) or 'sdxl' "
        "(~7 GB base + 3 ControlNets, 1024, ~5-10x slower on CPU). The seg "
        "ControlNet is supported on both. 'sd15-turbo' / 'sdxl-turbo' add "
        "ByteDance's Hyper-SD 8-step distilled LoRA on top of the same base "
        "+ ControlNets -- ~4x faster on CPU at a small quality cost (8 steps "
        "instead of 20-30, guidance scale 7.0/7.5). 'zimage' uses "
        "Tongyi-MAI/Z-Image-Turbo (6B bf16 DiT) + the alibaba-pai Union "
        "ControlNet (depth-only; canny and seg from the blueprint are "
        "ignored). Needs ~16 GB VRAM; 8 GB Apple Silicon should use 'sd15'. "
        "8 steps, fast on CUDA. 'flux-depth' uses FLUX.1-Depth-dev (~22 GB "
        "resident; pass --quantize for FP8 ~12 GB) and feeds the blueprint's "
        "depth map; 'flux-canny' is the same but with FLUX.1-Canny-dev + "
        "the canny map. Both ignore the other map and any seg map "
        "(channel-concat control, one image).",
    )
    args = parser.parse_args(argv)

    path = Path(args.brainimg)
    if not path.exists():
        print(f"error: brainimg file not found: {path}", file=sys.stderr)
        return 2

    data = load_brainimg(path)
    device = (
        "cpu"
        if args.device == "cpu"
        else (get_torch_device() if args.device == "auto" else args.device)
    )

    if args.model in ("flux-depth", "flux-canny"):
        # FLUX is bf16; --quantize FP8's the transformer + T5 (the big two).
        if args.quantize:
            mode = "bf16 + FP8 weights (transformer + T5)"
        elif device == "cuda":
            mode = "bf16"
        elif device == "mps":
            mode = "bf16 + cpu-offload"
        else:
            mode = "bf16 (resident in RAM, ~22 GB)"
    elif args.model == "zimage":
        # bf16 throughout. cuda: resident. mps: layers stream host<->device.
        # cpu: whole pipeline resident in host RAM (no offload -- diffusers'
        # enable_model_cpu_offload requires an accelerator to move *to*).
        if device == "cuda":
            mode = "bf16"
        elif device == "mps":
            mode = "bf16 + cpu-offload"
        else:
            mode = "bf16 (resident in RAM, ~18 GB)"
    elif args.model in ("sd15-turbo", "sdxl-turbo"):
        # Turbo LoRAs ride on the SD 1.5 / SDXL base + same ControlNets; the
        # mode string mirrors the non-turbo path but flags the distillation.
        if device == "cpu":
            mode = "fp32 + Hyper-SD 8-step LoRA"
        elif device == "mps":
            mode = "int8 weights + activations + Hyper-SD 8-step LoRA"
        else:
            mode = "fp16 + Hyper-SD 8-step LoRA"
    elif device == "cpu":
        mode = "int8 weights" if args.quantize else "fp32 (no quantization)"
    elif device == "mps":
        mode = "int8 weights + activations"
    else:
        mode = "fp16"
    print(f"Decoding {path} on {device} [{mode}] model={args.model} ...")
    print(f"  prompt : {data.prompt}")
    print(f"  seed   : {data.seed}")
    from brainimg.generate import _model_config

    if args.model in ("zimage", "flux-depth", "flux-canny"):
        # Z-Image + FLUX ignore the file's stored step count (tuned for SD 1.5);
        # show the effective step count they actually use.
        eff_steps = args.steps or _model_config(args.model)["default_steps"]
        print(f"  steps  : {eff_steps} ({args.model} default; file stored {data.steps})")
    elif args.model in ("sd15-turbo", "sdxl-turbo"):
        # Turbo stacks ignore the file's stored step count (tuned for the
        # 20-30 step SD schedule) and use the distilled LoRA's 8 steps unless
        # the user passes --steps explicitly.
        eff_steps = args.steps or _model_config(args.model)["default_steps"]
        print(f"  steps  : {eff_steps} ({args.model} default; file stored {data.steps})")
    else:
        print(f"  steps  : {args.steps or data.steps}")
    if (
        device == "cpu"
        and not args.quantize
        and args.model not in ("zimage", "flux-depth", "flux-canny")
        and args.model not in ("sd15-turbo", "sdxl-turbo")
    ):
        print("  note   : CPU fp32 is slow (minutes/image). Add --quantize for less memory.")
    if args.model in ("sd15-turbo", "sdxl-turbo"):
        print("  note   : Hyper-SD 8-step distilled LoRA; --cfg defaults to 7.0/7.5.")
    if args.model == "zimage" and device != "cuda":
        print("  note   : Z-Image without CUDA is slow. Prefer --device cuda for 8-step speed.")
    if args.model == "zimage" and device == "cpu":
        print("  note   : Z-Image on CPU keeps the whole bf16 pipeline in RAM (~18 GB).")
    if args.model == "zimage":
        print("  note   : Z-Image path uses depth only; canny/seg maps are ignored.")
    if args.model in ("flux-depth", "flux-canny"):
        other = "canny" if args.model == "flux-depth" else "depth"
        print(
            f"  note   : FLUX is bf16; {args.model} uses only its "
            f"{'depth' if args.model == 'flux-depth' else 'canny'} map "
            f"({other}/seg maps ignored). Add --quantize on CPU to drop RAM "
            f"from ~22 GB to ~12 GB via FP8 (transformer + T5)."
        )
    if args.model in ("flux-depth", "flux-canny") and device == "cpu" and not args.quantize:
        print(
            "  note   : FLUX on CPU without --quantize keeps the full bf16 pipeline "
            "in RAM (~22 GB)."
        )
    if args.model in ("flux-depth", "flux-canny") and device == "mps":
        print(
            "  note   : FLUX on MPS streams layers host<->device; "
            "8 GB Apple Silicon not supported (use --model sd15)."
        )

    t0 = time.time()
    _, image = decode_brainimg(
        path,
        args.output,
        size=args.size,
        steps=args.steps,
        device_override=device,
        quantize=args.quantize,
        guidance_scale=args.cfg,
        depth_scale=args.depth_scale,
        canny_scale=args.canny_scale,
        seg_scale=args.seg_scale,
        model=args.model,
    )
    dt = time.time() - t0

    print(f"  size   : {image.size[0]}x{image.size[1]}")
    print(f"  time   : {dt:.1f}s")
    print(f"  saved  : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
