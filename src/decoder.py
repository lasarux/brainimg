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
            "qwen-image",
            "hunyuan",
            "hunyuan-full",
            "sana",
            "flux2-klein",
            "flux-depth",
            "flux-canny",
            "flux-depth-turbo",
            "flux-canny-turbo",
            "sd35",
            "flux-union",
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
        "ignored). 'qwen-image' uses Alibaba's Qwen-Image (Apache 2.0 DiT) "
        "+ InstantX Union ControlNet (depth-only; canny and seg ignored, "
        "same as zimage). 50 steps, Qwen text encoder (512 tokens), "
        "true_cfg_scale 4.0. Needs ~20 GB RAM resident on CPU. "
        "'hunyuan' uses Tencent's HunyuanDiT v1.2 Distilled (bilingual "
        "DiT, 25 steps) with separate depth + canny ControlNets (same "
        "two-conditioner pattern as sd15/sdxl; seg map ignored). bf16, "
        "BERT + T5 text encoders. Needs ~12 GB RAM resident on CPU. "
        "'hunyuan-full' is the non-distilled variant (50 steps, same "
        "ControlNets). HunyuanDiT defaults to resolution binning (remaps "
        "--size to the nearest trained shape, e.g. 512x512 -> 1024x1024, "
        "on-distribution and artifact-free); pass --no-bin-resolution to "
        "honor --size exactly (off-distribution, ~4x faster but can produce "
        "severe artifacts). "
        "'sana' uses NVIDIA's SANA 600M (MIT, linear DiT) "
        "with an HED ControlNet -- the only available ControlNet type for "
        "SANA. The blueprint's canny map is fed to the HED ControlNet "
        "(edge-to-edge, closest match); depth and seg are ignored. "
        "bf16, T5 text encoder, 20 steps. Needs ~5 GB RAM on CPU. "
        "'flux2-klein' uses FLUX.2-klein-4B (Apache 2.0, 4B, ungated) as "
        "img2img -- feeds the depth map as the starting image (no "
        "ControlNet exists for klein). 4-step distilled, guidance 1.0. "
        "Experimental pseudo-ControlNet approach; canny/seg are ignored. "
        "~13 GB RAM on CPU. "
        "'flux-depth' uses FLUX.1-Depth-dev (~22 GB resident; pass "
        "--quantize for FP8 ~12 GB) and feeds the blueprint's depth map; "
        "'flux-canny' is the same but with FLUX.1-Canny-dev + the canny "
        "map. Both ignore the other map and any seg map (channel-concat "
        "control, one image). 'flux-depth-turbo' / 'flux-canny-turbo' add "
        "Hyper-SD's 8-step FLUX LoRA on top of the same control pipeline "
        "-- drops FLUX from 30 to 8 steps, guidance 3.5 (the dev default). "
        "~4-5x faster on CPU. "
        "'sd35' uses Stable Diffusion 3.5 Large (8B MMDiT) + official "
        "depth + canny ControlNets (two 8B nets, fed simultaneously via "
        "SD3MultiControlNetModel). 1024-native, 50 steps, guidance 4.5, "
        "bf16, gated (Stability AI community license). ~16-20 GB RAM on CPU. "
        "'flux-union' uses FLUX.1-dev + Shakker-Labs Union ControlNet "
        "(depth mode 2 + canny mode 0 fed simultaneously). 24 steps, "
        "guidance 3.5, bf16, gated (FLUX non-commercial). ~24 GB RAM on CPU "
        "(~12 GB with --quantize).",
    )
    parser.add_argument(
        "--no-bin-resolution",
        action="store_true",
        help="HunyuanDiT-only. By default HunyuanDiT uses resolution binning "
        "(diffusers remaps --size to the nearest trained shape, e.g. "
        "512x512 -> 1024x1024), which is on-distribution and artifact-free. "
        "Pass --no-bin-resolution to honor --size exactly -- off-distribution "
        "for HunyuanDiT (trained at 1024) and can produce severe artifacts "
        "(catastrophic noise), but ~4x faster. Ignored by all other backends.",
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

    if args.model in (
        "flux-depth", "flux-canny", "flux-depth-turbo", "flux-canny-turbo", "flux-union",
    ):
        # FLUX is bf16; --quantize FP8's the transformer + T5 (the big two).
        # Turbo variants add the Hyper-SD FLUX 8-step LoRA on top.
        turbo_suffix = " + Hyper-SD 8-step LoRA" if args.model.endswith("-turbo") else ""
        if args.model == "flux-union":
            ram_note = "~24 GB" if not args.quantize else "~12 GB"
        else:
            ram_note = "~22 GB" if not args.quantize else "~12 GB"
        if args.quantize:
            mode = f"bf16 + FP8 weights (transformer + T5){turbo_suffix}"
        elif device == "cuda":
            mode = f"bf16{turbo_suffix}"
        elif device == "mps":
            mode = f"bf16 + cpu-offload{turbo_suffix}"
        else:
            mode = f"bf16 (resident in RAM, {ram_note}){turbo_suffix}"
    elif args.model == "sd35":
        # SD3.5 Large: bf16, depth + canny ControlNets, 3 text encoders.
        if device == "cuda":
            mode = "bf16"
        elif device == "mps":
            mode = "bf16 + cpu-offload"
        else:
            mode = "bf16 (resident in RAM, ~16-20 GB)"
    elif args.model in ("hunyuan", "hunyuan-full"):
        # HunyuanDiT: bf16, depth + canny ControlNets (two separate nets).
        variant = "distilled (25-step)" if args.model == "hunyuan" else "full (50-step)"
        if device == "cuda":
            mode = f"bf16 {variant}"
        elif device == "mps":
            mode = f"bf16 + cpu-offload {variant}"
        else:
            mode = f"bf16 {variant} (resident in RAM, ~12 GB)"
    elif args.model == "qwen-image":
        # Qwen-Image: bf16, Union ControlNet (depth-only), Qwen text encoder.
        # Same memory strategy as Z-Image.
        if device == "cuda":
            mode = "bf16"
        elif device == "mps":
            mode = "bf16 + cpu-offload"
        else:
            mode = "bf16 (resident in RAM, ~20 GB)"
    elif args.model == "sana":
        # SANA: bf16, HED ControlNet (canny map fed to it), T5 text encoder.
        if device == "cuda":
            mode = "bf16"
        elif device == "mps":
            mode = "bf16 + cpu-offload"
        else:
            mode = "bf16 (resident in RAM, ~5 GB)"
    elif args.model == "flux2-klein":
        # FLUX.2-klein-4B: bf16, img2img (depth map as starting image).
        if device == "cuda":
            mode = "bf16"
        elif device == "mps":
            mode = "bf16 + cpu-offload"
        else:
            mode = "bf16 (resident in RAM, ~13 GB)"
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

    if args.model in (
        "zimage", "qwen-image", "hunyuan", "hunyuan-full", "sana",
        "flux2-klein",
        "flux-depth", "flux-canny",
        "flux-depth-turbo", "flux-canny-turbo",
        "sd35", "flux-union",
    ):
        # Z-Image + Qwen-Image + FLUX (+ turbo) + SD3.5 + FLUX-Union ignore
        # the file's stored step count (tuned for SD 1.5); show the effective
        # step count they use.
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
        and args.model not in (
            "zimage", "qwen-image", "hunyuan", "hunyuan-full", "sana",
            "flux2-klein",
            "flux-depth", "flux-canny",
            "sd35", "flux-union",
        )
        and args.model not in ("sd15-turbo", "sdxl-turbo")
        and args.model not in ("flux-depth-turbo", "flux-canny-turbo")
    ):
        print("  note   : CPU fp32 is slow (minutes/image). Add --quantize for less memory.")
    if args.model in ("sd15-turbo", "sdxl-turbo"):
        print("  note   : Hyper-SD 8-step distilled LoRA; --cfg defaults to 7.0/7.5.")
    if args.model in ("flux-depth-turbo", "flux-canny-turbo"):
        print("  note   : Hyper-SD FLUX 8-step LoRA; --cfg defaults to 3.5 (dev default).")
    if args.model == "zimage" and device != "cuda":
        print("  note   : Z-Image without CUDA is slow. Prefer --device cuda for 8-step speed.")
    if args.model == "zimage" and device == "cpu":
        print("  note   : Z-Image on CPU keeps the whole bf16 pipeline in RAM (~18 GB).")
    if args.model == "zimage":
        print("  note   : Z-Image path uses depth only; canny/seg maps are ignored.")
    if args.model == "qwen-image" and device != "cuda":
        print("  note   : Qwen-Image without CUDA is slow (50 steps on CPU).")
    if args.model == "qwen-image" and device == "cpu":
        print("  note   : Qwen-Image on CPU keeps the whole bf16 pipeline in RAM (~20 GB).")
    if args.model == "qwen-image":
        print("  note   : Qwen-Image path uses depth only; canny/seg maps are ignored.")
    if args.model in ("hunyuan", "hunyuan-full") and device != "cuda":
        print(f"  note   : HunyuanDiT without CUDA is slow ({args.model} on CPU).")
    if args.model in ("hunyuan", "hunyuan-full") and device == "cpu":
        print("  note   : HunyuanDiT on CPU keeps the whole bf16 pipeline in RAM (~12 GB).")
    if args.model in ("hunyuan", "hunyuan-full"):
        print("  note   : HunyuanDiT uses depth + canny; seg map is ignored (no seg ControlNet).")
    if args.model in ("hunyuan", "hunyuan-full") and args.no_bin_resolution:
        print(
            "  note   : --no-bin-resolution set; --size is honored exactly "
            "(HunyuanDiT trained at 1024, off-distribution -- can produce "
            "severe artifacts)."
        )
    if args.model == "sana" and device != "cuda":
        print("  note   : SANA without CUDA is slow (20 steps on CPU).")
    if args.model == "sana" and device == "cpu":
        print("  note   : SANA on CPU keeps the whole bf16 pipeline in RAM (~5 GB).")
    if args.model == "sana":
        print("  note   : SANA uses an HED ControlNet fed the canny map; depth/seg are ignored.")
    if args.model == "flux2-klein" and device != "cuda":
        print("  note   : FLUX.2-klein without CUDA is slower than GPU but only 4 steps.")
    if args.model == "flux2-klein" and device == "cpu":
        print("  note   : FLUX.2-klein on CPU keeps the whole bf16 pipeline in RAM (~13 GB).")
    if args.model == "flux2-klein":
        print("  note   : FLUX.2-klein uses img2img (depth as start image); canny/seg ignored.")
    if args.model in (
        "flux-depth", "flux-canny", "flux-depth-turbo", "flux-canny-turbo",
    ):
        is_turbo = args.model.endswith("-turbo")
        base = args.model[: -len("-turbo")] if is_turbo else args.model
        other = "canny" if base == "flux-depth" else "depth"
        cond = "depth" if base == "flux-depth" else "canny"
        print(
            f"  note   : FLUX is bf16; {base} uses only its {cond} map "
            f"({other}/seg maps ignored). Add --quantize on CPU to drop RAM "
            f"from ~22 GB to ~12 GB via FP8 (transformer + T5)."
        )
    if args.model == "flux-union":
        print(
            "  note   : FLUX Union is bf16; uses depth (mode 2) + canny (mode 0) "
            "simultaneously via a single Union ControlNet (seg map ignored). "
            "Add --quantize on CPU to drop RAM from ~24 GB to ~12 GB via FP8."
        )
    _flux_models = (
        "flux-depth", "flux-canny", "flux-depth-turbo", "flux-canny-turbo", "flux-union",
    )
    if args.model in _flux_models and device == "cpu" and not args.quantize:
        ram = "~24 GB" if args.model == "flux-union" else "~22 GB"
        print(
            f"  note   : FLUX on CPU without --quantize keeps the full bf16 pipeline "
            f"in RAM ({ram})."
        )
    if args.model in _flux_models and device == "mps":
        print(
            "  note   : FLUX on MPS streams layers host<->device; "
            "8 GB Apple Silicon not supported (use --model sd15)."
        )
    if args.model == "sd35":
        print(
            "  note   : SD3.5 is bf16; uses depth + canny ControlNets simultaneously "
            "(seg map ignored). Needs ~16-20 GB RAM resident on CPU."
        )
    if args.model == "sd35" and device == "mps":
        print(
            "  note   : SD3.5 on MPS streams layers host<->device; "
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
        bin_resolution=not args.no_bin_resolution,
    )
    dt = time.time() - t0

    print(f"  size   : {image.size[0]}x{image.size[1]}")
    print(f"  time   : {dt:.1f}s")
    print(f"  saved  : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
