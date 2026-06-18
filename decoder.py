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

    if device == "cpu":
        mode = "int8 weights" if args.quantize else "fp32 (no quantization)"
    elif device == "mps":
        mode = "int8 weights + activations"
    else:
        mode = "fp16"
    print(f"Decoding {path} on {device} [{mode}] ...")
    print(f"  prompt : {data.prompt}")
    print(f"  seed   : {data.seed}")
    print(f"  steps  : {args.steps or data.steps}")
    if device == "cpu" and not args.quantize:
        print("  note   : CPU fp32 is slow (minutes/image). Add --quantize for less memory.")

    t0 = time.time()
    _, image = decode_brainimg(
        path,
        args.output,
        size=args.size,
        steps=args.steps,
        device_override=device,
        quantize=args.quantize,
    )
    dt = time.time() - t0

    print(f"  size   : {image.size[0]}x{image.size[1]}")
    print(f"  time   : {dt:.1f}s")
    print(f"  saved  : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
