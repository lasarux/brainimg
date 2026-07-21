"""brainimg encoder CLI: image -> .brainimg blueprint.

Usage:
    python encoder.py samples/apple.jpg -o out.brainimg [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from brainimg.extract import encode_image
from brainimg.format import save_brainimg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="encoder",
        description="Encode an image into a tiny .brainimg semantic blueprint.",
    )
    parser.add_argument("image", help="path to the input image (jpg/png/etc.)")
    parser.add_argument(
        "-o", "--output", default="out.brainimg", help="output .brainimg path"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="fix the latent seed (default: random)"
    )
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"error: input image not found: {image_path}", file=sys.stderr)
        return 2

    original_bytes = image_path.stat().st_size

    print(f"Encoding {image_path} ...")
    data = encode_image(image_path, seed=args.seed)

    n = save_brainimg(args.output, data)
    ratio = original_bytes / n if n else float("inf")

    print(f"  caption : {data.prompt}")
    print(f"  seed    : {data.seed}")
    print(f"  original: {original_bytes:,} bytes")
    print(f"  brainimg: {n:,} bytes")
    print(f"  ratio   : {ratio:,.1f}x  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
