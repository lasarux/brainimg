"""Build a single side-by-side comparison image: original | reconstruction.

Produces comparison.jpg (or .png) with the two images scaled to a common
height, labeled, and joined horizontally. Handy for eyeballing how faithful
the brainimg round-trip is.

Usage:
    python scripts/make_comparison.py                       # defaults
    python scripts/make_comparison.py samples/real.jpg recon.png -o comparison.jpg
"""

from __future__ import annotations

import argparse

from PIL import Image, ImageDraw, ImageFont


def _fit_height(img: Image.Image, target_h: int) -> Image.Image:
    """Resize *img* so its height is *target_h*, preserving aspect ratio."""
    if img.height == target_h:
        return img
    scale = target_h / img.height
    return img.resize((max(1, round(img.width * scale)), target_h), Image.LANCZOS)


def _label_panel(img: Image.Image, label: str) -> Image.Image:
    """Return a new image with a caption bar above *img* containing *label*."""
    bar_h = 32
    panel = Image.new("RGB", (img.width, bar_h + img.height), (20, 20, 20))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((panel.width - tw) // 2, 6), label, fill=(235, 235, 235), font=font)
    panel.paste(img, (0, bar_h))
    return panel


def make_comparison(
    original: str,
    recon: str,
    out: str,
    height: int = 512,
    gap: int = 16,
) -> int:
    """Join *original* and *recon* side by side into *out*. Returns bytes written."""
    left = Image.open(original).convert("RGB")
    right = Image.open(recon).convert("RGB")

    target_h = min(height, max(left.height, right.height))
    left = _fit_height(left, target_h)
    right = _fit_height(right, target_h)

    ow, oh = Image.open(original).size
    rw, rh = Image.open(recon).size
    left = _label_panel(left, f"original  ({ow}x{oh})")
    right = _label_panel(right, f"recon  ({rw}x{rh})")

    total_w = left.width + gap + right.width
    total_h = max(left.height, right.height)
    canvas = Image.new("RGB", (total_w, total_h), (10, 10, 10))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))

    fmt = "PNG" if out.lower().endswith(".png") else "JPEG"
    save_kwargs = {"quality": 92} if fmt == "JPEG" else {}
    canvas.save(out, format=fmt, **save_kwargs)
    import os

    return os.path.getsize(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="make_comparison",
        description="Create a single side-by-side original-vs-recon comparison image.",
    )
    parser.add_argument(
        "original", nargs="?", default="samples/real.jpg", help="original image path"
    )
    parser.add_argument(
        "recon", nargs="?", default="recon.png", help="reconstructed image path"
    )
    parser.add_argument(
        "-o", "--output", default="comparison.jpg", help="output comparison file (jpg/png)"
    )
    parser.add_argument(
        "--height", type=int, default=512, help="common display height in pixels (default: 512)"
    )
    args = parser.parse_args()

    n = make_comparison(args.original, args.recon, args.output, height=args.height)
    print(f"wrote {args.output} ({n:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
