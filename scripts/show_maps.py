"""Visualize the conditioning maps stored in a .brainimg file.

Decodes the base64 depth, Canny, and (optional) segmentation maps and writes
them to PNGs (or one combined side-by-side image). Handier than eyeballing
base64 strings when tuning MAP_SIZE or ControlNet scales.

Usage:
    python scripts/show_maps.py out.brainimg                  # writes per-map PNGs
    python scripts/show_maps.py out.brainimg -o maps.png      # one side-by-side image
    python scripts/show_maps.py out.brainimg --show           # open each map in a viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainimg.extract import b64_to_image
from brainimg.format import load_brainimg


def _label_panel(img: Image.Image, label: str) -> Image.Image:
    """Return a new image with a caption bar above *img* containing *label*."""
    bar_h = 28
    panel = Image.new("RGB", (img.width, bar_h + img.height), (20, 20, 20))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((panel.width - tw) // 2, 5), label, fill=(235, 235, 235), font=font)
    panel.paste(img, (0, bar_h))
    return panel


def maps_of(path: str | Path) -> dict[str, Image.Image]:
    """Decode and return {"depth": ..., "canny": ..., "seg": ...} from *path*.

    The seg entry is omitted when the blueprint carries no segmentation map.
    """
    data = load_brainimg(path)
    maps: dict[str, Image.Image] = {
        "depth": b64_to_image(data.depth_map_b64).convert("RGB"),
        "canny": b64_to_image(data.canny_map_b64).convert("RGB"),
    }
    if getattr(data, "segmentation_map_b64", ""):
        maps["seg"] = b64_to_image(data.segmentation_map_b64).convert("RGB")
    return maps


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="show_maps",
        description="Visualize the depth/canny/seg conditioning maps in a .brainimg file.",
    )
    parser.add_argument("brainimg", help="path to the .brainimg file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-o",
        "--output",
        help="write one labeled side-by-side image to this path (png)",
    )
    group.add_argument(
        "--prefix",
        default=None,
        help="write each map to <prefix>-<name>.png (default: per-map PNGs next to the file)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open each map in the default image viewer (PIL.Image.show)",
    )
    args = parser.parse_args()

    src = Path(args.brainimg)
    maps = maps_of(src)

    if args.output:
        panels = [_label_panel(img, name) for name, img in maps.items()]
        h = max(p.height for p in panels)
        # Pad to common height so the join is clean.
        padded = []
        for p in panels:
            if p.height < h:
                bg = Image.new("RGB", (p.width, h), (10, 10, 10))
                bg.paste(p, (0, 0))
                padded.append(bg)
            else:
                padded.append(p)
        gap = 16
        total_w = sum(p.width for p in padded) + gap * (len(padded) - 1)
        canvas = Image.new("RGB", (total_w, h), (10, 10, 10))
        x = 0
        for p in padded:
            canvas.paste(p, (x, 0))
            x += p.width + gap
        canvas.save(args.output, format="PNG")
        print(f"wrote {args.output} ({Path(args.output).stat().st_size:,} bytes)")
        if args.show:
            canvas.show()
        return 0

    prefix = args.prefix or str(src.with_suffix(""))
    for name, img in maps.items():
        out = f"{prefix}-{name}.png"
        img.save(out, format="PNG")
        print(f"wrote {out} ({Path(out).stat().st_size:,} bytes)  [{img.size[0]}x{img.size[1]}]")
        if args.show:
            img.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
