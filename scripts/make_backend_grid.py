"""Build a single grid panel: original + all reconstructions for a sample.

Lays out the original image alongside every available reconstruction at a
common size, labeled with model name + PSNR, in a grid. Handy for eyeballing
the fidelity differences across all decoder backends at once.

Usage:
    python scripts/make_backend_grid.py mandril
    python scripts/make_backend_grid.py peppers --size 512
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.compare_backends import DEFAULT_BACKENDS, _find_source, _resize


def _mse_psnr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse <= 0 else 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)
    return mse, psnr


def _label_panel(img: Image.Image, label: str, sublabel: str = "") -> Image.Image:
    bar_h = 48
    panel = Image.new("RGB", (img.width, bar_h + img.height), (20, 20, 20))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        font_sub = font
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((panel.width - tw) // 2, 4), label, fill=(235, 235, 235), font=font)
    if sublabel:
        bbox2 = draw.textbbox((0, 0), sublabel, font=font_sub)
        tw2 = bbox2[2] - bbox2[0]
        draw.text(((panel.width - tw2) // 2, 26), sublabel, fill=(180, 180, 180), font=font_sub)
    panel.paste(img, (0, bar_h))
    return panel


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a grid of original + all backend reconstructions."
    )
    parser.add_argument("sample", help="sample name (e.g. mandril, peppers, cameraman)")
    parser.add_argument("--size", type=int, default=512, help="cell size (default: 512)")
    parser.add_argument(
        "--cols", type=int, default=4, help="grid columns (default: 4)"
    )
    args = parser.parse_args()

    src_path = _find_source(args.sample)
    src = _resize(Image.open(src_path), (args.size, args.size))
    src_arr = np.array(src)

    recons = [("ORIGINAL", "", src_path)]
    for label, suffix in DEFAULT_BACKENDS:
        recons.append((label, "", f"{args.sample}_{suffix}.png"))

    cell_w = args.size
    cell_h = args.size + 48
    cols = args.cols
    rows = (len(recons) + cols - 1) // cols
    gap = 12

    canvas_w = cols * cell_w + (cols + 1) * gap
    canvas_h = rows * cell_h + (rows + 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (10, 10, 10))

    for i, (label, sublabel, path) in enumerate(recons):
        col = i % cols
        row = i // cols
        x = gap + col * (cell_w + gap)
        y = gap + row * (cell_h + gap)

        if not Path(path).exists():
            panel = _label_panel(
                Image.new("RGB", (args.size, args.size), (40, 40, 40)), label, "(missing)"
            )
        else:
            img = _resize(Image.open(path), (args.size, args.size))
            if label == "ORIGINAL":
                sub = "source"
            else:
                mse, psnr = _mse_psnr(np.array(img), src_arr)
                if np.isfinite(psnr):
                    sub = f"PSNR {psnr:.2f} dB"
                else:
                    sub = "PSNR inf"
            panel = _label_panel(img, label, sub)

        canvas.paste(panel, (x, y))

    out = f"{args.sample}_grid.jpg"
    canvas.save(out, format="JPEG", quality=90)
    print(f"wrote {out} ({Path(out).stat().st_size:,} bytes)")
    print(f"grid: {len(recons)} panels, {cols}x{rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
