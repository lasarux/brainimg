"""Build a single grid panel: original + all Lenna reconstructions.

Lays out the original Lenna image alongside every available reconstruction
at 512x512, labeled with model name + PSNR, in a grid. Handy for eyeballing
the fidelity differences across all decoder backends at once.

Usage:
    python scripts/make_lenna_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
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
    src = Image.open("samples/lenna.tiff").convert("RGB").resize((512, 512), Image.LANCZOS)
    src_arr = np.array(src)

    recons = [
        ("ORIGINAL", "", "samples/lenna.tiff"),
        ("SD 1.5", "30-step, new scales", "lenna_sd15_new_scales.png"),
        ("SD 1.5 turbo", "8-step Hyper-SD", "lenna_sd15_turbo.png"),
        ("SDXL", "30-step, 512", "lenna_sdxl_512_new_scales.png"),
        ("SDXL turbo", "8-step Hyper-SD", "lenna_sdxl_turbo.png"),
        ("Z-Image", "depth-only, 8-step", "lenna_zimage.png"),
        ("Qwen-Image", "depth-only, 50-step", "lenna_qwen_image.png"),
        ("HunyuanDiT full", "d+c, 50-step, 1024", "lenna_hunyuan_full.png"),
        ("SANA", "HED/canny, 20-step, 1024", "lenna_sana_s0.8.png"),
        ("FLUX.2-klein", "img2img, 4-step, 512", "lenna_flux2_klein.png"),
        ("FLUX depth", "30-step, FP8", "lenna_flux_depth.png"),
        ("FLUX depth turbo", "8-step Hyper-SD, FP8", "lenna_flux_depth_turbo.png"),
    ]

    cell_w, cell_h = 512, 512 + 48
    cols = 4
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
            panel = _label_panel(Image.new("RGB", (512, 512), (40, 40, 40)), label, "(missing)")
        else:
            img = Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS)
            if label == "ORIGINAL":
                sub = "source"
            else:
                mse, psnr = _stats(np.array(img), src_arr)
                if np.isfinite(psnr):
                    sub = f"{sublabel}  |  PSNR {psnr:.2f} dB"
                else:
                    sub = f"{sublabel}  |  PSNR inf"
            panel = _label_panel(img, label, sub)

        canvas.paste(panel, (x, y))

    out = "lenna_grid.jpg"
    canvas.save(out, format="JPEG", quality=90)
    print(f"wrote {out} ({Path(out).stat().st_size:,} bytes)")
    print(f"grid: {len(recons)} panels, {cols}x{rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
