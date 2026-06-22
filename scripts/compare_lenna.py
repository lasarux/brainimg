"""Quick numeric comparison of lenna reconstructions against the source.

Computes per-channel MSE, PSNR, and mean absolute error between
samples/lenna.tiff and each reconstructed PNG, all resized to a common
size. Pure numpy + Pillow, no model deps. Intended for eyeballing the
SDXL vs SD15 fidelity gap on the Lenna test case.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.convert("RGB").resize(size, Image.LANCZOS)


def _stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse <= 0 else 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)
    mae = float(np.mean(np.abs(diff)))
    return {"mse": mse, "psnr_db": psnr, "mae": mae}


def main() -> int:
    src = _resize(Image.open("samples/lenna.tiff"), (512, 512))
    src_arr = np.array(src)
    recons = [
        ("SD15  512 30-step old scales", "lenna_recon.png"),
        ("SD15  512 30-step new scales", "lenna_sd15_new_scales.png"),
        ("SD15  512 turbo 8-step", "lenna_sd15_turbo.png"),
        ("SDXL  512 30-step old scales", "lenna_sdxl_512.png"),
        ("SDXL  512 30-step new scales", "lenna_sdxl_512_new_scales.png"),
        ("SDXL  512 turbo 8-step", "lenna_sdxl_turbo.png"),
        ("Z-Image depth 512", "lenna_zimage.png"),
        ("Qwen-Image depth 512", "lenna_qwen_image.png"),
        ("HunyuanDiT d+c 1024", "lenna_hunyuan.png"),
        ("SANA HED/canny 1024 s0.4", "lenna_sana_s0.4.png"),
        ("FLUX.2-klein img2img 512", "lenna_flux2_klein.png"),
        ("FLUX depth 512 30-step FP8", "lenna_flux_depth.png"),
        ("FLUX depth 512 turbo 8-step FP8", "lenna_flux_depth_turbo.png"),
    ]
    print(f"{'recon':28s}  {'size':>11s}  {'MSE':>10s}  {'PSNR':>8s}  {'MAE':>7s}")
    print("-" * 72)
    for label, path in recons:
        if not Path(path).exists():
            print(f"{label:28s}  (missing: {path})")
            continue
        img = Image.open(path).convert("RGB")
        size = f"{img.width}x{img.height}"
        img_resized = _resize(img, (512, 512))
        s = _stats(np.array(img_resized), src_arr)
        psnr = "inf" if not np.isfinite(s["psnr_db"]) else f"{s['psnr_db']:.2f}"
        print(f"{label:28s}  {size:>11s}  {s['mse']:>10.2f}  {psnr:>8s}  {s['mae']:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
