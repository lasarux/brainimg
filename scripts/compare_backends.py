"""Quick numeric comparison of reconstructions against a source image.

Computes per-channel MSE, PSNR, and mean absolute error between a source
image and each reconstructed PNG, all resized to a common size. Pure numpy +
Pillow, no model deps. Intended for eyeballing the fidelity gap across
decoder backends on a given sample.

Usage:
    python scripts/compare_backends.py mandril
    python scripts/compare_backends.py peppers --size 512
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# (display label, backend suffix, optional extra suffix used in sweep runs)
DEFAULT_BACKENDS = [
    ("SD 1.5 (30-step)", "sd15"),
    ("SD 1.5 turbo (8-step)", "sd15-turbo"),
    ("SDXL (30-step)", "sdxl"),
    ("SDXL turbo (8-step)", "sdxl-turbo"),
    ("Z-Image (depth-only)", "zimage"),
    ("Qwen-Image (depth-only)", "qwen-image"),
    ("SANA (HED/canny)", "sana"),
    ("FLUX.2-klein (img2img)", "flux2-klein"),
    ("FLUX depth (FP8)", "flux-depth"),
    ("FLUX depth turbo (FP8)", "flux-depth-turbo"),
    ("FLUX canny (FP8)", "flux-canny"),
    ("FLUX Union (depth+canny)", "flux-union"),
    ("SD 3.5 (depth+canny)", "sd35"),
]


def _resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.convert("RGB").resize(size, Image.LANCZOS)


def _stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse <= 0 else 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)
    mae = float(np.mean(np.abs(diff)))
    return {"mse": mse, "psnr_db": psnr, "mae": mae}


def _find_source(name: str) -> str:
    # Try exact name first, then name with common SIPI suffixes (_color, _gray).
    candidates = [name, f"{name}_color", f"{name}_gray"]
    for cand in candidates:
        for ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
            p = Path("samples") / f"{cand}{ext}"
            if p.exists():
                return str(p)
    raise FileNotFoundError(f"no samples/{name}*.tif/.jpg found")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare reconstructions against a source image."
    )
    parser.add_argument(
        "sample",
        help="sample name (e.g. mandril, peppers, cameraman); resolves to "
        "samples/<name>.* and <name>_<backend>.png reconstructions",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="common size for comparison (default: 512)",
    )
    args = parser.parse_args()

    src_path = _find_source(args.sample)
    src = _resize(Image.open(src_path), (args.size, args.size))
    src_arr = np.array(src)

    recons = [
        (label, str(Path("outputs") / f"{args.sample}_{suffix}.png"))
        for label, suffix in DEFAULT_BACKENDS
    ]
    size_str = f"{args.size}x{args.size}"
    print(f"source: {src_path}  compare size: {size_str}")
    print(f"{'recon':30s}  {'size':>11s}  {'MSE':>10s}  {'PSNR':>8s}  {'MAE':>7s}")
    print("-" * 75)
    for label, path in recons:
        if not Path(path).exists():
            print(f"{label:30s}  (missing: {path})")
            continue
        img = Image.open(path).convert("RGB")
        img_size = f"{img.width}x{img.height}"
        img_resized = _resize(img, (args.size, args.size))
        s = _stats(np.array(img_resized), src_arr)
        psnr = "inf" if not np.isfinite(s["psnr_db"]) else f"{s['psnr_db']:.2f}"
        print(f"{label:30s}  {img_size:>11s}  {s['mse']:>10.2f}  {psnr:>8s}  {s['mae']:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
