"""Sweep ControlNet scales + CFG on a sample to find better defaults.

Loads the SD 1.5 turbo pipeline once and runs multiple inferences with
different depth/canny/seg/cfg combinations, printing MSE/PSNR/MAE for each.
Reuses the same blueprint + seed so the only variable is the scale/cfg pair.

Usage:
    python scripts/sweep_scales.py mandril
    python scripts/sweep_scales.py test512
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from brainimg.format import load_brainimg
from brainimg.generate import (
    CLIP_MAX_TOKENS,
    _build_pipeline,
    _build_prompt,
    _load_conditioning_maps,
    _match_color_statistics,
)


def _stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse <= 0 else 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)
    mae = float(np.mean(np.abs(diff)))
    return {"mse": mse, "psnr_db": psnr, "mae": mae}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Sweep ControlNet scales on a sample.")
    parser.add_argument(
        "sample", help="sample name (e.g. mandril, peppers, cameraman, test512)"
    )
    args = parser.parse_args()

    src_path = None
    for cand in (args.sample, f"{args.sample}_color", f"{args.sample}_gray"):
        for ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
            p = Path("samples") / f"{cand}{ext}"
            if p.exists():
                src_path = str(p)
                break
        if src_path:
            break
    if src_path is None:
        print(f"unknown sample: {args.sample} (no samples/{args.sample}*.*)", file=sys.stderr)
        return 2
    blueprint = f"{args.sample}.brainimg"

    src = Image.open(src_path).convert("RGB").resize((512, 512), Image.LANCZOS)
    src_arr = np.array(src)

    data = load_brainimg(blueprint)
    device = "cpu"
    target_w, target_h = 512, 512

    has_seg = bool(getattr(data, "segmentation_map_b64", ""))
    conditioning = _load_conditioning_maps(data, target_w, target_h)

    pipe, torch = _build_pipeline(
        device, None, quantize=False, with_seg=has_seg, model="sd15-turbo"
    )
    prompt = _build_prompt(data, pipe.tokenizer, max_tokens=CLIP_MAX_TOKENS)

    # The sweep grid: (label, depth, canny, seg, cfg)
    # Final-pass compromise grid: depth 0.6-1.0, canny 1.0-1.2, seg 0.9-1.2, cfg 7.5.
    # Archived (retired) sample winner: (1.0/1.0/1.2/7.5); test512 winner:
    # (0.6/1.2/1.2/7.5). Pick scales that work well on both.
    sweep = [
        ("d1.0 seg1.2 (1.0/1.0/1.2/7.5)", 1.0, 1.0, 1.2, 7.5),
        ("d0.6 seg1.2 (0.6/1.2/1.2/7.5)", 0.6, 1.2, 1.2, 7.5),
        # Compromise: depth 0.8 (between the two winners).
        ("compromise (0.8/1.0/1.2/7.5)", 0.8, 1.0, 1.2, 7.5),
        ("compromise (0.8/1.2/1.2/7.5)", 0.8, 1.2, 1.2, 7.5),
        # Seg 1.0 (middle of 0.9 vs 1.2 split).
        ("compromise (0.8/1.0/1.0/7.5)", 0.8, 1.0, 1.0, 7.5),
        ("compromise (0.8/1.2/1.0/7.5)", 0.8, 1.2, 1.0, 7.5),
        # Keep depth at 1.0 but seg 1.0 (middle).
        ("d1.0 seg1.0 (1.0/1.0/1.0/7.5)", 1.0, 1.0, 1.0, 7.5),
        ("d1.0 seg1.0 (1.0/1.2/1.0/7.5)", 1.0, 1.2, 1.0, 7.5),
        # Depth 0.6 with seg 1.0 (middle).
        ("d0.6 seg1.0 (0.6/1.0/1.0/7.5)", 0.6, 1.0, 1.0, 7.5),
        ("d0.6 seg1.0 (0.6/1.2/1.0/7.5)", 0.6, 1.2, 1.0, 7.5),
    ]

    print(f"{'config':38s}  {'MSE':>10s}  {'PSNR':>8s}  {'MAE':>7s}  {'time':>6s}")
    print("-" * 80)

    gen_device = "cpu"
    best = None

    for label, ds, cs, ss, cfg in sweep:
        scales = [ds, cs]
        if has_seg:
            scales.append(ss)

        gen = torch.Generator(gen_device).manual_seed(data.seed)
        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt=data.negative_prompt,
            image=conditioning,
            controlnet_conditioning_scale=scales,
            guidance_scale=cfg,
            num_inference_steps=8,
            generator=gen,
        )
        img = result.images[0]
        img = _match_color_statistics(img, data.target_brightness, data.target_saturation)
        dt = time.time() - t0

        img_resized = img.convert("RGB").resize((512, 512), Image.LANCZOS)
        s = _stats(np.array(img_resized), src_arr)
        psnr = "inf" if not np.isfinite(s["psnr_db"]) else f"{s['psnr_db']:.2f}"
        print(f"{label:38s}  {s['mse']:>10.2f}  {psnr:>8s}  {s['mae']:>7.2f}  {dt:>5.1f}s")

        if best is None or s["mse"] < best[1]:
            best = (label, s["mse"], s["psnr_db"], s["mae"])

    print("-" * 80)
    if best:
        psnr = "inf" if not np.isfinite(best[2]) else f"{best[2]:.2f}"
        print(f"BEST: {best[0]}  MSE={best[1]:.2f}  PSNR={psnr}  MAE={best[3]:.2f}")

    del pipe
    from brainimg.device import free_torch
    free_torch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
