"""Per-region color analysis: segment the source by color regions, then
measure brightness/saturation/hue per region across all reconstructions.

Uses the ADE20K segmentation map stored in lenna.brainimg as a mask. Groups
adjacent ADE20K class colors into larger semantic regions (skin, hair, face,
background, fabric) by hand-tuned color matching -- the seg map is palette
colorized, so a quick nearest-color lookup is enough for Lenna.

Output: a table per reconstruction showing how each region's hue/brightness/
saturation compares to the source.
"""
from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image

from brainimg.format import load_brainimg

# ADE20K palette anchors (rough hand-picked centers from the swin-tiny palette).
# These give us a coarse semantic grouping. Each is a (name, rgb-triplet).
REGION_ANCHORS: list[tuple[str, tuple[int, int, int]]] = [
    ("skin",    (180, 130, 110)),   # ADE20K skin-ish tones
    ("hair",    (60,  40,  90)),    # dark hair
    ("face",    (200, 160, 140)),   # lighter skin/face
    ("background", (140, 160, 200)),  # blue/gray sky or backdrop
    ("fabric",  (110, 80,  70)),    # clothing
    ("hat",     (90,  60,  50)),    # dark accessories
    ("plant",   (60,  110, 50)),    # green vegetation
    ("building",(160, 160, 150)),   # neutral structure
]


def _classify_region(rgb: tuple[int, int, int]) -> str:
    best, best_d = "unknown", 1e18
    for name, ref in REGION_ANCHORS:
        d = sum((a - b) ** 2 for a, b in zip(rgb, ref))
        if d < best_d:
            best_d, best = d, name
    return best


def _seg_to_mask(seg_img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Return an HxW uint8 label map where each pixel is a region index.

    The seg map is downscaled to *size* with NEAREST (palette colors must
    stay crisp) and each pixel is mapped to its nearest region anchor.
    Returns the region *name* per pixel as a string array of shape (H, W).
    """
    seg = np.array(seg_img.convert("RGB").resize(size, Image.NEAREST))
    h, w, _ = seg.shape
    flat = seg.reshape(-1, 3).astype(np.int32)
    # Vectorized nearest-anchor classification.
    names = np.empty(flat.shape[0], dtype=object)
    for i, px in enumerate(flat):
        names[i] = _classify_region(tuple(int(c) for c in px))
    return names.reshape(h, w)


def _region_stats(img: Image.Image, mask: np.ndarray, region: str) -> dict | None:
    sel = mask == region
    if sel.sum() < 100:
        return None
    crop = img.crop((0, 0, img.width, img.height))  # no-op, just for typing
    arr = np.array(crop)[sel]
    brightness = float((0.299 * arr[:, 0] + 0.587 * arr[:, 1] + 0.114 * arr[:, 2]).mean())
    maxc = arr.max(1)
    minc = arr.min(1)
    sat = float(np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1.0), 0.0).mean() * 255.0)
    # Hue: skip near-gray pixels.
    sat_ch = arr.max(1).astype(np.float32) - arr.min(1).astype(np.float32)
    usable = sat_ch >= 5.0
    if not usable.any():
        hue = -1.0
    else:
        pixels = arr[usable].astype(np.uint8).reshape(-1, 1, 3)
        hsv = np.array(Image.fromarray(pixels, "RGB").convert("HSV"))
        hue_rad = np.deg2rad(hsv[:, 0, 0] * 360.0 / 255.0)
        x = float(np.cos(hue_rad).mean())
        y = float(np.sin(hue_rad).mean())
        if x == 0.0 and y == 0.0:
            hue = -1.0
        else:
            hue = math.degrees(math.atan2(y, x)) % 360.0
    return {
        "n": int(sel.sum()),
        "brightness": round(brightness, 1),
        "saturation": round(sat, 1),
        "hue": round(hue, 1) if hue >= 0 else -1.0,
        "mean_rgb": tuple(int(c) for c in arr.mean(0)),
    }


def _circular_delta(a: float, b: float) -> float:
    """Smallest signed delta from a to b, in [-180, 180]."""
    if a < 0 or b < 0:
        return float("nan")
    return round((b - a + 540.0) % 360.0 - 180.0, 1)


def main() -> int:
    src = Image.open("samples/lenna.tiff").convert("RGB")
    src_512 = src.resize((512, 512), Image.LANCZOS)

    bi = load_brainimg("lenna.brainimg")
    if not bi.segmentation_map_b64:
        print("lenna.brainimg has no segmentation_map_b64; nothing to analyze.")
        return 1
    seg_img = Image.open(io.BytesIO(base64.b64decode(bi.segmentation_map_b64)))

    mask = _seg_to_mask(seg_img, src_512.size)
    regions_present = sorted(set(mask.ravel()))
    print(f"source size: {src_512.size}, regions in seg: {len(regions_present)}")
    print("  region pixel counts: " + ", ".join(
        f"{r}={int((mask == r).sum())}" for r in regions_present
    ))

    decodes = [
        ("SD15  512 existing",  "lenna_recon.png"),
        ("SDXL 1024 existing",  "lenna_sdxl.png"),
        ("SDXL  512 existing",  "lenna_sdxl_512.png"),
        ("SDXL  512 fixed-prom", "lenna_sdxl_fixed_prompt.png"),
        ("SDXL  512 hue-fix",   "lenna_sdxl_512_with_hue.png"),
    ]

    print()
    print("Per-region stats vs source. Delta hue is signed (target - source).")
    print("-" * 90)

    for label, path in decodes:
        if not Path(path).exists():
            print(f"{label}: missing")
            continue
        img = Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS)
        print(f"\n{label}  ({path})")
        for region in regions_present:
            src_s = _region_stats(src_512, mask, region)
            out_s = _region_stats(img, mask, region)
            if src_s is None or out_s is None:
                continue
            dh = _circular_delta(src_s["hue"], out_s["hue"])
            print(
                f"  {region:11s} n={src_s['n']:6d}  "
                f"src b={src_s['brightness']:5.1f} s={src_s['saturation']:5.1f} "
                f"hue={src_s['hue']:6.1f}  rgb={src_s['mean_rgb']}  |  "
                f"out b={out_s['brightness']:5.1f} s={out_s['saturation']:5.1f} "
                f"hue={out_s['hue']:6.1f}  rgb={out_s['mean_rgb']}  |  "
                f"dH={dh:+6.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
