"""Generate a small bundled test image: a simple scene the captioner can describe.

Produces samples/apple.jpg -- a stylized scene of a red apple on a wooden table
near a window with warm light. Synthetic, royalty-free, deterministic.

Usage:
    python scripts/make_sample.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "samples" / "apple.jpg"
W, H = 512, 512


def _vertical_gradient(draw, w, h, top, bottom):
    for y in range(h):
        t = y / (h - 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))


def main() -> None:
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)

    # Sky / wall: warm gradient (top warm light, bottom dimmer).
    _vertical_gradient(draw, W, H, (250, 215, 160), (170, 130, 95))

    # Window of light: a soft bright rectangle top-right.
    draw.rectangle([360, 70, 470, 230], fill=(255, 245, 200), outline=(255, 255, 255))

    # Wooden table: a horizontal brown band.
    table_top = 360
    draw.rectangle([0, table_top, W, H], fill=(120, 75, 40))
    draw.rectangle([0, table_top, W, table_top + 8], fill=(150, 95, 55))

    # Apple: red body + green stem + leaf.
    apple_cx, apple_cy, apple_r = 200, 330, 60
    draw.ellipse(
        [apple_cx - apple_r, apple_cy - apple_r, apple_cx + apple_r, apple_cy + apple_r],
        fill=(200, 30, 30),
        outline=(120, 10, 10),
    )
    # Highlight.
    draw.ellipse([apple_cx - 35, apple_cy - 40, apple_cx - 10, apple_cy - 15], fill=(240, 90, 90))
    # Stem.
    draw.rectangle(
        [apple_cx - 4, apple_cy - apple_r - 18, apple_cx + 4, apple_cy - apple_r],
        fill=(80, 50, 20),
    )
    # Leaf.
    draw.ellipse(
        [apple_cx + 4, apple_cy - apple_r - 22, apple_cx + 34, apple_cy - apple_r + 2],
        fill=(40, 130, 40),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, format="JPEG", quality=88)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
