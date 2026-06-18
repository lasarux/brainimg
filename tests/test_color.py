"""Tests for the decoder's color post-processing.

These run without any ML dependencies (only Pillow + numpy). They verify that
``_match_color_statistics`` moves an image's brightness/saturation toward the
stored targets and is a no-op for old files (targets == 0.0).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from brainimg.extract import brightness_saturation_of
from brainimg.generate import _match_color_statistics


def _solid(rgb: tuple[int, int, int], size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), rgb)


def _noisy(brightness: int, sat_spread: int, size: int = 96) -> Image.Image:
    """A synthetic image with a known brightness and varied saturation."""
    rng = np.random.default_rng(0)
    base = np.full((size, size, 3), brightness, dtype=np.float32)
    # Add per-channel noise so saturation > 0; scale controls the spread.
    noise = rng.normal(0, sat_spread, (size, size, 3))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def test_brightness_saturation_of_solid_gray():
    b, s = brightness_saturation_of(_solid((128, 128, 128)))
    assert b == 128.0
    assert s == 0.0  # gray has zero saturation


def test_brightness_saturation_of_solid_color():
    b, s = brightness_saturation_of(_solid((255, 0, 0)))
    # Rec.601 luminance of pure red (function rounds to 1 decimal).
    assert b == pytest.approx(0.299 * 255, abs=0.1)
    assert s == 255.0  # fully saturated


def test_match_is_noop_for_old_files():
    """Older v0.1 files have target_brightness/saturation == 0.0 -> no-op."""
    img = _noisy(180, 40)
    out = _match_color_statistics(img, 0.0, 0.0)
    assert out == img


def test_match_brightness_moves_toward_target():
    """A bright image should be darkened toward a target within the clamp range.

    The gain is clamped to [0.5, 2.0], so the target must be reachable from
    the source within that range. 160 -> 100 needs a 0.625 gain (in range).
    """
    bright = _noisy(160, 10)  # brightness ~160
    b0, _ = brightness_saturation_of(bright)
    assert b0 > 140

    out = _match_color_statistics(
        bright, target_brightness=100.0, target_saturation=0.0
    )
    b1, _ = brightness_saturation_of(out)
    assert b1 < b0  # moved toward target
    assert abs(b1 - 100.0) <= 3.0  # converged within tolerance


def test_match_brightness_respects_lower_clamp():
    """A target beyond the 0.5 gain clamp gets as close as the clamp allows."""
    bright = _noisy(210, 8)  # brightness ~210
    b0, _ = brightness_saturation_of(bright)
    # 80/210 = 0.38 -> clamped to 0.5 -> result ~105, not 80.
    out = _match_color_statistics(bright, target_brightness=80.0, target_saturation=0.0)
    b1, _ = brightness_saturation_of(out)
    assert b1 == pytest.approx(b0 * 0.5, abs=3.0)  # hit the clamp, not the target


def test_match_saturation_moves_toward_target():
    """A saturated image should be desaturated toward a target in clamp range.

    220 -> 120 needs a ~0.55 S-ratio (within [0.5, 2.0]).
    """
    sat = _solid((220, 30, 30))
    _, s0 = brightness_saturation_of(sat)
    assert s0 > 200

    out = _match_color_statistics(sat, target_brightness=0.0, target_saturation=120.0)
    _, s1 = brightness_saturation_of(out)
    assert s1 < s0  # moved toward target
    assert abs(s1 - 120.0) <= 25.0  # HSV-S scaling is approximate, converges roughly


def test_match_preserves_color_balance_for_brightness_only():
    """A uniform brightness gain must keep hue ratios (channel ratios) intact."""
    img = _solid((100, 60, 40))
    out = _match_color_statistics(img, target_brightness=200.0, target_saturation=0.0)
    arr = np.array(out, dtype=np.float32)
    # Each channel scaled by the same gain -> ratios preserved.
    r, g, b = arr[0, 0]
    assert abs(r / g - 100 / 60) < 1e-2
    assert abs(g / b - 60 / 40) < 1e-2


def test_match_already_close_is_noop():
    """If the image is already within the tolerance, no change is made."""
    img = _noisy(120, 20)
    b, s = brightness_saturation_of(img)
    out = _match_color_statistics(img, b, s)
    # Within the 2.0 tolerance -> identical pixels.
    assert np.array_equal(np.array(img), np.array(out))
