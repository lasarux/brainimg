"""Tests for the FLUX decoder backend config.

Pure dict-shape tests -- no model download required. Verifies that
_model_config("flux-depth"|"flux-canny") returns the keys that
_generate_flux / _build_flux_pipeline depend on, and that an
unknown FLUX variant fails loudly.
"""
from __future__ import annotations

from brainimg.generate import (
    FLUX_CANNY_GUIDANCE_SCALE,
    FLUX_CANNY_MODEL_ID,
    FLUX_DEFAULT_STEPS,
    FLUX_DEPTH_GUIDANCE_SCALE,
    FLUX_DEPTH_MODEL_ID,
    FLUX_MAX_DEFAULT_SIDE,
    FLUX_MAX_TOKENS,
    _model_config,
)


def test_flux_depth_config_shape():
    cfg = _model_config("flux-depth")
    assert cfg["kind"] == "flux"
    assert cfg["base_id"] == FLUX_DEPTH_MODEL_ID
    assert cfg["base_id"].endswith("FLUX.1-Depth-dev")
    assert cfg["control_source"] == "depth"
    assert "control_scale" not in cfg  # channel-concat has no per-call scale
    assert cfg["guidance"] == FLUX_DEPTH_GUIDANCE_SCALE
    assert cfg["max_side"] == FLUX_MAX_DEFAULT_SIDE == 1024
    assert cfg["default_steps"] == FLUX_DEFAULT_STEPS == 30
    assert cfg["max_tokens"] == FLUX_MAX_TOKENS == 512


def test_flux_canny_config_shape():
    cfg = _model_config("flux-canny")
    assert cfg["kind"] == "flux"
    assert cfg["base_id"] == FLUX_CANNY_MODEL_ID
    assert cfg["base_id"].endswith("FLUX.1-Canny-dev")
    assert cfg["control_source"] == "canny"
    assert "control_scale" not in cfg
    # Canny variant wants higher CFG than depth (model card / community).
    assert cfg["guidance"] == FLUX_CANNY_GUIDANCE_SCALE
    assert cfg["guidance"] > FLUX_DEPTH_GUIDANCE_SCALE
    assert cfg["max_side"] == 1024
    assert cfg["default_steps"] == 30
    assert cfg["max_tokens"] == 512


def test_flux_variants_are_distinct():
    """The depth and canny variants must differ in base_id and control_source."""
    a = _model_config("flux-depth")
    b = _model_config("flux-canny")
    assert a["base_id"] != b["base_id"]
    assert a["control_source"] != b["control_source"]
    assert a["guidance"] != b["guidance"]


def test_flux_variants_dont_have_sd_specific_keys():
    """FLUX configs should not carry SD 1.5/SDXL-only fields like depth_id/seg_id."""
    for variant in ("flux-depth", "flux-canny"):
        cfg = _model_config(variant)
        for sd_key in ("depth_id", "canny_id", "seg_id", "vae_id", "seg_weight_name"):
            assert sd_key not in cfg, f"FLUX {variant} should not carry {sd_key!r}"


def test_unknown_model_falls_back_to_sd15():
    """Unrecognized model names default to the SD 1.5 stack (back-compat).

    ``argparse`` is responsible for rejecting truly unknown --model values at
    the CLI level; the function-level fallback is a safety net for
    programmatic callers.
    """
    cfg = _model_config("not-a-real-model")
    assert cfg["base_id"].endswith("stable-diffusion-v1-5")
