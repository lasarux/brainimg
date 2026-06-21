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
    FLUX_TURBO_DEFAULT_STEPS,
    FLUX_TURBO_GUIDANCE_SCALE,
    FLUX_TURBO_LORA_FILE,
    HYPER_SD_REPO,
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


def test_turbo_models_reuse_sd_stack():
    """sd15-turbo / sdxl-turbo must reuse the SD 1.5 / SDXL base + ControlNets.

    Hyper-SD is a LoRA on top of the existing stack, not a new base -- so the
    turbo configs must carry the same depth_id / canny_id / seg_id as their
    non-turbo siblings, plus a ``turbo=True`` flag and the LoRA file metadata.
    """
    sd15 = _model_config("sd15")
    sd15_turbo = _model_config("sd15-turbo")
    for stack_key in ("base_id", "vae_id", "depth_id", "canny_id", "seg_id"):
        assert sd15_turbo[stack_key] == sd15[stack_key]
    assert sd15_turbo["turbo"] is True
    assert sd15["turbo"] is False
    assert sd15_turbo["turbo_lora_repo"] == "ByteDance/Hyper-SD"
    assert sd15_turbo["turbo_lora_file"].endswith(".safetensors")
    assert sd15_turbo["default_steps"] == 8

    sdxl = _model_config("sdxl")
    sdxl_turbo = _model_config("sdxl-turbo")
    for stack_key in ("base_id", "vae_id", "depth_id", "canny_id", "seg_id"):
        assert sdxl_turbo[stack_key] == sdxl[stack_key]
    assert sdxl_turbo["turbo"] is True
    assert sdxl["turbo"] is False
    assert sdxl_turbo["default_steps"] == 8


def test_turbo_guidance_matches_non_turbo_defaults():
    """Hyper-SD CFG-preserved LoRAs support the same 5-8 guidance range as
    the non-turbo SD stacks, so the turbo defaults must equal the non-turbo
    defaults (7.5 SD 1.5, 7.0 SDXL).
    """
    assert _model_config("sd15-turbo")["guidance"] == _model_config("sd15")["guidance"]
    assert _model_config("sdxl-turbo")["guidance"] == _model_config("sdxl")["guidance"]


def test_all_model_configs_carry_turbo_flag():
    """Every _model_config entry must set ``turbo`` so downstream code can
    do ``cfg.get("turbo")`` without None ambiguity. The non-turbo / non-SD
    paths must set it to False explicitly.
    """
    for model in (
        "sd15", "sd15-turbo", "sdxl", "sdxl-turbo", "zimage",
        "flux-depth", "flux-canny", "flux-depth-turbo", "flux-canny-turbo",
    ):
        cfg = _model_config(model)
        assert "turbo" in cfg, f"{model} config missing turbo key"
        assert isinstance(cfg["turbo"], bool)


def test_flux_turbo_reuses_flux_base():
    """flux-depth-turbo / flux-canny-turbo must reuse the same FLUX.1-*-dev
    base + control_source as their non-turbo siblings, with turbo=True
    + the Hyper-SD FLUX LoRA file metadata + 8 steps + guidance 3.5.
    """
    for base, turbo in [("flux-depth", "flux-depth-turbo"), ("flux-canny", "flux-canny-turbo")]:
        base_cfg = _model_config(base)
        turbo_cfg = _model_config(turbo)
        assert turbo_cfg["base_id"] == base_cfg["base_id"]
        assert turbo_cfg["control_source"] == base_cfg["control_source"]
        assert turbo_cfg["kind"] == "flux"
        assert turbo_cfg["turbo"] is True
        assert base_cfg["turbo"] is False
        assert turbo_cfg["turbo_lora_repo"] == HYPER_SD_REPO
        assert turbo_cfg["turbo_lora_file"] == FLUX_TURBO_LORA_FILE
        assert turbo_cfg["turbo_lora_scale"] == 0.125
        assert turbo_cfg["default_steps"] == FLUX_TURBO_DEFAULT_STEPS == 8
        assert turbo_cfg["guidance"] == FLUX_TURBO_GUIDANCE_SCALE == 3.5


def test_flux_turbo_guidance_differs_from_non_turbo():
    """FLUX turbo uses guidance 3.5 (the dev default per Hyper-SD card),
    which is much lower than the non-turbo control defaults (10.0 depth,
    30.0 canny). The SD turbo paths keep the same guidance as non-turbo;
    FLUX turbo does not."""
    assert _model_config("flux-depth-turbo")["guidance"] != _model_config("flux-depth")["guidance"]
    assert _model_config("flux-canny-turbo")["guidance"] != _model_config("flux-canny")["guidance"]
    assert _model_config("flux-depth-turbo")["guidance"] == 3.5
    assert _model_config("flux-canny-turbo")["guidance"] == 3.5
