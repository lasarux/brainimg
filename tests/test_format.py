"""Tests for the .brainimg file format.

These run without any ML dependencies (pure stdlib). They verify the schema,
round-trip save/load, and rejection of malformed files.
"""

import base64
import json
from pathlib import Path

import pytest

from brainimg.format import (
    SCHEMA_VERSION,
    BrainimgData,
    BrainimgError,
    load_brainimg,
    save_brainimg,
    validate,
)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _make_data(**overrides) -> BrainimgData:
    base = dict(
        format_version=SCHEMA_VERSION,
        original_width=512,
        original_height=512,
        prompt="a red apple on a wooden table",
        depth_map_b64=_b64(b"\xff\xd8\xff\xe0fake-depth"),
        canny_map_b64=_b64(b"\x89PNGfake-canny"),
        seed=42,
    )
    base.update(overrides)
    return BrainimgData(**base)


def test_valid_data_passes_validation():
    validate(_make_data())


def test_save_load_roundtrip(tmp_path: Path):
    data = _make_data(prompt="a blue sphere on glass", seed=7)
    fp = tmp_path / "out.brainimg"
    n = save_brainimg(fp, data)
    assert n > 0
    assert fp.exists()

    loaded = load_brainimg(fp)
    assert loaded.prompt == "a blue sphere on glass"
    assert loaded.seed == 7
    assert loaded.original_width == 512
    assert loaded.depth_map_b64 == data.depth_map_b64
    assert loaded.canny_map_b64 == data.canny_map_b64
    assert loaded.format_version == SCHEMA_VERSION


def test_unknown_fields_preserved_in_extra(tmp_path: Path):
    data = _make_data()
    fp = tmp_path / "out.brainimg"
    save_brainimg(fp, data)
    # Inject an unknown forward-compat field directly into the JSON.
    doc = json.loads(fp.read_text())
    doc["future_field"] = "hello"
    fp.write_text(json.dumps(doc))

    loaded = load_brainimg(fp)
    assert loaded.extra.get("future_field") == "hello"


def test_wrong_version_rejected():
    with pytest.raises(BrainimgError, match="format_version"):
        validate(_make_data(format_version="9.9"))


def test_missing_field_rejected():
    data = _make_data()
    obj = json.loads(json.dumps(data.__dict__))
    del obj["prompt"]
    with pytest.raises(BrainimgError, match="prompt"):
        validate(obj)


def test_nonpositive_dims_rejected():
    with pytest.raises(BrainimgError, match="original_width/height"):
        validate(_make_data(original_width=0))
    with pytest.raises(BrainimgError, match="original_width/height"):
        validate(_make_data(original_height=-1))


def test_empty_prompt_rejected():
    with pytest.raises(BrainimgError, match="prompt"):
        validate(_make_data(prompt="   "))


def test_negative_seed_rejected():
    with pytest.raises(BrainimgError, match="seed"):
        validate(_make_data(seed=-5))


def test_bad_base64_rejected():
    with pytest.raises(BrainimgError, match="depth_map_b64"):
        validate(_make_data(depth_map_b64="%%%not-base64%%%"))


def test_bad_json_rejected(tmp_path: Path):
    fp = tmp_path / "bad.brainimg"
    fp.write_text("{not json")
    with pytest.raises(BrainimgError, match="not valid JSON"):
        load_brainimg(fp)


def test_size_bytes_positive():
    data = _make_data()
    assert data.size_bytes > 100
    assert data.size_bytes < 5_000  # maps are tiny fake blobs in tests
