"""The .brainimg file format.

A `.brainimg` file is a small JSON document that stores the *meaning* and
*structure* of an image instead of its pixels:

    {
      "format_version": "0.1",
      "caption_model": "qwen2-vl-2b-4bit",
      "original_width": 1024,
      "original_height": 768,
      "prompt": "a red apple on a wooden table next to a window",
      "negative_prompt": "blurry, low quality, deformed",
      "depth_map_b64": "<base64 64x64 JPEG>",
      "canny_map_b64": "<base64 64x64 PNG>",
      "seed": 42,
      "steps": 20
    }

This module is intentionally free of any ML/torch/mlx imports so the format
can be round-trip tested without downloading models.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1"

DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, deformed, watermark, jpeg artifacts"
DEFAULT_STEPS = 20
DEFAULT_CAPTION_MODEL = "qwen2-vl-2b-4bit"

# Downscaled conditioning map resolution. Larger -> better structural fidelity
# at the cost of a slightly bigger file. 128 keeps the file in the low-KB range
# while noticeably improving reconstruction quality over 64.
MAP_SIZE = 128

REQUIRED_FIELDS = (
    "format_version",
    "original_width",
    "original_height",
    "prompt",
    "depth_map_b64",
    "canny_map_b64",
    "seed",
)


class BrainimgError(ValueError):
    """Raised when a .brainimg file is malformed or fails validation."""


@dataclass
class BrainimgData:
    format_version: str
    original_width: int
    original_height: int
    prompt: str
    depth_map_b64: str
    canny_map_b64: str
    seed: int
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    steps: int = DEFAULT_STEPS
    caption_model: str = DEFAULT_CAPTION_MODEL
    # Color statistics of the original image, used by the decoder to
    # post-process the generation so its brightness/saturation matches the
    # source (SD 1.5 tends to produce oversaturated, too-bright images).
    target_brightness: float = 0.0
    target_saturation: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        """Approximate on-disk size of the serialized file (UTF-8 bytes)."""
        return len(json.dumps(asdict(self), ensure_ascii=False).encode("utf-8"))


def validate(data: BrainimgData | dict[str, Any]) -> None:
    """Check that *data* satisfies the brainimg v0.1 schema.

    Raises :class:`BrainimgError` on any problem.
    """
    d = asdict(data) if isinstance(data, BrainimgData) else dict(data)

    if d.get("format_version") != SCHEMA_VERSION:
        raise BrainimgError(
            f"unsupported format_version: {d.get('format_version')!r} "
            f"(expected {SCHEMA_VERSION!r})"
        )

    for name in REQUIRED_FIELDS:
        if name not in d:
            raise BrainimgError(f"missing required field: {name!r}")
        if d[name] is None:
            raise BrainimgError(f"required field is None: {name!r}")

    w, h = d["original_width"], d["original_height"]
    if not (isinstance(w, int) and w > 0 and isinstance(h, int) and h > 0):
        raise BrainimgError(f"original_width/height must be positive ints, got {w!r}, {h!r}")

    if not isinstance(d["prompt"], str) or not d["prompt"].strip():
        raise BrainimgError("prompt must be a non-empty string")

    seed = d["seed"]
    if not isinstance(seed, int) or seed < 0:
        raise BrainimgError(f"seed must be a non-negative int, got {seed!r}")

    steps = d.get("steps", DEFAULT_STEPS)
    if not isinstance(steps, int) or steps <= 0:
        raise BrainimgError(f"steps must be a positive int, got {steps!r}")

    # base64 maps must be decodable.
    for name in ("depth_map_b64", "canny_map_b64"):
        val = d[name]
        if not isinstance(val, str) or not val:
            raise BrainimgError(f"{name} must be a non-empty base64 string")
        try:
            decoded = base64.b64decode(val, validate=True)
        except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
            raise BrainimgError(f"{name} is not valid base64: {exc}") from exc
        if not decoded:
            raise BrainimgError(f"{name} decodes to empty bytes")


def save_brainimg(path: str | Path, data: BrainimgData) -> int:
    """Validate *data* and write it to *path* as JSON. Returns bytes written."""
    validate(data)
    payload = json.dumps(asdict(data), ensure_ascii=False, indent=2)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload, encoding="utf-8")
    return len(payload.encode("utf-8"))


def load_brainimg(path: str | Path) -> BrainimgData:
    """Read and validate a .brainimg file, returning a :class:`BrainimgData`."""
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise BrainimgError(f"could not read {path!r}: {exc}") from exc

    try:
        d = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BrainimgError(f"{path!r} is not valid JSON: {exc}") from exc

    if not isinstance(d, dict):
        raise BrainimgError("top-level brainimg document must be a JSON object")

    # Pull known fields; stash anything unknown into `extra` for forward-compat.
    known = {f for f in BrainimgData.__dataclass_fields__} - {"extra"}
    fields_in: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for k, v in d.items():
        if k in known:
            fields_in[k] = v
        else:
            extra[k] = v

    fields_in["extra"] = extra
    try:
        data = BrainimgData(**fields_in)
    except TypeError as exc:
        raise BrainimgError(f"schema mismatch in {path!r}: {exc}") from exc

    validate(data)
    return data
