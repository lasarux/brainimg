"""Build per-sample grids of original + all backend reconstructions.

Encodes each SIPI sample to a .brainimg blueprint (seed 200), decodes it
through every backend in ``compare_backends.DEFAULT_BACKENDS`` to a
``<sample>_<backend>.png`` reconstruction, then assembles a grid via
``scripts/make_backend_grid.py`` and moves it to ``docs/grids/``.

Designed to be resumable: each step is idempotent and skips itself if its
output already exists, so an interrupted ~10-hour run can be re-invoked
without redoing finished work. Per-step wall time is appended to
``grid_runs.log`` so progress is monitorable from another shell via
``tail -f grid_runs.log``.

Encoder and decoder are invoked as subprocesses (the decoder in
particular must be a separate process so model memory is fully released
between backends -- a long-lived Python process would accumulate GPU/RAM
fragmentation across 13 model loads).

Usage:
    python scripts/run_all_grids.py                 # all 4 samples
    python scripts/run_all_grids.py mandril          # one sample
    python scripts/run_all_grids.py mandril airplane # selected samples
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "outputs" / "grid_runs.log"
GRIDS_DIR = ROOT / "docs" / "grids"
SAMPLES_DIR = ROOT / "samples"
OUTPUTS_DIR = ROOT / "outputs"

# (sample name, source filename). Resolved to match compare_backends._find_source.
SIPI_SAMPLES: list[tuple[str, str]] = [
    ("mandril", "mandril_color.tif"),
    ("peppers", "peppers_color.tif"),
    ("cameraman", "cameraman.tif"),
    ("airplane", "airplane.tif"),
]

SEED = 200  # matches the verified-results table in README.md

# Backend -> decoder flag mapping. Order matches DEFAULT_BACKENDS in
# compare_backends.py. FLUX variants get --quantize (FP8 per the table);
# SDXL at native 1024x1024 (its training resolution -- smaller sizes drift,
# see README "SDXL hue drift at small sizes"); everything else uses
# decoder.py defaults (which already pick per-backend max_side = 1024
# via _model_config, so most backends generate at 1024 then downscale to
# the requested output size only when --size is passed).
#
# Note: make_backend_grid.py resizes each reconstruction to 512x512 for the
# grid layout regardless of decode resolution, so decoding SDXL at 1024
# does not change the grid cell size -- it just gives SDXL its native-res
# fidelity advantage that the verified-results table reports.
BACKEND_FLAGS: list[tuple[str, list[str]]] = [
    ("sd15", []),
    ("sd15-turbo", ["--model", "sd15-turbo"]),
    ("sdxl", ["--model", "sdxl", "--size", "1024x1024"]),
    ("sdxl-turbo", ["--model", "sdxl-turbo"]),
    ("zimage", ["--model", "zimage"]),
    ("qwen-image", ["--model", "qwen-image"]),
    ("sana", ["--model", "sana"]),
    ("flux2-klein", ["--model", "flux2-klein"]),
    ("flux-depth", ["--model", "flux-depth", "--quantize"]),
    ("flux-depth-turbo", ["--model", "flux-depth-turbo", "--quantize"]),
    ("flux-canny", ["--model", "flux-canny", "--quantize"]),
    ("flux-union", ["--model", "flux-union", "--quantize"]),
    ("sd35", ["--model", "sd35"]),
]


def log(msg: str) -> None:
    """Append a timestamped line to grid_runs.log and echo to stdout."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with LOG.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run(cmd: list[str], label: str) -> int:
    """Run a subprocess, streaming stdout/stderr to grid_runs.log.

    Returns the exit code. Raises SystemExit on non-zero exit so the
    orchestrator halts on the first failure (resumability comes from
    the skip-if-exists checks, not from continuing past errors).
    """
    start = time.time()
    log(f"BEGIN {label}: {' '.join(cmd)}")
    with LOG.open("a") as f:
        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
        )
    dt = time.time() - start
    status = "OK" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
    log(f"END   {label}: {status}  {dt:.1f}s")
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed (exit {proc.returncode}); see {LOG}")
    return proc.returncode


def encode_sample(sample_name: str, src_filename: str) -> Path:
    """Encode sample -> outputs/<sample>.brainimg (seed 200). Skips if it exists."""
    src = SAMPLES_DIR / src_filename
    out = OUTPUTS_DIR / f"{sample_name}.brainimg"
    if out.exists():
        log(f"SKIP encode {sample_name}: {out} exists")
        return out
    if not src.exists():
        raise SystemExit(f"source image not found: {src}")
    run(
        [sys.executable, "src/encoder.py", str(src), "-o", str(out), "--seed", str(SEED)],
        f"encode {sample_name}",
    )
    return out


def decode_backend(sample_name: str, brainimg: Path, backend: str, flags: list[str]) -> Path:
    """Decode <sample>.brainimg through one backend -> outputs/<sample>_<backend>.png.

    Skips if the output PNG already exists (resumable).
    """
    out = OUTPUTS_DIR / f"{sample_name}_{backend}.png"
    if out.exists():
        log(f"SKIP decode {sample_name}/{backend}: {out} exists")
        return out
    cmd = [
        sys.executable, "src/decoder.py",
        str(brainimg),
        "-o", str(out),
        "--device", "cpu",
        *flags,
    ]
    run(cmd, f"decode {sample_name}/{backend}")
    return out


def build_grid(sample_name: str) -> Path:
    """Run make_backend_grid.py for the sample, then move the grid to docs/grids/."""
    root_grid = ROOT / f"{sample_name}_grid.jpg"
    dest = GRIDS_DIR / f"{sample_name}_grid.jpg"
    if dest.exists():
        log(f"SKIP grid {sample_name}: {dest} exists")
        return dest
    run(
        [
            sys.executable, "scripts/make_backend_grid.py",
            sample_name, "--size", "512",
        ],
        f"grid {sample_name}",
    )
    if not root_grid.exists():
        raise SystemExit(f"make_backend_grid.py did not produce {root_grid}")
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(root_grid), str(dest))
    log(f"MOVED {root_grid.name} -> {dest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    """Orchestrate encode -> decode-all-backends -> grid for each sample."""
    requested = argv if argv else [name for name, _ in SIPI_SAMPLES]
    samples = [(n, f) for n, f in SIPI_SAMPLES if n in requested]
    if not samples:
        raise SystemExit(
            f"no matching samples; known: {[n for n,_ in SIPI_SAMPLES]}"
        )
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    log(f"=== run_all_grids start: {len(samples)} sample(s), "
        f"{len(BACKEND_FLAGS)} backends each ===")
    total = len(samples) * len(BACKEND_FLAGS)
    log(f"=== up to {total} decoder runs (skips existing outputs) ===")
    t_start = time.time()

    for sample_name, src_filename in samples:
        log(f"--- sample: {sample_name} ({src_filename}) ---")
        brainimg = encode_sample(sample_name, src_filename)
        for backend, flags in BACKEND_FLAGS:
            decode_backend(sample_name, brainimg, backend, flags)
        build_grid(sample_name)

    dt = time.time() - t_start
    log(f"=== run_all_grids done: {dt:.0f}s ({dt/3600:.2f}h) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
