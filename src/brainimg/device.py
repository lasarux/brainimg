"""Device, dtype, and memory helpers for the PyTorch + MLX stack.

Centralizes the cuda/mps/cpu decision so encoder/decoder don't repeat it, and
provides memory-release helpers for the 8 GB Apple Silicon target where the
encoder's MLX VLM and the decoder's SD pipeline must never overlap.
"""

from __future__ import annotations

import gc

# torch / mlx are imported lazily inside functions so that `format.py` and its
# tests never need them.


def get_torch_device() -> str:
    """Return the best available torch device name: 'cuda', 'mps', or 'cpu'."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str | None = None):
    """fp16 on GPU devices, fp32 on CPU."""
    import torch

    dev = device or get_torch_device()
    return torch.float16 if dev in ("cuda", "mps") else torch.float32


def free_torch() -> None:
    """Release PyTorch GPU/MPS memory."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def free_mlx() -> None:
    """Release MLX Metal memory."""
    gc.collect()
    try:
        import mlx.core as mx

        clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
        if clear is not None:
            clear()
    except Exception:
        # mlx not installed or not on Apple Silicon -- nothing to do.
        pass


def device_info() -> dict[str, str]:
    """Small summary for logging."""
    return {"torch_device": get_torch_device()}
