"""brainimg: semantic image compression.

Stores an image as a tiny blueprint (caption + depth map + Canny edges + seed)
and regenerates it with Stable Diffusion + ControlNet on decode.
"""

from .format import SCHEMA_VERSION, BrainimgData, load_brainimg, save_brainimg, validate

__all__ = [
    "BrainimgData",
    "load_brainimg",
    "save_brainimg",
    "validate",
    "SCHEMA_VERSION",
]
__version__ = "0.1.0"
