"""Patch lenna.brainimg's prompt and write to a sibling path.

The original lenna.brainimg was encoded with a caption that misidentifies
Lenna's hair as a 'wide-brimmed straw hat adorned with purple feathers'
(see TODO.md 'Captioner accuracy on Lenna'). We want to test whether the
SDXL decoder is being pulled toward that hallucination, so we rewrite the
prompt to something more faithful and decode only the patched file. The
original is untouched.
"""
from __future__ import annotations

from brainimg.format import load_brainimg, save_brainimg

src = "lenna.brainimg"
dst = "lenna_fixed_prompt.brainimg"

data = load_brainimg(src)
data.prompt = (
    "a portrait photograph of a young woman with shoulder-length wavy brown "
    "hair, fair skin, soft studio lighting, warm-toned blurred background, "
    "head and shoulders framing, photographic portrait"
)
n = save_brainimg(dst, data)
print(f"wrote {dst} ({n} bytes)")
print(f"  prompt: {data.prompt}")
