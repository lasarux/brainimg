"""The encoder: extract a semantic + structural blueprint from an image.

Four extractors run in sequence (memory is released between heavy stages so an
8 GB Apple Silicon Mac can cope):

  1. Captioning   -- Qwen2.5-VL-7B via MLX (Apple) or transformers (CPU/CUDA)
                                                                -> text prompt
  2. Depth map    -- Depth-Anything-V2-Base   (transformers)  -> 128x128 grayscale
  3. Canny edges  -- OpenCV                                    -> 128x128 binary edges
  4. Segmentation -- OneFormer ADE20K (colorized)              -> 128x128 palette PNG

The result is a :class:`brainimg.format.BrainimgData` ready to save as a
tiny ``.brainimg`` file.
"""

from __future__ import annotations

import base64
import io
import random
from pathlib import Path

from PIL import Image

from .device import free_mlx, free_torch, get_torch_device, get_dtype
from .format import (
    DEFAULT_CAPTION_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_STEPS,
    MAP_SIZE,
    BrainimgData,
)

CAPTION_MODEL_ID = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
CAPTION_MODEL_ID_TORCH = "Qwen/Qwen2.5-VL-7B-Instruct"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
SEG_MODEL_ID = "shi-labs/oneformer_ade20k_swin_tiny"

CAPTION_INSTRUCTION = (
    "In one short sentence, describe this image for reconstruction: "
    "main objects, layout, lighting, and colors."
)


def _mlx_available() -> bool:
    """True if the MLX runtime + mlx-vlm can actually import on this machine.

    MLX is Apple-Silicon-only; on x86/x64 it ships a stub wheel whose
    ``libmlx.so`` is missing, so ``import mlx.core`` raises ImportError.
    """
    try:
        import mlx.core as mx  # noqa: F401
        import mlx_vlm  # noqa: F401

        return True
    except Exception:
        return False

# Map RGB mean -> color name (for the dominant-hue descriptor).
_COLOR_NAMES = [
    ((200, 200, 200), "white"),
    ((80, 80, 80), "black"),
    ((120, 120, 120), "gray"),
    ((150, 120, 90), "brown"),
    ((180, 140, 100), "tan"),
    ((140, 60, 50), "red"),
    ((160, 100, 60), "orange"),
    ((180, 170, 80), "yellow"),
    ((80, 130, 70), "green"),
    ((60, 100, 130), "blue"),
    ((100, 80, 130), "purple"),
    ((150, 130, 160), "pink"),
]

# ADE20K 150-class color palette. This is the exact palette the SD 1.5 seg
# ControlNet (lllyasviel/control_v11p_sd15_seg) was trained on, so the
# colorized OneFormer output matches its conditioning distribution. Inlined
# from controlnet_aux.util.ade_palette() to avoid importing controlnet_aux
# (which drags in timm/segment-anything) at runtime.
_ADE20K_PALETTE = [
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
    [102, 255, 0], [92, 0, 255],
]


def _name_color(rgb: tuple[float, float, float]) -> str:
    """Return the name of the closest reference color to *rgb*."""
    best, best_d = None, float("inf")
    for ref, name in _COLOR_NAMES:
        d = sum((a - b) ** 2 for a, b in zip(rgb, ref))
        if d < best_d:
            best_d, best = d, name
    return best or "neutral"


def extract_color_style(img: Image.Image) -> tuple[str, float, float]:
    """Return (style_prefix, brightness, saturation) for *img*.

    The style prefix is prepended to the caption so CLIP weights the mood
    (grayscale, dark, warm, etc.) first -- the front of the prompt has the
    strongest weight. brightness and saturation (0-255) are stored in the
    brainimg file so the decoder can post-process the generation to match.
    """
    import numpy as np

    arr = np.array(img.convert("RGB")).astype(float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = float((0.299 * r + 0.587 * g + 0.114 * b).mean())
    maxc = arr.max(2)
    minc = arr.min(2)
    sat = float(np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1.0), 0.0).mean() * 255.0)
    mean_rgb = tuple(arr.reshape(-1, 3).mean(0).round(1))

    parts: list[str] = []
    if sat < 30:
        parts.append("monochrome, grayscale, black and white photograph")
    elif sat < 85:
        parts.append("muted, desaturated tones")
    else:
        parts.append("vivid, saturated colors")

    if brightness < 85:
        parts.append("dark, low-key lighting")
    elif brightness > 175:
        parts.append("bright, high-key lighting")
    else:
        parts.append("natural lighting")

    if sat >= 30:
        parts.append(f"{_name_color(mean_rgb)} dominant tones")

    return ", ".join(parts), round(brightness, 1), round(sat, 1)


# --------------------------------------------------------------------------- #
# image <-> base64 helpers
# --------------------------------------------------------------------------- #
def image_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 70) -> str:
    buf = io.BytesIO()
    save_kwargs: dict = {"quality": quality}
    if fmt.upper() == "PNG":
        save_kwargs = {}
    img.save(buf, format=fmt, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# --------------------------------------------------------------------------- #
# 1. captioning (MLX on Apple Silicon, transformers Qwen2-VL elsewhere)
# --------------------------------------------------------------------------- #
def extract_caption(image_path: str | Path, max_tokens: int = 60) -> str:
    """Caption *image_path* with a Qwen2-VL vision-language model.

    Uses the MLX 4-bit model on Apple Silicon (fast, low memory) and falls
    back to the HuggingFace transformers Qwen2-VL-2B model on any other
    platform (x86/x64 CPUs, CUDA). Both produce an equivalent caption.
    """
    if _mlx_available():
        return _extract_caption_mlx(image_path, max_tokens)
    return _extract_caption_transformers(image_path, max_tokens)


def _extract_caption_mlx(image_path: str | Path, max_tokens: int = 60) -> str:
    """Run the MLX Qwen2-VL-4bit captioner on *image_path*."""
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    image_path = str(image_path)
    model, processor = load(CAPTION_MODEL_ID)
    config = load_config(CAPTION_MODEL_ID)

    images = [image_path]
    formatted = apply_chat_template(
        processor, config, CAPTION_INSTRUCTION, num_images=len(images)
    )

    output = generate(
        model,
        processor,
        formatted,
        images,
        max_tokens=max_tokens,
        verbose=False,
    )
    # mlx-vlm 0.6+ returns a GenerationResult; older versions returned a str.
    if hasattr(output, "text"):
        caption = output.text
    elif isinstance(output, str):
        caption = output
    else:
        caption = str(output)
    caption = caption.strip()

    # Release MLX memory before any PyTorch stage runs.
    del model, processor
    free_mlx()
    return caption


def _extract_caption_transformers(image_path: str | Path, max_tokens: int = 60) -> str:
    """Run the transformers Qwen2-VL captioner (CPU/CUDA fallback for non-MLX)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = get_torch_device()
    dtype = get_dtype(device)

    model = AutoModelForImageTextToText.from_pretrained(
        CAPTION_MODEL_ID_TORCH, dtype=dtype
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID_TORCH)

    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": CAPTION_INSTRUCTION},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=text, images=img, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    # Slice off the prompt tokens so we only decode the new caption.
    prompt_len = inputs["input_ids"].shape[1]
    generated = out[0, prompt_len:]
    caption = processor.decode(generated, skip_special_tokens=True).strip()

    del model, processor, inputs, out
    free_torch()
    return caption


# --------------------------------------------------------------------------- #
# 2. depth map (PyTorch + MPS)
# --------------------------------------------------------------------------- #
def extract_depth(img: Image.Image) -> str:
    """Return a base64 MAP_SIZExMAP_SIZE JPEG depth map (near=bright, far=dark)."""
    import torch
    from transformers import pipeline as hf_pipeline

    device = get_torch_device()
    dtype = get_dtype(device)
    pipe = hf_pipeline(
        "depth-estimation",
        model=DEPTH_MODEL_ID,
        device=device,
        torch_dtype=dtype,
    )
    result = pipe(img)
    depth: Image.Image = result["depth"]
    depth = depth.convert("L").resize((MAP_SIZE, MAP_SIZE), Image.LANCZOS)

    b64 = image_to_b64(depth, fmt="JPEG", quality=70)

    del pipe
    free_torch()
    return b64


# --------------------------------------------------------------------------- #
# 3. canny edges (OpenCV, CPU)
# --------------------------------------------------------------------------- #
def extract_canny(img: Image.Image, low: int = 50, high: int = 150) -> str:
    """Return a base64 MAP_SIZExMAP_SIZE PNG Canny edge map."""
    import cv2
    import numpy as np

    gray = np.array(img.convert("L"))
    gray_small = cv2.resize(gray, (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(gray_small, low, high)

    ok, buf = cv2.imencode(".png", edges)
    if not ok:
        raise RuntimeError("failed to encode Canny edge map")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# --------------------------------------------------------------------------- #
# 4. segmentation (OneFormer ADE20K -> colorized, for the seg ControlNet)
# --------------------------------------------------------------------------- #
def extract_segmentation(img: Image.Image) -> str:
    """Return a base64 MAP_SIZExMAP_SIZE PNG ADE20K colorized segmentation map.

    Runs OneFormer (ADE20K, Swin-Tiny) semantic segmentation, then maps each
    class id to its ADE20K palette color. The output matches the conditioning
    distribution of ``lllyasviel/control_v11p_sd15_seg``.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForUniversalSegmentation, AutoProcessor

    device = get_torch_device()
    dtype = get_dtype(device)

    processor = AutoProcessor.from_pretrained(SEG_MODEL_ID)
    model = AutoModelForUniversalSegmentation.from_pretrained(
        SEG_MODEL_ID, dtype=dtype
    ).to(device)
    model.eval()

    inputs = processor(images=img, task_inputs=["semantic"], return_tensors="pt").to(
        device
    )
    with torch.inference_mode():
        outputs = model(**inputs)

    seg = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[img.size[::-1]]
    )[0]
    seg_np = seg.cpu().numpy().astype(np.int64)
    h, w = seg_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in np.unique(seg_np):
        color[seg_np == cid] = _ADE20K_PALETTE[int(cid) % len(_ADE20K_PALETTE)]
    seg_img = Image.fromarray(color, "RGB").resize(
        (MAP_SIZE, MAP_SIZE), Image.NEAREST
    )

    buf = io.BytesIO()
    seg_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    del model, processor, outputs, seg
    free_torch()
    return b64


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def encode_image(image_path: str | Path, seed: int | None = None) -> BrainimgData:
    """Extract the full blueprint from *image_path* into a BrainimgData."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    # Stage 1: caption (MLX on Apple Silicon, transformers Qwen2.5-VL elsewhere).
    # Run first; the file path is needed because mlx-vlm reads images from disk/URL.
    caption = extract_caption(image_path)

    # Stages 2-4: depth + canny + segmentation. Reload the PIL image (captioner
    # may have closed file handles).
    img = Image.open(image_path).convert("RGB")
    color_style, target_brightness, target_saturation = extract_color_style(img)
    depth_b64 = extract_depth(img)
    canny_b64 = extract_canny(img)
    seg_b64 = extract_segmentation(img)

    # Use the raw caption as the prompt. Prepending the color style made the
    # prompt too long and CLIP truncated it (77-token limit). The color stats
    # are stored in the file separately for potential post-processing.
    prompt = caption

    return BrainimgData(
        format_version="0.1",
        caption_model=DEFAULT_CAPTION_MODEL,
        original_width=width,
        original_height=height,
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        depth_map_b64=depth_b64,
        canny_map_b64=canny_b64,
        segmentation_map_b64=seg_b64,
        seed=seed,
        steps=DEFAULT_STEPS,
        target_brightness=target_brightness,
        target_saturation=target_saturation,
    )
