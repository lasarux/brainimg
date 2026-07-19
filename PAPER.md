<!-- Canonical source: PAPER.typ. This Markdown is a mirror for GitHub
     rendering; if it drifts, PAPER.typ wins. -->

---
title: "brainimg: A Reproducible Systems Study of Generative-Recall Image Compression"
authors:
  - name: Pedro A. Gracia Fajardo
    email: lasarux@gmail.com
date: 2026-07-19
abstract: |
  Classical image formats store pixels—either directly, as transform coefficients,
  or as latent codes—and are therefore resolution-bound and tied to the exact
  appearance of the original. This paper presents *brainimg*, a small prototype
  image format that stores the *meaning* of an image (a text caption) plus a
  tiny *structural blueprint* (128×128 depth, Canny-edge, and ADE20K
  segmentation maps) and a seed, and regenerates a visually faithful image on
  decode using one of fourteen pluggable diffusion decoder backends: Stable
  Diffusion 1.5 (default) or SDXL with two-to-three ControlNets, either of
  those plus ByteDance's Hyper-SD 8-step distilled LoRA (turbo), Z-Image-Turbo
  with a single Union ControlNet, Qwen-Image (Apache 2.0) with InstantX's
  Union ControlNet, HunyuanDiT v1.2 with separate depth +
  canny ControlNets, NVIDIA SANA 600M with an HED ControlNet fed the canny
  map, FLUX.2-klein-4B as an img2img pseudo-ControlNet fed the depth map, or
  FLUX.1-Depth-dev / FLUX.1-Canny-dev with channel-concat conditioning,
  optionally with Hyper-SD's 8-step FLUX LoRA. We frame brainimg
  as a working, reproducible instantiation of the "Semantic-Relational Field /
  generative-recall" paradigm: rather than compressing appearance, it stores
  the scene's semantics and geometry and lets a diffusion model repaint it. We
  describe the format schema, the four-stage encoder (VLM captioning, depth
  estimation, edge extraction, semantic segmentation), the fourteen decoder
  backends, the per-device memory/precision strategies, and the
  brightness/saturation post-processing with a gamma fallback for extreme
  targets. Using measurements reproducible from the committed repository on an
  AMD x86_64 CPU-only target with 188 GB RAM, we report blueprint sizes of
  2.7–8.4 KB (compression ratios of 2.2×–102.8×), deterministic reconstruction
  given a fixed seed, and a per-backend fidelity and speed comparison on four
  classic USC-SIPI test images (mandril, peppers, cameraman, airplane) in which
  Hyper-SD 8-step distilled schedules *beat* their 30-step counterparts on
  SD 1.5 (+0.54 dB), and a ControlNet scale sweep finds that lower depth
  scales (0.8 vs 1.5) improve fidelity with the Depth-Anything-V2-Base stack. We are explicit that brainimg
  is lossy-by-design, decoder-dependent, and unsuited to forensic/medical use;
  we position it as a systems study of a novel paradigm rather than a
  replacement for JPEG.
---

# 1. Introduction

Every widely used image format answers a variant of the same question: *how do
we store these pixels efficiently?* PNG stores exact colors and predicts the
next pixel; JPEG/WebP discard transform coefficients the human visual system
barely notices; AVIF generalizes this to AV1 intra-frames. All three paradigms
share two properties: the file is bound to a native resolution, and "loss"
manifests as visible artefacts—blockiness, ringing, colour bleeding.

Neuroscience offers a different model. The retina compresses roughly 130
million photoreceptor signals into about one million optic-nerve fibres
(a ~130:1 ratio) by transmitting *edges and contrast* rather than absolute
brightness. Deeper in the visual cortex, representation becomes increasingly
semantic—neurons respond to faces, objects, materials—until recall itself
becomes a generative act: to imagine an apple is not to "open a file" but to
drive top-down signals from concept areas back into the visual cortex and
*hallucinate* the sensory experience back into existence. The brain stores
meaning and the rules to rebuild, not a pixel grid.

This paper is a reproducible systems study of a compression paradigm that
mimics that idea. We ask:

> **Can a modern diffusion model serve as the decoder for a kilobyte-scale
> image format, conditioned on a text caption and a handful of low-resolution
> structural maps?**

We do **not** claim a new state of the art, nor that the result replaces JPEG.
We claim that the paradigm is implementable today, on commodity hardware
(including an 8 GB Apple Silicon laptop), that it is deterministic given a
seed, and that it exhibits qualitatively different—and in some ways
useful—failure modes from transform codecs.

### Contributions

1. **Paradigm framing.** We articulate the "Semantic-Relational Field / 
   generative-recall" image-compression paradigm and map it onto a concrete
   encoder–format–decoder architecture.
2. **A versioned, forward-compatible file format.** `brainimg` v0.1 is a small
   JSON document carrying a caption, three optional conditioning maps, a seed,
   and colour statistics. Optional fields allow older files to decode
   unchanged as new conditioners are added.
3. **A four-stage encoder** combining a vision-language model (Qwen2-VL on
   MLX, Qwen2.5-VL-7B on CPU/CUDA), Depth-Anything-V2-Base, OpenCV Canny, and
   OneFormer ADE20K semantic segmentation, with explicit memory release
   between stages so the pipeline fits in 8 GB.
4. **Fourteen pluggable decoder backends** spanning seven model families
   (SD 1.5, SDXL, Z-Image, Qwen-Image, HunyuanDiT, SANA, FLUX), each with an
   optional Hyper-SD 8-step distilled turbo variant for SD 1.5, SDXL, and
   FLUX. All backends consume the same blueprint; new backends require no
   schema change.
5. **Empirical findings on distilled schedules and scale tuning.** On the
   SIPI mandril test image, Hyper-SD's 8-step distilled LoRA *beats* the
   30-step non-turbo path on SD 1.5 (+0.54 dB) while running 3.1× faster on
   CPU. A grid sweep of ControlNet conditioning scales finds that
   Depth-Anything-V2-Base's sharper depth map over-constrains at the
   historical default of 1.5; lowering depth to 0.8 and raising seg to
   parity (1.0) yields a measurable lift on the turbo path.
6. **A gamma fallback for the brightness-clamp edge case.** When the uniform
   gain clamp $[0.5, 2.0]$ cannot reach an extreme target, a per-channel
   gamma curve (same exponent, clamped to $[0.3, 3.0]$) closes the residual
   gap without the clipping artefacts a >2× gain would cause.

The remainder of the paper is organised as follows. §2 surveys related work
in classical and learned compression, text-to-image generation, and
ControlNet conditioning. §3 describes the method (format, encoder, decoder,
device strategy). §4 reports measurements reproducible from the committed
repository. §5 discusses the paradigm's properties—resolution independence,
semantic loss, decoder dependency—and honest limitations. §6 concludes and
points to planned work.

# 2. Related Work

### 2.1 Classical image codecs

The three dominant compression paradigms are *spatial/predictive* (PNG,
run-length + DEFLATE over filtered scanlines), *spectral/transform*
(JPEG's discrete cosine transform, WebP's VP8 intra, AVIF's AV1 intra), and
*fractal/wavelet* (JPEG 2000). All are pixel-bound: the file encodes a fixed
grid at a fixed resolution, and "loss" is a transform artefact. brainimg
departs from all of these by storing *no pixel data at all*.

### 2.2 Learned / neural compression

Neural compression typically maps pixels to a lower-dimensional latent space
via an autoencoder and entropy-codes the latents. The decoder is a learned
up-sampler. brainimg is adjacent but distinct: its "latent" is not a
continuous vector but a *human-readable caption plus a few low-resolution
maps and a seed*, and the decoder is a general-purpose text-to-image
diffusion model conditioned by ControlNets rather than a codec-specific
autoencoder. The decoder is therefore large (~4 GB for SD 1.5) and shared
across all files, while each file is tiny and self-describing.

### 2.3 Text-to-image diffusion and ControlNet

Stable Diffusion 1.5 [Rombach et al., 2022] performs latent diffusion
conditioned on a CLIP text embedding. ControlNet [Zhang et al., 2023] adds a
trainable copy of the UNet encoder blocks that forces the diffusion process
to follow an external spatial conditioner (depth, edges, segmentation,
pose). brainimg uses two ControlNets unconditionally (depth + Canny) and a
third (ADE20K segmentation) when the blueprint carries a seg map. The
combination of CLIP caption conditioning with multi-map ControlNet
conditioning and a stored seed is, to our knowledge, novel as a *file-format*
design, though the individual components are off-the-shelf.

### 2.4 Vision-language captioning

Accurate captions are the semantic backbone of the format. We use the
Qwen2-VL-2B-Instruct 4-bit model via MLX on Apple Silicon for fast, low-memory
captioning, and fall back to the Qwen2.5-VL-7B-Instruct model via
transformers elsewhere. A known failure mode (§4.5) is that the captioner can
misidentify scene elements; because the conditioning maps drive structural
fidelity, a wrong caption biases mood more than geometry.

### 2.5 The biological inspiration

The retina's ~130:1 compression via lateral inhibition, sparse coding in V1,
and top-down generative recall are well documented in the neuroscience
literature. We use these only as *motivation*; brainimg is an engineering
artefact, not a brain model.

# 3. Method

brainimg has three parts: an *encoder* that extracts a blueprint from a
source image, a *file* that stores it, and a *decoder* that regenerates an
image from the blueprint. Encoder and decoder run as **separate processes**
so their heavy models are never resident simultaneously—an important
constraint on the 8 GB Apple Silicon target.

```
 Source image --> ENCODER --> .brainimg (JSON, ~3-10 KB) --> DECODER --> image
    (jpg/png)        |                                    (SD+ControlNet)   (any size)
                     +- 1. Caption (Qwen-VLM)        --> prompt
                     +- 2. Depth   (Depth-Anything-V2) --> 128^2 JPEG
                     +- 3. Canny   (OpenCV)            --> 128^2 PNG
                     +- 4. Seg     (OneFormer ADE20K)  --> 128^2 PNG (optional)
```

## 3.1 The `.brainimg` file format

A `.brainimg` file is a small UTF-8 JSON document, typically 3–10 KB
regardless of source resolution. The schema is versioned (`format_version`
is currently `"0.1"`). Required fields are `format_version`,
`original_width`, `original_height`, `prompt`, `depth_map_b64`,
`canny_map_b64`, and `seed`. Optional fields, added after the initial release,
are stored so that older files still decode: `segmentation_map_b64` enables
the seg ControlNet only when present and non-empty. The `extra` dictionary
absorbs unknown fields for forward compatibility. A representative file:

```json
{
  "format_version": "0.1",
  "original_width": 1024, "original_height": 768,
  "prompt": "a red apple on a wooden table next to a window",
  "negative_prompt": "blurry, low quality, deformed, watermark, jpeg artifacts",
  "depth_map_b64": "<base64 128×128 JPEG>",
  "canny_map_b64":  "<base64 128×128 PNG>",
  "segmentation_map_b64": "<base64 128×128 PNG, optional>",
  "seed": 42, "steps": 30,
  "target_brightness": 142.5,
  "target_saturation":  88.3,
  "extra": { "color_style": "vivid, saturated colors, natural lighting, red dominant tones" }
}
```

The format module (`brainimg/format.py`) is deliberately free of any ML
import so the schema and round-trip tests run without downloading models.
`MAP_SIZE` (currently 128) controls conditioning-map resolution; raising it
improves structural fidelity at the cost of a slightly larger file and
requires re-encoding existing samples.

## 3.2 Encoder

The encoder (`brainimg/extract.py`) runs four extractors in sequence,
releasing memory between heavy stages via `free_torch()` / `free_mlx()` so
the pipeline fits in 8 GB.

**Stage 1 — Captioning.** A Qwen2-VL vision-language model is prompted with
the instruction *"In one short sentence, describe this image for
reconstruction: main objects, layout, lighting, and colors."* On Apple
Silicon the MLX 4-bit `mlx-community/Qwen2-VL-2B-Instruct-4bit` model is used
(fast, low memory); on any other platform the transformers
`Qwen/Qwen2.5-VL-7B-Instruct` model is used. Both produce an equivalent
caption; the 7B is noticeably more detailed. MLX memory is released before
any PyTorch stage runs.

**Stage 2 — Depth map.** Depth-Anything-V2-Base (`depth-anything/Depth-Anything-V2-Base-hf`)
via a HuggingFace `depth-estimation` pipeline produces a grayscale depth map
(near = bright, far = dark), downscaled to `MAP_SIZE × MAP_SIZE` with Lanczos
resampling and JPEG-encoded at quality 70.

**Stage 3 — Canny edges.** OpenCV's Canny detector (low=50, high=150) is run
on the grayscale image downscaled with `INTER_AREA`, then PNG-encoded to keep
edges crisp.

**Stage 4 — Segmentation (optional).** OneFormer ADE20K
(`shi-labs/oneformer_ade20k_swin_tiny`) produces a semantic map that is
colourised with the exact 150-class ADE20K palette the SD 1.5 seg ControlNet
was trained on, then nearest-neighbour downscaled to `MAP_SIZE × MAP_SIZE`
and PNG-encoded. This stage was added after the initial v0.1 release;
older files omit it and decode unchanged.

**Colour-style and statistics.** Alongside the four conditioners the encoder
computes two scalar statistics of the original image, stored for decoder-side
post-processing. *Brightness* is the mean Rec.609 luminance on a 0–255 scale:

$$ B = \mathrm{mean}\!\left(0.299\,R + 0.587\,G + 0.114\,B\right). $$

*Saturation* is the mean normalised channel spread, scaled to 0–255 (0 =
grey, 255 = fully saturated):

$$ S = 255 \cdot \mathrm{mean}\!\left(\frac{\max_c - \min_c}{\max(\max_c, 1)}\right). $$

A short *colour-style prefix* is also derived (e.g. *"dark, low-key lighting,
red dominant tones"*) and stored in `extra`; the decoder prepends it to the
caption only when the combined length fits the CLIP 77-token limit, so the
caption itself is never truncated.

## 3.3 Decoder

The decoder (`brainimg/generate.py`) regenerates an image from the blueprint
using one of fourteen pluggable backends, all sharing the same blueprint schema
and the same brightness/saturation post-processing:

| `--model` | Base | Conditioning | Steps (default) | License |
|---|---|---|---|---|
| `sd15` (default) | `stable-diffusion-v1-5` | depth + canny (+ seg) ControlNets | 30 | CreativeML Open RAIL-M |
| `sd15-turbo` | SD 1.5 + Hyper-SD 8-step LoRA | depth + canny (+ seg) ControlNets | 8 | CreativeML Open RAIL-M |
| `sdxl` | `stable-diffusion-xl-base-1.0` | depth + canny (+ seg) ControlNets | 30 | CreativeML Open RAIL-M |
| `sdxl-turbo` | SDXL + Hyper-SD 8-step LoRA | depth + canny (+ seg) ControlNets | 8 | CreativeML Open RAIL-M |
| `zimage` | `Tongyi-MAI/Z-Image-Turbo` | single Union ControlNet (depth) | 9 (8-step Turbo) | Tongyi-MAI non-commercial |
| `qwen-image` | `Qwen/Qwen-Image` | single Union ControlNet (depth) | 50 | **Apache 2.0** |
| `hunyuan` | `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled` | depth + canny ControlNets | 25 | tencent-hunyuan-community |
| `sana` | `Efficient-Large-Model/Sana_600M_1024px_diffusers` | single HED ControlNet (canny map) | 20 | **MIT** |
| `flux2-klein` | `black-forest-labs/FLUX.2-klein-4B` | img2img (depth map as starting image) | 4 | **Apache 2.0** |
| `flux-depth` | `black-forest-labs/FLUX.1-Depth-dev` | channel-concat `depth_map_b64` | 30 | FLUX.1-dev non-commercial |
| `flux-canny` | `black-forest-labs/FLUX.1-Canny-dev` | channel-concat `canny_map_b64` | 30 | FLUX.1-dev non-commercial |
| `flux-depth-turbo` | FLUX.1-Depth-dev + Hyper-SD 8-step FLUX LoRA | channel-concat `depth_map_b64` | 8 | FLUX.1-dev non-commercial |
| `flux-union` | `black-forest-labs/FLUX.1-dev` + Shakker-Labs Union ControlNet | Union ControlNet (depth + canny) | 24 | FLUX.1-dev non-commercial |
| `sd35` | `stabilityai/stable-diffusion-3.5-large` | depth + canny ControlNets | 50 | stabilityai-ai-community |

All fourteen backends consume the **same blueprint**; the schema is unchanged
when a new decoder is added. Two additional CLI variants of the counted
backends are supported by `decoder.py` but not enumerated as separate paper
backends: `--model hunyuan-full` (the non-distilled 50-step HunyuanDiT
v1.2, used in §4.6 to isolate the distillation artefact) and `--model
flux-canny-turbo` (Hyper-SD 8-step LoRA on `FLUX.1-Canny-dev`, the
canny-channel counterpart of `flux-depth-turbo`). We describe the SD 1.5 / SDXL path in detail
(the historical default), then summarise the turbo, Z-Image, Qwen-Image,
HunyuanDiT, SANA, FLUX.2-klein, and FLUX paths that diverge structurally.

### 3.3.1 SD 1.5 / SDXL (ControlNet stack)

The default backend. SD 1.5 ships at 512², SDXL at 1024².

**Conditioning.** The depth and Canny maps are decoded from base64 and
upscaled to the target size (Lanczos for depth, nearest-neighbour for Canny
and seg to keep edges/palette crisp). When a segmentation map is present it is
appended as a third conditioner. Default conditioning scales for SD 1.5 are
depth 0.8, Canny 1.0, seg 1.0 — tuned via a grid sweep on the SIPI samples
(§4.8). The historical defaults (depth 1.5, canny 1.2, seg 0.9) were set for
the older Depth-Anything-Small + no-seg pipeline; Depth-Anything-V2-Base's
sharper depth map over-constrains at 1.5, and the ADE20K seg ControlNet adds
material cues that warrant parity with Canny.

**VAE substitution.** The stock SD 1.5 VAE is replaced with the fine-tuned
`stabilityai/sd-vae-ft-mse` for cleaner decode, better skin tones and
colours, and fewer washed-out highlights, at negligible runtime cost.

**Scheduler and steps.** The pipeline uses a `UniPCMultistepScheduler` with a
default of 30 inference steps (raised from 20). ControlNet scales and
classifier-free guidance (default 7.5) are CLI-tunable.

**Determinism.** A `torch.Generator` seeded with `data.seed` makes generation
deterministic: re-decoding with the same seed reproduces the same image
exactly (verified to produce 0 pixel difference between runs).

**Colour post-processing.** SD 1.5 systematically over-brightens and
over-saturates. The decoder corrects this against the stored targets in two
ordered steps, chosen so each step does not undo the other:

1. *Saturation first*, by scaling the HSV-S channel by
   `target_saturation / current_saturation`, iterated 1–2× to converge (HSV-S
   is not linear in the metric above). Ratios are clamped to $[0.5, 2.0]$ to
   avoid clipping artefacts. This perturbs brightness, which the next step
   corrects.
2. *Brightness last*, by a uniform RGB gain `target_brightness /
   current_brightness`. A uniform gain preserves colour balance and leaves
   the saturation metric (a ratio of channels) invariant, so correcting
   brightness last does not undo the saturation work. The same $[0.5, 2.0]$
   clamp applies.

The step is a no-op for older files whose targets are 0.0, or when the
generation's stats are already within ~2 % of the targets. When the clamped
gain cannot reach an extreme target (e.g. darkening 210→80 needs ratio 0.38,
clamped to 0.5 → 105), a **gamma fallback** applies a per-channel gamma curve
$\text{arr}^\gamma$ (same exponent on every channel, clamped to
$[0.3, 3.0]$) to close the residual gap. Gamma preserves colour balance
approximately (not exactly like a uniform gain, but far better than clipping)
and reaches extreme targets without the posterisation a >2× gain would
cause. The gamma exponent is $\gamma = \log(t/255) / \log(c/255)$ where $c$
and $t$ are the current and target brightnesses normalised to $[0, 1]$.

**Style-prefix gating.** The stored colour-style prefix is prepended to the
caption only when the combined length fits within the CLIP 77-token limit,
biasing mood without ever truncating the caption. Older files with no stored
style use the caption verbatim. (For Z-Image-Turbo and FLUX, which use a
Qwen / T5 text encoder with a 512-token limit, the style prefix is
prepended unconditionally.)

### 3.3.2 Z-Image-Turbo (Union ControlNet)

`--model zimage` swaps the SD stack for **Tongyi-MAI/Z-Image-Turbo** (a
6 B-parameter single-stream DiT distilled for ~8 steps) plus the
`alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` *Union* ControlNet
(the full `2.1-8steps` variant, ~6.4 GB). The Union net is a single
network that takes one conditioning image per call and bakes the
conditioning *type* (depth/canny/pose/mlsd/hed) into its training; we feed
the blueprint's `depth_map_b64` because depth carries the most structural
information. The blueprint's `canny_map_b64` and `segmentation_map_b64` are
**silently ignored** on this path—no schema change. Z-Image runs in **bf16
throughout**, which sidesteps the MPS fp16 NaN bug entirely (a different
dtype, no int8 quantization is used). Defaults: `guidance_scale = 0.0`
(Turbo is distilled for zero CFG), 9 inference steps (8 DiT forward
passes on the Turbo schedule), 1024 max side.

### 3.3.3 FLUX.1-Depth-dev / FLUX.1-Canny-dev (channel-concat)

`--model flux-depth` and `--model flux-canny` load Black Forest Labs'
**FLUX.1** guidance-distilled Control variants. FLUX.1 is a 12 B-parameter
MMDiT transformer with a T5-XXL text encoder (CLIP-L is the small auxiliary
encoder). The FLUX.1-*-dev Control checkpoints bake the conditioning into
the transformer via **channel-wise concatenation** of the conditioning
image—not a separate ControlNet model. Diffusers' `FluxControlPipeline`
takes a single `control_image` per call; we feed `depth_map_b64` for
`flux-depth` and `canny_map_b64` for `flux-canny`. The other map and any
`segmentation_map_b64` are silently ignored. bf16 throughout (FLUX's
native dtype, sidesteps the MPS fp16 NaN bug). Defaults: cfg 10.0 (depth)
or 30.0 (canny), 30 inference steps, 1024 max side, `max_sequence_length =
512` (T5-XXL). No per-call `controlnet_conditioning_scale`—the
conditioning strength is fixed by the trained checkpoint, so the
`--depth-scale` / `--canny-scale` CLI flags are accepted for argument
parity but silently ignored on FLUX paths. **License**: FLUX.1-Depth-dev /
-Canny-dev carry the FLUX.1-dev non-commercial license.

### 3.3.4 Hyper-SD turbo distillation (SD 1.5 / SDXL / FLUX)

ByteDance's **Hyper-SD** [Ren et al., 2024] applies trajectory-segmented
consistency distillation to produce small (~70–150 MB) LoRAs that fold the
SD 1.5, SDXL, and FLUX.1-dev base models down to **8 inference steps** while
preserving the stock ControlNets / channel-concat conditioning. The LoRA is
loaded via `pipe.load_lora_weights` and fused with `pipe.fuse_lora(0.125)`
(per the model card) after device placement, so the UNet / transformer
weights are permanently adjusted — no per-call LoRA overhead. For SD 1.5
and SDXL, the scheduler is swapped to
`DDIMScheduler(timestep_spacing="trailing")`; for FLUX, no scheduler swap
is needed (`FlowMatchEulerDiscreteScheduler` works natively with fewer
steps). The turbo paths ignore the file's stored step count (tuned for
20–30 step SD schedules) and use 8 steps unless `--steps` is passed.

**FLUX turbo compatibility fix.** The Hyper-SD FLUX LoRA was trained on the
base `FLUX.1-dev`, not the Control variants. The Control transformer's
`x_embedder` (patch embedding) has extra input channels for the control
image (128 vs 64) and a `context_embedder` that base dev doesn't have, so
those LoRA deltas are shape-incompatible. The decoder strips the
`x_embedder` and `context_embedder` keys (and the `transformer.` prefix,
which diffusers adds internally) before loading. The attention/FFN LoRA
deltas — the bulk of the distillation — load cleanly and are what matters
for the 8-step schedule. Guidance drops to 3.5 (the FLUX dev default, not
the 10.0 / 30.0 the non-turbo control variants use).

### 3.3.5 Qwen-Image (Union ControlNet, Apache 2.0)

`--model qwen-image` loads Alibaba's **Qwen-Image** [Wu et al., 2025]
(arXiv 2508.02324, Apache 2.0) — a DiT with a Qwen2.5-VL text encoder
(512-token limit) — plus InstantX's `Qwen-Image-ControlNet-Union`, a single
Union ControlNet supporting canny + depth + pose + soft-edge. Same pattern
as Z-Image: depth-only conditioning, blueprint's canny/seg ignored, bf16
throughout. Defaults: 50 steps, `true_cfg_scale = 4.0` (Qwen-Image's CFG
parameter, distinct from `guidance_scale`), `controlnet_conditioning_scale
= 0.9`, 1024 max side. The Qwen-Image backend is the only fully-open
(Apache 2.0) high-quality option in the stack; FLUX is non-commercial and
Z-Image is non-commercial.

### 3.3.6 HunyuanDiT v1.2 (depth + canny ControlNets)

`--model hunyuan` loads Tencent's **Hunyuan-DiT** [Li et al., 2024]
(arXiv 2405.08748) v1.2 Distilled — a bilingual (Chinese + English) DiT with
**two separate ControlNets** (depth + canny), the same two-conditioner
pattern as SD 1.5 / SDXL rather than a Union net. The blueprint's seg map
is **silently ignored** (no seg ControlNet exists for HunyuanDiT). HunyuanDiT
only supports fixed resolutions (1024, 1280, …); on a 512-side blueprint it
auto-upscales to 1024. Text encoding uses BERT + T5 (the CLIP 77-token
limit does not apply; the colour-style prefix is prepended unconditionally).
bf16 throughout (HunyuanDiT's native dtype, sidesteps the MPS fp16 NaN bug).
Defaults: depth scale 0.8, canny scale 0.8, `guidance_scale = 3.0`, 25 steps
(distilled), 1024 max side. `--model hunyuan-full` swaps the Distilled base
for the non-distilled `HunyuanDiT-v1.2-Diffusers` (50 steps, otherwise
identical) so we can isolate whether distillation is the source of the
visual artefacts (§4.6). **License**: tencent-hunyuan-community.

### 3.3.7 FLUX.1-dev Union ControlNet (depth + canny)

`--model flux-union` loads Black Forest Labs' **FLUX.1-dev** base plus the
**Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro** checkpoint [Shakker Labs
& InstantX, 2024]. The Union ControlNet bundles seven conditioning modes
(canny, tile, depth, blur, pose, gray, low-quality) into a single network;
we feed **depth (mode 2) and canny (mode 0) simultaneously** via
`FluxMultiControlNetModel` and `control_mode=[2, 0]`. The blueprint's seg
map is **silently ignored** (no seg mode). `guidance_scale = 3.5`, 24 steps,
`controlnet_conditioning_scale = 0.4` per map, 1024 max side,
`max_sequence_length = 512`. `--quantize` FP8-quantizes the transformer and
T5-XXL (weights only), dropping resident CPU RAM from ~24 GB to ~12 GB.
**License**: FLUX.1-dev non-commercial (gated).

### 3.3.8 Stable Diffusion 3.5 Large (depth + canny ControlNets)

`--model sd35` loads Stability AI's **Stable Diffusion 3.5 Large**
[Stability AI, 2024] (8B MMDiT) with the official **depth** and **canny**
ControlNets. Both ControlNets are 8B checkpoints; we load them separately
and wrap them in `SD3MultiControlNetModel` so depth and canny are fed
together. Text encoding uses CLIP-L, CLIP-G, and T5-XXL (the SD3 text
encoder stack). bf16 throughout. Defaults: depth scale 0.7, canny scale 0.7,
`guidance_scale = 4.5`, 50 steps, 1024 max side,
`max_sequence_length = 256`. The seg map is **silently ignored** (no SD3.5
seg ControlNet). **License**: stabilityai-ai-community (gated, free for
non-commercial and for commercial use up to $1M annual revenue).

### 3.3.9 SANA 600M (HED ControlNet, MIT)

`--model sana` loads NVIDIA's **SANA** [Xie et al., 2024]
(arXiv 2410.10629, MIT license) — a 600 M-parameter linear DiT with a DC-AE
VAE (32× spatial compression) and a T5 text encoder. diffusers ships
`SanaControlNetPipeline` + `SanaControlNetModel` since 0.38. **Only an HED
(soft-edge) ControlNet exists for SANA** — no depth or canny ControlNet has
been trained. We feed the blueprint's **canny** map to the HED ControlNet
as the closest available conditioning type. This is a type mismatch: HED
produces soft probability edges while Canny produces hard binary edges,
creating a PSNR-vs-colour trade-off (§4.6, §4.7). The blueprint's depth and
seg maps are **silently ignored**. The diffusers-format ControlNet is a
community conversion (`ishan24/Sana_600M_1024px_ControlNet_diffusers`) of
the official NVlabs checkpoint; the base model is the diffusers port from
`Efficient-Large-Model`. bf16 throughout. Defaults: HED scale 0.4 (tuned via
sweep on the SIPI samples at 1024² — §4.8), `guidance_scale = 4.5`, 20 steps, 1024 max
side, `max_sequence_length = 300`. SANA is the fastest 1024-native backend
(54 s at 1024², ~5 GB RAM) but the lowest-PSNR backend due to the HED/canny
mismatch.

### 3.3.10 FLUX.2-klein-4B (img2img pseudo-ControlNet, Apache 2.0)

`--model flux2-klein` loads Black Forest Labs' **FLUX.2-klein-4B**
(Apache 2.0, ungated, 4 B-parameter rectified-flow transformer, 4-step
distilled) as an **image-to-image** model. FLUX.2-klein has **no
ControlNet**; we use it as a pseudo-ControlNet by feeding the blueprint's
**depth map** as the `image` parameter (the starting point for img2img). The
model encodes the depth map into latents and denoises from there with the
caption as the text guide, "editing" the depth map into a photorealistic
image rather than being structurally constrained by a ControlNet. The
blueprint's canny and seg maps are **silently ignored** (only one image
input). bf16 throughout. Defaults: `guidance_scale = 1.0`, 4 steps, 1024
max side, `max_sequence_length = 512`. The only real FLUX.2 ControlNet
(alibaba-pai Union, depth + canny) requires the VideoX-Fun library + the
32 B gated FLUX.2-dev, which is impractical on CPU; the img2img path is the
practical fallback.

## 3.4 Device and precision strategy

The current dev target is an **AMD x86_64 CPU-only box with 188 GB RAM**
(no CUDA/MPS), so the CPU fp32 path is the primary measurement platform.
The decoder also supports Apple Silicon (MPS int8) and NVIDIA CUDA (fp16),
but those are secondary. On MPS, SD 1.5 fp16 matmuls produce NaNs (a black
output frame); the decoder therefore int8-quantizes the UNet and all
ControlNets (weights *and* activations) via `optimum-quanto` to avoid fp16
matmuls entirely, while the VAE runs in fp32 for a clean final decode. On
CPU, full fp32 is supported (best fidelity, slow) and an optional int8
weights mode fits in ~5 GB at a small quality cost. On CUDA, fp16 works
correctly and is fast. Z-Image, Qwen-Image, HunyuanDiT, SANA, FLUX.2-klein,
FLUX, and SD3.5 are bf16-native and sidestep the MPS fp16 NaN bug entirely; on a
CPU-only box they are kept resident in host RAM (diffusers'
`enable_model_cpu_offload` raises `RuntimeError` without an accelerator to
offload *to*).

| `--device` / `--model` | Precision | RAM | Speed | Fidelity |
|---|---|---|---|---|
| `cpu` (sd15, default target) | fp32 (no quant) | ~10 GB | 156 s @ 512² | **best** (sd15) |
| `cpu --model sd15-turbo` | fp32 + Hyper-SD 8-step LoRA | ~10 GB | **51 s** @ 512² | good (+0.54 dB vs 30-step) |
| `cpu --model sdxl` | fp32 (no quant) | ~17 GB | 989 s @ 512² | **best** (sdxl) |
| `cpu --model sdxl-turbo` | fp32 + Hyper-SD 8-step LoRA | ~17 GB | **76 s** @ 512² | good (−2.58 dB vs 30-step) |
| `cpu --model zimage` | bf16 resident | ~18 GB | 308 s @ 512² | good (depth-only) |
| `cpu --model qwen-image` | bf16 resident | ~20 GB | 1006 s @ 512² | good (depth-only) |
| `cpu --model hunyuan` | bf16 resident | ~12 GB | 912 s @ 1024² | good PSNR, poor visual (§4.6) |
| `cpu --model sana` | bf16 resident | ~5 GB | **54 s** @ 1024² | lowest PSNR (HED/canny mismatch) |
| `cpu --model flux2-klein` | bf16 resident | ~13 GB | 42 s @ 512² | good PSNR, poor colour |
| `cpu --model flux-depth --quantize` | bf16 + FP8 (host RAM) | ~12 GB | 510 s @ 512² | **best** (FLUX) |
| `cpu --model flux-depth-turbo --quantize` | bf16 + FP8 + Hyper-SD 8-step | ~12 GB | **475 s** @ 512² | within noise of 30-step |
| `cpu --model flux-union --quantize` | bf16 + FP8 (host RAM) | ~12 GB | ~860 s @ 512² | mid PSNR |
| `cpu --model sd35` | bf16 resident | ~16-20 GB | ~3100 s @ 1024²→512² | mid PSNR, can zoom at 512² |
| `mps` (sd15, Apple Silicon) | int8 weights + activations | ~5 GB | medium (8 GB Mac) | fair |
| `cuda` (sd15) | fp16 | ~5 GB | **fast** | good |

The encoder/decoder device selection is centralised in `brainimg/device.py`,
which also provides `free_torch()` / `free_mlx()` memory-release helpers.

# 4. Experiments

All measurements below are reproducible from the committed repository; we use
only data already present in the repo and do not run the heavy ML models
here. Where a number is documented in the project (`README.md`, `TODO.md`)
we cite it; where it is an on-disk file size we measured it directly with
`stat`. We do not report PSNR/SSIM/LPIPS/FID because those runs are out of
scope for this writing session; §5 flags the planned evaluation.

## 4.1 Setup

- **Hardware:** AMD x86_64 CPU-only, 188 GB RAM (the current dev target),
  32 cores, 16 torch threads. All measurements in §4 are from this machine.
  The M1/8 GB Apple Silicon results in `README.md` (59 s @ 256²) are
  historical and noted where relevant.
- **Software:** Python 3.12, `uv`-managed environment, `requirements.txt`.
  `diffusers` 0.38, `peft` 0.19 (for LoRA loading), `optimum-quanto` (FP8).
- **Models:** SD 1.5 + `sd-vae-ft-mse` + 3 ControlNets (default), SDXL,
  Hyper-SD 8-step LoRAs (SD 1.5, SDXL, FLUX), Z-Image-Turbo, Qwen-Image,
  HunyuanDiT v1.2 Distilled + full, SANA 600M, FLUX.2-klein-4B,
  FLUX.1-Depth-dev / -Canny-dev. Captioner is transformers Qwen2.5-VL-7B
  (CPU fallback; MLX is Apple-Silicon-only).
- **Samples:** `samples/real.jpg` (256×256 puppy JPEG, 13,430 B),
  `samples/mandril_color.tif` (512×512 SIPI mandril, 787,420 B),
  `samples/peppers_color.tif` (512×512 SIPI peppers, 786,572 B),
  `samples/cameraman.tif` (512×512 SIPI cameraman, grayscale, 262,750 B),
  `samples/airplane.tif` (512×512 SIPI F-16 fighter jet, 786,572 B),
  `samples/test512.jpg` (512×512, 49,690 B). Blueprints: `real.brainimg`,
  `mandril.brainimg`, `peppers.brainimg`, `cameraman.brainimg`,
  `airplane.brainimg`, `test512.brainimg`. Mandril decodes use seed 200,
  peppers seed 300, cameraman seed 100, airplane seed 400, all at 512×512
  output unless noted. The four SIPI images are public-domain standard test
  images from the USC-SIPI database, used here in place of the historically
  common Lenna image (retired from many venues for reasons documented at
  <https://en.wikipedia.org/wiki/Lenna>).

## 4.2 Compression

Table 2 reports blueprint sizes and compression ratios against the source
files. The "5.0×" figure for the puppy in `README.md` corresponds to an
earlier pre-segmentation v0.1 file of ~2.7 KB; the committed `real.brainimg`
includes the seg map and colour statistics and is 6,120 B (2.2×), reflecting
the richer current schema.

| Source file | Source size | `.brainimg` size | Ratio |
|---|---|---|---|
| `samples/real.jpg` (puppy, 256²) | 13,430 B | 6,120 B (`real.brainimg`) | 2.2× |
| `samples/test512.jpg` (512²) | 49,690 B | 5,777 B (`test512.brainimg`) | 8.6× |
| `samples/cameraman.tif` (512², grayscale) | 262,750 B | 7,693 B (`cameraman.brainimg`) | 34.2× |
| `samples/mandril_color.tif` (512²) | 787,420 B | 8,386 B (`mandril.brainimg`) | 93.9× |
| `samples/peppers_color.tif` (512²) | 786,572 B | 7,652 B (`peppers.brainimg`) | 102.8× |
| `samples/airplane.tif` (512²) | 786,572 B | 6,305 B (`airplane.brainimg`) | 124.8× |
| (documented) `samples/real.jpg` | 13.4 KB | 2.7 KB (pre-seg v0.1) | 5.0× |

Two properties are worth noting. First, the blueprint size is **roughly
constant** (a few KB) regardless of source resolution or colour depth: a
256² puppy, a 512² grayscale cameraman, and a 512² colour mandril all
produce 6–8 KB files, because the stored maps are fixed at 128². Second,
the compression ratio therefore *grows with source size*: the large
uncompressed TIFFs compress ~94–103× while the already-compressed puppy
JPEG compresses ~2×. This is the opposite of transform codecs, whose ratios
are largely independent of whether the source is raw or pre-compressed, and
is a direct consequence of storing meaning rather than pixels.

## 4.3 Reconstruction quality (qualitative)

Reconstruction is semantically faithful (same scene, layout, and lighting)
but not pixel-identical—by design. The committed repository ships one
side-by-side figure:

- `comparison.jpg` — original vs. brainimg reconstruction of `samples/real.jpg`.
  Per `README.md`, the captioner correctly described the scene ("a black puppy
  sitting on a wooden surface") and the decoder produced a visually faithful
  reconstruction at 256×256 in 59 s on the M1/8 GB machine.

Across the SIPI subjects the per-backend character is best read from the
fidelity numbers in §4.7: SDXL at its native 1024² reproduces the mandril's
broad palette most faithfully; SD 1.5 turbo gives the best speed/fidelity
trade-off on the same image; FLUX.1-Depth-dev at 512² stays close to the
source palette where SDXL@512 drifts (§4.6); HunyuanDiT and FLUX.2-klein
score in the top half by PSNR yet are visually weak due to palette
collapse/shift (§4.6, §4.7). Cross-subject, the grayscale cameraman is the
easiest to match (FLUX depth turbo reaches 15.80 dB) and the F-16's clean
lines suit the canny ControlNet (SD 1.5 turbo reaches 15.05 dB on
`airplane.brainimg`, the highest PSNR of any SIPI subject at 512²).

Side-by-side grids of all backends on each SIPI subject can be regenerated
with `scripts/make_backend_grid.py <subject>` (mandril, peppers,
cameraman, airplane); the script writes `<subject>_grid.jpg` next to the
source. The per-backend reconstruction PNGs and `<subject>_<backend>_comparison.jpg`
side-by-sides are similarly produced by `decoder.py` and
`scripts/make_comparison.py`. These artifacts are deliberately gitignored
(see `.gitignore`: they are large decoder outputs, not source) and
regenerated on demand.

## 4.4 Determinism

`README.md` reports that re-running the decoder with the same seed reproduces
the same image exactly, verified as 0 pixel difference between runs. The
SDXL mandril run was additionally verified to produce an identical md5 across
runs. Determinism is a property of the format: the seed is stored in the file,
so a `.brainimg` file plus a fixed decoder yields a bit-identical image.

## 4.5 Device and precision ablation

Table 1 (§3.4) summarises the supported operating modes. The key
engineering finding is that on Apple Silicon the SD 1.5 naive fp16 path is
unusable (NaNs), forcing int8 weights + activations; this halves memory
but degrades structural fidelity relative to CPU fp32. CPU fp32 is the
recommended SD 1.5 mode on a high-RAM machine for best quality; CUDA fp16
is the recommended mode for speed. Z-Image-Turbo, Qwen-Image, HunyuanDiT,
SANA, FLUX.2-klein, and FLUX are bf16 throughout (their native dtype),
which sidesteps the MPS fp16 NaN bug entirely; FLUX additionally supports
FP8 quantization of the transformer + T5 via `optimum.quanto` to halve
resident memory at a small quality cost.

## 4.6 Known failure modes

Five failure modes remain documented in `TODO.md`:

- **Captioner misidentification.** The 7B captioner can misidentify scene
  elements. On the SIPI mandril it correctly identified the colourful face
  but on some dark subjects has read hair as a hat or accessories. Because
  the conditioning maps (depth/Canny/seg) capture the true structure
  regardless, a wrong caption biases mood more than geometry—but it does
  bias generation. Mitigations noted for future work: a larger VLM or
  caption ensembling.
- **SDXL hue-distribution drift at small sizes.** When SDXL is decoded at
  512×512 on a source whose palette spans multiple hue bands (e.g. the
  mandril's blue face stripes + orange/red nose + green/yellow fur), the
  output can collapse the blue band: the mandril source is 30.7% blue/purple
  pixels, SDXL@512 drops it to 7.6%, while SDXL-turbo@512 overshots to
  49.7%. This is a content/palette drift, not a stat drift: a single
  global HSV-H rotation aligns means but cannot reshape distributions, and
  a rotation large enough to chase a different distribution recolours
  neutrals and skin in a way that reads worse than the original drift.
  Workaround: prefer SDXL at 1024×1024, where the drift is much smaller.
  FLUX.1-Depth-dev at 512×512 produces a less-drifted palette than SDXL@512
  on the same blueprint (§4.7).
- **HunyuanDiT pixel-metric-vs-perceptual disconnect.** HunyuanDiT scores
  10.68 dB PSNR (mid-pack, §4.7) and is visually the worst backend by a
  wide margin — visible artefacts and a blue/purple band collapse (8.8%
  vs the mandril source's 30.7%) that MSE/PSNR do not capture. The good
  MSE relative to SD 1.5 likely comes from getting overall
  brightness/layout right at 1024² while producing texture/feature
  artefacts. The likely cause is a language mismatch: HunyuanDiT is
  bilingual with a BERT tokenizer trained primarily on Chinese data; the
  English caption may produce inferior results regardless of tuning. Not
  recommended for visual use; kept for the systems-study comparison.
- **SANA HED/canny mismatch.** SANA's only available ControlNet is an HED
  (soft-edge) net; we feed the blueprint's canny (hard binary edge) map to
  it. This type mismatch creates a PSNR-vs-colour trade-off: on the mandril
  at 1024² the default scale 0.4 gives 7.69 dB — the lowest PSNR of any
  backend — while preserving the blue band at 28.5% (close to the source's
  30.7%). Raising the scale would trade colour fidelity for PSNR. SANA is
  the fastest 1024-native backend (54 s at 1024², ~5 GB RAM) but the
  lowest-PSNR backend due to the mismatch.
- **FLUX.2-klein img2img palette shift.** The pseudo-ControlNet img2img
  path (§3.3.10) reaches #2 PSNR overall (11.01 dB at 512²) but shifts the
  colour palette: 41.9% blue/purple vs the mandril source's 30.7%, and on
  the peppers (12% blue source) the FLUX depth turbo path collapses to
  0.3% blue — the model converts the depth map's grayscale into warm tones
  regardless of the caption. As with HunyuanDiT, this is a case where
  pixel-level MSE rewards overall brightness/layout accuracy while
  missing a colour-distribution failure that the human eye sees
  immediately.

The brightness-clamp edge case that was previously listed here has been
**fixed** with a gamma fallback (§3.3.1): when the uniform gain clamp
$[0.5, 2.0]$ cannot reach an extreme target, a per-channel gamma curve
closes the residual gap. Three new tests in `tests/test_color.py` cover
the gamma darkening, brightening, and approximate ratio preservation paths.

## 4.7 Per-backend fidelity and speed comparison (mandril, 512×512)

We report three pixel-level metrics (MSE, PSNR, MAE) and wall time for each
decoder backend on the SIPI mandril blueprint (`samples/mandril_color.tif`,
512×512, seed 200), all decoded at 512×512 on the AMD CPU target (188 GB RAM)
unless noted. Metrics are computed by `scripts/compare_backends.py` against
the source resized to 512×512. Two cross-subject sanity rows (peppers,
cameraman) at the best-performing backend confirm the codec works across
palettes and grayscale.

| Backend | Steps | Time (s) | MSE ↓ | PSNR (dB) ↑ | MAE ↓ |
|---|---:|---:|---:|---:|---:|
| SD 1.5 (tuned scales) | 30 | 156 | 8696 | 8.74 | 75.6 |
| SD 1.5 turbo (Hyper-SD, tuned) | 8 | **51** | 7683 | 9.28 | 70.3 |
| SDXL | 30 | 989 | 3253 | **13.01** | 46.1 |
| SDXL turbo (Hyper-SD) | 8 | **76** | 5891 | 10.43 | 60.4 |
| Z-Image (depth-only) | 8 | 308 | 7023 | 9.67 | 67.6 |
| Qwen-Image (depth-only) | 50 | 1006 | 5944 | 10.39 | 61.5 |
| SANA (HED/canny, scale 0.4) | 20 | 54 @ 1024² | 11067 | 7.69 | 88.1 |
| HunyuanDiT (depth+canny, 1024²) | 25 | 912 | 5559 | 10.68 | 59.8 |
| FLUX.2-klein (img2img, 512²) | 4 | 42 | 5159 | 11.01 | 57.2 |
| FLUX.1-Depth-dev (FP8) | 30 | 510 | 6620 | 9.92 | 64.6 |
| FLUX.1-Depth-dev turbo (FP8) | 8 | **475** | 6648 | 9.90 | 64.5 |
| FLUX Union (depth+canny, FP8) | 24 | ~860 | 7817 | 9.20 | 72.2 |
| SD 3.5 (depth+canny, 1024²→512²) | 50 | ~3100 | 8048 | 9.07 | 72.8 |

| **Cross-subject sanity (FLUX depth turbo, FP8)** | | | | | |
| `peppers_color.tif` (512²) | 8 | 187 | 4142 | 11.96 | 51.5 |
| `cameraman.tif` (512² grayscale) | 8 | 207 | 1712 | **15.80** | 28.4 |

Six observations:

1. **Hyper-SD 8-step distilled schedules beat their 30-step counterparts on
   SD 1.5.** The turbo path scores 9.28 dB vs 8.74 dB for the 30-step path
   (+0.54 dB) — at 3.1× less wall time. On FLUX, however, the turbo path
   (9.90 dB) does *not* beat the 30-step path (9.92 dB) on this image —
   within noise. The distilled-schedule-wins finding is SD 1.5-specific on
   this subject; it held strongly on the retired Lenna sample for both SD
   and FLUX (see §A) but does not generalise to the mandril's broad palette.

2. **SDXL is the best result across all backends** (13.01 dB, 989 s) — the
   30-step SDXL at its native 1024² resolution beats every other backend on
   the mandril. FLUX.2-klein img2img (11.01 dB, 42 s) follows at #2 by
   PSNR, and Qwen-Image (10.39 dB) at #3 — both ahead of the FLUX depth
   path. The ranking differs from the retired Lenna sample (§A), where
   FLUX depth turbo was #1 — a reminder that single-image rankings do not
   generalise.

3. **ControlNet scale tuning on SD 1.5.** The tuned defaults (depth 0.8,
   canny 1.0, seg 1.0) reflect the §4.8 scale sweep, which found depth 0.6
   marginally better than 0.8 on the mandril (9.37 vs 9.28 dB) — the
   0.8/1.0/1.0 compromise is retained because it is robust across both
   broad-palette (mandril) and narrow-palette subjects.

4. **Qwen-Image (Apache 2.0) is competitive with SDXL turbo** (10.39 dB
   vs 10.43 dB) despite being depth-only (no canny/seg), but slow at 50
   steps (1006 s). It beats Z-Image (9.67 dB) on the same depth-only
   pattern. The Apache 2.0 license is a practical advantage over FLUX's
   non-commercial license for distribution.

5. **Z-Image is the weakest depth-only backend** (9.67 dB, 308 s) —
   slightly below SDXL turbo (10.43 dB, 76 s) and Qwen-Image (10.39 dB).
   SANA (7.69 dB, 54 s @ 1024²) is the lowest-PSNR backend on this subject
   due to the HED/canny mismatch (§4.6), though it is the fastest
   1024-native backend.

6. **Pixel-level PSNR does not track perceptual quality for two backends.**
   HunyuanDiT (10.68 dB, mid-pack) is visually the worst backend by a
   wide margin — visible artefacts and a blue/purple band collapse
   (8.8% vs source 30.7%) — and FLUX.2-klein (#2 by PSNR, 11.01 dB)
   shifts the palette (41.9% blue vs source 30.7%). MSE rewards getting
   overall brightness/layout right but does not penalise texture/feature
   or colour-distribution failures; this is the concrete
   pixel-metric-vs-perceptual disconnect flagged in §5.3 as motivation
   for adding LPIPS / CLIP-score / FID in a real evaluation.

7. **The two newest ControlNet backends sit in the mid-PSNR range.**
   FLUX Union (9.20 dB, ~860 s) and SD 3.5 (9.07 dB, ~3100 s) both
   underperform the headline FLUX depth turbo and SDXL results on the
   mandril. SD 3.5 additionally requires native 1024² generation: when
   forced to 512² it produces a zoomed/cropped composition, so the
   decoder generates at 1024² and downscales. Both are included as
   additional ControlNet options rather than top-tier fidelity choices.

**These numbers do not generalise beyond the SIPI samples** (four sources
with varied palettes); §5.3 notes that a real evaluation requires more
subjects and perceptual metrics (LPIPS, CLIP-score, FID). Side-by-side
grids of all backends on each SIPI subject can be regenerated with
`scripts/make_backend_grid.py` (mandril, peppers, cameraman, airplane).

## 4.8 ControlNet scale sweep

The historical SD 1.5 conditioning defaults (depth 1.5, canny 1.2, seg 0.9,
cfg 7.5) were set for the older Depth-Anything-Small + no-seg pipeline. With
Depth-Anything-V2-Base (sharper depth) and the ADE20K seg ControlNet now in
the stack, we ran a grid sweep on `samples/mandril_color.tif` at 512×512
with `sd15-turbo` (8 steps, seed from the blueprint), using
`scripts/sweep_scales.py` — ~10 configurations, loading the pipeline once
and varying only the scale/cfg pair. (The original sweep also covered
`samples/test512.jpg`; see §A for those archived numbers.)

**Findings:**

- **Lower depth helps.** Depth 0.6 (9.37 dB) beats 0.8 (9.28 dB) beats 1.0
  (9.05 dB) on the mandril. The V2 depth map is sharper than the Small
  model's, so high scales over-constrain and fight the caption.
- **Seg at parity (1.0) is the compromise.** The ADE20K seg ControlNet adds
  material cues that were missing in the old no-seg pipeline; parity with
  canny (1.0) is robust.
- **Canny 1.0 beats 1.2.** A small but consistent improvement.

**New SD 1.5 defaults:** depth 0.8, canny 1.0, seg 1.0, cfg 7.5 (was
1.5/1.2/0.9/7.5). The 0.8 compromise is retained: 0.6 wins on the mandril by
0.09 dB but the older sweep found 0.6 hurt the narrower-palette test512
sample, and 0.8 is robust across both broad and narrow palettes. SDXL
defaults (1.0/0.8/0.6) were left unchanged — they were already in the good
region.

## 4.9 MAP_SIZE regression

A TODO item proposed raising `MAP_SIZE` from 128 to 256 for sharper
conditioning maps. We tested this on a 512×512 sample and it **regressed
on every backend**: SD 1.5 30-step −0.65 dB, SD 1.5 turbo −0.85 dB, SDXL
turbo −0.57 dB (archived numbers, see §A). File size also grew 2.5×
(~8 KB → ~20 KB). The ControlNets appear over-constrained by the sharper
maps at 512×512 output — 128 maps upscaled 4× to 512 give the right amount
of structural grip, while 256 maps upscaled 2× over-specify edges/depth.
`MAP_SIZE` stays at 128; the finding is documented in `TODO.md` and
`format.py`. Re-evaluation is warranted if the default output size moves
to 1024.

# 5. Discussion

### 5.1 Properties of the paradigm

**Resolution independence.** A `.brainimg` file has no native resolution: it
is a set of instructions for a generative model, so the same few-KB file can
be decoded at 256, 512, 1024, or higher, the decoder simply generating more
pixels. This is qualitatively different from transform codecs, where
up-sampling a small file produces a blurry large image.

**Semantic, not pixel, loss.** In JPEG, loss is block artefacts and colour
bleeding. In brainimg, "loss" means the model may render a slightly different
wood grain or fabric pattern than the original, while the fabric still looks
photorealistic. The lost data is micro-detail, replaced by synthesised
micro-detail. This is desirable for human-visual consumption and unacceptable
for forensic/medical/legal use where exact pixels matter.

**Editability.** Because the file is a self-describing JSON blueprint, a user
can edit the caption, mood prefix, or conditioning scales and re-decode to
relight or re-pose the scene—an early hint of the "edit the recipe, not the
pixels" property of the Semantic-Relational Field paradigm.

**Determinism.** Stored seed + fixed decoder $\Rightarrow$ bit-identical re-decode. This
addresses the usual "hallucination is non-deterministic" objection to
generative codecs: within a fixed decoder version, a file *is* a stable image.

### 5.2 Limitations

- **Decoder dependency.** Unlike JPEG's kilobyte decoder, brainimg's decoder
  is a multi-gigabyte neural network (anywhere from ~4 GB for SD 1.5 to
  ~22 GB for FLUX.1 bf16, reducible to ~12 GB with FP8 quantization). A
  device without the standard decoder cannot view the image—the codec is
  more like a video codec in this respect. On the other hand, the decoder
  is *shared* across all files, so the per-file cost remains a few KB.
- **Compute cost.** Decompression runs a diffusion model: 50 s for SD 1.5
  turbo at 512² on the AMD CPU target, 166 s for FLUX turbo (FP8), ~24 min
  for Qwen-Image at 50 steps. The turbo paths make interactive use
  plausible on CPU; the 30-step paths remain in the minutes-per-image
  range. This still precludes gallery scrolling or video use cases.
- **Quality depends on device.** CPU fp32 gives the best SD 1.5
  reconstruction; on 8 GB Apple Silicon, int8 quantization (required by
  the MPS fp16 NaN bug) degrades structural fidelity. Z-Image-Turbo,
  Qwen-Image, HunyuanDiT, SANA, FLUX.2-klein, FLUX, and SD3.5 are
  bf16-native and are not supported on 8 GB Apple Silicon; FLUX CPU
  requires ~22 GB host RAM (or ~12 GB with `--quantize`), while SD3.5
  requires ~16-20 GB. SANA is the lightest of the bf16-native backends
  (~5 GB RAM).
- **Lossy by design.** Reconstruction is semantically faithful, not
  pixel-identical. The format is explicitly unsuited to medical, legal, or
  forensic images.
- **Captioner accuracy.** Wrong captions bias generation (§4.6). Structure
  is the primary fidelity driver, which limits but does not eliminate the
  impact of caption errors.
- **Per-backend palette fidelity.** SDXL and Z-Image, run at sizes smaller
  than their native training resolution (1024), can drift into a different
  hue *distribution* than the source (§4.6). HunyuanDiT and FLUX.2-klein
  exhibit a more severe form of this — a palette *collapse/shift*
  (HunyuanDiT 8.8% blue vs mandril source 30.7%; FLUX.2-klein 41.9% blue)
  that even MSE does not catch (§4.6, §4.7). On the peppers (12% blue
  source), the FLUX depth turbo path collapses to 0.3% blue. This is a
  content/palette drift the existing brightness/saturation post-processor
  cannot correct. Prefer each backend's native resolution (512 for SD 1.5,
  1024 for SDXL / Z-Image / HunyuanDiT / SANA / FLUX.2-klein / FLUX) when
  colour fidelity matters.
- **License footprint.** The default backend (SD 1.5 + ControlNets) is
  CreativeML Open RAIL-M. SDXL is the same. Z-Image-Turbo is non-commercial.
  HunyuanDiT is tencent-hunyuan-community (a bespoke community license,
  not OSI-approved). FLUX.1-Depth-dev / -Canny-dev are FLUX.1-dev
  non-commercial; SD3.5 is stabilityai-ai-community (gated, free up to
  $1M annual revenue). Qwen-Image, SANA, and FLUX.2-klein-4B are **Apache 2.0**
  — the three fully-open options. If distribution matters, the SD 1.5/SDXL,
  Qwen-Image, SANA, and FLUX.2-klein paths are the permissively licensed
  choices.

### 5.3 Planned evaluation (not run here)

A complete evaluation would include: PSNR/SSIM/LPIPS against originals for
the sample images across all backends and device modes; CLIP-score as a
semantic-fidelity proxy; FID against a small natural-image set; a
bitrate-matched comparison to JPEG/WebP/AVIF at similar file sizes; and an
ablation removing each ControlNet in turn to quantify each conditioner's
contribution. The current measurements are on four SIPI test images
(mandril, peppers, cameraman, airplane) with varied palettes; a real
evaluation requires more subjects. The repository is structured to support
these via `encoder.py`/`decoder.py` and the `samples/` directory.

# 6. Conclusion

brainimg is a small, reproducible prototype of a different way to compress
images: store the *meaning and structure* of a scene and let a diffusion model
repaint it. We have described the format schema, the four-stage encoder, the
fourteen decoder backends (SD 1.5, SDXL, their Hyper-SD turbo variants,
Z-Image-Turbo, Qwen-Image, HunyuanDiT v1.2, SANA,
FLUX.2-klein-4B img2img, FLUX.1-Depth-dev / -Canny-dev with
Hyper-SD turbo on the depth variant, FLUX.1-dev + Shakker-Labs Union ControlNet,
and Stable Diffusion 3.5 Large) with their quality post-processing and gamma
brightness fallback, and the device/precision tradeoffs across CPU fp32,
MPS int8, and CUDA fp16. Using measurements reproducible from the
committed repository on an AMD CPU target with 188 GB RAM, we observe
few-kilobyte blueprints (2.2×–102.8× compression), deterministic
reconstruction given a seed, and two empirical findings: (1) Hyper-SD's
8-step distilled schedules *beat* their 30-step counterparts on SD 1.5
(+0.54 dB) while running 3.1× faster on CPU, and (2) lower ControlNet
depth scales (0.8 vs 1.5) improve fidelity with the Depth-Anything-V2-Base
stack. SDXL (13.01 dB PSNR, 989 s) is the best result across all backends
on the mandril. We also report a concrete
pixel-metric-vs-perceptual disconnect: HunyuanDiT and FLUX.2-klein score
in the top half by PSNR yet are visually weak backends due to palette
collapse/shift that MSE does not capture. We frame this as a systems study
of a paradigm, not a JPEG replacement, and we are explicit about its
limitations—decoder dependency, compute cost, per-backend palette
fidelity, and lossy-by-design semantics.

Planned work tracked in `TODO.md` includes per-region hue transfer (a fix
for the SDXL hue-distribution drift documented in §4.6) and a captioner
upgrade (§4.6). The MAP_SIZE 128→256 bump, ControlNet scale tuning,
brightness-clamp gamma fix, and all turbo distillation backends are done.

# 7. Reproducibility

The project is self-contained and engineered to run on commodity hardware.
Setup (Python 3.12 via `uv`):

```bash
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
# On non-Apple platforms, uninstall the non-functional MLX stub wheels:
pip uninstall -y mlx mlx-vlm mlx-lm
```

Encode and decode:

```bash
# photo -> tiny .brainimg blueprint
python encoder.py samples/real.jpg -o out.brainimg --seed 42

# blueprint -> regenerated image (CPU fp32, best fidelity)
python decoder.py out.brainimg -o recon.png --device cpu

# CUDA fp16 (fast), or CPU int8 (low RAM)
python decoder.py out.brainimg -o recon.png --device cuda
python decoder.py out.brainimg -o recon.png --device cpu --quantize
```

The format and colour post-processing modules are deliberately ML-free and
can be tested without downloading models:

```bash
pytest            # tests/test_format.py + tests/test_color.py + tests/test_flux_config.py, runs in seconds
```

Known gotchas (from `AGENTS.md`): the MPS fp16 NaN bug is *not* a bug to
"fix" by switching to fp16—int8 quantization is the intended workaround
(Apple Silicon only; not relevant on the AMD CPU target). The Hyper-SD
FLUX LoRA was trained on base FLUX.1-dev, not the Control variants; the
decoder strips the shape-incompatible `x_embedder` / `context_embedder`
deltas before loading (§3.3.4). Re-decoding with the same seed reproduces
the image exactly.

# Appendix A. Archived measurements (retired Lenna sample)

The measurements in §4.7–§4.9 were originally conducted on the Lenna test
image (`samples/lenna.tiff`, 512×512, seed 916570520). Lenna was removed
from the sample set prior to publication per the well-known objection
documented at <https://en.wikipedia.org/wiki/Lenna>; the numbers below are
retained for provenance and to document how the headline findings changed
(or did not) when the sample set moved to the four USC-SIPI images.

## A.1 Per-backend fidelity and speed (Lenna, 512×512)

| Backend | Steps | Time (s) | MSE ↓ | PSNR (dB) ↑ | MAE ↓ |
|---|---:|---:|---:|---:|---:|
| SD 1.5 (old scales) | 30 | ~180 | 8763 | 8.70 | 77.5 |
| SD 1.5 (tuned scales) | 30 | 156 | 7560 | 9.35 | 70.8 |
| SD 1.5 turbo (Hyper-SD, tuned) | 8 | **50** | 7055 | 9.65 | 68.1 |
| SDXL | 30 | 220 | 5774 | 10.52 | 58.8 |
| SDXL turbo (Hyper-SD) | 8 | **69** | 6085 | 10.29 | 61.1 |
| Z-Image (depth-only) | 8 | 237 | 7651 | 9.29 | 70.3 |
| Qwen-Image (depth-only) | 50 | 1436 | 6810 | 9.80 | 68.4 |
| SANA (HED/canny, scale 0.4) | 20 | 52 @ 1024² | 6633 | 9.91 | 64.2 |
| HunyuanDiT (depth+canny, 1024²) | 25 | 1004 | 2977 | 13.39 | 44.3 |
| FLUX.2-klein (img2img, 512²) | 4 | 240 | 2736 | 13.76 | 41.9 |
| FLUX.1-Depth-dev (FP8) | 30 | 654 | 3202 | 13.08 | 43.6 |
| **FLUX.1-Depth-dev turbo (FP8)** | 8 | **166** | **2314** | **14.49** | **37.1** |

On Lenna, FLUX.1-Depth-dev turbo was the best result (14.49 dB) and the
distilled schedule beat the 30-step path on both SD 1.5 (+0.95 dB) and
FLUX (+1.41 dB). On the mandril (§4.7), SDXL is best (13.01 dB) and the
FLUX distilled schedule does *not* beat the 30-step path. The
distilled-schedule-wins finding is therefore image-dependent, not a
universal property of the distillation.

## A.2 ControlNet scale sweep (Lenna + test512)

The original sweep covered `samples/lenna.tiff` and `samples/test512.jpg`.
Findings: depth 1.0 beats 1.5 on both; 0.8 beats 1.0; 0.6 beats 0.8 on
test512 but hurt Lenna. Seg 1.2 beat 0.9 on Lenna; 0.9 beat 1.2 on test512.
The 0.8/1.0/1.0 compromise was chosen as robust across both. The mandril
sweep (§4.8) confirms 0.6 edges out 0.8 by 0.09 dB, but 0.8 is retained
for cross-sample robustness.

## A.3 MAP_SIZE regression (Lenna)

The 128→256 MAP_SIZE bump regressed on every backend on Lenna at 512×512:
SD 1.5 30-step −0.65 dB (8763 → 10185 MSE), SD 1.5 turbo −0.85 dB
(7934 → 9640), SDXL turbo −0.57 dB (6085 → 6928). File size grew 2.5×
(7.9 KB → 19.7 KB).

# References

The references below are to canonical works and projects used by brainimg.
They are given in a lightweight author–year–title form suitable for a
Markdown draft; a submission version would expand them into a `.bib`.

- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. (2022).
  *High-Resolution Image Synthesis with Latent Diffusion Models* (Stable
  Diffusion, arXiv 2112.10752). CVPR 2022.
- Zhang, L., Rao, A., Agrawala, M. (2023). *Adding Conditional Control to
  Text-to-Image Diffusion Models* (ControlNet, arXiv 2302.05543). ICCV 2023.
- Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H. (2024).
  *Depth Anything V2* (Depth-Anything-V2-Base, arXiv 2406.09414). NeurIPS 2024.
- Jain, J., Li, J., Chiu, M.-T., et al. (2023). *OneFormer: One Transformer to
  Rule Universal Image Segmentation* (arXiv 2211.06220, CVPR 2023;
  `shi-labs/oneformer_ade20k_swin_tiny`).
- Wang, P. et al. (2024). *Qwen2-VL: Enhancing Vision-Language Model's
  Perception of the World at Any Resolution* (arXiv 2409.12191).
  `mlx-community/Qwen2-VL-2B-Instruct-4bit` (Apple Silicon MLX 4-bit
  captioning) and `Qwen/Qwen2.5-VL-7B-Instruct` (transformers CPU/CUDA
  fallback).
- Apple MLX framework and `mlx-vlm` (Apple Silicon 4-bit captioning).
- HuggingFace `diffusers` library; `transformers` library.
- HuggingFace `optimum-quanto` (int8 quantization for MPS/CPU).
- `stabilityai/sd-vae-ft-mse` (fine-tuned SD 1.5 VAE).
- `stabilityai/stable-diffusion-xl-base-1.0`; `diffusers/controlnet-{depth,canny}-sdxl-1.0`;
  `abovzv/sdxl_segmentation_controlnet_ade20k` (SDXL stack).
- `lllyasviel/control_v11f1p_sd15_depth`,
  `lllyasviel/control_v11p_sd15_canny`,
  `lllyasviel/control_v11p_sd15_seg` (SD 1.5 ControlNets).
- `Tongyi-MAI/Z-Image-Turbo` (6 B-parameter DiT distilled for 8-step
  inference) with `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`
  (the full `2.1-8steps` variant; the `*lite*` 2601/2602 files are rejected
  by diffusers 0.38 due to a widened `control_all_x_embedder` input dim).
- `black-forest-labs/FLUX.1-Depth-dev` and `black-forest-labs/FLUX.1-Canny-dev`
  (FLUX.1 guidance-distilled Control variants; channel-concat
  conditioning). `black-forest-labs/FLUX.1-dev` (the underlying 12 B
  MMDiT + T5-XXL base) — gated on Hugging Face; both FLUX Control
  variants carry the FLUX.1-dev non-commercial license.
- Ren, Y., Xia, X., Lu, Y., et al. (2024). *Hyper-SD: Trajectory Segmented
  Consistency Model for Efficient Image Synthesis* (arXiv 2404.13686).
  ByteDance/Hyper-SD LoRAs for SD 1.5, SDXL, and FLUX.1-dev (8-step
  distillation).
- Wu, C., Li, J., Zhou, J., et al. (2025). *Qwen-Image Technical Report*
  (arXiv 2508.02324). `Qwen/Qwen-Image` (Apache 2.0 DiT) with
  `InstantX/Qwen-Image-ControlNet-Union` (Union ControlNet, Apache 2.0).
- Li, Z., Zhang, J., Lin, Q., et al. (2024). *Hunyuan-DiT: A Powerful
  Multi-Resolution Diffusion Transformer with Fine-Grained Chinese
  Understanding* (arXiv 2405.08748). `Tencent-Hunyuan/HunyuanDiT-v1.2-
  Diffusers-Distilled` and `HunyuanDiT-v1.2-Diffusers` (non-distilled) with
  separate `HunyuanDiT-v1.2-ControlNet-Diffusers-{Depth,Canny}`
  (tencent-hunyuan-community license).
- Xie, E., Chen, J., Chen, J., et al. (2024). *SANA: Efficient
  High-Resolution Image Synthesis with Linear Diffusion Transformers*
  (arXiv 2410.10629, MIT). `Efficient-Large-Model/Sana_600M_1024px_diffusers`
  + `ishan24/Sana_600M_1024px_ControlNet_diffusers` (community diffusers
  conversion of the official NVlabs HED ControlNet).
- Black Forest Labs. *FLUX.2-klein-4B* (Apache 2.0, ungated, 4 B
  rectified-flow transformer, 4-step distilled img2img).
  `black-forest-labs/FLUX.2-klein-4B`.
- `peft` (Parameter-Efficient Fine-Tuning library, HuggingFace) for LoRA
  loading on the turbo paths.
- Wallace, G. K. (1992). *The JPEG Still Picture Compression Standard.*
  Communications of the ACM.
- WebP / VP8 intra-frame specification (Google).
- AVIF / AV1 intra-frame specification (Alliance for Open Media).
- JPEG 2000 (wavelet/fractal) standard.
- Olshausen, B. A., Field, D. J. (1996). *Emergence of simple-cell receptive
  field properties by learning a sparse code for natural images* (sparse
  coding / brain motivation).
- Rao, R., Ballard, D. (1999). *Predictive coding of natural images*
  (predictive coding / brain motivation).
- Hubel, D., Wiesel, T. (1959/1962). *Receptive fields of single neurons in
  the cat's striate cortex* (V1 feature extraction motivation).
- Classic retinal compression / lateral inhibition accounts (visual
  neuroscience, motivational background only).