# Decode quality TODO

Tracked improvements for the brainimg decoder. Tier 1 (color matching, VAE
swap, steps bump, style prefix, tunable CLI flags) is done — see commit
`ef4e1f9`. Remaining work, in priority order.

## Tier 2 — moderate lift (next up)

- [ ] **Raise MAP_SIZE 128 -> 256.** Sharper conditioning maps (depth/canny/seg)
      -> better structural fidelity at decode. File grows ~2-3x but stays in
      the low-KB range. Requires re-encoding existing samples. Touch
      `brainimg/format.py` `MAP_SIZE`; no decoder change (maps already upscale
      to target size).
- [ ] **Tune ControlNet scales / CFG for the new stack.** The current defaults
      (depth 1.5, canny 1.2, seg 0.9, cfg 7.5) were set for the old
      Depth-Anything-Small + no-seg pipeline. The CLI flags now exist
      (`--depth-scale`, `--canny-scale`, `--seg-scale`, `--cfg`); run a small
      sweep on `samples/lenna.tiff` + `samples/test512.jpg` and pick better
      defaults for Depth-Anything-V2-Base + seg. Then update the module
      constants in `brainimg/generate.py`.
- [ ] **Brightness clamp edge case.** The `[0.5, 2.0]` gain clamp in
      `_match_color_statistics` can't reach extreme targets (e.g. darkening
      210 -> 80 needs ratio 0.38, clamped to 0.5). Consider a per-channel
      gamma or a wider clamp for out-of-range cases, weighed against clipping
      artifacts.

## Tier 3 — big lift (separate project)

- [x] **Hyper-SD turbo backends.** `--model sd15-turbo` / `sdxl-turbo` add
      ByteDance's Hyper-SD 8-step distilled LoRA on top of the existing
      SD 1.5 / SDXL base + depth/canny/seg ControlNets. No new base model,
      no schema change -- the LoRA is ~70-150 MB, loaded + `fuse_lora(0.125)`
      + scheduler swapped to `DDIMScheduler(timestep_spacing="trailing")`
      inside `_build_pipeline` (gated by `cfg["turbo"]`). ~4x faster on CPU
      than the 20-30 step non-turbo paths, at a small quality cost. Turbo
      paths ignore the file's stored step count and use 8 steps unless
      `--steps` is passed; `--cfg` defaults stay at 7.5/7.0 (CFG-preserved
      LoRAs support 5-8). The biggest win on the AMD CPU target where every
      step costs the same wall time.
- [x] **Z-Image-Turbo backend.** `--model zimage` adds Tongyi-MAI/Z-Image-Turbo
      (6B bf16 DiT) + alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
      (full 2.1-8steps, depth-only). Differs from the SD path:
        * **Depth-only conditioning.** The Union ControlNet takes one image per
          call; the blueprint's canny + seg maps are ignored (no schema change).
        * bf16 everywhere (sidesteps the MPS fp16 NaN bug); no `optimum.quanto`.
        * guidance_scale 0.0 (Turbo-distilled); 8 steps; Qwen 512-token encoder
          (color_style prefix prepended unconditionally).
        * ~18 GB VRAM on GPU; ~18 GB RAM resident on CPU (no offload trick on
          CPU -- `enable_model_cpu_offload` needs an accelerator). 8 GB Apple
          Silicon should use `sd15`.
        * The *lite* 2.1-2601/2602-8steps files (~2 GB) are rejected by diffusers
          0.38 (widened `control_all_x_embedder`, shape mismatch); the full
          2.1-8steps (~6.4 GB) loads cleanly and is what the code pins.
      Schema-unchanged, encoder-untouched. Verified end-to-end on CPU (pipeline
      constructs, generation runs). Quality vs sd15/sdxl not yet benchmarked.
- [x] **SDXL base model.** Swap `stable-diffusion-v1-5` for
      `stabilityai/stable-diffusion-xl-base-1.0` +
      `diffusers/controlnet-{depth,canny}-sdxl-1.0`. Gated behind
      `--model sdxl` (SD 1.5 stays the default). Much higher quality, but:
        * ~5-10x slower on CPU (~17 min/image at 1024 fp32), heavier download
          (~7 GB base + ~2.4 GB per ControlNet).
        * Default size 1024; Conditioning scales run lower than SD 1.5
          (depth 1.0, canny 0.8, seg 0.6, cfg 7.0).
        * **Seg ControlNet now supported**: `abovzv/sdxl_segmentation_controlnet_ade20k`
          (ungated) loads via `ControlNetModel.from_single_file` since it ships a
          checkpoint-format safetensors, not a diffusers repo layout. The old
          xinsir 401 blocker is gone.
      Verified on lenna.brainimg: 1024x1024 fp32, deterministic (identical md5
      across runs), color stats match targets. Scale tuning still TODO on a GPU.

- [x] **FLUX.1 Control backend.** `--model flux-depth` /
      `--model flux-canny` add Black Forest Labs' FLUX.1 guidance-distilled
      Control variants. Same channel-concat conditioning pattern as Z-Image's
      Union net (one image per call), wired through diffusers'
      `FluxControlPipeline`:
        * **`flux-depth`**: `FLUX.1-Depth-dev` + the blueprint's `depth_map_b64`.
        * **`flux-canny`**: `FLUX.1-Canny-dev` + the blueprint's `canny_map_b64`.
        * bf16 throughout (FLUX is bf16-native; sidesteps the MPS fp16 NaN
          bug); no int8 quant. `--quantize` FP8-quantizes the transformer +
          T5-XXL via `optimum.quanto` (`qfloat8` weights-only), dropping
          resident memory from ~22 GB to ~12 GB.
        * Per-device: cuda resident, mps cpu-offload, cpu resident in host
          RAM (no offload trick on CPU-only boxes, same constraint as
          Z-Image). 8 GB Apple Silicon is not supported -- use `sd15`.
        * Defaults: 30 steps; cfg 10.0 (depth) / 30.0 (canny); max
          sequence length 512 (T5-XXL); control scale 0.85.
        * Non-commercial license (FLUX.1-dev family).
      Schema-unchanged (the other map and the seg map are silently
      ignored, like Z-Image). Same SDXL hue-distribution drift caveat
      applies at small sizes -- prefer `--size 1024x1024` for palette
      fidelity.

## Known issues (not pure decode-quality)

- [ ] **Captioner accuracy on Lenna.** The 7B captioner misidentifies Lenna's
      dark curled hair as "a wide-brimmed straw hat adorned with purple
      feathers." The conditioning maps (depth/canny/seg) capture the true
      structure regardless, but a wrong caption biases generation. Could try a
      larger VLM or ensemble captions; low priority since structure is what
      drives fidelity.

- [ ] **SDXL hue distribution drift.** SDXL @ 512 outputs land in the
      orange/yellow band (60-90 deg) when the source is pink/magenta (330-30
      deg) -- a hue *distribution* shift, not just a mean shift. Brightness
      and saturation post-processing can't fix this; a global HSV-H rotation
      can align means but not reshape distributions, and a large enough
      rotation to chase a different distribution recolors neutrals and skin
      badly. Per-region hue transfer (segmentation mask + per-region target
      hue) or histogram matching (LAB space, per-region) would help; not
      implemented. Workaround: prefer SDXL @ 1024, where the drift is much
      smaller, or accept SDXL's color choice and treat it as "photorealism
      interpretation" rather than a bug.
