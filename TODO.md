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

- [ ] **SDXL base model.** Swap `stable-diffusion-v1-5` for
      `stabilityai/stable-diffusion-xl-base-1.0` +
      `diffusers/controlnet-{depth,canny}-sdxl-1.0`. Much higher quality, but:
        * ~5-10x slower on CPU, heavier download (~7 GB base).
        * Different default size (1024) and scale ranges.
        * **Loses the seg ControlNet** — no reliable SDXL ADE20K seg net
          (`xinsir/controlnet-sdxl-segmentation-ade20k` returned 401 on the
          Hub). Would drop back to 2-ControlNet unless a seg net is found.
      Worth gating behind `--model sdxl` rather than replacing SD 1.5.

## Known issues (not pure decode-quality)

- [ ] **Captioner accuracy on Lenna.** The 7B captioner misidentifies Lenna's
      dark curled hair as "a wide-brimmed straw hat adorned with purple
      feathers." The conditioning maps (depth/canny/seg) capture the true
      structure regardless, but a wrong caption biases generation. Could try a
      larger VLM or ensemble captions; low priority since structure is what
      drives fidelity.
