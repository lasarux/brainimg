# brainimg — Licensing Notice

## Code

The brainimg source code in this repository (the `brainimg/` package,
`encoder.py`, `decoder.py`, `scripts/`, `tests/`, and the documentation) is
licensed under the **MIT License** — see [`LICENSE`](LICENSE).

Copyright (c) 2026 Pedro A. Gracia Fajardo <lasarux@gmail.com>.

## Model weights are NOT covered by this license

brainimg's decoder does **not** bundle model weights. On first run it downloads
pretrained checkpoints from Hugging Face, each under its **own** license that
the end user must accept separately (some repos are *gated* and require
`huggingface-cli login`). The MIT license above applies only to this
repository's code and documentation; it grants no rights to the model weights
or to images produced by them.

The model licenses constrain how the **output** of a decode can be used,
independently of this code license. In particular:

| Backend (`--model`)        | Model license                  | Commercial use of output |
|----------------------------|--------------------------------|--------------------------|
| `sd15`, `sd15-turbo`       | CreativeML Open RAIL-M         | Yes, with use-based restrictions |
| `sdxl`, `sdxl-turbo`       | CreativeML Open RAIL-M (SDXL)  | Yes, with use-based restrictions |
| `zimage`                   | Apache 2.0                     | Yes (clean) |
| `qwen-image`               | Apache 2.0                     | Yes (clean) |
| `sana`                     | MIT                            | Yes (clean) |
| `flux2-klein`              | Apache 2.0                     | Yes (clean) |
| `hunyuan` / `hunyuan-full` | tencent-hunyuan-community      | Research / non-commercial-leaning |
| `flux-depth`, `flux-canny`, `flux-*-turbo`, `flux-union` | FLUX.1-dev non-commercial | No |
| `sd35`                     | Stability AI community license | No (low-revenue cap) |

If you need a commercially unencumbered decode, use one of the Apache-2.0 / MIT
backends (`zimage`, `qwen-image`, `sana`, `flux2-klein`). The default `sd15`
backend is CreativeML Open RAIL-M: it permits commercial use but carries
use-based restrictions (e.g. no malware, no deception) that are *not* OSI-open.

## Dependencies

The Python dependencies listed in `requirements.txt` (PyTorch, diffusers,
transformers, MLX, OpenCV, etc.) are each under their own licenses — see their
respective repositories. None of them are bundled here.
