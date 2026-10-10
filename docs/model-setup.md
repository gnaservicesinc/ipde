# Native material models

The material workbench implements the pinned PBRnxt SCUNetV2 and RRDB mapping in Apple MPSGraph. Training and inference use Float32 with reduced precision fast math disabled. All learned decoder and output-branch operations are retained; the final enlargement is omitted so predictions use the declared native grid. No alternative model or reduced resolution is silently substituted.

Open the tool's settings to select a working folder and model weights. Download Base Model obtains the pinned 349,493,406-byte PBRnxt tensor archive at revision `73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35`, with SHA-256 `3f25b03e950c6199b53a3e1581296831e71555e1928ad209232b757f75153b7d`. The native data parser reads only the expected ZIP tensor state dictionary and rejects executable object constructors. Architecture implementation and original license notices are shipped with the application.

A LoRA requires its exact recorded base. A full fused material checkpoint contains the complete learned weights and can run after the separately downloaded base is removed. Export packages contain safetensors, configuration, checksums and license notices. They carry no executable model sources or source images.

Hugging Face account, catalog, download and upload requests use HTTPS through URLSession. Save an access token in settings; the app stores it in Apple Keychain. `HF_TOKEN` is also available for development automation. Model downloads bind to an exact revision and verify the complete package before publishing it locally. Uploads verify weights, destination and visibility before creating the commit.

The migration is checked with synthetic complete-operation Metal fixtures, real dataset metadata and exact numeric export fixtures. Full pretrained-weight quality, peak memory and production-grid throughput must be measured with the actual pinned model installed. Smaller fixture results are not an image-quality or full-model performance claim.

## Upstream training resolution

At the pinned revision, the published [base-generator training script](https://github.com/aaf6aa/PBRnxt/blob/73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35/train.py#L325-L356) uses 128 × 128 random patches. The separate [4× upscaler script](https://github.com/aaf6aa/PBRnxt/blob/73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35/train_sr.py#L264-L297) uses 192 × 192 targets and 48 × 48 inputs. These published configurations do not establish the complete training history of `pbrnxt_402236.pth`.

The [pinned upstream README](https://github.com/aaf6aa/PBRnxt/blob/73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35/README.md#showcase) demonstrates 256 × 256 inputs and benchmarks 512 × 512 inputs; it gives no preferred 1K or 2K training resolution. Texture Studio's adaptation omits the original final 4× enlargement. Its best training grid therefore needs held-out material comparisons, including the pixel scale of surface details, rather than assuming an upstream ideal canvas size.

## Training configuration

Model Training exposes learning rate, gradient accumulation, AdamW or Adam, weight decay, constant or cosine scheduling, the minimum learning rate ratio, warmup updates, optimizer betas and epsilon, gradient clipping, random seed, LoRA rank and alpha. It remembers these choices and records the effective configuration in checkpoints, exports and run records. Physical batch size is one complete native map; accumulation averages several map gradients before updating the adapter. Larger accumulation increases work per optimizer update.

The native defaults preserve the previous learning rate and update behavior: `1e-5`, accumulation `1`, AdamW with decay `0`, betas `0.9/0.999`, epsilon `1e-8`, maximum gradient norm `1`, a constant schedule and no warmup. AdamW applies weight decay separately from moment estimates when a nonzero decay is selected. See [PyTorch's AdamW algorithm](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).

These are Texture Studio adapter defaults. They do not claim to reproduce the upstream full-generator training recipe, which uses a different optimizer, losses, batch size and precision. See the [training audit](training-audit.md) for the source comparison, fixed choices and performance limits.
