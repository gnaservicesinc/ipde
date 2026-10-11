# Compact native material prototype — October 10, 2026

Texture Studio now provides opt-in **Compact scalar** and **Compact normals** families. They train standalone weights with Apple MPSGraph on Metal, using the existing dataset preparation, sample recovery, epoch/step scheduling, validation and checkpoint lifecycle. They do not download or refine PBRnxt weights. The existing PBRnxt family remains available, and older preferences retain it.

## Try it

Open **Model Training**, choose **Compact scalar** for displacement/height or roughness, or **Compact normals** for normals. Select **512 × 512**, choose the dataset and target, and train. Switching from PBRnxt sets the directly editable learning rate to **0.001**; subsequent edits are preserved. Each run trains one target. Compact models export complete `model.safetensors` weights in both normal and developer modes. Saved models support continued training, comparison, export and application in Texture Studio. The comparison baseline is the checkpoint's exact seeded **Untrained initialization**.

Keep entire material subjects out of training for the quality trial. Automatic extra-crop validation is a diagnostic on known subjects, so it does not establish generalization to unseen materials. A few hundred distinct examples may suffice for a useful model within their represented material types; this remains a hypothesis. Multiple crops provide detail supervision without adding independent material subjects. This prototype is not quality-qualified on a real held-out dataset.

## Architecture and numeric contract

| Family | Target | Input channels | Output channels | Width 32 parameters | Width 16 parameters |
| --- | --- | ---: | ---: | ---: | ---: |
| Compact scalar | Height or roughness | 3 | 1 | 999,937 | 256,257 |
| Compact normals | Normal | 3 | 3 | 1,000,515 | 256,547 |

Width 32 is the UI default. The native trainer also accepts `--compact-width 16` for a smaller trial. Checkpoint loading derives its recorded width; the network cannot silently change width or target on a warm start.

There are three downsampling levels, channels 32→64→128→256, seven NAF-style blocks (one per encoder/decoder level and one bottleneck), skip additions and a single target-specific linear head. Upsampling uses a channel-reducing 1×1 convolution and differentiable nearest-neighbor repetition. Layer normalization operates over channels at each pixel; channel attention averages the spatial dimensions. Residual gates initialize to 0.1 so internal block weights receive gradients immediately. The backbone and head are fully trainable. RGB diffuse input remains useful even when its target is scalar.

This smaller task network adapts the [NAFNet block design](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py), with its [MIT notice](https://github.com/megvii-research/NAFNet/blob/main/LICENSE) retained in the app and compact packages. It does not load the authors' restoration weights, retain their RGB residual output, use pixel shuffle, or perform their automatic input padding. The numeric input grid, target codes and declared normal convention follow the existing native material contract. Internal feature downsampling does not resize stored sources or targets. Float32 predictions are exported directly without per-image normalization, stretching, color conversion or clamping. Normal components remain encoded data values, without automatic vector renormalization.

The schema is `texture-studio-compact-material-v1`, with separate `texture-studio-compact-scalar-native-v1` and `texture-studio-compact-normal-native-v1` architecture identities. Checkpoints contain complete named F32 tensors and a deterministic initialization seed/hash. Inspection binds target, dimensions, width, tensor shapes, output channel count and provenance. Invalid weights cannot become a warm start. Compact checkpoints cannot load as PBRnxt LoRA or participate in adapter mixtures. Packages include exact hashes and the Texture Studio/NAFNet notices; no base, interpreter or source images are included.

## Measured 512 throughput

The [JSON receipt](compact-native-material-prototype-2026-10-10.json) records six successful full-weight scalar training updates on an Apple M2 Max, width 32, Float32, optimized arm64 Release test host. The fixture has three distinct native 512×512 procedural RGB/16-bit scalar pairs, trained for two epochs. This was a short execution probe rather than a quality fit.

- First step, including graph compilation: **1.9506 s**.
- Five warmed steps: **0.2142–0.2180 s**, mean **0.2167 s**.
- Complete training, checkpoint and package probe: **3.1390 s**.
- Data preparation: about **3.2–3.5 ms**; CPU optimizer: about **14–20 ms** per step.
- Test-host lifetime peak resident memory: **0.379 GiB**. Metal allocated memory at completion: **0.438 GiB**. These overlap and are not physical-footprint or peak-Metal measurements.

The earlier PBRnxt 512 changed-input step was **5.7065 s**, approximately **26×** this warmed scalar step. These runs have different architectures, training scope and fixtures; this is not a quality-equivalent controlled comparison. Scalar timing does not establish normal timing, convergence, larger-grid viability, or a safe concurrent-training count. The compact backend's compilation and execution wall time are combined in its cold step; its zero `compilationSeconds` counter is not a measured zero compilation cost.

## Verification

The first focused Release batch passed **69 tests, one opt-in UI-control skip, zero failures**. It covered all three targets through real gradient updates, standalone export and inference; height warm start; exact seeded baseline comparison; grayscale versus three-channel output; all-weight autodiff with finite differences; early/middle/head weight updates; exact save/reload predictions; strict metadata/tensor validation; package/Hub transfer; UI selection/persistence/handoffs; and final validation corruption followed by successful checkpoint publication. Subsequent checkpoint boundary checks additionally bind baseline overrides and inference receipts to validated checkpoint identities.

The complete Release regression suite finished on October 11 with **406 tests, five optional skips, zero failures**, in 556.800 seconds of test execution. It includes the additional baseline identity/digest regressions and existing PBRnxt numerical, optimizer, recovery, dataset, package and UI tests. Only the isolated XCTest host disables hardened runtime to load the test bundle.

Normal `make package`, native build-tool checks, strict recursive code-signature verification, interpreter-free bundle checks and source/release version checks passed. **Texture Studio 0.9.13 build 108** is installed at `/Applications/Texture Studio.app`, including all four material tools. Recursive installed/built byte comparison found no differences. The installed executable's `--smoke-test` passed. Production bundles use hardened runtime and ad-hoc signatures. Binary/archive/source hashes are retained in the JSON receipt.

Reproduce the compact throughput probe with `TEST_RUNNER_TEXTURE_STUDIO_COMPACT_BENCHMARK=1 xcodebuild -project src/TextureStudio/TextureStudio.xcodeproj -scheme TextureStudio -configuration Release -destination 'platform=macOS,arch=arm64' -derivedDataPath build/TransparencyTrainingValidation ENABLE_TESTABILITY=YES ENABLE_HARDENED_RUNTIME=NO -only-testing:TextureStudioTests/NativeCompactMaterialTrainerTests/testCompact512TrainingBenchmark test`. Omitting the environment variable makes the throughput probe an optional skip. The regression result bundle is `build/TransparencyTrainingValidation/Logs/Test/Test-TextureStudio-2026.10.10_23-51-09--0400.xcresult`.

## Quality expectation

This is an experiment in reducing training cost. Its random initialization carries no pretrained material prior. Success on a narrow distribution is plausible; reliable prediction across arbitrary material classes remains uncertain. Relevant published [single-image SVBRDF work](https://www-sop.inria.fr/reves/Basilic/2018/DADDB18/) uses broad procedural material supervision, varied lighting and rendering-aware losses to address ambiguity. The compact prototype uses existing paired native maps and value/detail losses, so those research results are not evidence of its accuracy. Judge the trial on unseen material subjects, reference maps and rendered relief/roughness, alongside validation error.
