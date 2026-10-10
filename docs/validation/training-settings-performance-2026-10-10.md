# Training settings and throughput validation — 2026-10-10

Host: Apple M2 Max, 64 GiB unified memory, macOS 27.2, Xcode 27.2.

## Installed-model engine measurements

An optimized standalone Swift harness loaded the installed revision-pinned PBRnxt weights with SHA-256 `3f25b03e950c6199b53a3e1581296831e71555e1928ad209232b757f75153b7d`. Every update evaluated a complete 1024 × 1024 height grid in Float32. No source maps, selected dimensions or learned operations were reduced. The harness uses deterministic RGB/reference arrays and the common legacy Adam graph update at `1e-4` to isolate staging/checkpoint changes. The application separately tests its new compact Adam/AdamW update and gradient accumulation.

| Scope and adapter | Previous first update | Updated first update | Previous repeated-input update | Updated repeated-input update |
| --- | ---: | ---: | ---: | ---: |
| Map decoder, rank 64 / alpha 16 — saved user scope/grid/adapter | 216.869 s | 173.109 s | 51.645 s | 40.021 s |
| Map output branch, rank 8 / alpha 8 | 121.830 s | 77.161 s | 11.781 s | 11.873 s |

The saved decoder configuration improved by about 20.2% on its first update and 22.5% on its repeated-input update. The output-branch first update improved by 36.7%; repeated-input output-branch throughput was effectively unchanged. First updates include lazy compilation and package loading, while repeated inputs can reuse only adapter-independent features. These are individual observations, not averages or throughput guarantees. The earlier baseline cold runs overlapped some CPU-only application compilation; the updated decoder's first two measurements ran without an Xcode build. Cache state was not forcibly flushed.

A third updated decoder update used different RGB and a different feature identity. It took 45.292 s, demonstrating a complete changed-input update after compilation. That observation overlapped CPU-only Xcode compilation and has no paired previous-code measurement. Losses on the first two identical-input updates agreed to the harness's printed Float32 precision in both scopes; automated numerical regressions separately compare every adapter factor and optimizer moment.

Frozen-block coalescing reduced stage count from 704 to 288 for the decoder configuration and from 692 to 116 for the output branch. Blocks containing adapters or receiving adapter-dependent inputs retain fine backward boundaries. Activation checkpoint storage is capped at the smallest of 12 GiB, one quarter of safe training capacity and half the estimated spare capacity. Compiler/executable arenas still release after use; frozen-feature storage remains separately capped at 2 GiB.

Previous decoder measurement: two updates, maximum RSS 2,874,802,176 bytes, macOS peak memory footprint 20,322,208,672 bytes. Updated decoder measurement: three updates, maximum RSS 2,892,709,888 bytes, peak footprint 26,145,835,888 bytes. Both reported zero swaps. These are process-wide high-water measurements across different update counts, not per-update allocation or an equal-length peak comparison. The larger activation budget spends additional available memory to reduce replay.

Local harness sources, model-code snapshots, executable binaries and complete logs are under `out/training-throughput-audit/`. The measured commands were:

```sh
/usr/bin/time -l out/training-throughput-audit/baseline-benchmark 1024 2 map-decoder 64 16
/usr/bin/time -l out/training-throughput-audit/coalesced-benchmark 1024 3 map-decoder 64 16
/usr/bin/time -l out/training-throughput-audit/baseline-benchmark 1024 2 final-map 8 8
/usr/bin/time -l out/training-throughput-audit/coalesced-benchmark 1024 2 final-map 8 8
```

The smaller output-branch updated measurement preceded the checkpoint-budget increase. The decoder measurement includes both performance changes. No image-quality acceptance, 2K throughput, or exact upstream training equivalence is claimed by these engine measurements.

## Configuration and controls

Focused Release tests verified AdamW decay, coupled Adam, moment updates and bias correction, averaged accumulation before clipping, scheduler boundaries and parser rejection of invalid settings. Store tests verified every UI option reaches the native command and preferences persist. Hosted AppKit field-editor tests verified that focusing both automatic and stored model names preserves the text and that edits persist immediately. The focused test initially exposed a focus echo that pinned the automatic name; the corrected field and layout rerun passed all three tests.

## Final regression and bundle results

The complete Release XCTest suite passed: **351 tests, three opt-in skips, zero failures**, in 386.167 seconds. All 18 model tests passed, including two-step comparisons of every adapter factor and Adam moment across both scopes and zero/partial/full checkpoint budgets, raw-gradient equivalence, cancellation recovery and frozen-block staging. Trainer integration tests passed full/partial accumulation, saved configuration, Stop & Save, Abort and deadline boundaries.

The skips were the original brown-wall RAW fixture, real-dataset 1K/2K production training qualification, and externally driven control buttons. The installed pinned-weight engine benchmarks above were run separately; they are not a substitute for image-quality or real-dataset 2K qualification.

The final small UI clarification disables new rank/alpha settings when refining a recorded LoRA. It followed the full-suite build and passed a separate fresh Release build of the two name-editor tests and minimum-window action-bar test: **three tests, zero failures**. No backend code changed after the complete-suite build.

Commands:

```sh
xcodebuild -project src/TextureStudio/TextureStudio.xcodeproj -scheme TextureStudio \
  -configuration Release -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath build/TrainingSettingsValidation \
  ENABLE_TESTABILITY=YES ENABLE_HARDENED_RUNTIME=NO build-for-testing
xcodebuild -project src/TextureStudio/TextureStudio.xcodeproj -scheme TextureStudio \
  -configuration Release -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath build/TrainingSettingsValidation \
  ENABLE_TESTABILITY=YES ENABLE_HARDENED_RUNTIME=NO test-without-building
xcodebuild -project src/TextureStudio/TextureStudio.xcodeproj -scheme TextureStudio \
  -configuration Release -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath build/TrainingSettingsFinalUI \
  ENABLE_TESTABILITY=YES ENABLE_HARDENED_RUNTIME=NO \
  -only-testing:TextureStudioTests/TrainingModelNameFieldTests \
  -only-testing:TextureStudioTests/TrainingWorkbenchLayoutTests test
make package
```

The testability and hardened-runtime overrides applied only to the separate test builds. The production Release package used normal project settings. Build-tool/installer regressions, version agreement, built and installed bundle checks, deep strict code signatures, installed native smoke test and a byte-for-byte built/installed directory comparison passed. The final build was installed at `/Applications/Texture Studio.app`, including all four material tools. Local package SHA-256: `e313cc65501c30aaa59edd2ba59904ed5c20cd9caf6d022f1fc4f54a3edc61cd`.

Full local test/build logs are `/tmp/ipde-training-final-native-tests.log`, `/tmp/ipde-training-final-ui-tests.log`, `/tmp/ipde-training-final-ui-package.log` and `/tmp/ipde-training-config-build-tools.log`. Result bundles remain under `build/TrainingSettingsValidation/Logs/Test/` and `build/TrainingSettingsFinalUI/Logs/Test/`.
