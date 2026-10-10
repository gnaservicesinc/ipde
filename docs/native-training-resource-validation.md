# Native resource validation — 2026-10-09

Current throughput work and fresh validation results are recorded in
[the October 10 throughput investigation](validation/native-training-throughput-2026-10-10.md).
The dated measurements below describe their stated source versions.

## Training CPU overhead — 2026-10-10

Training activation checkpoints now retain independent, compact GPU buffers directly within the existing byte cap. Replay reuses those buffers without reading whole-grid activations into CPU `Data` or uploading another copy. Compiler/executable ownership and reverse-consumer eviction remain bounded as before.

Staged execution also reuses its uncompiled symbolic graph, derivative plans and immutable weight/mask uploads. Actual stage compilation uses separate disposable graph owners, released after every execution; loaded executables remain transient. Any failure or cancellation discards the retained metadata, and factor keys, dimensions, precision and byte counts are checked before reuse or rebuilding. Nonstaged execution continues to use a fresh graph.

The worklog uses one file handle and a bounded UTF-8 display buffer. UI delivery batches the complete event stream at 100 ms intervals; the final flush consumes only undelivered events. An optimized Foundation-only comparison of the previous and new log helpers appended 5,000 identical 79-byte Unicode JSON events (395,000 bytes total), with one warmup and three measured trials in alternating order. Mean append time was **1.69438 s before** and **0.010863 s after**, about **156× faster for this logging workload**. Disk bytes matched exactly, and the new helper delivered every pending event once. OS caches were not cleared. Evidence: `docs/validation/native-training-log-2026-10-10.txt`.

This comparison measures log handling only. It does not measure UI rendering, whole training updates or GPU utilization. Current full-size throughput remains unmeasured for these changes; the historical production results below describe their recorded implementations.

The final selected regression run executed **93 tests with two opt-in skips and zero failures**. It covers two-step factor/Adam parity in both scopes across three checkpoint budgets, cancellation recovery, factor contracts, naming through snapshots/exports/library display, progress delivery and stop/deadline behavior. The normal Release build, build-tool checks, native bundle verification and code-signature verification also passed. Evidence and the test command: [October 10 validation record](validation/native-training-tests-2026-10-10.txt). The production 1K/2K benchmark and externally driven control-button test were the opt-in skips.

## Source scanning: measured

Machine: Apple M2 Max, 12 CPU cores, 64 GiB unified memory. Optimized arm64 Swift harnesses exercised the native dataset service against the existing `sources_mats` inventory: **936 PNGs, 31.20 GiB, 224 registered native source sets**. Serial and concurrent runs found the same 35 ignored images and 29 discovery warnings.

| Operation | Observed time |
| --- | ---: |
| Original serial full scan | 19.28 s |
| New full scan, first observed scan (caches not forcibly flushed), 5 workers | 17.63 s |
| New full scan, repeat observation, 5 workers | 4.55 s |
| Import the verified scan preview | 0.95–1.00 s |
| Repeat scan using registered file-state cache | 0.81–0.84 s |
| Initial automatic source discovery on dataset open | 18.47 s |
| Repeated dataset open, unchanged index hash | 0.81 s |

Caches were not forcibly flushed. Whole-process maximum RSS, including import/reuse: serial 365 MiB, new first run 785 MiB, new repeat run 895 MiB, automatic discovery 798 MiB. Every run reported **zero swaps**. The repeated concurrent scan was approximately 4.2× faster than the observed serial scan; cache state was not controlled for this comparison.

Automatic-open validation used APFS copy-on-write clones. Temporary datasets were removed; no originals were written, and source inventories stayed stable. Harnesses, commands and `/usr/bin/time -l` measurements remain in `out/native-training-regression/source-scan/`.

PNG inspection, SHA-256 and provider MD5 audits use bounded workers; cancellation stops queued jobs and joins readers. Deterministic metadata assembly retains native pixels, color variants and provider evidence, and preserves the selected layer setting.

## Behavioral tests: passed

Swift 6 typechecks and diff checks passed. All **26 `NativeMaterialDatasetTests` passed** in the coordinated Debug Xcode run, including:

- `testSourceScanWorkersRespectCPUAndMappedFileMemory`
- `testSourceScanReadsConcurrentlyPreservesOrderAndIsolatesCorruptFiles`
- `testCancellingSourceScanStopsAndJoinsReadersBeforeReturning`
- `testParallelFolderScanPreservesNativeDimensionsVariantsAndImportProof`

From the repository root:

```sh
xcodebuild -project src/TextureStudio/TextureStudio.xcodeproj \
  -scheme TextureStudio -configuration Debug \
  -destination 'platform=macOS,arch=arm64' \
  -only-testing:TextureStudioTests/NativeMaterialDatasetTests test
```

The earlier coordinated `make test-native` run passed **295 tests, with three opt-in skips and zero failures**, in 891.85 seconds. This includes all 26 dataset tests, 14 model tests, and the trainer/preparation Stop, Abort, snapshot and deadline checks. Evidence: `/tmp/ipde-final-native-regression-tests.log`. A focused Release rerun also passed the constructor's negative-grid guard before the production 1K benchmark began. That earlier suite predates the latest linear/GELU and generator-tail changes. The new full suite passed **295 tests, with three opt-in skips and zero failures**, in **953.376 seconds**, on the current integrated model. Permanent filtered results: `docs/validation/native-training-suite-2026-10-09.log`; original full log: `/tmp/ipde-final-vjp-native-tests.log`. The two final-validation Abort checks were added after this suite was built and are verified separately below.

An earlier full run exposed callback fixtures whose identical training and validation pixels reused frozen features and bypassed their intended callbacks. Those fixtures now use distinct pixels; the focused trainer/preparation rerun passed all 24 cases, with two opt-in skips. Physical button-click automation remains unavailable because the native Computer Use pipe failed to start; the controls' state and trainer behavior were tested programmatically.

Two final-validation cancellation checks were added after the current full-suite build. The focused **Release run passed all three boundary tests**, with zero failures in **125.056 seconds**: Abort from final validation prevents checkpoint/export publication; Abort after checkpoint publication preserves that checkpoint and prevents export; Stop completes its active update while Abort discards an unfinished update. Permanent results: `docs/validation/native-training-abort-boundaries-2026-10-09.log`.

## Training implementation and standalone numeric check

Backward replay collects unique stage dependencies and runs them once in forward order, avoiding repeated shared residual branches. Reverse stages request only feeds consumed by their derivative graphs. Constructor-registered vector-Jacobian products (VJPs) cover same-shaped residual additions, frozen convolutions, exact channel slices/concatenations and erf GELU. Adapted linear/GELU pairs now have separate boundaries; their linear factor derivatives still use MPSGraph autodifferentiation. Immutable linear/GELU pairs retain one boundary with an explicit input derivative. Generator-tail branches have separate input and convolution boundaries; the selected tail convolution retains its normal input and LoRA A/B derivatives. The forward operations, full native pixel grid, loss and complete selected adapter scope remain unchanged.

Checkpoint placement uses the dependency graph and retained bytes: it keeps the generator output when it fits and chooses cuts that reduce subsequent recomputation. The plan is reused across updates. The default RAM pool is capped at the smallest of **6 GiB**, one eighth of training capacity, and half the estimated spare capacity. Spare capacity is zero when the working estimate exceeds capacity. For the pinned map-decoder model on this 64 GiB Mac, the current policy selects **6 GiB at both 1K and 2K**. These are checkpoint storage limits, not total process memory estimates or measured peaks.

The October 9 implementation released its cold symbolic graph after each execution. The October 10 changes above reuse only the uncompiled staged graph and continue releasing compiler and executable workspaces. Compiled stage code uses a temporary cache capped at **1 GiB**, removed when the training program is released. Frozen features have a separate **2 GiB** cache limit; no activation files are written. Fresh production benchmarks at both sizes remain pending for the latest cuts and checkpoint policy.

A **standalone small-fixture check**, with the activation checkpoint budget forced to zero, completed two Adam updates in both training scopes. Every trainable factor and Adam first/second moment matched the reference within 0.3% relative tolerance; predictions and loss matched within 2 × 10⁻⁶. The fixture's compiled code cache peaked at approximately 17.8 MB, and Metal allocations were approximately 21 MB after execution returned. This verifies replay numerics on the small fixture; the production measurements below exercise the learned base at its full channel widths.

The combined compiler-owner implementation passed the same checks with a **512 KiB checkpoint limit**, forcing partial checkpoint replay in both scopes. Compiler owners are discarded after eight cache misses or 512 MiB of positive allocation growth, with one job's transient allocation separate from that bound. Normal graph results go directly into independent shared buffers; read-only results that ignore supplied buffers are copied only for the affected slots. The automated numerical test exercises both zero and 512 KiB limits, including every factor and Adam moment over two updates. All 14 model tests passed in the earlier coordinated Xcode run.

After integrating the static VJPs, the standalone 512 KiB check again passed every adapter factor and Adam moment in both scopes over two updates. Duplicate requested outputs also matched direct, cold-package and warm-package execution bit for bit, including signed zero and subnormal values. Evidence: `/tmp/ipde-static-vjp-current-proof.log`; earlier compiler-owner evidence: `/tmp/ipde-compiler-owner-batched-proof.log`. This small-fixture proof does not establish production 2K memory use or speed.

The latest linear/GELU and generator-tail implementation also passed the strict standalone proof: both scopes, zero and 512 KiB checkpoint budgets, and two updates per case. Every factor and Adam first/second moment matched monolithic execution within 0.3% maximum-scaled error, with a 16-subnormal absolute floor; every output pixel and the total/value/gradient losses matched within 2 × 10⁻⁶ absolute tolerance. Evidence: `/tmp/ipde-linear-gelu-hybrid-static-tail-proof.log`. This covers complete and partial replay on the small fixture, not production 1K/2K resources.

A CPU-only audit of the pinned 2K model found 704 stages after these cuts, versus 684 previously. Its largest modeled persistent job fell from 43.84 to 33.34 GiB; with the new 6 GiB checkpoint pool, the plan recomputes 2,743 stages, versus 5,983 with the earlier implementation's 2 GiB pool. These are symbolic storage and work estimates, not observed memory peaks or elapsed times. Backend workspace, heap caching and other runtime allocations can raise actual use. Audit and candidate details: `/tmp/ipde-linear-gelu-tail-candidate-summary.md`.

## Production training resource proof: current-code 1K and 2K pending

The engineering handoff includes verified and installed **0.9.13 build 107**. All five Release app bundles passed deep, strict signature verification with hardened runtime enabled and no XCTest bundle. Their installed binaries matched the built binaries by SHA-256; the installed Material Trainer launched and remained running. The full native suite and focused Abort boundary tests passed as recorded above. Fresh production resource benchmarks are the remaining validation work; the user authorized handing those routine runs to a lighter model once the engineering work was complete.

The saved Mat3 workload uses **200 updates per input/height pair** and a **90-minute training limit**. A JSON-only inventory of its 224 registered source sets, applying the native crop and color-variant rules, gives:

| Native training size | Eligible height source sets | Input/height pairs | Configured updates | Automatic validation pairs |
| --- | ---: | ---: | ---: | ---: |
| 1K | 218 | 229 | **45,800** | 1 |
| 2K | 217 | 228 | **45,600** | 1 |

The 1K-only height source is too small for 2K; six source sets have no height map. Validation uses the saved 2% selection with a maximum of one crop. These are descriptor-derived workload counts, not measured training timings. The time limit can end training before all configured updates complete.

An earlier Release 1K implementation completed one actual map-decoder update at rank 64/alpha 16. Initial validation took 178 s; the first update then took approximately 3,695 s. Peak physical footprint was 53.07 GiB and sampled Metal allocation was 52.02 GiB. System swap decreased by 96 MiB. The old 30-minute scheduling limit ended that run after one update, and the six-update benchmark **failed**. These are diagnostic results, not a passing fix or acceptable training rate. That benchmark's top-level completed count and cached free-space reading were incorrect; the new harness records the actual count and obtains fresh filesystem capacity readings. Artifacts: `out/native-training-regression/run-1A9B3A51-4993-4CC2-AD88-27B92B608082/` and `/tmp/ipde-production-1k-packaged-release.log`.

### Passing 1K measurement before the first static VJPs

The subsequent compiler-owner and checkpoint-placement implementation, using a **2 GiB checkpoint pool**, passed the Release production benchmark on the pinned learned base, map-decoder scope, height target, rank 64/alpha 16. It completed all **six requested updates**, comprising three shuffled passes over two real materials at the exact 1024 × 1024 grid, then validated and exported the package. Source checksums stayed unchanged. Its validation pair repeats the first training material, so this run measures numeric execution and resources rather than material quality.

| Measurement | Observed result |
| --- | ---: |
| Initial full validation | 97.18 s |
| First update, including cold backward compilation | 189.86 s |
| Subsequent five updates | 111.72–123.89 s each; mean 119.10 s |
| Entire six-update run, including validation and export | 901.27 s (15.02 min) |
| Process peak physical footprint | 19.29 GiB |
| Sampled peak Metal allocation | 16.31 GiB |
| System swap change | −8 MiB |
| Available volume capacity change | −0.84 GiB |

The benchmark resource-abort flag remained false. Physical footprint is a process-lifetime high-water mark; Metal allocation was sampled every 100 ms. Swap and free-space readings are system-wide and volume-wide respectively, so their deltas can include unrelated activity. This passing result predates the first static VJPs and the previous adaptive 6 GiB/2 GiB checkpoint policy; it must not be presented as a current-code measurement. Artifacts: `out/native-training-regression/run-79C7597C-859D-4F70-A7E0-B96D548493D0/training-1024.json`, its `trained-1024/run.json`, and `/tmp/ipde-production-1k-owner-anchors-release.log`.

### Passing 1K measurement before the latest linear/GELU and tail changes

The preceding implementation passed a fresh-process Release benchmark with the first static VJPs and **6 GiB checkpoint pool**. It completed all **six requested updates** over the same two real materials, using map-decoder scope, height target, rank 64/alpha 16, and the exact 1024 × 1024 grid. It then completed full validation and exported the package. The benchmark resource-abort flag remained false, and the source checksums stayed unchanged. This result predates the latest linear/GELU and generator-tail cuts and the revised checkpoint policy; it is not a current-code measurement.

| Measurement | Observed result |
| --- | ---: |
| Initial full validation | 92.02 s |
| First update, including cold backward compilation | 137.17 s |
| Subsequent five updates | 59.88–71.25 s each; mean 67.57 s |
| Entire six-update run, including validation and export | 583.71 s (9.73 min) |
| Process peak physical footprint | 25,366,373,040 bytes (23.62 GiB) |
| Sampled peak Metal allocation | 16,777,412,608 bytes (15.63 GiB) |
| System swap change | 0 bytes |
| Persistent report and output files | 114,682,306 bytes (109.37 MiB) |
| Observed temporary compiled-code cache peak | 15.06 MiB |

The temporary compiled-code cache directory was removed after the program was released. The saved report and output files remain for inspection. Available volume capacity increased by approximately 1.26 GiB during the run; because this reading includes unrelated filesystem activity, it does not represent this run's disk consumption. The persistent file total above measures the retained benchmark artifacts directly. Physical footprint is a process-lifetime high-water mark, Metal was sampled every 100 ms, and system swap is shared with other applications.

Artifacts: `out/native-training-regression/run-F9DD3FC5-7EC1-4CD8-AFF1-E481A1AFBA66/training-1024.json`, its `trained-1024/run.json`, and `/tmp/ipde-production-1k-final-release.log`.

This is a **two-material, six-update resource regression proof for that preceding implementation**. It establishes that version's ability to update the selected learned layers, validate, save and export under the measured memory and disk bounds. It does not measure the full **45,800-update Mat3 base-model run**, establish long-term throughput across all materials, or assess resulting model quality. The five warmed updates provide a short-run speed measurement; larger dataset composition, feature-cache reuse, validation and checkpoint frequency can affect sustained throughput. The configured 90-minute limit can end a full-dataset run before its requested updates complete.

### Interrupted 2K diagnostic before the latest changes

The preceding implementation's 2K run used a **2 GiB checkpoint pool**, map-decoder scope, height target, rank 64/alpha 16 and the exact 2048 × 2048 grid. It completed one update in **1,010.48 s (16.84 min)**, then was interrupted while profiling the second update to address replay cost. A three-second sample of that partial warm update placed all 109 worker samples in replay, including 105 in loaded-package execution; it recorded no compiler or owner-reconstruction samples. This local sample identifies replay as a bottleneck but does not give a completed warm-update duration.

The saved diagnostic explicitly records `passing_regression=false`, one completed update out of six requested, and `status=interrupted_for_replay_optimization`. It is **not a passing six-update resource proof**. Artifact: `out/native-training-regression/run-2EC8746D-4544-4DE2-BEBF-DDCA46645BC9/training-2048-diagnostic.json`; profiles: `/tmp/ipde-2k-first-step-sample.txt` and `/tmp/ipde-2k-warm-step-sample.txt`.

### Current-code production measurements: pending

Fresh-process Release benchmarks at **both 1K and 2K** remain pending for the latest linear/GELU and generator-tail implementation with the revised checkpoint policy. Each must complete all six requested updates, validation and export before it is recorded as passing. Current-code update timings, peak memory and retained disk use are not established by the preceding measurements or symbolic audit.

A time-limited partial run reports `stopped` with `stopped_reason=time_limit` and actual/requested update counts. The budget includes baseline and interval validation, excludes model/setup, and allows an active update plus final validation/save to finish.

## Benchmark handoff

Permanent evidence is recorded in `docs/validation/native-training-2026-10-09.json`, with its format in `native-training-evidence.schema.json`. The prior 1K report's metrics are copied exactly and labeled as prior code; its implementation source hash was not captured and remains unavailable. The strict proof records the actual candidate-model and binary hashes separately from the current integrated source hash. Its explicit zero/512 KiB budgets do not test the revised production checkpoint policy.

Run 1K first, then 2K, using one GPU process at a time. The commands below rebuild the current source in a separate Release test directory and run six updates over two real materials, at map-decoder rank 64/alpha 16. The subshell stops on a failed build or benchmark. The test host disables hardened runtime to load the ad hoc XCTest bundle; this setting is limited to the benchmark build.

```sh
(
  set -e
  cd /opt/ipde/ipde
  benchmark_args=(
    -project src/TextureStudio/TextureStudio.xcodeproj
    -scheme TextureStudio -configuration Release
    -destination 'platform=macOS,arch=arm64'
    -derivedDataPath build/TextureStudioBenchmarkFinal
    ENABLE_TESTABILITY=YES ENABLE_HARDENED_RUNTIME=NO
  )
  xcodebuild "${benchmark_args[@]}" build-for-testing \
    > /tmp/ipde-benchmark-final-build.log 2>&1
  for size in 1024 2048; do
    TEST_RUNNER_TEXTURE_STUDIO_PRODUCTION_TRAINING_BENCHMARK=1 \
    TEST_RUNNER_TEXTURE_STUDIO_PRODUCTION_TRAINING_SIZES="$size" \
    TEST_RUNNER_TEXTURE_STUDIO_PRODUCTION_TRAINING_UPDATES=3 \
    TEST_RUNNER_TEXTURE_STUDIO_PRODUCTION_TRAINING_PAIRS=2 \
    TEST_RUNNER_TEXTURE_STUDIO_PRODUCTION_TRAINING_SCOPE=map-decoder \
    TEST_RUNNER_TEXTURE_STUDIO_PRODUCTION_TRAINING_RANK=64 \
    TEST_RUNNER_TEXTURE_STUDIO_PRODUCTION_TRAINING_ALPHA=16 \
      xcodebuild "${benchmark_args[@]}" \
        -only-testing:TextureStudioTests/NativeMaterialTrainerTests/testProductionExactGridTrainingResourceRegression \
        test-without-building > "/tmp/ipde-production-$size-current-release.log" 2>&1
  done
)
```

Each size must finish with six actual update events, completed status, finite baseline/final validation, changed factors in the selected early/late/decoder branches, verified export, unchanged source hashes, and both resource peaks below the benchmark guard. A guard stop, deadline stop, error or interrupted run remains partial or failed evidence. Copy the completed logs and selected report metrics into `docs/validation/`, preserving SHA-256 hashes; retain per-update durations so the first compilation-heavy update and subsequent updates can be compared. Physical footprint and sampled Metal allocation are separate readings of unified-memory use and must not be added together.
