# Native training throughput investigation — October 10, 2026

## Reproduction

Machine: Apple M2 Max, 12 CPU cores, 64 GiB unified memory, macOS 27.2.
The synthetic engine measurements use an optimized arm64 Release XCTest host, the installed
pinned `pbrnxt_402236.pth` base (SHA-256
`3f25b03e950c6199b53a3e1581296831e71555e1928ad209232b757f75153b7d`),
and the height target. The original and intermediate 1K comparisons use map-decoder
scope, rank 64, alpha 16, and exact 1024 × 1024 Float32 pixels. Later grid, scope,
rank and execution-policy comparisons state their settings separately. No synthetic
engine measurement resizes or crops its input pixels. The fast-transform numerical
experiment discussed below deliberately permits reduced-precision convolution
intermediates; the throughput rows below retain full Float32 math.

`script/benchmark_native_training.sh` builds and runs the opt-in installed-model
engine benchmark. It performs a first update (cold compilation only with an
empty or disabled code cache), an update using the same input,
and an update with changed pixels. Every update computes gradients and applies
the production CPU optimizer. These synthetic pixels remove PNG
decoding and image-file I/O; compiled-package and OS filesystem caches still
affect execution. The benchmark does not measure material quality.
Use `TEXTURE_STUDIO_BENCHMARK_BUILD`, `TEXTURE_STUDIO_THROUGHPUT_REPORT`,
`TEXTURE_STUDIO_THROUGHPUT_SIZE`, `TEXTURE_STUDIO_THROUGHPUT_STEPS`,
`TEXTURE_STUDIO_THROUGHPUT_SCOPE`, `TEXTURE_STUDIO_THROUGHPUT_RANK`, and
`TEXTURE_STUDIO_THROUGHPUT_ALPHA` to choose isolated paths and settings.
`--no-build` uses the previously built test host.

Each measured run uses a fresh process with one GPU training workload and no
concurrent build. OS caches are not forcibly flushed. Reported physical footprint
is the app process lifetime high-water mark obtained with `task_info`, not
`time`'s measurement of the parent `xcodebuild` process. Metal allocation and
physical footprint describe overlapping unified memory and must not be added.

## Diagnosis and intermediate experiments

The instrumented original engine at `74a8d6e` took 165.67 s for the cold update,
39.94 s for the same-input warmed update, and 45.51 s for the changed-input
update. Compilation accounted for 115.49 s of the first update. Peak physical
footprint was 24.68 GiB. The original CPU sample was predominantly graph
compilation during startup. Low utilization at that point is not a measurement
of sustained training throughput.

An initial background test host was terminated by macOS RunningBoard for a CPU
limit violation. Explicit `ProcessInfo` user-initiated activity covers model
execution and the complete training job, with balanced cleanup on every exit.
The baseline benchmark was repeated under the same activity assertion so this
fix did not bias the comparison.

The first asynchronous GPU submission candidate took 153.85 / 40.10 / 45.25 s.
It did not materially improve warmed throughput. Its executable LRU retained
up to 3 GiB but recorded only 11 hits per update. That cache was removed.

Reusing freed checkpoint capacity reduced repeated forward stages from 275 to
152 per update. This intermediate candidate took 146.69 / 34.89 / 39.58 s, with
the same three reported losses as the baseline. It still included the discarded
executable cache, so these are not final-code measurements.

Raw experiment reports and source snapshots are in
`out/training-throughput-audit/`. They include `baseline-activity-1024.json`,
`candidate-1024.json`, `replay-1024.json`, and `candidate-source-sha256.json`.

## Execution changes

- A program-owned ordered Metal stream lets CPU encoding overlap GPU execution.
  At most two stage submissions are outstanding; estimated live bytes constrain
  overlap, and large stages execute alone. Submission ownership preserves input,
  output, and executable storage through completion, including MPS internally
  committing and replacing command buffers.
- Adapter gradients stay in compact GPU buffers throughout the backward pass.
  CPU readback happens after its final GPU completion instead of after every
  reverse stage. Error and cancellation paths join outstanding GPU work before
  releasing its storage.
- Checkpoint selection scores the actual derivative dependencies. Reverse
  traversal reuses freed checkpoint capacity for useful replayed activations,
  subject to the same byte budget. Frozen-feature cache hits refresh LRU order.
- The default checkpoint pool uses at most half the estimated spare training
  capacity and half the machine training limit. Its cap is 8 GiB with adaptive
  active grouping through 1K, 24 GiB for the fine-stage opt-out through 1K, and
  6 GiB for larger grids. The fine-stage policy replaced an additional 12 GiB
  ceiling that forced avoidable recomputation on this 64 GiB Mac. Explicit test
  budgets still force replay; the adaptive workspace reserve is described below.
- Update logs include data preparation, optimizer timing and cumulative engine
  counts/timing, making subsequent slowdowns attributable to a specific phase.

Engine timing categories can overlap. `finalCommandBufferGPUSeconds` measures
only the final command buffer roots; MPS may submit additional buffers internally.
It is not total GPU time or a utilization percentage. Optional IORegistry device
utilization samples are system-wide and include other applications.

## Completed execution candidate measurements

The candidate with freed-capacity reuse, transient executables, GPU convolution
layout conversion enabled, and the larger adaptive checkpoint cap completed all
three engine updates:

| Measurement | Original | Execution candidate |
| --- | ---: | ---: |
| Cold update | 165.67 s | 147.25 s |
| Same-input warmed update | 39.94 s | 34.35 s |
| Changed-input warmed update | 45.51 s | 38.94 s |
| Peak physical footprint | 24.68 GiB | 34.28 GiB |

This individual run reduced the same-input update by 14.0% and changed-input
update by 14.4%. All three losses match the baseline's printed Float32 values.
The candidate recomputed 107 forward stages per update and used at most
19.91 GiB for checkpoints. System swap remained zero. Its cold compilation
still took 116.43 s, which motivates subsequent cross-job compiled-code reuse.
Report: `out/training-throughput-audit/layout-1024.json`. Source and binary
hashes: `out/training-throughput-audit/layout-source-sha256.json`.

One-second IORegistry samples in this run recorded system GPU device utilization
averaging 82.6% during the same-input warmed update and 81.9% during the
changed-input warmed update. These system-wide driver readings are different
from Activity Monitor's per-process GPU percentage and cannot be compared
directly with the user's screenshot.

The complete Release suite for this execution candidate executed **352 tests,
four opt-in skips, zero failures**, in 359.80 s. It includes two-step comparisons
of every LoRA factor and Adam moment in both scopes with zero, partial and full
checkpoint budgets; compact strided-output bit preservation; and recovery after
forward/backward cancellation. The normal Release package and native installer
build-tool checks passed. Production builds use hardened runtime; only the
isolated XCTest host disables it to load the ad hoc test bundle.

## Larger-grid memory correction

The initial larger-checkpoint candidate was interrupted during the first 2K
backward pass after it reached **53.46 GiB physical footprint**, **51.02 GiB
sampled Metal allocation**, and **2.54 GiB system swap** (initial swap zero).
That run is a memory regression, not a passing production qualification.
Evidence is retained in
`out/training-throughput-audit/production-2048/memory-regression.log`.

The integrated policy caps larger-grid checkpoints at 6 GiB. Cold package
compilation also joins outstanding GPU submissions before allocating a separate
compiler workspace. Cached stage execution continues to overlap CPU encoding
and GPU work.

A production 2K recovery run with this policy was stopped after the user requested
that the investigation focus on 1K and 512 grids. At approximately **589.52 s of
job elapsed time**, it had completed **zero optimizer updates** and was replaying
inputs for **backward stage 119 of 249**. It is not a completed per-update timing
or a passing production qualification. Its recorded partial-run peaks were
**47.93 GiB physical footprint** and **45.34 GiB Metal allocation**. Sampled system
swap changed from 1.19 to 1.14 GiB; that observation covers only the interrupted
run and does not establish the full job's eventual memory behavior. Evidence:
`out/training-throughput-audit/production-2048/xcodebuild.log`, run
`run-1EC02E52-B394-4FDA-9531-08CF7B68BAB5`. No completed-update, final-validation or
checkpoint-export assertions were reached. The 2K policy remains unqualified.

## Cross-job compiled-code reuse

Equivalent model programs can reuse immutable compiled code from
`~/Library/Caches/Texture Studio/NativeTrainingPrograms/v1`. The shared code and
metadata quota is 1 GiB. No pixels, learned factors, activations, gradients or
optimizer values are persisted in this cache.

The identity includes the executable SHA-256, OS version/build, GPU identity,
training capacity, model base SHA-256, tensor descriptors, architecture, exact
grid/target and adapter layer shapes/rank/alpha. The adaptive execution variant
and its resolved workspace allowance also participate in identity. Code and metadata are checked
with SHA-256 before loading; incomplete or corrupt records become compilation
misses. Atomic publication and identity leases preserve code used by another
program, while nonblocking locks make contention fall back immediately to the
bounded temporary cache.

Set `TEXTURE_STUDIO_DISABLE_PROGRAM_CACHE=1` to measure compilation from scratch.
`TEXTURE_STUDIO_PROGRAM_CACHE_DIRECTORY` isolates benchmark cache storage. The
benchmark script forwards these settings to its XCTest host.

Cache recovery also takes a nonblocking exclusive identity lease before the
Program's shared lease. It reclaims only package generations no longer named
by a stage index. Active Programs preserve every generation until all their
leases end. Regression coverage includes interrupted publication, replacements,
indexed-code preservation, recovered quota, corruption, macOS path aliases and
nonblocking contention.

The integrated engine before the final orphan-recovery-only change executed
**357 tests, four opt-in skips, zero failures**, in 243.97 s. The normal package
built successfully. The final recovery change is qualified separately below.

An empty-cache engine run took **159.02 / 33.55 / 38.37 s**, with 115.21 s
compilation, 553 compiled stages and zero disk hits. Peak physical footprint
was 32.31 GiB. Observed system swap before/after these runs was 1274.25 / 1242.25 MiB;
the OS still retained swap from the interrupted earlier 2K experiment.
The synthetic benchmark does not continuously sample swap. In a fresh second process, the
first update took **42.60 s**, with **553 disk hits and zero compilations**.
These execution measurements precede only the cache orphan-recovery change;
encoder, checkpoint, compiler and optimizer code is identical to the integrated
engine. Reports: `integrated-cold-1024.json` and `integrated-reused-1024.json`
under `out/training-throughput-audit/`.

The final cache-recovery change passed **six focused cache tests, zero failures**,
in 0.047 s. The normal hardened-runtime Release package built and installed to
`/Applications/Texture Studio.app`. All five app bundles passed deep/strict
signature verification, and installed executable SHA-256 values match the built
bundles. The installed main app's smoke test exited zero. Evidence:
`final-cache-tests.log`, `recovery-package.log`, `final-install.log`,
`final-installed-binaries.json`, `final-installed-smoke.log`, and
`final-source-sha256.json` in the audit directory. The earlier production attempts
and recovery-qualified fine-stage rows below use this version. Later adaptive
measurements identify their source and test host separately. This installation
evidence is historical and does not qualify or identify the adaptive changes as
installed. The adaptive package's separate build and installation verification
are recorded below.

## Grid and scope measurements

These are individual fresh-process synthetic engine runs, not end-to-end dataset
job times. The first update includes cold compilation. The second update reuses
identical RGB; the third changes RGB and exercises frozen-feature misses. The
changed-input column is the more relevant planning value when cycling through
different source maps. Peak footprint is the app process lifetime high-water mark.

| Grid | Adapter scope | Rank / alpha | Execution policy | Cold first update | Same-input warm update | Changed-input warm update | Peak footprint |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1024 | map-decoder | 64 / 16 | Fine stages, integrated cache candidate | 159.02 s | 33.55 s | 38.37 s | 32.31 GiB |
| 512 | map-decoder | 64 / 16 | Fine stages, recovery-qualified version | 88.76 s | 7.33 s | 8.14 s | 22.59 GiB |
| 1024 | final-map | 64 / 16 | Fine stages, recovery-qualified version | 75.20 s | 11.61 s | 18.60 s | 16.45 GiB |
| 512 | final-map | 64 / 16 | Fine stages, recovery-qualified version | 41.95 s | 2.43 s | 3.71 s | 7.38 GiB |
| 1024 | map-decoder | 8 / 8 | Fine stages, matched rank-8 baseline | 156.28 s | 32.96 s | 37.49 s | 32.23 GiB |
| 1024 | map-decoder | 8 / 8 | Experimental coalesced active blocks, full Float32 | 106.31 s | 23.53 s | 28.36 s | 29.90 GiB |
| 1024 | final-map | 8 / 8 | Experimental coalesced RRDBs, active-block coalescing off | 54.00 s | 11.39 s | 18.35 s | 19.61 GiB |

Reports under `out/training-throughput-audit/`: `integrated-cold-1024.json`,
`final-512.json`, `final-output-1024.json`, `final-output-512.json`,
`fine-rank8-1024.json`, `coarse-rank8-1024.json`, and
`rrdb-output-rank8-1024.json`, respectively. The `final-*.json` filenames identify
specific benchmark artifacts; they do not mark later experiments as a shipping
configuration. The coalesced rank-8 run recorded 198 compilations, 68.12 s of cold
compilation, and no disk-code hits; fast convolution transforms were explicitly
disabled. Its checkpoint peak was 7.94 GiB.

The matched rank-8 fine-stage baseline completed all three updates at
**156.275940 / 32.962607 / 37.490782 s**. The active-coalescing run uses the same
1K grid, map-decoder scope, rank 8, alpha 8 and full Float32 arithmetic. Its
same-input and changed-input warmed updates were respectively **28.6%** and
**24.3%** shorter in these individual runs, and all three printed losses match
exactly. This comparison changes both the stage cuts and the default checkpoint
cap from 24 to 8 GiB; it does not isolate the cuts from checkpoint policy. The
fine baseline recorded 553 compilations, 112.33 s of compilation, 107 explicit
replayed stages per update and a 19.91 GiB checkpoint peak. The active candidate
recorded 198 compilations and ten explicit replayed stages per update. Both
reported zero disk-code hits. The fine baseline's historical
`fast_convolution_transforms: false` report field is preserved even though that
experiment's implementation has since been removed.

The RRDB output-branch candidate uses rank 8, alpha 8, RRDB coalescing enabled and
active-block coalescing disabled. It completed at
**53.998540 / 11.394166 / 18.345184 s**, with 100 compilations, 33.52 s of cold
compilation, zero explicit replayed stages and a 2.125 GiB checkpoint peak. Its
warm same-input update submitted 37 stages; the changed-input update submitted
100 stages. Those lower submission counts did not produce a few-second 1K
update. Changed-input time was only about 1.4% below the earlier fine
output-branch run, which uses rank 64 and alpha 16; that rank/alpha difference
prevents a matched attribution of RRDB coalescing's effect. The selected
throughput test passed, but the later broader-scope learned-update comparison
failed as described below. RRDB coalescing was rejected and has been removed.
Completion logs for the two added runs are `fine-rank8-1024.json.log` and
`rrdb-output-rank8-1024.json.log`.

Rank-8 versus rank-64 rows change adapter capacity and alpha scaling and do not
establish equal material quality. The 1K broad scope remains above the user's
target of a few seconds per update. The measured 512 final-map case meets that
timing target, while the measured 512 broader scope stays below ten seconds.
The current adaptive rank-8 profile is measured separately below. These
historical rows do not establish a final installed configuration.

## Current adaptive production-source measurements

These three runs use the production-source execution defaults: adaptive active
grouping enabled through 1K with workspace headroom and isolated active stages,
RRDB coalescing removed, rank 8, alpha 8, height target and full Float32 math.
They are synthetic engine measurements of that source, separate from the
completed dataset job and normal-package installation evidence below.

| Grid | Scope | Cold first update | Same-input warm update | Changed-input warm update | Peak footprint |
| --- | --- | ---: | ---: | ---: | ---: |
| 512 | map-decoder | 53.38 s | 4.94 s | 5.71 s | 10.32 GiB |
| 1024 | map-decoder | 97.23 s | 23.36 s | 27.92 s | 30.85 GiB |
| 1024 | final-map | 73.48 s | 11.36 s | 18.61 s | 16.03 GiB |

Exact reports are `adaptive-rank8-512.json`, `adaptive-rank8-1024.json` and
`adaptive-output-rank8-1024.json` in the audit directory. The 512 broad run took
**53.383394 / 4.937150 / 5.706484 s**, compiled 198 packages in 42.65 s, retained
at most 3.383 GiB of checkpoints and performed zero explicit stage replay. Its
warm updates submitted 173 and 212 stages. The 1K broad run took
**97.232253 / 23.359422 / 27.924949 s**, with 198 compilations, 65.63 s of
compilation, a 7.938 GiB checkpoint peak and ten explicit replayed stages per
update. Against the matched fine R8/A8 baseline, its two warmed updates were
29.1% and 25.5% shorter, with the same three printed losses; stage cuts and
checkpoint policy both differ. The current 1K output-branch run took
**73.482510 / 11.364777 / 18.608300 s** and performed zero explicit replay.
All three runs reported zero disk-code hits.

The source and Release test-host hashes are in `adaptive-source-sha256.json`;
the host SHA-256 is
`9a416a7c7d25509fe1f7ed91b02819f37f98963c41560e3ff67173f263b2b47a`.
The current source has adaptive active grouping on by default, with a
fine-stage opt-out. This execution-policy default preserves saved grid, scope
and rank choices; it does not silently change a 1K dataset to 512 or expand an
output-branch adapter to the decoder. New UI options remain 1K, output branch,
rank 8 and alpha 8. Existing LoRAs retain their recorded layer rank and alpha.

`adaptive-rank8-512.gpu.csv` contains five system-device utilization samples
with `completed_updates=1` and five with `completed_updates=2`, covering the
second and third warmed update windows. Their means are **86.6%** and **89.4%**.
These are brief, system-wide IORegistry samples, include other applications,
and are not Activity Monitor's per-process GPU percentage or a full-run average.

On this Mac, **512 map-decoder/R8/A8 is the practical performance candidate**:
its changed-input engine update is about 5.71 s while retaining the broader
trainable scope. Both current 1K scopes still miss ten seconds per changed-input
update. The real 512 dataset run below completed all four requested updates and
its adapter export. Material-quality equivalence has not been established. The
final normal package is built, installed and verified below.

## Real 512 dataset production qualification

The current source completed height training at **512 × 512, map-decoder scope,
rank 8, alpha 8** on two real 16-bit material pairs, `white_stucco_02` and
`red_brick`, with two updates per pair. The fixture uses native central crops
without resampling or padding; the original source-file hashes were unchanged.
The complete job took **29.840368 s**, including model setup, baseline and final
validation, and saving. The trainer's elapsed-time field was **28.592322 s**;
the selected XCTest passed in **30.847 s**. These are separate timing intervals.

| Update | Material | Update duration | Data preparation | CPU optimizer |
| --- | --- | ---: | ---: | ---: |
| 1 | white_stucco_02 | 8.533407 s | 0.008992 s | 0.025999 s |
| 2 | red_brick | 5.762392 s | 0.009059 s | 0.034279 s |
| 3 | white_stucco_02 | 5.006647 s | 0.009040 s | 0.033699 s |
| 4 | red_brick | 4.934533 s | 0.008889 s | 0.033395 s |

The fresh process reused **198 disk packages with zero compilations** and zero
explicit replay. Each update reports those cumulative cache totals; they are
not 198 additional hits per update. Peak process physical footprint was
**10.4167 GiB** and sampled Metal allocation peaked at **7.9983 GiB**, sampled
every 100 ms. These describe overlapping memory and must not be added. System
swap was **754,778,112 bytes** before and after, with **zero observed growth**;
this is system-wide swap, not a per-process allocation.

The test verified all requested updates, both materials' recurrence, the complete
selected factor set, changed early/output/decoder adapter factors, native
dimensions, rank/alpha/scope metadata and the exported package. The original
source hashes, cropped-input hashes, selected update telemetry and validation
summaries are preserved in the companion JSON. Baseline and final validation
MAE were **0.1005356163** and **0.1000050157**, respectively. Validation repeats
the training `white_stucco_02` pair and is a numeric execution diagnostic; it is
not held-out evidence of material quality or generalization.

Raw report:
`out/training-throughput-audit/production-512-map-decoder-r8/run-647ADCEB-0A4A-4A97-9A25-6E9A95AB8E97/training-512.json`.
Completion and assertions:
`out/training-throughput-audit/production-512-map-decoder-r8/xcodebuild.log`.

## Adaptive active-stage memory and scheduling policy

The production source resolves one shared local workspace allowance by
subtracting retained checkpoint storage (capped at half the machine training
capacity), the frozen-feature budget (the smaller of 2 GiB and one-sixteenth of
capacity), and the estimated model working set from training capacity. Each
subtraction saturates at zero. Active coalescing requires its conservative
72-times-input-byte workspace estimate to fit this allowance. Existing frozen
forward-block admission is unchanged. The rejected full-RRDB experiment also
required its workspace estimate to fit both 24 GiB and half the machine capacity;
that admission policy is retained here as historical evidence.

The same resolved allowance reaches initial, rebuilt and disposable compiler
graph owners, and participates in compiled-code identity whenever a coarse
policy is enabled. This keeps stage cuts deterministic when explicit checkpoint
budgets differ. Active grouping is enabled by default through 1K, subject to
adaptive memory admission; larger grids retain fine cuts. An explicit false
override or `TEXTURE_STUDIO_COALESCE_ACTIVE_BLOCKS=0` retains the fine-stage
control. The rejected RRDB implementation and flag have been removed.

Coalesced active blocks are marked during graph construction; the historical
RRDB experiment used the same isolation. Forward and reverse executions drain
the ordered stream before and after the stage, including replay, errors and
cancellation. This prevents their hidden
backend workspace from overlapping another stage in the program stream.
Compact result buffers and submission ownership retain storage through GPU
completion. The current adaptive measurements and Release suite below include
these safeguards; earlier experimental rows precede them. The completed real
512 job and normal package are recorded separately. The workspace figures
are conservative admission estimates, not guarantees of actual backend peak
allocation.

## Scope, native crops and quality

`final-map` trains the selected `ups.<branch>.` output branch: the RRDB refinement
network and its output convolutions. It does **not** mean only the literal last
convolution. `map-decoder` includes those same adapters plus the selected generator
decoder `gen.m_dec_<branch>.` and generator tail `gen.m_tail_<branch>.`. The broader
scope can change earlier features for that target but requires more forward and
reverse computation. Fewer trained layers are a real capacity tradeoff, not an
implementation-only acceleration.

The native dataset service takes aligned crops from original maps without
resampling. Moving from a 1K crop to a 512 crop preserves the sampled pixel detail
and reduces spatial context per example. It can also change crop count and the
number of updates in a dataset loop, so per-update timing alone cannot predict
an entire run. Evaluation should keep held-out source families separate, use the
same native source pixels and target encoding, and compare checkpoints trained
for equal wall-clock budgets. A separate crop from a known training source is a
useful diagnostic but does not establish generalization to unseen materials.

The working decision criterion is to prefer 1K with the broader adapter scope if
verified changed-input updates can be brought below ten seconds. If that cannot
be reached, evaluate 512 with the broader scope as the fallback, then compare it
with the faster final-map scope on held-out sources at equal training time. These
are proposed performance/quality comparisons; this investigation has not measured
one choice to produce better material quality.

## Experimental numerical qualification

Coalesced active blocks with ordinary full Float32 math passed the existing
two-step fixture comparison of every factor and Adam moment in both scopes,
including forced replay and a generous checkpoint budget. The selected test took
68.30 s. This establishes the covered fixture's equivalence and does not qualify
all later coalescing policies or production performance.

Permitting FP16 Winograd convolution intermediates passed the varied 64-pixel
fine-stage numerical test but **failed** the varied 128-pixel coalesced test. On
the second map-decoder Adam step, the cumulative learned change for
`gen.m_dec_3.m_up2.0.up.1.lora_A` differed by **7.03e-5**, above its explicit
**8.99e-6** regression bound. This is about 35% of that tensor's largest expected
learned change. Output/loss, raw-gradient and moment comparisons alone did not
catch this optimizer sensitivity. The tolerance was not relaxed, and this
fast-transform candidate has not been adopted and its implementation was removed.
This failure alone does not attribute the sensitivity to reduced precision.

The matching full-Float32 active-coalescing control differed by **7.0217e-5**;
the existing fine-stage engine, with both coalescing policies disabled, produced
the **same difference at coordinate 679** against the monolithic reference.
Diagnostics show second-step gradients of **-8.36e-9** for the monolithic reference
and **1.82e-8** for fine stages, near the optimizer epsilon of **1e-8**, despite
only **2.70e-8** difference between the pre-update factors. The opposite small
gradients yielded materially different normalized Adam directions. This is
existing staged-versus-monolithic numerical sensitivity and does not establish
an active-coalescing regression or isolate layout/kernel arithmetic as its cause.

The varied-weight qualification subsequently compared each coalescing candidate
with the existing fine-stage engine, using identical checkpoint budgets and
explicit policy flags; the bounds remained unchanged. Independent monolithic
comparisons remain in the standard all-factor fixture tests. The active-block
128-pixel, both-scope comparison of gradients, every factor, all Adam moments and
learned changes over two steps **passed in 45.446 s**. Both pure workspace-policy
tests also passed, including low-memory rejection, the 64 GiB reference budget,
saturated subtraction and checkpoint reservation clamping.

The full-RRDB candidate passed its output-scope cases but **failed four learned
change bounds on the second map-decoder step** against the existing fine-stage
baseline. The largest error was **6.9190748e-5** versus **8.9884434e-6** allowed,
at coordinate 679 of `gen.m_dec_3.m_up2.0.up.1.lora_A`. That candidate's gradient
was **-7.7453e-9** versus the fine control's **1.8192e-8**, producing opposing
normalized Adam directions near epsilon **1e-8**. Unlike the earlier
staged-versus-monolithic discrepancy, this failure demonstrates an additional
candidate-versus-existing-engine optimizer regression under the declared bounds.
The tolerance was not relaxed. Combined with its small observed timing change,
the RRDB experiment was rejected and its implementation has been removed. This does not prove a
particular backend arithmetic cause or retroactively attribute the separate
fast-transform failure to FP16 intermediates.

Evidence: `final-coarse-controls.log` executed four selected tests with four
assertion failures, all from the RRDB comparison; the active comparison and two
policy tests passed. Historical logs `kernel-candidate-tests.log`,
`rrdb-candidate-tests.log`, and `fine-varied-control-tests.log` remain in
`out/training-throughput-audit/`.

The current adaptive source subsequently passed the complete Release suite:
**363 tests executed, four opt-in skips, zero failures**, in **454.133 s**.
Within that suite, varied 128-pixel height training in both scopes passed in
**45.653 s**, and the normal/roughness test passed both scopes in **92.265 s**.
Each comparison uses two independently evolving production CPU Adam updates and
checks outputs/loss, all gradients, moments and learned changes against the
existing fine-stage baseline with unchanged bounds. Independent two-step
monolithic all-factor fixture tests also passed for fine and adaptive stages,
including forced replay. Both pure headroom-policy tests passed. Evidence:
`adaptive-final-suite.log`. This establishes the covered regression tests;
the completed 512 dataset job is a separate production execution check.
Material-quality comparisons remain unmeasured.

## Final normal package and installation

The normal arm64 Release package built successfully from the same source
snapshot used for the adaptive measurements. All five app bundles passed
deep/strict signature verification and have hardened runtime enabled. The
package contains no XCTest bundles, and the built main app's smoke test exited
zero. Evidence: `adaptive-package.log`, `adaptive-built-binaries.json` and
`adaptive-package-validation.json` in the audit directory.

The ready package is `dist/Texture-Studio-macos-arm64.zip`, SHA-256
`248b6157b1ea36cc09c4ebe7906089dc94ddfae3b86e18fe0452906d95fbec58`.
The main executable SHA-256 is
`1a0255667d6d93eb2a13ad91e62d6ec730b7d8d48c0d85af4018d2712432d249`.
It differs from the isolated Release test host because normal packaging retains
hardened runtime and does not enable XCTest testability.

Installation initially waited for the running user app to close. After the user
closed it, the installer succeeded and installed the adaptive package to
`/Applications/Texture Studio.app` (`adaptive-install.log`). All **five installed
bundles** passed deep/strict signature verification with hardened runtime enabled,
and every installed executable hash matches `adaptive-built-binaries.json`.
The installed main app is version **0.9.13, build 107**, with SHA-256
`1a0255667d6d93eb2a13ad91e62d6ec730b7d8d48c0d85af4018d2712432d249`.
Its smoke test exited zero. Proof: `adaptive-installed-binaries.json`,
`adaptive-installed-smoke.log` and the updated `adaptive-package-validation.json`.
This separately verifies the installed adaptive version; the earlier fine-stage
installation remains historical evidence.
