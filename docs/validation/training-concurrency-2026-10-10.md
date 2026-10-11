# Concurrent material training at 512 × 512

Assessment dated October 10, 2026. **Two independent 512 jobs are plausible in memory on this 64 GiB M2 Max, but a useful throughput improvement has not been demonstrated. Keep one GPU trainer as the default. For several materials contributing to one adapter, investigate a physical batch of two before adding simultaneous trainers.** Neither option is enabled by this assessment.

## Measured starting point

The host reports an Apple M2 Max with 12 CPU cores, 30 GPU cores and 64 GiB unified memory. The retained [October 10 evidence JSON](native-training-throughput-2026-10-10.json) and [measurement notes](native-training-throughput-2026-10-10.md) provide these single-job observations:

| 512 configuration | Changed-input step | Peak process footprint | Evidence scope |
| --- | ---: | ---: | --- |
| Adaptive map-decoder, rank 8 / alpha 8 | 5.706484 s | 10.32 GiB | Three synthetic engine steps; cold first step 53.383394 s |
| Adaptive map-decoder, rank 8 / alpha 8 | 5.762392 s on first red-brick input; later repeated inputs 5.006647 / 4.934533 s | 10.4167 GiB | Two real materials, four total steps, validation and export; 29.840368 s complete job |
| Earlier fine-stage map-decoder, rank 64 / alpha 16 | 8.138213 s | 22.59 GiB | Historical execution policy; three synthetic steps |

The measured adaptive model source SHA-256 is `6341ea5e771e58605ca6b039f9146684040bf56a91a7a196661040249c40de82`, which matches `NativeMaterialModel.swift` at assessment time. The trainer and UI are being revised separately; these timings qualify the recorded source, not a fresh run of those revisions. The permanent JSON retains selected raw values and provenance; the original `out/training-throughput-audit/` reports referenced in it are absent from this checkout.

The real job reused 198 compiled packages, made no compilations, peaked at 7.9983 GiB of sampled Metal allocations, and observed no growth in system swap. Footprint and Metal allocations overlap and must not be added. Two times the observed process footprint is approximately **20.83 GiB**; this is a sizing illustration, not a measured concurrent peak. Compiler workspaces, other applications, different ranks/scopes and large-dataset cache behavior can raise use.

Ten brief system-wide GPU samples during warmed synthetic steps averaged 86.6% and 89.4% in their two windows. This suggests limited unused GPU capacity and makes a twofold speedup from a second trainer unlikely; it does not prove a utilization ceiling or predict a speedup. Real-job input preparation took about 9 ms per step, at most 0.181% of the measured step time, so prefetching alone offers little benefit on that fixture.

As a planning illustration, 5.706484 s × 1,000 steps is 1.59 hours, and × 10,000 is 15.85 hours, before validation, saving and setup. These are linear projections from one changed-input measurement, not sustained full-dataset timings. More 512 crops can also increase the number of steps per epoch.

## What the current implementation supports

| Approach | Present behavior and required change |
| --- | --- |
| Several materials train one adapter | Already supported: the trainer shuffles materials each epoch and evaluates them serially. All update the same adapter and Adam state. |
| Gradient accumulation | Already supported, with physical batch size one. Maps run serially at the same weight snapshot; gradients are averaged before one optimizer step. Increasing this setting evaluates more maps and does not parallelize them. |
| Physical batch size two | Requires graph changes. Inputs and outputs explicitly use batch dimension one, including attention-window reshapes and feed construction. Loss normalization, stage storage estimates, cache identity and sample failure handling must support a batch. A batch of two should match accumulation of two at the same weight snapshot within existing numerical tolerances; it does not reproduce two sequential Adam steps. |
| Two independent trainers | Requires separate model/program instances, optimizer state, controls, progress/logs and output directories. They produce separate adapters; independently trained adapters cannot simply be combined into the shared adapter that a serial multi-material run produces. |

Relevant source: [trainer](../../src/TextureStudio/Services/NativeMaterialTrainer.swift), [model and GPU execution](../../src/TextureStudio/Services/NativeMaterialModel.swift), [machine policy](../../src/TextureStudio/Models/MachineResources.swift), [disk code cache](../../src/TextureStudio/Services/NativeGraphCodeCache.swift), and [workbench operation state](../../src/TextureStudio/Stores/WorkbenchStore.swift).

The model/program holds mutable graph plans, package statistics, feature-cache order and an ordered GPU submission stream. Sharing it across worker tasks would race. Parallel gradient workers for one adapter need independent execution contexts, an immutable weight snapshot and one coordinator that applies the averaged gradient once. Independent trainers avoid that algorithmic coupling but duplicate retained GPU state.

Current memory admission gives each program a budget calculated from the whole machine. It has **no shared reservation for multiple jobs**. Active stages drain their own stream around expensive workspaces; they do not exclude expensive stages or compilation in another program. The per-stage Metal allocation check does not reserve future allocations atomically across trainers. A concurrency feature therefore needs admission for the combined working sets and a shared gate for cold compilation and workspace-heavy stages, including validation and cancellation cleanup. Apple describes the recommended Metal working set as a performance threshold for the total resources and heaps, not an additional allowance for each trainer. [Apple memory guidance](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize).

Dataset shared-read locks allow concurrent readers and protect against edits. Disk packages use identity leases, immutable generations and cross-process publication/eviction locks; contention falls back to compilation. That protects stored code, but is not a mechanism for sharing optimizer state or coordinating GPU memory. The workbench allows one operation per store, without a machine-wide training scheduler. Opening separate windows is not a qualified throughput mode.

Metal permits concurrent commands, but shared read/write resources require synchronization. A queue per trainer provides submission opportunities; it does not promise doubled compute capacity. Immutable base buffers could be shared only with verified lifetime and read-only use; mutable adapter, activation and result buffers require isolated ownership or explicit synchronization. [Apple synchronization guidance](https://developer.apple.com/documentation/metal/resource-synchronization).

## Recommended experiment and acceptance criteria

1. Compare serial batch-one accumulation-two with a physical batch-two candidate on the same two distinct 512 inputs and weight snapshot. Verify gradients, every adapter factor and Adam moments, finite losses, mean-loss normalization, and one optimizer step. Cover Stop & Save, Abort, a failed member of a batch, validation and export. Share immutable base storage and retain isolated writable results.
2. If independent adapters are the desired output, benchmark two isolated trainers against the **same total work** completed serially. Start with rank 8 / alpha 8 and cap admission at two jobs. Warm compiled code serially; measure cold starts separately. Keep a shared compiler/workspace gate and reserve the aggregate memory before admitting a second job.
3. Use at least 20 changed-input steps per job and representative distinct materials, repeated trials, no overlapping application build, and matched validation/checkpoint settings. Report aggregate successful sample evaluations per second, total completion time, each job's latency, combined footprint, sampled Metal allocation and system swap delta. A two-job test has no speed benefit if each job slows enough that total completion time equals the serial baseline.
4. Require both jobs to finish validation and verified exports, with no training failures, no swap growth attributable to the experiment, and useful repeatable aggregate throughput improvement before offering concurrency by default. Do not infer reliability for an hours-long run from a short benchmark; run a longer representative soak after the short comparison passes.

No GPU benchmark was run for this assessment: the existing benchmark script explicitly exercises one isolated workload and has no qualified concurrency mode. The conclusions are based on retained single-job measurements, a current hardware inventory, source review and the linked Apple documentation. Physical batching is the first engineering candidate for one shared adapter; simultaneous independent training remains an opt-in experimental candidate whose benefit must be measured.
