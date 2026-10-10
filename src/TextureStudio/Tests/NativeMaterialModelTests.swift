import Foundation
import Metal
import MetalPerformanceShadersGraph
import XCTest
@testable import TextureStudio

final class NativeMaterialModelTests: XCTestCase {
    /// Opt-in throughput evidence on the installed, checksum-pinned learned
    /// model. Synthetic pixels isolate the engine from file cache effects.
    func testInstalledModelTrainingThroughput() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["TEXTURE_STUDIO_THROUGHPUT_BENCHMARK"] == "1" else {
            throw XCTSkip("Set TEXTURE_STUDIO_THROUGHPUT_BENCHMARK=1 for the installed-model engine benchmark.")
        }
        let size = Int(environment["TEXTURE_STUDIO_THROUGHPUT_SIZE"] ?? "1024") ?? 1024
        let steps = Int(environment["TEXTURE_STUDIO_THROUGHPUT_STEPS"] ?? "3") ?? 3
        let scope = environment["TEXTURE_STUDIO_THROUGHPUT_SCOPE"] ?? "map-decoder"
        let report = try XCTUnwrap(environment["TEXTURE_STUDIO_THROUGHPUT_REPORT"])
        guard let rank = Int(environment["TEXTURE_STUDIO_THROUGHPUT_RANK"] ?? "64"), rank > 0,
              let alpha = Float(environment["TEXTURE_STUDIO_THROUGHPUT_ALPHA"] ?? "16"), alpha.isFinite, alpha > 0,
              [256, 512, 1024, 2048].contains(size), (1...10).contains(steps), ["final-map", "map-decoder"].contains(scope) else {
            throw StudioError("Invalid installed-model benchmark grid, step count, layer scope, rank or alpha.")
        }
        try await Task.detached {
            let activity = ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: "Benchmarking material training")
            defer { ProcessInfo.processInfo.endActivity(activity) }
            let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
                .appendingPathComponent("Texture Studio/Material Models/pbrnxt-base/" + NativeMaterialModel.pinnedFilename)
            let model = try NativeMaterialModel.load(checkpointURL: nil, baseURL: base, target: "height", scope: scope,
                rank: rank, alpha: alpha, training: true)
            let program = try model.program(width: size, height: size, target: "height")
            let reference = (0..<size * size).map { Float(($0 * 17) % 251) / 251 }
            var state: [String: NativeTensor] = [:]
            var observations: [[String: Any]] = []
            for step in 1...steps {
                // Repeat once, then change pixels to exercise feature misses.
                let variation = max(0, step - 2)
                let rgb = (0..<3 * size * size).map { Float(($0 * 37 + variation * 19) % 251) / 251 }
                let started = ProcessInfo.processInfo.systemUptime
                print("THROUGHPUT_BEGIN step=\(step) size=\(size) scope=\(scope) rank=\(rank) alpha=\(alpha)")
                let result = try program.execute(rgb: rgb, adapters: model.adapterWeights, reference: reference,
                    gradientsOnly: true, featureKey: "fixture-\(variation)")
                let update = try NativeMaterialOptimizer.apply(gradients: result.gradients, weights: model.adapterWeights,
                    state: state, learningRate: 1e-4, step: step, configuration: .init())
                state = update.state; model.updateAdapters(update.weights)
                let elapsed = ProcessInfo.processInfo.systemUptime - started
                XCTAssertTrue(result.loss!.isFinite)
                XCTAssertEqual(Set(result.gradients.keys), Set(model.adapterWeights.keys))
                let stats = try JSONSerialization.jsonObject(with: JSONEncoder().encode(program.executionStatistics))
                var memory = task_vm_info_data_t()
                var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
                let status = withUnsafeMutablePointer(to: &memory) { pointer in
                    pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
                    }
                }
                let observation: [String: Any] = ["step": step, "seconds": elapsed, "loss": result.loss!,
                    "cumulative_execution": stats, "metal_allocated_bytes": MTLCreateSystemDefaultDevice()!.currentAllocatedSize,
                    "peak_physical_footprint_bytes": status == KERN_SUCCESS ? UInt64(max(0, memory.ledger_phys_footprint_peak)) : 0]
                observations.append(observation)
                let data = try JSONSerialization.data(withJSONObject: ["size": size, "scope": scope,
                    "rank": rank, "alpha": alpha,
                    "coalesce_active_blocks": environment["TEXTURE_STUDIO_COALESCE_ACTIVE_BLOCKS"] != "0" && size <= 1024,
                    "base_sha256": model.baseSHA256, "observations": observations], options: [.prettyPrinted, .sortedKeys])
                try data.write(to: URL(fileURLWithPath: report), options: .atomic)
                print("THROUGHPUT_UPDATE " + String(decoding: try JSONSerialization.data(withJSONObject: observation, options: [.sortedKeys]), as: UTF8.self))
            }
        }.value
    }

    func testRawAdapterGradientsMatchLegacyAdamUpdateWithoutMutatingWeights() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let model = try NativeMaterialModelFixture().model(adapter: true)
            let rgb = (0..<3 * 64 * 64).map { Float($0 % 97) / 97 }
            let reference = [Float](repeating: 0.3, count: 64 * 64)
            let program = try model.program(width: 64, height: 64, target: "height")
            let before = model.adapterWeights
            let raw = try program.execute(rgb: rgb, adapters: before, reference: reference, gradientsOnly: true)
            XCTAssertEqual(Set(raw.gradients.keys), Set(before.keys))
            XCTAssertTrue(raw.updated.isEmpty); XCTAssertTrue(raw.optimizerState.isEmpty)
            for name in before.keys { XCTAssertEqual(before[name]!.bytes, model.adapterWeights[name]!.bytes) }
            let native = try NativeMaterialOptimizer.apply(gradients: raw.gradients, weights: before, state: [:],
                learningRate: 1e-3, step: 1, configuration: .init())
            let legacy = try program.execute(rgb: rgb, adapters: before, reference: reference, learningRate: 1e-3)
            XCTAssertEqual(raw.loss!, legacy.loss!, accuracy: 1e-6)
            for name in native.weights.keys {
                for (current, expected) in zip(try native.weights[name]!.floatValues(), try legacy.updated[name]!.floatValues()) {
                    XCTAssertEqual(current, expected, accuracy: 2e-6)
                }
            }
        }.value
    }

    func testGraphResultCompactionPreservesStridedLogicalOrderAndFloat32Bits() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            for stream in [nil, NativeGraphExecution.Stream()] {
                let words: [UInt32] = [
                    0x3f800001, 0x80000000, 1, 0x40000001,
                    0x3e000003, 0x007fffff, 0x00800000, 0xbf800008,
                    0x3f000009, 0xbe80000a, 0x3fc0000b, 0x3d00000c,
                    0x4020000d, 0xc010000e, 0x3f400005, 0x3e800006,
                    0x3fa00007, 0x3e800011, 0xbe000004, 0x3e000013,
                    0x3f100015, 0x3f200017, 0xbf300019, 0x3f400021
                ]
                var transposed: [UInt32] = [], sliced: [UInt32] = []
                for channel in 0..<2 {
                    for x in 0..<4 {
                        for y in 0..<3 {
                            let word = words[(channel * 3 + y) * 4 + x]
                            transposed.append(word)
                            if x == 1 || x == 2 { sliced.append(word) }
                        }
                    }
                }
                let source = NativeTensor(dtype: "F32", shape: [1, 2, 3, 4], bytes: words.withUnsafeBytes { Data($0) })
                // Keep only returned data after graph/executable destruction. A
                // strided slice must own precisely its logical, contiguous pixels.
                let retained: [MPSGraphTensorData] = try autoreleasepool {
                    let graph = MPSGraph()
                    let input = graph.placeholder(shape: [1, 2, 3, 4], dataType: .float32, name: "integer_float_words")
                    let transpose = graph.transpose(input, permutation: [0, 1, 3, 2], name: "strided_transpose")
                    let slice = graph.sliceTensor(transpose, dimension: 2, start: 1, length: 2, name: "noncontiguous_slice")
                    var cache: [String: MPSGraphExecutable] = [:]
                    let result = try NativeGraphExecution.runData(graph,
                        feeds: [input: NativeGraphExecution.tensorData(source)], targets: [slice, transpose, input, input, slice], cache: &cache, stream: stream)
                    cache.removeAll()
                    try stream?.finish()
                    return result
                }
                XCTAssertEqual(retained[0].shape, [1, 2, 2, 3])
                XCTAssertEqual(retained[1].shape, [1, 2, 4, 3])
                for (index, expected) in [sliced, transposed, words, words, sliced].enumerated() {
                    let array = retained[index].mpsndarray()
                    XCTAssertNil(array.parent, "Compacted output must not retain a parent scratch arena.")
                    XCTAssertEqual(array.resourceSize(), expected.count * MemoryLayout<UInt32>.size,
                                   "Each result must own only its contiguous logical Float32 bytes.")
                    var actual = [UInt32](repeating: 0, count: expected.count)
                    actual.withUnsafeMutableBytes { array.readBytes($0.baseAddress!, strideBytes: nil) }
                    XCTAssertEqual(actual, expected, "Logical order, signed zero, subnormal and mantissa bits must survive compaction.")
                }
                // Activation replay feeds the retained compact GPU buffer directly
                // into a fresh graph after the producer and executable are gone.
                // Reusing it repeatedly must preserve both samples and storage.
                let checkpoint = retained[0]
                var expectedReplay: [UInt32] = []
                for channel in 0..<2 {
                    for y in 0..<3 {
                        for x in 1...2 { expectedReplay.append(words[(channel * 3 + y) * 4 + x]) }
                    }
                }
                for _ in 0..<2 {
                    let replay: [UInt32] = try autoreleasepool {
                        let graph = MPSGraph()
                        let input = graph.placeholder(shape: checkpoint.shape, dataType: .float32, name: "gpu_checkpoint")
                        let output = graph.transpose(input, permutation: [0, 1, 3, 2], name: "replayed_channels")
                        var cache: [String: MPSGraphExecutable] = [:]
                        let result = try NativeGraphExecution.runData(graph, feeds: [input: checkpoint], targets: [output], cache: &cache, stream: stream)[0]
                        try stream?.finish()
                        XCTAssertNil(result.mpsndarray().parent)
                        XCTAssertEqual(result.mpsndarray().resourceSize(), expectedReplay.count * MemoryLayout<UInt32>.size)
                        var values = [UInt32](repeating: 0, count: expectedReplay.count)
                        values.withUnsafeMutableBytes { result.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
                        return values
                    }
                    XCTAssertEqual(replay, expectedReplay, "GPU checkpoint replay must preserve every Float32 bit.")
                    XCTAssertNil(checkpoint.mpsndarray().parent)
                    XCTAssertEqual(checkpoint.mpsndarray().resourceSize(), sliced.count * MemoryLayout<UInt32>.size)
                    var unchanged = [UInt32](repeating: 0, count: sliced.count)
                    unchanged.withUnsafeMutableBytes { checkpoint.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
                    XCTAssertEqual(unchanged, sliced, "Replay must not overwrite a retained checkpoint.")
                }
            }
        }.value
    }
    func testCheckpointedReverseMatchesMonolithicAdamForEveryFactorOverTwoSteps() async throws {
        try await verifyCheckpointedReverseAgainstMonolithic(coalesceActiveBlocks: false)
    }
    func testCoalescedActiveBlocksMatchMonolithicAdamForEveryFactorOverTwoSteps() async throws {
        try await verifyCheckpointedReverseAgainstMonolithic(coalesceActiveBlocks: true)
    }
    private func verifyCheckpointedReverseAgainstMonolithic(coalesceActiveBlocks: Bool) async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let verification: Task<Void, Error> = Task.detached {
            let fixture = NativeMaterialModelFixture()
            var rgb = [Float](repeating: 0, count: 12_288)
            for index in rgb.indices {
                let code = (index * 37 + index / 64) % 251
                rgb[index] = Float(code) / Float(251)
            }
            var reference = [Float](repeating: 0, count: 4_096)
            for index in reference.indices {
                let horizontal = Float(index % 64) * Float(0.001)
                let vertical = Float(index / 64) * Float(0.00037)
                reference[index] = Float(0.2) + horizontal + vertical
            }
            var coordinates = [Float](repeating: 0, count: 15 * 15 * 2)
            for y in 0..<15 {
                for x in 0..<15 {
                    for (axis, coordinate) in [y - 7, x - 7].enumerated() {
                        let normalized = Float(coordinate) * Float(8) / Float(7)
                        let logarithm = Float(log2(Double(abs(normalized)) + 1) / 3)
                        coordinates[(y * 15 + x) * 2 + axis] = normalized < 0 ? -logarithm : logarithm
                    }
                }
            }
            var positionIndices = [Int64](repeating: 0, count: 64 * 64)
            for a in 0..<64 {
                for b in 0..<64 {
                    let dy = a / 8 - b / 8 + 7
                    let dx = a % 8 - b % 8 + 7
                    positionIndices[a * 64 + b] = Int64(dy * 15 + dx)
                }
            }
            func compare(_ actual: NativeTensor, _ expected: NativeTensor, name: String) throws {
                XCTAssertEqual(actual.shape, expected.shape, name)
                let a = try actual.floatValues()
                let e = try expected.floatValues()
                XCTAssertEqual(a.count, e.count, name)
                var scale = Float.leastNormalMagnitude
                var error = Float(0)
                for index in e.indices {
                    scale = max(scale, abs(e[index]))
                    error = max(error, abs(a[index] - e[index]))
                }
                let tolerance = scale * Float(0.003) + Float.leastNonzeroMagnitude * Float(16)
                XCTAssertLessThanOrEqual(error, tolerance,
                    "All factor gradients and Adam state must agree, including tiny upstream gradients: \(name)")
            }
            let cases = ["final-map", "map-decoder"].flatMap { scope in
                // Cover no checkpoints, partial replay and a generous compact
                // GPU checkpoint pool, including plan reuse on the next step.
                let budgets: [UInt64] = coalesceActiveBlocks ? [0, 16 * 1024 * 1024] : [0, 512 * 1024, 16 * 1024 * 1024]
                return budgets.map { (scope, $0) }
            }
            for (scope, checkpointByteLimit) in cases {
                let original = try fixture.model(scope: scope, rank: 4, alpha: 8)
                var starting = original.adapterWeights
                // Nonzero B makes both A and B derivatives observable on the
                // first step. Independent columns avoid artificial rank-one
                // cancellation in attention/normalization derivatives.
                var random = NativeMaterialRandom(seed: 9007)
                for name in starting.keys.sorted() where name.hasSuffix(".lora_B") {
                    let tensor = starting[name]!
                    var values = [Float](repeating: 0, count: tensor.shape.reduce(1, *))
                    for index in values.indices { values[index] = (random.unit() * 2 - 1) * Float(0.005) }
                    starting[name] = .floats(values, shape: tensor.shape)
                }
                var base = original.baseWeights
                // The shape fixture has zero coordinates and all-zero gather
                // indices, making every relative bias uniform. Softmax removes
                // a uniform bias, so its true gradient is zero; relative error
                // on its ~1e-29 rounding residue is not a useful comparison.
                // Restore canonical Swin coordinates/indices to test a real
                // position-dependent bias and its complete gradient path.
                for name in base.keys.sorted() {
                    if name.hasSuffix(".relative_coords_table") {
                        base[name] = .floats(coordinates, shape: [1, 15, 15, 2])
                    } else if name.hasSuffix(".relative_position_index") {
                        base[name] = NativeTensor(dtype: "I64", shape: [64, 64], bytes: positionIndices.withUnsafeBytes { Data($0) })
                    }
                }
                let model = try NativeMaterialModel(baseWeights: base, adapterWeights: starting,
                    layers: original.layers, configuration: original.configuration, baseSHA256: original.baseSHA256, architecture: .test)
                let monolithic = try NativeMaterialModel.Program(model: model, width: 64, height: 64, target: "height", staged: false,
                    coalesceActiveBlocks: false)
                let staged = try NativeMaterialModel.Program(model: model, width: 64, height: 64, target: "height", staged: true,
                    checkpointByteLimit: checkpointByteLimit, coalesceActiveBlocks: coalesceActiveBlocks)
                if coalesceActiveBlocks && scope == "map-decoder" {
                    let fine = try NativeMaterialModel.Program(model: model, width: 64, height: 64, target: "height",
                        checkpointByteLimit: checkpointByteLimit, coalesceActiveBlocks: false)
                    XCTAssertLessThan(staged.frozenStageCount, fine.frozenStageCount,
                        "The coarse numerical cases must exercise fewer complete block VJPs.")
                }
                var expectedFactors = starting, actualFactors = starting
                var expectedState: [String: NativeTensor] = [:], actualState: [String: NativeTensor] = [:]
                for step in 1...2 {
                    let expected = try monolithic.execute(rgb: rgb, adapters: expectedFactors, reference: reference,
                        learningRate: 1e-4, step: step, optimizerState: expectedState)
                    let actual = try staged.execute(rgb: rgb, adapters: actualFactors, reference: reference,
                        learningRate: 1e-4, step: step, optimizerState: actualState, featureKey: "same-native-rgb")
                    XCTAssertEqual(actual.loss!, expected.loss!, accuracy: 2e-6, "\(scope), checkpoint budget \(checkpointByteLimit), step \(step)")
                    XCTAssertEqual(actual.valueLoss!, expected.valueLoss!, accuracy: 2e-6)
                    XCTAssertEqual(actual.gradientLoss!, expected.gradientLoss!, accuracy: 2e-6)
                    XCTAssertEqual(Set(actual.updated.keys), Set(starting.keys))
                    XCTAssertEqual(Set(actual.optimizerState.keys), Set(expected.optimizerState.keys))
                    for name in expected.updated.keys {
                        try compare(actual.updated[name]!, expected.updated[name]!, name: "\(scope), checkpoint budget \(checkpointByteLimit), step \(step), \(name)")
                    }
                    // m/v reveal gradient disagreement that a tiny learning
                    // rate could hide when only updated factors are compared.
                    for name in expected.optimizerState.keys {
                        try compare(actual.optimizerState[name]!, expected.optimizerState[name]!, name: "\(scope), checkpoint budget \(checkpointByteLimit), step \(step), \(name)")
                    }
                    for (a, e) in zip(actual.output, expected.output) { XCTAssertEqual(a, e, accuracy: 2e-6) }
                    expectedFactors = expected.updated; actualFactors = actual.updated
                    expectedState = expected.optimizerState; actualState = actual.optimizerState
                }
                let statistics = staged.executionStatistics
                XCTAssertGreaterThan(statistics.commandBuffers, 0)
                XCTAssertLessThanOrEqual(statistics.maximumInFlightStages, 2)
                XCTAssertLessThanOrEqual(statistics.peakCheckpointBytes, checkpointByteLimit,
                    "Replayed features must reuse the checkpoint budget, never grow it.")
                if coalesceActiveBlocks, checkpointByteLimit > 0 {
                    XCTAssertEqual(statistics.recomputedStages, 0, "The full-budget coarse case must cover backward execution without activation replay.")
                }
            }
        }
        try await verification.value
    }
    func testBoundedStagesMatchWholeGraphAndReuseOnlyUnchangedFeatures() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let model = try NativeMaterialModelFixture().model(adapter: true)
            let rgb = (0..<3 * 64 * 64).map { Float($0 % 97) / 97 }
            let reference = [Float](repeating: 0.1, count: 64 * 64)
            let whole = try NativeMaterialModel.Program(model: model, width: 64, height: 64, target: "height", staged: false)
            let staged = try model.program(width: 64, height: 64, target: "height")
            let expected = try whole.execute(rgb: rgb, adapters: model.adapterWeights, reference: reference)
            var stageCalls = 0
            let actual = try staged.execute(rgb: rgb, adapters: model.adapterWeights, reference: reference,
                featureKey: "same-input", onStage: { _, _ in stageCalls += 1 })
            XCTAssertGreaterThan(stageCalls, 10)
            for (a, b) in zip(expected.output, actual.output) { XCTAssertEqual(a, b, accuracy: 2e-6) }
            XCTAssertEqual(expected.loss!, actual.loss!, accuracy: 2e-6)
            stageCalls = 0
            var graphBuilds = 0, graphReuses = 0
            let changedReference = reference.map { $0 + 0.1 }
            let update = try staged.execute(rgb: rgb, adapters: model.adapterWeights, reference: changedReference,
                learningRate: 1e-3, featureKey: "same-input", onStage: { _, _ in stageCalls += 1 },
                onOperation: { operation, done, _ in
                    if operation == "Building model execution graph", done == 0 { graphBuilds += 1 }
                    if operation == "Reusing model execution graph", done == 1 { graphReuses += 1 }
                })
            XCTAssertEqual(stageCalls, 0, "Frozen features are reused, while changed target and adapter feeds stay live")
            XCTAssertEqual(graphBuilds, 0)
            XCTAssertEqual(graphReuses, 1, "Successive staged executions reuse their uncompiled symbolic graph.")
            XCTAssertNotEqual(actual.loss, update.loss)
            XCTAssertNotEqual(update.updated["ups.3.model.10.lora_B"]!.bytes, model.adapterWeights["ups.3.model.10.lora_B"]!.bytes)
            let control = NativeMaterialTrainingControl()
            XCTAssertThrowsError(try staged.execute(rgb: rgb, adapters: model.adapterWeights, featureKey: "different-input",
                checkCancellation: { try control.check() }, onStage: { _, _ in control.stop() })) {
                XCTAssertTrue($0 is CancellationError)
            }
        }.value
    }
    func testCapacityAdmissionRejectsBroadLegacyAdaptersBeforeGraphAllocation() throws {
        let fixture = NativeMaterialModelFixture()
        let focused = try fixture.model(adapter: true)
        let broad = try fixture.model(scope: "map-decoder", rank: 8, alpha: 8)
        let focusedBytes = NativeMaterialTrainer.estimatedWorkingBytes(model: focused, size: 1024)
        let broadBytes = NativeMaterialTrainer.estimatedWorkingBytes(model: broad, size: 1024)
        XCTAssertGreaterThanOrEqual(broadBytes, UInt64(3 * 1_073_741_824))
        XCTAssertNoThrow(try NativeMaterialTrainer.admitTraining(model: focused, size: 1024, budget: focusedBytes))
        XCTAssertThrowsError(try NativeMaterialTrainer.admitTraining(model: broad, size: 1024, budget: broadBytes - 1))
    }

    func testCoarseWorkspaceBudgetReservesCheckpointsFrozenFeaturesAndModelWorkingSet() {
        let gib = UInt64(1_073_741_824)
        let lowMemory = NativeMaterialModel.Program.coarseWorkspaceBudget(capacity: 18 * gib,
            checkpointBytes: 15 * gib / 4, workingBytes: 21 * gib / 2)
        XCTAssertEqual(lowMemory, 21 * gib / 8)
        XCTAssertLessThan(lowMemory, 27 * gib / 2,
            "A 13.5 GiB coarse VJP must retain fine cuts when an 18 GiB machine has only 2.625 GiB of unreserved workspace.")

        let capacity = 51 * gib + 84 * gib / 100
        let largeMachine = NativeMaterialModel.Program.coarseWorkspaceBudget(capacity: capacity,
            checkpointBytes: 8 * gib, workingBytes: 12 * gib)
        XCTAssertEqual(largeMachine, capacity - 22 * gib)
        XCTAssertGreaterThanOrEqual(largeMachine, 27 * gib,
            "The 64 GiB reference capacity retains enough reserved headroom to admit a 27 GiB coarse block.")
    }

    func testCoarseWorkspaceBudgetSaturatesAndClampsCheckpointReservations() {
        let gib = UInt64(1_073_741_824)
        XCTAssertEqual(NativeMaterialModel.Program.coarseWorkspaceBudget(capacity: 18 * gib,
            checkpointBytes: .max, workingBytes: .max), 0)
        XCTAssertEqual(NativeMaterialModel.Program.coarseWorkspaceBudget(capacity: 0,
            checkpointBytes: .max, workingBytes: .max), 0)
        XCTAssertEqual(NativeMaterialModel.Program.coarseWorkspaceBudget(capacity: 18 * gib,
            checkpointBytes: .max, workingBytes: 0), 63 * gib / 8,
            "Checkpoint storage is independently capped at half the machine capacity.")
    }

    func testFrozenBlocksUseFewerStagesWhileAdaptedDecoderKeepsItsCuts() throws {
        let fixture = NativeMaterialModelFixture()
        let focused = try fixture.model(scope: "final-map", rank: 8, alpha: 8)
        let decoder = try fixture.model(scope: "map-decoder", rank: 8, alpha: 8)
        let focusedProgram = try NativeMaterialModel.Program(model: focused, width: 64, height: 64, target: "height", coalesceActiveBlocks: false)
        let decoderProgram = try NativeMaterialModel.Program(model: decoder, width: 64, height: 64, target: "height", coalesceActiveBlocks: false)
        XCTAssertLessThan(focusedProgram.frozenStageCount, 150,
            "Frozen generator blocks should need one forward package each, not cuts for each normalization and activation.")
        XCTAssertGreaterThan(decoderProgram.frozenStageCount, focusedProgram.frozenStageCount + 60,
            "An adapted decoder and all downstream derivative paths must keep their bounded backward cuts.")
    }

    func testStagedMetadataRebuildsAfterEntryForwardAndBackwardCancellation() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let model = try NativeMaterialModelFixture().model(adapter: true)
            let program = try model.program(width: 64, height: 64, target: "height")
            let rgb = (0..<12_288).map { Float($0 % 97) / 97 }
            let reference = [Float](repeating: 0.2, count: 4_096)
            let baseline = try program.execute(rgb: rgb, adapters: model.adapterWeights,
                reference: reference, featureKey: "same-input")
            let entryControl = NativeMaterialTrainingControl()
            entryControl.stop()
            XCTAssertThrowsError(try program.execute(rgb: rgb, adapters: model.adapterWeights,
                reference: reference, featureKey: "same-input", checkCancellation: { try entryControl.check() })) {
                XCTAssertTrue($0 is CancellationError)
            }
            var graphBuilds = 0, stageCalls = 0
            let update = try program.execute(rgb: rgb, adapters: model.adapterWeights,
                reference: reference, learningRate: 1e-3, featureKey: "same-input",
                onStage: { _, _ in stageCalls += 1 }, onOperation: { operation, done, _ in
                    if operation == "Building model execution graph", done == 0 { graphBuilds += 1 }
                })
            XCTAssertEqual(graphBuilds, 1, "Entry cancellation must discard the retained metadata graph.")
            XCTAssertGreaterThan(stageCalls, 10, "An aborted execution must discard its frozen feature cache.")
            XCTAssertEqual(Set(update.updated.keys), Set(model.adapterWeights.keys))
            XCTAssertEqual(update.optimizerState.count, model.adapterWeights.count * 2)
            XCTAssertNotEqual(update.updated["ups.3.model.10.lora_B"]!.bytes,
                model.adapterWeights["ups.3.model.10.lora_B"]!.bytes)
            let backwardControl = NativeMaterialTrainingControl()
            XCTAssertThrowsError(try program.execute(rgb: rgb, adapters: model.adapterWeights,
                reference: reference, gradientsOnly: true, featureKey: "same-input",
                checkCancellation: { try backwardControl.check() }, onOperation: { operation, done, _ in
                    if operation == "Computing loss gradients", done == 1 { backwardControl.stop() }
                })) { XCTAssertTrue($0 is CancellationError) }
            let recovered = try program.execute(rgb: rgb, adapters: model.adapterWeights,
                reference: reference, learningRate: 1e-3, featureKey: "same-input")
            for name in update.updated.keys { XCTAssertEqual(recovered.updated[name]!.bytes, update.updated[name]!.bytes) }
            for name in update.optimizerState.keys { XCTAssertEqual(recovered.optimizerState[name]!.bytes, update.optimizerState[name]!.bytes) }
            let forwardControl = NativeMaterialTrainingControl()
            XCTAssertThrowsError(try program.execute(rgb: rgb, adapters: model.adapterWeights,
                reference: reference, featureKey: "uncached-input", checkCancellation: { try forwardControl.check() },
                onStage: { _, _ in forwardControl.stop() })) {
                XCTAssertTrue($0 is CancellationError)
            }
            graphBuilds = 0
            let retried = try program.execute(rgb: rgb, adapters: model.adapterWeights,
                reference: reference, featureKey: "same-input", onOperation: { operation, done, _ in
                    if operation == "Building model execution graph", done == 0 { graphBuilds += 1 }
                })
            XCTAssertEqual(graphBuilds, 1)
            XCTAssertEqual(retried.output, baseline.output)
            XCTAssertEqual(retried.loss, baseline.loss)
        }.value
    }

    func testReusedStagedMetadataRejectsChangedFactorContractsBeforeGraphRebuild() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let model = try NativeMaterialModelFixture().model(adapter: true)
            let program = try model.program(width: 64, height: 64, target: "height")
            let rgb = [Float](repeating: 0.5, count: 12_288)
            let baseline = try program.execute(rgb: rgb, adapters: model.adapterWeights)
            let name = "ups.3.model.10.lora_A", original = model.adapterWeights["ups.3.model.10.lora_A"]!
            var missing = model.adapterWeights; missing.removeValue(forKey: name)
            var extra = model.adapterWeights; extra["unexpected.lora_A"] = .floats([1], shape: [1])
            var reshaped = model.adapterWeights
            reshaped[name] = NativeTensor(dtype: "F32", shape: Array(original.shape.reversed()), bytes: original.bytes)
            var integer = model.adapterWeights
            integer[name] = NativeTensor(dtype: "I32", shape: original.shape, bytes: original.bytes)
            var truncated = model.adapterWeights
            truncated[name] = NativeTensor(dtype: "F32", shape: original.shape, bytes: Data(original.bytes.dropLast()))
            for factors in [missing, extra, reshaped, integer, truncated] {
                var graphBuilds = 0
                XCTAssertThrowsError(try program.execute(rgb: rgb, adapters: factors, onOperation: { operation, done, _ in
                    if operation == "Building model execution graph", done == 0 { graphBuilds += 1 }
                }))
                XCTAssertEqual(graphBuilds, 0, "Reject changed cached-package structure before building a new graph.")
            }
            let retried = try program.execute(rgb: rgb, adapters: model.adapterWeights)
            XCTAssertEqual(retried.output, baseline.output, "Valid factors can retry after a rejected execution.")
        }.value
    }

    func testCachedProgramDoesNotRetainOwningModel() throws {
        weak var releasedModel: NativeMaterialModel?
        weak var releasedProgram: NativeMaterialModel.Program?
        do {
            let model = try NativeMaterialModelFixture().model()
            let program = try model.program(width: 64, height: 64, target: "height")
            releasedModel = model; releasedProgram = program
            XCTAssertNotNil(releasedModel); XCTAssertNotNil(releasedProgram)
        }
        XCTAssertNil(releasedModel)
        XCTAssertNil(releasedProgram)
    }
    func testGraphResultsFollowRequestedTensorIdentitiesWithEqualShapes() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let graph = MPSGraph()
            let input = graph.placeholder(shape: [2], dataType: .float32, name: "values")
            let doubled = graph.multiplication(input, graph.constant(2, dataType: .float32), name: nil)
            let offset = graph.addition(input, graph.constant(10, dataType: .float32), name: nil)
            let feeds = [input: try NativeGraphExecution.tensorData(.floats([3, 7], shape: [2]))]
            var cache: [String: MPSGraphExecutable] = [:]
            for _ in 0..<2 {
                let first = try NativeGraphExecution.run(graph, feeds: feeds, targets: [offset, input, doubled], cache: &cache)
                XCTAssertEqual(first, [[13, 17], [3, 7], [6, 14]])
                let second = try NativeGraphExecution.run(graph, feeds: feeds, targets: [doubled, offset, input], cache: &cache)
                XCTAssertEqual(second, [[6, 14], [13, 17], [3, 7]])
            }
        }.value
    }
    func testNearestNeighborUpsamplingPreservesFloatBitsAndSumsFourGradientCopies() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let samples: [Float] = [Float(bitPattern: 0x80000000), .leastNonzeroMagnitude, 0.125, -2, 17, 0.75]
            let graph = MPSGraph()
            let input = graph.placeholder(shape: [1, 1, 2, 3], dataType: .float32, name: "native_samples")
            let output = NativeGraphExecution.nearestNeighbor2(input, graph: graph)
            let sum = graph.reductionSum(with: output, axes: [0, 1, 2, 3], name: nil)
            let gradient = graph.gradients(of: sum, with: [input], name: nil)[input]!
            let result = try NativeGraphExecution.run(graph,
                feeds: [input: NativeGraphExecution.tensorData(.floats(samples, shape: [1, 1, 2, 3]))], targets: [output, gradient])
            XCTAssertEqual(output.shape, [1, 1, 4, 6])
            let expected = [samples[0], samples[0], samples[1], samples[1], samples[2], samples[2],
                            samples[0], samples[0], samples[1], samples[1], samples[2], samples[2],
                            samples[3], samples[3], samples[4], samples[4], samples[5], samples[5],
                            samples[3], samples[3], samples[4], samples[4], samples[5], samples[5]]
            XCTAssertEqual(result[0].map(\.bitPattern), expected.map(\.bitPattern))
            XCTAssertEqual(result[1], [Float](repeating: 4, count: samples.count))
        }.value
    }
    func testPinnedLayoutMatchesPublishedCheckpointTensorDescriptors() {
        // Digest independently recorded from the pinned archive's data.pkl
        // metadata, fetched with bounded range requests without weight data.
        // All 2892 consumed tensors match; the four auxiliary heads are unused.
        let fixture = NativeMaterialModelFixture(architecture: .pinned, shapesOnly: true)
        let record = fixture.weights.keys.sorted().map { name in
            let tensor = fixture.weights[name]!
            return name + "|" + tensor.dtype + "|" + tensor.shape.map(String.init).joined(separator: ",")
        }.joined(separator: "\n")
        XCTAssertEqual(fixture.weights.count, 2892)
        XCTAssertEqual(NativeMaterialTrainer.checksum(Data(record.utf8)), "b0761102648c4bde2debd3b6ae7828a18f6e5e2605289af5e31c51866bdb20d9")
    }
    func testCompleteNativeGraphPreservesRectangleAndPredictsAllMapShapes() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
        XCTAssertFalse(Thread.isMainThread)
        let fixture = NativeMaterialModelFixture()
        let model = try fixture.model()
        let rgb = (0..<3 * 64 * 128).map { Float($0 % 100) / 100 }
        for target in ["height", "normal", "roughness"] {
            let result = try model.predict(rgb: rgb, width: 128, height: 64, target: target)
            XCTAssertEqual(result.width, 128); XCTAssertEqual(result.height, 64)
            XCTAssertEqual(result.values.count, 128 * 64 * (target == "normal" ? 3 : 1))
            XCTAssertTrue(result.values.allSatisfy(\.isFinite))
        }
        }.value
    }
    func testLoRAGradientAdamUpdateAndLossAreRealOnFullOperationSequence() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
        XCTAssertFalse(Thread.isMainThread)
        let fixture = NativeMaterialModelFixture()
        let model = try fixture.model(adapter: true)
        let rgb = (0..<3 * 64 * 64).map { Float($0 % 100) / 100 }
        let program = try model.program(width: 64, height: 64, target: "height")
        let before = try program.execute(rgb: rgb, adapters: model.adapterWeights)
        let reference = before.output.map { $0 + 0.01 }
        let update = try program.execute(rgb: rgb, adapters: model.adapterWeights, reference: reference, learningRate: 1e-3)
        XCTAssertEqual(update.valueLoss!, 0.01, accuracy: 2e-6)
        XCTAssertGreaterThan(update.loss!, 0)
        XCTAssertFalse(update.updated.isEmpty)
        XCTAssertEqual(update.optimizerState.count, update.updated.count * 2)
        let name = "ups.3.model.10.lora_B"
        XCTAssertNotEqual(update.updated[name]!.bytes, model.adapterWeights[name]!.bytes)
        let changed = try program.execute(rgb: rgb, adapters: update.updated)
        XCTAssertLessThan(zip(changed.output, reference).reduce(Float(0)) { $0 + abs($1.0 - $1.1) } / Float(reference.count), update.valueLoss!)
        XCTAssertEqual(try model.fusedWeights()["ups.3.model.10.weight"]!.bytes, model.baseWeights["ups.3.model.10.weight"]!.bytes)
        model.updateAdapters(update.updated)
        let fused = try NativeMaterialModel(baseWeights: model.fusedWeights(), baseSHA256: "fixture", architecture: .test)
        let merged = try fused.predict(rgb: rgb, width: 64, height: 64, target: "height")
        for (first, second) in zip(changed.output, merged.values) { XCTAssertEqual(first, second, accuracy: 2e-5) }
        }.value
    }
    func testValidationDoesNotConstructAutodiffOrRequireAdapters() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            XCTAssertFalse(Thread.isMainThread)
            let model = try NativeMaterialModelFixture().model()
            let program = try model.program(width: 64, height: 64, target: "height")
            let result = try program.execute(rgb: [Float](repeating: 0.5, count: 3 * 64 * 64),
                adapters: [:], reference: [Float](repeating: 0.5, count: 64 * 64))
            XCTAssertTrue(result.loss!.isFinite)
            XCTAssertTrue(result.updated.isEmpty)
            XCTAssertTrue(result.optimizerState.isEmpty)
            XCTAssertFalse(program.optimizerPrepared)
        }.value
    }
    func testBothTrainingScopesUpdateAllLayersIncludingDecoderUpsamplingAtRank64() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            XCTAssertFalse(Thread.isMainThread)
            let fixture = NativeMaterialModelFixture()
            let rgb: [Float] = (0..<12_288).map { index in Float((index * 17) % 255) / 255 }
            let baseline = try fixture.model().predict(rgb: rgb, width: 64, height: 64, target: "height")
            for scope in ["final-map", "map-decoder"] {
                let model = try fixture.model(scope: scope, rank: 64, alpha: 16)
                let program = try model.program(width: 64, height: 64, target: "height")
                XCTAssertGreaterThan(program.frozenStageCount, 10)
                let reference = baseline.values.map { $0 + 0.01 }
                let validation = try program.execute(rgb: rgb, adapters: model.adapterWeights, reference: reference)
                XCTAssertFalse(program.optimizerPrepared)
                for (before, staged) in zip(baseline.values, validation.output) {
                    XCTAssertEqual(before, staged, accuracy: 2e-6)
                }
                let updated = try program.execute(rgb: rgb, adapters: model.adapterWeights,
                    reference: reference, learningRate: 1e-3)
                XCTAssertTrue(program.optimizerPrepared)
                XCTAssertGreaterThan(program.frozenStageCount, 10)
                XCTAssertEqual(Set(updated.updated.keys), Set(model.adapterWeights.keys))
                XCTAssertEqual(updated.optimizerState.count, model.adapterWeights.count * 2)
                let outputLayer = "ups.3.model.10.lora_B"
                XCTAssertNotEqual(updated.updated[outputLayer]!.bytes, model.adapterWeights[outputLayer]!.bytes)
                if scope == "map-decoder" {
                    // This reaches the nearest-neighbor upsampler derivative
                    // that used to abort inside MPSGraphTileOp.
                    let firstUpsample = "gen.m_dec_3.m_up3.0.up.1.lora_B"
                    XCTAssertNotEqual(updated.updated[firstUpsample]!.bytes, model.adapterWeights[firstUpsample]!.bytes)
                }
            }
        }.value
    }
    func testRejectsAlteredPixelGridAndMissingLearnedLayer() throws {
        let fixture = NativeMaterialModelFixture()
        let model = try fixture.model()
        XCTAssertThrowsError(try model.predict(rgb: [Float](repeating: 0, count: 65 * 64 * 3), width: 65, height: 64, target: "height"))
        var weights = fixture.weights
        weights.removeValue(forKey: "gen.m_head.weight")
        let incomplete = try NativeMaterialModel(baseWeights: weights, baseSHA256: "fixture", architecture: .test)
        XCTAssertThrowsError(try incomplete.predict(rgb: [Float](repeating: 0, count: 64 * 64 * 3), width: 64, height: 64, target: "height"))
        XCTAssertThrowsError(try model.program(width: 0, height: 64, target: "height"))
        XCTAssertThrowsError(try model.program(width: -64, height: -64, target: "height"))
        var invalid = fixture.weights
        let name = "gen.m_body.0.trans_block.msa.relative_position_index"
        let indices = [Int64](repeating: 225, count: 4096)
        invalid[name] = .init(dtype: "I64", shape: [64, 64], bytes: indices.withUnsafeBytes { Data($0) })
        XCTAssertThrowsError(try NativeMaterialModel(baseWeights: invalid, baseSHA256: "fixture", architecture: .test))
    }
    func testMalformedAdapterDimensionsFailBeforeGraphCompilation() throws {
        let name = "ups.3.model.10", base = [name + ".weight": NativeTensor.floats([1, 2], shape: [1, 2])]
        let layers = [name: NativeMaterialModel.AdapterLayer(weightShape: [1, 2], rank: 1, alpha: 1)]
        let factors = [name + ".lora_A": NativeTensor.floats([1, 2], shape: [1, 2]), name + ".lora_B": NativeTensor.floats([0], shape: [1, 1])]
        XCTAssertNoThrow(try NativeMaterialModel(baseWeights: base, adapterWeights: factors, layers: layers, baseSHA256: "fixture"))
        // Equal element counts with a different matrix layout are incompatible.
        let transposed = [name: NativeMaterialModel.AdapterLayer(weightShape: [2, 1], rank: 1, alpha: 1)]
        XCTAssertThrowsError(try NativeMaterialModel(baseWeights: base, adapterWeights: factors, layers: transposed, baseSHA256: "fixture"))
        var missing = factors; missing.removeValue(forKey: name + ".lora_B")
        XCTAssertThrowsError(try NativeMaterialModel(baseWeights: base, adapterWeights: missing, layers: layers, baseSHA256: "fixture"))
        var extra = factors; extra["unexpected.lora_A"] = .floats([1], shape: [1])
        XCTAssertThrowsError(try NativeMaterialModel(baseWeights: base, adapterWeights: extra, layers: layers, baseSHA256: "fixture"))
        var wrongType = factors; wrongType[name + ".lora_B"] = .init(dtype: "I32", shape: [1, 1], bytes: Data(repeating: 0, count: 4))
        XCTAssertThrowsError(try NativeMaterialModel(baseWeights: base, adapterWeights: wrongType, layers: layers, baseSHA256: "fixture"))
        for rank in [0, Int.max] {
            let invalid = [name: NativeMaterialModel.AdapterLayer(weightShape: [1, 2], rank: rank, alpha: 1)]
            XCTAssertThrowsError(try NativeMaterialModel(baseWeights: base, adapterWeights: factors, layers: invalid, baseSHA256: "fixture"))
        }
        let overflow = [name + ".weight": NativeTensor(dtype: "F32", shape: [Int.max, 2], bytes: Data())]
        XCTAssertThrowsError(try NativeMaterialModel(baseWeights: overflow, baseSHA256: "fixture"))
    }
}

/// Every PBRnxt operation is present with smaller channel widths; fixture
/// tensors are synthesized directly in Swift, without a Python runtime.
struct NativeMaterialModelFixture {
    var weights: [String: NativeTensor] = [:]
    init(architecture a: NativeMaterialModel.Architecture = .test, shapesOnly: Bool = false) {
        let d = a.dim
        func add(_ name: String, _ shape: [Int], _ value: Float) {
            weights[name] = shapesOnly ? NativeTensor(dtype: "F32", shape: shape, bytes: Data()) : .floats([Float](repeating: value, count: shape.reduce(1, *)), shape: shape)
        }
        func conv(_ name: String, _ incoming: Int, _ outgoing: Int, kernel: Int = 3, bias: Bool = true) {
            add(name + ".weight", [outgoing, incoming, kernel, kernel], 0.05 / Float(incoming * kernel * kernel))
            if bias { add(name + ".bias", [outgoing], 0.01) }
        }
        func linear(_ name: String, _ incoming: Int, _ outgoing: Int, bias: Bool = true) {
            add(name + ".weight", [outgoing, incoming], 0.05 / Float(incoming))
            if bias { add(name + ".bias", [outgoing], 0.01) }
        }
        func norm(_ name: String, _ dim: Int) { add(name + ".weight", [dim], 1); add(name + ".bias", [dim], 0) }
        func block(_ prefix: String, _ channels: Int) {
            let half = channels / 2
            conv(prefix + ".conv1_1", channels, channels, kernel: 1)
            conv(prefix + ".conv1_2", channels, channels, kernel: 1)
            conv(prefix + ".conv_block.dwconv", 1, half, kernel: 7)
            norm(prefix + ".conv_block.norm", half)
            linear(prefix + ".conv_block.pwconv1", half, half * 4)
            add(prefix + ".conv_block.grn.gamma", [1, 1, 1, half * 4], 0)
            add(prefix + ".conv_block.grn.beta", [1, 1, 1, half * 4], 0)
            linear(prefix + ".conv_block.pwconv2", half * 4, half)
            add(prefix + ".conv_block.gamma", [half], 1e-6)
            let p = prefix + ".trans_block"
            norm(p + ".ln1", half); norm(p + ".ln2", half)
            linear(p + ".mlp.0", half, half * 4); linear(p + ".mlp.2", half * 4, half)
            linear(p + ".msa.embedding_layer", half, half * 3, bias: false)
            add(p + ".msa.q_bias", [half], 0); add(p + ".msa.v_bias", [half], 0)
            add(p + ".msa.logit_scale", [a.heads, 1, 1], log(10))
            add(p + ".msa.relative_coords_table", [1, 15, 15, 2], 0)
            weights[p + ".msa.relative_position_index"] = NativeTensor(dtype: "I64", shape: [64, 64], bytes: [Int64](repeating: 0, count: 4096).withUnsafeBytes { Data($0) })
            linear(p + ".msa.cpb_mlp.0", 2, 512); linear(p + ".msa.cpb_mlp.2", 512, a.heads, bias: false)
            linear(p + ".msa.linear", half, half)
        }
        conv("gen.m_head", 3, d, bias: false)
        for level in 1...3 {
            let channels = d << (level - 1), prefix = "gen.m_enc.m_down\(level)"
            for i in 0..<a.encoderBlocks { block(prefix + ".\(i)", channels) }
            conv(prefix + ".\(a.encoderBlocks)", channels, channels * 2, kernel: 2, bias: false)
        }
        for i in 0..<a.encoderBlocks { block("gen.m_body.\(i)", d * 8) }
        for branch in 0..<4 {
            for level in (1...3).reversed() {
                let channels = d << level, prefix = "gen.m_dec_\(branch).m_up\(level)"
                conv(prefix + ".0.up.1", channels, channels, bias: false)
                conv(prefix + ".0.up.3", channels, channels / 2, bias: false)
                for i in 0..<a.decoderBlocks { block(prefix + ".\(i + 1)", channels / 2) }
            }
            conv("gen.m_tail_\(branch).0", d, [3, 3, 1, 1][branch], bias: false)
        }
        conv("gen.m_fuse.0", d * 4, d)
        for i in 0..<a.fusionBlocks { block("gen.m_fuse.\(i + 1)", d) }
        conv("gen.m_fuse.\(a.fusionBlocks + 1)", d, d * 4)
        for branch in 0..<4 {
            let p = "ups.\(branch).model", width = a.rrdbWidth, growth = a.growth
            conv(p + ".0", 11, width)
            for i in 0..<a.rrdbBlocks { for r in 1...3 {
                let q = p + ".1.sub.\(i).RDB\(r)"
                conv(q + ".conv1x1", width, growth, kernel: 1, bias: false)
                for index in 1...4 { conv(q + ".conv\(index).0", width + (index - 1) * growth, growth) }
                conv(q + ".conv5.0", width + 4 * growth, width)
            } }
            conv(p + ".1.sub.\(a.rrdbBlocks)", width, width)
            for index in [3, 6, 8] { conv(p + ".\(index)", width, width) }
            conv(p + ".10", width, [3, 3, 1, 1][branch])
        }
    }
    func model(adapter: Bool = false) throws -> NativeMaterialModel {
        let name = "ups.3.model.10", shape = weights[name + ".weight"]!.shape
        let specs: [String: NativeMaterialModel.AdapterLayer] = adapter ? [name: .init(weightShape: shape, rank: 1, alpha: 1)] : [:]
        let adapters: [String: NativeTensor] = adapter ? [name + ".lora_A": .floats([Float](repeating: 0.1, count: shape.dropFirst().reduce(1, *)), shape: [1, shape.dropFirst().reduce(1, *)]), name + ".lora_B": .floats([0], shape: [1, 1])] : [:]
        return try NativeMaterialModel(baseWeights: weights, adapterWeights: adapters, layers: specs, baseSHA256: "fixture", architecture: .test)
    }
    func model(scope: String, rank: Int, alpha: Float, target: String = "height") throws -> NativeMaterialModel {
        let branch = try XCTUnwrap(["normal": 1, "roughness": 2, "height": 3][target])
        let prefixes = ["ups.\(branch)."] + (scope == "map-decoder" ? ["gen.m_dec_\(branch).", "gen.m_tail_\(branch)."] : [])
        var layers: [String: NativeMaterialModel.AdapterLayer] = [:], factors: [String: NativeTensor] = [:]
        var random = NativeMaterialRandom(seed: 17)
        for key in weights.keys.sorted() where key.hasSuffix(".weight") && !key.contains(".dwconv.") && prefixes.contains(where: { key.hasPrefix($0) }) {
            let weight = weights[key]!
            guard [2, 4].contains(weight.shape.count) else { continue }
            let name = String(key.dropLast(7)), incoming = weight.shape.dropFirst().reduce(1, *)
            layers[name] = .init(weightShape: weight.shape, rank: rank, alpha: alpha)
            factors[name + ".lora_A"] = .floats((0..<rank * incoming).map { _ in
                (random.unit() * 2 - 1) / sqrt(Float(incoming))
            }, shape: [rank, incoming])
            factors[name + ".lora_B"] = .floats([Float](repeating: 0, count: weight.shape[0] * rank), shape: [weight.shape[0], rank])
        }
        return try NativeMaterialModel(baseWeights: weights, adapterWeights: factors, layers: layers,
            baseSHA256: "fixture", architecture: .test)
    }
}
