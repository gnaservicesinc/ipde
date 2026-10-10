import Foundation
import Metal
import XCTest
@testable import TextureStudio

final class NativeGraphStageCoalescingTests: XCTestCase {
    func testActiveBlockCoalescingPreservesVariedFloat32TrainingOverTwoSteps() async throws {
        try await verifyTraining(size: 128, coalesceActiveBlocks: true)
    }

    func testActiveBlockCoalescingPreservesNormalAndRoughnessFloat32TrainingOverTwoSteps() async throws {
        for target in ["normal", "roughness"] {
            try await verifyTraining(size: 128, coalesceActiveBlocks: true, target: target)
        }
    }

    private func verifyTraining(size: Int, coalesceActiveBlocks: Bool, target: String = "height") async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            // More channels and varied learned weights make the 3x3 convolutions
            // nontrivial and avoid the constant shape fixture's nearly uniform
            // attention and vanishing upstream derivatives.
            let architecture = NativeMaterialModel.Architecture(dim: 8, heads: 1,
                encoderBlocks: 2, decoderBlocks: 2, fusionBlocks: 2,
                rrdbBlocks: 1, rrdbWidth: 8, growth: 4)
            let fixture = NativeMaterialModelFixture(architecture: architecture)
            let base = try Self.variedWeights(fixture.weights)
            let pixels = size * size
            let outputCount = pixels * (target == "normal" ? 3 : 1)
            // Synthetic references exercise the numerical training path for
            // each branch, without making a claim about material-map quality.
            let reference = (0..<outputCount).map { index in
                Float(0.25) + Float((index * 17 + index / size) % 127) / Float(635)
            }
            for scope in ["final-map", "map-decoder"] {
                print("STAGE_COALESCING_NUMERICAL target=\(target) scope=\(scope) size=\(size) coarseActive=\(coalesceActiveBlocks) precision=Float32")
                let original = try fixture.model(scope: scope, rank: 4, alpha: 8, target: target)
                var starting = original.adapterWeights
                var random = NativeMaterialRandom(seed: 9007)
                for name in starting.keys.sorted() where name.hasSuffix(".lora_B") {
                    let tensor = starting[name]!
                    let values = (0..<tensor.shape.reduce(1, *)).map { _ in
                        (random.unit() * 2 - 1) * Float(0.005)
                    }
                    starting[name] = .floats(values, shape: tensor.shape)
                }
                let model = try NativeMaterialModel(baseWeights: base, adapterWeights: starting,
                    layers: original.layers, configuration: original.configuration,
                    baseSHA256: original.baseSHA256, architecture: architecture)
                // Compare active-block coalescing with the fine-stage execution
                // path, preserving its layout and numerical boundary behavior.
                // A separate monolithic control differed identically with fine
                // and coarse stages at a near-zero gradient around Adam epsilon;
                // that comparison cannot attribute a new coalescing regression.
                // Standard fixture tests retain independent monolithic checks.
                let baseline = try NativeMaterialModel.Program(model: model, width: size, height: size,
                    target: target, staged: true, checkpointByteLimit: 32 * 1024 * 1024,
                    coalesceActiveBlocks: false)
                let candidate = try NativeMaterialModel.Program(model: model, width: size, height: size,
                    target: target, staged: true, checkpointByteLimit: 32 * 1024 * 1024,
                    coalesceActiveBlocks: coalesceActiveBlocks)
                var expectedWeights = starting, actualWeights = starting
                var expectedState: [String: NativeTensor] = [:], actualState: [String: NativeTensor] = [:]
                let optimizer = NativeMaterialOptimizerConfiguration(maxGradientNorm: 0)
                for step in 1...2 {
                    // Changed pixels on step two also exercise replay/cache
                    // invalidation while independently evolving both Adam states.
                    let rgb = (0..<3 * pixels).map { index in
                        Float((index * 37 + index / size + (step - 1) * 19) % 251) / Float(251)
                    }
                    let expected = try baseline.execute(rgb: rgb, adapters: expectedWeights,
                        reference: reference, gradientsOnly: true, featureKey: "stage-coalescing-step-\(step)")
                    let actual = try candidate.execute(rgb: rgb, adapters: actualWeights,
                        reference: reference, gradientsOnly: true, featureKey: "stage-coalescing-step-\(step)")
                    let label = "\(target), \(scope), \(size), coarseActive=\(coalesceActiveBlocks), step \(step)"
                    XCTAssertEqual(expected.output.count, outputCount, label)
                    XCTAssertEqual(actual.output.count, outputCount, label)
                    XCTAssertTrue(expected.output.allSatisfy(\.isFinite), label)
                    XCTAssertTrue(actual.output.allSatisfy(\.isFinite), label)
                    for (name, a, e) in [("loss", actual.loss, expected.loss),
                                         ("value loss", actual.valueLoss, expected.valueLoss),
                                         ("gradient loss", actual.gradientLoss, expected.gradientLoss)] {
                        let a = try XCTUnwrap(a), e = try XCTUnwrap(e)
                        XCTAssertTrue(a.isFinite && e.isFinite, "\(label), \(name)")
                        XCTAssertEqual(a, e, accuracy: abs(e) * 0.005 + 1e-4, "\(label), \(name)")
                    }
                    try Self.compare(actual.output, expected.output, relativeTolerance: 0.005,
                        absoluteTolerance: 1e-4, name: "\(label), native \(target) map")
                    XCTAssertEqual(Set(actual.gradients.keys), Set(starting.keys), label)
                    XCTAssertEqual(Set(expected.gradients.keys), Set(starting.keys), label)
                    for name in starting.keys.sorted() {
                        try Self.compare(try XCTUnwrap(actual.gradients[name]), try XCTUnwrap(expected.gradients[name]),
                            relativeTolerance: 0.025, name: "\(label), gradient \(name)")
                    }
                    let expectedUpdate = try NativeMaterialOptimizer.apply(gradients: expected.gradients,
                        weights: expectedWeights, state: expectedState, learningRate: 1e-4,
                        step: step, configuration: optimizer)
                    let actualUpdate = try NativeMaterialOptimizer.apply(gradients: actual.gradients,
                        weights: actualWeights, state: actualState, learningRate: 1e-4,
                        step: step, configuration: optimizer)
                    XCTAssertEqual(Set(actualUpdate.weights.keys), Set(starting.keys), label)
                    XCTAssertEqual(Set(actualUpdate.state.keys), Set(expectedUpdate.state.keys), label)
                    for name in expectedUpdate.state.keys.sorted() {
                        // Squared gradients double the first-order relative
                        // error envelope, so v has a separate explicit bound.
                        try Self.compare(try XCTUnwrap(actualUpdate.state[name]), try XCTUnwrap(expectedUpdate.state[name]),
                            relativeTolerance: name.hasSuffix(".v") ? 0.05 : 0.025,
                            name: "\(label), Adam state \(name)")
                    }
                    for name in starting.keys.sorted() {
                        let initial = try starting[name]!.floatValues()
                        let actualParameters = try XCTUnwrap(actualUpdate.weights[name]).floatValues()
                        let expectedParameters = try XCTUnwrap(expectedUpdate.weights[name]).floatValues()
                        let actualDelta = zip(actualParameters, initial).map { $0 - $1 }
                        let expectedDelta = zip(expectedParameters, initial).map { $0 - $1 }
                        let scale = expectedDelta.reduce(Float.leastNormalMagnitude) { max($0, abs($1)) }
                        let tolerance = scale * Float(0.025) + Float(4e-6)
                        if let index = actualDelta.indices.max(by: {
                            abs(actualDelta[$0] - expectedDelta[$0]) < abs(actualDelta[$1] - expectedDelta[$1])
                        }), abs(actualDelta[index] - expectedDelta[index]) > tolerance {
                            func coordinate(_ tensor: NativeTensor?) throws -> Float {
                                guard let tensor else { return 0 }
                                return try tensor.floatValues()[index]
                            }
                            func details(weights: [String: NativeTensor], state: [String: NativeTensor],
                                         gradients: [String: NativeTensor], update: NativeMaterialOptimizer.Update,
                                         delta: Float) throws -> [String: Any] {
                                let before = try coordinate(weights[name])
                                let after = try coordinate(update.weights[name])
                                let m = try coordinate(update.state[name + ".m"])
                                let v = try coordinate(update.state[name + ".v"])
                                let correctedM = Double(m) / (1 - pow(Double(optimizer.beta1), Double(step)))
                                let correctedV = Double(v) / (1 - pow(Double(optimizer.beta2), Double(step)))
                                let denominator = sqrt(correctedV) + Double(optimizer.epsilon)
                                return ["gradient_before_update": try coordinate(gradients[name]),
                                    "factor_before_update": before, "factor_after_update": after,
                                    "m_before_update": try coordinate(state[name + ".m"]),
                                    "v_before_update": try coordinate(state[name + ".v"]),
                                    "m_after_update": m, "v_after_update": v,
                                    "bias_corrected_m": correctedM, "bias_corrected_v": correctedV,
                                    "adam_denominator": denominator, "adam_direction": correctedM / denominator,
                                    "delta_this_step": after - before, "delta_since_initial": delta]
                            }
                            let diagnostic: [String: Any] = ["case": label, "target": target, "parameter": name,
                                "coordinate": index, "initial_factor": initial[index],
                                "maximum_delta_error": abs(actualDelta[index] - expectedDelta[index]),
                                "delta_error_bound": tolerance, "adam_epsilon": optimizer.epsilon,
                                "learning_rate": Float(1e-4),
                                "expected": try details(weights: expectedWeights, state: expectedState,
                                    gradients: expected.gradients, update: expectedUpdate, delta: expectedDelta[index]),
                                "actual": try details(weights: actualWeights, state: actualState,
                                    gradients: actual.gradients, update: actualUpdate, delta: actualDelta[index])]
                            let data = try JSONSerialization.data(withJSONObject: diagnostic, options: [.sortedKeys])
                            print("STAGE_COALESCING_ADAM_DIAGNOSTIC " + String(decoding: data, as: UTF8.self))
                        }
                        // Compare the learned change, not the large initial A
                        // factors that could hide a wrong optimizer update.
                        try Self.compare(actualDelta, expectedDelta,
                            relativeTolerance: 0.025, absoluteTolerance: 4e-6,
                            name: "\(label), learned delta \(name)")
                    }
                    expectedWeights = expectedUpdate.weights; actualWeights = actualUpdate.weights
                    expectedState = expectedUpdate.state; actualState = actualUpdate.state
                }
            }
        }.value
    }

    private static func compare(_ actual: NativeTensor, _ expected: NativeTensor,
                                relativeTolerance: Float, name: String) throws {
        XCTAssertEqual(actual.dtype, "F32", name)
        XCTAssertEqual(expected.dtype, "F32", name)
        XCTAssertEqual(actual.shape, expected.shape, name)
        try Self.compare(actual.floatValues(), expected.floatValues(), relativeTolerance: relativeTolerance,
            absoluteTolerance: Float.leastNonzeroMagnitude * 16, name: name)
    }

    private static func compare(_ actual: [Float], _ expected: [Float], relativeTolerance: Float,
                                absoluteTolerance: Float, name: String) throws {
        XCTAssertEqual(actual.count, expected.count, name)
        guard actual.count == expected.count else { return }
        XCTAssertTrue(actual.allSatisfy(\.isFinite) && expected.allSatisfy(\.isFinite), name)
        let scale = expected.reduce(Float.leastNormalMagnitude) { max($0, abs($1)) }
        let maximumError = zip(actual, expected).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        // Preserve the varied-weight qualification's predeclared error envelope
        // to attribute failures without moving its bounds. Standard fixture
        // tests independently impose tighter Float32 gradient/moment bounds.
        // Tiny gradients still retain a scale-relative comparison here.
        XCTAssertLessThanOrEqual(maximumError, scale * relativeTolerance + absoluteTolerance, name)
    }

    private static func variedWeights(_ original: [String: NativeTensor]) throws -> [String: NativeTensor] {
        var weights = original
        var random = NativeMaterialRandom(seed: 123_457)
        for name in weights.keys.sorted() where name.hasSuffix(".weight") {
            let tensor = weights[name]!
            guard tensor.dtype == "F32", [2, 4].contains(tensor.shape.count) else { continue }
            let incoming = tensor.shape.dropFirst().reduce(1, *)
            let scale = Float(0.8) / sqrt(Float(incoming))
            let values = (0..<tensor.shape.reduce(1, *)).map { _ in (random.unit() * 2 - 1) * scale }
            weights[name] = .floats(values, shape: tensor.shape)
        }
        var coordinates = [Float](repeating: 0, count: 15 * 15 * 2)
        for y in 0..<15 {
            for x in 0..<15 {
                for (axis, coordinate) in [y - 7, x - 7].enumerated() {
                    let normalized = Float(coordinate) * 8 / 7
                    let logarithm = Float(log2(Double(abs(normalized)) + 1) / 3)
                    coordinates[(y * 15 + x) * 2 + axis] = normalized < 0 ? -logarithm : logarithm
                }
            }
        }
        var positionIndices = [Int64](repeating: 0, count: 64 * 64)
        for a in 0..<64 {
            for b in 0..<64 {
                let dy = a / 8 - b / 8 + 7, dx = a % 8 - b % 8 + 7
                positionIndices[a * 64 + b] = Int64(dy * 15 + dx)
            }
        }
        for name in weights.keys.sorted() {
            if name.hasSuffix(".relative_coords_table") {
                weights[name] = .floats(coordinates, shape: [1, 15, 15, 2])
            } else if name.hasSuffix(".relative_position_index") {
                weights[name] = NativeTensor(dtype: "I64", shape: [64, 64], bytes: positionIndices.withUnsafeBytes { Data($0) })
            }
        }
        return weights
    }
}
