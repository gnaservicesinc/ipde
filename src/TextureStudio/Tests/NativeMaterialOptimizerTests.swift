import Foundation
import XCTest
@testable import TextureStudio

final class NativeMaterialOptimizerTests: XCTestCase {
    private let arguments = ["train", "--dataset", "/dataset", "--output", "/output"]

    func testSampleNumericalOverflowIsRecoverableAndDoesNotChangeWeightsOrAccumulation() throws {
        let weights = ["a": NativeTensor.floats([1], shape: [1])]
        let extreme = ["a": NativeTensor.floats([Float.greatestFiniteMagnitude], shape: [1])]
        XCTAssertThrowsError(try NativeMaterialOptimizer.apply(gradients: extreme, weights: weights, state: [:],
            learningRate: 0.001, step: 1, configuration: .init(maxGradientNorm: 0))) {
            XCTAssertTrue($0 is NativeMaterialSampleError)
        }
        XCTAssertEqual(try weights["a"]!.floatValues(), [1])
        var accumulated = NativeMaterialGradientAccumulator()
        try accumulated.add(extreme)
        var candidate = accumulated
        XCTAssertThrowsError(try candidate.add(extreme)) { XCTAssertTrue($0 is NativeMaterialSampleError) }
        XCTAssertEqual(accumulated.count, 1)
        XCTAssertEqual(try accumulated.averaged()["a"]!.floatValues(), [Float.greatestFiniteMagnitude])
        XCTAssertThrowsError(try accumulated.add(["missing": .floats([1], shape: [1])])) {
            XCTAssertFalse($0 is NativeMaterialSampleError, "Structural optimizer contracts must remain fatal")
        }
    }

    func testDefaultsPreservePreviousNumericsAndLegacyPreferencesDecode() throws {
        let options = try NativeMaterialTrainer.Options(arguments)
        XCTAssertEqual(options.learningRate, 1e-5)
        XCTAssertEqual(options.gradientAccumulationSteps, 1)
        XCTAssertEqual(options.optimizerConfiguration, .init())
        let saved = try JSONDecoder().decode(MaterialTrainingOptions.self, from: Data("{\"loraRank\":16}".utf8))
        XCTAssertEqual(saved.loraRank, 16)
        XCTAssertEqual(saved.optimizer, "adamw")
        XCTAssertNil(saved.configurationIssue)
        let roundTrip = try JSONDecoder().decode(MaterialTrainingOptions.self, from: JSONEncoder().encode(saved))
        XCTAssertEqual(saved, roundTrip)
    }

    func testRejectsUnsupportedNonfiniteUnderflowAndRoundedBetaValues() throws {
        for (flag, value) in [("--optimizer", "lion"), ("--learning-rate", "nan"), ("--learning-rate", "1e-100"),
            ("--gradient-accumulation-steps", "0"), ("--optimizer-beta1", "1"), ("--optimizer-beta2", "0.999999999999"),
            ("--optimizer-beta1", "-1e-100"), ("--optimizer-epsilon", "1e-100"), ("--weight-decay", "-1"),
            ("--max-gradient-norm", "inf"), ("--max-gradient-norm", "1e-100"), ("--weight-decay", "1e-100"),
            ("--warmup-updates", "-1"), ("--learning-rate-schedule", "unknown"),
            ("--minimum-learning-rate-ratio", "0"), ("--minimum-learning-rate-ratio", "2"), ("--seed", "invalid"),
            ("--seed", "18446744073709551616")] {
            XCTAssertThrowsError(try NativeMaterialTrainer.Options(arguments + [flag, value]), "\(flag) \(value)")
        }
        var saved = MaterialTrainingOptions()
        saved.learningRate = 1e-100; XCTAssertNotNil(saved.configurationIssue)
        saved.learningRate = 1e-5; saved.optimizerBeta1 = 0.999999999999; XCTAssertNotNil(saved.configurationIssue)
        saved = .init(); saved.loraAlpha = 1e100; XCTAssertNotNil(saved.configurationIssue)
    }

    func testWarmupAndCosineUseOptimizerUpdatesAndPositiveConfiguredFloor() throws {
        let options = try NativeMaterialTrainer.Options(arguments + ["--learning-rate", "0.01", "--warmup-updates", "2",
            "--learning-rate-schedule", "cosine", "--minimum-learning-rate-ratio", "0.2"])
        XCTAssertEqual(options.effectiveLearningRate(update: 1, totalUpdates: 6), 0.005, accuracy: 1e-8)
        XCTAssertEqual(options.effectiveLearningRate(update: 2, totalUpdates: 6), 0.01, accuracy: 1e-8)
        XCTAssertEqual(options.effectiveLearningRate(update: 3, totalUpdates: 6), 0.01, accuracy: 1e-8)
        XCTAssertEqual(options.effectiveLearningRate(update: 6, totalUpdates: 6), 0.002, accuracy: 1e-8)
        let large = try NativeMaterialTrainer.Options(arguments + ["--warmup-updates", String(Int.max)])
        XCTAssertGreaterThan(large.effectiveLearningRate(update: 1, totalUpdates: 2), 0)
    }

    func testGradientAccumulationAveragesActualCountBeforeGlobalClipping() throws {
        var accumulator = NativeMaterialGradientAccumulator()
        try accumulator.add(["a": .floats([2, 0], shape: [2]), "b": .floats([0], shape: [1])])
        try accumulator.add(["a": .floats([4, 0], shape: [2]), "b": .floats([8], shape: [1])])
        XCTAssertEqual(accumulator.count, 2)
        let gradients = try accumulator.averaged()
        XCTAssertEqual(try gradients["a"]!.floatValues(), [3, 0])
        XCTAssertEqual(try gradients["b"]!.floatValues(), [4])
        let update = try NativeMaterialOptimizer.apply(gradients: gradients,
            weights: ["a": .floats([1, 2], shape: [2]), "b": .floats([3], shape: [1])], state: [:],
            learningRate: 0.1, step: 1, configuration: .init(beta1: 0, beta2: 0, epsilon: 1, maxGradientNorm: 1))
        XCTAssertEqual(update.gradientNorm, 5, accuracy: 1e-8)
        let a = try update.weights["a"]!.floatValues(), b = try update.weights["b"]!.floatValues()
        XCTAssertEqual(a[0], 1 - 0.1 * 0.6 / 1.6, accuracy: 1e-6)
        XCTAssertEqual(a[1], 2)
        XCTAssertEqual(b[0], 3 - 0.1 * 0.8 / 1.8, accuracy: 1e-6)
        XCTAssertThrowsError(try accumulator.add(["a": .floats([1, 1], shape: [2])]))
    }

    func testAdamWDecaysWeightsSeparatelyFromAdamMomentsAcrossUpdates() throws {
        let weights = ["factor": NativeTensor.floats([2, -3], shape: [2])]
        let zero = ["factor": NativeTensor.floats([0, 0], shape: [2])]
        let config = NativeMaterialOptimizerConfiguration(weightDecay: 0.1, maxGradientNorm: 0)
        let first = try NativeMaterialOptimizer.apply(gradients: zero, weights: weights, state: [:], learningRate: 0.01, step: 1, configuration: config)
        let values = try first.weights["factor"]!.floatValues()
        XCTAssertEqual(values[0], 1.998, accuracy: 1e-6); XCTAssertEqual(values[1], -2.997, accuracy: 1e-6)
        XCTAssertEqual(try first.state["factor.m"]!.floatValues(), [0, 0])
        XCTAssertEqual(try first.state["factor.v"]!.floatValues(), [0, 0])
        let second = try NativeMaterialOptimizer.apply(gradients: zero, weights: first.weights, state: first.state, learningRate: 0.01, step: 2, configuration: config)
        XCTAssertEqual(try second.weights["factor"]!.floatValues()[0], 1.996002, accuracy: 1e-6)
        var coupled = config; coupled.algorithm = "adam"
        let adam = try NativeMaterialOptimizer.apply(gradients: zero, weights: weights, state: [:], learningRate: 0.01, step: 1, configuration: coupled)
        XCTAssertEqual(try adam.weights["factor"]!.floatValues()[0], 1.99, accuracy: 1e-6)
        XCTAssertGreaterThan(try adam.state["factor.v"]!.floatValues()[0], 0)
    }

    func testAdamMomentsAndBiasCorrectionMatchHandCalculatedSecondStep() throws {
        let configuration = NativeMaterialOptimizerConfiguration(beta1: 0.5, beta2: 0.5, epsilon: 1e-6, maxGradientNorm: 0)
        let first = try NativeMaterialOptimizer.apply(gradients: ["a": .floats([2], shape: [1])],
            weights: ["a": .floats([1], shape: [1])], state: [:], learningRate: 0.1, step: 1, configuration: configuration)
        let second = try NativeMaterialOptimizer.apply(gradients: ["a": .floats([4], shape: [1])],
            weights: first.weights, state: first.state, learningRate: 0.1, step: 2, configuration: configuration)
        XCTAssertEqual(try second.state["a.m"]!.floatValues()[0], 2.5)
        XCTAssertEqual(try second.state["a.v"]!.floatValues()[0], 9)
        let expected = Double(try first.weights["a"]!.floatValues()[0]) - 0.1 * (2.5 / 0.75) / (sqrt(9 / 0.75) + 1e-6)
        XCTAssertEqual(Double(try second.weights["a"]!.floatValues()[0]), expected, accuracy: 1e-6)
        XCTAssertThrowsError(try NativeMaterialOptimizer.apply(gradients: ["a": .floats([.infinity], shape: [1])],
            weights: first.weights, state: first.state, learningRate: 0.1, step: 2, configuration: configuration))
        XCTAssertThrowsError(try NativeMaterialOptimizer.apply(gradients: ["a": .floats([4], shape: [1])],
            weights: first.weights, state: ["a.m": first.state["a.m"]!], learningRate: 0.1, step: 2, configuration: configuration))
    }
}
