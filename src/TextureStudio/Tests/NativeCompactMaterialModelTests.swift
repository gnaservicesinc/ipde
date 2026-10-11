import Foundation
import Metal
import XCTest
@testable import TextureStudio

final class NativeCompactMaterialModelTests: XCTestCase {
    func testScratchInitializationAndOutputFamiliesHaveDeterministicSeparateIdentities() throws {
        let height = try NativeCompactMaterialModel(target: "height", width: 16, seed: 19)
        let repeated = try NativeCompactMaterialModel(target: "height", width: 16, seed: 19)
        let otherSeed = try NativeCompactMaterialModel(target: "height", width: 16, seed: 20)
        let normal = try NativeCompactMaterialModel(target: "normal", width: 16, seed: 19)
        let roughness = try NativeCompactMaterialModel(target: "roughness", width: 16, seed: 19)
        XCTAssertEqual(height.baseSHA256, repeated.baseSHA256)
        XCTAssertNotEqual(height.baseSHA256, otherSeed.baseSHA256)
        XCTAssertNotEqual(height.baseSHA256, normal.baseSHA256)
        XCTAssertEqual(height.architectureID, NativeCompactMaterialModel.scalarArchitecture)
        XCTAssertEqual(roughness.architectureID, height.architectureID)
        XCTAssertEqual(normal.architectureID, NativeCompactMaterialModel.normalArchitecture)
        XCTAssertEqual(height.outputChannels, 1); XCTAssertEqual(roughness.outputChannels, 1); XCTAssertEqual(normal.outputChannels, 3)
        XCTAssertEqual(height.weights["output.weight"]!.shape, [1, 16, 3, 3])
        XCTAssertEqual(normal.weights["output.weight"]!.shape, [3, 16, 3, 3])
        XCTAssertEqual(Set(height.weights.keys), Set(repeated.weights.keys))
        for name in height.weights.keys { XCTAssertEqual(height.weights[name]!.bytes, repeated.weights[name]!.bytes) }
        XCTAssertNil(height.configuration["base"]); XCTAssertNil(height.configuration["layers"])
        XCTAssertEqual(height.configuration["from_scratch"] as? Bool, true)
        XCTAssertThrowsError(try NativeCompactMaterialModel(target: "albedo"))
        XCTAssertThrowsError(try NativeCompactMaterialModel(target: "height", width: 96))
    }

    func testFullCheckpointRejectsWrongFamilyMissingWeightAndMalformedValues() throws {
        let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 17)
        let configuration = try model.checkpointConfiguration(size: 256, step: 1)
        func snapshot(_ weights: [String: NativeTensor]) throws -> NativeSafetensors {
            let metadata = String(decoding: try JSONSerialization.data(withJSONObject: configuration, options: [.sortedKeys]), as: UTF8.self)
            return try NativeSafetensors(bytes: NativeSafetensors.encoded(tensors: weights, metadata: ["configuration": metadata]))
        }
        let original = try snapshot(model.weights)
        XCTAssertNoThrow(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: configuration))
        for field in ["step", "training_size", "image_padding", "image_resizing", "scope", "native_runtime", "initial_weights_sha256", "target", "network_width"] {
            var missingField = configuration; missingField.removeValue(forKey: field)
            XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: missingField), field)
        }
        for (field, invalid) in [("step", -1 as Any), ("training_size", 255 as Any), ("image_padding", true as Any),
                                 ("image_resizing", 0 as Any), ("native_runtime", "unknown" as Any), ("scope", "final-map" as Any)] {
            var wrongField = configuration; wrongField[field] = invalid
            XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: wrongField), field)
        }
        var wrongFamily = configuration
        wrongFamily["architecture"] = NativeCompactMaterialModel.normalArchitecture
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: wrongFamily))
        var wrongTarget = configuration; wrongTarget["target"] = "normal"
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: wrongTarget))
        var wrongSeed = configuration; wrongSeed["initialization_seed"] = UInt64(18)
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: wrongSeed))
        var noninteger = configuration; noninteger["network_width"] = 16.5
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: noninteger))
        var missing = model.weights; missing.removeValue(forKey: "input.weight")
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(snapshot(missing), configuration: configuration))
        var malformed = model.weights
        malformed["output.weight"] = .floats([Float](repeating: 0, count: 16 * 9), shape: [16, 1, 3, 3])
        XCTAssertThrowsError(try model.updateWeights(malformed))
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(snapshot(malformed), configuration: configuration))
        var bytes = original.bytes, infinity = Float.infinity.bitPattern.littleEndian
        let offset = original.payloadStart + original.tensors["output.bias"]!.dataOffsets[0]
        withUnsafeBytes(of: &infinity) { bytes.replaceSubrange(offset..<(offset + 4), with: $0) }
        let nonfinite = try NativeSafetensors(bytes: bytes)
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(nonfinite, configuration: configuration)) { error in
            XCTAssertFalse(error is NativeMaterialSampleError, "Malformed stored model weights are fatal model failures.")
        }
    }

    func testNativeScalarAndNormalPredictionPreserveInputGridAndHeadDimensions() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let rgb = Self.rgb(width: 128, height: 64)
            for target in ["height", "roughness", "normal"] {
                let model = try NativeCompactMaterialModel(target: target, width: 16)
                let result = try model.predict(rgb: rgb, width: 128, height: 64, target: target)
                XCTAssertEqual(result.width, 128); XCTAssertEqual(result.height, 64)
                XCTAssertEqual(result.channels, target == "normal" ? 3 : 1)
                XCTAssertEqual(result.values.count, 128 * 64 * result.channels)
                XCTAssertTrue(result.values.allSatisfy(\.isFinite))
                let program = try model.program(width: 128, height: 64, target: target)
                XCTAssertEqual(program.executionStatistics.compilations, 1)
                let repeated = try model.predict(rgb: rgb, width: 128, height: 64, target: target)
                XCTAssertEqual(program.executionStatistics.compilations, 1, "Prediction reuses the executable and never builds autodiff.")
                XCTAssertEqual(result.values.map(\.bitPattern), repeated.values.map(\.bitPattern))
                XCTAssertThrowsError(try model.predict(rgb: rgb, width: 128, height: 64, target: target == "normal" ? "height" : "normal"))
                XCTAssertThrowsError(try model.predict(rgb: rgb, width: 127, height: 64, target: target))
            }
        }.value
    }

    func testEveryWeightHasFiniteAutodiffAndAdamUpdatesEarlyMiddleAndOutputWeights() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 23)
            let before = model.weights, rgb = Self.rgb(width: 64, height: 64)
            let reference = (0..<64 * 64).map { Float(($0 * 11) % 251) / 251 }
            let program = try model.program(width: 64, height: 64, target: "height")
            let evaluated = try program.execute(rgb: rgb, weights: before, reference: reference, gradientsOnly: true)
            XCTAssertEqual(Set(evaluated.gradients.keys), Set(before.keys))
            XCTAssertTrue(evaluated.updated.isEmpty); XCTAssertTrue(evaluated.optimizerState.isEmpty)
            XCTAssertTrue(try XCTUnwrap(evaluated.loss).isFinite)
            for name in before.keys {
                XCTAssertEqual(before[name]!.bytes, model.weights[name]!.bytes, "Gradient evaluation cannot mutate weights.")
                XCTAssertEqual(before[name]!.shape, evaluated.gradients[name]!.shape)
                XCTAssertTrue(try evaluated.gradients[name]!.floatValues().allSatisfy(\.isFinite))
            }
            // Compare one graph gradient with an independently perturbed loss.
            let epsilon: Float = 1e-3
            func loss(shift: Float) throws -> Float {
                var shifted = before
                let bias = try before["output.bias"]!.floatValues()[0]
                shifted["output.bias"] = .floats([bias + shift], shape: [1, 1, 1, 1])
                return try XCTUnwrap(program.execute(rgb: rgb, weights: shifted, reference: reference).loss)
            }
            let finiteDifference = try (loss(shift: epsilon) - loss(shift: -epsilon)) / (2 * epsilon)
            XCTAssertEqual(try evaluated.gradients["output.bias"]!.floatValues()[0], finiteDifference, accuracy: 0.005)
            let update = try NativeMaterialOptimizer.apply(gradients: evaluated.gradients, weights: before, state: [:],
                learningRate: 1e-3, step: 1, configuration: .init())
            XCTAssertEqual(Set(update.weights.keys), Set(before.keys))
            XCTAssertEqual(update.state.count, before.count * 2)
            for name in ["input.weight", "encoder.0.expand.weight", "middle.expand.weight", "up.2.weight", "output.weight", "output.bias"] {
                XCTAssertNotEqual(before[name]!.bytes, update.weights[name]!.bytes, "Full-backbone training must update \(name).")
            }
            try model.updateWeights(update.weights)
            let second = try program.execute(rgb: rgb, weights: model.weights, reference: reference, gradientsOnly: true)
            XCTAssertTrue(try XCTUnwrap(second.loss).isFinite)
            // Validation and training retain their own executables.
            XCTAssertEqual(program.executionStatistics.compilations, 2)
        }.value
    }

    func testSavedFullWeightsReloadPredictExactlyAndResumeWithoutPBRnxt() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let model = try NativeCompactMaterialModel(target: "normal", width: 16, seed: 29)
            let rgb = Self.rgb(width: 64, height: 64), before = model.weights
            let reference = (0..<3 * 64 * 64).map { Float(($0 * 7) % 251) / 251 }
            let program = try model.program(width: 64, height: 64, target: "normal")
            let derivative = try program.execute(rgb: rgb, weights: before, reference: reference, gradientsOnly: true)
            let update = try NativeMaterialOptimizer.apply(gradients: derivative.gradients, weights: before, state: [:],
                learningRate: 1e-4, step: 1, configuration: .init())
            try model.updateWeights(update.weights)
            let prediction = try model.predict(rgb: rgb, width: 64, height: 64, target: "normal")
            let checkpoint = directory.appendingPathComponent("model.safetensors")
            let configuration = try model.checkpointConfiguration(size: 256, step: 1)
            let metadata = String(decoding: try JSONSerialization.data(withJSONObject: configuration, options: [.sortedKeys]), as: UTF8.self)
            try NativeSafetensors.write(tensors: model.weights, metadata: ["configuration": metadata], to: checkpoint)
            let hash = try NativeMaterialTransfer.hash(checkpoint)
            let reloaded = try NativeCompactMaterialModel.load(checkpointURL: checkpoint, expectedSHA256: hash,
                target: "normal", width: 16, seed: 99)
            XCTAssertEqual(reloaded.networkWidth, 16); XCTAssertEqual(reloaded.initializationSeed, 29)
            XCTAssertEqual(reloaded.configuration["step"] as? Int, 1)
            XCTAssertEqual(reloaded.baseSHA256, model.baseSHA256)
            for name in model.weights.keys { XCTAssertEqual(reloaded.weights[name]!.bytes, model.weights[name]!.bytes) }
            let repeated = try reloaded.predict(rgb: rgb, width: 64, height: 64, target: "normal")
            XCTAssertEqual(prediction.values.map(\.bitPattern), repeated.values.map(\.bitPattern))
            XCTAssertThrowsError(try NativeCompactMaterialModel.load(checkpointURL: checkpoint,
                expectedSHA256: String(repeating: "0", count: 64), target: "normal"))
            XCTAssertThrowsError(try NativeCompactMaterialModel.load(checkpointURL: checkpoint, target: "height"))
            XCTAssertThrowsError(try NativeCompactMaterialModel.load(checkpointURL: checkpoint, target: "normal", width: 32))
        }.value
    }

    func testFiniteSampleOverflowSkipsWithoutCorruptingFollowingEvaluation() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let model = try NativeCompactMaterialModel(target: "height", width: 16)
            let before = model.weights, rgb = Self.rgb(width: 64, height: 64)
            let program = try model.program(width: 64, height: 64, target: "height")
            let extreme = (0..<64 * 64).map { $0 % 2 == 0 ? Float.greatestFiniteMagnitude : -Float.greatestFiniteMagnitude }
            XCTAssertThrowsError(try program.execute(rgb: rgb, weights: before, reference: extreme, gradientsOnly: true)) { error in
                XCTAssertTrue(error is NativeMaterialSampleError)
            }
            for name in before.keys { XCTAssertEqual(before[name]!.bytes, model.weights[name]!.bytes) }
            let next = try program.execute(rgb: rgb, weights: before,
                reference: [Float](repeating: 0.25, count: 64 * 64), gradientsOnly: true)
            XCTAssertTrue(try XCTUnwrap(next.loss).isFinite)
            XCTAssertEqual(Set(next.gradients.keys), Set(before.keys))
        }.value
    }

    private static func rgb(width: Int, height: Int) -> [Float] {
        (0..<width * height * 3).map { Float(($0 * 37 + $0 / width) % 251) / 251 }
    }
}
