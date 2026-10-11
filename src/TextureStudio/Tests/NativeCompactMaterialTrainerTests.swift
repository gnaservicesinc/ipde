import Darwin
import Foundation
import Metal
import XCTest
@testable import TextureStudio

final class NativeCompactMaterialTrainerTests: XCTestCase {
    func testOptionsChooseCompatibleFamiliesAndKeepLegacyDefaults() throws {
        let arguments = ["train", "--dataset", "/unused", "--output", "/new"]
        let legacy = try NativeMaterialTrainer.Options(arguments)
        XCTAssertEqual(legacy.modelFamily, .pbrnxt); XCTAssertEqual(legacy.scope, "final-map")
        let scalar = try NativeMaterialTrainer.Options(arguments + ["--model-family", "compact-scalar"])
        XCTAssertEqual(scalar.size, 512); XCTAssertEqual(scalar.compactWidth, 32)
        XCTAssertEqual(scalar.scope, "full-model"); XCTAssertEqual(scalar.learningRate, 0.001)
        XCTAssertThrowsError(try NativeMaterialTrainer.Options(arguments + ["--model-family", "compact-scalar", "--target", "normal"]))
        XCTAssertThrowsError(try NativeMaterialTrainer.Options(arguments + ["--model-family", "compact-normal"]))
        XCTAssertThrowsError(try NativeMaterialTrainer.Options(arguments + ["--model-family", "compact-scalar", "--scope", "final-map"]))
        XCTAssertThrowsError(try NativeMaterialTrainer.Options(arguments + ["--model-family", "compact-scalar", "--compact-width", "24"]))
    }

    func testCheckpointOptionsPinWeightDigestAndRejectLegacyFamilyConfusion() throws {
        let fixture = try NativeMaterialPackageFixture(); defer { fixture.remove() }
        let checkpoint = fixture.root.appendingPathComponent("adapter.safetensors")
        let original = try fixture.adapter().bytes
        try original.write(to: checkpoint)
        let arguments = ["infer", "--baseline", "--checkpoint", checkpoint.path,
            "--image", "/unused.png", "--output", fixture.root.appendingPathComponent("output").path]
        let options = try NativeMaterialTrainer.Options(arguments)
        XCTAssertEqual(options.expectedSHA256, NativeMaterialTrainer.checksum(original))
        let confused = try NativeMaterialTrainer.Options(arguments + ["--model-family", "compact-scalar", "--scope", "full-model"])
        XCTAssertThrowsError(try NativeTrainableMaterialModel.load(confused, training: false))
        try fixture.adapter(a: [9, 10]).bytes.write(to: checkpoint)
        XCTAssertThrowsError(try NativeMaterialCheckpoint.inspect(at: checkpoint, expectedSHA256: options.expectedSHA256))
    }

    func testAllFamiliesTrainExportAndInferWithoutDownloadedBase() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        for target in ["height", "roughness", "normal"] {
            let dataset = root.appendingPathComponent("dataset-" + target)
            try fixture(dataset, target: target)
            let output = root.appendingPathComponent("trained-" + target)
            let family = target == "normal" ? "compact-normal" : "compact-scalar"
            let arguments = ["train", "--dataset", dataset.path, "--output", output.path,
                "--model-family", family, "--target", target, "--size", "256", "--compact-width", "16",
                "--updates-per-map", "1", "--seed", "31", "--model-directory", root.appendingPathComponent("missing-base").path]
            let recorder = Recorder()
            let result = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: arguments,
                onEvent: { recorder.record($0) }).utf8))
            XCTAssertEqual(result["completed_updates"] as? Int, 1)
            XCTAssertEqual(result["model_family"] as? String, family)
            XCTAssertEqual(result["output_channels"] as? Int, target == "normal" ? 3 : 1)
            let checkpoint = URL(fileURLWithPath: try XCTUnwrap(result["checkpoint_path"] as? String))
            XCTAssertEqual(checkpoint.lastPathComponent, "model.safetensors")
            let (configuration, inventory) = try NativeMaterialPackage.verify(checkpoint.deletingLastPathComponent())
            XCTAssertEqual(configuration["target"] as? String, target)
            XCTAssertNil(configuration["base"]); XCTAssertNil(inventory["adapter.safetensors"])
            let initial = try NativeCompactMaterialModel(target: target, width: 16, seed: 31)
            let trained = try NativeSafetensors(contentsOf: checkpoint).nativeTensors()
            XCTAssertNotEqual(initial.weights["input.weight"]!.bytes, trained["input.weight"]!.bytes)
            XCTAssertNotEqual(initial.weights["output.weight"]!.bytes, trained["output.weight"]!.bytes)
            let hash = try NativeMaterialTransfer.hash(checkpoint)
            let inferred = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: ["infer",
                "--checkpoint", checkpoint.path, "--expected-sha256", hash,
                "--image", dataset.appendingPathComponent("samples/train-0/diffuse.png").path,
                "--output", root.appendingPathComponent("inferred-" + target).path]).utf8))
            XCTAssertEqual(inferred["native_dimensions"] as? [Int], [256, 256])
            XCTAssertEqual(inferred["output_channels"] as? Int, target == "normal" ? 3 : 1)
            XCTAssertEqual(inferred["model_family"] as? String, family)
            XCTAssertEqual(inferred["checkpoint_sha256"] as? String, hash)
            if target == "height" {
                let baseline = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: ["infer", "--baseline",
                    "--checkpoint", checkpoint.path, "--image", dataset.appendingPathComponent("samples/train-0/diffuse.png").path,
                    "--output", root.appendingPathComponent("untrained").path]).utf8))
                XCTAssertEqual(baseline["untrained_initialization"] as? Bool, true)
                XCTAssertEqual(baseline["checkpoint_step"] as? Int, 0)
                for override in [["--target", "roughness"], ["--compact-width", "32"], ["--seed", "32"]] {
                    let changed = try NativeMaterialTrainer.Options(["infer", "--baseline", "--checkpoint", checkpoint.path,
                        "--image", dataset.appendingPathComponent("samples/train-0/diffuse.png").path,
                        "--output", root.appendingPathComponent("invalid-baseline").path] + override)
                    XCTAssertThrowsError(try NativeTrainableMaterialModel.load(changed, training: false))
                }
                let resumed = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: ["train",
                    "--dataset", dataset.path, "--checkpoint", checkpoint.path, "--expected-sha256", hash,
                    "--size", "256", "--updates-per-map", "1", "--output", root.appendingPathComponent("resumed").path]).utf8))
                XCTAssertEqual(resumed["completed_updates"] as? Int, 1)
                let inspected = try NativeMaterialCheckpoint.inspect(at: URL(fileURLWithPath: try XCTUnwrap(resumed["checkpoint_path"] as? String)))
                let info = try NativeMaterialTransfer.object(Data(inspected.utf8))
                XCTAssertEqual(info["step"] as? Int, 2)
                XCTAssertEqual(info["network_width"] as? Int, 16)
                XCTAssertEqual(info["initialization_seed"] as? UInt64, 31)
            }
            XCTAssertEqual(recorder.events.filter { $0["event"] as? String == "update" }.count, 1)
        }
    }

    func testFinalValidationFailureStillPublishesCompactWeights() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset"), output = root.appendingPathComponent("trained")
        try fixture(dataset, target: "height", validation: true)
        let recorder = Recorder()
        let validationMap = dataset.appendingPathComponent("samples/validation/height.png")
        let result = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: ["train",
            "--dataset", dataset.path, "--output", output.path, "--model-family", "compact-scalar", "--size", "256",
            "--compact-width", "16", "--updates-per-map", "1"], onEvent: { line in
                recorder.record(line)
                if let event = try? NativeMaterialTransfer.object(Data(line.utf8)), event["event"] as? String == "update" {
                    try? Data("corrupt validation sample".utf8).write(to: validationMap)
                }
            }).utf8))
        XCTAssertEqual(result["completed_updates"] as? Int, 1)
        XCTAssertEqual(result["skipped_sample_count"] as? Int, 1)
        let checkpoint = URL(fileURLWithPath: try XCTUnwrap(result["checkpoint_path"] as? String))
        XCTAssertNoThrow(try NativeMaterialPackage.verify(checkpoint.deletingLastPathComponent()))
        XCTAssertTrue(recorder.events.contains { $0["event"] as? String == "sample_skipped" && $0["phase"] as? String == "validation" })
    }

    /// Opt-in short throughput probe. It deliberately does not assess trained quality.
    func testCompact512TrainingBenchmark() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["TEXTURE_STUDIO_COMPACT_BENCHMARK"] == "1" else { throw XCTSkip("Opt-in 512 compact training throughput probe.") }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset")
        try fixture(dataset, target: "height", size: 512, trainingCount: 3)
        let report = URL(fileURLWithPath: environment["TEXTURE_STUDIO_COMPACT_BENCHMARK_OUTPUT"] ?? "/opt/ipde/ipde/out/compact-native-benchmark.json")
        let recorder = Recorder(), start = ProcessInfo.processInfo.systemUptime
        let result = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: ["train",
            "--dataset", dataset.path, "--output", root.appendingPathComponent("trained").path,
            "--model-family", "compact-scalar", "--size", "512", "--updates-per-map", "2"],
            onEvent: { recorder.record($0) }).utf8))
        let updates = recorder.events.filter { $0["event"] as? String == "update" }
        XCTAssertEqual(result["completed_updates"] as? Int, 6)
        XCTAssertEqual(updates.count, 6)
        XCTAssertTrue(updates.allSatisfy { ($0["total"] as? Double)?.isFinite == true })
        var resources = rusage(); getrusage(RUSAGE_SELF, &resources)
        let receipt: [String: Any] = ["model_family": "compact-scalar", "size": 512, "network_width": 32,
            "parameters": 999_937, "fixture": "three distinct procedural native RGB/16-bit scalar pairs; two epochs",
            "device": device.name, "total_wall_seconds": ProcessInfo.processInfo.systemUptime - start,
            "peak_resident_bytes": resources.ru_maxrss, "metal_allocated_bytes_at_completion": device.currentAllocatedSize,
            "updates": updates, "completed_updates": 6,
            "quality_qualification": false, "cold_step_includes_graph_compilation": true]
        try FileManager.default.createDirectory(at: report.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys]).write(to: report)
    }

    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var recorded: [[String: Any]] = []
        func record(_ text: String) {
            guard let event = try? NativeMaterialTransfer.object(Data(text.utf8)) else { return }
            lock.withLock { recorded.append(event) }
        }
        var events: [[String: Any]] { lock.withLock { recorded } }
    }

    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("compact-training-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true); return root
    }

    private func fixture(_ root: URL, target: String, size: Int = 256, trainingCount: Int = 1, validation: Bool = false) throws {
        var entries: [[String: Any]] = []
        let identities = (0..<trainingCount).map { ("train-\($0)", "train", $0) } + (validation ? [("validation", "validation", trainingCount)] : [])
        for (id, split, variant) in identities {
            let folder = root.appendingPathComponent("samples/" + id)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let input = try NativePNG(header: .init(width: size, height: size, bits: 8, channels: 3, color: 2, interlace: 0),
                pixels: Data((0..<size * size * 3).map { UInt8(($0 * 7 + $0 / size + variant * 31) % 251) }), colorChunks: []).encoded()
            let normal = target == "normal", channels = normal ? 3 : 1, bits = normal ? 8 : 16
            let reference = try NativePNG(header: .init(width: size, height: size, bits: bits, channels: channels, color: normal ? 2 : 0, interlace: 0),
                pixels: Data((0..<size * size * channels * (bits / 8)).map { UInt8(($0 * 3 + variant * 13) % 251) }), colorChunks: []).encoded()
            try input.write(to: folder.appendingPathComponent("diffuse.png")); try reference.write(to: folder.appendingPathComponent(target + ".png"))
            entries.append(["sample_id": id, "material_id": id, "status": "approved", "split": split, "path": "samples/" + id])
            let sample: [String: Any] = ["sample_id": id, "material_id": id, "status": "approved", "split": split,
                "sample_pixel_dimensions": [size, size], "maps": ["input": "diffuse.png", target: target + ".png"],
                "map_metadata": ["input": ["filename": "diffuse.png", "encoding": "srgb", "sample_sha256": NativeMaterialTrainer.checksum(input)],
                    target: ["filename": target + ".png", "sample_sha256": NativeMaterialTrainer.checksum(reference)]]]
            try JSONSerialization.data(withJSONObject: sample).write(to: folder.appendingPathComponent("sample.json"))
        }
        try JSONSerialization.data(withJSONObject: ["schema_version": 2, "samples": entries,
            "validation": ["enabled": validation, "quick_count": 1]]).write(to: root.appendingPathComponent("dataset.json"))
    }
}
