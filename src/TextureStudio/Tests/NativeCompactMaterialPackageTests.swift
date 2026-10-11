import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class NativeCompactMaterialPackageTests: XCTestCase {
    func testEveryCompactTargetExportsStandaloneFullWeightsInBothModes() throws {
        let fixture = try NativeMaterialPackageFixture(); defer { fixture.remove() }
        for target in ["height", "roughness", "normal"] { for developer in [false, true] {
            let model = try NativeCompactMaterialModel(target: target, width: 16, seed: 17)
            var configuration = try model.checkpointConfiguration(size: 256, step: 4, validation: nil)
            configuration["model_name"] = "Compact \(target)"
            let package = fixture.root.appendingPathComponent("\(target)-\(developer)")
            let information = try NativeMaterialTransfer.object(Data(NativeMaterialPackage.export(model: model,
                configuration: configuration, to: package, developer: developer).utf8))
            let (savedConfiguration, hashes) = try NativeMaterialPackage.verify(package)
            XCTAssertEqual(hashes.count, 6)
            XCTAssertNotNil(hashes["ModelLicenses/NAFNet_LICENSE"])
            XCTAssertNotNil(hashes["model.safetensors"])
            XCTAssertNil(hashes["adapter.safetensors"])
            XCTAssertNil(savedConfiguration["adapter_filename"])
            XCTAssertNil(savedConfiguration["base"])
            XCTAssertNil(savedConfiguration["layers"])
            XCTAssertEqual(savedConfiguration["schema"] as? String, NativeCompactMaterialModel.schema)
            XCTAssertEqual(savedConfiguration["checkpoint_filename"] as? String, "model.safetensors")
            XCTAssertEqual(savedConfiguration["full_checkpoint"] as? Bool, true)
            XCTAssertEqual(information["variant"] as? String, "full")
            XCTAssertEqual(information["supports_training_warm_start"] as? Bool, true)
            XCTAssertEqual(information["supports_studio_inference"] as? Bool, true)
            XCTAssertEqual(information["base_required"] as? Bool, false)
            XCTAssertNil(information["adapter_path"])
            XCTAssertEqual(information["output_channels"] as? Int, target == "normal" ? 3 : 1)
            let snapshot = try NativeSafetensors(contentsOf: package.appendingPathComponent("model.safetensors"))
            XCTAssertEqual(Set(snapshot.tensors.keys), Set(model.weights.keys))
            for (name, weight) in model.weights { XCTAssertEqual(try snapshot.tensorBytes(named: name), weight.bytes) }
            XCTAssertTrue(try String(contentsOf: package.appendingPathComponent("README.md"), encoding: .utf8).contains("every weight from scratch"))
        } }
    }

    func testRepackagingCompactCheckpointPreservesExactBytesWithoutItsBaseOrAdapter() async throws {
        let fixture = try NativeMaterialPackageFixture(); defer { fixture.remove() }
        let model = try NativeCompactMaterialModel(target: "normal", width: 16, seed: 19)
        let configuration = try model.checkpointConfiguration(size: 256, step: 7, validation: nil)
        let selected = fixture.root.appendingPathComponent("selected.safetensors")
        let bytes = try NativeSafetensors.encoded(tensors: model.weights,
            metadata: ["configuration": NativeMaterialTransfer.json(configuration), "provenance": "exact compact snapshot"])
        try bytes.write(to: selected)
        let snapshot = try NativeSafetensors(bytes: bytes)
        for developer in [false, true] {
            let output = fixture.root.appendingPathComponent("repacked-\(developer)")
            var arguments = ["--checkpoint", selected.path, "--expected-sha256", snapshot.sha256, "--output", output.path]
            if developer { arguments.append("--developer-mode") }
            let result = try NativeMaterialTransfer.object(Data(try await NativeMaterialPackage.run(arguments: arguments).utf8))
            XCTAssertEqual(result["source_checkpoint_sha256"] as? String, snapshot.sha256)
            XCTAssertEqual(try Data(contentsOf: output.appendingPathComponent("model.safetensors")), bytes)
            XCTAssertNoThrow(try NativeMaterialPackage.verify(output))
        }
        do {
            _ = try await NativeMaterialPackage.run(arguments: ["--checkpoint", selected.path, "--checkpoint-weight", "0.5",
                                                               "--output", fixture.root.appendingPathComponent("weighted").path])
            XCTFail("A complete scratch model must not be interpreted as a weighted LoRA adapter")
        } catch { XCTAssertTrue(error.localizedDescription.contains("standalone")) }
        XCTAssertThrowsError(try NativeMaterialPackage.combine([(snapshot, 1)]))
        XCTAssertThrowsError(try NativeMaterialCheckpoint.inspect(at: selected, expectedSHA256: String(repeating: "0", count: 64)))
        let other = try NativeCompactMaterialModel(target: "normal", width: 16, seed: 23)
        XCTAssertThrowsError(try NativeMaterialPackage.export(model: model,
            configuration: other.checkpointConfiguration(size: 256, step: 7, validation: nil),
            to: fixture.root.appendingPathComponent("wrong-initialization"), developer: false))
    }

    func testInspectionRejectsCompactArchitectureTargetChannelsAndTensorContractMismatch() throws {
        let fixture = try NativeMaterialPackageFixture(); defer { fixture.remove() }
        let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 17)
        let configuration = try model.checkpointConfiguration(size: 256, step: 4, validation: nil)
        let file = fixture.root.appendingPathComponent("invalid.safetensors")
        func rejects(_ configuration: [String: Any], tensors: [String: NativeTensor]) throws {
            try NativeSafetensors.write(tensors: tensors, metadata: ["configuration": NativeMaterialTransfer.json(configuration)], to: file)
            XCTAssertThrowsError(try NativeMaterialCheckpoint.inspect(at: file))
        }
        for (key, value) in [("architecture", "texture-studio-compact-normal-native-v1" as Any),
                             ("target", "normal"), ("output_channels", 3), ("input_channels", 1),
                             ("model_family", "compact-normal"), ("network_width", 32), ("from_scratch", false),
                             ("image_padding", true), ("step", -1)] {
            var changed = configuration; changed[key] = value
            try rejects(changed, tensors: model.weights)
        }
        let name = try XCTUnwrap(model.weights.keys.sorted().first), weight = try XCTUnwrap(model.weights[name])
        var missing = model.weights; missing.removeValue(forKey: name)
        try rejects(configuration, tensors: missing)
        var extra = model.weights; extra["unexpected.weight"] = .floats([0], shape: [1])
        try rejects(configuration, tensors: extra)
        var wrongShape = model.weights; wrongShape[name] = .floats([0], shape: [1])
        try rejects(configuration, tensors: wrongShape)
        var wrongPrecision = model.weights
        wrongPrecision[name] = NativeTensor(dtype: "F16", shape: weight.shape, bytes: Data(repeating: 0, count: weight.bytes.count / 2))
        try rejects(configuration, tensors: wrongPrecision)
        var finite = try NativeSafetensors.encoded(tensors: model.weights, metadata: ["configuration": NativeMaterialTransfer.json(configuration)])
        let payload = try NativeSafetensors(bytes: finite).payloadStart
        finite.replaceSubrange(payload..<payload + 4, with: [0, 0, 192, 127])
        try finite.write(to: file)
        XCTAssertThrowsError(try NativeMaterialCheckpoint.inspect(at: file), "Nonfinite learned weights cannot be imported")
    }

    func testInspectionRejectsAmbiguousConfigurationAndPackageSidecarMutation() throws {
        let fixture = try NativeMaterialPackageFixture(); defer { fixture.remove() }
        let model = try NativeCompactMaterialModel(target: "roughness", width: 16, seed: 17)
        let configuration = try model.checkpointConfiguration(size: 256, step: 4, validation: nil)
        let file = fixture.root.appendingPathComponent("ambiguous.safetensors")
        let json = try NativeMaterialTransfer.json(configuration)
        let duplicate = String(json.dropLast()) + ",\"target\":\"roughness\"}"
        try NativeSafetensors.write(tensors: model.weights, metadata: ["configuration": duplicate], to: file)
        XCTAssertThrowsError(try NativeMaterialCheckpoint.inspect(at: file))
        var fractional = configuration; fractional["step"] = 4.5
        try NativeSafetensors.write(tensors: model.weights, metadata: ["configuration": NativeMaterialTransfer.json(fractional)], to: file)
        XCTAssertThrowsError(try NativeMaterialCheckpoint.inspect(at: file))
        let package = fixture.root.appendingPathComponent("package")
        _ = try NativeMaterialPackage.export(model: model, configuration: configuration, to: package, developer: false)
        var sidecar = try NativeMaterialTransfer.object(package.appendingPathComponent("config.json"))
        sidecar["step"] = 5
        try NativeMaterialTransfer.writeJSON(sidecar, to: package.appendingPathComponent("config.json"))
        try fixture.refreshManifest(package)
        XCTAssertThrowsError(try NativeMaterialPackage.verify(package), "Even a checksummed sidecar must match the exact embedded training configuration")
    }

    func testCompactCapabilitiesAndVerifiedHubDownloadNeedNoPBRBase() async throws {
        let fixture = try NativeMaterialPackageFixture(); defer { fixture.remove() }
        for (family, targets) in [("compact-scalar", ["height", "roughness"]), ("compact-normal", ["normal"])] {
            let text = try await NativeMaterialCommands.run(arguments: ["capabilities", "--model-family", family], onEvent: { _ in }, control: .init())
            let result = try NativeMaterialTransfer.object(Data(text.utf8))
            XCTAssertEqual(result["base_required"] as? Bool, false)
            XCTAssertEqual(result["all_weights_trainable"] as? Bool, true)
            XCTAssertEqual(result["scope"] as? String, "full-model")
            XCTAssertEqual(result["targets"] as? [String], targets)
        }
        let model = try NativeCompactMaterialModel(target: "normal", width: 16, seed: 17)
        let package = fixture.root.appendingPathComponent("package")
        _ = try NativeMaterialPackage.export(model: model, configuration: model.checkpointConfiguration(size: 256, step: 3, validation: nil),
                                             to: package, developer: false)
        let files = try fixture.files(package), revision = String(repeating: "c", count: 40)
        let transfer = NativeMaterialTransfer(token: nil, fetch: { request, destination, maximum in
            let prefix = "/fixture/compact/resolve/\(revision)/"
            guard let url = request.url, url.scheme == "https", url.host == "huggingface.co", url.path.hasPrefix(prefix),
                  let bytes = files[String(url.path.dropFirst(prefix.count))], Int64(bytes.count) <= maximum else {
                throw StudioError("Unexpected compact model download fixture request")
            }
            try bytes.write(to: destination)
        }, catalogURL: fixture.root.appendingPathComponent("catalog.json"))
        let destination = fixture.root.appendingPathComponent("downloaded")
        let result = try NativeMaterialTransfer.object(Data(try await transfer.download(repository: "fixture/compact", revision: revision, to: destination).utf8))
        XCTAssertEqual(result["architecture"] as? String, "texture-studio-compact-normal-native-v1")
        XCTAssertEqual(result["base_required"] as? Bool, false)
        XCTAssertNoThrow(try NativeMaterialPackage.verify(destination))
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("model.safetensors")), files["model.safetensors"])
    }
}
