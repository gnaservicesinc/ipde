import CryptoKit
import Foundation

/// Native model packages contain numeric weights, configuration and notices.
/// They do not carry executable architecture sources or an interpreter.
enum NativeMaterialPackage {
    static let licenses = ["TextureStudio_LICENSE", "PBRnxt_LICENSE", "SCUNet_LICENSE", "SwinTransformer_LICENSE", "ESRGANplus_LICENSE"]
    static let compactLicenses = ["TextureStudio_LICENSE", "NAFNet_LICENSE"]
    static func allowedFile(_ name: String) -> Bool {
        ["adapter.safetensors", "model.safetensors", "config.json", "README.md", "LICENSE"].contains(name)
            || (licenses + compactLicenses).contains(where: { name == "ModelLicenses/" + $0 })
    }

    static func verify(_ directory: URL) throws -> ([String: Any], [String: Any]) {
        try Task.checkCancellation()
        let fm = FileManager.default
        let root = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard root.isDirectory == true, root.isSymbolicLink != true else { throw StudioError("Choose a real model package directory.") }
        let directory = directory.standardizedFileURL.resolvingSymlinksInPath()
        let manifest = directory.appendingPathComponent("SHA256SUMS.json")
        let manifestValues = try manifest.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard manifestValues.isRegularFile == true, manifestValues.isSymbolicLink != true,
              (manifestValues.fileSize ?? Int.max) <= 1_048_576 else { throw StudioError("Invalid native package manifest.") }
        let hashes = try NativeMaterialTransfer.object(manifest)
        let required = Set(["config.json", "README.md", "LICENSE"])
        guard required.isSubset(of: Set(hashes.keys)),
              hashes.count <= 16 else { throw StudioError("The native model package is incomplete.") }
        for (name, value) in hashes {
            guard allowedFile(name), let digest = value as? String, NativeMaterialTransfer.isDigest(digest) else {
                throw StudioError("The model package contains an unsupported file or digest.")
            }
            let file = directory.appendingPathComponent(name)
            let attributes = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
            guard attributes.isRegularFile == true, attributes.isSymbolicLink != true,
                  Int64(attributes.fileSize ?? Int.max) <= (name.hasSuffix(".safetensors") ? 4_294_967_296 : 4_194_304),
                  try NativeMaterialTransfer.hash(file) == digest else { throw StudioError("A model package file failed SHA-256 verification: \(name)") }
        }
        var enumerationError: Error?
        guard let walker = fm.enumerator(at: directory, includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey], errorHandler: { _, error in enumerationError = error; return false }) else {
            throw StudioError("The model package could not be enumerated.")
        }
        for case let file as URL in walker {
            try Task.checkCancellation()
            let values = try file.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
            guard values.isSymbolicLink != true else { throw StudioError("Symbolic links are not accepted in model packages.") }
            let path = file.standardizedFileURL.resolvingSymlinksInPath().path
            guard path.hasPrefix(directory.path + "/") else { throw StudioError("Model package enumeration escaped its directory.") }
            let relative = String(path.dropFirst(directory.path.count + 1))
            if values.isDirectory == true {
                guard relative == "ModelLicenses" else { throw StudioError("Unsupported directory in model package.") }
            } else {
                guard values.isRegularFile == true,
                      relative != ".texture-studio-native-model.json" || (values.fileSize ?? Int.max) <= 1_048_576 else {
                    throw StudioError("Unsupported file type or model ownership marker in package.")
                }
                guard hashes[relative] != nil || ["SHA256SUMS.json", ".texture-studio-native-model.json"].contains(relative) else {
                    throw StudioError("Unlisted file in model package: \(relative)")
                }
            }
        }
        if let enumerationError { throw enumerationError }
        let config = try NativeMaterialTransfer.object(directory.appendingPathComponent("config.json"))
        let compact = config["schema"] as? String == NativeCompactMaterialModel.schema
        let requiredNotices = compact ? compactLicenses : licenses
        guard Set(requiredNotices.map { "ModelLicenses/" + $0 }).isSubset(of: Set(hashes.keys)),
              compact || hashes["adapter.safetensors"] != nil else { throw StudioError("The native model package is incomplete.") }
        guard let filename = config["checkpoint_filename"] as? String,
              ["adapter.safetensors", "model.safetensors"].contains(filename), hashes[filename] != nil else {
            throw StudioError("The model package does not identify its checkpoint.")
        }
        var recorded: [String: [String: Any]] = [:]
        var snapshots: [String: NativeSafetensors] = [:]
        for name in ["adapter.safetensors", "model.safetensors"] where hashes[name] != nil {
            let file = directory.appendingPathComponent(name)
            _ = try NativeMaterialCheckpoint.inspect(at: file, expectedSHA256: hashes[name] as? String)
            let snapshot = try NativeSafetensors(contentsOf: file, expectedSHA256: hashes[name] as? String)
            snapshots[name] = snapshot
            guard let metadata = snapshot.metadata["configuration"] else { throw StudioError("Missing checkpoint configuration.") }
            recorded[name] = try NativeMaterialTransfer.object(Data(metadata.utf8))
        }
        let full = hashes["model.safetensors"] != nil
        if compact {
            guard filename == "model.safetensors", full, hashes["adapter.safetensors"] == nil,
                  config["adapter_filename"] == nil,
                  config["full_checkpoint"] as? Bool == true,
                  config["optimizer_included"] as? Bool == false,
                  config["source_images_included"] as? Bool == false else {
                throw StudioError("A compact package must contain a complete standalone model without a LoRA adapter.")
            }
            var embedded = config
            for key in ["checkpoint_filename", "full_checkpoint", "optimizer_included", "source_images_included"] { embedded.removeValue(forKey: key) }
            guard let selected = recorded[filename], NSDictionary(dictionary: embedded).isEqual(to: selected) else {
                throw StudioError("Package configuration differs from its recorded weights.")
            }
            return (config, hashes)
        }
        guard filename == (full ? "model.safetensors" : "adapter.safetensors"),
              config["adapter_filename"] as? String == "adapter.safetensors",
              config["full_checkpoint"] as? Bool == full,
              config["optimizer_included"] as? Bool == false,
              config["source_images_included"] as? Bool == false else { throw StudioError("Invalid native package inventory configuration.") }
        var embedded = config
        for key in ["checkpoint_filename", "adapter_filename", "full_checkpoint", "optimizer_included", "source_images_included"] { embedded.removeValue(forKey: key) }
        guard let selected = recorded[filename], NSDictionary(dictionary: embedded).isEqual(to: selected) else {
            throw StudioError("Package configuration differs from its recorded weights.")
        }
        guard recorded["adapter.safetensors"]?["schema"] as? String == "texture-studio-material-lora-v1" else { throw StudioError("The package adapter has the wrong checkpoint schema.") }
        if full {
            guard var fullConfig = recorded["model.safetensors"], let adapterConfig = recorded["adapter.safetensors"],
                  fullConfig["schema"] as? String == "texture-studio-material-checkpoint-v1",
                  fullConfig["fused_adapter_sha256"] as? String == hashes["adapter.safetensors"] as? String else {
                throw StudioError("The complete checkpoint differs from its saved adapter.")
            }
            fullConfig["schema"] = "texture-studio-material-lora-v1"
            fullConfig.removeValue(forKey: "fused_adapter_sha256")
            guard NSDictionary(dictionary: fullConfig).isEqual(to: adapterConfig) else { throw StudioError("The complete checkpoint and adapter record different training configuration.") }
            let layers = try JSONDecoder().decode([String: NativeMaterialModel.AdapterLayer].self, from: JSONSerialization.data(withJSONObject: adapterConfig["layers"]!))
            let snapshot = snapshots["model.safetensors"]!
            guard layers.allSatisfy({ snapshot.tensors[$0.key + ".weight"]?.shape == $0.value.weightShape && snapshot.tensors[$0.key + ".weight"]?.dtype == "F32" }) else {
                throw StudioError("The complete checkpoint is missing its adapter's recorded native layers.")
            }
        }
        return (config, hashes)
    }

    static func export(model: NativeMaterialModel, configuration: [String: Any], to output: URL, developer: Bool) throws -> String {
        var configuration = configuration
        guard (configuration["base"] as? [String: Any])?["sha256"] as? String == model.baseSHA256 else {
            throw StudioError("Export configuration differs from the model's exact recorded base weights.")
        }
        configuration["schema"] = "texture-studio-material-lora-v1"
        configuration["layers"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(model.layers))
        let adapter = try NativeSafetensors.encoded(tensors: model.adapterWeights, metadata: metadata(configuration))
        var files = ["adapter.safetensors": adapter]
        var packageConfig = configuration
        if developer {
            packageConfig["schema"] = "texture-studio-material-checkpoint-v1"
            packageConfig["fused_adapter_sha256"] = hash(adapter)
            files["model.safetensors"] = try NativeSafetensors.encoded(tensors: model.fusedWeights(), metadata: metadata(packageConfig))
        }
        return try publish(files: files, configuration: packageConfig, output: output)
    }

    /// Compact training updates every weight and exports a complete standalone
    /// model in every mode. It has no downloaded base or separately fused adapter.
    static func export(model: NativeCompactMaterialModel, configuration: [String: Any], to output: URL, developer: Bool) throws -> String {
        guard configuration["target"] as? String == model.target,
              configuration["architecture"] as? String == model.architectureID,
              configuration["network_width"] as? Int == model.networkWidth,
              configuration["initialization_seed"] as? UInt64 == model.initializationSeed,
              configuration["initial_weights_sha256"] as? String == model.baseSHA256 else {
            throw StudioError("Export configuration differs from the compact model's exact architecture or initialization identity.")
        }
        let bytes = try NativeSafetensors.encoded(tensors: model.weights, metadata: metadata(configuration))
        let snapshot = try NativeSafetensors(bytes: bytes)
        try NativeCompactMaterialModel.validateCheckpoint(snapshot, configuration: configuration)
        return try publish(files: ["model.safetensors": bytes], configuration: configuration, output: output)
    }

    static func run(arguments: [String]) async throws -> String {
        let job = Task.detached(priority: .userInitiated) { try package(arguments: arguments) }
        return try await withTaskCancellationHandler { try await job.value } onCancel: { job.cancel() }
    }

    private static func package(arguments: [String]) throws -> String {
        func value(_ key: String) -> String? { arguments.firstIndex(of: key).flatMap { arguments.indices.contains($0 + 1) ? arguments[$0 + 1] : nil } }
        guard let path = value("--checkpoint"), let output = value("--output") else { throw StudioError("Select a checkpoint and a new export folder.") }
        let source = NativeMaterialModel.resolvedCheckpoint(URL(fileURLWithPath: path))
        let selected = try NativeSafetensors(contentsOf: source, expectedSHA256: value("--expected-sha256"))
        _ = try NativeMaterialCheckpoint.inspect(at: source, expectedSHA256: selected.sha256)
        var configuration = try adapterConfiguration(selected)
        let full = configuration["schema"] as? String == "texture-studio-material-checkpoint-v1"
        let extras = arguments.indices.filter { arguments[$0] == "--adapter" }.map { index -> String in
            arguments.indices.contains(index + 1) ? arguments[index + 1] : ""
        }
        guard let weight = Float(value("--checkpoint-weight") ?? "1"), weight.isFinite else { throw StudioError("The selected adapter weight must be finite.") }
        let developer = arguments.contains("--developer-mode")
        if configuration["schema"] as? String == NativeCompactMaterialModel.schema {
            guard extras.isEmpty, weight == 1 else {
                throw StudioError("A compact model is a complete standalone checkpoint; weighted LoRA combinations do not apply.")
            }
            return try addingSource(try publish(files: ["model.safetensors": selected.bytes], configuration: configuration,
                                               output: URL(fileURLWithPath: output)), digest: selected.sha256)
        }
        var adapter = selected
        if full {
            let partner = try NativeSafetensors(contentsOf: source.deletingLastPathComponent().appendingPathComponent("adapter.safetensors"))
            guard configuration["fused_adapter_sha256"] as? String == partner.sha256 else { throw StudioError("The full checkpoint needs its matching saved adapter for this export.") }
            _ = try NativeMaterialCheckpoint.inspect(at: source.deletingLastPathComponent().appendingPathComponent("adapter.safetensors"), expectedSHA256: partner.sha256)
            adapter = partner
            if extras.isEmpty && weight == 1 {
                if !developer { configuration = try adapterConfiguration(adapter) }
                var files = ["adapter.safetensors": adapter.bytes]
                if developer { files["model.safetensors"] = selected.bytes }
                return try addingSource(try publish(files: files, configuration: configuration, output: URL(fileURLWithPath: output)), digest: selected.sha256)
            }
        }
        configuration = try adapterConfiguration(adapter)
        var tensors = try adapter.nativeTensors()
        if !extras.isEmpty || weight != 1 {
            var inputs: [(NativeSafetensors, Float)] = [(adapter, weight)]
            for entry in extras {
                guard let separator = entry.lastIndex(of: "="), let weight = Float(entry[entry.index(after: separator)...]), weight.isFinite else {
                    throw StudioError("Each weighted adapter requires PATH=WEIGHT with a finite weight.")
                }
                let url = NativeMaterialModel.resolvedCheckpoint(URL(fileURLWithPath: String(entry[..<separator])))
                let snapshot = try NativeSafetensors(contentsOf: url)
                _ = try NativeMaterialCheckpoint.inspect(at: url, expectedSHA256: snapshot.sha256)
                inputs.append((snapshot, weight))
            }
            (configuration, tensors) = try combine(inputs)
        }
        // Unmodified export is an exact snapshot, including original metadata,
        // header spacing and every original numeric bit pattern.
        var files = ["adapter.safetensors": extras.isEmpty && weight == 1 ? adapter.bytes : try NativeSafetensors.encoded(tensors: tensors, metadata: metadata(configuration))]
        if developer {
            guard let modelPath = value("--model-directory"), let target = configuration["target"] as? String else { throw StudioError("Locate the recorded base weights for a complete export.") }
            let base = baseURL(URL(fileURLWithPath: modelPath), configuration: configuration)
            let layers = try JSONDecoder().decode([String: NativeMaterialModel.AdapterLayer].self, from: JSONSerialization.data(withJSONObject: configuration["layers"]!))
            let loaded = try NativeMaterialModel.load(checkpointURL: nil, baseURL: base, target: target)
            guard loaded.baseSHA256 == (configuration["base"] as? [String: Any])?["sha256"] as? String else { throw StudioError("The adapter requires its exact recorded base weights.") }
            let model = try NativeMaterialModel(baseWeights: loaded.baseWeights, adapterWeights: tensors, layers: layers, configuration: configuration, baseSHA256: loaded.baseSHA256)
            configuration["schema"] = "texture-studio-material-checkpoint-v1"
            configuration["fused_adapter_sha256"] = hash(files["adapter.safetensors"]!)
            files["model.safetensors"] = try NativeSafetensors.encoded(tensors: model.fusedWeights(), metadata: metadata(configuration))
        }
        return try addingSource(try publish(files: files, configuration: configuration, output: URL(fileURLWithPath: output)), digest: selected.sha256)
    }

    static func baseURL(_ directory: URL, configuration: [String: Any]) -> URL {
        if directory.pathExtension == "safetensors" || directory.pathExtension == "pth" { return directory }
        let full = directory.appendingPathComponent("model.safetensors")
        return FileManager.default.fileExists(atPath: full.path) ? full : directory.appendingPathComponent(NativeMaterialModel.pinnedFilename)
    }

    static func combine(_ inputs: [(NativeSafetensors, Float)]) throws -> ([String: Any], [String: NativeTensor]) {
        guard let first = inputs.first else { throw StudioError("Select an adapter to combine.") }
        func layers(_ config: [String: Any]) throws -> [String: NativeMaterialModel.AdapterLayer] {
            guard let value = config["layers"] else { throw StudioError("Missing adapter layers.") }
            let decoded = try JSONDecoder().decode([String: NativeMaterialModel.AdapterLayer].self, from: JSONSerialization.data(withJSONObject: value))
            guard !decoded.isEmpty, decoded.values.allSatisfy({ [2, 4].contains($0.weightShape.count) && $0.weightShape.allSatisfy { $0 > 0 } && $0.rank > 0 && $0.alpha.isFinite && $0.alpha > 0 }) else { throw StudioError("Invalid adapter layer specification.") }
            return decoded
        }
        var configuration = try adapterConfiguration(first.0)
        guard configuration["schema"] as? String == "texture-studio-material-lora-v1" else {
            throw StudioError("Weighted combinations require LoRA adapters; complete standalone models cannot be combined as adapters.")
        }
        let specs = try layers(configuration)
        var decoded: [([String: NativeTensor], [String: NativeMaterialModel.AdapterLayer], Float)] = []
        for (snapshot, weight) in inputs {
            _ = try snapshot.validateFiniteFloatingPoint()
            let next = try Self.adapterConfiguration(snapshot)
            guard weight.isFinite, next["schema"] as? String == "texture-studio-material-lora-v1",
                  next["target"] as? String == configuration["target"] as? String,
                  (next["base"] as? [String: Any])?["sha256"] as? String == (configuration["base"] as? [String: Any])?["sha256"] as? String,
                  next["scope"] as? String == configuration["scope"] as? String else { throw StudioError("Combined adapters must use the same exact base, target and scope.") }
            let nextLayers = try layers(next)
            guard Set(nextLayers.keys) == Set(specs.keys), nextLayers.allSatisfy({ specs[$0.key]?.weightShape == $0.value.weightShape }),
                  Set(snapshot.tensors.keys) == Set(specs.keys.flatMap { [$0 + ".lora_A", $0 + ".lora_B"] }) else { throw StudioError("Combined adapters have different trained layers.") }
            decoded.append((try snapshot.nativeTensors(), nextLayers, weight))
        }
        var result: [String: NativeTensor] = [:], combinedSpecs: [String: NativeMaterialModel.AdapterLayer] = [:]
        for name in specs.keys.sorted() {
            try Task.checkCancellation()
            var incoming = 1
            for dimension in specs[name]!.weightShape.dropFirst() {
                let product = incoming.multipliedReportingOverflow(by: dimension)
                guard !product.overflow else { throw StudioError("Adapter dimensions overflow.") }
                incoming = product.partialValue
            }
            let outgoing = specs[name]!.weightShape[0]
            var rank = 0
            for (_, layers, _) in decoded {
                let sum = rank.addingReportingOverflow(layers[name]!.rank)
                guard !sum.overflow else { throw StudioError("The combined adapter rank overflows its integer representation.") }
                rank = sum.partialValue
            }
            let aCount = rank.multipliedReportingOverflow(by: incoming), bCount = outgoing.multipliedReportingOverflow(by: rank)
            guard !aCount.overflow, !bCount.overflow, aCount.partialValue <= 268_435_456, bCount.partialValue <= 268_435_456 else { throw StudioError("The combined adapter exceeds the format budget.") }
            var a: [Float] = [], b = [Float](repeating: 0, count: outgoing * rank), column = 0
            for (values, layers, weight) in decoded {
                let spec = layers[name]!, factor = weight * spec.alpha / Float(spec.rank)
                guard let left = values[name + ".lora_A"], let right = values[name + ".lora_B"], left.shape == [spec.rank, incoming], right.shape == [outgoing, spec.rank] else { throw StudioError("Invalid adapter factors.") }
                a += try left.floatValues()
                let source = try right.floatValues()
                for row in 0..<outgoing {
                    for j in 0..<spec.rank { b[row * rank + column + j] = source[row * spec.rank + j] * factor }
                }
                column += spec.rank
            }
            guard b.allSatisfy(\.isFinite) else { throw StudioError("The weighted adapter overflows Float32.") }
            result[name + ".lora_A"] = .floats(a, shape: [rank, incoming]); result[name + ".lora_B"] = .floats(b, shape: [outgoing, rank])
            combinedSpecs[name] = .init(weightShape: specs[name]!.weightShape, rank: rank, alpha: Float(rank))
        }
        configuration["layers"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(combinedSpecs))
        configuration["adapter_mixture"] = inputs.map { ["sha256": $0.0.sha256, "weight": $0.1] as [String: Any] }
        return (configuration, result)
    }

    private static func publish(files supplied: [String: Data], configuration: [String: Any], output: URL) throws -> String {
        try Task.checkCancellation()
        let fm = FileManager.default
        guard !fm.fileExists(atPath: output.path) else { throw StudioError("Choose a new model export directory.") }
        var files = supplied, configuration = configuration
        let full = files["model.safetensors"] != nil
        let compact = configuration["schema"] as? String == NativeCompactMaterialModel.schema
        configuration["checkpoint_filename"] = full ? "model.safetensors" : "adapter.safetensors"
        if !compact { configuration["adapter_filename"] = "adapter.safetensors" }
        configuration["full_checkpoint"] = full; configuration["optimizer_included"] = false; configuration["source_images_included"] = false
        files["config.json"] = try JSONSerialization.data(withJSONObject: configuration, options: [.prettyPrinted, .sortedKeys])
        for name in compact ? compactLicenses : licenses {
            let source = Bundle.main.url(forResource: name, withExtension: nil, subdirectory: "ModelLicenses")
                ?? Bundle.main.url(forResource: name, withExtension: nil)
                ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Resources/ModelLicenses/" + name)
            files["ModelLicenses/" + name] = try Data(contentsOf: source)
        }
        files["LICENSE"] = files["ModelLicenses/TextureStudio_LICENSE"]
        let modelName = (configuration["model_name"] as? String).map { "Model name: \($0)\n\n" } ?? ""
        let familyDescription = compact ? "This compact model trains every weight from scratch and includes its complete standalone weights. No downloaded base or LoRA adapter is required." : "Original model notices are in ModelLicenses."
        files["README.md"] = Data("---\nlicense: gpl-3.0\ntags:\n- texture-studio-material\n- material-maps\n- safetensors\n---\n\n# Texture Studio native material model\n\n\(modelName)Target: \(configuration["target"] ?? ""). Every training map uses its declared native grid without padding or resizing. The app evaluates this numeric checkpoint with Apple MPSGraph in Float32. \(familyDescription) Source images and an interpreter are not included. Visual review is required before choosing a model.\n".utf8)
        let size = files.values.reduce(Int64(0)) { $0 + Int64($1.count) }
        try fm.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        let capacity = try output.deletingLastPathComponent().resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage
        guard capacity.map({ $0 > size + 67_108_864 }) ?? false else { throw StudioError("Free disk space is insufficient for this model export.") }
        let stage = output.deletingLastPathComponent().appendingPathComponent(".native-package-" + UUID().uuidString)
        try fm.createDirectory(at: stage, withIntermediateDirectories: false)
        defer { try? fm.removeItem(at: stage) }
        var hashes: [String: String] = [:]
        for (name, bytes) in files {
            try Task.checkCancellation()
            let file = stage.appendingPathComponent(name)
            try fm.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bytes.write(to: file, options: [.atomic])
            let digest = hash(bytes)
            guard try NativeMaterialTransfer.hash(file) == digest else { throw StudioError("Exported model bytes failed verification.") }
            hashes[name] = digest
        }
        try NativeMaterialTransfer.writeJSON(hashes, to: stage.appendingPathComponent("SHA256SUMS.json"))
        _ = try verify(stage)
        try Task.checkCancellation()
        try fm.moveItem(at: stage, to: output)
        var result = try NativeMaterialTransfer.object(Data(NativeMaterialCheckpoint.inspect(at: output.appendingPathComponent(full ? "model.safetensors" : "adapter.safetensors")).utf8))
        result["package_path"] = output.path
        if !compact { result["adapter_path"] = output.appendingPathComponent("adapter.safetensors").path }
        return try NativeMaterialTransfer.json(result)
    }
    private static func metadata(_ configuration: [String: Any]) throws -> [String: String] { ["configuration": try NativeMaterialTransfer.json(configuration)] }
    private static func adapterConfiguration(_ snapshot: NativeSafetensors) throws -> [String: Any] {
        guard let json = snapshot.metadata["configuration"] else { throw StudioError("Missing adapter configuration.") }
        return try NativeMaterialTransfer.object(Data(json.utf8))
    }
    private static func hash(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
    private static func addingSource(_ text: String, digest: String) throws -> String {
        var object = try NativeMaterialTransfer.object(Data(text.utf8)); object["source_checkpoint_sha256"] = digest
        return try NativeMaterialTransfer.json(object)
    }
}
