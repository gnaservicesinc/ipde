import Foundation

enum NativeMaterialCommands {
    static func run(arguments: [String], onEvent: @escaping @Sendable (String) -> Void,
                    control: NativeMaterialTrainingControl) async throws -> String {
        try Task.checkCancellation()
        func value(_ flag: String) -> String? { arguments.firstIndex(of: flag).flatMap { arguments.indices.contains($0 + 1) ? arguments[$0 + 1] : nil } }
        func url(_ flag: String) throws -> URL {
            guard let path = value(flag), !path.isEmpty else { throw StudioError("Missing native material operation argument: \(flag)") }
            return URL(fileURLWithPath: path)
        }
        if let result = try await NativeMaterialDatasetService.run(arguments: arguments, onEvent: onEvent) { return result }
        switch arguments.first {
        case "checkpoint":
            let file = try url("--checkpoint"), expected = value("--expected-sha256")
            let job = Task.detached { try NativeMaterialCheckpoint.inspect(at: file, expectedSHA256: expected) }
            return try await withTaskCancellationHandler { try await job.value } onCancel: { job.cancel() }
        case "capabilities":
            let family = value("--model-family") ?? "pbrnxt"
            guard ["pbrnxt", "compact-scalar", "compact-normal"].contains(family) else { throw StudioError("Unsupported material model family.") }
            let scope = value("--scope") ?? "final-map"
            guard family != "pbrnxt" || ["final-map", "map-decoder"].contains(scope) else { throw StudioError("Unsupported material training scope.") }
            let families: [[String: Any]] = [
                ["id": "pbrnxt", "architecture": "pbrnxt-native-v1", "targets": ["height", "roughness", "normal"], "base_required": true, "training_policy": "lora"],
                ["id": "compact-scalar", "architecture": "texture-studio-compact-scalar-native-v1", "targets": ["height", "roughness"], "base_required": false, "training_policy": "all_weights_from_scratch"],
                ["id": "compact-normal", "architecture": "texture-studio-compact-normal-native-v1", "targets": ["normal"], "base_required": false, "training_policy": "all_weights_from_scratch"]]
            let targets = family == "compact-scalar" ? ["height", "roughness"] : family == "compact-normal" ? ["normal"] : ["height", "roughness", "normal"]
            return try NativeMaterialTransfer.json(["training_sizes": [256,512,1024,2048,4096], "inference_sizes": [256,512,1024,2048,4096,8192],
                "targets": targets, "scope": family == "pbrnxt" ? scope : "full-model", "model_family": family, "model_families": families,
                "base_required": family == "pbrnxt", "all_weights_trainable": family != "pbrnxt", "image_size_matches_training_size": true,
                "hidden_encoder_resize": false, "memory_admission_enabled": true])
        case "hub-account": return try await NativeHuggingFaceService().accountJSON()
        case "hub-models": return try await NativeHuggingFaceService().modelsJSON()
        case "train", "refine", "infer": return try await NativeMaterialTrainer.run(arguments: arguments, onEvent: onEvent, control: control)
        case "review-source":
            guard let result = try await ReviewImageLoader.runNativeSource(arguments: arguments) else { throw StudioError("Invalid native source review request.") }
            return result
        case "package": return try await NativeMaterialPackage.run(arguments: arguments)
        case "upload-selected":
            let package = try url("--output")
            let packageResult = try await NativeMaterialPackage.run(arguments: arguments)
            let packaged = try NativeMaterialTransfer.object(Data(packageResult.utf8))
            guard let repository = value("--repo") else { throw StudioError("Choose a Hub repository.") }
            let uploadResult = try await NativeMaterialTransfer().upload(package: package, repository: repository, isPublic: arguments.contains("--public"))
            var uploaded = try NativeMaterialTransfer.object(Data(uploadResult.utf8))
            uploaded["source_checkpoint_sha256"] = packaged["source_checkpoint_sha256"]
            uploaded["package_path"] = package.path
            return try NativeMaterialTransfer.json(uploaded)
        case "upload":
            guard let repository = value("--repo") else { throw StudioError("Choose a Hub repository.") }
            return try await NativeMaterialTransfer().upload(package: url("--package"), repository: repository, isPublic: arguments.contains("--public"))
        case "download-model":
            guard let repository = value("--repo"), let revision = value("--revision") else { throw StudioError("Choose a Hub model with an exact recorded revision.") }
            return try await NativeMaterialTransfer().download(repository: repository, revision: revision, to: url("--destination"))
        case "install-base": return try await NativeMaterialTransfer.installBase(at: url("--destination"))
        case "remove-base": return try NativeMaterialTransfer.removeBase(at: url("--directory"))
        default: throw StudioError("Unsupported native material operation: \(arguments.first ?? "missing")")
        }
    }
}

/// Native events are complete Swift strings. Keep the complete worklog on disk,
/// a bounded UTF-8 display tail, and lossless chunks awaiting one UI delivery.
/// Writes and delivery reservations share a lock off the app's main actor.
final class NativeWorkbenchLog: @unchecked Sendable {
    private let lock = NSLock()
    private let file: FileHandle?
    private var displayBytes: [UInt8]
    private var displayStart = 0
    private var displayCount = 0
    private var pending: [String] = []
    private var deliveryReserved = false

    init(url: URL, displayByteLimit: Int = 100000) {
        displayBytes = [UInt8](repeating: 0, count: max(0, displayByteLimit))
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        file = try? FileHandle(forWritingTo: url)
        _ = try? file?.seekToEnd()
    }
    deinit { try? file?.close() }

    var text: String {
        lock.withLock {
            guard displayCount > 0 else { return "" }
            let firstCount = min(displayCount, displayBytes.count - displayStart)
            var bytes = Data(displayBytes[displayStart..<(displayStart + firstCount)])
            if firstCount < displayCount { bytes.append(contentsOf: displayBytes[0..<(displayCount - firstCount)]) }
            // A byte cap can cut through the oldest scalar. Remove its trailing
            // bytes so the displayed suffix never introduces replacement text.
            var start = bytes.startIndex
            while start < bytes.endIndex, bytes[start] & 0xc0 == 0x80 { start += 1 }
            return String(decoding: bytes[start...], as: UTF8.self)
        }
    }

    /// Returns true only when this append reserves the next delivery. Callers
    /// schedule one delayed main-actor drain for that reservation, not per event.
    @discardableResult func append(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        let bytes = Data(text.utf8)
        return lock.withLock {
            try? file?.write(contentsOf: bytes)
            appendDisplay(bytes)
            pending.append(text)
            guard !deliveryReserved else { return false }
            deliveryReserved = true
            return true
        }
    }

    /// Clears the reservation and returns each queued chunk exactly once. A
    /// final drain consumes only undelivered events, never the display tail.
    func drainPending() -> String {
        lock.withLock {
            let result = pending.joined()
            pending.removeAll(keepingCapacity: true)
            deliveryReserved = false
            return result
        }
    }

    private func appendDisplay(_ bytes: Data) {
        let capacity = displayBytes.count
        guard capacity > 0 else { return }
        if bytes.count >= capacity {
            displayBytes.replaceSubrange(0..<capacity, with: bytes.suffix(capacity))
            displayStart = 0; displayCount = capacity
            return
        }
        let end = (displayStart + displayCount) % capacity
        let firstCount = min(bytes.count, capacity - end)
        displayBytes.replaceSubrange(end..<(end + firstCount), with: bytes.prefix(firstCount))
        if firstCount < bytes.count {
            displayBytes.replaceSubrange(0..<(bytes.count - firstCount), with: bytes.dropFirst(firstCount))
        }
        let overflow = max(0, displayCount + bytes.count - capacity)
        displayStart = (displayStart + overflow) % capacity
        displayCount = min(capacity, displayCount + bytes.count)
    }
}
