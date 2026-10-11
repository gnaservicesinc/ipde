import Foundation

/// An explicit document handoff keeps an already-running Trainer independent
/// of another tool's in-memory state and shared preferences.
struct MaterialTrainingHandoff: Codable, Sendable {
    static let schemaName = "texture-studio-material-training-handoff-v1"
    static let documentName = "material-training-handoff.json"

    let schema: String
    let requestID: UUID
    let checkpointURL: URL
    let checkpointSHA256: String
    let datasetURL: URL?
    let training: MaterialTrainingOptions
    let sampleID: String?
    let inputVariantID: String?

    init(checkpoint: WorkbenchCheckpoint, dataset: URL?, training: MaterialTrainingOptions,
         sampleID: String?, inputVariantID: String?) throws {
        guard checkpoint.supportsTrainingWarmStart else {
            throw StudioError("The selected checkpoint does not support material refinement.")
        }
        guard checkpoint.modelFamily == training.modelFamily,
              !training.modelFamily.isCompact || checkpoint.target == training.target else {
            throw StudioError("The starting checkpoint must match the compact training family and target.")
        }
        schema = Self.schemaName
        requestID = UUID()
        checkpointURL = checkpoint.url.standardizedFileURL
        checkpointSHA256 = checkpoint.sha256.lowercased()
        datasetURL = dataset?.standardizedFileURL
        self.training = training
        self.sampleID = sampleID
        self.inputVariantID = inputVariantID
        try validate()
    }

    static func read(from url: URL) throws -> Self {
        guard localFile(url) else { throw StudioError("A Trainer handoff must be a local file.") }
        let handoff = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        try handoff.validate()
        return handoff
    }

    func write(to url: URL) throws {
        try validate()
        guard Self.localFile(url) else { throw StudioError("Save the Trainer handoff to a local file.") }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: url, options: .atomic)
    }

    func writeTemporary() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(temporaryFolderName, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false,
                                               attributes: [.posixPermissions: 0o700])
        let document = folder.appendingPathComponent(Self.documentName)
        do {
            try write(to: document)
            return document
        } catch {
            try? FileManager.default.removeItem(at: folder)
            throw error
        }
    }

    /// Only remove the temporary document whose directory matches this request.
    /// Imported or saved copies outside that directory remain untouched.
    @discardableResult
    func discardTemporaryFile(at url: URL) throws -> Bool {
        let expected = FileManager.default.temporaryDirectory
            .appendingPathComponent(temporaryFolderName, isDirectory: true)
            .appendingPathComponent(Self.documentName)
        guard url.standardizedFileURL.resolvingSymlinksInPath() == expected.standardizedFileURL.resolvingSymlinksInPath() else { return false }
        let folder = url.deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath: folder.path) else { return true }
        if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
        if (try FileManager.default.contentsOfDirectory(atPath: folder.path)).isEmpty {
            try FileManager.default.removeItem(at: folder)
        }
        return true
    }

    private var temporaryFolderName: String { "texture-studio-training-handoff-" + requestID.uuidString }
    private static func localFile(_ url: URL) -> Bool {
        url.isFileURL && url.path.hasPrefix("/") && (url.host == nil || url.host == "" || url.host == "localhost")
    }
    private func validate() throws {
        guard schema == Self.schemaName else { throw StudioError("This Trainer handoff version is unsupported.") }
        guard Self.localFile(checkpointURL), datasetURL.map(Self.localFile) != false else {
            throw StudioError("Trainer handoffs must identify local checkpoint and dataset files.")
        }
        guard checkpointSHA256.count == 64, checkpointSHA256.allSatisfy(\.isHexDigit) else {
            throw StudioError("The Trainer handoff needs the selected checkpoint's exact SHA256.")
        }
        if let issue = training.configurationIssue { throw StudioError(issue) }
        guard ["height", "roughness", "normal"].contains(training.target),
              training.modelFamily.isCompact || ["final-map", "map-decoder"].contains(training.scope),
              training.modelFamily.supports(target: training.target),
              [256, 512, 1024, 2048].contains(training.size), training.useWarmStart,
              training.updatesPerCrop > 0,
              training.maxMinutes.isFinite, training.maxMinutes > 0,
              training.modelFamily.isCompact || training.loraRank > 0 && training.loraAlpha.isFinite && training.loraAlpha > 0,
              training.validationEvery >= 0,
              training.checkpointEvery >= 0,
              sampleID?.isEmpty != true, inputVariantID?.isEmpty != true else {
            throw StudioError("The Trainer handoff contains unsupported training settings.")
        }
    }
}
