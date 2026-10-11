import Foundation
import Darwin

enum DatasetSheetRoute: String, Identifiable {
    case new, add, info, folder
    var id: String { rawValue }
}

enum MaterialTool: String, CaseIterable, Identifiable {
    case review, compare, dataset, train
    var id: String { rawValue }
    var title: String {
        switch self {
        case .review: "Material Review"
        case .compare: "Checkpoint Compare"
        case .dataset: "Material Dataset"
        case .train: "Material Trainer"
        }
    }
    var symbol: String {
        switch self { case .review: "photo.on.rectangle"; case .compare: "rectangle.split.2x1"; case .dataset: "square.stack.3d.up"; case .train: "cpu" }
    }
    static var launchRole: MaterialTool? {
        if let i = CommandLine.arguments.firstIndex(of: "--tool"), CommandLine.arguments.indices.contains(i + 1) {
            return MaterialTool(rawValue: CommandLine.arguments[i + 1])
        }
        return (Bundle.main.object(forInfoDictionaryKey: "MaterialToolRole") as? String).flatMap(MaterialTool.init(rawValue:))
    }
}

struct WorkbenchMap: Decodable, Identifiable, Sendable {
    let path: String
    let sha256: String?
    let sourceBits: Int?
    let encoding: String?
    let width: Int?
    let height: Int?
    var originalSourcePath: String? = nil
    var originalSourceSha256: String? = nil
    var originalSourceWidth: Int? = nil
    var originalSourceHeight: Int? = nil
    var originalNormalConvention: String? = nil
    var sourceNormalConvention: String? = nil
    var cropRectangle: [Int]? = nil
    var variantId: String? = nil
    var id: String { path }
    var url: URL { URL(fileURLWithPath: path) }
}

struct WorkbenchSample: Decodable, Identifiable, Sendable {
    let sampleId: String
    let status: String
    let split: String
    let width: Int
    let height: Int
    let maps: [String: WorkbenchMap]
    let note: String?
    var sourceFamilyId: String? = nil
    var sourceSetId: String? = nil
    var inputVariants: [WorkbenchMap]? = nil
    var availableTargets: [String]? = nil
    var splitAssignment: String? = nil
    var sourceRegionId: String? = nil
    var id: String { sampleId }
}

struct WorkbenchMaterial: Decodable, Identifiable, Sendable {
    let materialId: String
    let samples: [WorkbenchSample]
    var name: String? = nil
    var subjectId: String? = nil
    var sourceDirectory: String? = nil
    var id: String { materialId }
}

struct WorkbenchDataset: Decodable, Sendable {
    let datasetPath: String
    let indexSha256: String
    let materials: [WorkbenchMaterial]
    let validationScope: String?
    let crossSizeValidationNotice: String?
    let preparation: WorkbenchDatasetPreparation?
    let automaticValidation: WorkbenchAutomaticValidation?
    var supportedTrainingSizes: [Int]? = nil
    var name: String? = nil
    var description: String? = nil
    var materialCount: Int? = nil
    var sampleCount: Int? = nil
    var reviewSha256: String? = nil
    var addedMaterialCount: Int? = nil
    var duplicateMaterialCount: Int? = nil
    var trainingSize: Int? = nil
    var reviewSize: Int? = nil
    var sourceSetCount: Int? = nil
    var trainingPlans: [String: WorkbenchDatasetPlan]? = nil
    var resolutionPlans: [String: [String: WorkbenchDatasetPlan]]? = nil
    var validation: WorkbenchValidationSettings? = nil
    var subjects: [WorkbenchSubject]? = nil
    var samples: [WorkbenchSample] { materials.flatMap(\.samples) }
    func readyForTraining(size: Int, material: String?, target: String? = nil) -> Bool {
        guard hasNativeSize(size), let policy = automaticValidation,
              policy.policy == "subject-extra-crops-v2",
              target == nil || policy.target == nil || policy.target == target else { return false }
        if let target {
            let candidates = materials.filter { material == nil || $0.id == material }.flatMap(\.samples)
                .filter { !["excluded", "rejected"].contains($0.status) }
            guard !candidates.isEmpty, candidates.allSatisfy({ Set($0.maps.keys) == Set(["input", target]) }) else { return false }
        }
        if let material { return policy.quickFitMaterialId == material }
        return policy.quickFitMaterialId == nil
    }
    func hasNativeSize(_ size: Int) -> Bool {
        !samples.isEmpty && samples.allSatisfy { sample in
            sample.width == size && sample.height == size && !sample.maps.isEmpty &&
                sample.maps.values.allSatisfy { $0.width == size && $0.height == size }
        }
    }
}

struct WorkbenchDatasetPlan: Decodable, Sendable {
    let size: Int
    let cropCount: Int
    let sourceSetCount: Int
    let trainCount: Int
    let validationCount: Int
    let excludedCount: Int
    let unavailableTargetCount: Int
    let undersizedSourceSetCount: Int?
    var smallerAlternateSourceSetCount: Int? = nil
    var regionalFamilies: [String] = []
    var subjectCount: Int? = nil
    var sharedValidationCount: Int? = nil
    var validationLimit: Int? = nil
    var validationCandidateCount: Int? = nil
    var subjects: [WorkbenchSubject]? = nil
    var sourceIssues: [WorkbenchImportSourceIssue] = []
}

extension WorkbenchDatasetPlan {
    private enum CodingKeys: String, CodingKey {
        case size, cropCount, sourceSetCount, trainCount, validationCount, excludedCount, unavailableTargetCount
        case undersizedSourceSetCount, smallerAlternateSourceSetCount, regionalFamilies, subjectCount, sharedValidationCount, validationLimit
        case validationCandidateCount, subjects, sourceIssues
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.init(size: try values.decode(Int.self, forKey: .size),
                  cropCount: try values.decode(Int.self, forKey: .cropCount),
                  sourceSetCount: try values.decode(Int.self, forKey: .sourceSetCount),
                  trainCount: try values.decode(Int.self, forKey: .trainCount),
                  validationCount: try values.decode(Int.self, forKey: .validationCount),
                  excludedCount: try values.decode(Int.self, forKey: .excludedCount),
                  unavailableTargetCount: try values.decode(Int.self, forKey: .unavailableTargetCount),
                  undersizedSourceSetCount: try values.decodeIfPresent(Int.self, forKey: .undersizedSourceSetCount),
                  smallerAlternateSourceSetCount: try values.decodeIfPresent(Int.self, forKey: .smallerAlternateSourceSetCount),
                  regionalFamilies: try values.decodeIfPresent([String].self, forKey: .regionalFamilies) ?? [],
                  subjectCount: try values.decodeIfPresent(Int.self, forKey: .subjectCount),
                  sharedValidationCount: try values.decodeIfPresent(Int.self, forKey: .sharedValidationCount),
                  validationLimit: try values.decodeIfPresent(Int.self, forKey: .validationLimit),
                  validationCandidateCount: try values.decodeIfPresent(Int.self, forKey: .validationCandidateCount),
                  subjects: try values.decodeIfPresent([WorkbenchSubject].self, forKey: .subjects),
                  sourceIssues: try values.decodeIfPresent([WorkbenchImportSourceIssue].self, forKey: .sourceIssues) ?? [])
    }
}

struct WorkbenchImportSourceIssue: Decodable, Identifiable, Sendable {
    let materialId: String
    let sourcePath: String
    let target: String?
    let code: String
    let reason: String
    let cropCount: Int
    var id: String { [materialId, sourcePath, target ?? "", code].joined(separator: "|") }
}

struct WorkbenchValidationSettings: Codable, Equatable, Sendable {
    var enabled = true
    var percent = 5.0
    var maxCrops = 0
    var quickCount = 4
    var folders: [String: Bool] = [:]
    var arguments: [String] {
        ["--validation-enabled", enabled ? "yes" : "no", "--validation-percent", String(percent),
         "--validation-max-crops", String(maxCrops), "--validation-quick-count", String(quickCount)]
    }
}

struct WorkbenchSubject: Decodable, Identifiable, Sendable {
    let subjectId: String
    let name: String
    let selected: Bool
    let available: Bool
    let preference: Bool?
    let reason: String?
    var id: String { subjectId }
}

struct WorkbenchFolderImport: Decodable, Sendable {
    let folderPath: String
    let planPath: String
    let planSha256: String
    let indexSha256: String
    let sourceSetCount: Int
    let addedMaterialCount: Int
    let duplicateMaterialCount: Int
    let ignoredFileCount: Int
    let warnings: [String]
    let plans: [String: [String: WorkbenchDatasetPlan]]
    var recoveredMapCount: Int? = nil
}

struct WorkbenchDatasetLocation: Codable, Identifiable, Equatable, Sendable {
    var name: String
    var path: String
    var id: String { path }
    var url: URL { URL(fileURLWithPath: path) }
}

struct WorkbenchDatasetDeletion: Decodable, Sendable {
    let datasetPath: String
    let indexSha256: String
    let safeToTrashFolder: Bool
    let trashPaths: [String]
}

struct WorkbenchAutomaticValidation: Decodable, Sendable {
    let policy: String
    let materialIds: [String]
    let quickFitMaterialId: String?
    var target: String? = nil
}

struct WorkbenchDatasetPreparation: Decodable, Sendable {
    let sourceDatasetPath: String
    let sourceIndexSha256: String
    let preparedDatasetPath: String
    let cropSize: Int
    let reused: Bool
    let targetResized: Bool
    let originalDatasetModified: Bool
    let splitLineageChanged: Bool?
    let crossSizeValidationNotice: String?
    var targetCropped: Bool? = nil
}

struct WorkbenchCheckpoint: Decodable, Identifiable, Sendable {
    struct Base: Decodable, Sendable { let sha256: String }
    let checkpointPath: String
    let sha256: String
    let schema: String
    let target: String
    let step: Int
    let compatible: Bool
    let variant: String?
    let warmStartSupported: Bool?
    let refinementPolicy: String?
    var architecture: String? = nil
    var scope: String? = nil
    var base: Base? = nil
    var modelName: String? = nil
    enum CodingKeys: String, CodingKey {
        case checkpointPath, sha256, schema, target, step, compatible, variant, refinementPolicy, architecture, scope, base, modelName
        case warmStartSupported = "supportsTrainingWarmStart"
    }
    var modelFamily: MaterialTrainingModelFamily? {
        if schema == "texture-studio-compact-material-v1" {
            let family = MaterialTrainingModelFamily(architecture: architecture)
            return family?.supports(target: target) == true ? family : nil
        }
        return ["texture-studio-material-lora-v1", "texture-studio-material-checkpoint-v1"].contains(schema) ? .pbrnxt : nil
    }
    var supportsTrainingWarmStart: Bool { compatible && warmStartSupported == true && modelFamily != nil }
    var supportsStudioInference: Bool {
        compatible && modelFamily != nil
    }
    func matchesTraining(_ options: MaterialTrainingOptions) -> Bool {
        guard modelFamily == options.modelFamily else { return false }
        if options.modelFamily.isCompact { return target == options.target }
        return schema != "texture-studio-material-lora-v1" ||
            target == options.target && (scope ?? "final-map") == options.scope
    }
    var availabilityLabel: String { modelFamily?.isCompact == true ? "Full compact model" : variant == "full" ? "Full material checkpoint" : "Material LoRA" }
    var id: String { sha256 }
    var url: URL { URL(fileURLWithPath: checkpointPath) }
    var title: String {
        let name = modelName?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return (name.isEmpty ? url.deletingLastPathComponent().lastPathComponent : name) + " · " + url.lastPathComponent
    }
    var trainingBaseLabel: String {
        architecture ?? "PBRnxt material model"
    }
    var modelSummary: String {
        let map = target == "height" ? "Surface height / displacement" : target.capitalized
        return "\(map) · \(trainingBaseLabel) · step \(step.formatted())"
    }
}

enum MaterialTrainingIntervalUnit: String, Codable, CaseIterable, Hashable, Sendable {
    case epoch
    case step

    var label: String { rawValue }
}

enum MaterialTrainingModelFamily: String, Codable, CaseIterable, Hashable, Sendable {
    case pbrnxt
    case compactScalar = "compact-scalar"
    case compactNormal = "compact-normal"

    var isCompact: Bool { self != .pbrnxt }
    var label: String {
        switch self {
        case .pbrnxt: "PBRnxt refinement"
        case .compactScalar: "Compact scalar"
        case .compactNormal: "Compact normals"
        }
    }
    var defaultLearningRate: Double { isCompact ? 0.001 : 0.00001 }
    var architecture: String? {
        switch self {
        case .pbrnxt: nil
        case .compactScalar: "texture-studio-compact-scalar-native-v1"
        case .compactNormal: "texture-studio-compact-normal-native-v1"
        }
    }
    init?(architecture: String?) {
        guard let value = Self.allCases.first(where: { $0.isCompact && $0.architecture == architecture }) else { return nil }
        self = value
    }
    func supports(target: String) -> Bool {
        switch self {
        case .pbrnxt: ["height", "roughness", "normal"].contains(target)
        case .compactScalar: ["height", "roughness"].contains(target)
        case .compactNormal: target == "normal"
        }
    }
}

struct MaterialTrainingOptions: Codable, Equatable, Sendable {
    var modelName = ""
    var modelFamily: MaterialTrainingModelFamily = .pbrnxt
    var target = "height"
    var scope = "final-map"
    var size = 1024
    var updatesPerCrop = 100
    var maxMinutes = 30.0
    var useSelectedMaterialOnly = false
    var useWarmStart = false
    var loraRank = 8
    var loraAlpha = 8.0
    var learningRate = 0.00001
    var gradientAccumulationSteps = 1
    var optimizer = "adamw"
    var optimizerBeta1 = 0.9
    var optimizerBeta2 = 0.999
    var optimizerEpsilon = 1e-8
    var weightDecay = 0.0
    var maxGradientNorm = 1.0
    var learningRateSchedule = "constant"
    var minimumLearningRateRatio = 0.1
    var warmupUpdates = 0
    var seed: UInt64 = 17
    var validationEvery = 0
    var validationUnit: MaterialTrainingIntervalUnit = .epoch
    var checkpointEvery = 0
    var checkpointUnit: MaterialTrainingIntervalUnit = .epoch

    init() {}

    enum CodingKeys: String, CodingKey {
        case modelName, modelFamily, target, scope, size, updatesPerCrop, maxMinutes, useSelectedMaterialOnly, useWarmStart, loraRank, loraAlpha, validationEvery, checkpointEvery
        case validationUnit, checkpointUnit
        case learningRate, gradientAccumulationSteps, optimizer, optimizerBeta1, optimizerBeta2, optimizerEpsilon, weightDecay, maxGradientNorm, learningRateSchedule, minimumLearningRateRatio, warmupUpdates, seed
    }

    init(from decoder: Decoder) throws {
        self.init()
        let values = try decoder.container(keyedBy: CodingKeys.self)
        modelName = try values.decodeIfPresent(String.self, forKey: .modelName) ?? modelName
        modelFamily = try values.decodeIfPresent(MaterialTrainingModelFamily.self, forKey: .modelFamily) ?? .pbrnxt
        target = try values.decodeIfPresent(String.self, forKey: .target) ?? target
        scope = try values.decodeIfPresent(String.self, forKey: .scope) ?? scope
        size = try values.decodeIfPresent(Int.self, forKey: .size) ?? size
        updatesPerCrop = try values.decodeIfPresent(Int.self, forKey: .updatesPerCrop) ?? updatesPerCrop
        maxMinutes = try values.decodeIfPresent(Double.self, forKey: .maxMinutes) ?? maxMinutes
        useSelectedMaterialOnly = try values.decodeIfPresent(Bool.self, forKey: .useSelectedMaterialOnly) ?? useSelectedMaterialOnly
        useWarmStart = try values.decodeIfPresent(Bool.self, forKey: .useWarmStart) ?? useWarmStart
        loraRank = try values.decodeIfPresent(Int.self, forKey: .loraRank) ?? loraRank
        loraAlpha = try values.decodeIfPresent(Double.self, forKey: .loraAlpha) ?? loraAlpha
        learningRate = try values.decodeIfPresent(Double.self, forKey: .learningRate) ?? modelFamily.defaultLearningRate
        gradientAccumulationSteps = try values.decodeIfPresent(Int.self, forKey: .gradientAccumulationSteps) ?? gradientAccumulationSteps
        optimizer = try values.decodeIfPresent(String.self, forKey: .optimizer) ?? optimizer
        optimizerBeta1 = try values.decodeIfPresent(Double.self, forKey: .optimizerBeta1) ?? optimizerBeta1
        optimizerBeta2 = try values.decodeIfPresent(Double.self, forKey: .optimizerBeta2) ?? optimizerBeta2
        optimizerEpsilon = try values.decodeIfPresent(Double.self, forKey: .optimizerEpsilon) ?? optimizerEpsilon
        weightDecay = try values.decodeIfPresent(Double.self, forKey: .weightDecay) ?? weightDecay
        maxGradientNorm = try values.decodeIfPresent(Double.self, forKey: .maxGradientNorm) ?? maxGradientNorm
        learningRateSchedule = try values.decodeIfPresent(String.self, forKey: .learningRateSchedule) ?? learningRateSchedule
        minimumLearningRateRatio = try values.decodeIfPresent(Double.self, forKey: .minimumLearningRateRatio) ?? minimumLearningRateRatio
        warmupUpdates = try values.decodeIfPresent(Int.self, forKey: .warmupUpdates) ?? warmupUpdates
        seed = try values.decodeIfPresent(UInt64.self, forKey: .seed) ?? seed
        validationEvery = try values.decodeIfPresent(Int.self, forKey: .validationEvery) ?? validationEvery
        // Existing saved schedules counted optimizer updates. Preserve their
        // frequency as steps while new or disabled settings default to epochs.
        validationUnit = try values.decodeIfPresent(MaterialTrainingIntervalUnit.self, forKey: .validationUnit) ??
            (validationEvery > 0 ? .step : .epoch)
        checkpointEvery = try values.decodeIfPresent(Int.self, forKey: .checkpointEvery) ?? checkpointEvery
        checkpointUnit = try values.decodeIfPresent(MaterialTrainingIntervalUnit.self, forKey: .checkpointUnit) ??
            (checkpointEvery > 0 ? .step : .epoch)
    }

    /// Preserve supported choices when reopening on another Mac.
    func restored(for resources: MachineResources) -> Self {
        var result = self
        if !["height", "roughness", "normal"].contains(result.target) { result.target = "height" }
        if !result.modelFamily.supports(target: result.target) { result.target = result.modelFamily == .compactNormal ? "normal" : "height" }
        if !["final-map", "map-decoder", "full-model"].contains(result.scope) || !result.modelFamily.isCompact && result.scope == "full-model" { result.scope = "final-map" }
        if ![256, 512, 1024, 2048].contains(result.size) { result.size = 1024 }
        result.loraRank = max(1, result.loraRank)
        result.loraAlpha = result.loraAlpha.isFinite && Float(result.loraAlpha).isFinite && Float(result.loraAlpha) > 0 ? result.loraAlpha : 8
        let defaults = Self()
        if !result.learningRate.isFinite || Float(result.learningRate) <= 0 || !Float(result.learningRate).isFinite { result.learningRate = result.modelFamily.defaultLearningRate }
        result.gradientAccumulationSteps = max(1, result.gradientAccumulationSteps)
        if !["adam", "adamw"].contains(result.optimizer) { result.optimizer = defaults.optimizer }
        if !result.optimizerBeta1.isFinite || result.optimizerBeta1 < 0 || !(0..<1).contains(Float(result.optimizerBeta1)) || result.optimizerBeta1 > 0 && Float(result.optimizerBeta1) == 0 { result.optimizerBeta1 = defaults.optimizerBeta1 }
        if !result.optimizerBeta2.isFinite || result.optimizerBeta2 < 0 || !(0..<1).contains(Float(result.optimizerBeta2)) || result.optimizerBeta2 > 0 && Float(result.optimizerBeta2) == 0 { result.optimizerBeta2 = defaults.optimizerBeta2 }
        if !result.optimizerEpsilon.isFinite || Float(result.optimizerEpsilon) <= 0 || !Float(result.optimizerEpsilon).isFinite { result.optimizerEpsilon = defaults.optimizerEpsilon }
        if !result.weightDecay.isFinite || result.weightDecay < 0 || !Float(result.weightDecay).isFinite || result.weightDecay > 0 && Float(result.weightDecay) == 0 { result.weightDecay = defaults.weightDecay }
        if !result.maxGradientNorm.isFinite || result.maxGradientNorm < 0 || !Float(result.maxGradientNorm).isFinite || result.maxGradientNorm > 0 && Float(result.maxGradientNorm) == 0 { result.maxGradientNorm = defaults.maxGradientNorm }
        if !["constant", "cosine"].contains(result.learningRateSchedule) { result.learningRateSchedule = defaults.learningRateSchedule }
        if !result.minimumLearningRateRatio.isFinite || result.minimumLearningRateRatio > 1 || Float(result.minimumLearningRateRatio) <= 0 { result.minimumLearningRateRatio = defaults.minimumLearningRateRatio }
        result.warmupUpdates = max(0, result.warmupUpdates)
        result.validationEvery = max(0, result.validationEvery)
        result.checkpointEvery = max(0, result.checkpointEvery)
        result.updatesPerCrop = max(1, result.updatesPerCrop)
        result.maxMinutes = result.maxMinutes.isFinite && result.maxMinutes > 0 ? result.maxMinutes : 30
        return result
    }

    var effectiveScope: String { modelFamily.isCompact ? "full-model" : scope }

    /// Both controls and the launcher reject values the Float32 trainer cannot represent.
    var configurationIssue: String? {
        guard modelFamily.supports(target: target) else { return "Choose a map supported by the selected model family." }
        if !modelFamily.isCompact {
            guard loraRank > 0 else { return "LoRA rank must be at least one." }
            guard loraAlpha.isFinite, Float(loraAlpha).isFinite, Float(loraAlpha) > 0 else { return "LoRA alpha must be a positive finite Float32 value." }
        }
        guard updatesPerCrop > 0 else { return "Updates per map must be at least one." }
        guard maxMinutes.isFinite, maxMinutes > 0 else { return "The training time limit must be a positive finite value." }
        guard validationEvery >= 0, checkpointEvery >= 0 else { return "Validation and checkpoint intervals cannot be negative." }
        guard learningRate.isFinite, Float(learningRate).isFinite, Float(learningRate) > 0 else { return "Learning rate must be a positive finite Float32 value." }
        guard gradientAccumulationSteps > 0 else { return "Gradient accumulation must be at least one map." }
        guard ["adam", "adamw"].contains(optimizer) else { return "Choose Adam or AdamW." }
        guard optimizerBeta1.isFinite, optimizerBeta2.isFinite, optimizerBeta1 >= 0, optimizerBeta2 >= 0,
              optimizerBeta1 == 0 || Float(optimizerBeta1) > 0, optimizerBeta2 == 0 || Float(optimizerBeta2) > 0,
              (0..<1).contains(Float(optimizerBeta1)), (0..<1).contains(Float(optimizerBeta2)) else { return "Optimizer beta values must be from zero up to, but below, one." }
        guard optimizerEpsilon.isFinite, Float(optimizerEpsilon).isFinite, Float(optimizerEpsilon) > 0 else { return "Optimizer epsilon must be a positive finite Float32 value." }
        guard weightDecay.isFinite, Float(weightDecay).isFinite, weightDecay >= 0, weightDecay == 0 || Float(weightDecay) > 0 else { return "Weight decay must be a nonnegative finite Float32 value." }
        guard maxGradientNorm.isFinite, Float(maxGradientNorm).isFinite, maxGradientNorm >= 0, maxGradientNorm == 0 || Float(maxGradientNorm) > 0 else { return "Gradient norm limit must be a nonnegative finite Float32 value; zero disables clipping." }
        guard ["constant", "cosine"].contains(learningRateSchedule) else { return "Choose a constant or cosine learning rate schedule." }
        guard minimumLearningRateRatio.isFinite, minimumLearningRateRatio <= 1, Float(minimumLearningRateRatio) > 0 else { return "Minimum learning rate ratio must be above zero and at most one." }
        guard warmupUpdates >= 0 else { return "Warmup updates cannot be negative." }
        return nil
    }
}

struct WorkbenchPreferences: Codable {
    var training: MaterialTrainingOptions?
    var selectedSampleId: String?
    var selectedRole: String?
    var selectedInputVariantId: String?
    var selectedCheckpointId: String?
    var comparisonCheckpointIds: Set<String>?
    var comparisonIncludesBase: Bool?
    var sourceImagePath: String?
    var lastOutputPath: String?
    var lastLogPath: String?
    var lastPackagePath: String?
    var lastPackageCheckpointId: String?
    static let key = "workbenchSettings.v1"

    static func load(from defaults: UserDefaults) -> Self {
        guard let data = defaults.data(forKey: key), let value = try? JSONDecoder().decode(Self.self, from: data) else { return Self() }
        return value
    }
    func save(to defaults: UserDefaults) {
        guard let data = try? JSONEncoder().encode(self) else { return }
        defaults.set(data, forKey: Self.key)
    }
}

struct MaterialInferenceResponse: Decodable, Sendable {
    struct Output: Decodable, Sendable { let path: String }
    let outputs: [String: Output]
    let checkpointSha256: String
}

struct SelectedMaterialCheckpoint: Codable, Sendable {
    let checkpointPath: String
    let sha256: String
    let target: String
    let workspacePath: String
    let modelDirectory: String
    var displayName: String? = nil
    var modelSummary: String? = nil
    var supportsStudioInference: Bool { ["height", "normal", "roughness"].contains(target) && URL(fileURLWithPath: checkpointPath).pathExtension == "safetensors" }
    var selectionIdentity: String { checkpointPath + "|" + sha256 }
    var title: String {
        displayName ?? (URL(fileURLWithPath: checkpointPath).deletingLastPathComponent().lastPathComponent
            + " · " + URL(fileURLWithPath: checkpointPath).lastPathComponent)
    }
    static let changeNotification = Notification.Name("org.ipde.texture-studio.material-selection-changed")
    static var registryURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/selected-material-checkpoint.json")
    }
    static func registryURL(for target: String, heightRegistryURL: URL = registryURL) -> URL {
        target == "height" ? heightRegistryURL : heightRegistryURL.deletingPathExtension().appendingPathExtension(target + ".json")
    }
    static func readAll(heightRegistryURL: URL = registryURL) -> [String: Self] {
        Dictionary(uniqueKeysWithValues: ["height", "roughness", "normal"].compactMap { target in
            guard let checkpoint = try? read(from: registryURL(for: target, heightRegistryURL: heightRegistryURL)),
                  checkpoint.target == target, checkpoint.supportsStudioInference else { return nil }
            return (target, checkpoint)
        })
    }
    static func read(from url: URL = registryURL) throws -> Self { try JSONDecoder().decode(Self.self, from: Data(contentsOf: url)) }

    func save(to requestedURL: URL? = nil) throws {
        let url = requestedURL ?? Self.registryURL(for: target)
        try Self.withRegistryLock(at: url) {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            try encoder.encode(self).write(to: url, options: .atomic)
        }
        if ["height", "roughness", "normal"].contains(where: { url.standardizedFileURL == Self.registryURL(for: $0).standardizedFileURL }) {
            let information = ["selectionID": UUID().uuidString, "target": target]
            NotificationCenter.default.post(name: Self.changeNotification, object: nil, userInfo: information)
            DistributedNotificationCenter.default().postNotificationName(Self.changeNotification, object: nil, userInfo: information, deliverImmediately: true)
        }
    }

    /// Runtime changes reconnect the existing selected model. They never select
    /// a library row or change its checkpoint identity as a side effect.
    static func refreshRuntime(workspacePath: String, modelDirectory: String, at url: URL = registryURL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        try withRegistryLock(at: url) {
            let raw = try Data(contentsOf: url)
            let selected = try JSONDecoder().decode(Self.self, from: raw)
            guard selected.workspacePath != workspacePath || selected.modelDirectory != modelDirectory else { return }
            guard var document = try JSONSerialization.jsonObject(with: raw) as? [String: Any] else {
                throw CocoaError(.fileReadCorruptFile)
            }
            // Preserve any newer metadata fields alongside the exact selected
            // checkpoint path, SHA256 and target.
            document["workspacePath"] = workspacePath
            document["modelDirectory"] = modelDirectory
            try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys])
                .write(to: url, options: .atomic)
        }
    }

    private static func withRegistryLock(at url: URL, body: () throws -> Void) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let descriptor = Darwin.open(url.appendingPathExtension("lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { Darwin.close(descriptor) }
        guard flock(descriptor, LOCK_EX) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { flock(descriptor, LOCK_UN) }
        try body()
    }
}

struct WorkbenchTrainingCapabilities: Decodable, Sendable {
    let trainingSizes: [Int]
}
struct WorkbenchTrainingResponse: Decodable, Sendable {
    let checkpointPath: String
    let packagePath: String?
    let status: String?
    let stoppedReason: String?
    let completedUpdates: Int?
    let requestedUpdates: Int?
}
struct WorkbenchAdapterWeight: Identifiable {
    let id = UUID()
    var path: String
    var weight = 1.0
}
struct WorkbenchHubModel: Decodable, Identifiable, Sendable {
    let repository: String
    let revision: String?
    let target: String?
    var id: String { repository }
}
struct WorkbenchHubModels: Decodable, Sendable {
    let models: [WorkbenchHubModel]
}

struct WorkbenchDatasetCleanup: Decodable {
    let datasetPath: String
    let sourceDatasetPath: String?
    let removed: Bool
}
