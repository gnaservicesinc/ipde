import AppKit
import Observation
import UniformTypeIdentifiers

@MainActor @Observable
final class WorkbenchStore {
    var dataset: WorkbenchDataset?
    var datasetSheet: DatasetSheetRoute?
    var showNewDatasetSheet: Bool {
        get { datasetSheet == .new }
        set { if newValue { datasetSheet = .new } else if datasetSheet == .new { datasetSheet = nil } }
    }
    var showAddMaterialSheet: Bool {
        get { datasetSheet == .add }
        set { if newValue { datasetSheet = .add } else if datasetSheet == .add { datasetSheet = nil } }
    }
    var showDatasetInfoSheet: Bool {
        get { datasetSheet == .info }
        set { if newValue { datasetSheet = .info } else if datasetSheet == .info { datasetSheet = nil } }
    }
    var showImportFolderSheet: Bool {
        get { datasetSheet == .folder }
        set { if newValue { datasetSheet = .folder } else if datasetSheet == .folder { datasetSheet = nil } }
    }
    var pendingSourceFolder: URL?
    var folderImport: WorkbenchFolderImport?
    var folderImportSize = 1024
    var isScanningFolder = false
    var folderImportURL: URL?
    @ObservationIgnored var folderImportPlanURL: URL?
    @ObservationIgnored private var queuedDatasetURL: URL?
    @ObservationIgnored private var queuedTrainingHandoff: MaterialTrainingHandoff?
    @ObservationIgnored private var queuedResumeURL: URL?
    var showTrashDatasetConfirmation = false
    var recentDatasets: [WorkbenchDatasetLocation] = []
    var selectedSampleId: String? { didSet { saveUserSettings() } }
    var selectedRole = "height" { didSet { saveUserSettings() } }
    var selectedInputVariantId: String? { didSet { saveUserSettings() } }
    var checkpoints: [WorkbenchCheckpoint] = []
    var selectedCheckpointId: String? { didSet { saveUserSettings() } }
    var comparisonCheckpointIds: Set<String> = [] { didSet { saveUserSettings() } }
    var comparisonIncludesBase = true { didSet { saveUserSettings() } }
    var training = MaterialTrainingOptions() { didSet { saveUserSettings() } }
    var trainingNavigationRequest = UUID()
    var developerMode: Bool {
        get { StudioPreferences.defaults.bool(forKey: StudioPreferences.developerModeKey) }
        set { StudioPreferences.defaults.set(newValue, forKey: StudioPreferences.developerModeKey) }
    }
    var uploadAfterTraining = true { didSet { preferences.set(uploadAfterTraining, forKey: "uploadAfterTraining") } }
    private var configuredPreparationWorkers: Int
    var preparationWorkers: Int {
        get { configuredPreparationWorkers }
        set {
            configuredPreparationWorkers = min(resources.availableProcessorCount, max(1, newValue))
            preferences.set(configuredPreparationWorkers, forKey: "preparationWorkers")
        }
    }
    private(set) var supportedTrainingSizes: [Int] = [256, 512, 1024, 2048]
    private var backendTrainingSizes: [Int] = [256, 512, 1024, 2048]
    private(set) var hubModels: [WorkbenchHubModel] = []
    var adapterMix: [WorkbenchAdapterWeight] = []
    var activity = ""
    var logText = ""
    var error: String?
    private(set) var isBusy = false
    private(set) var isTraining = false
    private(set) var isResumingTraining = false
    private(set) var isStopping = false
    private(set) var hasTrainingStarted = false
    private(set) var isSavingTraining = false
    private(set) var isCheckpointPending = false
    private(set) var validationSummary = ""
    private(set) var trainingProgress: WorkbenchTrainingProgress?
    @ObservationIgnored private var trainingEventBuffer = ""
    private(set) var isPreparingDataset = false
    var datasetPreparationSummary = ""
    private(set) var trainingPreferenceNotice: String?
    var lastOutputURL: URL? { didSet { saveUserSettings() } }
    var lastLogURL: URL? { didSet { saveUserSettings() } }
    var lastPackageURL: URL? { didSet { saveUserSettings() } }
    var lastPackageCheckpointId: String? { didSet { saveUserSettings() } }
    var sourceImageURL: URL? { didSet { saveUserSettings() } }
    var testPhotoSettings = TextureSettings()
    var comparisonCandidates: [MapReviewCandidate] = []
    var uploadRepo = "" { didSet { saveUploadConfiguration() } }
    var uploadPublic = false { didSet { saveUploadConfiguration() } }
    private(set) var uploadAccount: String?
    private(set) var uploadAccountMessage = "Checking your saved Hugging Face login…"
    private(set) var uploadAccountChecked = false
    private(set) var lastUploadURL: URL?
    var workspacePath: String { didSet { preferences.set(workspacePath, forKey: "workspace") } }
    var modelDirectory: String { didSet { preferences.set(modelDirectory, forKey: "materialModelDirectory") } }
    let resources: MachineResources
    @ObservationIgnored private var runner: NativeMaterialTrainingControl?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var activeWorkerId: UUID?
    @ObservationIgnored let preferences: UserDefaults
    @ObservationIgnored let trashHandler: (URL) throws -> Void
    @ObservationIgnored private let workerOverride: (@MainActor ([String], String) async throws -> String)?
    @ObservationIgnored private let managedWorkspaceURL: URL
    @ObservationIgnored private let selectedCheckpointRegistryURL: URL
    @ObservationIgnored private var hasRestored = false
    @ObservationIgnored private var materialBaseLocations: [String: String] {
        didSet { preferences.set(materialBaseLocations, forKey: "nativeMaterialBaseLocations") }
    }
    @ObservationIgnored private var isRestoringPreferences = false

    var samples: [WorkbenchSample] { dataset?.samples ?? [] }
    var canStopAndSave: Bool { isTraining && hasTrainingStarted && !isStopping }
    var canAbort: Bool { isBusy && (!isStopping || isSavingTraining) }
    var selectedSample: WorkbenchSample? { samples.first { $0.id == selectedSampleId } }
    var selectedMaterialId: String? { dataset?.materials.first { $0.samples.contains { $0.id == selectedSampleId } }?.materialId }
    var selectedMaterialName: String? {
        dataset?.materials.first { $0.samples.contains { $0.id == selectedSampleId } }.map { $0.name ?? $0.materialId.replacingOccurrences(of: "_", with: " ") }
    }
    var selectedCheckpoint: WorkbenchCheckpoint? { checkpoints.first { $0.id == selectedCheckpointId } }
    var comparisonBaselineLabel: String {
        checkpoints.first { comparisonCheckpointIds.contains($0.id) }?.modelFamily?.isCompact == true ?
            "Include untrained compact initialization" : "Include the material base before refinement"
    }
    var selectedDiffuseMap: WorkbenchMap? {
        selectedSample?.inputVariants?.first { $0.variantId == selectedInputVariantId } ?? selectedSample?.inputVariants?.first ?? selectedSample?.maps["input"]
    }
    var selectedMap: WorkbenchMap? { selectedRole == "input" ? selectedDiffuseMap : selectedSample?.maps[selectedRole] }
    func datasetReviewURL(_ map: WorkbenchMap) -> URL {
        map.originalSourcePath.map { URL(fileURLWithPath: $0) } ?? map.url
    }
    func datasetReviewSHA256(_ map: WorkbenchMap) -> String? { map.originalSourceSha256 ?? map.sha256 }
    func datasetDisplayTransform(_ map: WorkbenchMap, role: String) -> MapReviewDisplayTransform? {
        let width = map.originalSourceWidth ?? map.width
        let height = map.originalSourceHeight ?? map.height
        // Sources too small for the chosen training grid remain manageable and
        // display at their original size instead of requesting an invalid crop.
        if let width, let height, min(width, height) < training.size { return nil }
        let convention = map.originalNormalConvention ?? map.sourceNormalConvention ?? "opengl"
        guard let hash = datasetReviewSHA256(map),
              map.cropRectangle != nil || width != training.size || height != training.size || convention == "directx" else { return nil }
        let rectangle = map.cropRectangle ?? width.flatMap { w in height.map { h in
            [max(0, (w - training.size) / 2), max(0, (h - training.size) / 2), training.size, training.size]
        } }
        return MapReviewDisplayTransform(size: training.size, sourceSHA256: hash,
            algorithm: MapReviewDisplayTransform.exactCrop, mapType: role,
            normalConvention: convention, cropRectangle: rectangle)
    }

    var datasetURL: URL? { dataset.map { URL(fileURLWithPath: $0.datasetPath) } }
    var datasetDisplayURL: URL? {
        guard let canonical = datasetURL else { return nil }
        guard let saved = preferences.string(forKey: "dataset") else { return canonical }
        let located = URL(fileURLWithPath: saved)
        return located.resolvingSymlinksInPath().standardizedFileURL == canonical.resolvingSymlinksInPath().standardizedFileURL
            ? located : canonical
    }
    var datasetNativeSizeLabel: String {
        guard let dataset, let first = dataset.samples.first?.maps.values.first,
              let width = first.width, let height = first.height else { return "Map dimensions not verified" }
        let matching = dataset.samples.allSatisfy { !$0.maps.isEmpty && $0.maps.values.allSatisfy { $0.width == width && $0.height == height } }
        return matching ? "\(width.formatted()) × \(height.formatted()) native maps" : "Mixed native map sizes"
    }
    var workspaceURL: URL { URL(fileURLWithPath: workspacePath).standardizedFileURL }
    var dependencyArguments: [String] { ["--model-directory", modelDirectory] }
    func dependencyArguments(for checkpoint: WorkbenchCheckpoint) -> [String] {
        if checkpoint.modelFamily?.isCompact == true { return [] }
        return ["--model-directory", baseDirectory(for: checkpoint)]
    }

    init(preferences defaults: UserDefaults = UserDefaults(suiteName: "org.ipde.material-tools")!,
         managedWorkspaceURL: URL? = nil,
         resources: MachineResources = .current,
         selectedCheckpointRegistryURL: URL = SelectedMaterialCheckpoint.registryURL,
         trashHandler: @escaping (URL) throws -> Void = { url in try FileManager.default.trashItem(at: url, resultingItemURL: nil) },
         workerOverride: (@MainActor ([String], String) async throws -> String)? = nil) {
        preferences = defaults
        materialBaseLocations = defaults.dictionary(forKey: "nativeMaterialBaseLocations") as? [String: String] ?? [:]
        self.trashHandler = trashHandler
        if let data = defaults.data(forKey: "recentDatasets.v1"),
           let locations = try? JSONDecoder().decode([WorkbenchDatasetLocation].self, from: data) {
            recentDatasets = locations
        }
        uploadRepo = defaults.string(forKey: "uploadRepository") ?? ""
        uploadPublic = defaults.bool(forKey: "uploadPublic")
        uploadAfterTraining = defaults.object(forKey: "uploadAfterTraining") == nil ? true : defaults.bool(forKey: "uploadAfterTraining")
        self.resources = resources
        configuredPreparationWorkers = defaults.object(forKey: "preparationWorkers") == nil
            ? resources.availableProcessorCount
            : min(resources.availableProcessorCount, max(1, defaults.integer(forKey: "preparationWorkers")))
        self.selectedCheckpointRegistryURL = selectedCheckpointRegistryURL
        self.workerOverride = workerOverride
        self.managedWorkspaceURL = (managedWorkspaceURL ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/Material Workspace")).standardizedFileURL
        let local = Bundle.main.object(forInfoDictionaryKey: "IPDEWorkspace") as? String ?? ""
        let configuredWorkspace = defaults.string(forKey: "workspace") ?? local
        let workspace = configuredWorkspace.isEmpty ? self.managedWorkspaceURL.path : URL(fileURLWithPath: configuredWorkspace).standardizedFileURL.path
        workspacePath = workspace
        let cache = URL(fileURLWithPath: workspace).appendingPathComponent("out/material-training/transfer-models")
        let materialBasePath = defaults.string(forKey: "materialModelDirectory") ?? cache.appendingPathComponent("pbrnxt-base").path
        modelDirectory = materialBasePath
        let saved = WorkbenchPreferences.load(from: defaults)
        if let options = saved.training {
            training = options.restored(for: resources)
            if training != options {
                trainingPreferenceNotice = "Saved training settings were adapted to supported values. Review the training settings before starting."
                activity = trainingPreferenceNotice!
            }
        }
        selectedSampleId = saved.selectedSampleId
        selectedInputVariantId = saved.selectedInputVariantId
        if let role = saved.selectedRole, ["input", "height", "roughness", "normal"].contains(role) { selectedRole = role }
        selectedCheckpointId = saved.selectedCheckpointId
        comparisonCheckpointIds = saved.comparisonCheckpointIds ?? []
        comparisonIncludesBase = saved.comparisonIncludesBase ?? true
        sourceImageURL = saved.sourceImagePath.map { URL(fileURLWithPath: $0) }
        lastOutputURL = saved.lastOutputPath.map { URL(fileURLWithPath: $0) }
        lastLogURL = saved.lastLogPath.map { URL(fileURLWithPath: $0) }
        lastPackageURL = saved.lastPackagePath.map { URL(fileURLWithPath: $0) }
        lastPackageCheckpointId = saved.lastPackageCheckpointId
    }

    /// Save changes as they happen; closing a window or the app is not a save
    /// boundary, and starting a worker must not be required to retain a choice.
    private func saveUserSettings() {
        guard !isRestoringPreferences else { return }
        WorkbenchPreferences(training: training, selectedSampleId: selectedSampleId, selectedRole: selectedRole,
            selectedInputVariantId: selectedInputVariantId,
            selectedCheckpointId: selectedCheckpointId, comparisonCheckpointIds: comparisonCheckpointIds,
            comparisonIncludesBase: comparisonIncludesBase, sourceImagePath: sourceImageURL?.path,
            lastOutputPath: lastOutputURL?.path, lastLogPath: lastLogURL?.path,
            lastPackagePath: lastPackageURL?.path, lastPackageCheckpointId: lastPackageCheckpointId).save(to: preferences)
    }

    func restore() {
        guard !isBusy, !hasRestored else { return }
        hasRestored = true
        let args = CommandLine.arguments
        var datasetPath = preferences.string(forKey: "dataset")
        if datasetPath == nil {
            let localDataset = workspaceURL.deletingLastPathComponent().appendingPathComponent("material-dataset/dataset.json")
            if FileManager.default.fileExists(atPath: localDataset.path) { datasetPath = localDataset.path }
        }
        if let i = args.firstIndex(of: "--dataset"), args.indices.contains(i + 1) { datasetPath = args[i + 1] }
        let paths = preferences.stringArray(forKey: "checkpoints") ?? []
        let saved = WorkbenchPreferences.load(from: preferences)
        operation("Opening workspace…") {
            do { try await self.loadTrainingCapabilities() }
            catch { self.logText += "Training setup is unavailable: \(error.localizedDescription)\n" }
            self.isRestoringPreferences = true
            defer { self.isRestoringPreferences = false; self.saveUserSettings() }
            if let datasetPath, FileManager.default.fileExists(atPath: datasetPath) {
                let url = URL(fileURLWithPath: datasetPath)
                var directory: ObjCBool = false
                if FileManager.default.fileExists(atPath: datasetPath, isDirectory: &directory), directory.boolValue,
                   !FileManager.default.fileExists(atPath: url.appendingPathComponent("dataset.json").path),
                   !(url.lastPathComponent == "sources" && FileManager.default.fileExists(atPath: url.deletingLastPathComponent().appendingPathComponent("dataset.json").path)) {
                    self.pendingSourceFolder = url
                    self.showNewDatasetSheet = true
                } else { try await self.loadDataset(url) }
            }
            for path in paths where FileManager.default.fileExists(atPath: path) {
                do { try await self.loadCheckpoint(URL(fileURLWithPath: path), select: false) }
                catch { self.logText += "Could not reconnect \(path): \(error.localizedDescription)\n" }
            }
            if !self.checkpoints.contains(where: { $0.id == self.selectedCheckpointId }) {
                self.selectedCheckpointId = self.checkpoints.first?.id
            }
            let available = Set(self.checkpoints.map(\.id))
            self.comparisonCheckpointIds = saved.comparisonCheckpointIds.map { $0.intersection(available) } ?? available
        }
    }

    func saveConfiguration(refreshSelectedRuntime: Bool = true) {
        preferences.set(workspacePath, forKey: "workspace")
        preferences.set(modelDirectory, forKey: "materialModelDirectory")
        saveUploadConfiguration()
        if refreshSelectedRuntime {
            do {
                for target in ["height", "roughness", "normal"] {
                    let registry = SelectedMaterialCheckpoint.registryURL(for: target, heightRegistryURL: selectedCheckpointRegistryURL)
                    let selectedBase = (try? SelectedMaterialCheckpoint.read(from: registry))?.modelDirectory
                    let base = selectedBase.flatMap { materialBaseLocations.values.contains($0) ? $0 : nil } ?? modelDirectory
                    try SelectedMaterialCheckpoint.refreshRuntime(workspacePath: workspacePath,
                        modelDirectory: base, at: registry)
                }
            } catch {
                self.error = "Runtime settings were saved, but Texture Studio could not reconnect its selected checkpoint: \(error.localizedDescription)"
            }
        }
    }

    func chooseDataset() {
        guard !isBusy else { return }
        let panel = NSOpenPanel()
        panel.title = "Open Dataset"
        panel.message = "Open a saved dataset or choose a folder of material maps to set up and import. Subfolders are scanned automatically."
        panel.prompt = "Open Dataset"
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.folder, .json]
        panel.directoryURL = datasetFolderURL?.deletingLastPathComponent()
        panel.begin { response in
            if response == .OK, let url = panel.url { self.openDataset(url) }
        }
    }
    func openDataset(_ url: URL) {
        guard !isBusy else { error = "Stop the current operation before changing datasets."; return }
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue,
           !FileManager.default.fileExists(atPath: url.appendingPathComponent("dataset.json").path),
           !(url.lastPathComponent == "sources" && FileManager.default.fileExists(atPath: url.deletingLastPathComponent().appendingPathComponent("dataset.json").path)) {
            if dataset != nil { importMaterialFolder(url) }
            else { pendingSourceFolder = url; showNewDatasetSheet = true }
            return
        }
        operation("Reading dataset…") {
            try await self.loadDataset(url)
            self.activity = "Opened \(self.datasetName)."
        }
    }
    func receiveDataset(_ url: URL) {
        if isBusy { queuedDatasetURL = url }
        else { openDataset(url) }
    }
    func openCheckpoint(_ url: URL) {
        guard !isBusy else { error = "Stop the current operation before changing checkpoints."; return }
        operation("Inspecting checkpoint…") { try await self.loadCheckpoint(url) }
    }
    func chooseCheckpoint() {
        let panel = NSOpenPanel(); panel.allowsMultipleSelection = true
        panel.title = "Choose material checkpoints"; panel.prompt = "Inspect Checkpoints"
        panel.begin { response in
            guard response == .OK else { return }
            self.operation("Inspecting checkpoints…") {
                for url in panel.urls { try await self.loadCheckpoint(url) }
            }
        }
    }
    func chooseSourceImage() {
        chooseFile(types: [.image], title: "Choose a surface photo to prepare as diffuse for every checkpoint") { self.sourceImageURL = $0 }
    }
    func chooseWorkspace() {
        chooseFolder(title: "Choose a working folder for material runs") { url in self.workspacePath = url.path; self.saveConfiguration() }
    }
    func chooseEncoder() {
        guard !isBusy else { return }
        let panel = NSOpenPanel(); panel.title = "Locate material weights or their folder"
        panel.canChooseFiles = true; panel.canChooseDirectories = true
        panel.begin { response in
            if response == .OK, let url = panel.url { self.modelDirectory = url.path; self.saveConfiguration() }
        }
    }

    func loadDataset(_ url: URL) async throws {
        let result: WorkbenchDataset = try WorkbenchResult.decode(WorkbenchDataset.self,
            output: await worker(["dataset", "--dataset", url.path, "--target", training.target,
                                  "--default-review-size", String(training.size)]))
        adoptDataset(result)
        datasetPreparationSummary = ""
        saveDatasetLocation(result, requested: url)
    }

    func saveDatasetLocation(_ result: WorkbenchDataset, requested: URL? = nil) {
        let canonical = URL(fileURLWithPath: result.datasetPath)
        let located = requested ?? preferences.string(forKey: "dataset").map { URL(fileURLWithPath: $0) }
        let selected = located.flatMap {
            $0.resolvingSymlinksInPath().standardizedFileURL == canonical.resolvingSymlinksInPath().standardizedFileURL ? $0 : nil
        } ?? canonical
        preferences.set(selected.path, forKey: "dataset")
        rememberDataset(result)
    }

    func adoptDataset(_ result: WorkbenchDataset, preferredMaterial: String? = nil) {
        let material = preferredMaterial ?? selectedMaterialId
        dataset = result
        if let size = result.trainingSize {
            training.size = size
        }
        supportedTrainingSizes = backendTrainingSizes.filter { result.supportedTrainingSizes?.contains($0) ?? true }
        if !result.samples.contains(where: { $0.id == selectedSampleId }) {
            selectedSampleId = result.materials.first(where: { $0.id == material })?.samples.first?.id ?? result.samples.first?.id
        }
    }

    func selectTrainingSize(_ size: Int) {
        guard !isBusy else { return }
        guard supportedTrainingSizes.contains(size) else { error = "Choose a training size supplied by the original source maps."; return }
        guard dataset != nil else { training.size = size; return }
        updateDatasetInfo(name: datasetName, description: datasetDescription, size: size)
    }

    func selectTrainingTarget(_ target: String) {
        guard !isBusy, ["height", "roughness", "normal"].contains(target) else { return }
        training.target = target
        if training.modelFamily.isCompact {
            training.modelFamily = target == "normal" ? .compactNormal : .compactScalar
        }
        clearIncompatibleTrainingCheckpoint()
    }

    func selectTrainingModelFamily(_ family: MaterialTrainingModelFamily) {
        guard !isBusy, family != training.modelFamily else { return }
        let previous = training.modelFamily
        training.modelFamily = family
        if previous.isCompact != family.isCompact { training.learningRate = family.defaultLearningRate }
        if !family.supports(target: training.target) { training.target = family == .compactNormal ? "normal" : "height" }
        if !family.isCompact, !["final-map", "map-decoder"].contains(training.scope) { training.scope = "final-map" }
        clearIncompatibleTrainingCheckpoint()
    }

    private func clearIncompatibleTrainingCheckpoint() {
        if let checkpoint = selectedCheckpoint, !checkpoint.matchesTraining(training) {
            selectedCheckpointId = nil
            training.useWarmStart = false
        } else if selectedCheckpoint == nil { training.useWarmStart = false }
    }

    func prepareTrainingDataset() {
        guard !isBusy, dataset != nil else { return }
        let options = training
        let size = options.size
        let selectedMaterial = selectedMaterialId
        let material = options.useSelectedMaterialOnly ? selectedMaterial : nil
        guard dataset?.readyForTraining(size: size, material: material, target: options.target) != true else { return }
        operation("Preparing \(size) × \(size) complete training maps…") {
            _ = try await self.ensureTrainingDataset(options: options, selectedMaterial: selectedMaterial)
            self.activity = self.datasetPreparationSummary
        }
    }

    private func ensureTrainingDataset(options: MaterialTrainingOptions, selectedMaterial: String?) async throws -> WorkbenchDataset {
        let size = options.size
        let target = options.target
        guard let current = dataset else { throw StudioError("Open a material dataset first.") }
        guard supportedTrainingSizes.contains(size) else { throw StudioError("Choose a training size supported by the model and original source maps.") }
        let checkMaterial = options.useSelectedMaterialOnly ? selectedMaterial : nil
        if current.readyForTraining(size: size, material: checkMaterial, target: target) { return current }
        isPreparingDataset = true
        activity = "Preparing \(size) × \(size) complete training maps from the original materials…"
        defer { isPreparingDataset = false }
        var arguments = [
            "prepare-size", "--dataset", current.datasetPath, "--size", String(size), "--target", target,
            "--automatic-validation", "--expected-index-sha256", current.indexSha256]
        if let reviewHash = current.reviewSha256 { arguments += ["--expected-review-sha256", reviewHash] }
        if let checkMaterial { arguments += ["--material", checkMaterial] }
        let result: WorkbenchDataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: await worker(arguments))
        guard result.readyForTraining(size: size, material: checkMaterial, target: target), let preparation = result.preparation,
              preparation.cropSize == size, !preparation.originalDatasetModified,
              URL(fileURLWithPath: preparation.preparedDatasetPath).standardizedFileURL == URL(fileURLWithPath: result.datasetPath).standardizedFileURL else {
            throw StudioError("Dataset preparation did not verify matching map dimensions and preserved originals.")
        }
        try Task.checkCancellation()
        adoptDataset(result, preferredMaterial: selectedMaterial)
        preferences.set(preparation.sourceDatasetPath, forKey: "dataset")
        datasetPreparationSummary = "\(size) × \(size) native pixel maps: matching originals are referenced directly; each crop is saved once in temporary training storage."
        return result
    }
    func loadCheckpoint(_ url: URL, select: Bool = true) async throws {
        let output = try await worker(["checkpoint", "--checkpoint", url.path])
        let checkpoint = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: output)
        if checkpoint.variant == "full" { materialBaseLocations[checkpoint.sha256] = checkpoint.checkpointPath }
        guard checkpoint.compatible else { throw StudioError("This checkpoint is not supported by the material backend.") }
        if let i = checkpoints.firstIndex(where: { $0.id == checkpoint.id }) { checkpoints[i] = checkpoint }
        else { checkpoints.append(checkpoint) }
        if select {
            selectedCheckpointId = checkpoint.id
            comparisonCheckpointIds.insert(checkpoint.id)
        }
        if !isRestoringPreferences { preferences.set(checkpoints.map(\.checkpointPath), forKey: "checkpoints") }
    }

    func baseDirectory(for checkpoint: WorkbenchCheckpoint) -> String {
        checkpoint.base.flatMap { materialBaseLocations[$0.sha256] } ?? modelDirectory
    }

    func curateSelected(status: String, split: String? = nil, note: String? = nil) {
        guard let dataset, let sample = selectedSample else { return }
        operation("Saving material review…") {
            var args = ["curate", "--dataset", dataset.datasetPath, "--sample", sample.id,
                        "--status", status, "--expected-index-sha256", dataset.indexSha256, "--review-size", String(self.training.size)]
            if let reviewHash = dataset.reviewSha256 { args += ["--expected-review-sha256", reviewHash] }
            if let split { args += ["--split", split] }
            if let note { args += ["--note", note] }
            _ = try await self.worker(args)
            try await self.loadDataset(URL(fileURLWithPath: dataset.datasetPath))
        }
    }
    func inspectSelectedMap() {
        guard let map = selectedMap else { return }
        ReviewWindowController.shared.open(candidates: [MapReviewCandidate(id: map.path, label: "\(selectedSampleId ?? "Material") · \(selectedRole)",
            mapURL: datasetReviewURL(map), numeric: selectedRole != "input",
            sourceIdentity: MapReviewSourceIdentity(path: datasetReviewURL(map).path, sha256: datasetReviewSHA256(map)),
            displayTransform: datasetDisplayTransform(map, role: selectedRole))])
    }

    func compare() {
        if let issue = comparisonConfigurationIssue { error = issue; return }
        let diffuseMap = sourceImageURL == nil ? selectedDiffuseMap : nil
        guard let image = sourceImageURL ?? diffuseMap.map(datasetReviewURL) else {
            error = "Choose a surface photo or select a dataset diffuse map first."; return
        }
        let selected = checkpoints.filter { comparisonCheckpointIds.contains($0.id) }
        let includeBase = comparisonIncludesBase
        guard !selected.isEmpty, selected.count + (includeBase ? 1 : 0) >= 2, Set(selected.map(\.target)).count == 1 else {
            error = "Select one checkpoint plus its base, or two checkpoints predicting the same map type."; return
        }
        let target = selected[0].target
        let sampleLabel = sourceImageURL == nil ? selectedSampleId ?? image.lastPathComponent : image.lastPathComponent
        let referenceMap = sourceImageURL == nil ? selectedSample?.maps[target] : nil
        let referenceSampleID = sourceImageURL == nil ? selectedSampleId : nil
        let diffuseTransform = diffuseMap.flatMap { datasetDisplayTransform($0, role: "input") }
        let referenceTransform = referenceMap.flatMap { datasetDisplayTransform($0, role: target) }
        let referenceURL = referenceMap.map(datasetReviewURL)
        let photoSettings = testPhotoSettings
        let comparisonSize = training.size
        let inputIsPhoto = sourceImageURL != nil
        let targetLabel = target == "height" ? "Displacement" : target.capitalized
        operation("Comparing \(selected.count + (includeBase ? 1 : 0)) models on the \(comparisonSize) × \(comparisonSize) diffuse grid…") {
            let parent = try self.newOutputURL(prefix: "comparison")
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
            self.lastOutputURL = parent
            self.comparisonCandidates = []
            var modelImage = image
            var reviewedDiffuse = image
            var temporaryInput: URL?
            defer { if let temporaryInput { try? FileManager.default.removeItem(at: temporaryInput) } }
            if inputIsPhoto {
                let engine = TextureEngine()
                let photo = try await engine.importPhoto(image)
                var settings = photoSettings
                settings.outputSize = comparisonSize
                let prepared = try await engine.prepareDiffuse(source: photo, settings: settings)
                modelImage = parent.appendingPathComponent("diffuse.png")
                try await engine.writeDiffuse(prepared.diffuse, to: modelImage)
                reviewedDiffuse = modelImage
            } else if let diffuseTransform {
                let directory = FileManager.default.temporaryDirectory.appendingPathComponent("material-comparison-input-\(UUID().uuidString)")
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
                temporaryInput = directory
                modelImage = directory.appendingPathComponent("diffuse.png")
                var arguments = ["review-source", "--image", image.path, "--expected-sha256", diffuseTransform.sourceSHA256,
                    "--size", String(diffuseTransform.size), "--map-type", "input", "--output", modelImage.path]
                if let rectangle = diffuseTransform.cropRectangle { arguments += ["--source-rectangle"] + rectangle.map(String.init) }
                _ = try await self.worker(arguments)
            }
            var candidates = [MapReviewCandidate(id: "diffuse|" + reviewedDiffuse.path, label: "Diffuse · model input", mapURL: reviewedDiffuse, numeric: false,
                sampleLabel: sampleLabel, detail: reviewedDiffuse.path, role: "diffuse",
                modelIdentity: MapReviewModelIdentity(mapType: "input"), displayTransform: diffuseTransform)]
            if let referenceMap {
                let sourceIdentity: MapReviewSourceIdentity
                if let original = referenceMap.originalSourcePath {
                    sourceIdentity = MapReviewSourceIdentity(path: original, sha256: referenceMap.originalSourceSha256,
                        bits: referenceMap.sourceBits, cropRectangle: referenceMap.cropRectangle, pixelDimensions: referenceMap.originalSourceWidth.flatMap { width in
                            referenceMap.originalSourceHeight.map { [width, $0] } })
                } else {
                    sourceIdentity = await Task.detached {
                        MapReviewSourceIdentity.fromDatasetMap(referenceMap, target: target, sampleID: referenceSampleID)
                    }.value
                }
                candidates.append(MapReviewCandidate(id: "target|" + referenceMap.path, label: "Source \(targetLabel.lowercased()) · reference",
                    mapURL: referenceURL ?? referenceMap.url, numeric: true, sampleLabel: sampleLabel,
                    detail: (["Real source map · not a model output"] + sourceIdentity.recordedDetails).joined(separator: " · "), role: "target",
                    modelIdentity: MapReviewModelIdentity(mapType: target, modelName: "Dataset reference"), sourceIdentity: sourceIdentity,
                    displayTransform: referenceTransform))
            }
            if includeBase, let reference = selected.first {
                let compact = reference.modelFamily?.isCompact == true
                self.activity = compact ? "Running untrained compact initialization" : "Running the material base model"
                let result: MaterialInferenceResponse = try WorkbenchResult.decode(MaterialInferenceResponse.self, output: await self.worker([
                    "infer", "--baseline", "--checkpoint", reference.checkpointPath, "--expected-sha256", reference.sha256,
                    "--image", modelImage.path, "--output", parent.appendingPathComponent("base-untrained").path] + self.dependencyArguments(for: reference)))
                guard result.checkpointSha256 == reference.sha256, let map = result.outputs[target] else {
                    throw StudioError("The base comparison did not match the selected checkpoint architecture and map type.")
                }
                candidates.append(MapReviewCandidate(id: "base|" + reference.id, label: "\(compact ? "Untrained initialization" : "Base") · \(targetLabel)",
                    mapURL: URL(fileURLWithPath: map.path), numeric: true, sampleLabel: sampleLabel,
                    detail: compact ? "Compact model prediction from its recorded random initialization before training" : "Material base prediction before refinement", role: "base",
                    modelIdentity: MapReviewModelIdentity(architecture: reference.trainingBaseLabel,
                        mapType: target, modelName: compact ? "Untrained compact initialization" : "Material base")))
            }
            for (i, checkpoint) in selected.enumerated() {
                self.activity = "Running checkpoint \(i + 1)/\(selected.count): \(checkpoint.title)"
                let child = parent.appendingPathComponent("candidate-\(i + 1)")
                let result: MaterialInferenceResponse = try WorkbenchResult.decode(MaterialInferenceResponse.self, output: await self.worker([
                    "infer", "--checkpoint", checkpoint.checkpointPath, "--expected-sha256", checkpoint.sha256,
                    "--image", modelImage.path, "--output", child.path] + self.dependencyArguments(for: checkpoint)))
                guard result.checkpointSha256 == checkpoint.sha256, let map = result.outputs[checkpoint.target] else {
                    throw StudioError("The comparison did not use the selected checkpoint or map type.")
                }
                candidates.append(MapReviewCandidate(id: checkpoint.id, label: "\(checkpoint.url.deletingLastPathComponent().lastPathComponent) · \(targetLabel)",
                    mapURL: URL(fileURLWithPath: map.path), numeric: true, sampleLabel: sampleLabel,
                    detail: "Trained model prediction · not the real source map · \(checkpoint.trainingBaseLabel) · \(checkpoint.url.lastPathComponent) · step \(checkpoint.step.formatted()) · SHA256 \(checkpoint.sha256.prefix(12))", role: "checkpoint",
                    modelIdentity: MapReviewModelIdentity(checkpointPath: checkpoint.checkpointPath,
                        checkpointSHA256: checkpoint.sha256, checkpointStep: checkpoint.step,
                        architecture: checkpoint.trainingBaseLabel, mapType: checkpoint.target)))
            }
            self.comparisonCandidates = candidates
            self.lastOutputURL = parent
            var material: [String: Any] = ["material_id": sampleLabel, "diffuse": reviewedDiffuse.path,
                    "variants": candidates.filter { $0.role != "diffuse" }.map { candidate in
                        var item: [String: Any] = ["name": candidate.label, "candidate_id": candidate.id, target: candidate.mapURL.path,
                            "sample_label": sampleLabel, "detail": candidate.detail ?? "", "role": candidate.role, "numeric": candidate.numeric]
                        if let identity = candidate.modelIdentity {
                            item.merge(identity.manifestFields) { _, recorded in recorded }
                        }
                        if let identity = candidate.sourceIdentity {
                            item.merge(identity.manifestFields) { _, recorded in recorded }
                        }
                        if let transform = candidate.displayTransform {
                            item.merge(transform.manifestFields) { _, recorded in recorded }
                        }
                        if candidate.role == "base" { item["baseline"] = true }
                        return item
                    }]
            material["diffuse_variant_id"] = diffuseMap?.variantId
            if let diffuseTransform {
                material["diffuse_native_size"] = diffuseTransform.size
                material["diffuse_source_sha256"] = diffuseTransform.sourceSHA256
                material["diffuse_resize_algorithm"] = diffuseTransform.algorithm
                material["diffuse_source_crop_rectangle"] = diffuseTransform.cropRectangle
            }
            let manifest: [String: Any] = ["schema": "texture-studio-material-quality-review-v1", "comparison_target": target,
                "materials": [material]]
            try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
                .write(to: parent.appendingPathComponent("review-manifest.json"), options: .atomic)
        }
    }

    func loadTrainingCapabilities() async throws {
        let result = try WorkbenchResult.decode(WorkbenchTrainingCapabilities.self,
            output: await worker(["capabilities", "--scope", training.effectiveScope, "--model-family", training.modelFamily.rawValue]))
        backendTrainingSizes = result.trainingSizes.filter { [256, 512, 1024, 2048].contains($0) }.sorted()
        supportedTrainingSizes = backendTrainingSizes.filter { dataset?.supportedTrainingSizes?.contains($0) ?? true }
        if dataset?.trainingSize == nil, !supportedTrainingSizes.contains(training.size), let size = supportedTrainingSizes.last {
            training.size = size
        }
    }

    func refreshTrainingCapabilities() {
        guard !isBusy else { return }
        operation("Checking supported training sizes…") {
            let previousSize = self.training.size
            try await self.loadTrainingCapabilities()
            if previousSize != self.training.size, let datasetURL = self.datasetURL { try await self.loadDataset(datasetURL) }
        }
    }

    var suggestedTrainingModelName: String {
        if training.useWarmStart,
           let name = selectedCheckpoint?.modelName?.trimmingCharacters(in: .whitespacesAndNewlines), !name.isEmpty {
            return name
        }
        let map = ["height": "Displacement", "roughness": "Roughness", "normal": "Normals"][training.target] ?? training.target.capitalized
        return "\(datasetName) \(map)"
    }

    var effectiveTrainingModelName: String {
        let name = training.modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        return name.isEmpty ? suggestedTrainingModelName : name
    }

    var trainingConfigurationIssue: String? {
        if let issue = training.configurationIssue { return issue }
        if dataset == nil { return "Open your source dataset first." }
        if !supportedTrainingSizes.contains(training.size) {
            if dataset?.supportedTrainingSizes?.contains(training.size) == false { return "Choose a training size supplied by the original source maps." }
            return "Choose a supported training size."
        }
        if training.useSelectedMaterialOnly && selectedMaterialId == nil { return "Select a material first." }
        let selected = training.useSelectedMaterialOnly ? dataset?.materials.first { $0.id == selectedMaterialId }?.samples ?? [] : samples
        let eligible = selected.filter { self.sampleIsTrainable($0) }
        if eligible.isEmpty { return "No included material provides a registered \(training.target == "height" ? "16-bit displacement" : training.target) target at this size." }
        if eligible.allSatisfy({ $0.split == "validation" && $0.splitAssignment == "manual" }) {
            return "Assign at least one included material to Training in Dataset. All available materials are assigned to Validation."
        }
        if training.useWarmStart {
            guard let checkpoint = selectedCheckpoint, checkpoint.supportsTrainingWarmStart else { return "Select a material checkpoint to refine." }
            if checkpoint.modelFamily != training.modelFamily { return "Choose a starting checkpoint from the selected model family." }
            if training.modelFamily.isCompact, checkpoint.target != training.target { return "Use the compact checkpoint's recorded \(checkpoint.target) target when continuing training." }
            if checkpoint.schema == "texture-studio-material-lora-v1", checkpoint.target != training.target || (checkpoint.scope ?? "final-map") != training.scope {
                return "Use the selected LoRA's \(checkpoint.target) target and \(checkpoint.scope ?? "final-map") scope when refining it."
            }
        }
        return nil
    }

    func sampleIsTrainable(_ sample: WorkbenchSample) -> Bool {
        !["excluded", "rejected"].contains(sample.status) && min(sample.width, sample.height) >= training.size &&
            (sample.availableTargets?.contains(training.target) ?? (sample.maps[training.target] != nil))
    }

    func startTraining() {
        if let issue = trainingConfigurationIssue { error = issue; return }
        runTraining(checkpoint: training.useWarmStart ? selectedCheckpoint : nil)
    }

    private func runTraining(checkpoint: WorkbenchCheckpoint?) {
        let options = training
        let modelName = effectiveTrainingModelName
        let selectedMaterial = selectedMaterialId
        let dependencies = options.modelFamily.isCompact ? [] : checkpoint.map { dependencyArguments(for: $0) } ?? dependencyArguments
        let developer = developerMode
        let publishAfterTraining = developer && uploadAfterTraining
        operation(options.modelFamily.isCompact ? "Training compact material model…" : checkpoint == nil ? "Training material LoRA…" : "Refining material LoRA…", training: true) {
            let prepared = try await self.ensureTrainingDataset(options: options, selectedMaterial: selectedMaterial)
            let output = try self.newOutputURL(prefix: "material-\(options.target)")
            self.lastOutputURL = output
            var args = [checkpoint == nil ? "train" : "refine", "--dataset", prepared.datasetPath,
                "--output", output.path, "--size", String(options.size), "--whole-maps",
                "--model-name", modelName,
                "--target", options.target, "--scope", options.effectiveScope, "--model-family", options.modelFamily.rawValue,
                "--max-minutes", String(options.maxMinutes),
                "--updates-per-map", String(options.updatesPerCrop),
                "--validation-every", String(options.validationEvery), "--checkpoint-every", String(options.checkpointEvery),
                "--validation-unit", options.validationUnit.rawValue, "--checkpoint-unit", options.checkpointUnit.rawValue,
                "--learning-rate", String(options.learningRate),
                "--gradient-accumulation-steps", String(options.gradientAccumulationSteps),
                "--optimizer", options.optimizer,
                "--optimizer-beta1", String(options.optimizerBeta1), "--optimizer-beta2", String(options.optimizerBeta2),
                "--optimizer-epsilon", String(options.optimizerEpsilon), "--weight-decay", String(options.weightDecay),
                "--max-gradient-norm", String(options.maxGradientNorm),
                "--learning-rate-schedule", options.learningRateSchedule,
                "--minimum-learning-rate-ratio", String(options.minimumLearningRateRatio),
                "--warmup-updates", String(options.warmupUpdates), "--seed", String(options.seed)] + dependencies
            if !options.modelFamily.isCompact { args += ["--lora-rank", String(options.loraRank), "--lora-alpha", String(options.loraAlpha)] }
            if developer { args += ["--developer-mode"] }
            if options.useSelectedMaterialOnly, let id = selectedMaterial { args += ["--material", id] }
            if let checkpoint { args += ["--checkpoint", checkpoint.checkpointPath, "--expected-sha256", checkpoint.sha256] }
            self.isResumingTraining = checkpoint != nil
            do {
                let result = try WorkbenchResult.decode(WorkbenchTrainingResponse.self, output: await self.worker(args))
                self.trainingProgress?.finish(state: result.status == "stopped" ? .stopped : .completed,
                    completed: result.completedUpdates, total: result.requestedUpdates)
                try await self.loadCheckpoint(URL(fileURLWithPath: result.checkpointPath))
                self.lastPackageURL = result.packagePath.map { URL(fileURLWithPath: $0) }
                self.lastPackageCheckpointId = self.selectedCheckpointId
                if result.stoppedReason == "time_limit" {
                    let counts = result.completedUpdates.flatMap { completed in
                        result.requestedUpdates.map { " Saved \(completed.formatted()) of \($0.formatted()) updates." }
                    } ?? " Saved the completed updates."
                    self.activity = "Training time limit reached." + counts
                } else if result.stoppedReason == "no_valid_training_samples" {
                    self.activity = result.completedUpdates == 0 ?
                        "No valid training samples remain. Saved the current weights without training." :
                        "No valid training samples remain. Saved \((result.completedUpdates ?? 0).formatted()) completed steps."
                } else if self.isSavingTraining || result.status == "stopped" {
                    self.activity = options.modelFamily.isCompact ? "Stopped and saved compact material model." : "Stopped and saved material LoRA."
                } else {
                    self.activity = "Training finished. Review the material maps before using this model."
                }
                try await self.cleanupTrainingDataset(prepared)
                if publishAfterTraining && !self.isStopping {
                    let account = try WorkbenchResult.decode(HuggingFaceAccountResponse.self, output: await self.worker(["hub-account"]))
                    self.uploadAccount = account.authenticated ? account.username : nil
                    self.uploadAccountChecked = true
                    if let checkpoint = self.selectedCheckpoint, self.canUploadSelectedCheckpoint {
                        try await self.performUpload(checkpoint)
                    } else {
                        self.activity += " Saved locally; sign in to Hugging Face to publish it."
                    }
                }
            } catch {
                // Cancellation must not prevent cleanup of a positively owned stage.
                await Task { @MainActor in try? await self.cleanupTrainingDataset(prepared) }.value
                throw error
            }
        }
    }

    private func cleanupTrainingDataset(_ prepared: WorkbenchDataset) async throws {
        guard prepared.preparation != nil else { return }
        let output = try await worker(["cleanup-size", "--dataset", prepared.datasetPath])
        _ = try WorkbenchResult.decode(WorkbenchDatasetCleanup.self, output: output)
        try await loadDataset(URL(fileURLWithPath: prepared.preparation!.sourceDatasetPath))
        datasetPreparationSummary = "Training files cleared. Original source maps are selected."
    }

    func chooseResumeCheckpoint() {
        chooseFile(title: "Choose material safetensors to refine") { self.resumeTraining(from: $0) }
    }

    func resumeTraining(from url: URL) {
        operation("Opening material checkpoint…") {
            try await self.loadCheckpoint(url)
            if let checkpoint = self.selectedCheckpoint {
                guard checkpoint.supportsTrainingWarmStart else { throw StudioError("This checkpoint does not support material refinement.") }
                self.configureCheckpointForRefinement(checkpoint)
            }
        }
    }

    func receiveResumeCheckpoint(_ url: URL) {
        if isBusy { queuedResumeURL = url }
        else { resumeTraining(from: url) }
    }

    func receiveTrainingHandoff(_ handoff: MaterialTrainingHandoff) {
        if isBusy { queuedTrainingHandoff = handoff; return }
        operation("Opening the selected model in Trainer…") {
            try await self.loadCheckpoint(handoff.checkpointURL, select: false)
            guard let checkpoint = self.checkpoints.first(where: { $0.url.standardizedFileURL == handoff.checkpointURL.standardizedFileURL }),
                  checkpoint.sha256.lowercased() == handoff.checkpointSHA256,
                  checkpoint.supportsTrainingWarmStart else {
                throw StudioError("The requested checkpoint changed or cannot be refined. Choose the model again.")
            }
            guard checkpoint.modelFamily == handoff.training.modelFamily,
                  !handoff.training.modelFamily.isCompact || checkpoint.target == handoff.training.target else {
                throw StudioError("The requested training family or target does not match this compact checkpoint.")
            }
            self.training = handoff.training
            if let dataset = handoff.datasetURL { try await self.loadDataset(dataset) }
            self.training = handoff.training
            if let sample = handoff.sampleID {
                guard self.samples.contains(where: { $0.id == sample }) else {
                    throw StudioError("The requested material is no longer in this dataset. Select it in Dataset before training.")
                }
                self.selectedSampleId = sample
            }
            self.selectedInputVariantId = handoff.inputVariantID
            self.configureCheckpointForRefinement(checkpoint)
            self.activity = "Ready to refine \(checkpoint.title)."
            self.trainingNavigationRequest = UUID()
        }
    }

    func removeMissingSource(_ url: URL, sampleID: String) {
        guard !FileManager.default.fileExists(atPath: url.path), let dataset, !isBusy else { return }
        operation("Updating missing source reference…") {
            _ = try await self.worker([
                "remove-missing", "--dataset", dataset.datasetPath, "--sample", sampleID,
                "--path", url.path, "--expected-index-sha256", dataset.indexSha256, "--review-size", String(self.training.size)])
            try await self.loadDataset(URL(fileURLWithPath: dataset.datasetPath))
        }
    }

    func stop() {
        if isTraining { stopAndSave() }
        else { abort() }
    }

    func abort() {
        guard canAbort else { return }
        isStopping = true
        isSavingTraining = false
        activity = isTraining ? (hasTrainingStarted ? "Aborting training…" : "Aborting training setup…") : "Stopping the operation…"
        runner?.stop()
        task?.cancel()
    }

    func stopAndSave() {
        guard canStopAndSave else {
            if !isStopping { abort() }
            return
        }
        isStopping = true
        isSavingTraining = true
        activity = training.modelFamily.isCompact ? "Finishing the current step and saving the compact material model…" : "Finishing the current update and saving the material LoRA…"
        runner?.stopAndSave()
    }

    func saveCheckpointNow() {
        guard canStopAndSave, !isCheckpointPending else { return }
        isCheckpointPending = true
        activity = "Full validation and checkpoint queued after this update…"
        runner?.saveCheckpoint()
    }

    func recordTrainingProgress(_ chunk: String) {
        trainingEventBuffer += chunk
        while let newline = trainingEventBuffer.firstIndex(of: "\n") {
            let line = String(trainingEventBuffer[..<newline])
            trainingEventBuffer.removeSubrange(...newline)
            guard let data = line.data(using: .utf8),
                  let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let kind = event["event"] as? String else { continue }
            if isTraining {
                if trainingProgress == nil { trainingProgress = WorkbenchTrainingProgress() }
                trainingProgress?.consume(event)
            }
            if kind == "source_recovery_progress" {
                if isBusy, !isStopping, let message = event["message"] as? String { activity = message }
                continue
            }
            if ["preparation_started", "preparation_progress", "preparation_completed"].contains(kind) {
                guard isPreparingDataset, !isStopping,
                      let completed = event["completed"] as? Int, let total = event["total"] as? Int,
                      let size = event["training_size"] as? Int, let workers = event["worker_count"] as? Int,
                      total >= 0, completed >= 0, completed <= total, size > 0, workers > 0 else { continue }
                activity = "Preparing \(size) × \(size) maps: \(completed)/\(total) materials · \(workers) \(workers == 1 ? "worker" : "workers")"
                continue
            }
            guard isTraining else { continue }
            switch kind {
            case "training_started":
                hasTrainingStarted = true
                if !isStopping { activity = training.modelFamily.isCompact ? "Training compact material model…" : "Training material LoRA…" }
            case "training_stopped":
                guard let reason = event["stopped_reason"] as? String,
                      ["time_limit", "no_valid_training_samples"].contains(reason), !isStopping else { continue }
                isStopping = true
                isSavingTraining = true
                activity = reason == "time_limit" ? "Training time limit reached. Validating and saving completed updates…" :
                    "No valid training samples remain. Saving current weights…"
            case "training_setup", "update_started", "operation_progress", "update", "validation_started", "validation_sample", "checkpoint_started", "export_started":
                if let progress = trainingProgress {
                    activity = [progress.currentUpdateSummary, progress.operationDetail.isEmpty ? nil : progress.operationDetail,
                        progress.operationLabel, progress.stageSummary].compactMap { $0 }.joined(separator: " · ")
                }
            case "sample_skipped":
                if !isStopping {
                    let sample = event["sample_id"] as? String ?? "sample"
                    let reason = event["error"] as? String
                    activity = "Skipped \(sample)" + (reason.map { ": \($0)" } ?? "") + " · Training continues…"
                }
            case "checkpoint_saved":
                if let checkpoint = try? WorkbenchResult.decode(WorkbenchCheckpoint.self, output: line) {
                    if !checkpoints.contains(where: { $0.id == checkpoint.id }) { checkpoints.append(checkpoint) }
                    preferences.set(checkpoints.map(\.checkpointPath), forKey: "checkpoints")
                }
                isCheckpointPending = false
                if !isStopping { activity = "Checkpoint saved. Training continues…" }
            case "validation":
                let count = event["sample_count"] as? Int ?? 0
                let pool = event["pool_count"] as? Int ?? 0
                let skipped = max(0, event["validation_skipped_sample_count"] as? Int ?? 0)
                let label = event["scope"] as? String == "full" ? "Full validation" : "Quick check"
                let skippedSummary = skipped > 0 ? " · \(skipped) skipped" : ""
                if let error = event["mae"] as? Double {
                    validationSummary = "\(label): \(count)/\(pool) crops · error \(error.formatted(.number.precision(.fractionLength(6))))\(skippedSummary)"
                } else if event["status"] as? String == "unavailable" {
                    validationSummary = "\(label) unavailable: \(count)/\(pool) crops\(skippedSummary). Training and saving continue."
                } else { validationSummary = "" }
            default: break
            }
        }
        if trainingEventBuffer.count > 100000 { trainingEventBuffer = String(trainingEventBuffer.suffix(100000)) }
    }

    func useSelectedInStudio() {
        guard let checkpoint = selectedCheckpoint, checkpoint.supportsStudioInference else { return }
        do {
            try SelectedMaterialCheckpoint(checkpointPath: checkpoint.checkpointPath, sha256: checkpoint.sha256,
                target: checkpoint.target, workspacePath: workspacePath,
                modelDirectory: baseDirectory(for: checkpoint),
                displayName: checkpoint.title, modelSummary: checkpoint.modelSummary)
                .save(to: SelectedMaterialCheckpoint.registryURL(for: checkpoint.target, heightRegistryURL: selectedCheckpointRegistryURL))
            activity = "Selected \(checkpoint.title) for Texture Studio."
        } catch { self.error = error.localizedDescription }
    }
    func exportSelectedCheckpoint() {
        guard let checkpoint = selectedCheckpoint else { return }
        chooseFolder(title: "Choose a folder for a new model package") { parent in
            self.operation("Exporting selected model package…") {
                let destination = parent.appendingPathComponent("material-\(checkpoint.target)-\(UUID().uuidString.prefix(8))")
                let args = self.checkpointPackageArguments(for: checkpoint, output: destination, developer: self.developerMode)
                _ = try await self.worker(args)
                self.lastPackageURL = destination
                self.lastPackageCheckpointId = checkpoint.id
                NSWorkspace.shared.activateFileViewerSelecting([destination])
            }
        }
    }
    var suggestedUploadRepo: String {
        guard let uploadAccount, let checkpoint = selectedCheckpoint else { return "" }
        return HuggingFaceUpload.repository(account: uploadAccount, checkpoint: checkpoint)
    }
    var effectiveUploadRepo: String {
        let configured = uploadRepo.trimmingCharacters(in: .whitespacesAndNewlines)
        return configured.isEmpty ? suggestedUploadRepo : configured
    }
    var canUploadSelectedCheckpoint: Bool {
        selectedCheckpoint?.compatible == true && uploadAccount != nil && HuggingFaceUpload.validRepository(effectiveUploadRepo)
    }
    func saveUploadConfiguration() {
        preferences.set(uploadRepo, forKey: "uploadRepository")
        preferences.set(uploadPublic, forKey: "uploadPublic")
    }
    func refreshUploadAccount() {
        guard !isBusy else { return }
        operation("Checking saved Hugging Face account…") {
            let result = try WorkbenchResult.decode(HuggingFaceAccountResponse.self,
                output: await self.worker(["hub-account"]))
            self.uploadAccount = result.authenticated ? result.username : nil
            self.uploadAccountChecked = true
            self.uploadAccountMessage = result.message
            self.activity = result.message
            if result.authenticated {
                let models = try WorkbenchResult.decode(WorkbenchHubModels.self, output: await self.worker(["hub-models"]))
                self.hubModels = models.models
            }
        }
    }
    func uploadPackage() {
        guard let checkpoint = selectedCheckpoint else { error = "Select a model checkpoint to upload."; return }
        let repository = effectiveUploadRepo
        guard uploadAccount != nil else {
            uploadAccountMessage = "Save a Hugging Face token in settings, then click Refresh Account."
            return
        }
        guard HuggingFaceUpload.validRepository(repository) else {
            error = "Enter a Hugging Face repository as owner/model-name."; return
        }
        saveUploadConfiguration()
        operation("Packaging and uploading \(checkpoint.title)…") { try await self.performUpload(checkpoint) }
    }
    private func performUpload(_ checkpoint: WorkbenchCheckpoint) async throws {
        let repository = effectiveUploadRepo, isPublic = uploadPublic
        let package = try newOutputURL(prefix: "upload-\(checkpoint.target)")
        defer { try? FileManager.default.removeItem(at: package) }
        var args = ["upload-selected", "--checkpoint", checkpoint.checkpointPath,
            "--expected-sha256", checkpoint.sha256, "--output", package.path, "--repo", repository]
        if isPublic { args += ["--public"] }
        if developerMode { args += ["--developer-mode"] }
        args += dependencyArguments(for: checkpoint)
        let result = try WorkbenchResult.decode(HuggingFaceUploadResponse.self, output: await worker(args))
        guard result.sourceCheckpointSha256 == checkpoint.sha256, result.repository == repository,
              result.private == !isPublic else {
            throw StudioError("The upload response did not match the selected checkpoint and destination. See the operation log.")
        }
        let existingPackage = checkpoint.url.deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: existingPackage.appendingPathComponent("config.json").path) {
            lastPackageURL = existingPackage
            lastPackageCheckpointId = checkpoint.id
        }
        lastUploadURL = URL(string: result.commitUrl ?? result.url)
        activity = "Uploaded \(checkpoint.title) to \(repository) (\(isPublic ? "public" : "private"))."
        let models = try WorkbenchResult.decode(WorkbenchHubModels.self, output: await worker(["hub-models"]))
        hubModels = models.models
    }
    var managedEncoderURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/Material Models/pbrnxt-base")
    }
    func installEncoder() {
        operation("Downloading the material base…") {
            _ = try await self.worker(["install-base", "--destination", self.managedEncoderURL.path])
            self.modelDirectory = self.managedEncoderURL.path
            self.saveConfiguration()
            self.activity = "Material base installed."
        }
    }
    func removeDownloadedEncoder() {
        operation("Removing downloaded base weights…") {
            _ = try await self.worker(["remove-base", "--directory", self.managedEncoderURL.path])
            self.activity = "Base weights removed. Your full checkpoint remains usable; the base can be downloaded again."
        }
    }
    func refreshHubModels() {
        operation("Finding your Hugging Face material models…") {
            let result = try WorkbenchResult.decode(WorkbenchHubModels.self, output: await self.worker(["hub-models"]))
            self.hubModels = result.models
        }
    }
    func downloadHubModel(_ model: WorkbenchHubModel) {
        operation("Downloading \(model.repository)…") {
            let destination = self.managedEncoderURL.deletingLastPathComponent().appendingPathComponent(model.repository.replacingOccurrences(of: "/", with: "--"))
            var args = ["download-model", "--repo", model.repository, "--destination", destination.path]
            if let revision = model.revision { args += ["--revision", revision] }
            let result = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: await self.worker(args))
            self.saveConfiguration()
            try await self.loadCheckpoint(result.url)
        }
    }
    func chooseMixAdapter() {
        chooseFile(title: "Choose a compatible material LoRA") { url in
            self.adapterMix.append(WorkbenchAdapterWeight(path: url.path))
        }
    }
    func forgetSelectedCheckpoint() {
        guard let checkpoint = selectedCheckpoint else { return }
        checkpoints.removeAll { $0.id == checkpoint.id }
        comparisonCheckpointIds.remove(checkpoint.id)
        selectedCheckpointId = checkpoints.first?.id
        preferences.set(checkpoints.map(\.checkpointPath), forKey: "checkpoints")
    }

    func worker(_ args: [String]) async throws -> String {
        let includesTarget = ["create-dataset", "edit-dataset", "add-material", "import-folder", "remove-material"].contains(args.first ?? "") && !args.contains("--target")
        let includesPreparationWorkers = args.first == "prepare-size" && !args.contains("--preparation-workers")
        let args = args + (includesTarget ? ["--target", training.target] : [])
            + (includesPreparationWorkers ? ["--preparation-workers", String(preparationWorkers)] : [])
        try Task.checkCancellation()
        guard !WorkbenchLifecycle.shared.isTerminating else { throw CancellationError() }
        let trainingWorker = ["train", "refine"].contains(args.first ?? "")
        let progressWorker = trainingWorker || ["prepare-size", "scan-folder", "import-folder"].contains(args.first ?? "")
        if progressWorker { trainingEventBuffer = "" }
        if trainingWorker { hasTrainingStarted = false }
        defer { if trainingWorker { hasTrainingStarted = false } }
        if let workerOverride {
            let output = try await workerOverride(args, args.first ?? "")
            if progressWorker { recordTrainingProgress(output) }
            try Task.checkCancellation()
            return output
        }
        saveConfiguration(refreshSelectedRuntime: false)
        let logs = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/Worklogs")
        try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let log = logs.appendingPathComponent("\(UUID().uuidString).log")
        lastLogURL = log
        let events = NativeWorkbenchLog(url: log)
        let control = NativeMaterialTrainingControl(); runner = control
        let workerId = UUID(); activeWorkerId = workerId
        let event: @Sendable (String) -> Void = { [weak self] chunk in
            guard events.append(chunk) else { return }
            Task { @MainActor in
                // Deliver progress in order at most ten times per second. The
                // trainer never waits for UI updates, and its disk log is complete.
                try? await Task.sleep(for: .milliseconds(100))
                guard let self, self.activeWorkerId == workerId else { return }
                let pending = events.drainPending()
                self.logText = events.text
                if progressWorker, !pending.isEmpty { self.recordTrainingProgress(pending) }
            }
        }
        let operation = Task { try await NativeMaterialCommands.run(arguments: args, onEvent: event, control: control) }
        WorkbenchLifecycle.shared.add(control, cancel: { operation.cancel() })
        defer {
            // Flush any final events once before retiring the worker. A queued
            // UI delivery then fails its worker identity check and cannot replay them.
            let pending = events.drainPending()
            if progressWorker, !pending.isEmpty { recordTrainingProgress(pending) }
            runner = nil; activeWorkerId = nil
            logText = events.text
            WorkbenchLifecycle.shared.remove(control)
        }
        let output = try await withTaskCancellationHandler { try await operation.value } onCancel: { operation.cancel(); control.stop() }
        events.append(output + "\n")
        try Task.checkCancellation()
        return output
    }
    func operation(_ label: String, training: Bool = false, body: @escaping @MainActor () async throws -> Void) {
        guard !isBusy else { return }
        isCheckpointPending = false; validationSummary = ""
        isBusy = true; isTraining = training; isStopping = false; hasTrainingStarted = false; isSavingTraining = false
        error = nil; activity = label; logText = ""; trainingEventBuffer = ""
        trainingProgress = training ? WorkbenchTrainingProgress() : nil
        task = Task {
            defer {
                isBusy = false; isTraining = false; isResumingTraining = false; isStopping = false
                hasTrainingStarted = false; isSavingTraining = false; isCheckpointPending = false; isPreparingDataset = false; task = nil
                if let url = queuedDatasetURL {
                    queuedDatasetURL = nil
                    openDataset(url)
                } else if let handoff = queuedTrainingHandoff {
                    queuedTrainingHandoff = nil
                    receiveTrainingHandoff(handoff)
                } else if let url = queuedResumeURL {
                    queuedResumeURL = nil
                    receiveResumeCheckpoint(url)
                }
            }
            do {
                try await body()
                if activity == label { activity = "Ready." }
            }
            catch {
                if Task.isCancelled || error is CancellationError {
                    if training { trainingProgress?.finish(state: .aborted) }
                    activity = training ? "Training aborted. Previously saved checkpoints are kept." : "Operation stopped. See the log and output folder."
                } else {
                    if training { trainingProgress?.finish(state: .failed) }
                    self.error = error.localizedDescription; activity = "Operation stopped. See the error and log."
                }
            }
        }
    }
    private func newOutputURL(prefix: String) throws -> URL {
        if workspaceURL == managedWorkspaceURL, !FileManager.default.fileExists(atPath: workspacePath) {
            try FileManager.default.createDirectory(at: managedWorkspaceURL, withIntermediateDirectories: true)
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: workspacePath, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw StudioError("The chosen working folder is missing. Locate it in Local runtime settings.")
        }
        let root = workspaceURL.appendingPathComponent("out/material-training")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.appendingPathComponent("\(prefix)-\(UUID().uuidString.prefix(8))")
    }
    private func chooseFile(types: [UTType] = [], title: String, selected: @escaping (URL) -> Void) {
        guard !isBusy else { return }
        let panel = NSOpenPanel(); panel.title = title; panel.allowedContentTypes = types
        panel.begin { response in if response == .OK, let url = panel.url { selected(url) } }
    }
    private func chooseFolder(title: String, selected: @escaping (URL) -> Void) {
        guard !isBusy else { return }
        let panel = NSOpenPanel(); panel.title = title; panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.begin { response in if response == .OK, let url = panel.url { selected(url) } }
    }
}
